# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
import logging
from functools import partial
from typing import Optional, Tuple, Union
import math

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddleformers.transformers.model_outputs import (
    BaseModelOutputWithPooling,
)
from paddleformers.transformers.model_utils import PretrainedModel

from paddleformers.transformers.model_utils import _convert_state_dict_dtype_and_shape

from .configuration import CLIPVisionConfig

logger = logging.getLogger(__name__)


def quick_gelu(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return x * F.sigmoid(1.702 * x)


F.quick_gelu = quick_gelu


class CLIPVisionTransformer(PretrainedModel):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    config_class = CLIPVisionConfig

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        assert self.config.resolution_transform != "adaptive"
        embed_dim = config.hidden_size
        self.input_resolution = config.image_size
        dtype = paddle.get_default_dtype()
        paddle.set_default_dtype(
            "float32"
        )  # 在defualt type为bfloat16的情况下调用randn会报错，实际上randn已经支持bfloat16了，只是API检测层面的错误。
        embedding_params = paddle.randn((embed_dim,), dtype=dtype)
        paddle.set_default_dtype(dtype)
        self.class_embedding = self.create_parameter(
            (embed_dim,),
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Assign(embedding_params),
        )

        self.conv1 = nn.Conv2D(
            in_channels=config.num_channels,
            out_channels=embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias_attr=False,
        )
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.positional_embedding = nn.Embedding(self.num_positions, embed_dim)

        self.ln_pre = nn.LayerNorm(embed_dim)

        TransformerEncoderLayer = nn.TransformerEncoderLayer
        TransformerEncoder = nn.TransformerEncoder
        encoder_layer = TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            normalize_before=True,
            dropout=0.0,
            activation=config.hidden_act,
            attn_dropout=config.attention_dropout,
            act_dropout=0.0,
        )
        self.transformer = TransformerEncoder(encoder_layer, config.num_hidden_layers)
        # self.ln_post = nn.LayerNorm(embed_dim)
        self.register_buffer(
            "position_ids", paddle.arange(self.num_positions).reshape((1, -1))
        )
        self.resolution_transform = config.resolution_transform

        if config.use_recompute:
            self.apply(partial(self._set_gradient_checkpointing, value=True))

    def _set_gradient_checkpointing(self, module, value=False):
        """_summary_

        Args:
            module (_type_): _description_
            value (bool, optional): _description_. Defaults to False.
        """
        if isinstance(module, nn.TransformerEncoder):
            module.enable_recompute = value

    def forward(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        cumulative_indices: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Args:
            pixel_values (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Padding will be ignored by default should you provide it.
                Pixel values can be obtained using
                [`CLIPFeatureExtractor`]. See [`CLIPFeatureExtractor.__call__`] for details.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`BaseModelOutputWithPooling`] instead of a plain tuple.

        Returns:
            An instance of :class:`BaseModelOutputWithPooling` if `return_dict=True`.
            Otherwise it returns a tuple of tensors
            corresponding to ordered and not None (depending on the input arguments) fields of
            :class:`BaseModelOutputWithPooling`.

        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_dict = True  # hard code

        pixel_values = self.conv1(pixel_values)
        pixel_values = pixel_values.reshape(
            (pixel_values.shape[0], pixel_values.shape[1], -1)
        )
        pixel_values = pixel_values.transpose((0, 2, 1))

        if pixel_values.shape[1] != self.positional_embedding.weight.shape[0] - 1:
            # 可变分辨率
            old_shape = int(math.sqrt(self.positional_embedding.weight.shape[0] - 1))
            new_shape = int(math.sqrt(pixel_values.shape[1]))
            if self.resolution_transform == "interpolate":
                pos_emb_img = (
                    self.positional_embedding.weight[:, 1:]
                    .reshape([1, old_shape, old_shape, -1])
                    .transpose([0, 3, 1, 2])
                )
                pos_emb_img = paddle.nn.functional.interpolate(
                    pos_emb_img,
                    size=[new_shape, new_shape],
                    mode="bicubic",
                    align_corners=True,
                )
                pos_emb_img = pos_emb_img.transpose([0, 2, 3, 1]).reshape(
                    [1, new_shape * new_shape, -1]
                )
                positional_embeddings = paddle.concat(
                    [self.positional_embedding.weight[:, :1], pos_emb_img], axis=1
                )
            elif self.resolution_transform == "downsample":
                tmp = paddle.arange(
                    0, old_shape, int(old_shape / new_shape), dtype="int64"
                )
                position_ids = (
                    tmp.unsqueeze(0) + tmp.unsqueeze(-1) * old_shape
                ).flatten()
                positional_embeddings = paddle.concat(
                    [
                        self.positional_embedding.weight[:1, :],
                        self.positional_embedding.weight[1:, :][position_ids],
                    ],
                    axis=0,
                )
            else:
                raise ValueError(
                    f"resolution_transform {self.resolution_transform} is not supported"
                )
        else:
            positional_embeddings = self.positional_embedding.weight

        embedding_output = paddle.concat(
            [
                self.class_embedding.unsqueeze([0, 1]).expand(
                    [pixel_values.shape[0], -1, -1]
                ),
                pixel_values,
            ],
            axis=1,
        )
        hidden_states = embedding_output + positional_embeddings  # [B, seqlen, hidden]

        hidden_states = self.ln_pre(hidden_states)

        encoder_outputs = self.transformer(
            hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if isinstance(encoder_outputs, type(embedding_output)):
            last_hidden_state = encoder_outputs
        else:
            last_hidden_state = encoder_outputs[0]

        pooled_output = None

        if isinstance(encoder_outputs, type(embedding_output)):
            return (last_hidden_state, pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        """
        dummy
        """
        return {}

    def set_state_dict(self, state_dict, *args, **kwargs):
        """_summary_

        Args:
            state_dict (_type_): _description_
        """
        for key in list(state_dict.keys()):
            if key.startswith("vision_model."):
                new_key = key.replace("vision_model.", "")
                state_dict[new_key] = state_dict.pop(key)
        _convert_state_dict_dtype_and_shape(state_dict, self)
        ret = super().set_state_dict(state_dict, *args, **kwargs)
        logger.info(f"openclip set_state_dict: {ret}")
