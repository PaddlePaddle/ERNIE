# -*- coding: utf-8 -*-
# !/usr/bin/env python3

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

2025
import os
import copy
from typing import Dict, Union
from paddleformers.transformers.configuration_utils import PretrainedConfig
import logging


logger = logging.getLogger(__name__)


class EVAVisionTransformerConfig(PretrainedConfig):
    """_summary_

    Args:
        PretrainedConfig (_type_): _description_

    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """

    model_type = "evavision_transformer"
    attribute_map: Dict[str, str] = {"hidden_size": "width"}

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_chans=3,
        output_dim=1000,
        width=768,
        layers=12,
        head_width: int = 64,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_values=None,
        patch_dropout=0.0,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        rope=False,
        use_mean_pooling=True,
        attentional_pool=False,
        n_queries=256,
        attn_pooler_heads=8,
        init_scale=0.001,
        use_recompute_attn=False,
        xattn=False,
        postnorm=False,
        pt_hw_seq_len=16,
        intp_freq=False,
        naiveswiglu=False,
        subln=False,
        output_tokens=False,
        token_feats=False,  # whether tokens send to self.head
        fusedLN=False,
        inner_attn_ln=True,  # false in eva-01 clip
        use_flash_attn=True,
        fused_qkv_attn=False,
        tensor_parallel_degree=-1,
        resolution_transform="ordered",
        need_head=False,
        use_rms_norm=False,
        head_bias=True,
        fuse_rms_norm=True,
        seqlen=None,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.output_dim = output_dim
        self.width = width
        self.layers = layers
        self.head_width = head_width
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.init_values = init_values
        self.patch_dropout = patch_dropout
        self.use_abs_pos_emb = use_abs_pos_emb
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_shared_rel_pos_bias = use_shared_rel_pos_bias
        self.rope = rope
        self.use_mean_pooling = use_mean_pooling
        self.attentional_pool = attentional_pool
        self.n_queries = n_queries
        self.attn_pooler_heads = attn_pooler_heads
        self.init_scale = init_scale
        self.use_recompute_attn = use_recompute_attn
        self.xattn = xattn
        self.postnorm = postnorm
        self.pt_hw_seq_len = pt_hw_seq_len
        self.intp_freq = intp_freq
        self.naiveswiglu = naiveswiglu
        self.subln = subln
        self.output_tokens = output_tokens
        self.token_feats = token_feats
        self.fusedLN = fusedLN
        self.inner_attn_ln = inner_attn_ln
        self.use_flash_attn = use_flash_attn
        self.fused_qkv_attn = fused_qkv_attn
        self.tensor_parallel_degree = tensor_parallel_degree
        self.resolution_transform = resolution_transform
        self.need_head = need_head
        self.use_rms_norm = use_rms_norm
        self.head_bias = head_bias
        self.num_patches = (image_size // patch_size) ** 2
        self.fuse_rms_norm = fuse_rms_norm
        self.seqlen = seqlen
        assert self.resolution_transform in [
            "interpolate",
            "downsample",
            "adaptive",
            "ordered",
        ]

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        """_summary_

        Args:
            pretrained_model_name_or_path (Union[str, os.PathLike]): _description_

        Returns:
            PretrainedConfig: _description_

        Yields:
            Iterator[PretrainedConfig]: _description_
        """
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        if (
            "model_type" in config_dict
            and hasattr(cls, "model_type")
            and config_dict["model_type"] != cls.model_type
        ):
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        if "vision_cfg" in config_dict:
            config_dict = config_dict["vision_cfg"]

        return cls.from_dict(config_dict, **kwargs)


class ImageEncoderConfig(EVAVisionTransformerConfig):
    """ImageEncoderConfig

    Args:
        EVAVisionTransformerConfig (_type_): _description_
    """

    def __init__(
        self,
        im_patch_id=None,
        clip_data_rank=0,
        clip_data_world_size=1,
        clip_local_loss: bool = True,
        pad_token_id=None,
        clip_gather_with_grad: bool = True,
        clip_cache_labels: bool = False,
        freeze_vit: bool = False,
        freeze_text_tower: bool = False,
        train_clip_only=False,
        text_model_config=None,
        language_model_config=None,
        ignore_index=-100,
        init_exp_logit_scale=1 / 0.07,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.freeze_vit = freeze_vit
        self.freeze_text_tower = freeze_text_tower
        self.train_clip_only = train_clip_only
        self.text_model_config = text_model_config
        self.language_model_config = language_model_config
        self.im_patch_id = im_patch_id
        self.pad_token_id = pad_token_id
        self.clip_local_loss = clip_local_loss
        self.clip_gather_with_grad = clip_gather_with_grad
        self.clip_cache_labels = clip_cache_labels
        self.clip_data_rank = clip_data_rank
        self.clip_data_world_size = clip_data_world_size
        self.ignore_index = ignore_index
        self.init_exp_logit_scale = init_exp_logit_scale

    def to_dict(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        output = copy.deepcopy(self.__dict__)
        if self.language_model_config is not None:
            output["language_model_config"] = self.language_model_config.to_dict()
        if self.text_model_config is not None:
            output["text_model_config"] = self.text_model_config.to_dict()
        return output
