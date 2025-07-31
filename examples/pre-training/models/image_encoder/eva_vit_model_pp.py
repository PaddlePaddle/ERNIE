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

"""
@author: kebo
@contact: kebo01@baidu.com

@version: 1.0
@file: eva_vit_model_pp.py
@time: 2024/03/15 15:55:19
@Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved

这一行开始写关于本文件的说明与解释


"""
import logging
import contextlib
from functools import partial
from collections import defaultdict

import paddle
import paddle.distributed.fleet as fleet
import paddle.nn as nn
from paddle.nn import functional as F

from paddleformers.transformers import PretrainedModel
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker
from paddle.distributed.fleet.layers.mpu.mp_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from models.sequence_parallel_utils import (
    ColumnSequenceParallelLinear,
    RowSequenceParallelLinear,
)
from models.image_encoder.eva_vit_model import (
    LayerNorm,
    PatchEmbed,
    PatchEmbedForPatchified,
    Block,
    PatchDropout,
    PatchDropoutForAdaptive,
    EVAVisionTransformerConfig,
    EVAVisionTransformer,
)

from .utils import to_2tuple
from .modules.rope import VisionRotaryEmbeddingME

# from models.comm_utils import PrintOp

FusedLinear = type("Linear", (paddle.incubate.nn.FusedLinear,), {})

logger = logging.getLogger(__name__)

try:
    from .modules.fusedln import FusedLayerNorm
except Exception:
    from paddle.nn import LayerNorm as FusedLayerNorm

    logger.warning("Warning, FusedLn module is not available, use LayerNorm instead.")


def get_hcg():
    """_summary_

    Returns:
        _type_: _description_
    """
    return fleet.get_hybrid_communicate_group()


class PatchEmbedPipe(nn.Layer):
    """PatchEmbedPipe"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.use_recompute_attn = config.use_recompute_attn
        self.output_tokens = config.output_tokens
        self.token_feats = config.token_feats
        self.resolution_transform = config.resolution_transform
        self.seqlen = config.seqlen

        self.output_dim = output_dim = config.output_dim
        self.width = width = config.width
        self.naiveswiglu = config.naiveswiglu

        if self.resolution_transform != "adaptive":
            self.patch_embed = PatchEmbed(config)
        else:
            self.patch_embed = PatchEmbedForPatchified(config)
        num_patches = self.patch_embed.num_patches
        init_data = paddle.zeros(shape=[1, 1, width])
        self.cls_token = self.create_parameter(
            shape=[1, 1, width],
            default_initializer=paddle.nn.initializer.Assign(init_data),
        )
        if config.use_abs_pos_emb:
            init_data = paddle.zeros(shape=[1, num_patches + 1, width])
            self.pos_embed = self.create_parameter(
                shape=[1, num_patches + 1, width],
                default_initializer=paddle.nn.initializer.Assign(init_data),
            )
        else:
            self.pos_embed = None
        self.pos_drop = paddle.nn.Dropout(p=config.drop_rate)
        assert not config.use_shared_rel_pos_bias
        if self.resolution_transform != "adaptive":
            self.patch_dropout = PatchDropout(
                config.patch_dropout, rope=config.rope
            )  # if config.patch_dropout > 0.0 else paddle.nn.Identity()
        else:
            self.patch_dropout = PatchDropoutForAdaptive(
                config.patch_dropout, rope=config.rope
            )
        if config.tensor_parallel_degree > 1:
            self.rng_tracker = get_rng_state_tracker().rng_state
        else:
            self.rng_tracker = contextlib.nullcontext

    def forward(self, args):
        """forward"""

        if not isinstance(args, tuple):
            args = (args,)
        if len(args) > 1:
            x, image_sizes, *remain_args = args
        else:
            x, image_sizes, remain_args = args[0], None, ()

        if remain_args:
            (position_ids,) = remain_args
        else:
            position_ids = None

        if x.ndim == 2:
            assert (
                image_sizes is not None
                and len(x) - image_sizes.prod(-1).sum(0).item() >= 0
            ), f"patches in images_sizes:{image_sizes} > image_len:{len(x)}"

        x = self.patch_embed(x)

        if x.ndim == 2 and image_sizes is not None:
            image_sizes = image_sizes.numpy()
            images = x.split(image_sizes.prod(-1).tolist(), axis=0)
            cls_tokens = self.cls_token.reshape([1, -1])
            x = paddle.concat([t for im in images for t in [cls_tokens, im]], 0)
        else:
            batch_size = x.shape[0]
            cls_tokens = self.cls_token.expand(shape=[batch_size, -1, -1])
            x = paddle.concat([cls_tokens, x], 1)

        if x.ndim == 2 and image_sizes is not None:
            pos_emb, position_ids, position_ids_2d = (
                EVAVisionTransformer.get_adaptive_position_emb(
                    self, image_sizes, position_ids
                )
            )
            assert len(position_ids) == len(x), (len(position_ids), len(x))
        else:
            pos_emb, position_ids, position_ids_2d = (
                EVAVisionTransformer.get_position_emb(self, x)
            )

        with self.rng_tracker("global_seed"):
            if self.pos_embed is not None:
                x = x + pos_emb
            x = self.pos_drop(x)

        cumulative_indices = None

        if self.config.rope:
            if (
                self.training
                and not isinstance(self.patch_dropout, paddle.nn.Identity)
                and self.patch_dropout.prob != 0.0
            ):
                if self.resolution_transform != "adaptive":
                    x, patch_indices_keep = self.patch_dropout(x)
                    position_ids = position_ids[
                        paddle.arange(x.shape[0])[..., None], patch_indices_keep
                    ]
                    position_ids_2d = position_ids_2d[
                        paddle.arange(x.shape[0])[..., None], patch_indices_keep
                    ]
                else:
                    pass  # handle adaptive patch drop in dataloader

        else:
            if self.resolution_transform != "adaptive":
                x = self.patch_dropout(x)
            else:
                pass  # handle adaptive patch drop in dataloader

        if x.ndim == 2:
            (cls_pos,) = paddle.where(position_ids == 0)
            (non_cls_pos,) = paddle.where(position_ids != 0)
            cls_pos = cls_pos.squeeze(-1)
            non_cls_pos = non_cls_pos.squeeze(-1)
            cumulative_indices = F.pad(cls_pos, (0, 1), value=len(position_ids)).astype(
                "int32"
            )

        assert self.seqlen
        actual_len = paddle.full([1], len(x), dtype="int64")
        if x.ndim == 2:  # do padding
            pad_len = self.seqlen - len(x)
            # logger.info(f'[vit] before varlen:{actual_len}, x={x.shape}')
            if pad_len > 0:
                x = F.pad(x, (0, pad_len, 0, 0))
                cumulative_indices = paddle.concat(
                    [
                        cumulative_indices,
                        paddle.full((1,), len(x), dtype=cumulative_indices.dtype),
                    ],
                    0,
                )
                if self.config.rope:
                    position_ids = paddle.concat(
                        (
                            position_ids,
                            paddle.zeros([1], dtype=position_ids.dtype),
                            paddle.full(
                                [pad_len - 1], -1, dtype=position_ids.dtype
                            ),  # special hard code
                        ),
                        0,
                    )
                    position_ids_2d = paddle.concat(
                        (
                            position_ids_2d,
                            paddle.zeros([pad_len - 1, 2], dtype=position_ids_2d.dtype),
                        ),
                        0,
                    )

            else:
                raise RuntimeError(
                    f"unable to fix vit seqlen, seqlen={self.seqlen}, "
                    f"image-patch-sizes={len(x)}, image_sizes={image_sizes}"
                )
        # logger.info(f"fwd-shape:{x.shape}")

        if x.ndim == 2:
            return x, position_ids, position_ids_2d, cumulative_indices, actual_len
        return x, actual_len


class BlockPipe(Block):
    """
    BlockPipe
    """

    def __init__(self, config, drop_path, norm_layer, layer_idx):
        image_size = to_2tuple(config.image_size)
        patch_size = to_2tuple(config.patch_size)
        patch_shape = image_size[0] // patch_size[0], image_size[1] // patch_size[1]
        window_size = patch_shape if config.use_rel_pos_bias else None
        num_heads = config.width // config.head_width
        width = config.width
        if config.rope:
            assert not config.use_abs_pos_emb, "RoPE do need abs pos emb"
            half_head_dim = width // num_heads // 2
            hw_seq_len = config.image_size // config.patch_size
            rope = VisionRotaryEmbeddingME(
                dim=half_head_dim,
                pt_seq_len=config.pt_hw_seq_len,
                ft_seq_len=hw_seq_len if config.intp_freq else None,
            )
        else:
            rope = None

        super().__init__(
            config,
            drop_path=drop_path,
            norm_layer=norm_layer,
            window_size=window_size,
            rope=rope,
        )

        self.layer_idx = layer_idx

    def forward(self, args):
        """
        转换为固定长度的 VIT pp
        """

        if len(args) == 5:
            x, position_ids_, position_ids_2d_, cumulative_indices_, actual_len = args

            pad_len = len(x) - actual_len
            # logger.info(f'[vit] before len:{actual_len}, x={x.shape} 1d={position_ids_.shape} 2d={position_ids_2d_.shape} ')
            x = x[:actual_len]
            cumulative_indices = cumulative_indices_[:-1]
            position_ids = position_ids_[:actual_len]
            position_ids_2d = position_ids_2d_[: -(pad_len - 1)]
            # logger.info(f'[vit] actual len:{actual_len}, x={x.shape} 1d={position_ids.shape} 2d={position_ids_2d.shape} ')
        else:
            x, actual_len = args
            x = x[:actual_len]
            pad_len = len(x) - actual_len
            position_ids = position_ids_2d = cumulative_indices = None
            pad_len = None
        if self.attn.rope is not None:
            self.attn.rope.forward = partial(
                self.attn.rope.forward,
                position_ids=position_ids,
                position_ids_2d=position_ids_2d,
            )

        x = super().forward(x, cumulative_indices=cumulative_indices)

        if pad_len is not None:
            x = F.pad(x, (0, pad_len, 0, 0))
        # logger.info(f'[vit] output len:{actual_len}, x={x.shape} 1d={position_ids_.shape} 2d={position_ids_2d_.shape} ')
        if position_ids is None:
            return x, actual_len.clone()
        return (
            x,
            position_ids_.clone(),
            position_ids_2d_.clone(),
            cumulative_indices_.clone(),
            actual_len.clone(),
        )


class VisionHeadPipe(nn.Layer):
    """VisionHeadPipe"""

    def __init__(self, config, norm_layer):
        super().__init__()
        self.config = config
        use_mean_pooling = config.use_mean_pooling
        self.output_dim = output_dim = config.output_dim
        self.width = width = config.width
        self.seqlen = config.seqlen
        self.attn_pool = None
        self.norm = paddle.nn.Identity() if use_mean_pooling else norm_layer(width)
        # self.fc_norm = norm_layer(width) if use_mean_pooling else None
        assert config.need_head
        self.need_head = config.need_head
        if config.tensor_parallel_degree > 1:
            self.head = (
                ColumnParallelLinear(
                    width,
                    output_dim,
                    weight_attr=None,
                    has_bias=True,
                    gather_output=True,
                )
                if output_dim > 0
                else paddle.nn.Identity()
            )
        else:
            self.head = (
                paddle.nn.Linear(width, output_dim)
                if output_dim > 0
                else paddle.nn.Identity()
            )

    def forward(self, args):
        """fwd"""
        if isinstance(args, tuple):
            x, position_ids, position_ids_2d, cumulative_indices, actual_len = args
        else:
            x, actual_len = args
            position_ids = position_ids_2d = cumulative_indices = None
        x = x[:actual_len]

        all_features = self.norm(x)

        if x.ndim == 2:
            assert position_ids is not None
            position_ids = position_ids[:actual_len]
            # if len(position_ids) > self.seqlen:
            #     position_ids = position_ids[: self.seqlen]
            (cls_pos,) = paddle.where(position_ids == 0)
            # if cls_pos[-1].item() == len(position_ids) - 1: #pop padding pos
            #     cls_pos = cls_pos[:-1]
            (non_cls_pos,) = paddle.where(position_ids > 0)
            cls_pos = cls_pos.squeeze(-1)
            non_cls_pos = non_cls_pos.squeeze(-1)
            # cls_pos = cls_pos[cls_pos < non_cls_pos[-1]]  # remove padding last cls, tricky
            x = all_features[cls_pos]
            if x.ndim == 1 and len(cls_pos) == 1:
                x = x.unsqueeze(0)
            all_features = all_features[non_cls_pos]
        else:
            x = all_features[:, 0]
            all_features = all_features[:, 1:]
        if self.need_head:
            x = self.head(x)
        if x.dim == 2:
            return x, all_features
        return x, all_features


class PipelinePretrainedModel(PretrainedModel):
    """_summary_

    Args:
        PretrainedModel (_type_): _description_
    """

    def __init__(self, config, *args, **kwargs):  # no call
        super().__init__(config, *args, **kwargs)

    def init(self, config, *args, **kwargs):  # no call
        """_summary_

        Args:
            config (_type_): _description_
        """
        self._sequential_layers = []
        self._pipeline_name_mapping = None

    def add_sequential_layer(self, layer_desc, name_prefix=""):
        """_summary_

        Args:
            layer_desc (_type_): _description_
            name_prefix (str, optional): _description_. Defaults to "".
        """
        self._sequential_layers.append(
            {"layer": layer_desc, "name_prefix": name_prefix}
        )

    def get_sequential_layers(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return [x["layer"] for x in self._sequential_layers]

    def get_sequential_name_prefixs(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        res = {
            str(index): x["name_prefix"]
            for index, x in enumerate(self._sequential_layers)
        }
        return {**res, "_loss_fn": "criterion", "criterion": "criterion"}

    def _set_pipeline_name_mapping(self, mappings=None):
        """_summary_

        Args:
            mappings (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if mappings is not None:
            self._pipeline_name_mapping = mappings
        else:
            mapping = {}
            state_dict_keys = list(super().state_dict().keys())
            first_key = state_dict_keys[0].split(".")
            # if use virtual pp_degree, the prefix is like 0.0.xxx
            # else it will be like 0.xxx
            use_virtual_pp_degree = first_key[0].isdigit() and first_key[1].isdigit()

            prefixs = self.get_sequential_name_prefixs()
            for k in state_dict_keys:
                name_splited = k.split(".")
                if use_virtual_pp_degree:
                    idx = str(int(name_splited[0]) + int(name_splited[1]))
                    single_name = [prefixs[idx]]
                    single_name.extend(name_splited[2:])
                else:
                    idx = name_splited[0]
                    single_name = [prefixs[idx]]
                    single_name.extend(name_splited[1:])
                single_name = [s for s in single_name if s is not None]
                mapping[".".join(single_name)] = k

            self._pipeline_name_mapping = mapping

        return self._pipeline_name_mapping

    def state_dict(self, *args, **kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        state_dict = super().state_dict(*args, **kwargs)

        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()
        assert (
            len(self._pipeline_name_mapping) > 0
        ), "The pipeline stage must have parameters!"
        pp_to_single_mapping = {v: k for k, v in self._pipeline_name_mapping.items()}

        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            state_dict[pp_to_single_mapping[k]] = v

        return state_dict

    def set_state_dict(self, state_dict, *args, **kwargs):
        """_summary_

        Args:
            state_dict (_type_): _description_
        """
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()
        assert (
            len(self._pipeline_name_mapping) > 0
        ), "The pipeline stage must have parameters!"

        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if k not in self._pipeline_name_mapping:
                continue
            state_dict[self._pipeline_name_mapping[k]] = v

        res = super().set_state_dict(state_dict, *args, **kwargs)
        logger.info(f"PP state-dict: {res}")
        return res

    def _init_weights(self, layer):
        """Initialization hook"""
        if self.config.tensor_parallel_degree > 1:
            rng_tracker = get_rng_state_tracker().rng_state
        else:
            rng_tracker = contextlib.nullcontext

        if isinstance(
            layer,
            (
                ColumnParallelLinear,
                RowParallelLinear,
                ColumnSequenceParallelLinear,
                RowSequenceParallelLinear,
                VocabParallelEmbedding,
                nn.Conv2D,
                nn.Embedding,
                nn.Linear,
            ),
        ):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            # logger.info(f'initializing pp:{type(layer)}')

            with rng_tracker():
                dtype = paddle.get_default_dtype()
                paddle.set_default_dtype(
                    "float32"
                )  # 在defualt dtype为bfloat16的情况下调用randn会报错，实际上randn已经支持bfloat16了，只是API检测层面的错误。
                layer.weight.set_value(
                    paddle.randn(layer.weight.shape, dtype=dtype).scale(
                        self.config.initializer_range
                    )
                )
                paddle.set_default_dtype(dtype)
                logger.info(
                    f"dist-init-fc: shape={layer.weight.shape}, range={self.config.initializer_range}, "
                    f"type={type(layer)},norm={layer.weight.astype('float32').norm()}"
                )


class EVAVisionTransformerPipe(PipelinePretrainedModel, PipelineLayer):
    """EVAVisionTransformerPipe

    Args:
        PipelinePretrainedModel (_type_): _description_
        PipelineLayer (_type_): _description_

    Returns:
        _type_: _description_
    """

    config_class = EVAVisionTransformerConfig
    use_dummy_loss_fn = False
    _get_tensor_parallel_mappings = EVAVisionTransformer._get_tensor_parallel_mappings

    def __init__(self, config):
        self.config = config
        self.config.initializer_range = 0.02
        self.accum_info = defaultdict(float)
        PipelinePretrainedModel.init(self, config=config)
        self.width = width = config.width
        num_heads = config.width // config.head_width
        norm_layer = (
            partial(FusedLayerNorm, epsilon=1e-6)
            if config.fusedLN
            else partial(LayerNorm, epsilon=1e-6)
        )
        dpr = [
            x.item()
            for x in paddle.linspace(
                start=0, stop=config.drop_path_rate, num=config.layers
            )
        ]

        self.add_sequential_layer(LayerDesc(PatchEmbedPipe, config=config), None)
        for i in range(config.layers):
            self.add_sequential_layer(
                LayerDesc(
                    BlockPipe,
                    config=config,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    layer_idx=i,
                ),
                f"blocks.{i}",
            )
        self.add_sequential_layer(
            LayerDesc(
                VisionHeadPipe,
                config=config,
                norm_layer=norm_layer,
            ),
            None,
        )

        seg_method = "layer:Block|EmptyLayer"
        if config.layers % get_hcg().topology().get_dim_size("pipe") != 0:
            seg_method = "uniform"
        recompute_interval = 1 if config.use_recompute_attn else 0
        logger.info(
            f"using recompute_interval={recompute_interval}, seg_method={seg_method}"
        )

        PipelineLayer.__init__(
            self,
            layers=self.get_sequential_layers(),
            loss_fn=self.get_loss_fn(config),
            topology=get_hcg().topology(),
            seg_method=seg_method,
            recompute_interval=recompute_interval,
            recompute_ctx={
                "mp_group": get_hcg().get_model_parallel_group(),
                "offload": False,
                "partition": False,  # TODO：看看怎么 Partition recompute checkpoint。
            },
            num_virtual_pipeline_stages=1,
        )

    def get_loss_fn(self, config):
        """_summary_

        Args:
            config (_type_): _description_

        Returns:
            _type_: _description_
        """
        # return ErniePretrainingCriterion(config, return_tuple=False)

        # TODO dummy loss fn
        def func(pred, labels):  # lazy lossfn
            image_embs, text_embs, all_features = pred
            with paddle.amp.auto_cast(False):
                image_embs = F.normalize(image_embs.astype("float32"), axis=-1)
                text_embs = F.normalize(text_embs.astype("float32"), axis=-1)
            src_id, labels = labels[0], labels[1:]
            src_id = src_id[0].item()

            loss, info = self.criterion((image_embs, text_embs, all_features), labels)

            assert src_id in [0, 1], src_id
            logging_prefix = "cn" if src_id == 1 else "en"
            if info is not None:
                for k, v in info["clipdata_cliploss_info"].items():
                    self.accum_info[f"{logging_prefix}_clip_data/clip_" + k] += (
                        v.detach() if isinstance(v, paddle.Tensor) else v
                    )
                if "clipdata_decoderloss_info" in info:
                    for k, v in info["clipdata_decoderloss_info"].items():
                        self.accum_info[f"{logging_prefix}_clip_data/decoder_" + k] += (
                            v.detach() if isinstance(v, paddle.Tensor) else v
                        )
                self.accum_info["all_data/total_loss"] += loss.detach()

            if loss is None:
                loss = paddle.to_tensor(0.0, dtype="float32")  # 刷库
            return loss

        return func
