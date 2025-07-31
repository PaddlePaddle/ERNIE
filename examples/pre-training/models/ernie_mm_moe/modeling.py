# !/usr/bin/env python3

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Erniemm model"""
import logging
import re
import json
from typing import List, Optional
from collections import defaultdict
import copy
from functools import partial
import numpy as np
import math
from copy import deepcopy
from types import MethodType

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.distributed.fleet.utils import recompute
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu.mp_layers import (
    RowParallelLinear,
    ColumnParallelLinear,
    VocabParallelEmbedding,
)
from models.comm_utils import profile
from models.sequence_parallel_utils import (
    ScatterOp,
    GatherOp,
    mark_as_sequence_parallel_parameter,
    RowSequenceParallelLinear,
)
from models.ernie.modeling import (
    ErniePretrainingCriterion as ErniePretrainingCriterionBase,
)
from models.ernie.modeling import (
    RMSNorm,
    LayerNorm,
    FusedLayerNorm,
    ErnieLMHead,
    parallel_matmul,
)
from models.ernie.modeling import ErnieModel
from models.ernie_moe import ErnieMoEForCausalLM, CausalLMOutputWithCrossAttentions
from models.image_encoder import EVAVisionTransformer, EVAVisionTransformerConfig
from models.dfnrope.modeling import (
    DFNRopeVisionTransformerConfig,
)
from models.longcontext_ops import TensorBalanceByTokenType

from .configuration import ErniemmMoEConfig

try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}

logger = logging.getLogger(__name__)

__all__ = [
    "ErniemmMoEForCausalLM",
]


class TokenType:
    """token type 含义"""

    text = 0
    image = 1
    video = 2
    audio = 3
    # text_in_image = 2


IDTYPES_2_ID = {"text": 0, "image": 1, "video": 2, "audio": 3}
IMAGETYPES_2_ID = {"image": 0, "video": 1, "padded_image": 2}


def monkey_patch_param_hook(param):
    """
    因为 paddle 的 tensor hook 机制，不支持调整 hook 的顺序，
    所以对需要特殊操作的 param hack了 `_register_grad_hook`的实现，使其能支持调整 hook 顺序
    """
    hook_list = []

    def _register_grad_hook(self, hook):
        nonlocal hook_list
        hook_list.append(hook)

    def hook_of_hook(g):
        nonlocal hook_list
        for h in hook_list:
            g = h(g)
        return g

    def hooks(self):
        nonlocal hook_list
        return hook_list

    def register_hook(self, hook, pos=None):
        nonlocal hook_list
        if pos is None:
            pos = len(hook_list)
        hook_list.insert(pos, hook)

        class _Remover:
            """hook remover"""

            def remove(self):
                """hook remover"""
                for i, h in enumerate(hook_list):
                    if h is hook:
                        break
                else:
                    logger.error(f"can not remove hook {hook} from: {hook_list}")
                    return False
                hook_list.pop(i)
                return True

        return _Remover()

    param._register_grad_hook(hook_of_hook)
    param._register_grad_hook = MethodType(_register_grad_hook, param)
    param.register_hook = MethodType(register_hook, param)
    param.hooks = MethodType(hooks, param)


def get_backbone_lm_param_regex(config):
    """
    返回一个可以匹配所有 *backbone 网络中（不含experts)*  *纯LM* 参数名的 regex
    """
    moe_rank = dist.get_rank(config.moe_group)
    moe_world_size = dist.get_world_size(config.moe_group)
    num_local_experts = (
        sum(config.moe_num_experts) // moe_world_size
        if config.multimodel_experts
        else config.moe_num_experts // moe_world_size
    )
    num_freeze_expert = (
        config.moe_num_experts[0]
        if config.multimodel_experts
        else config.moe_num_experts
    )

    # SA-moe
    num_local_experts_sa = config.moe_num_attn_experts // moe_world_size

    freeze_part = [r"ernie\.norm.*", r"ernie\.layers.*norm.*"]  # freeze all norm
    # we do not include gate weight
    # gate weight 已经进行了模态隔离
    freeze_part += [
        r"ernie\.layers\.(\d+)\.mlp\.(up_gate|gate|up|down)_proj\.*",
        r"ernie\.layers\.(\d+)\.mlp\.shared_experts\.(up_gate|gate|up|down)_proj\.*",
        r"ernie\.layers\.(\d+)\.self_attn.(q|k|v|o|qkv)_proj\.(weight|bias)",
        r"ernie\.layers\.(\d)+\.mlp\.gate\.weight$",
    ]
    logger.info(f"FREEZE_DEBUG: { moe_rank * num_local_experts} {num_freeze_expert}")
    freeze_part += [r"ernie\.embed_tokens\.weight"]
    freeze_part += [r"lm_head\.weight", r"lm_head\.bias"]

    assert freeze_part, f"not freeze any part, moe: {moe_rank}/{moe_world_size}"
    logger.info(f"freeze pattern: {freeze_part}, moe: {moe_rank}/{moe_world_size}")
    freeze_part = re.compile("|".join(freeze_part))
    return freeze_part


def create_freeze_hook(name, param, factor=0.0):
    """
    创造一个 hook，用于将 `param` 收到的 gradient 缩小 `factor` 倍
    """

    def _stopgrad_hook(g):
        # logger.info(f"hook trigered for: {name} {param.shape} {factor}: {g.shape}")
        #     debug_norm = g.astype("float32").norm()
        with paddle.no_grad():
            return g.scale_(factor)  # 必须用 inplace 版的操作.

    return _stopgrad_hook


def create_partial_freeze_hook(name, param, factor, index):
    """
    创造一个 hook，用于将 `param[...,:index]`部分收到的 gradient 缩小 `factor` 倍
    """

    def _stopgrad_hook(g):
        with paddle.no_grad():
            # debug_norm = g[..., index:].astype("float32").norm()
            # logger.info(f'partial-grad <{name}>, idx:{index}, {debug_norm}')
            g[:, :index] = g[:, :index] * factor
        return g

    return _stopgrad_hook


@paddle.no_grad()
def construct_types_for_video(image_mask, token_type_ids, image_type_ids):
    """
    construct_types_for_video,
    Args:
        image_mask: [B], 1 if is `im_patch_id` else 0
        token_type_ids: [B], see `IDTYPES_2_ID`
        image_type_ids: [B], 其中非零值有 `B_image_video` 个，标定每个 image patch 的含义，定义见 `IMAGETYPES_2_ID`
    Returns:
        image_is_video: [image_B,]的tensor，image_B是image_features的bsz。
                    里面的值是0/1，1是video，0是image。
        compressed_image_indices: [image_seq,]的tensor，其中image_seq是input_ids中image_placeholder的数量。
                                  里面的值是0/1，1是video，0是image。
        video_images_with_placeholder: [padded_video_b,]的tensor，其中padded_video_b是加了padding后的video的数量，
                                        即过temporal_linear前的batch size。里面的值是0/1，1是video，0是pad。
    """
    if image_type_ids is not None:
        image_type_ids = image_type_ids[image_type_ids >= 0]  # remove padding
        # conv3d前的placeholder，里面带着pad。具体1是video feature在位置，0是pad在的位置
        video_images_with_placeholder = image_type_ids[
            image_type_ids != IMAGETYPES_2_ID["image"]
        ]
        if video_images_with_placeholder.shape[0] != 0:
            video_images_with_placeholder = (
                video_images_with_placeholder == IMAGETYPES_2_ID["video"]
            )
            video_images_with_placeholder = video_images_with_placeholder.astype(
                "int64"
            )
        else:
            video_images_with_placeholder = None

        # image_is_video是visual feature的type id，用于在visual feature中取出video和image。具体1是video，0是普通image。
        image_is_video = image_type_ids[
            image_type_ids != IMAGETYPES_2_ID["padded_image"]
        ]

        assert (
            image_is_video.shape[0] != 0
        ), f"image_is_video is 0 shape, {image_is_video.shape}"

        image_is_video = image_is_video == IMAGETYPES_2_ID["video"]
        image_is_video = image_is_video.astype("int64")

    else:
        video_images_with_placeholder = None
        image_is_video = None

    # compressed_image_indices是最终压缩后的visual feature的type id，0是image，1是conv3d后的video。
    compressed_image_indices = token_type_ids[image_mask]
    compressed_image_indices = compressed_image_indices == TokenType.video
    compressed_image_indices = compressed_image_indices.astype("int64")

    return image_is_video, compressed_image_indices, video_images_with_placeholder


class VariableResolutionResamplerModel(nn.Layer):
    """
    ResamplerModel, 支持变分, 负责空间、时间维度缩并。
    """

    def __init__(self, in_dim, out_dim, spatial_conv_size, temporal_conv_size, config):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.config = config
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size
        self.use_recompute_resampler = config.use_recompute_resampler
        self.use_temporal_conv = config.use_temporal_conv
        self.tensor_parallel_degree = config.tensor_parallel_degree

        # for 空间四合一
        self.spatial_dim = self.in_dim * self.spatial_conv_size * self.spatial_conv_size
        # for 时间二合一
        self.temporal_dim = (
            self.in_dim
            * self.spatial_conv_size
            * self.spatial_conv_size
            * self.temporal_conv_size
        )

        with paddle.utils.unique_name.guard("mm_resampler_"):

            self.spatial_linear = nn.Sequential(
                (
                    RowSequenceParallelLinear(
                        self.spatial_dim,
                        self.spatial_dim,
                        input_is_parallel=True,
                        has_bias=True,
                        fuse_matmul_bias=True,
                    )
                    if config.tensor_parallel_degree > 1
                    else nn.Linear(self.spatial_dim, self.spatial_dim)
                ),
                nn.GELU(),
                nn.Linear(self.spatial_dim, self.spatial_dim),
                nn.LayerNorm(self.spatial_dim, epsilon=1e-6),
            )

            if self.use_temporal_conv:
                self.temporal_linear = nn.Sequential(
                    nn.Linear(self.temporal_dim, self.spatial_dim),
                    nn.GELU(),
                    nn.Linear(self.spatial_dim, self.spatial_dim),
                    nn.LayerNorm(self.spatial_dim, epsilon=1e-6),
                )

            self.mlp = nn.Linear(self.spatial_dim, self.out_dim)

            out_config = deepcopy(config)
            out_config.hidden_size = out_dim
            # Note(GuoxiaWang): fuse can reduce gpu peak memory
            out_config.fuse_rms_norm = out_config.resampler_fuse_rms_norm
            self.after_norm = RMSNorm(out_config)

            if config.tensor_parallel_degree > 1:
                for idx in [2, 3]:
                    mark_as_sequence_parallel_parameter(self.spatial_linear[idx].weight)
                    mark_as_sequence_parallel_parameter(self.spatial_linear[idx].bias)

                if self.use_temporal_conv:
                    for idx in [0, 2, 3]:
                        mark_as_sequence_parallel_parameter(
                            self.temporal_linear[idx].weight
                        )
                        mark_as_sequence_parallel_parameter(
                            self.temporal_linear[idx].bias
                        )

                mark_as_sequence_parallel_parameter(self.mlp.weight)
                mark_as_sequence_parallel_parameter(self.mlp.bias)
                mark_as_sequence_parallel_parameter(self.after_norm.weight)

    def spatial_conv_reshape(self, x, spatial_conv_size):
        """
        Linear 前的 reshape，为了让 Linear 能模仿 conv 的感受野
        """
        S, C = x.shape
        x = x.reshape([-1, C * (spatial_conv_size**2)])
        return x

    def forward(self, x, image_mask, token_type_ids, image_type_ids, grid_thw):
        """
        x: image_features
        image_mask: [B]
        token_types_ids: [B]
        image_type_ids:  [B_image]
        grid_thw: [B_image, 3]
        """
        assert image_type_ids is not None

        def fwd_spatial(x):
            """
            x in the shape of [S, H]
            S is ordered in the following way: [ [patch_h*patch_w (row-major traversal)] * patch_time]
            H is simply hidden
            """
            x = self.spatial_conv_reshape(x, self.spatial_conv_size)

            num_pad = 0
            if self.tensor_parallel_degree > 1:
                num_pad = (
                    (x.shape[0] + self.tensor_parallel_degree - 1)
                    // self.tensor_parallel_degree
                    * self.tensor_parallel_degree
                    - x.shape[0]
                )

            if num_pad > 0:
                x = paddle.nn.functional.pad(x, [0, num_pad, 0, 0])

            x = self.spatial_linear(x)

            if self.tensor_parallel_degree > 1:
                x = GatherOp.apply(x)

            if num_pad > 0:
                x = x[:-num_pad]
            return x

        def fwd_placeholder(x, grid_thw, to_tensor=False):
            """
            x: [S, H]
            grid_thw: [S, 3]
                其中第二维是: [t, h, w]
            """

            grid_thw_cpu = grid_thw.numpy()
            grid_t, grid_hw = grid_thw_cpu[:, 0], grid_thw_cpu[:, 1:]
            grid_hw_after_conv = grid_hw.prod(-1) // (self.spatial_conv_size**2)

            tokens_per_img_or_vid = grid_thw_cpu.prod(-1) // (self.spatial_conv_size**2)
            batch_offset = np.empty(
                tokens_per_img_or_vid.size, dtype=tokens_per_img_or_vid.dtype
            )
            batch_offset[0] = 0
            batch_offset[1:] = tokens_per_img_or_vid.cumsum()[:-1]

            assert (
                self.temporal_conv_size == 2
            ), f"Hard Code: temporal_conv_size==2, got:{self.temporal_conv_size}"

            # TODO: support any temporal conv size
            slice_offsets = []
            for temporoal_size, spatial_size, b_offset in zip(
                grid_t, grid_hw_after_conv, batch_offset
            ):
                for temp_offset in range(0, temporoal_size, 2):
                    slice_offsets.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets = paddle.to_tensor(np.concatenate(slice_offsets, axis=-1))

            slice_offsets2 = []
            for temporoal_size, spatial_size, b_offset in zip(
                grid_t, grid_hw_after_conv, batch_offset
            ):
                for temp_offset in range(
                    1 if temporoal_size > 1 else 0, temporoal_size, 2
                ):
                    slice_offsets2.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets2 = paddle.to_tensor(np.concatenate(slice_offsets2, axis=-1))

            x_timestep_1 = paddle.gather(x, slice_offsets, axis=0)
            x_timestep_2 = paddle.gather(x, slice_offsets2, axis=0)
            x = paddle.concat([x_timestep_1, x_timestep_2], axis=-1)

            return x

        def fwd_temporal(x):
            num_pad = 0
            if self.tensor_parallel_degree > 1:
                num_pad = (
                    (x.shape[0] + self.tensor_parallel_degree - 1)
                    // self.tensor_parallel_degree
                    * self.tensor_parallel_degree
                    - x.shape[0]
                )
            if num_pad > 0:
                x = paddle.nn.functional.pad(x, [0, num_pad, 0, 0])
            if self.tensor_parallel_degree > 1:
                x = ScatterOp.apply(x, axis=0)
            x = self.temporal_linear(x)

            if self.use_recompute_resampler:
                # GuoxiaWang: make recompute happy
                num_pad = paddle.to_tensor(num_pad)

            return x, num_pad

        def fwd_mlp(x):
            x = self.mlp(x)
            x = self.after_norm(x)
            if self.tensor_parallel_degree > 1:
                x = GatherOp.apply(x)
            return x

        num_pad = 0
        if self.use_recompute_resampler:
            x = recompute(fwd_spatial, x)
            if self.use_temporal_conv:
                x = recompute(fwd_placeholder, x, grid_thw)
                x, num_pad = recompute(fwd_temporal, x)
            x = recompute(fwd_mlp, x)
        else:
            x = fwd_spatial(x)
            if self.use_temporal_conv:
                x = fwd_placeholder(x, grid_thw)
                x, num_pad = fwd_temporal(x)
            x = fwd_mlp(x)
        if num_pad is not None and num_pad > 0:
            x = x[:-num_pad]
        return x

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):

        from paddleformers.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )
        res = {"spatial_linear.0.weight": partial(fn, is_column=False)}
        cast_to_tensor_if_needed = lambda x: (
            x if isinstance(x, (paddle.Tensor, np.ndarray)) else x[:]
        )
        for k in (
            "spatial_linear.0.bias",  # row linear bias
            "spatial_linear.2.weight",
            "spatial_linear.2.bias",  # linear
            "spatial_linear.3.weight",
            "spatial_linear.3.bias",  # layernorm
            "temporal_linear.0.weight",
            "temporal_linear.0.weight",  # linear
            "temporal_linear.2.weight",
            "temporal_linear.2.bias",  # linear
            "temporal_linear.3.weight",
            "temporal_linear.3.bias",  # bias
        ):
            res.update({k: cast_to_tensor_if_needed})
        return res


class ResamplerModel(nn.Layer):
    """
    ResamplerModel, 负责空间、时间维度缩并。
    """

    def __init__(self, in_dim, out_dim, spatial_conv_size, temporal_conv_size, config):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size
        self.use_recompute_resampler = config.use_recompute_resampler
        with paddle.utils.unique_name.guard("mm_resampler_"):
            self.spatial_linear = RowParallelLinear(
                self.in_dim * self.spatial_conv_size * self.spatial_conv_size,
                self.out_dim,
                input_is_parallel=True,
                has_bias=True,
                fuse_matmul_bias=True,
            )
            self.act_fn = nn.Silu()
            self.temporal_linear = ColumnParallelLinear(
                self.out_dim * self.temporal_conv_size,
                self.out_dim,
                gather_output=True,
                has_bias=True,
                fuse_matmul_bias=True,
            )
            out_config = deepcopy(config)
            out_config.hidden_size = out_dim
            # Note(GuoxiaWang): fuse can reduce gpu peak memory
            out_config.fuse_rms_norm = out_config.resampler_fuse_rms_norm
            self.after_norm = RMSNorm(out_config)

    def spatial_conv_reshape(self, x, spatial_conv_size):
        """
        Linear 前的 reshape，为了让 Linear 能模仿 conv 的感受野
        """
        B, S, C = x.shape
        H = int(math.sqrt(S))

        # x = x.reshape([B, H, H, C])  # B, H, W, C
        x = x.reshape(
            [B, H, H // spatial_conv_size, int(C * spatial_conv_size)]
        )  # N, H, W/conv, C*conv
        x = x.transpose([0, 2, 1, 3])  # N, W/conv, H, C*conv
        x = x.reshape(
            [
                B,
                H // spatial_conv_size,
                H // spatial_conv_size,
                int(C * spatial_conv_size * spatial_conv_size),
            ]
        )  # N, W/conv, H/conv, C*conv*conv
        x = x.transpose([0, 2, 1, 3])  # N, H/conv, W/conv, C*conv*conv

        x = x.reshape(
            [B, -1, int(C * spatial_conv_size * spatial_conv_size)]
        )  # N, (H * W)/(conv * conv), C*conv*conv

        return x

    def forward(
        self,
        x,
        image_mask,
        token_type_ids,
        image_type_ids,
        grid_thw=None,
    ):
        """
        x: image_features
        image_mask: [B]
        token_types_ids: [B]
        image_type_ids:  [B_image]
        """
        assert image_type_ids is not None
        (
            image_is_video,
            compressed_image_indices,
            video_images_with_placeholder,
        ) = construct_types_for_video(image_mask, token_type_ids, image_type_ids)
        assert len(compressed_image_indices) == len(image_mask[image_mask]), (
            f"len(compressed_image_indices): {compressed_image_indices.shape}, "
            f"len(image_mask): {image_mask.astype('int64').sum()} "
        )

        def fwd_spatial(x):
            x = self.spatial_conv_reshape(x, self.spatial_conv_size)
            x = self.spatial_linear(x)
            x = self.act_fn(x)
            return x

        def fwd_placeholder(x, to_tensor=False):
            # separate image and video from features, # B, S, C
            video_indices = paddle.nonzero(image_is_video == 1).flatten()
            image_features_video = paddle.gather(x, video_indices)

            nonvideo_indices = paddle.nonzero(image_is_video == 0).flatten()
            image_features_nonvideo = paddle.gather(x, nonvideo_indices)

            if image_features_video.shape[0] != 0:
                # use video_images_with_placeholder to construct placeholder
                video_placeholder = paddle.zeros(
                    video_images_with_placeholder.shape[0:1]
                    + image_features_video.shape[1:],
                    dtype=image_features_video.dtype,
                )
                indices = paddle.nonzero(video_images_with_placeholder == 1).flatten()
                video_placeholder = paddle.scatter_(
                    video_placeholder, indices, image_features_video, overwrite=True
                )
                B_video_placeholder = (
                    video_placeholder.shape[0] // self.temporal_conv_size
                )
            else:
                B_video_placeholder = 0

            if image_features_nonvideo.shape[0] != 0:
                # same padding for image
                # TODO: check repeat_interleave
                image_placeholder = paddle.repeat_interleave(
                    image_features_nonvideo, self.temporal_conv_size, 0
                )
                B_image_placeholder = (
                    image_placeholder.shape[0] // self.temporal_conv_size
                )
            else:
                B_image_placeholder = 0

            if B_video_placeholder != 0 and B_image_placeholder != 0:
                # merge video and image
                placeholder = paddle.concat(
                    [image_placeholder, video_placeholder], axis=0
                )
            elif B_video_placeholder != 0:
                placeholder = video_placeholder
            elif B_image_placeholder != 0:
                placeholder = image_placeholder
            else:
                raise ValueError(
                    f"there is no image and video! image_is_video:{image_is_video}, "
                    f"compressed_image_indices:{compressed_image_indices}, "
                    f"video_images_with_placeholder:{video_images_with_placeholder}"
                )

            if to_tensor:
                # make recompute happy
                return (
                    placeholder,
                    paddle.to_tensor(B_video_placeholder),
                    paddle.to_tensor(B_image_placeholder),
                )
            else:
                return placeholder, B_video_placeholder, B_image_placeholder

        def fwd_temporal(placeholder):
            # TODO: check this
            B, S, C = placeholder.shape
            placeholder = placeholder.transpose([1, 0, 2])  # S, B, C
            placeholder = placeholder.reshape(
                [S, B // self.temporal_conv_size, int(C * self.temporal_conv_size)]
            )  # S, B/Sconv, C*Sconv
            placeholder = self.temporal_linear(placeholder)  # S, B/Sconv, C*Sconv
            placeholder = placeholder.transpose([1, 0, 2])  # B/Sconv, S, C*Sconv
            placeholder = self.after_norm(placeholder)  # B/Sconv, S, C*Sconv
            return placeholder

        def fwd_index_put(
            placeholder,
            B_video_placeholder,
            B_image_placeholder,
            compressed_image_indices,
        ):
            # # separate video and image from placeholder and put them into compressed placeholder
            compressed_placeholder = paddle.zeros(
                compressed_image_indices.shape[0:1] + [placeholder.shape[-1]],
                dtype=placeholder.dtype,
            )
            if B_video_placeholder != 0:
                compressed_video = placeholder[B_image_placeholder:, ...].reshape(
                    [-1, placeholder.shape[-1]]
                )
                indices = paddle.nonzero(compressed_image_indices == 1).flatten()
                compressed_placeholder = paddle.scatter_(
                    compressed_placeholder, indices, compressed_video, overwrite=True
                )
            if B_image_placeholder != 0:
                compressed_image = placeholder[:B_image_placeholder, ...].reshape(
                    [-1, placeholder.shape[-1]]
                )
                indices = paddle.nonzero(compressed_image_indices == 0).flatten()
                compressed_placeholder = paddle.scatter_(
                    compressed_placeholder, indices, compressed_image, overwrite=True
                )

            return compressed_placeholder

        if self.use_recompute_resampler:
            x = recompute(fwd_spatial, x)
            placeholder, B_video_placeholder, B_image_placeholder = recompute(
                fwd_placeholder, x, to_tensor=True
            )
            placeholder = recompute(fwd_temporal, placeholder)
            compressed_placeholder = recompute(
                fwd_index_put,
                placeholder,
                B_video_placeholder,
                B_image_placeholder,
                compressed_image_indices,
            )
        else:
            x = fwd_spatial(x)
            placeholder, B_video_placeholder, B_image_placeholder = fwd_placeholder(x)
            placeholder = fwd_temporal(placeholder)
            compressed_placeholder = fwd_index_put(
                placeholder,
                B_video_placeholder,
                B_image_placeholder,
                compressed_image_indices,
            )

        return compressed_placeholder


class ErniePretrainingCriterion(ErniePretrainingCriterionBase):
    """
    继承顺序：
    ErnieMMMoE -> ErnieMoE -> Ernie
    """

    def __init__(self, config):
        super().__init__(config)
        self.im_patch_id = config.im_patch_id
        self.max_text_id = config.max_text_id
        self.use_one_head = config.mm_vocab_size == 0 and config.audio_config is None
        if config.audio_config is not None:
            self.audio_frame_depth = config.audio_config["audio_frame_depth"]
            self.audio_decode_frame_depth = config.audio_config.get(
                "audio_decode_frame_depth", self.audio_frame_depth
            )
            assert (
                self.audio_decode_frame_depth == 1
            ), "为了方便复用token_balance_loss计算逻辑,目前只支持解码深度为1"

    def forward(
        self,
        scores_text,
        scores_image,
        labels,
        token_type_ids_shifted,
        token_type_ids_untouched,
        scores_audio=None,
        labels_audio=None,
        lm_weight=None,
        lm_bias=None,
        mm_weight=None,
        mm_bias=None,
        router_loss=None,
    ):
        """
        文图分离 Criterion,只返回文本 样本的CEloss，其他 loss 会 update 到 `global_training_logs` 里。
        Args:
            score_text: 文本 logits，只包含纯文数据。
            scores_text_in_image: 只包含文本数据 logits。
            scores_image: 只包含图片 sepecial-token 的 logits。
            labels: 原始 label，可能包含文本、图片 sepecial-token、ignored-index。
            token_type_ids_shifted: `labels` 对应的 token-type。
            router_loss: router_loss
        Returns:
            loss: 纯文样本的 CE loss
            loss_sum. 纯文样本的 CE loss_sum
        """
        if self.config.use_recompute_loss_fn and self.config.use_fused_head_loss_fn:
            has_tp = (
                hasattr(fleet.fleet, "_hcg")
                and fleet.get_hybrid_communicate_group().get_model_parallel_world_size()
                > 1
            )
            with paddle.no_grad():
                if token_type_ids_shifted.unique().shape[0] > 1 and has_tp:
                    labels, token_type_ids_shifted = TensorBalanceByTokenType.apply(
                        labels.squeeze(0),
                        token_type_ids_shifted,
                        is_tensor_sharded=False,
                    )
                else:
                    labels = ScatterOp.apply(labels, axis=-1)

        if self.use_one_head:
            assert scores_audio is None and scores_image is None, (
                scores_audio,
                scores_image,
            )
            if self.config.use_recompute_loss_fn:
                loss, loss_sum = super().forward(
                    (scores_text.unsqueeze(0), lm_weight, lm_bias), labels.unsqueeze(0)
                )
            else:
                loss, loss_sum = super().forward(
                    scores_text.unsqueeze(0), labels.unsqueeze(0)
                )
            self.update_log(loss, token_type_ids_untouched)
            return loss, loss_sum

        assert (
            scores_text is not None
        ), f"no text or image token provided, text: {scores_text}"

        if scores_text is not None:
            text_pos_shifted = token_type_ids_shifted == TokenType.text
            labels_text = labels[text_pos_shifted]
            assert labels_text.size > 0, labels
            assert (
                labels_text.shape[0] == scores_text.shape[0]
            ), f"labels_text.shape = {labels_text.shape}, scores_text.shape = {scores_text.shape}"
            if self.config.use_recompute_loss_fn:
                assert lm_weight is not None and mm_weight is not None
                loss, loss_sum = super().forward(
                    (scores_text.unsqueeze(0), lm_weight, lm_bias),
                    labels_text.unsqueeze(0),
                )
            else:
                loss, loss_sum = super().forward(
                    scores_text.unsqueeze(0), labels_text.unsqueeze(0)
                )
            self.update_log(loss, token_type_ids_untouched)
        else:
            assert 0
            loss = paddle.zeros([], dtype="float32")
            loss.stop_gradient = False

        if scores_image is not None:
            image_mask_shifted = token_type_ids_shifted == TokenType.image
            labels_image = labels[image_mask_shifted]
            assert labels_image.size > 0, labels
            assert (
                labels_image.shape[0] == scores_image.shape[0]
            ), f"labels_image.shape = {labels_image.shape}, scores_image.shape = {scores_image.shape}"
            labels_image = paddle.where(
                labels_image >= 0, labels_image - self.max_text_id, labels_image
            )  # do not move ignored-index
            if self.config.use_recompute_loss_fn:
                assert mm_weight is not None and mm_bias is not None
                loss_image, _ = super().forward(
                    (scores_image.unsqueeze(0), mm_weight, mm_bias),
                    labels_image.unsqueeze(0),
                )
            else:
                loss_image, _ = super().forward(
                    scores_image.unsqueeze(0), labels_image.unsqueeze(0)
                )
            global_training_logs.update(image_special_token_loss=loss_image.detach())
            loss = loss + loss_image - loss_image.detach()

        if scores_audio is not None and labels_audio is not None:
            audio_mask_shifted = token_type_ids_shifted == TokenType.audio
            # 这里labels_audio.shape = [max_seq_length, depth]
            # 为了让输入定长，在collate_func 里面被补充PAD了很多ignored_index
            # 但是实际帧数应该和token_type_ids_shifted 中音频token数量一致
            labels_audio = labels_audio.clone()
            audio_frame_num = paddle.sum(
                audio_mask_shifted.astype("int64")
            )  # 首先计算实际音频帧数
            labels_audio = labels_audio.reshape([-1, self.audio_decode_frame_depth])
            labels_audio = labels_audio[:audio_frame_num, :].reshape(
                [-1]
            )  # 取出实际音频帧，排除PAD部分
            assert (
                scores_audio.shape[0] == labels_audio.shape[0]
            ), f"labels_audio.shape = {labels_audio.shape}, scores_audio.shape = {scores_audio.shape}"

            lossmask_audio = labels_audio != self.ignored_index
            # 至少存在1个音频没被mask
            if lossmask_audio.any():
                loss_audio, _ = super().forward(
                    scores_audio.unsqueeze(0), labels_audio.unsqueeze(0)
                )
                loss_audio_val = loss_audio.detach()
                loss = loss + loss_audio - loss_audio.detach()
                global_training_logs.update(audio_data_audio_token_loss=loss_audio_val)

        if router_loss is not None and isinstance(router_loss, paddle.Tensor):
            global_training_logs.update(router_loss=router_loss.detach())
            loss = loss + router_loss - router_loss.detach()
        return loss, loss_sum

    def update_log(self, loss, token_type_ids_untouched):
        """update log"""
        pure_text = (
            token_type_ids_untouched == TokenType.text
        ).all()  # 所有token都是文本token，则为纯文数据
        has_video = (
            token_type_ids_untouched == TokenType.video
        ).any()  # 只要有1个视频token，则为视频数据
        has_image = (
            token_type_ids_untouched == TokenType.image
        ).any()  # 只要有1个视频token，则为图片数据
        has_audio = (
            token_type_ids_untouched == TokenType.audio
        ).any()  # 只要有1个音频token，则为音频数据

        if pure_text:
            global_training_logs.update(lm_loss=loss.detach())
        elif has_video:
            global_training_logs.update(video_loss=loss.detach())
        elif has_image:
            global_training_logs.update(image_loss=loss.detach())
        elif has_audio:
            # 由于音频同时存在理解和生成，直接用audio_loss 会混淆，所以用 audio_data_text_token_loss 表示音频理解loss
            global_training_logs.update(audio_data_text_token_loss=loss.detach())
        else:
            raise RuntimeError(
                f"输入token 必须是 [text, video, image, audio] 之一: {token_type_ids_untouched}"
            )
        return


class AudioLayerNorm(LayerNorm):
    """
    音频LayerNorm,主要是为了修改音频layernorm的名字
    """

    pass


class AudioEmbedding(VocabParallelEmbedding):
    """
    音频Embedding，和nn.Embedding 的主要差别是，audio_input_ids 存在深度维度（或者说残差维度），
    不同深度不共享词表，因此单独实现一版适合音频输入的Embedding
    """

    def __init__(self, config):
        self.config = config
        self.depth = config.audio_config["audio_encode_frame_depth"]
        self.vocab_size = config.audio_config["audio_vocab_size"]
        self.hidden_size = config.audio_config["audio_hidden_size"]

        if isinstance(self.vocab_size, int):
            self.vocab_size = [self.vocab_size] * self.depth
        assert isinstance(self.vocab_size, list)

        self.offset, full_vocab_size = [], 0
        for depth_vocab_size in self.vocab_size:
            self.offset.append(full_vocab_size)
            full_vocab_size += depth_vocab_size
        super().__init__(full_vocab_size, self.hidden_size)
        self.offset = np.array(self.offset, dtype=np.int64).reshape([1, self.depth])
        self.offset = paddle.to_tensor(self.offset, dtype="int64")

    def forward(self, audio_ids):
        """
        实际实现时，还是基于nn.Embedding，但是不同深度会加上offset，以此隔离不同深度audio_input_ids
        """
        assert len(audio_ids.shape) == 2
        assert audio_ids.shape[-1] == self.depth

        audio_ids = audio_ids[audio_ids >= 0]
        audio_ids = audio_ids.reshape([-1, self.depth])
        L, D = audio_ids.shape
        audio_ids = audio_ids + self.offset
        input_embeds = super().forward(audio_ids.reshape([-1]))
        input_embeds = input_embeds.reshape([L, D, self.hidden_size])
        return input_embeds


class AudioDepthOutputModule(nn.Layer):
    """
    音频 Local Transoformer 组网输出模块。
    音频模型参数、主要计算逻辑，都封装在这里。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.audio_out_adaptor = nn.Linear(
            config.hidden_size,
            config.audio_config["audio_hidden_size"],
        )

        self.parallel_matmul_tp = partial(
            parallel_matmul,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_output=False,
            fuse_linear=config.fuse_linear,
        )

        self.depth_transformer_config_dict = self.config.audio_config[
            "audio_depth_transformer"
        ]
        assert (
            self.depth_transformer_config_dict
        ), "使用depth transformer时，必须给定audio_depth_transformer"
        depth_transformer_config = copy.deepcopy(config)
        (
            depth_transformer_config.seqlen,
            depth_transformer_config.hidden_size,
            depth_transformer_config.num_attention_heads,
            depth_transformer_config.intermediate_size,
            depth_transformer_config.num_hidden_layers,
        ) = (
            config.audio_config["audio_frame_depth"],
            config.audio_config["audio_hidden_size"],
            self.depth_transformer_config_dict["num_attention_heads"],
            self.depth_transformer_config_dict["intermediate_size"],
            self.depth_transformer_config_dict["num_hidden_layers"],
        )
        depth_transformer_config.sequence_parallel = False
        self.audio_depth_transformer = ErnieModel(depth_transformer_config)
        self.audio_depth_transformer.embed_tokens = (
            None  # 音频Embedding 有独立实现，不复用ErnieModel 里面的实现
        )

        audio_vocab_config = deepcopy(config)
        audio_vocab_config.hidden_size = config.audio_config["audio_hidden_size"]
        audio_vocab_config.vocab_size = config.audio_config["audio_vocab_size"]
        assert audio_vocab_config.vocab_size > 0, audio_vocab_config
        self.audio_head = nn.LayerList(
            [
                ErnieLMHead(audio_vocab_config)
                for _ in range(config.audio_config["audio_frame_depth"])
            ]
        )

    def forward(
        self,
        audio_features,
        audio_hidden_state,
    ):
        """
        包含3步:
        1. audio_out_adaptor: 必有。用于对齐音频token hidden_size 。
        2. depth_transformer: 可选。通过配置，可以不使用这部分组网，配置不设置audio_depth_transformer 即可。
        3. audio_head: 必有。用于不同深度的音频token输出。
        """
        # audio_hidden_state.shape = [F, LH]   F表示音频帧数,严格 == paddle.sum(shift_audio_mask)
        # audio_hidden_state_proj = [F, AH]
        audio_hidden_state_proj = self.audio_out_adaptor(audio_hidden_state)

        # 构造Depth Transformer 输入(teacher forcing)
        # audio_hidden_state_proj.shape = [F, 1, AH]
        # audio_features.shape = [F, D, AH]
        # depth_input_embeds.shape = [F, D, AH]
        # depth_input_embeds = [audio_hidden_state, audio_feat_1, audio_feat_2, audio_feat_3, ...]
        audio_hidden_state_proj = audio_hidden_state_proj.unsqueeze(1)
        depth_input_embeds = paddle.concat(
            [audio_hidden_state_proj, audio_features], axis=1
        )[:, :-1, :]

        # audio_depth_hidden_state.shape = [F, D, AH]
        audio_depth_outputs = self.audio_depth_transformer(
            inputs_embeds=depth_input_embeds, return_dict=True
        )
        audio_depth_hidden_state = audio_depth_outputs.last_hidden_state

        score_audio = []
        for i in range(len(self.audio_head)):
            x = audio_depth_hidden_state[:, i, :].contiguous()  # shape = [F, AH]
            w = self.audio_head[i].weight  # shape = [AH, AV / mp]
            b = self.audio_head[i].bias  # shape = [AV / mp]
            o = self.parallel_matmul_tp(x, w, b).unsqueeze(1)  # shape = [F, 1, AV / mp]
            score_audio.append(o)
        score_audio = paddle.concat(score_audio, axis=1)  # shape = [F, D, AV / mp]
        score_audio = score_audio.reshape(
            [-1, score_audio.shape[-1]]
        )  # shape = [F * D, AV / mp]

        return score_audio


class AudioMultiHeadOutputModule(ErnieLMHead):
    """
    音频 MultiHead 组网输出模块。
    音频模型参数、主要计算逻辑，都封装在这里。
    """

    def __init__(self, config):
        # 在multihead 的场景下，将多个audio_head 合并为一个大audio_head，加速计算。
        audio_vocab_config = deepcopy(config)
        # 前置已经allgather，强给sequence_parallel
        audio_vocab_config.sequence_parallel = False
        audio_vocab_config.use_recompute_loss_fn = False
        audio_vocab_config.vocab_size = (
            config.audio_config["audio_vocab_size"]
            * config.audio_config["audio_frame_depth"]
        )
        if isinstance(config.audio_config["audio_vocab_size"], int):
            audio_vocab_config.vocab_size = (
                config.audio_config["audio_vocab_size"]
                * config.audio_config["audio_decode_frame_depth"]
            )
        elif isinstance(config.audio_config["audio_vocab_size"], list):
            assert config.audio_config["audio_vocab_size"][0] * config.audio_config[
                "audio_decode_frame_depth"
            ] == sum(
                config.audio_config["audio_vocab_size"][
                    : config.audio_config["audio_decode_frame_depth"]
                ]
            ), "暂不支持音频token decode时不同深度vocab_size 变化"
            audio_vocab_config.vocab_size = (
                config.audio_config["audio_vocab_size"][0]
                * config.audio_config["audio_decode_frame_depth"]
            )
        else:
            raise RuntimeError("audio_vocab_size 参数类型错误")

        assert audio_vocab_config.vocab_size > 0, audio_vocab_config
        super().__init__(audio_vocab_config)
        self.config = audio_vocab_config

    def forward(
        self,
        audio_hidden_state,
    ):
        """
        包含3步:
        3. audio_head: 必有。用于不同深度的音频token输出。
        """
        score_audio = super().forward(audio_hidden_state)
        F, AVD = score_audio.shape
        D = self.config.audio_config["audio_decode_frame_depth"]
        score_audio = score_audio.reshape([F * D, AVD // D])  # shape = [F * D, AV]
        return score_audio


def calc_multimodal_logits(
    last_hidden_state: paddle.Tensor,
    lm_head_weight: paddle.Tensor,
    lm_head_bias: paddle.Tensor,
    mm_head_weight: paddle.Tensor,
    mm_head_bias: paddle.Tensor,
    token_type_ids_shifted: paddle.Tensor,
    config: ErniemmMoEConfig,
    audio_features: paddle.Tensor = None,
    audio_module=None,
):
    """
    分别计算 纯文、图文、图片 位置的 logits
    Args:
        last_hidden_state: 最后一层的 hidden，在 sequence-parallel下，处于切分状态。
        ...
        token_type_ids_shifted: # 非 sp 切分tensor
            label 位置的 token-type-ids，用于选择每个 token 对应的 lm-head。
            注意：图文交替的 id 序列中，最后一个 文本 token 会预测 图id，反之亦然，
            所以需要选择 label type 对应的 lmhead weight。
    """
    # 将 ids 的类型对齐 label的类型，对于最后一个 ids，认为token type不变
    # TODO:从 reader 中传入 token-type-ids
    # token_type_ids_shifted = paddle.concat([token_type_ids[:, 1:], token_type_ids[:, -1:]], 1)  #

    if config.use_recompute_loss_fn and config.use_fused_head_loss_fn:
        has_tp = (
            hasattr(fleet.fleet, "_hcg")
            and fleet.get_hybrid_communicate_group().get_model_parallel_world_size() > 1
        )
        if config.sequence_parallel:
            if token_type_ids_shifted.unique().shape[0] > 1 and has_tp:  # 多模数据
                last_hidden_state, token_type_ids_shifted = (
                    TensorBalanceByTokenType.apply(
                        last_hidden_state, token_type_ids_shifted
                    )
                )
            else:
                with paddle.no_grad():
                    token_type_ids_shifted = ScatterOp.apply(
                        token_type_ids_shifted, axis=-1
                    )
                    token_type_ids_shifted = token_type_ids_shifted.reshape([-1])
        else:
            token_type_ids_shifted = token_type_ids_shifted.reshape([-1])
    else:
        if config.sequence_parallel:
            last_hidden_state = GatherOp.apply(last_hidden_state)
            last_hidden_state = last_hidden_state.reshape(
                [-1, config.seqlen, last_hidden_state.shape[-1]]
            )

        assert last_hidden_state.shape[:2] == token_type_ids_shifted.shape, (
            last_hidden_state.shape,
            token_type_ids_shifted.shape,
        )
    parallel_matmul_tp = partial(
        parallel_matmul,
        tensor_parallel_degree=config.tensor_parallel_degree,
        tensor_parallel_output=config.tensor_parallel_output,
        fuse_linear=config.fuse_linear,
        transpose_y=config.tie_word_embeddings,
    )

    if mm_head_weight is None and audio_module is None:
        # assert audio_module is None, f"audio should use 1 head when has mm_vocab_size=0"
        if config.use_recompute_loss_fn:
            return last_hidden_state, None, None
        score_text = parallel_matmul_tp(
            last_hidden_state,
            lm_head_weight,
            lm_head_bias,
            transpose_y=config.tie_word_embeddings,
        )
        return score_text, None, None

    image_mask_shifted = token_type_ids_shifted == TokenType.image
    audio_mask_shifted = token_type_ids_shifted == TokenType.audio
    text_pos_shifted = token_type_ids_shifted == TokenType.text

    if text_pos_shifted.any().item() > 0:
        if config.use_recompute_loss_fn:
            score_text = last_hidden_state[text_pos_shifted]
        else:
            score_text = parallel_matmul_tp(
                last_hidden_state[text_pos_shifted], lm_head_weight, lm_head_bias
            )
    else:
        score_text = None

    if mm_head_weight is not None and image_mask_shifted.any().item() > 0:
        if config.use_recompute_loss_fn:
            score_image = last_hidden_state[image_mask_shifted]
        else:
            score_image = parallel_matmul_tp(
                last_hidden_state[image_mask_shifted], mm_head_weight, mm_head_bias
            )
    else:
        score_image = None

    if audio_mask_shifted.any().item() > 0 and audio_module is not None:
        # last_hidden_state.shape = [B, L, LH]
        # audio_hidden_state.shape = [F, LH]
        audio_hidden_state = last_hidden_state[audio_mask_shifted]
        # score_audio.shape = [F * D, AV / mp]
        if isinstance(audio_module, AudioMultiHeadOutputModule):
            score_audio = audio_module.forward(audio_hidden_state)
        elif isinstance(audio_module, AudioDepthOutputModule):
            assert audio_features is not None
            score_audio = audio_module.forward(audio_features, audio_hidden_state)
        else:
            raise RuntimeError(f"unknown audio_out_module class: {type(audio_module)}")
    else:
        score_audio = None
    return score_text, score_image, score_audio


class ErniemmMoEHead(ErnieLMHead):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.sequence_parallel = config.sequence_parallel
        if config.mm_vocab_size > 0:
            mm_vocab_config = deepcopy(config)
            mm_vocab_config.vocab_size = config.mm_vocab_size
            assert mm_vocab_config.vocab_size > 0, mm_vocab_config
            assert (
                mm_vocab_config.im_patch_id >= mm_vocab_config.max_text_id
            ), mm_vocab_config
            self.mm_head = ErnieLMHead(mm_vocab_config)
        else:
            self.mm_head = None
        if config.audio_config is not None:
            if config.audio_config.get("output_module", "multihead") == "multihead":
                self.audio_out_module = AudioMultiHeadOutputModule(deepcopy(config))
            else:
                self.audio_out_module = AudioDepthOutputModule(deepcopy(config))
        else:
            self.audio_out_module = None

    def forward(
        self, hidden_state, token_type_ids_labels, audio_features=None, use_cache=False
    ):
        """_summary_

        Args:
            hidden_state (_type_): _description_
            token_type_ids_labels (_type_): _description_
            autio_features (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if not use_cache:
            mm_head_weight = self.mm_head.weight if self.mm_head is not None else None
            mm_head_bias = self.mm_head.bias if self.mm_head is not None else None
            logits_text, logits_image, logits_audio = calc_multimodal_logits(
                hidden_state,
                self.weight,
                self.bias,
                mm_head_weight,
                mm_head_bias,
                token_type_ids_labels,
                self.config,
                audio_features,
                self.audio_out_module,
            )
            return logits_text, logits_image, logits_audio
        else:
            if self.config.sequence_parallel:
                hidden_state = GatherOp.apply(hidden_state)
                logger.warning(
                    "you are trying to generate with sequence-parallel model"
                )
                hidden_state = hidden_state.reshape(
                    [-1, self.config.seqlen, hidden_state.shape[-1]]
                )
            # assert not self.config.sequence_parallel, "generate is not supported in sequence-parallel mode"
            # TODO，当前只支持lm_head decode
            return (
                parallel_matmul(
                    hidden_state[:, -1:, :],
                    self.weight,
                    self.bias,
                    transpose_y=self.config.tie_word_embeddings,
                    tensor_parallel_degree=self.config.tensor_parallel_degree,
                    tensor_parallel_output=False,
                    fuse_linear=self.config.fuse_linear,
                ),
                None,
                None,
            )


class ErniemmMoEForCausalLM(ErnieMoEForCausalLM):
    """ErniemmForCausalLM"""

    config_class = ErniemmMoEConfig
    main_input_name = "pixel_values"

    def __init__(
        self, config: ErniemmMoEConfig, vision_model=None, resampler_model=None
    ):
        super().__init__(config)
        self.criterion = ErniePretrainingCriterion(config)  # 复写

        if config.mm_vocab_size > 0:
            if config.tensor_parallel_degree > 1:
                self.ernie.mm_embed_tokens = VocabParallelEmbedding(
                    config.mm_vocab_size, config.hidden_size
                )
            else:
                self.ernie.mm_embed_tokens = nn.Embedding(
                    config.mm_vocab_size, config.hidden_size
                )
        else:
            self.ernie.mm_embed_tokens = None
        if isinstance(config.vision_config, EVAVisionTransformerConfig):
            logger.info(f"update vision config {config.use_recompute_attn_vision}")
            config.vision_config.use_recompute_attn = config.use_recompute_attn_vision

        elif isinstance(config.vision_config, DFNRopeVisionTransformerConfig):
            logger.info("variable resolution vision model")
            config.vision_config.variable_resolution = True

        resampler_cls = (
            VariableResolutionResamplerModel
            if getattr(config.vision_config, "variable_resolution", False)
            else ResamplerModel
        )
        self.ernie.resampler_model = resampler_cls(
            (
                config.inception_config.hidden_size
                if getattr(config, "inception_config", False)
                else config.pixel_hidden_size
            ),
            config.hidden_size,
            config.spatial_conv_size,
            config.temporal_conv_size,
            config=config,
        )

        self._modality_param_mapping = None
        self.image_preprocess = None

        if config.audio_config is not None:
            self.ernie.audio_embed_tokens = AudioEmbedding(config)
            with paddle.utils.unique_name.guard("audio_adapter_"):
                if self.config.tensor_parallel_degree > 1:
                    self.ernie.audio_adaptor = ColumnParallelLinear(
                        config.audio_config["audio_hidden_size"],
                        config.hidden_size,
                        gather_output=True,
                        has_bias=True,
                        fuse_matmul_bias=True,
                    )
                else:
                    self.ernie.audio_adaptor = nn.Linear(
                        config.audio_config["audio_hidden_size"],
                        config.hidden_size,
                    )
            Norm = RMSNorm if config.use_rmsnorm else LayerNorm
            if not config.use_rmsnorm and config.fuse_ln:
                Norm = FusedLayerNorm
            if isinstance(Norm, LayerNorm):
                self.ernie.audio_after_norm = AudioLayerNorm(config)
            else:
                self.ernie.audio_after_norm = Norm(config)
        else:
            self.ernie.audio_embed_tokens = None
            self.ernie.audio_after_norm = None
        self.lm_head = ErniemmMoEHead(config)
        self.tie_weights()

    def add_vision_model(
        self,
        encoder: nn.Layer,
    ):
        """add_vision_model"""
        self.vision_model = encoder
        self._set_modality_param_mapping()

    def add_image_preprocess(self, preprocess):
        """_summary_

        Args:
            preprocess (_type_): _description_
        """
        logger.info("image preprocess is set")
        self.image_preprocess = preprocess

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):

        from paddleformers.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = ErnieMoEForCausalLM._get_tensor_parallel_mappings(
                config, is_split=is_split
            )
            if config.vision_config is not None and isinstance(
                config.vision_config, EVAVisionTransformerConfig
            ):
                vision_actions = EVAVisionTransformer._get_tensor_parallel_mappings(
                    config.vision_config, is_split=is_split
                )
                vision_actions = {
                    f"vision_model.{k}": v for k, v in vision_actions.items()
                }
                final_actions.update(vision_actions)
            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)
        if isinstance(config.vision_config, DFNRopeVisionTransformerConfig):
            resampler_actions = (
                VariableResolutionResamplerModel._get_tensor_parallel_mappings(
                    config, is_split=is_split
                )
            )
            mappings.update(
                {f"resampler_model.{k}": v for k, v in resampler_actions.items()}
            )
        if config.mm_vocab_size > 0:
            mappings.update(
                {
                    "mm_embed_tokens.weight": partial(fn, is_column=False),
                    "lm_head.mm_head.weight": partial(fn, is_column=True),
                    "lm_head.mm_head.bias": partial(fn, is_column=True),
                }
            )
        return mappings

    def _set_modality_param_mapping(self):
        """_summary_"""
        lm_pattern = get_backbone_lm_param_regex(self.config)
        self._modality_param_mapping = defaultdict(lambda: [])
        for name, param in self.named_parameters():
            monkey_patch_param_hook(param)
            expert_type = getattr(param, "expert_type", None)
            if "vision_model" in name:
                self._modality_param_mapping["vit"].append(
                    (name, param, create_freeze_hook(name, param))
                )
                setattr(param, "color", "vit")
            elif expert_type == "expert_type_3":
                self._modality_param_mapping["audio"].append(
                    (name, param, create_freeze_hook(name, param))
                )
                setattr(param, "color", "audio")
            elif lm_pattern.match(name) or expert_type == "expert_type_0":
                self._modality_param_mapping["lm"].append(
                    (name, param, create_freeze_hook(name, param))
                )
                setattr(param, "color", "lm")
            else:
                self._modality_param_mapping["mm"].append(
                    (name, param, create_freeze_hook(name, param))
                )
                setattr(param, "color", "mm")
        debug_msg = {
            k: [i[0] for i in v] for k, v in self._modality_param_mapping.items()
        }
        logger.info(
            f"modality_param_mapping: {json.dumps(debug_msg, ensure_ascii=False, indent=2)}"
        )

    def update_params_stat(self, param_group, stop_gradient):
        """freeze mm"""
        assert param_group in (
            "lm",
            "mm",
            "audio",
            "vit",
        ), "param_group must be in ('lm', 'mm', 'audio', 'vit')"
        if self._modality_param_mapping is None:
            self._set_modality_param_mapping()
        if self._modality_param_mapping.get(param_group):
            for name, param, _ in self._modality_param_mapping[param_group]:
                # logger.info(f"mm: {name} set_stop_gradient to {stop_gradient}")
                param.stop_gradient = stop_gradient

    def freeze_vision(self):
        """freeze_vision"""
        if self._modality_param_mapping is None:
            self._set_modality_param_mapping()
        for name, param, _ in self._modality_param_mapping.get("vit", []):
            logger.info("Freezing vision parameter: {}".format(name))
            param.stop_gradient = True
        self.vision_model.config.freeze_vision = True

    def vision_forward(
        self,
        images,
        image_position_ids,
        image_attention_mask,
        grid_thw,
    ):
        """vision_forward"""
        with profile("extract_image_fea"):
            if self.image_preprocess is not None:
                assert images.dtype == paddle.uint8, images.dtype
                images = self.image_preprocess.rescale_factor * images.astype("float32")
                images = (
                    images - self.image_preprocess.image_mean_tensor
                ) / self.image_preprocess.image_std_tensor
                images = images.astype("bfloat16")
            else:
                assert images.dtype == paddle.bfloat16, images.dtype
            # logger.info(f"extract feature input - {images}--{grid_thw}")
            if grid_thw is not None:
                grid_thw = grid_thw[grid_thw > 0].reshape([-1, 3])
                grid_thw = F.pad(
                    paddle.repeat_interleave(grid_thw[:, 1:], grid_thw[:, 0], 0),
                    [0, 0, 1, 0],
                    value=1,
                )
            image_features = self.vision_model.extract_feature(images, grid_thw)
        return image_features

    def add_image_preprocess(self, preprocess):
        logger.info("image preprocess is set")
        self.image_preprocess = preprocess

    def vision_mapping_forward(
        self,
        token_type_ids,
        token_type_ids_w_video,
        input_ids,
        mm_input_ids,
        image_features,
        inputs_embeds,
        image_type_ids,
        grid_thw,
    ):
        """vision_mapping_forward"""
        # debug = mm_input_ids - self.config.max_text_id
        if self.ernie.mm_embed_tokens is not None:
            mm_ids_features = self.ernie.mm_embed_tokens(
                mm_input_ids - self.config.max_text_id
            )
            inputs_embeds[token_type_ids == TokenType.image] = mm_ids_features[
                token_type_ids == TokenType.image
            ]
        image_mask = input_ids == self.config.im_patch_id
        image_features = self.ernie.resampler_model(
            image_features,
            image_mask,
            token_type_ids_w_video,
            image_type_ids,
            grid_thw,
        )

        if image_features.dim == 2:
            B, N, C = image_features.shape
            image_features = image_features.reshape([B * N, C]).astype(
                inputs_embeds.dtype
            )
        # 会覆盖 `mm_ids_features` 中 `ids==im_patch_id` 的部分
        inputs_embeds[image_mask] = image_features
        # # TODO 对部分参数进行normalize 对文部分detach 打印图token
        # text_token_norm = inputs_embeds[input_ids != self.config.im_patch_id].norm(axis=-1).mean()
        # image_token_norm = image_features.norm(axis=-1).mean()
        # image_features /= paddle.sqrt(image_token_norm / text_token_norm)
        return inputs_embeds

    def audio_forward(self, audio_ids):
        """_summary_

        Args:
            audio_ids (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.ernie.audio_embed_tokens(audio_ids)

    def audio_mapping_forward(self, token_type_ids, audio_input_ids, inputs_embeds):
        """_summary_

        Args:
            token_type_ids (_type_): _description_
            audio_features (_type_): _description_
            inputs_embeds (_type_): _description_

        Returns:
            _type_: _description_
        """
        D = self.config.audio_config["audio_encode_frame_depth"]
        # audio_input_ids.shape = [F (音频帧数), D (音频每帧深度)]
        # audio_features.shape = [F, D, AH (音频hidden_size) ]
        # audio_features_reduce.shape = [F, AH]
        # audio_features_proj.shape = [F, LH (文本hidden_size)]
        audio_input_ids = audio_input_ids.reshape([-1, D])

        audio_pad_mask = (
            audio_input_ids == self.config.audio_config["audio_special_tokens"]["PAD"]
        ) | (audio_input_ids == self.config.ignored_index)
        audio_unpad_mask = ~audio_pad_mask
        audio_input_ids[audio_pad_mask] = 0
        audio_pad_mask = audio_pad_mask.astype("float32")
        audio_unpad_mask = audio_unpad_mask.astype("float32")
        audio_features = self.ernie.audio_embed_tokens(audio_input_ids)

        audio_feature_unpad_mask = audio_features * audio_unpad_mask.unsqueeze([-1])
        audio_features_reduce = paddle.sum(audio_feature_unpad_mask, axis=1)

        audio_indices = token_type_ids == TokenType.audio
        audio_frame_num = paddle.sum(audio_indices.astype("int64"))  # 计算实际音频帧数
        audio_features_reduce = audio_features_reduce[:audio_frame_num]
        audio_features_reduce = audio_features_reduce.astype(inputs_embeds.dtype)
        audio_features_proj = self.ernie.audio_adaptor(audio_features_reduce)
        inputs_embeds[audio_indices] = audio_features_proj
        return inputs_embeds

    def prepare_inputs_for_generation(
        self,
        input_ids,
        images=None,
        use_cache=False,
        past_key_values=None,
        inputs_embeds=None,
        image_position_ids=None,
        image_attention_mask=None,
        token_type_ids=None,
        image_type_ids=None,
        grid_thw=None,
        **kwargs,
    ):
        """prepare_inputs_for_generation"""
        if past_key_values:
            input_ids = input_ids[:, -1:]
            token_type_ids = token_type_ids[:, -1:]
            image_type_ids = (
                image_type_ids[:, -1:] if image_type_ids is not None else None
            )

        attention_mask = kwargs.get("attention_mask", None)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": True,
                "attention_mask": attention_mask,
                "return_dict": True,
                "images": images,
                "image_position_ids": image_position_ids,
                "image_attention_mask": image_attention_mask,
                "image_type_ids": image_type_ids,
                "token_type_ids": paddle.concat(
                    [
                        token_type_ids,
                        paddle.zeros([len(token_type_ids), 1], token_type_ids.dtype),
                    ],
                    axis=-1,
                ),
                "grid_thw": grid_thw,
            }
        )

        if self.config.rope_3d:
            model_inputs.update({"position_ids": kwargs["position_ids"]})

        return model_inputs

    def _post_init(self, original_init, *args, **kwargs):
        """
        标记模型中所有的多模态参数, 只处理 head 和 Embedding
        experts 参数在`ernie_moe/modeling.py`中已经完成了标记
        """
        super()._post_init(self, original_init, *args, **kwargs)
        if self.ernie.mm_embed_tokens is not None:
            self.ernie.mm_embed_tokens.weight.expert_type = (
                "expert_type_1"  # 借用 `ernie_moe/modeling.py` 中的标记。
            )
        if self.lm_head.mm_head is not None:
            self.lm_head.mm_head.weight.expert_type = "expert_type_1"
        if getattr(self.lm_head.mm_head, "bias", None) is not None:
            self.lm_head.mm_head.bias.expert_type = "expert_type_1"
        if self.ernie.audio_embed_tokens is not None:
            self.ernie.audio_embed_tokens.weight.expert_type = "expert_type_3"
        if self.ernie.audio_after_norm is not None:
            self.ernie.audio_after_norm.weight.expert_type = "expert_type_3"
            if getattr(self.ernie.audio_after_norm, "bias", None) is not None:
                self.ernie.audio_after_norm.bias.expert_type = "expert_type_3"
        if self.lm_head.audio_out_module is not None:
            self.lm_head.audio_out_module.weight.expert_type = "expert_type_3"
            if getattr(self.lm_head.audio_out_module, "bias", None) is not None:
                self.lm_head.audio_out_module.bias.expert_type = "expert_type_3"

    def forward(
        self,
        input_ids: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[paddle.Tensor] = None,
        images: Optional[paddle.Tensor] = None,
        ignored_index: Optional[int] = 0,
        return_dict: Optional[bool] = None,
        image_position_ids: Optional[paddle.Tensor] = None,
        image_attention_mask: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        image_type_ids: Optional[paddle.Tensor] = None,
        audio_input_ids: Optional[paddle.Tensor] = None,
        audio_labels: Optional[paddle.Tensor] = None,
        grid_thw: Optional[paddle.Tensor] = None,
        # cumulative_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        """forward"""
        if grid_thw is not None:
            grid_thw = grid_thw[grid_thw > 0].reshape([-1, 3])
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        image_mask = input_ids == self.config.im_patch_id

        image_rate = image_mask.astype("float32").mean()
        if labels is not None:
            pad_rate = (
                ((labels == self.criterion.ignored_index) & (~image_mask))
                .astype("float32")
                .mean()
            )
            global_training_logs.update(image_rate=image_rate, pad_rate=pad_rate)
            # assert paddle.any(labels > 0).item(), labels

        if past_key_values is None:
            if images is not None:
                assert (image_mask).any().item(), (
                    image_mask.numpy().tolist(),
                    input_ids.numpy().tolist(),
                    self.config.im_patch_id,
                    images.shape,
                )
                image_features = self.vision_forward(
                    images,
                    image_position_ids,
                    image_attention_mask,
                    grid_thw,
                )
                if self.config.tensor_parallel_degree > 1:
                    if getattr(self.config.vision_config, "variable_resolution", False):
                        S, C = image_features.shape
                        # scatterOp切fea时候 + 4合一，提前把4个token的feature并到一起。
                        image_features = image_features.reshape(
                            [-1, C * self.config.spatial_conv_size**2]
                        )
                    image_features = ScatterOp.apply(image_features, axis=-1)
                    if getattr(self.config.vision_config, "variable_resolution", False):
                        image_features = image_features.reshape([S, -1])
            else:
                image_features = None  # no more faking
        else:
            image_features = None
        # inputs_embeds.stop_gradient = False
        # 0 == 纯文， 1 == 图片，1 会激活 >1 的 expert
        if token_type_ids is None:
            # assert 0, f"别自己造了，请用数据流给的. token_type_ids: {token_type_ids}, image_type_ids: {image_type_ids}"
            token_type_ids = image_mask.astype("int64")
            token_type_ids_labels = paddle.concat(
                [token_type_ids[:, 1:], token_type_ids[:, -1:]], 1
            )
        else:
            assert (
                token_type_ids.shape[1] == input_ids.shape[1] + 1
            ), f"token_type:{token_type_ids.shape}, ids:{input_ids.shape}"
            token_type_ids_labels = token_type_ids[..., 1:]
            # token_type_ids = token_type_ids[..., :-1]

        lm_input_ids = input_ids.clone()
        mm_input_ids = input_ids.clone()
        if self.ernie.mm_embed_tokens is not None:
            lm_input_ids[token_type_ids[..., :-1] == TokenType.image] = 0
            mm_input_ids[token_type_ids[..., :-1] == TokenType.text] = (
                self.config.max_text_id
            )
        # 在 embedding lookup 的时候会统一减去 `max_text_id`
        # 用 `max_text_id` + 1 来替换文本部分 id，是为了跟 `im_patch_id` 区分开。替换部分不会加到最终的input_embeds上所以无所谓。
        # assert self.config.max_text_id + 1 != self.config.im_patch_id,  \
        #      f'max_text_id:{self.config.max_text_id}, im_pach_id:{self.config.im_patch_id}'
        # TODO： audio token

        inputs_embeds = self.ernie.embed_tokens(lm_input_ids)
        token_type_ids_w_video = token_type_ids[..., :-1].clone()
        token_type_ids[token_type_ids == TokenType.video] = TokenType.image

        if images is not None and image_features is not None:
            inputs_embeds = self.vision_mapping_forward(
                token_type_ids[..., :-1],
                token_type_ids_w_video,
                input_ids,
                mm_input_ids,
                image_features,
                inputs_embeds,
                image_type_ids,
                grid_thw,
            )
        else:
            pass  # do nothing, should not hang under DygraphShardingOptimizerV2

        audio_features = None
        if audio_input_ids is not None:
            inputs_embeds = self.audio_mapping_forward(
                token_type_ids[..., :-1],
                audio_input_ids,
                inputs_embeds,
            )
        else:
            pass  # do nothing, should not hang under DygraphShardingOptimizerV2

        outputs = self.ernie(
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        logits_all = self.lm_head(
            outputs.last_hidden_state,
            token_type_ids_labels,
            audio_features,
            use_cache,
        )
        logits, logits_image, logits_audio = logits_all
        router_loss = outputs.router_loss

        mm_head_weight = (
            self.lm_head.mm_head.weight if self.lm_head.mm_head is not None else None
        )
        mm_head_bias = (
            self.lm_head.mm_head.bias if self.lm_head.mm_head is not None else None
        )
        if return_dict:  # aka Generate Decoding
            if labels is not None:
                loss, _ = self.criterion(
                    logits,
                    None,
                    labels,
                    token_type_ids_labels,
                    token_type_ids,
                    logits_audio,
                    audio_labels,
                    self.lm_head.weight,
                    self.lm_head.bias,
                    mm_head_weight,
                    mm_head_bias,
                    router_loss=outputs.router_loss,
                )
            else:
                loss = None
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_loss=outputs.router_loss,
            )
        # Pretrain & Eval 必须有labels
        assert labels is not None
        loss = self.criterion(
            logits,
            logits_image,
            labels,
            token_type_ids_labels,
            token_type_ids,
            logits_audio,
            audio_labels,
            self.lm_head.weight,
            self.lm_head.bias,
            mm_head_weight,
            mm_head_bias,
            router_loss=router_loss,
        )
        return loss

    @staticmethod
    def _resolve_prefix_keys(
        state_keys_base, state_keys_real, ignore_error=False, base_model_prefix=None
    ):
        """_resolve_prefix_keys"""
        # state_keys_map base to real
        state_keys_map = {}

        state_keys_base = set(state_keys_base)
        state_keys_real = set(state_keys_real)

        for key in state_keys_base:
            for x in state_keys_real:
                if "mm_embed_tokens" in x:
                    if "mm_embed_tokens" in key:
                        state_keys_map[key] = x
                        break
                elif x.endswith(key):
                    state_keys_map[key] = x
                    break
            if key not in state_keys_map:
                if not ignore_error:
                    logger.error(f"could not find name {key} in loaded state dict!")
            else:
                state_keys_real.remove(state_keys_map[key])

        return state_keys_map
