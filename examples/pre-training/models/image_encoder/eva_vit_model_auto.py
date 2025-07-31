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

import contextlib
import math
from functools import partial

import paddle
from paddle import nn
from paddle.nn import functional as F

from .eva_text_model import LayerNorm, PatchDropout, PatchDropoutForAdaptive
from .modules.rope import VisionRotaryEmbeddingME
from .eva_vit_model import DropPath, RelativePositionBias


from models.ernie.modeling_auto import (
    get_mesh,
)

try:
    from .modules.fusedln import FusedLayerNorm
except Exception:
    from paddle.nn import LayerNorm as FusedLayerNorm

    print("Warning, FusedLn module is not available, use LayerNorm instead.")
import paddle.distributed as dist
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.incubate.nn.memory_efficient_attention import memory_efficient_attention

# from paddlemix.models.model_utils import MixPretrainedModel
from paddleformers.transformers.model_utils import PretrainedModel
import logging

from .utils import to_2tuple, trunc_normal_
from .configuration import (
    EVAVisionTransformerConfig,
)

logger = logging.getLogger(__name__)

try:
    import fused_ln as fused
except Exception:
    logging.warning("fail to load fused_ln, will not use fused_rms_norm")
    fused = None

try:
    from paddle.nn.functional.flash_attention import (
        flash_attention,
        flash_attn_unpadded,
    )

    logger.warning(
        "Use flash attention in scaled-dot-product. Attention mask is deprecated"
    )
except Exception:
    flash_attention = None


class RMSNorm(paddle.nn.Layer):
    """
    adepted from transformers T5LayerNorm
    """

    def __init__(self, config, hidden_size, fuse_rms_norm=True, eps=1e-6, ipp=0):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.ipp = ipp
        self.weight = paddle.create_parameter(
            [hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Assign(
                paddle.ones([hidden_size])
            ),
        )
        self.fuse_rms_norm = fuse_rms_norm
        self.variance_epsilon = eps

        if config.pipeline_parallel_degree > 1:
            self.weight = dist.shard_tensor(
                self.weight, get_mesh(self.ipp), [dist.Replicate(), dist.Replicate()]
            )

    def forward(self, hidden_states):
        """
        forward
        """
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        if self.fuse_rms_norm and fused is not None:
            return fused.fused_rms_norm(
                hidden_states, self.weight, self.variance_epsilon
            )[0]

        with paddle.amp.auto_cast(False):
            variance = (
                hidden_states.astype(paddle.float32).pow(2).mean(-1, keepdim=True)
            )
            hidden_states = hidden_states * paddle.rsqrt(
                variance + self.variance_epsilon
            )

        # convert into half-precision if necessary
        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = hidden_states.astype(self.weight.dtype)

        return self.weight * hidden_states


class Mlp(paddle.nn.Layer):
    """_summary_

    Args:
        paddle (_type_): _description_
    """

    def __init__(
        self, config, act_layer=paddle.nn.GELU, norm_layer=paddle.nn.LayerNorm, ipp=0
    ):
        super().__init__()
        self.ipp = ipp
        in_features = config.width
        hidden_features = int(config.width * config.mlp_ratio)
        out_features = in_features
        hidden_features = hidden_features or in_features
        if config.tensor_parallel_degree > 1:
            self.fc1 = nn.Linear(
                in_features,
                hidden_features,
                weight_attr=None,
                bias_attr=True,
            )
            self.fc2 = nn.Linear(
                hidden_features, out_features, weight_attr=None, bias_attr=True
            )
            self.rng_tracker = lambda x: get_rng_state_tracker().rng_state(x)
        else:
            self.fc1 = paddle.nn.Linear(in_features, hidden_features)
            self.fc2 = paddle.nn.Linear(hidden_features, out_features)
            self.rng_tracker = lambda x: contextlib.nullcontext()
        self.act = act_layer()
        if config.subln is True:
            raise ValueError("Don't use subln foreveer! This shit fuck distributed")
        self.drop_rate = config.drop_rate
        self.drop = paddle.nn.Dropout(p=config.drop_rate)

        if config.tensor_parallel_degree > 1:
            self.fc1.weight = dist.shard_tensor(
                self.fc1.weight,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(1)],
            )
            self.fc1.bias = dist.shard_tensor(
                self.fc1.bias,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(0)],
            )
            self.fc2.weight = dist.shard_tensor(
                self.fc2.weight,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(0)],
            )
            self.fc2.bias = dist.shard_tensor(
                self.fc2.bias,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Replicate()],
            )

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        if self.drop_rate > 0.0:
            #     with self.rng_tracker("global_seed"):
            x = self.drop(x)
        return x


class SwiGLU(paddle.nn.Layer):
    """_summary_

    Args:
        paddle (_type_): _description_
    """

    def __init__(
        self,
        config,
        drop=0.0,
        act_layer=paddle.nn.Silu,
        norm_layer=paddle.nn.LayerNorm,
        ipp=0,
    ):
        super().__init__()
        self.ipp = ipp
        in_features = config.width
        hidden_features = int(config.width * config.mlp_ratio)
        out_features = in_features
        if config.tensor_parallel_degree > 1:
            self.w1 = nn.Linear(
                in_features,
                hidden_features,
                weight_attr=None,
                bias_attr=True,
            )
            self.w2 = nn.Linear(
                in_features,
                hidden_features,
                weight_attr=None,
                bias_attr=True,
            )
            self.w3 = nn.Linear(
                hidden_features,
                out_features,
                weight_attr=None,
                bias_attr=True,
            )
        else:
            self.w1 = paddle.nn.Linear(in_features, hidden_features)
            self.w2 = paddle.nn.Linear(in_features, hidden_features)
            self.w3 = paddle.nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.ffn_ln = (
            norm_layer(hidden_features) if config.subln else paddle.nn.Identity()
        )
        self.drop = paddle.nn.Dropout(p=drop)
        if config.tensor_parallel_degree > 1:
            self.rng_tracker = lambda x: get_rng_state_tracker().rng_state(x)
        else:
            self.rng_tracker = lambda x: contextlib.nullcontext()

        if config.tensor_parallel_degree > 1:
            self.w1.weight = dist.shard_tensor(
                self.w1.weight,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(1)],
            )
            self.w2.weight = dist.shard_tensor(
                self.w2.weight,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(1)],
            )
            self.w3.weight = dist.shard_tensor(
                self.w3.weight,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(1)],
            )
            self.w1.bias = dist.shard_tensor(
                self.w1.bias,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(0)],
            )
            self.w2.bias = dist.shard_tensor(
                self.w2.bias,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(0)],
            )
            self.w3.bias = dist.shard_tensor(
                self.w3.bias,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(0)],
            )

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        # with self.rng_tracker("global_seed"):
        x = self.drop(x)
        return x


class Attention(paddle.nn.Layer):
    """_summary_

    Args:
        paddle (_type_): _description_
    """

    def __init__(
        self, config, window_size=None, rope=None, norm_layer=paddle.nn.LayerNorm, ipp=0
    ):
        super().__init__()
        self.ipp = ipp
        self.config = config
        dim = config.width
        self.xattn_drop = config.attn_drop_rate
        self.xattn = config.xattn
        self.subln = config.subln
        self.fused_qkv_attn = config.fused_qkv_attn
        self.adaptive = config.resolution_transform == "adaptive"

        self.num_heads = config.width // config.head_width
        self.head_dim = dim // self.num_heads
        if hasattr(config, "attn_head_dim") and config.attn_head_dim is not None:
            self.head_dim = config.attn_head_dim
        all_head_dim = self.head_dim * self.num_heads
        self.scale = config.qk_scale or self.head_dim**-0.5
        if config.tensor_parallel_degree > 1:
            if not self.fused_qkv_attn:
                self.q_proj = nn.Linear(
                    dim,
                    all_head_dim,
                    weight_attr=None,
                    bias_attr=config.qkv_bias,
                )

                self.k_proj = nn.Linear(
                    dim,
                    all_head_dim,
                    weight_attr=None,
                    bias_attr=config.qkv_bias,
                )
                self.v_proj = nn.Linear(
                    dim,
                    all_head_dim,
                    weight_attr=None,
                    bias_attr=config.qkv_bias,
                )
            else:
                self.qkv_proj = nn.Linear(
                    dim,
                    all_head_dim * 3,
                    weight_attr=None,
                    bias_attr=config.qkv_bias,
                )
        else:
            if not self.fused_qkv_attn:
                self.q_proj = paddle.nn.Linear(
                    dim, all_head_dim, bias_attr=config.qkv_bias
                )
                self.k_proj = paddle.nn.Linear(
                    dim, all_head_dim, bias_attr=config.qkv_bias
                )
                self.v_proj = paddle.nn.Linear(
                    dim, all_head_dim, bias_attr=config.qkv_bias
                )
            else:
                self.qkv_proj = paddle.nn.Linear(
                    dim, all_head_dim * 3, bias_attr=config.qkv_bias
                )

        if config.tensor_parallel_degree > 1:
            if not self.fused_qkv_attn:
                self.q_proj.weight = dist.shard_tensor(
                    self.q_proj.weight,
                    get_mesh(self.ipp),
                    [dist.Replicate(), dist.Shard(1)],
                )
                self.k_proj.weight = dist.shard_tensor(
                    self.k_proj.weight,
                    get_mesh(self.ipp),
                    [dist.Replicate(), dist.Shard(1)],
                )
                self.v_proj.weight = dist.shard_tensor(
                    self.v_proj.weight,
                    get_mesh(self.ipp),
                    [dist.Replicate(), dist.Shard(1)],
                )
                if config.qkv_bias:
                    self.q_proj.bias = dist.shard_tensor(
                        self.q_proj.bias,
                        get_mesh(self.ipp),
                        [dist.Replicate(), dist.Shard(0)],
                    )
                    self.k_proj.bias = dist.shard_tensor(
                        self.k_proj.bias,
                        get_mesh(self.ipp),
                        [dist.Replicate(), dist.Shard(0)],
                    )
                    self.v_proj.bias = dist.shard_tensor(
                        self.v_proj.bias,
                        get_mesh(self.ipp),
                        [dist.Replicate(), dist.Shard(0)],
                    )

            else:
                self.qkv_proj.weight = dist.shard_tensor(
                    self.qkv_proj.weight,
                    get_mesh(self.ipp),
                    [dist.Replicate(), dist.Shard(1)],
                )
                if config.qkv_bias:
                    self.qkv_proj.bias = dist.shard_tensor(
                        self.qkv_proj.bias,
                        get_mesh(self.ipp),
                        [dist.Replicate(), dist.Shard(0)],
                    )

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                2 * window_size[1] - 1
            ) + 3
            init_data = paddle.zeros(shape=[self.num_relative_distance, self.num_heads])
            self.relative_position_bias_table = self.create_parameter(
                shape=[self.num_relative_distance, self.num_heads],
                default_initializer=paddle.nn.initializer.Assign(init_data),
            )
            coords_h = paddle.arange(end=window_size[0])
            coords_w = paddle.arange(end=window_size[1])
            coords = paddle.stack(x=paddle.meshgrid([coords_h, coords_w]))
            coords_flatten = paddle.flatten(x=coords, start_axis=1)
            relative_coords = (
                coords_flatten[:, :, (None)] - coords_flatten[:, (None), :]
            )
            relative_coords = relative_coords.transpose(perm=[1, 2, 0])
            relative_coords[:, :, (0)] += window_size[0] - 1
            relative_coords[:, :, (1)] += window_size[1] - 1
            relative_coords[:, :, (0)] *= 2 * window_size[1] - 1
            relative_position_index = paddle.zeros(
                shape=(window_size[0] * window_size[1] + 1,) * 2,
                dtype=relative_coords.dtype,
            )
            relative_position_index[1:, 1:] = relative_coords.sum(axis=-1)
            relative_position_index[(0), 0:] = self.num_relative_distance - 3
            relative_position_index[0:, (0)] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None
        self.attn_drop = paddle.nn.Dropout(p=self.xattn_drop)
        if config.subln and config.inner_attn_ln:
            raise ValueError("Don't fucking use inner attn ln. This fuck distributed")
        if config.tensor_parallel_degree > 1:
            self.proj = nn.Linear(all_head_dim, dim, weight_attr=None, bias_attr=True)

            self.proj.weight = dist.shard_tensor(
                self.proj.weight,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(0)],
            )
            self.proj.bias = dist.shard_tensor(
                self.proj.bias,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Replicate()],
            )
        else:
            self.proj = paddle.nn.Linear(all_head_dim, dim)
        self.drop_rate = config.drop_rate
        self.proj_drop = paddle.nn.Dropout(p=config.drop_rate)
        self.rope = rope
        if config.tensor_parallel_degree > 1:
            self.rng_tracker = lambda x: get_rng_state_tracker().rng_state(x)
        else:
            self.rng_tracker = lambda x: contextlib.nullcontext()

    def forward(self, x, rel_pos_bias=None, cumulative_indices=None, attn_mask=None):
        """_summary_

        Args:
            x (_type_): _description_
            rel_pos_bias (_type_, optional): _description_. Defaults to None.
            cumulative_indices (_type_, optional): _description_. Defaults to None.
            attn_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if not self.adaptive:
            B, N, C = x.shape
        else:
            total_N, C = x.shape
        if self.fused_qkv_attn:
            mix_layer = self.qkv_proj(x)
            if not self.adaptive:
                mix_layer = mix_layer.reshape([B, N, -1, self.head_dim * 3])
            else:
                mix_layer = mix_layer.reshape([total_N, -1, self.head_dim * 3])
            q, k, v = paddle.split(mix_layer, 3, axis=-1)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            if not self.adaptive:
                q = q.reshape((B, N, -1, self.head_dim))
                k = k.reshape((B, N, -1, self.head_dim))
                v = v.reshape((B, N, -1, self.head_dim))
            else:
                q = q.reshape((total_N, -1, self.head_dim))
                k = k.reshape((total_N, -1, self.head_dim))
                v = v.reshape((total_N, -1, self.head_dim))

        if self.rope is not None:
            # assert not self.adaptive, "Rope is not supported in adaptive mode"
            if not self.adaptive:
                q = q.transpose(perm=[0, 2, 1, 3])
                q_t = q[:, :, 1:, :]
                ro_q_t = self.rope(q_t)
                q = paddle.concat(x=(q[:, :, :1, :], ro_q_t), axis=-2).astype(
                    dtype=v.dtype
                )
                q = q.transpose(perm=[0, 2, 1, 3])

                k = k.transpose(perm=[0, 2, 1, 3])
                k_t = k[:, :, 1:, :]
                ro_k_t = self.rope(k_t)
                k = paddle.concat(x=(k[:, :, :1, :], ro_k_t), axis=-2).astype(
                    dtype=v.dtype
                )
                k = k.transpose(perm=[0, 2, 1, 3])
            else:
                q = self.rope(q)  # leave rope to handle cls
                k = self.rope(k)  # leave rope to handle cls

        if self.adaptive:
            assert (
                self.config.use_flash_attn
            ), "Only Flash Attention support adaptive mode"
            image_max_len = paddle.max(
                cumulative_indices[1:] - cumulative_indices[:-1]
            ).item()
            # print(f'q:{q.norm()} k:{k.norm()} v:{v.norm()}')
            attn_output, _ = flash_attn_unpadded(
                q,
                k,
                v,
                cumulative_indices,
                cumulative_indices,
                image_max_len,
                image_max_len,
                self.head_dim**-0.5,
                self.xattn_drop,
                False,
                False,
            )

            x = attn_output.reshape([0, attn_output.shape[-1] * attn_output.shape[-2]])
            x = self.proj(x)

            if self.drop_rate > 0.0:
                #     with self.rng_tracker("global_seed"):
                x = self.proj_drop(x)
        elif self.config.use_flash_attn:
            # Flash Attention now ignore attention mask
            # Current Flash Attention doesn't support attn maskt
            # Paddle Flash Attention input [ bz, seqlen, nhead, head_dim]
            # Torch Flash Attention input [ bz, nhead, seqlen, head_dim]
            # without past keys
            attn_output, _ = flash_attention(
                q,
                k,
                v,
                causal=False,
                return_softmax=False,
                dropout=self.xattn_drop,
            )
            x = attn_output.reshape([B, N, -1])
            x = self.proj(x)
            if self.drop_rate > 0.0:
                #     with self.rng_tracker("global_seed"):
                x = self.proj_drop(x)
        elif self.xattn:
            x = memory_efficient_attention(q, k, v, p=self.xattn_drop, scale=self.scale)
            x = x.reshape((B, N, -1))
            x = self.proj(x)
            if self.drop_rate > 0.0:
                #     with self.rng_tracker("global_seed"):
                x = self.proj_drop(x)
        else:
            q = q.transpose(perm=[0, 2, 1, 3])
            k = k.transpose(perm=[0, 2, 1, 3])
            v = v.transpose(perm=[0, 2, 1, 3])
            q = q * self.scale
            x = k
            perm_0 = list(range(x.ndim))
            perm_0[-2] = x.ndim - 1
            perm_0[-1] = x.ndim - 2
            attn = q @ x.transpose(perm=perm_0)
            if self.relative_position_bias_table is not None:
                relative_position_bias = self.relative_position_bias_table[
                    self.relative_position_index.reshape((-1))
                ].reshape(
                    (
                        self.window_size[0] * self.window_size[1] + 1,
                        self.window_size[0] * self.window_size[1] + 1,
                        -1,
                    )
                )
                relative_position_bias = relative_position_bias.transpose(
                    perm=[2, 0, 1]
                )
                attn = attn + relative_position_bias.unsqueeze(axis=0).astype(
                    dtype=attn.dtype
                )
            if rel_pos_bias is not None:
                attn = attn + rel_pos_bias.astype(dtype=attn.dtype)
            if attn_mask is not None:
                attn_mask = attn_mask.astype(dtype="bool")
                attn = paddle.where(
                    ~attn_mask[:, (None), (None), :], attn, float("-inf")
                )
            attn = paddle.nn.functional.softmax(attn, axis=-1)
            with self.rng_tracker("local_seed"):
                attn = self.attn_drop(attn)
            x = attn @ v
            perm_1 = list(range(x.ndim))
            perm_1[1] = 2
            perm_1[2] = 1
            x = x.transpose(perm=perm_1).reshape((B, N, -1))
            x = self.proj(x)

            if self.drop_rate > 0.0:
                # with self.rng_tracker("global_seed"):
                x = self.proj_drop(x)
        return x


class Block(paddle.nn.Layer):
    """_summary_

    Args:
        paddle (_type_): _description_
    """

    def __init__(
        self,
        config,
        drop_path=0.0,
        window_size=None,
        rope=None,
        act_layer=paddle.nn.GELU,
        norm_layer=paddle.nn.LayerNorm,
        layer_id=0,
    ):
        super().__init__()
        self.layer_id = layer_id
        dim = config.width
        init_values = config.init_values
        self.postnorm = config.postnorm

        self.norm1 = norm_layer(dim)
        self.attn = Attention(config, window_size=window_size, rope=rope)
        self.drop_path = (
            DropPath(drop_path, config.tensor_parallel_degree)
            if drop_path > 0.0
            else paddle.nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        if config.naiveswiglu:
            self.mlp = SwiGLU(config, norm_layer=norm_layer)
        else:
            self.mlp = Mlp(config, act_layer=act_layer)
        if init_values is not None and init_values > 0:
            init_data = init_values * paddle.ones(shape=dim)
            self.gamma_1 = self.create_parameter(
                shape=dim, default_initializer=paddle.nn.initializer.Assign(init_data)
            )
            init_data = init_values * paddle.ones(shape=dim)
            self.gamma_2 = self.create_parameter(
                shape=dim, default_initializer=paddle.nn.initializer.Assign(init_data)
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None, cumulative_indices=None, attn_mask=None):
        """_summary_

        Args:
            x (_type_): _description_
            rel_pos_bias (_type_, optional): _description_. Defaults to None.
            position_ids (_type_, optional): _description_. Defaults to None.
            cumulative_indices (_type_, optional): _description_. Defaults to None.
            attn_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(
                    self.norm1(
                        self.attn(
                            x,
                            rel_pos_bias=rel_pos_bias,
                            attn_mask=attn_mask,
                            cumulative_indices=cumulative_indices,
                        )
                    )
                )
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(
                    self.attn(
                        self.norm1(x),
                        rel_pos_bias=rel_pos_bias,
                        attn_mask=attn_mask,
                        cumulative_indices=cumulative_indices,
                    )
                )
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.postnorm:
            x = x + self.drop_path(
                self.gamma_1
                * self.norm1(
                    self.attn(
                        x,
                        rel_pos_bias=rel_pos_bias,
                        attn_mask=attn_mask,
                        cumulative_indices=cumulative_indices,
                    )
                )
            )
            x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
        else:
            x = x + self.drop_path(
                self.gamma_1
                * self.attn(
                    self.norm1(x),
                    rel_pos_bias=rel_pos_bias,
                    attn_mask=attn_mask,
                    cumulative_indices=cumulative_indices,
                )
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(paddle.nn.Layer):
    """Image to Patch Embedding"""

    def __init__(self, config):
        super().__init__()
        image_size = to_2tuple(config.image_size)
        patch_size = to_2tuple(config.patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.patch_shape = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = paddle.nn.Conv2D(
            in_channels=config.in_chans,
            out_channels=config.width,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x, **kwargs):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        B, C, H, W = x.shape
        # assert (
        #     H == self.image_size[0] and W == self.image_size[1]
        # ), f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
        x = self.proj(x).flatten(start_axis=2)
        perm_2 = list(range(x.ndim))
        perm_2[1] = 2
        perm_2[2] = 1
        x = x.transpose(perm=perm_2)
        return x


class EVAVisionTransformerPretrainedModel(PretrainedModel):
    """
    #See :class:`paddlemix.models.model_utils.MixPretrainedModel` for more details.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = "config.json"
    config_class = EVAVisionTransformerConfig
    resource_files_names = {"model_state": "vision_model_state.pdparams"}
    base_model_prefix = "evavision_transformer"


class EVAVisionTransformerAuto(EVAVisionTransformerPretrainedModel):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(self, config: EVAVisionTransformerConfig):
        super(EVAVisionTransformerAuto, self).__init__(config)
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
        use_mean_pooling = config.use_mean_pooling
        if config.use_rms_norm:
            logger.info("use rms norm!!")
            norm_layer = partial(
                RMSNorm,
                config=self.config,
                fuse_rms_norm=config.fuse_rms_norm,
                eps=1e-6,
            )
        else:
            norm_layer = (
                partial(FusedLayerNorm, epsilon=1e-6)
                if config.fusedLN
                else partial(LayerNorm, epsilon=1e-6)
            )
        num_heads = config.width // config.head_width
        if self.resolution_transform != "adaptive":
            self.patch_embed = PatchEmbed(config)
        else:
            raise NotImplementedError
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
        if config.use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads
            )
        else:
            self.rel_pos_bias = None
        if config.rope:
            assert not config.use_abs_pos_emb, "RoPE do need abs pos emb"
            half_head_dim = width // num_heads // 2
            hw_seq_len = config.image_size // config.patch_size
            self.rope = VisionRotaryEmbeddingME(
                dim=half_head_dim,
                pt_seq_len=config.pt_hw_seq_len,
                ft_seq_len=hw_seq_len if config.intp_freq else None,
            )
        else:
            self.rope = None
        dpr = [
            x.item()
            for x in paddle.linspace(
                start=0, stop=config.drop_path_rate, num=config.layers
            )
        ]
        self.blocks = paddle.nn.LayerList(
            sublayers=[
                Block(
                    config,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    window_size=(
                        self.patch_embed.patch_shape
                        if config.use_rel_pos_bias
                        else None
                    ),
                    rope=self.rope,
                    layer_id=i,
                )
                for i in range(config.layers)
            ]
        )
        if config.attentional_pool:
            raise NotImplementedError
        else:
            self.attn_pool = None
            self.norm = paddle.nn.Identity() if use_mean_pooling else norm_layer(width)
            self.fc_norm = norm_layer(width) if use_mean_pooling else None
            if config.need_head:
                raise NotImplementedError

        if self.resolution_transform != "adaptive":
            self.patch_dropout = PatchDropout(
                config.patch_dropout, rope=config.rope
            )  # if config.patch_dropout > 0.0 else paddle.nn.Identity()
        else:
            self.patch_dropout = PatchDropoutForAdaptive(
                config.patch_dropout, rope=config.rope
            )
        if config.tensor_parallel_degree > 1:
            self.rng_tracker = lambda x: get_rng_state_tracker().rng_state(x)
        else:
            self.rng_tracker = lambda x: contextlib.nullcontext()

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):

        from paddleformers.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.width // config.head_width,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}
            base_actions = {
                # Column Linear
                "blocks.0.mlp.fc1.weight": partial(fn, is_column=True),
                "blocks.0.mlp.fc1.bias": partial(fn, is_column=True),
                "blocks.0.mlp.fc2.weight": partial(fn, is_column=False),
                "blocks.0.attn.proj.weight": partial(fn, is_column=False),
            }
            if config.fused_qkv_attn:
                base_actions.update(
                    {
                        "blocks.0.attn.qkv_proj.weight": partial(fn, is_column=True),
                    }
                )
                if config.qkv_bias:
                    base_actions.update(
                        {
                            "blocks.0.attn.qkv_proj.bias": partial(fn, is_column=True),
                        }
                    )
            else:
                base_actions.update(
                    {
                        "blocks.0.attn.q_proj.weight": partial(fn, is_column=True),
                        "blocks.0.attn.k_proj.weight": partial(fn, is_column=True),
                        "blocks.0.attn.v_proj.weight": partial(fn, is_column=True),
                    }
                )
                if config.qkv_bias:
                    base_actions.update(
                        {
                            "blocks.0.attn.q_proj.bias": partial(fn, is_column=True),
                            "blocks.0.attn.k_proj.bias": partial(fn, is_column=True),
                            "blocks.0.attn.v_proj.bias": partial(fn, is_column=True),
                        }
                    )
            for key, action in base_actions.items():
                if "blocks.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("blocks.0.", f"blocks.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.layers)

        return mappings

    def _post_init(self, original_init, *args, **kwargs):
        super()._post_init(self, original_init, *args, **kwargs)
        """_summary_"""

        # def rescale(param, layer_id):
        #     origin_dtype = paddle.get_default_dtype()
        #     paddle.set_default_dtype("float32")
        #     tmp = paddle.to_tensor(math.sqrt(2.0 * layer_id))
        #     paddle.set_default_dtype(origin_dtype)
        #     if origin_dtype != "float32":
        #         tmp = tmp.astype(origin_dtype)
        #     param = param.divide(tmp)

        # for layer_id, layer in enumerate(self.blocks):
        #     rescale(layer.attn.proj.weight, layer_id + 1)
        #     if self.naiveswiglu:
        #         rescale(layer.mlp.w3.weight, layer_id + 1)
        #     else:
        #         rescale(layer.mlp.fc2.weight, layer_id + 1)

        # with self.rng_tracker("global_seed"):
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
            logger.info(
                f"dist-init-fc: <pos_embed> shape={self.pos_embed.shape}, "
                f' type={type(self.pos_embed)},norm={self.pos_embed.astype("float32").norm()}'
            )
        trunc_normal_(self.cls_token, std=0.02)
        logger.info(
            f"dist-init-fc: <cls_token> shape={self.cls_token.shape}, "
            f' type={type(self.cls_token)},norm={self.cls_token.astype("float32").norm()}'
        )

    def get_cast_dtype(self) -> paddle.dtype:
        """_summary_

        Returns:
            paddle.dtype: _description_
        """
        return self.blocks[0].mlp.fc2.weight.dtype

    # def _init_weights(self, m):
    #     """Initialization hook"""
    #     zeros_params = paddle.nn.initializer.Constant(0.0)
    #     ones_params = paddle.nn.initializer.Constant(1.0)
    #     if isinstance(m, (ColumnParallelLinear, RowParallelLinear)):
    #         with self.rng_tracker("model_parallel_rng"):  # dp 之间一样，mp 间不一样
    #             trunc_normal_(m.weight, std=0.02)
    #             if m.bias is not None:
    #                 zeros_params(m.bias)
    #             logger.info(
    #                 f"dist-init-fc: shape={m.weight.shape}, "
    #                 f' type={type(m)},norm={m.weight.astype("float32").norm()}'
    #             )
    #     elif isinstance(m, (paddle.nn.Linear,)):
    #         # 在 eb 的 seeding 环境下，dp间不一样，会在 TensroParallel 中进行 dp 间广播
    #         trunc_normal_(m.weight, std=0.02)
    #         if m.bias is not None:
    #             zeros_params(m.bias)
    #         logger.info(
    #             f"dist-init-fc: shape={m.weight.shape}, " f' type={type(m)},norm={m.weight.astype("float32").norm()}'
    #         )
    #     elif isinstance(m, paddle.nn.LayerNorm):
    #         # 在 eb 的 seeding 环境下，dp间不一样，会在 TensroParallel 中进行 dp 间广播
    #         zeros_params(m.bias)
    #         ones_params(m.weight)
    #         logger.info(
    #             f"dist-init-fc: shape={m.weight.shape}, " f' type={type(m)},norm={m.weight.astype("float32").norm()}'
    #         )

    def get_num_layers(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return len(self.blocks)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """_summary_

        Args:
            unlocked_groups (int, optional): _description_. Defaults to 0.
            freeze_bn_stats (bool, optional): _description_. Defaults to False.
        """
        assert (
            unlocked_groups == 0
        ), "partial locking not currently supported for this model"
        for param in self.parameters():
            param.stop_gradient = not False

    def set_grad_checkpointing(self, enable=True):
        """_summary_

        Args:
            enable (bool, optional): _description_. Defaults to True.
        """
        self.use_recompute_attn = enable

    def no_weight_decay(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return {"pos_embed", "cls_token"}

    def get_adaptive_position_emb(
        self,
        image_sizes,
        position_ids_2d,
    ):
        """
        adaptive 分辨率情况下，查询绝对位置 id，当前支持方法:`interpolate`, `downsample`。
        position_ids_2d: 位置不包含 [CLS]
        """
        assert self.resolution_transform == "adaptive", self.resolution_transform

        def _build_ordered_pos(i, j):
            pos = paddle.arange(i).unsqueeze(-1) * self.patch_embed.num_patches_2d[
                1
            ] + paddle.arange(j).unsqueeze(0)
            pos = pos.reshape([-1])
            pos = F.pad(pos + 1, (1, 0))  # + 1 for cls
            return pos.astype("int64")

        def _build_ordered_pos1d_from_2d(pos2d, lens):
            i, j = pos2d.unbind(-1)
            pos1d = i * self.patch_embed.num_patches_2d[1] + j
            pos_list = pos1d.split(lens.tolist(), 0)
            cls_pos = paddle.zeros([1], dtype=pos1d.dtype)
            pos1d = paddle.concat([l for p in pos_list for l in [cls_pos, p + 1]], 0)
            return pos1d.astype("int64")

        def _build_ordered_pos_2d(i, j):
            pos = paddle.stack(
                [
                    paddle.arange(i).unsqueeze(-1).expand([i, j]),
                    paddle.arange(j).unsqueeze(0).expand([i, j]),
                ],
                -1,
            )
            return pos.reshape([-1, 2]).astype("int64")

        if position_ids_2d is None:
            image_sizes_list = image_sizes.tolist()
            position_ids = paddle.concat(
                [_build_ordered_pos(*s) for s in image_sizes_list],
                axis=0,
            )
            position_ids_2d = paddle.concat(
                [_build_ordered_pos_2d(*s) for s in image_sizes_list],
                axis=0,
            )
        else:
            position_ids = _build_ordered_pos1d_from_2d(
                position_ids_2d, image_sizes.prod(-1)
            )
        if self.pos_embed is not None:
            assert (
                paddle.max(position_ids).item() < self.pos_embed.shape[1]
            ), f"position ids {paddle.max(position_ids)} must be \
                less than {self.pos_embed.shape}"

        if self.pos_embed is not None:
            positional_embeddings = self.pos_embed[0, :, :][position_ids]
        else:
            positional_embeddings = 0.0
        return positional_embeddings, position_ids, position_ids_2d

    def get_position_emb(self, x):
        """
        固定分辨率情况下，查询绝对位置 id，当前支持方法:`interpolate`, `downsample`。
        """
        # if x.shape[1] - 1 == self.patch_embed.num_patches:
        #     return self.pos_embed
        # 可变分辨率
        old_shape = int(math.sqrt(self.patch_embed.num_patches))
        new_shape = int(math.sqrt(x.shape[1] - 1))
        if self.resolution_transform == "interpolate":
            assert (
                not self.config.rope and self.pos_embed is not None
            ), "pos_embed must be provided for interpolation"
            pos_emb_img = (
                self.pos_embed[:, 1:]
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
                [self.pos_embed[:, :1], pos_emb_img], axis=1
            )
            position_ids = None
            position_ids_2d = None
        elif self.resolution_transform == "downsample":
            tmp = paddle.arange(0, old_shape, int(old_shape / new_shape), dtype="int64")
            position_ids = (tmp.unsqueeze(0) + tmp.unsqueeze(-1) * old_shape).flatten()
            if self.pos_embed is not None:
                positional_embeddings = paddle.concat(
                    [
                        self.pos_embed[:, :1, :],
                        self.pos_embed[0, 1:, :][position_ids].unsqueeze(0),
                    ],
                    axis=1,
                )
            else:
                positional_embeddings = 0.0
            position_ids_2d = paddle.stack(
                [
                    tmp.repeat_interleave(tmp.shape[0], axis=0),
                    tmp.tile([tmp.shape[0]]),
                ],
                axis=-1,
            )
            position_ids = position_ids.tile([x.shape[0], 1])
            position_ids_2d = position_ids_2d.tile([x.shape[0], 1, 1])
        elif self.resolution_transform == "ordered":
            tmp = paddle.arange(0, new_shape, dtype="int64")
            position_ids = (tmp.unsqueeze(0) + tmp.unsqueeze(-1) * old_shape).flatten()
            if self.pos_embed is not None:
                positional_embeddings = paddle.concat(
                    [
                        self.pos_embed[:, :1, :],
                        self.pos_embed[0, 1:, :][position_ids].unsqueeze(0),
                    ],
                    axis=1,
                )
            else:
                positional_embeddings = 0.0
            position_ids_2d = paddle.stack(
                [
                    tmp.repeat_interleave(tmp.shape[0], axis=0),
                    tmp.tile([tmp.shape[0]]),
                ],
                axis=-1,
            )
            position_ids = position_ids.tile([x.shape[0], 1])
            position_ids_2d = position_ids_2d.tile([x.shape[0], 1, 1])
        else:
            raise ValueError(
                f"resolution_transform {self.resolution_transform} is not supported"
            )
        return positional_embeddings, position_ids, position_ids_2d

    def forward_features(
        self, x, image_sizes=None, position_ids=None, return_all_features=False
    ):
        """
        Args:
            x :Tensor[B,C,H,W] or [num_tokens, 3 * patch_size * patch_siz], raw-pixels.
               如果输入为固定分辨率，x.shape == [B,C,H,W]
               如果输入为 adaptive 分辨率，需要提前 patchify。x.shape == [num_tokens, 3 * patch_size * patch_size]

            position_ids: Tensor[num_tokens], 如果在ImageEncoder 中做了 patch-drop，直接传入 position_id (2维)。
            image_sizes :Tensor[num_image, 2], int64, defalts to None,
                list of [patch_size_H, patch_size_W] of all images in `x`, 固定分辨率下为 None。

        Returns:
            x: Tensor[num_sample, C], cls_feature
            image_sizes: Tensor[num_sample, 1], 经过 patch-drop 后adaptive图片的大小。

            all_feature: Tensor[num_tokens,C], optional, all token feature exclude cls.
            hiddens: List[Tensor[num_tokens, C]], optional, all tokens feature for all layers exclude cls.
        """
        if x.ndim == 2:
            assert (
                image_sizes is not None
                and len(x) - image_sizes.prod(-1).sum(0).item() >= 0
            ), f"patches in images_sizes:{image_sizes} > image_len:{len(x)}"
        # print(f'x init cls:{x.astype("float32").norm()}')
        x = self.patch_embed(x)
        # print(f'x_before cls:{x.astype("float32").norm()}')

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
            pos_emb, position_ids, position_ids_2d = self.get_adaptive_position_emb(
                image_sizes, position_ids
            )
            assert len(position_ids) == len(x), (len(position_ids), len(x))
        else:
            pos_emb, position_ids, position_ids_2d = self.get_position_emb(x)

        # with self.rng_tracker("global_seed"):
        if self.pos_embed is not None:
            x = x + pos_emb
        x = self.pos_drop(x)

        cumulative_indices = None

        if self.rope is not None:
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

            self.rope.forward = partial(
                self.rope.forward,
                position_ids=position_ids,
                position_ids_2d=position_ids_2d,
            )
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

        if self.seqlen and x.ndim == 2:  # do padding
            pad_len = self.seqlen - len(x)
            if pad_len > 0:
                x = F.pad(x, (0, pad_len, 0, 0))
                cumulative_indices = paddle.concat(
                    [
                        cumulative_indices,
                        paddle.full((1,), len(x), dtype=cumulative_indices.dtype),
                    ],
                    0,
                )
                if self.rope is not None:
                    position_ids = paddle.concat(
                        (
                            position_ids,
                            paddle.zeros([1], dtype=position_ids.dtype),
                            paddle.ones([pad_len - 1], dtype=position_ids.dtype),
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
                    self.rope.forward = partial(
                        self.rope.forward,
                        position_ids=position_ids,
                        position_ids_2d=position_ids_2d,
                    )
            else:
                raise RuntimeError(
                    f"unable to fix vit seqlen, seqlen={self.seqlen}, "
                    f"image-patch-sizes={len(x)}, image_sizes={image_sizes}"
                )

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        cnt = 0
        hiddens = []
        # logger.info(f"fwd-shape:{x.shape}")
        for blk in self.blocks:
            cnt += 1
            if self.use_recompute_attn:
                x = paddle.distributed.fleet.utils.recompute(
                    blk, x, rel_pos_bias, cumulative_indices, use_reentrant=False
                )
            else:
                x = blk(
                    x, rel_pos_bias=rel_pos_bias, cumulative_indices=cumulative_indices
                )
            hiddens.append(x)

        if self.attn_pool is not None:
            raise NotImplementedError

        all_features = self.norm(x)
        if self.fc_norm is not None:
            assert (
                self.resolution_transform != "adaptive"
            ), "resolution_transform='adaptive' is not supported for AttentionalPooler"
            x = self.fc_norm(all_features.mean(1))
        else:
            if self.resolution_transform == "adaptive":
                x = all_features[cls_pos]
                if x.ndim == 1 and len(cls_pos) == 1:
                    x = x.unsqueeze(0)
                # 考虑了 patch-drop 后的实际 image-sizes，-1 以移除 cls
                image_sizes = paddle.diff(cumulative_indices).unsqueeze(-1) - 1
            else:
                # 考虑了 patch-drop 后的实际 image-sizes，-1 以移除 cls
                x = all_features[:, 0]
                image_sizes = paddle.full(
                    [all_features.shape[0], 1], all_features.shape[1] - 1, dtype="int64"
                )

        if not return_all_features:
            return x, image_sizes

        if self.resolution_transform == "adaptive":
            # separate the cls token
            all_features = all_features[non_cls_pos]
            hiddens = [h[non_cls_pos] for h in hiddens]  # 统一在组网中移除cls
            return x, image_sizes, all_features, hiddens

        hiddens = [h[:, 1:] for h in hiddens]  # 统一在组网中移除cls
        all_features = all_features[:, 1:]
        return x, image_sizes, all_features, hiddens

    def forward(
        self, x, image_sizes=None, position_ids=None, return_all_features=False
    ):
        """
        Args:
            x :Tensor[B,C,H,W] or [num_tokens,C], raw-pixels
            image_sizes :Tensor[num_image, 2], int64, defalts to None,
            position_ids: Tensor[num_tokens], int64 position-ids for image patches
        Returns:
            x: Tensor[num_sample, C], cls_feature
            image_sizes: image_sizes after patch-drop
            all_feature: Tensor[num_tokens,C], optional, all token feature exclude cls.
            all hiddens, optional, all hiddens.
        """

        model_output = self.forward_features(
            x,
            image_sizes,
            position_ids,
            return_all_features=return_all_features,
        )
        if isinstance(model_output, tuple):
            x = model_output[0]
        else:
            x = model_output

        if self.config.need_head:
            x = model_output[0] if isinstance(model_output, tuple) else model_output
            x = self.head(x)
            model_output = (
                ((x,) + model_output[1:]) if isinstance(model_output, tuple) else x
            )
        return model_output
