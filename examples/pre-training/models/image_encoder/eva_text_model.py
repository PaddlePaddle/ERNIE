# coding=utf-8
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

import collections
import logging
import math
import os
from typing import Callable, Optional, Union

import paddle
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle import nn
from paddle.common_ops_import import convert_dtype
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker

from .utils import params_normal_

try:
    from .modules.fusedln import FusedLayerNorm
except Exception:
    from paddle.nn import LayerNorm as FusedLayerNorm

    print("Warning, FusedLn module is not available, use LayerNorm instead.")
try:
    from paddle.incubate.nn.memory_efficient_attention import (
        LowerTriangularMask,
        memory_efficient_attention,
    )
except Exception:
    print("Warning: import memory_efficient_attention error")

from paddleformers.transformers.configuration_utils import PretrainedConfig

# from paddlemix.models.model_utils import MixPretrainedModel
from paddleformers.transformers.model_utils import PretrainedModel


logger = logging.getLogger(__name__)

USE_TENSOR_PARALLEL = False


def _convert_attention_mask(attn_mask, dtype):
    """
    Convert the attention mask to the target dtype we expect.

    Parameters:
        attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.
        dtype (VarType): The target type of `attn_mask` we expect.

    Returns:
        Tensor: A Tensor with shape same as input `attn_mask`, with data type `dtype`.
    """
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = convert_dtype(attn_mask.dtype)
        if attn_mask_dtype == "bool" or "int" in attn_mask_dtype:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e9
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask


class MultiHeadAttention(paddle.nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    """

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        kdim=None,
        vdim=None,
        need_weights=False,
        weight_attr=None,
        bias_attr=None,
        fuse_attention_qkv=False,
        num_partitions=1,
    ):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights
        self.fuse_attention_qkv = fuse_attention_qkv

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        assert self.num_heads % num_partitions == 0
        self.num_heads = self.num_heads // num_partitions

        if self.fuse_attention_qkv:
            assert self.kdim == embed_dim, "embed_dim should be equal to kdim"
            assert self.vdim == embed_dim, "embed_dim should be equal to vidm"

            if USE_TENSOR_PARALLEL:
                self.qkv_proj = fleet.meta_parallel.ColumnParallelLinear(
                    embed_dim,
                    3 * embed_dim,
                    weight_attr=weight_attr,
                    has_bias=True,
                    gather_output=False,
                )
            else:
                self.qkv_proj = paddle.nn.Linear(
                    embed_dim,
                    3 * embed_dim,
                    weight_attr=weight_attr,
                )
        else:
            if USE_TENSOR_PARALLEL:
                self.q_proj = fleet.meta_parallel.ColumnParallelLinear(
                    embed_dim,
                    embed_dim,
                    weight_attr=weight_attr,
                    has_bias=True,
                    gather_output=False,
                )

                self.k_proj = fleet.meta_parallel.ColumnParallelLinear(
                    self.kdim,
                    embed_dim,
                    weight_attr=weight_attr,
                    has_bias=True,
                    gather_output=False,
                )

                self.v_proj = fleet.meta_parallel.ColumnParallelLinear(
                    self.vdim,
                    embed_dim,
                    weight_attr=weight_attr,
                    has_bias=True,
                    gather_output=False,
                )
            else:
                self.q_proj = paddle.nn.Linear(
                    embed_dim,
                    embed_dim,
                    weight_attr=weight_attr,
                )
                self.k_proj = paddle.nn.Linear(
                    self.kdim,
                    embed_dim,
                    weight_attr=weight_attr,
                )
                self.v_proj = paddle.nn.Linear(
                    self.vdim,
                    embed_dim,
                    weight_attr=weight_attr,
                )

        # TODO
        if USE_TENSOR_PARALLEL:
            self.out_proj = fleet.meta_parallel.RowParallelLinear(
                embed_dim,
                embed_dim,
                weight_attr=weight_attr,
                has_bias=True,
                input_is_parallel=True,
            )
        else:
            self.out_proj = paddle.nn.Linear(
                embed_dim,
                embed_dim,
                weight_attr=weight_attr,
            )

    def _fuse_prepare_qkv(self, query):
        mix_layer = self.qkv_proj(query)
        mix_layer = paddle.reshape_(mix_layer, [0, 0, -1, 3 * self.head_dim])
        mix_layer = paddle.transpose(mix_layer, [0, 2, 1, 3])
        q, k, v = paddle.split(mix_layer, num_or_sections=3, axis=-1)
        return q, k, v

    def _prepare_qkv(self, query, key, value, use_cache=False, cache=None):
        r"""
        Prapares linear projected queries, keys and values for usage of subsequnt
        multiple parallel attention. If `cache` is not None, using cached results
        to reduce redundant calculations.

        """
        q = self.q_proj(query)
        q = tensor.reshape(x=q, shape=[0, 0, -1, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=2)
            v = tensor.concat([cache.v, v], axis=2)
        if use_cache is True:
            cache = self.Cache(k, v)

        return (q, k, v) if use_cache is False else (q, k, v, cache)

    def compute_kv(self, key, value):
        r"""
        Applies linear projection on input keys and values, then splits heads
        (reshape and transpose) to get keys and values from different representation
        subspaces. The results are used as key-values pairs for subsequent multiple
        parallel attention.

        It is part of calculations in multi-head attention, and is provided as
        a method to pre-compute and prefetch these results, thus we can use them
        to construct cache for inference.

        """
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = tensor.reshape(x=k, shape=[0, 0, -1, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, -1, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def gen_cache(self, key, value=None, type=Cache):
        """
        Generates cache for `forward` usage in inference accroding to arguments.
        The generated cache is an instance of `MultiHeadAttention.Cache` or an
        instance of `MultiHeadAttention.StaticCache`.
        """
        if type == MultiHeadAttention.StaticCache:  # static_kv
            k, v = self.compute_kv(key, value)
            return self.StaticCache(k, v)
        elif value is None:  # incremental_state
            shape = [key.shape[0], -1, self.num_heads, 0, self.head_dim]
            k = paddle.zeros(shape, dtype=key.dtype, value=0)
            v = paddle.zeros(shape, dtype=key.dtype, value=0)
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def forward(self, query, key, value, attn_mask=None, use_cache=False, cache=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if use_cache is False:
            if self.fuse_attention_qkv:
                q, k, v = self._fuse_prepare_qkv(query)
            else:
                q, k, v = self._prepare_qkv(query, key, value, use_cache, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, use_cache, cache)
        # scale dot product attention
        product = paddle.matmul(x=q * (self.head_dim**-0.5), y=k, transpose_y=True)

        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        weights = F.softmax(product)

        if self.dropout:
            with get_rng_state_tracker().rng_state("local_seed"):
                weights = F.dropout(
                    weights,
                    self.dropout,
                    training=self.training,
                    mode="upscale_in_train",
                )

        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if use_cache:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


class LayerNormFp32(paddle.nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: paddle.Tensor):
        output = paddle.nn.functional.layer_norm(
            x=x.astype(dtype="float32"),
            normalized_shape=self._normalized_shape,
            weight=(
                self.weight.astype(dtype="float32") if self.weight is not None else None
            ),
            bias=self.bias.astype(dtype="float32") if self.bias is not None else None,
            epsilon=self._epsilon,
        )
        return output.astype(dtype=x.dtype)


class LayerNorm(paddle.nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: paddle.Tensor):
        orig_type = x.dtype
        x = paddle.nn.functional.layer_norm(
            x=x,
            normalized_shape=self._normalized_shape,
            weight=self.weight,
            bias=self.bias,
            epsilon=self._epsilon,
        )
        if isinstance(orig_type, paddle.dtype):
            dtype = orig_type
        elif isinstance(orig_type, str) and orig_type not in [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
        ]:
            dtype = orig_type
        elif isinstance(orig_type, paddle.Tensor):
            dtype = orig_type.dtype
        else:
            dtype = x.dtype
        return x.cast(dtype)


class QuickGELU(paddle.nn.Layer):
    def forward(self, x: paddle.Tensor):
        return x * paddle.nn.functional.sigmoid(x=1.702 * x)


class LayerScale(paddle.nn.Layer):
    def __init__(self, dim, init_values=1e-05):
        super().__init__()
        init_data = init_values * paddle.ones(shape=[dim])
        self.gamma = self.create_parameter(
            shape=[dim], default_initializer=paddle.nn.initializer.Assign(init_data)
        )

    def forward(self, x):
        return x * self.gamma


class PatchDropout(paddle.nn.Layer):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, rope, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token
        self.rope = rope

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x
        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = x[:, :1]
        batch = x.shape[0]
        num_tokens = x.shape[1]
        batch_indices = paddle.arange(end=batch)
        batch_indices = batch_indices[..., None]
        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))
        rand = paddle.randn(shape=[batch, num_tokens])
        _, patch_indices_keep = rand.topk(k=num_patches_keep, axis=-1)
        x = x[batch_indices, patch_indices_keep]
        if self.exclude_first_token:
            x = paddle.concat(x=(cls_tokens, x), axis=1)
        if self.training and self.rope:
            return x, patch_indices_keep
        return x


class PatchDropoutForAdaptive(paddle.nn.Layer):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, rope, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token
        self.rope = rope

    def forward(self, x, position_ids, image_sizes, position_ids_2d):
        """
        Args:
            x: Tensor[num_tokens, Dim],
            position_ids: Tensor[num_tokens],
            image_sizes: Tensor[num_samples, 2], tensor of (patch_height, patch_width)
            position_ids_2d: 2d position, but without cls token
            position_ids: flatten position, with cls token
        Returns:
            x: Tensor[num_tokens * drop_rate, Dim],
            position_ids: Tensor[num_tokens * drop_rate],
        """
        if not self.training or self.prob == 0.0:
            return x, position_ids, position_ids_2d

        if isinstance(image_sizes, paddle.Tensor):
            image_sizes = image_sizes.numpy()
        cls_pos_norm = paddle.to_tensor([-1.0])
        rand_pos_norm_list = [
            paddle.concat(
                [
                    cls_pos_norm,
                    (paddle.randperm(int(s.prod()), dtype="float32") + 1.0) / s.prod(),
                ],
                0,
            )
            for s in image_sizes
        ]
        rand_pos_norm = paddle.concat(rand_pos_norm_list, 0)
        keep_prob = 1.0 - self.prob
        assert rand_pos_norm.shape[0] == x.shape[0], (rand_pos_norm.shape, x.shape)
        (keep_pos,) = paddle.where(rand_pos_norm < keep_prob)
        (keep_pos_2d,) = paddle.where((rand_pos_norm < keep_prob) & (position_ids != 0))
        keep_pos = keep_pos.squeeze()
        keep_pos_2d = keep_pos_2d.squeeze()
        retained_x = x[keep_pos]
        retained_pid = position_ids[keep_pos]
        retained_pid_2d = paddle.zeros(
            [position_ids.shape[0], 2], dtype=position_ids_2d.dtype
        )
        retained_pid_2d[position_ids != 0] = position_ids_2d
        retained_pid_2d = retained_pid_2d[keep_pos_2d]

        return retained_x, retained_pid, retained_pid_2d


def _in_projection_packed(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    w: paddle.Tensor,
    b: Optional[paddle.Tensor] = None,
):
    """
    https://github.com/pytorch/pytorch/blob/db2a237763eb8693a20788be94f8c192e762baa8/torch/nn/functional.py#L4726
    """
    E = q.shape[-1]
    if k is v:
        if q is k:
            return paddle.nn.functional.linear(x=q, weight=w, bias=b).chunk(
                chunks=3, axis=-1
            )
        else:
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (
                paddle.nn.functional.linear(x=q, weight=w_q, bias=b_q),
            ) + paddle.nn.functional.linear(x=k, weight=w_kv, bias=b_kv).chunk(
                chunks=2, axis=-1
            )
    else:
        w_q, w_k, w_v = w.chunk(chunks=3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(chunks=3)
        return (
            paddle.nn.functional.linear(x=q, weight=w_q, bias=b_q),
            paddle.nn.functional.linear(x=k, weight=w_k, bias=b_k),
            paddle.nn.functional.linear(x=v, weight=w_v, bias=b_v),
        )


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


class Attention(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        scaled_cosine=False,
        scale_heads=False,
        logit_scale_max=math.log(1.0 / 0.01),
        attn_drop=0.0,
        proj_drop=0.0,
        xattn=False,
        rope=False,
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.logit_scale_max = logit_scale_max
        origin_dtype = paddle.get_default_dtype()
        paddle.set_default_dtype("float32")
        init_data = paddle.randn(shape=[dim, dim * 3]) * self.scale
        if origin_dtype != "float32":
            init_data.astype(origin_dtype)
        paddle.set_default_dtype(origin_dtype)
        self.in_proj_weight = self.create_parameter(
            shape=[dim, dim * 3],
            default_initializer=paddle.nn.initializer.Assign(init_data),
        )
        if qkv_bias:
            init_data = paddle.zeros(shape=[dim * 3])
            self.in_proj_bias = self.create_parameter(
                shape=[dim * 3],
                default_initializer=paddle.nn.initializer.Assign(init_data),
            )
        else:
            self.in_proj_bias = None
        if self.scaled_cosine:
            init_data = paddle.log(x=10 * paddle.ones(shape=[num_heads, 1, 1]))
            self.logit_scale = self.create_parameter(
                shape=[num_heads, 1, 1],
                default_initializer=paddle.nn.initializer.Assign(init_data),
            )
        else:
            self.logit_scale = None
        self.attn_drop = paddle.nn.Dropout(p=attn_drop)
        if self.scale_heads:
            init_data = paddle.ones(shape=[num_heads, 1, 1])
            self.head_scale = self.create_parameter(
                shape=[num_heads, 1, 1],
                default_initializer=paddle.nn.initializer.Assign(init_data),
            )
        else:
            self.head_scale = None
        if USE_TENSOR_PARALLEL:
            self.out_proj = fleet.meta_parallel.ColumnParallelLinear(
                dim, dim, weight_attr=None, has_bias=True, gather_output=True
            )
        else:
            self.out_proj = paddle.nn.Linear(dim, dim)
        self.out_drop = paddle.nn.Dropout(p=proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop
        self.rope = rope

    def forward(self, x, attn_mask: Optional[paddle.Tensor] = None):
        L, N, C = x.shape
        q, k, v = paddle.nn.functional.linear(
            x=x, weight=self.in_proj_weight, bias=self.in_proj_bias
        ).chunk(chunks=3, axis=-1)
        if self.xattn:
            x = q.reshape((L, N, self.num_heads, -1))
            perm_3 = list(range(x.ndim))
            perm_3[0] = 1
            perm_3[1] = 0
            q = x.transpose(perm=perm_3)
            x = k.reshape((L, N, self.num_heads, -1))
            perm_4 = list(range(x.ndim))
            perm_4[0] = 1
            perm_4[1] = 0
            k = x.transpose(perm=perm_4)
            x = v.reshape((L, N, self.num_heads, -1))
            perm_5 = list(range(x.ndim))
            perm_5[0] = 1
            perm_5[1] = 0
            v = x.transpose(perm=perm_5)
            x = memory_efficient_attention(
                q,
                k,
                v,
                p=self.xattn_drop,
                scale=self.scale if self.logit_scale is None else None,
                attn_bias=LowerTriangularMask() if attn_mask is not None else None,
            )
        else:
            x = q.reshape((L, N * self.num_heads, -1))
            perm_6 = list(range(x.ndim))
            perm_6[0] = 1
            perm_6[1] = 0
            q = x.transpose(perm=perm_6)
            x = k.reshape((L, N * self.num_heads, -1))
            perm_7 = list(range(x.ndim))
            perm_7[0] = 1
            perm_7[1] = 0
            k = x.transpose(perm=perm_7)
            x = v.reshape((L, N * self.num_heads, -1))
            perm_8 = list(range(x.ndim))
            perm_8[0] = 1
            perm_8[1] = 0
            v = x.transpose(perm=perm_8)
            if self.logit_scale is not None:
                x = paddle.nn.functional.normalize(x=k, axis=-1)
                perm_9 = list(range(x.ndim))
                perm_9[-1] = -2
                perm_9[-2] = -1
                attn = paddle.bmm(
                    x=paddle.nn.functional.normalize(x=q, axis=-1),
                    y=x.transpose(perm=perm_9),
                )
                logit_scale = paddle.clip(
                    x=self.logit_scale, max=self.logit_scale_max
                ).exp()
                attn = attn.reshape((N, self.num_heads, L, L)) * logit_scale
                attn = attn.reshape((-1, L, L))
            else:
                q = q * self.scale
                x = k
                perm_10 = list(range(x.ndim))
                perm_10[-1] = -2
                perm_10[-2] = -1
                attn = paddle.bmm(x=q, y=x.transpose(perm=perm_10))
            if attn_mask is not None:
                if attn_mask.dtype == "bool":
                    new_attn_mask = paddle.zeros_like(x=attn_mask).astype(q.dtype)
                    # new_attn_mask.masked_fill_(attn_mask, float('-inf'))
                    new_attn_mask = masked_fill(new_attn_mask, attn_mask, float("-inf"))
                    attn_mask = new_attn_mask
                attn += attn_mask
            attn = paddle.nn.functional.softmax(attn, axis=-1)
            with get_rng_state_tracker().rng_state("global_seed"):
                attn = self.attn_drop(attn)
            x = paddle.bmm(x=attn, y=v)
        if self.head_scale is not None:
            x = x.reshape((N, self.num_heads, L, C)) * self.head_scale
            x = x.reshape((-1, L, C))
        x = x
        perm_11 = list(range(x.ndim))
        perm_11[0] = 1
        perm_11[1] = 0
        x = x.transpose(perm=perm_11).reshape((L, N, C))
        x = self.out_proj(x)
        with get_rng_state_tracker().rng_state("global_seed"):
            x = self.out_drop(x)
        return x


class CustomAttention(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        scaled_cosine=True,
        scale_heads=False,
        logit_scale_max=math.log(1.0 / 0.01),
        attn_drop=0.0,
        proj_drop=0.0,
        xattn=False,
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.logit_scale_max = logit_scale_max
        origin_dtype = paddle.get_default_dtype()
        paddle.set_default_dtype("float32")
        init_data = paddle.randn(shape=[dim, dim * 3]) * self.scale
        if origin_dtype != "float32":
            init_data.astype(origin_dtype)
        paddle.set_default_dtype(origin_dtype)
        self.in_proj_weight = self.create_parameter(
            shape=[dim, dim * 3],
            default_initializer=paddle.nn.initializer.Assign(init_data),
        )
        if qkv_bias:
            self.in_proj_bias = self.create_parameter(
                shape=[dim * 3],
                default_initializer=paddle.nn.initializer.Assign(
                    paddle.zeros(shape=[dim * 3])
                ),
            )
        else:
            self.in_proj_bias = None
        if self.scaled_cosine:
            init_data = paddle.log(x=10 * paddle.ones(shape=[num_heads, 1, 1]))
            self.logit_scale = self.create_parameter(
                shape=[num_heads, 1, 1],
                default_initializer=paddle.nn.initializer.Assign(init_data),
            )
        else:
            self.logit_scale = None
        self.attn_drop = paddle.nn.Dropout(p=attn_drop)
        if self.scale_heads:
            init_data = paddle.ones(shape=[num_heads, 1, 1])
            self.head_scale = self.create_parameter(
                shape=[num_heads, 1, 1],
                default_initializer=paddle.nn.initializer.Assign(init_data),
            )
        else:
            self.head_scale = None
        if USE_TENSOR_PARALLEL:
            self.out_proj = fleet.meta_parallel.ColumnParallelLinear(
                dim, dim, weight_attr=None, has_bias=True, gather_output=True
            )
        else:
            self.out_proj = paddle.nn.Linear(dim, dim)
        self.out_drop = paddle.nn.Dropout(p=proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop

    def forward(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor,
        attn_mask: Optional[paddle.Tensor] = None,
    ):
        q, k, v = _in_projection_packed(
            query, key, value, self.in_proj_weight, self.in_proj_bias
        )
        N_q, B_q, C_q = q.shape
        N_k, B_k, C_k = k.shape
        N_v, B_v, C_v = v.shape
        if self.xattn:
            q = q.transpose(perm=[1, 0, 2]).reshape((B_q, N_q, self.num_heads, -1))
            k = k.transpose(perm=[1, 0, 2]).reshape((B_k, N_k, self.num_heads, -1))
            v = v.transpose(perm=[1, 0, 2]).reshape((B_v, N_v, self.num_heads, -1))
            x = memory_efficient_attention(
                q,
                k,
                v,
                p=self.xattn_drop,
                scale=self.scale if self.logit_scale is None else None,
                attn_bias=LowerTriangularMask() if attn_mask is not None else None,
            )

        else:
            x = q.reshape((N_q, B_q * self.num_heads, -1))
            perm_12 = list(range(x.ndim))
            perm_12[0] = 1
            perm_12[1] = 0
            q = x.transpose(perm=perm_12)
            x = k.reshape((N_k, B_k * self.num_heads, -1))
            perm_13 = list(range(x.ndim))
            perm_13[0] = 1
            perm_13[1] = 0
            k = x.transpose(perm=perm_13)
            x = v.reshape((N_v, B_v * self.num_heads, -1))
            perm_14 = list(range(x.ndim))
            perm_14[0] = 1
            perm_14[1] = 0
            v = x.transpose(perm=perm_14)
            if self.logit_scale is not None:
                x = paddle.nn.functional.normalize(x=k, axis=-1)
                perm_15 = list(range(x.ndim))
                perm_15[-1] = -2
                perm_15[-2] = -1
                attn = paddle.bmm(
                    x=paddle.nn.functional.normalize(x=q, axis=-1),
                    y=x.transpose(perm=perm_15),
                )
                logit_scale = paddle.clip(
                    x=self.logit_scale, max=self.logit_scale_max
                ).exp()
                attn = attn.reshape((B_q, self.num_heads, N_q, N_k)) * logit_scale
                attn = attn.reshape((-1, N_q, N_k))
            else:
                q = q * self.scale
                x = k
                perm_16 = list(range(x.ndim))
                perm_16[-1] = -2
                perm_16[-2] = -1
                attn = paddle.bmm(x=q, y=x.transpose(perm=perm_16))
            if attn_mask is not None:
                if attn_mask.dtype == "bool":
                    new_attn_mask = paddle.zeros_like(x=attn_mask).astype(q.dtype)
                    new_attn_mask = masked_fill(new_attn_mask, attn_mask, float("-inf"))
                    attn_mask = new_attn_mask
                attn += attn_mask
            attn = paddle.nn.functional.softmax(attn, axis=-1)
            with get_rng_state_tracker().rng_state("global_seed"):
                attn = self.attn_drop(attn)
            x = paddle.bmm(x=attn, y=v)
        if self.head_scale is not None:
            x = x.reshape((B_q, self.num_heads, N_q, C_q)) * self.head_scale
            x = x.reshape((-1, N_q, C_q))
        x = x
        perm_17 = list(range(x.ndim))
        perm_17[0] = 1
        perm_17[1] = 0
        x = x.transpose(perm=perm_17).reshape((N_q, B_q, C_q))
        x = self.out_proj(x)
        with get_rng_state_tracker().rng_state("global_seed"):
            x = self.out_drop(x)
        return x


class CustomResidualAttentionBlock(paddle.nn.Layer):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = paddle.nn.GELU,
        norm_layer: Callable = LayerNorm,
        scale_cosine_attn: bool = False,
        scale_heads: bool = False,
        scale_attn: bool = False,
        scale_fc: bool = False,
        cross_attn: bool = False,
        xattn: bool = False,
    ):
        super().__init__()
        self.ln_1 = norm_layer(d_model)
        self.ln_1_k = norm_layer(d_model) if cross_attn else self.ln_1
        self.ln_1_v = norm_layer(d_model) if cross_attn else self.ln_1
        self.attn = CustomAttention(
            d_model,
            n_head,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
            xattn=xattn,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else paddle.nn.Identity()
        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else paddle.nn.Identity()
        )
        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        if USE_TENSOR_PARALLEL:
            self.mlp = paddle.nn.Sequential(
                *[
                    (
                        "c_fc",
                        fleet.meta_parallel.ColumnParallelLinear(
                            d_model,
                            mlp_width,
                            weight_attr=None,
                            has_bias=True,
                            gather_output=True,
                        ),
                    ),
                    ("ln", norm_layer(mlp_width) if scale_fc else paddle.nn.Identity()),
                    ("gelu", act_layer()),
                    (
                        "c_proj",
                        fleet.meta_parallel.ColumnParallelLinear(
                            mlp_width,
                            d_model,
                            weight_attr=None,
                            has_bias=True,
                            gather_output=True,
                        ),
                    ),
                ]
            )
        else:
            self.mlp = paddle.nn.Sequential(
                *[
                    ("c_fc", paddle.nn.Linear(d_model, mlp_width)),
                    ("ln", norm_layer(mlp_width) if scale_fc else paddle.nn.Identity()),
                    ("gelu", act_layer()),
                    ("c_proj", paddle.nn.Linear(mlp_width, d_model)),
                ]
            )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else paddle.nn.Identity()
        )

    def forward(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        attn_mask: Optional[paddle.Tensor] = None,
    ):
        q = q + self.ls_1(
            self.ln_attn(
                self.attn(
                    self.ln_1(q), self.ln_1_k(k), self.ln_1_v(v), attn_mask=attn_mask
                )
            )
        )
        q = q + self.ls_2(self.mlp(self.ln_2(q)))
        return q


class ResidualAttentionBlock(paddle.nn.Layer):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        xattn: bool = False,
        is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        if xattn:
            self.attn = Attention(d_model, n_head, xattn=True)
        else:
            self.attn = MultiHeadAttention(d_model, n_head)
        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        if USE_TENSOR_PARALLEL:
            self.mlp = paddle.nn.Sequential(
                *[
                    (
                        "c_fc",
                        fleet.meta_parallel.ColumnParallelLinear(
                            d_model,
                            mlp_width,
                            weight_attr=None,
                            has_bias=True,
                            gather_output=True,
                        ),
                    ),
                    ("gelu", act_layer()),
                    (
                        "c_proj",
                        fleet.meta_parallel.ColumnParallelLinear(
                            mlp_width,
                            d_model,
                            weight_attr=None,
                            has_bias=True,
                            gather_output=True,
                        ),
                    ),
                ]
            )
        else:
            self.mlp = paddle.nn.Sequential(
                *[
                    ("c_fc", paddle.nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", paddle.nn.Linear(mlp_width, d_model)),
                ]
            )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        self.xattn = xattn

    def attention(
        self,
        q_x,
        k_x=None,
        v_x=None,
        attn_mask=None,
    ):

        if isinstance(q_x.dtype, paddle.dtype):
            dtype = q_x.dtype
        elif isinstance(q_x.dtype, str) and q_x.dtype not in [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
        ]:
            dtype = q_x.dtype
        elif isinstance(q_x.dtype, paddle.Tensor):
            dtype = q_x.dtype.dtype
        else:
            dtype = attn_mask.dtype
        attn_mask = attn_mask.cast(dtype) if attn_mask is not None else None
        if self.xattn:
            return self.attn(q_x, attn_mask=attn_mask)

        attn_mask = attn_mask if attn_mask is not None else None
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        out = self.attn(q_x, k_x, v_x, attn_mask=attn_mask)
        return out

    def forward(
        self,
        q_x,
        k_x=None,
        v_x=None,
        attn_mask=None,
    ):
        k_x = (
            self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        )
        v_x = (
            self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        )

        x = self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        x = x + q_x
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(paddle.nn.Layer):
    def __init__(
        self,
        config,
        act_layer: Callable = paddle.nn.GELU,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.enable_recompute = False
        self.width = config.width
        self.layers = config.layers
        self.resblocks = paddle.nn.LayerList(
            sublayers=[
                ResidualAttentionBlock(
                    config.width,
                    config.heads,
                    mlp_ratio=config.mlp_ratio,
                    ls_init_value=config.ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    xattn=config.xattn,
                )
                for _ in range(config.layers)
            ]
        )

    def get_cast_dtype(self) -> paddle.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: paddle.Tensor, attn_mask: Optional[paddle.Tensor] = None):
        cnt = 0
        for r in self.resblocks:
            if self.enable_recompute:
                x = paddle.distributed.fleet.utils.recompute(
                    r, x, attn_mask, use_reentrant=False
                )
            else:
                x = r(x, attn_mask=attn_mask)
                cnt += 1
        return x


class EVATextTransformerConfig(PretrainedConfig):

    model_type = "evatext_transformer"

    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        ls_init_value: float = None,
        embed_dim: int = 512,
        output_dim: int = 1024,
        # act_layer: Callable = paddle.nn.GELU,
        # norm_layer: Callable = LayerNorm,
        xattn: bool = False,
        attn_mask: bool = True,
        pad_id: int = 0,
        embed_cls: bool = False,
        output_tokens: bool = False,
        quick_gelu: bool = False,
        mlp_ratio: float = 4.0,
        fusedLN: bool = False,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.heads = heads
        self.layers = layers
        self.ls_init_value = ls_init_value
        self.output_dim = output_dim
        # self.act_layer = act_layer
        # self.norm_layer = norm_layer
        self.xattn = xattn
        self.attn_mask = attn_mask
        self.pad_id = pad_id
        self.embed_cls = embed_cls
        self.output_tokens = output_tokens

        self.quick_gelu = quick_gelu
        self.mlp_ratio = mlp_ratio
        self.fusedLN = fusedLN

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
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

        if "text_cfg" in config_dict:
            config_dict = config_dict["text_cfg"]

        return cls.from_dict(config_dict, **kwargs)


class EVATextTransformerPretrainedModel(PretrainedModel):
    """
    See :class:`paddlemix.models.model_utils.MixPretrainedModel` for more details.
    """

    model_config_file = "config.json"
    config_class = EVATextTransformerConfig
    resource_files_names = {"model_state": "model_state_text.pdparams"}
    base_model_prefix = "evatext_transformer"


class EVATextTransformer(EVATextTransformerPretrainedModel):
    def __init__(self, config: EVATextTransformerConfig):
        super().__init__(config)

        self.width = width = config.width
        self.output_dim = config.output_dim
        self.pad_id = config.pad_id
        norm_layer = FusedLayerNorm if config.fusedLN else LayerNorm
        act_layer = QuickGELU if config.quick_gelu else paddle.nn.GELU

        self.num_pos = config.context_length
        self.heads = config.heads
        if USE_TENSOR_PARALLEL:
            self.token_embedding = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size, width
            )
        else:
            self.token_embedding = paddle.nn.Embedding(config.vocab_size, width)
        self.transformer = Transformer(
            config, act_layer=act_layer, norm_layer=norm_layer
        )
        self.ln_final = norm_layer(width)
        init_data = paddle.empty(shape=[width, self.output_dim])
        self.text_projection = self.create_parameter(
            shape=[width, self.output_dim],
            default_initializer=paddle.nn.initializer.Assign(init_data),
        )
        self.text_projection = params_normal_(
            self.text_projection, std=self.transformer.width**-0.5
        )
        self.output_tokens = config.output_tokens
        if config.embed_cls:
            self.cls_emb = self.create_parameter(shape=[width])
            self.num_pos += 1
        else:
            self.cls_emb = None
        init_data = paddle.empty(shape=[self.num_pos, width])
        self.positional_embedding = self.create_parameter(
            shape=[self.num_pos, width],
            default_initializer=paddle.nn.initializer.Assign(init_data),
        )
        if config.attn_mask:
            self.register_buffer(
                "attn_mask", self.build_attention_mask(), persistable=False
            )
        else:
            self.attn_mask = None
        # self.init_parameters()

    def init_parameters(self):
        self.token_embedding.weight = params_normal_(
            self.token_embedding.weight, std=0.02
        )
        self.positional_embedding = params_normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            self.cls_emb = params_normal_(self.cls_emb, std=0.01)

        proj_std = self.transformer.width**-0.5 * (2 * self.transformer.layers) ** -0.5
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            block.attn.q_proj.weight = params_normal_(
                block.attn.q_proj.weight, std=attn_std
            )
            block.attn.k_proj.weight = params_normal_(
                block.attn.k_proj.weight, std=attn_std
            )
            block.attn.v_proj.weight = params_normal_(
                block.attn.v_proj.weight, std=attn_std
            )
            block.attn.out_proj.weight = params_normal_(
                block.attn.out_proj.weight, std=proj_std
            )
            block.mlp.c_fc.weight = params_normal_(block.mlp.c_fc.weight, std=fc_std)
            block.mlp.c_proj.weight = params_normal_(
                block.mlp.c_proj.weight, std=proj_std
            )
        if self.text_projection is not None:
            self.text_projection = params_normal_(
                self.text_projection, std=self.transformer.width**-0.5
            )

    def set_grad_checkpointing(self, enable=True):
        self.transformer.enable_recompute = enable

    def no_weight_decay(self):
        return {"positional_embedding"}

    def get_num_layers(self):
        return self.transformer.layers

    def build_attention_mask(self):
        mask = paddle.empty(shape=[self.num_pos, self.num_pos])
        mask.fill_(value=float("-inf"))
        mask = paddle.triu(mask, 1)
        return mask

    def build_cls_mask(self, text, cast_dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1).astype(text.dtype)
        cls_mask = paddle.nn.functional.pad(
            cls_mask,
            (0, 0, cls_mask.shape[2], 0, 1, 0),
            value=1.0,
        )
        additive_mask = paddle.zeros(cls_mask.shape, dtype=cast_dtype)
        additive_mask = masked_fill(
            additive_mask, (1.0 - cls_mask).astype("bool"), float("-inf")
        )
        additive_mask = additive_mask.unsqueeze(1)
        additive_mask = paddle.repeat_interleave(additive_mask, self.heads, 1)
        return additive_mask

    def _repeat(self, t, N: int):
        return t.reshape((1, 1, -1)).repeat_interleave(N, 0)

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]
        if isinstance(cast_dtype, paddle.dtype):
            dtype = cast_dtype
        elif isinstance(cast_dtype, str) and cast_dtype not in [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
        ]:
            dtype = cast_dtype
        elif isinstance(cast_dtype, paddle.Tensor):
            dtype = cast_dtype.dtype
        else:
            dtype = self.token_embedding(text).dtype
        x = self.token_embedding(text).cast(dtype)
        attn_mask = self.attn_mask

        if self.cls_emb is not None:
            seq_len += 1
            x = paddle.concat([x, self._repeat(self.cls_emb, x.shape[0])], axis=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            attn_mask = (
                attn_mask[:seq_len, :seq_len].unsqueeze(0)
                + cls_mask[:, :seq_len, :seq_len]
            )

        if isinstance(cast_dtype, paddle.dtype):
            dtype = cast_dtype
        elif isinstance(cast_dtype, str) and cast_dtype not in [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
        ]:
            dtype = cast_dtype
        elif isinstance(cast_dtype, paddle.Tensor):
            dtype = cast_dtype.dtype
        else:
            dtype = self.positional_embedding.dtype
        x = x + self.positional_embedding.cast(dtype)[:seq_len, :]
        attn_mask = attn_mask[:seq_len, :seq_len]

        x = self.transformer(x, attn_mask=attn_mask)

        if self.cls_emb is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            eos_indices = (text != self.pad_id).sum(-1) - 1
            pooled, tokens = x[paddle.arange(x.shape[0]), eos_indices], x
            if x.shape[0] == 1 and len(pooled.shape) == 1:
                pooled = pooled.unsqueeze(0)

        if self.text_projection is not None:
            pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens
        return pooled
