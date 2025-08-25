# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
"""Paddle Ernie model"""
import math
import functools
import logging
from typing import Optional, Tuple
import contextlib


from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker
from paddle.incubate.nn.memory_efficient_attention import (
    memory_efficient_attention,
    BlockDiagonalCausalMask,
)
from paddle.distributed import in_auto_parallel_align_mode


from models.moe.top2_gate_auto import Top2Gate, TopKGateFusedAuto


from paddleformers.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions as _BaseModelOutput,
)
from paddleformers.transformers.model_outputs import CausalLMOutputWithCrossAttentions

from paddleformers.transformers.model_utils import PretrainedModel, register_base_model


from models.moe.moe_layer_auto import (
    MOELayerAuto,
)
from models.ernie.configuration_auto import ErnieMoEConfig
from models.moe.moe_utils_auto import get_mesh

from paddle.nn.functional.flash_attention import (
    scaled_dot_product_attention as flash_attention_with_mask,
)
from paddle.nn.functional.flash_attention import flash_attention

from to_block_diag_causal_mask import to_block_diag_causal_mask


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(_BaseModelOutput):

    router_loss: Optional[paddle.Tensor] = None
    gate_logits: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class CausalLMOutputWithCrossAttentionsAuto(CausalLMOutputWithCrossAttentions):

    router_loss: Optional[paddle.Tensor] = None


logger = logging.getLogger(__name__)


try:
    from fast_ln import fast_ln
except ImportError:
    fast_ln = None


try:
    from paddle.incubate.nn.functional import (
        fused_rotary_position_embedding as fused_rope,
    )
except (ImportError, ModuleNotFoundError):
    logger.warning("fused_rotary_position_embedding not found")
    fused_rope = None

try:
    from paddle.incubate.nn.functional import swiglu as fused_swiglu
except (ImportError, ModuleNotFoundError):
    fused_swiglu = None


__all__ = [
    "ErnieModelAuto",
    "ErniePretrainedModelAuto",
    "ErnieForCausalLMAuto",
]


def subbatch(f, arg_idx, axis, bs, out_idx, use_recompute=False, same_arg_idx={}):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):

        assert len(arg_idx) == len(
            axis
        ), "Number of batching args and number of batching dims should match."

        inps = [args[i] for i in arg_idx]
        axis_width = [inp.shape[d] for inp, d in zip(inps, axis)]
        assert len(set(axis_width)) == 1, "Batch sizes should be kept equal."

        inp_axis = {inp: d for inp, d in zip(inps, axis)}

        axis_width = axis_width[0]
        if axis_width < bs:
            return f(*args, **kwargs)

        outs = []
        for slice_at in np.arange(0, axis_width, bs):
            _args = []
            for i, inp in enumerate(args):
                if i in same_arg_idx:
                    assert (
                        i > same_arg_idx[i]
                    ), f"expect i > same_arg_idx[i], but got i: {i} and same_arg_idx[i]: {same_arg_idx[i]}"
                    _args.append(_args[same_arg_idx[i]])
                elif i in arg_idx:
                    inp = inp.slice(
                        [inp_axis[inp]],
                        [slice_at],
                        [min(inp.shape[inp_axis[inp]], slice_at + bs)],
                    )
                    _args.append(inp)
                else:
                    _args.append(inp)
            if use_recompute:
                out = paddle.distributed.fleet.utils.recompute(f, *_args, **kwargs)
            else:
                out = f(*_args, **kwargs)
            outs.append(out)

        return paddle.concat(outs, out_idx)

    return wrapper


class FusedDropoutImpl(nn.Layer):

    def __init__(self, prob, mode):
        super().__init__()
        self.prob = prob
        self.mode = mode

        self.dropout = nn.Dropout(p=prob, mode=mode)

    def forward(self, x, y):
        if self.prob > 0:
            x = self.dropout(x)
        output = x + y

        return output


def is_pp_enable():

    mesh = fleet.auto.get_mesh()
    return "pp" in mesh.dim_names


def global_mesh_starts_with_pp():

    mesh = fleet.auto.get_mesh()
    if is_pp_enable():
        return mesh.get_mesh_with_dim("pp")
    else:
        return mesh


def get_triangle_upper_mask(x, mask=None):

    if mask is not None:
        return mask
    shape = x.shape
    shape[1] = 1
    mask = paddle.full(shape, -np.inf, dtype=x.dtype)
    mask.stop_gradient = True
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


def parallel_matmul(
    x,
    y,
    bias=None,
    transpose_y=False,
    tensor_parallel_degree=1,
    tensor_parallel_output=True,
):

    if transpose_y:
        logits = paddle.matmul(x, y, transpose_y=True)
        if bias is not None:
            logits += bias
    else:
        logits = F.linear(x, y, bias)

    if tensor_parallel_degree > 1 and not tensor_parallel_output:
        logits = dist.reshard(logits, get_mesh(-1), [dist.Shard(0), dist.Replicate()])

    return logits


def calc_lm_head_logits(
    config,
    hidden_states,
    weight,
    bias,
    sparse_label_idx=None,
    tensor_parallel_output=None,
):
    """the core function to calc lm head"""
    if config.sequence_parallel:

        assert (
            not config.use_sparse_head_and_loss_fn
        ), "use_sparse_head_and_loss_fn is not supported now."

        hcg = paddle.distributed.fleet.get_hybrid_communicate_group()
        dp_rank = hcg.get_data_parallel_rank()
        sharding_rank = hcg.get_sharding_parallel_rank()
        if dp_rank <= 1 and sharding_rank <= 1:
            hidden_states = dist.reshard(
                hidden_states,
                get_mesh(-1),
                [dist.Replicate(), dist.Replicate()],
            )
        else:
            hidden_states = dist.reshard(
                hidden_states,
                get_mesh(-1),
                [dist.Shard(1), dist.Replicate()],
            )
        # [S, B, H] to [B, S, H]
        hidden_states = paddle.transpose(hidden_states, [1, 0, 2])
        if not config.using_dynamic_sequence_length:
            hidden_states = hidden_states.reshape(
                [-1, config.seqlen, hidden_states.shape[-1]]
            )
        else:
            assert (
                config.micro_batch_size
            ), "micro_batch_size should be set when using dygramic sequence length."
            hidden_states = hidden_states.reshape(
                [config.micro_batch_size, -1, hidden_states.shape[-1]]
            )
    if tensor_parallel_output is None:
        tensor_parallel_output = config.tensor_parallel_output
    logits = parallel_matmul(
        hidden_states,
        weight,
        bias=bias,
        transpose_y=config.tie_word_embeddings,
        tensor_parallel_degree=config.tensor_parallel_degree,
        tensor_parallel_output=tensor_parallel_output,
    )

    return logits


def finfo(dtype: paddle.dtype = None):

    if dtype is None:
        dtype = paddle.get_default_dtype()

    if dtype == paddle.bfloat16:

        class BFloatFInfo:
            min = -3.3895313892515355e38

        return BFloatFInfo
    if dtype == paddle.float32:
        return np.finfo(np.float32)
    if dtype == paddle.float16:
        return np.finfo(np.float16)
    if dtype == paddle.float64:
        return np.finfo(np.float64)


def masked_fill(x, mask, value):

    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def mem_eff_attn(
    query, key, value, pack_offset, drop_prob=0.0, dtype=paddle.bfloat16, training=True
):

    pack_offset = pack_offset.numpy()
    shape = pack_offset.shape
    assert len(shape) == 2, len(shape)
    assert shape[0] == 1, shape[0]
    n = pack_offset.size
    pack_offset = pack_offset.flatten()
    seqlens = []
    assert pack_offset[0] == 0, pack_offset[0]
    for i in range(1, n):
        if pack_offset[i] < 0:
            break
        cur = pack_offset[i] - pack_offset[i - 1]
        assert cur > 0
        seqlens.append(cur)

    assert drop_prob == 0.0, drop_prob
    assert dtype == paddle.bfloat16, dtype

    def cast(x):
        return x.astype(dtype) if x.dtype != dtype else x

    if len(seqlens) == 1:
        out, _ = flash_attention(
            query, key, value, drop_prob, causal=True, training=training
        )
    else:
        mask = BlockDiagonalCausalMask.from_seqlens(seqlens)
        out = memory_efficient_attention(
            cast(query),
            cast(key),
            cast(value),
            attn_bias=mask,
            p=drop_prob,
            training=training,
        )
    return out


def scaled_dot_product_attention(
    query_states,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    config,
    is_causal=True,
    rr_flash_attn=None,
    inbatch_pack_offset=None,
    training=True,
):

    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, num_key_value_heads, _ = value_states.shape

    can_use_fa = config.use_flash_attn

    if not can_use_fa:
        if query_states.shape[-2] != key_states.shape[-2]:
            key_states = key_states.repeat_interleave(
                num_heads // num_key_value_heads, axis=-2
            )
        if query_states.shape[-2] != value_states.shape[-2]:
            value_states = value_states.repeat_interleave(
                num_heads // num_key_value_heads, axis=-2
            )

    if can_use_fa:
        if rr_flash_attn is not None:
            attn_output, attn_weights = rr_flash_attn(
                query_states,
                key_states,
                value_states,
                dropout=config.attention_probs_dropout_prob,
                causal=is_causal and query_states.shape[1] != 1,
                return_softmax=output_attentions,
            )
        else:
            attn_output, attn_weights = flash_attention(
                query_states,
                key_states,
                value_states,
                dropout=config.attention_probs_dropout_prob,
                causal=is_causal and query_states.shape[1] != 1,
                return_softmax=output_attentions,
            )

        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        return attn_output, attn_weights
    elif config.use_mem_eff_attn and inbatch_pack_offset is not None:
        assert (
            not output_attentions
        ), "output_attentions should be False when use_mem_eff_attn=True"
        if config.use_flash_attn_with_mask:

            attn_mask = to_block_diag_causal_mask(
                inbatch_pack_offset, q_len, float("-inf"), "bfloat16"
            )
            attn_output = flash_attention_with_mask(
                query_states,
                key_states,
                value_states,
                attn_mask,
                config.attention_probs_dropout_prob,
            )
        else:
            attn_output = mem_eff_attn(
                query_states,
                key_states,
                value_states,
                inbatch_pack_offset,
                drop_prob=config.attention_probs_dropout_prob,
            )
        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        return attn_output, None
    else:

        query_states = paddle.transpose(query_states, [0, 2, 1, 3]) / math.sqrt(
            head_dim
        )
        key_states = paddle.transpose(key_states, [0, 2, 1, 3])
        value_states = paddle.transpose(value_states, [0, 2, 1, 3])

        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2]))

        if attn_weights.shape != [bsz, num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention weights should be of shape {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is None:
            attention_mask = get_triangle_upper_mask(attn_weights)

        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])
        if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention mask should be of shape {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
            )
        if training:
            attn_weights = attention_mask + attn_weights
            attn_weights = paddle.maximum(
                attn_weights,
                paddle.to_tensor(
                    float(finfo(query_states.dtype).min), dtype=query_states.dtype
                ),
            )

            if paddle.in_dynamic_mode():
                with paddle.amp.auto_cast(False):
                    attn_weights = F.softmax(
                        attn_weights, axis=-1, dtype="float32"
                    ).astype(query_states.dtype)
            else:
                attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(
                    query_states.dtype
                )
        else:
            attn_weights = attn_weights.cast(paddle.float32)
            attention_mask = attention_mask.cast(paddle.float32)
            attn_weights = attn_weights.add_(attention_mask)
            attn_weights = F.softmax_(attn_weights, axis=-1).astype(query_states.dtype)

        if config.attention_probs_dropout_prob > 0.0:
            if config.tensor_parallel_degree > 1:
                with get_rng_state_tracker().rng_state("local_seed"):
                    attn_weights = F.dropout(
                        attn_weights,
                        config.attention_probs_dropout_prob,
                        training=training,
                        mode="upscale_in_train",
                    )
            else:
                attn_weights = F.dropout(
                    attn_weights,
                    config.attention_probs_dropout_prob,
                    training=training,
                    mode="upscale_in_train",
                )

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None


def _make_causal_mask(input_ids_shape, past_key_values_length, dtype):
    batch_size, target_length = input_ids_shape

    mask = paddle.full((target_length, target_length), float(finfo(dtype).min))

    mask_cond = paddle.arange(mask.shape[-1])
    mask = masked_fill(
        mask, mask_cond < (mask_cond + 1).reshape([mask.shape[-1], 1]), 0
    )

    if past_key_values_length > 0:
        mask = paddle.concat(
            [paddle.zeros([target_length, past_key_values_length]), mask], axis=-1
        )

    return mask[None, None, :, :].expand(
        [batch_size, 1, target_length, target_length + past_key_values_length]
    )


def _expand_mask(mask, dtype, tgt_length):
    if mask.ndim == 4:
        expanded_mask = mask
    elif mask.ndim == 3:
        expanded_mask = mask[:, None, :, :]
    else:
        batch_size, src_length = mask.shape[0], mask.shape[-1]
        tgt_length = tgt_length if tgt_length is not None else src_length

        expanded_mask = mask[:, None, None, :].expand(
            [batch_size, 1, tgt_length, src_length]
        )

    inverted_mask = 1.0 - expanded_mask
    return masked_fill(
        inverted_mask, inverted_mask.cast("bool"), float(finfo(dtype).min)
    )


def slice_experts(experts, moe_world_size):
    moe_num_experts_per_device = len(experts) // moe_world_size
    experts_per_device = [[] for _ in range(moe_world_size)]

    for i, expert in enumerate(experts):
        ep_group_id = i // moe_num_experts_per_device
        experts_per_device[ep_group_id].append(expert)

    lm_experts = nn.LayerList([])
    for experts_list in experts_per_device:
        lm_experts.extend(experts_list[: moe_num_experts_per_device // 2])
    return lm_experts


def get_gate(
    config: ErnieMoEConfig,
    expert: Tuple[Tuple[int, nn.Layer]],
    layer_idx: int,
    ipp: int = 0,
) -> Tuple[nn.Layer, nn.LayerList]:

    moe_num_experts = config.moe_num_experts
    assert (
        moe_num_experts >= config.moe_world_size
    ), f"expert moe_num_experts={moe_num_experts} >= moe_world_size={config.moe_world_size}"
    assert (
        moe_num_experts % config.moe_world_size == 0
    ), f"expert moe_num_experts={moe_num_experts} % moe_world_size={config.moe_world_size} == 0"
    moe_num_experts_per_device = moe_num_experts // config.moe_world_size
    experts = nn.LayerList([])
    for expert_id, (experts_num, fc) in enumerate(expert):
        assert experts_num % config.moe_world_size == 0
        experts_to_append = []
        if not hasattr(fc, "__len__"):
            experts_to_append.append(fc)
            if expert_id == 1:
                with paddle.utils.unique_name.guard("_mm_deepcopy"):
                    for _ in range(experts_num - 1):
                        experts_to_append.append(deepcopy(fc))
            else:
                for _ in range(experts_num - 1):
                    experts_to_append.append(deepcopy(fc))
        else:
            experts_to_append = fc
        for ex in experts_to_append:
            for p in ex.parameters():
                p.expert_type = f"expert_type_{expert_id}"
        experts.extend(experts_to_append)

    logger.info(
        f"using moe-world-size: {config.moe_world_size} "
        f"expert-per-device: {moe_num_experts_per_device} "
    )
    if config.moe_use_hard_gate and moe_num_experts <= 2:
        gate = None
        logger.info("MOE-GATE:-hard-gate")
    else:
        logger.info(f"MOE-GATE:-{config.moe_gate}")
        gate = TopKGateFusedAuto(
            config, layer_idx=layer_idx, group=config.moe_group, ipp=ipp
        )

    lm_gate, lm_experts = None, None
    logger.info(f"LM-experts-{lm_experts} -- experts-{experts}")

    index = 0 if config.moe_group == "dp" else 1
    ep_sub_meshes = dist.auto_parallel.api.split_mesh(get_mesh(ipp), index)

    for i, expert in enumerate(experts):
        ep_group_id = i // moe_num_experts_per_device
        if isinstance(expert, (ErnieMoeMLPFused, ErnieMoeMLP)):
            experts[i].redistribute_expert(
                ep_sub_meshes[ep_group_id], [dist.Replicate(), dist.Replicate()]
            )
            experts[i].ep_group_id = ep_group_id

    return gate, experts, lm_gate, lm_experts


def _parse_moe_group(moe_group: str):
    moe_group = moe_group.lower()
    assert moe_group in {
        "dp",
        "mp",
        "none",
    }, f"moe-group not supported, got: {moe_group}"
    logger.info(f"using moe-group: {moe_group}")

    return moe_group


class RMSNorm(nn.Layer):

    def __init__(self, config, ipp=0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

    def forward(self, hidden_states):

        if self.config.fuse_rms_norm:
            return paddle.incubate.nn.functional.fused_rms_norm_ext(
                hidden_states, self.weight, self.variance_epsilon
            )[0]
        if paddle.in_dynamic_mode():
            with paddle.amp.auto_cast(False):
                variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
                hidden_states = (
                    paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
                )
        else:
            variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
            hidden_states = (
                paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
            )

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return hidden_states * self.weight


class LayerNorm(nn.LayerNorm):

    def __init__(self, config, ipp=0):
        super().__init__(config.hidden_size, epsilon=config.rms_norm_eps)

        self.use_fast_ln = config.use_fast_ln
        if self.use_fast_ln:
            assert fast_ln is not None
        self.ipp = ipp
        if config.pipeline_parallel_degree > 1:
            self.weight = dist.shard_tensor(
                self.weight, get_mesh(self.ipp), [dist.Replicate(), dist.Replicate()]
            )
            self.bias = dist.shard_tensor(
                self.bias, get_mesh(self.ipp), [dist.Replicate(), dist.Replicate()]
            )

    def forward(self, hidden_states):
        if self.use_fast_ln:
            return fast_ln(hidden_states, self.weight, self.bias, self._epsilon)[0]
        else:
            return super().forward(hidden_states)


class RotaryEmbedding(nn.Layer):

    def __init__(self, dim, max_position_embeddings=4096, base=10000):

        super().__init__()
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (
            base ** (paddle.cast(paddle.arange(0, dim, 2), dtype="float32") / dim)
        )

        t = paddle.arange(max_position_embeddings, dtype="float32")
        freqs = paddle.einsum("i,j->ij", t, inv_freq.cast("float32"))
        emb = paddle.concat([freqs, freqs], axis=-1)

        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

        self._cast_to_low_precision = False
        self._cast_to_low_precison = False

    def forward(self, x, seq_len=None):

        return (
            self.cos_cached[:seq_len, :],
            self.sin_cached[:seq_len, :],
        )

    @classmethod
    def rotate_half(cls, x):

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return paddle.concat([-x2, x1], axis=-1)

    @classmethod
    def apply_rotary_pos_emb(cls, q, k, cos, sin, offset: int = 0, position_ids=None):
        if position_ids is not None:
            assert offset == 0, offset
            cos = F.embedding(position_ids, cos)
            sin = F.embedding(position_ids, sin)
        else:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        cos = cos[:, offset : q.shape[1] + offset, None, :]
        sin = sin[:, offset : q.shape[1] + offset, None, :]

        q_embed = paddle.add(
            paddle.multiply(q, cos), paddle.multiply(cls.rotate_half(q), sin)
        )
        k_embed = paddle.add(
            paddle.multiply(k, cos), paddle.multiply(cls.rotate_half(k), sin)
        )
        q_embed = q_embed.astype(q.dtype)
        k_embed = k_embed.astype(k.dtype)
        return q_embed, k_embed


class RopeEmbeddingLegacy(nn.Layer):

    def __init__(self, head_dim, compression_ratio=1.0, base=10000):
        super().__init__()
        self.head_dim = head_dim
        self.compression_ratio = compression_ratio
        self.base = base

    def forward(self, seq_length, position_ids=None):

        indices = paddle.arange(0, self.head_dim, 2, dtype="float32")
        indices = 1 / self.base ** (indices / self.head_dim)
        if position_ids is None:
            position_ids = paddle.arange(0, seq_length, 1, dtype="float32").unsqueeze(1)
            position_ids = position_ids / self.compression_ratio
            sinusoid_inp = position_ids * indices.unsqueeze(0)
        else:
            position_ids = position_ids / self.compression_ratio
            seq_length = position_ids.shape[-1]
            sinusoid_inp = position_ids.unsqueeze(-1).astype(
                "float32"
            ) * indices.unsqueeze(0)
        pos_emb = paddle.concat(
            [paddle.sin(sinusoid_inp), paddle.cos(sinusoid_inp)], axis=-1
        )
        pos_emb = paddle.reshape(pos_emb, (-1, 1, seq_length, self.head_dim))
        pos_emb.stop_gradient = True
        return pos_emb

    def apply_rotary(self, rp, q, k):

        sin, cos = paddle.chunk(rp, 2, axis=-1)
        sin_pos = paddle.reshape(paddle.stack([sin, sin], axis=-1), rp.shape)
        cos_pos = paddle.reshape(paddle.stack([cos, cos], axis=-1), rp.shape)
        rotate_half_q = paddle.reshape(
            paddle.stack([-q[:, :, :, 1::2], q[:, :, :, 0::2]], axis=-1),
            paddle.shape(q),
        )
        query = paddle.add(
            paddle.multiply(q.astype("float32"), cos_pos),
            paddle.multiply(rotate_half_q.astype("float32"), sin_pos),
        )
        rotate_half_k = paddle.reshape(
            paddle.stack([-k[:, :, :, 1::2], k[:, :, :, 0::2]], axis=-1),
            paddle.shape(k),
        )
        key = paddle.add(
            paddle.multiply(k.astype("float32"), cos_pos),
            paddle.multiply(rotate_half_k.astype("float32"), sin_pos),
        )
        return query, key

    def forward_single(self, position_ids):

        batch_size, seq_length = position_ids.shape[:2]
        rope_emb = paddle.zeros(
            (2, batch_size, seq_length, 1, self.head_dim), dtype="float32"
        )
        inv_freq = self.base ** (
            -paddle.arange(0, self.head_dim, 2, dtype="float32") / self.head_dim
        )
        position_ids = position_ids.cast("float32")
        position_ids = position_ids / self.compression_ratio
        freqs = paddle.einsum("ij,k->ijk", position_ids.cast("float32"), inv_freq)
        emb = paddle.stack([freqs, freqs], axis=-1).reshape(
            (batch_size, seq_length, self.head_dim)
        )
        emb = paddle.unsqueeze(emb, 2)

        rope_emb[0] = paddle.cos(emb)
        rope_emb[1] = paddle.sin(emb)
        return rope_emb

    @staticmethod
    def apply_rotary_single(x, rope_emb):

        rotate_half_x = paddle.reshape(
            paddle.stack([-x[:, :, :, 1::2], x[:, :, :, 0::2]], axis=-1),
            paddle.shape(x),
        )
        return x * rope_emb[0] + rotate_half_x * rope_emb[1]


class ErnieLinear(nn.Layer):

    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        bias_attr=None,
        name=None,
        ipp=0,
    ):
        super(ErnieLinear, self).__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )
        self.bias = self.create_parameter(
            shape=[out_features],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )
        self.name = name
        self.ipp = ipp

    def forward(self, input):

        out = F.linear(x=input, weight=self.weight, bias=None, name=self.name)
        out = dist.reshard(
            out,
            get_mesh(self.ipp),
            [dist.Shard(1), dist.Shard(0)],
        )
        if self.bias:
            out += self.bias
        return out


class ErnieMLP(nn.Layer):

    def __init__(self, config, ipp=None, do_shard_tensor=True):
        super().__init__()
        self.config = config
        self.ipp = ipp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        LinearFN = nn.Linear
        self.gate_proj = LinearFN(
            self.hidden_size, self.intermediate_size, bias_attr=config.use_bias
        )
        self.up_proj = LinearFN(
            self.hidden_size, self.intermediate_size, bias_attr=config.use_bias
        )

        if config.sequence_parallel:
            self.down_proj = ErnieLinear(
                self.intermediate_size,
                self.hidden_size,
                bias_attr=config.use_bias,
                ipp=self.ipp,
            )
        else:
            self.down_proj = LinearFN(
                self.intermediate_size, self.hidden_size, bias_attr=config.use_bias
            )

        if do_shard_tensor and (
            self.config.tensor_parallel_degree > 1
            or self.config.pipeline_parallel_degree > 1
        ):
            self.gate_proj.weight = dist.shard_tensor(
                self.gate_proj.weight,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(1)],
            )
            self.up_proj.weight = dist.shard_tensor(
                self.up_proj.weight,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(1)],
            )
            if config.use_bias:
                self.gate_proj.bias = dist.shard_tensor(
                    self.gate_proj.bias,
                    get_mesh(self.ipp),
                    [dist.Replicate(), dist.Shard(0)],
                )
                self.up_proj.bias = dist.shard_tensor(
                    self.up_proj.bias,
                    get_mesh(self.ipp),
                    [dist.Replicate(), dist.Shard(0)],
                )
            self.down_proj.weight = dist.shard_tensor(
                self.down_proj.weight,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(0)],
            )
            if config.use_bias:
                self.down_proj.bias = dist.shard_tensor(
                    self.down_proj.bias,
                    get_mesh(self.ipp),
                    [dist.Replicate(), dist.Replicate()],
                )

        self.fuse_swiglu = config.fuse_swiglu
        if self.fuse_swiglu:
            assert fused_swiglu is not None, "fused_swiglu operator is not found."

    def forward(self, x):

        if self.fuse_swiglu:
            x = fused_swiglu(self.gate_proj(x), self.up_proj(x))
        else:
            x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(x)


class ErnieAttentionAuto(nn.Layer):

    def __init__(self, config, ipp: Optional[int] = None):
        super().__init__()
        self.ipp = ipp
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.use_recompute_attn = config.use_recompute_attn
        self.is_gqa = (
            config.num_key_value_heads is not None
            and config.num_key_value_heads != self.num_heads
        )
        if config.fuse_rope:
            assert fused_rope is not None, "fused_rope is not supported"
        self.fuse_rope = config.fuse_rope

        if self.is_gqa:
            logger.info(
                f"use GQA - num_heads: {self.num_heads}- num_key_value_heads: {self.num_key_value_heads}"
            )
            assert (
                self.num_heads % self.num_key_value_heads == 0
            ), f"num_heads: {self.num_heads}, num_key_value_heads: {self.num_key_value_heads}"
            kv_hidden_size = (
                self.hidden_size // self.num_heads * self.num_key_value_heads
            )

        LinearFN = nn.Linear
        self.q_proj = LinearFN(
            self.hidden_size,
            self.hidden_size,
            bias_attr=config.use_bias,
        )
        self.k_proj = LinearFN(
            self.hidden_size,
            self.hidden_size if not self.is_gqa else kv_hidden_size,
            bias_attr=config.use_bias,
        )
        self.v_proj = LinearFN(
            self.hidden_size,
            self.hidden_size if not self.is_gqa else kv_hidden_size,
            bias_attr=config.use_bias,
        )

        if config.sequence_parallel:
            self.o_proj = ErnieLinear(
                self.hidden_size,
                self.hidden_size,
                bias_attr=config.use_bias,
                ipp=self.ipp,
            )
        else:
            self.o_proj = LinearFN(
                self.hidden_size,
                self.hidden_size,
                bias_attr=config.use_bias,
            )
        if config.rope_reorder:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        else:
            self.rotary_emb = RopeEmbeddingLegacy(
                self.head_dim,
                compression_ratio=config.compression_ratio,
                base=config.rope_theta,
            )

        self.config = config

        if (
            self.config.tensor_parallel_degree > 1
            or self.config.pipeline_parallel_degree > 1
        ):
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
            if config.use_bias:
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
            self.o_proj.weight = dist.shard_tensor(
                self.o_proj.weight,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(0)],
            )

    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        inbatch_pack_offset: Optional[Tuple[paddle.Tensor]] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if self.config.sequence_parallel:
            hidden_states = dist.reshard(
                hidden_states, get_mesh(self.ipp), [dist.Shard(1), dist.Replicate()]
            )

        query_states = self.q_proj(hidden_states).reshape(
            shape=[0, 0, self.num_heads, self.head_dim]
        )
        key_states = self.k_proj(hidden_states).reshape(
            shape=[
                0,
                0,
                self.num_key_value_heads if self.is_gqa else self.num_heads,
                self.head_dim,
            ]
        )
        value_states = self.v_proj(hidden_states).reshape(
            shape=[
                0,
                0,
                self.num_key_value_heads if self.is_gqa else self.num_heads,
                self.head_dim,
            ]
        )

        if self.config.sequence_parallel:
            query_states = paddle.transpose(query_states, [1, 0, 2, 3])
            key_states = paddle.transpose(key_states, [1, 0, 2, 3])
            value_states = paddle.transpose(value_states, [1, 0, 2, 3])

        if self.use_recompute_attn:
            assert past_key_value is None, "do not use kv cache in recompute"
            assert not use_cache
            attn_output, attn_weights, past_key_value = recompute(
                self.rope_attn,
                None,
                query_states,
                key_states,
                value_states,
                attention_mask,
                position_ids,
                output_attentions,
                past_key_value,
                use_cache,
                inbatch_pack_offset,
                use_reentrant=False,
            )
        else:
            attn_output, attn_weights, past_key_value = self.rope_attn(
                mix_layer=None,
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                past_key_value=past_key_value,
                use_cache=use_cache,
                inbatch_pack_offset=inbatch_pack_offset,
            )

        if self.config.sequence_parallel:
            attn_output = paddle.transpose(attn_output, [1, 0, 2])

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def rope_attn(
        self,
        mix_layer,
        query_states,
        key_states,
        value_states,
        attention_mask,
        position_ids,
        output_attentions=False,
        past_key_value=None,
        use_cache=False,
        inbatch_pack_offset=None,
    ):
        if mix_layer is not None:
            query_states, key_states, value_states = paddle.split(mix_layer, 3, axis=-1)
        query_states_dtype = query_states.dtype

        kv_seq_len = key_states.shape[-3]
        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].shape[-3]
            kv_seq_len += offset

        if self.config.rope_reorder:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

            query_states, key_states = self.rotary_emb.apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                position_ids=position_ids,
                offset=offset if position_ids is None else 0,
            )
        else:
            if offset > 0 or position_ids is not None or not self.fuse_rope:
                cos_sin = self.rotary_emb(kv_seq_len, position_ids).transpose(
                    [0, 2, 1, 3]
                )
                if offset > 0 and position_ids is None:
                    cos_sin = cos_sin[:, offset:]
                query_states, key_states = self.rotary_emb.apply_rotary(
                    cos_sin, query_states, key_states
                )
            else:
                bsz, q_len, num_heads, head_dim = query_states.shape
                _, kv_seq_len, num_key_value_heads, _ = key_states.shape
                if num_heads != num_key_value_heads:
                    query_states, _, _ = fused_rope(query_states, None, None)
                    key_states, _, _ = fused_rope(key_states, None, None)
                else:
                    query_states, key_states, _ = fused_rope(
                        query_states, key_states, None
                    )

        if use_cache:
            query_states = query_states.astype(query_states_dtype)
            key_states = key_states.astype(query_states_dtype)
        if past_key_value is not None:
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)

        past_key_value = [key_states, value_states] if use_cache else None

        attn_output, attn_weights = scaled_dot_product_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            config=self.config,
            inbatch_pack_offset=inbatch_pack_offset,
            training=self.training,
        )
        return attn_output, attn_weights, past_key_value


class ErnieMoeMLP(ErnieMLP):
    """_summary_

    Args:
        ErnieMoeMLP (_type_): _description_
    """

    def __init__(self, config, ipp=0):
        """
        doc
        """
        disable_ffn_model_parallel = getattr(
            config, "disable_ffn_model_parallel", False
        )
        if disable_ffn_model_parallel:
            config = deepcopy(config)
            config.tensor_parallel_degree = 1
            config.sequence_parallel = False

        super().__init__(config, ipp, do_shard_tensor=not disable_ffn_model_parallel)
        self.moe_dropout_prob = config.moe_dropout_prob
        self.fuse_swiglu = config.fuse_swiglu
        if self.fuse_swiglu:
            assert fused_swiglu is not None, "fused_swiglu operator is not found."

    def redistribute_expert(self, mesh, placements):
        """
        Place the experts on different devices.
        """
        self.gate_proj.weight = dist.shard_tensor(
            self.gate_proj.weight, mesh, placements
        )
        self.up_proj.weight = dist.shard_tensor(self.up_proj.weight, mesh, placements)
        self.down_proj.weight = dist.shard_tensor(
            self.down_proj.weight, mesh, placements
        )
        if self.config.use_bias:
            self.gate_proj.bias = dist.shard_tensor(
                self.gate_proj.bias, mesh, placements
            )
            self.up_proj.bias = dist.shard_tensor(self.up_proj.bias, mesh, placements)
            self.down_proj.bias = dist.shard_tensor(
                self.down_proj.bias, mesh, placements
            )

    def forward(self, x):

        if self.fuse_swiglu:
            x = fused_swiglu(self.gate_proj(x), self.up_proj(x))
        else:
            x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        if self.moe_dropout_prob > 0:
            with get_rng_state_tracker().rng_state("local_seed"):
                x = F.dropout(x=x, p=self.moe_dropout_prob)
        ret = self.down_proj(x)
        return ret


class BMMLinear(nn.Layer):

    def __init__(self, experts, d_in, d_out, use_bias=False):
        super().__init__()
        self.weight = self.create_parameter(
            [experts, d_in, d_out], dtype=paddle.get_default_dtype()
        )
        if use_bias:
            self.bias = self.create_parameter(
                [experts, d_out], dtype=paddle.get_default_dtype(), is_bias=True
            )
        else:
            self.bias = None

    def forward(self, x):
        """x: [num_experts, Seq, dim]"""
        if self.bias is not None:
            return paddle.bmm(x, self.weight) + self.bias
        return paddle.bmm(x, self.weight)


class ErnieMoeMLPFused(nn.Layer):

    def __init__(self, config):

        assert (
            hasattr(config, "disable_ffn_model_parallel")
            or config.tensor_parallel_degree == 1
        ), f"fused mlp only suport mp-moe, mp={config.tensor_parallel_degree}"
        assert config.fuse_attn_ffn, "fused mlp only support fuse_attn_ffn"
        super().__init__()
        self.moe_dropout_prob = config.moe_dropout_prob
        self.num_local_experts = config.moe_num_experts // config.moe_world_size
        logger.info(
            f"fused-expert-weight-shape: {[self.num_local_experts, config.hidden_size, config.intermediate_size]}"
        )

        self.up_gate_proj = BMMLinear(
            self.num_local_experts, config.hidden_size, config.intermediate_size * 2
        )
        self.down_proj = BMMLinear(
            self.num_local_experts, config.intermediate_size, config.hidden_size
        )
        self.fuse_swiglu = config.fuse_swiglu
        if self.fuse_swiglu:
            assert fused_swiglu is not None, "fused_swiglu operator is not found."

    def __len__(self):
        return self.num_local_experts

    def __iter__(self):
        return (self for _ in range(1))

    def forward(self, x):
        if self.fuse_swiglu:
            x = fused_swiglu(self.up_gate_proj(x))
        else:
            gate, x = self.up_gate_proj(x).chunk(2, axis=-1)
            x = F.silu(gate) * x
        x = self.down_proj(x)
        return x


class ErnieDecoderLayerAuto(nn.Layer):
    """
    ErnieDecoderLayerAuto is a decoder layer in Ernie model.
    It is composed of self-attention, cross-attention and feedforward layers.
    """

    def __init__(self, config, layer_idx=0, ipp=0):
        """
            Initializes the ErnieBlock module.

        Args:
            config (ErnieConfig): The model configuration.
            layer_idx (int, optional): The index of this block in the model. Defaults to 0.
            ipp (int, optional): The index of this block in the pipeline parallelism. Defaults to 0.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.ipp = ipp
        self.hidden_size = config.hidden_size
        self.self_attn = ErnieAttentionAuto(config, ipp)
        self.use_moe = config.use_moe if hasattr(config, "use_moe") else False
        if self.use_moe:
            moe_layer_start_index = (
                min(config.moe_layer_start_index)
                if isinstance(config.moe_layer_start_index, (tuple, list))
                else config.moe_layer_start_index
            )
            moe_layer_end_index = (
                max(config.moe_layer_end_index)
                if isinstance(config.moe_layer_end_index, (tuple, list))
                else config.moe_layer_end_index
            )

        if (
            self.use_moe
            and ((layer_idx + 1) % config.moe_layer_interval == 0)
            and layer_idx >= moe_layer_start_index
            and layer_idx <= moe_layer_end_index
        ):
            self.create_moe_mlp_layer(layer_idx, ipp)
        else:
            self.mlp = ErnieMLP(config, ipp)
        Norm = RMSNorm if config.use_rmsnorm else LayerNorm
        self.input_layernorm = Norm(config, ipp)
        self.post_attention_layernorm = Norm(config, ipp)
        self.residual_add1 = FusedDropoutImpl(
            config.hidden_dropout_prob, mode="upscale_in_train"
        )
        self.residual_add2 = FusedDropoutImpl(
            config.hidden_dropout_prob, mode="upscale_in_train"
        )

    def create_moe_mlp_layer(self, layer_idx, ipp):
        _ex_cfg = deepcopy(self.config)
        fc_cls = ErnieMoeMLPFused if _ex_cfg.moe_fuse_experts else ErnieMoeMLP
        if _ex_cfg.moe_intermediate_size:
            if isinstance(_ex_cfg.moe_intermediate_size, (tuple, list)):
                assert isinstance(_ex_cfg.moe_num_experts, (tuple, list)) and len(
                    _ex_cfg.moe_num_experts
                ) == len(_ex_cfg.moe_intermediate_size)
                fc = []
                for _i, (num_experts, intermediate_size) in enumerate(
                    zip(_ex_cfg.moe_num_experts, _ex_cfg.moe_intermediate_size)
                ):
                    _ex_cfg_real = deepcopy(_ex_cfg)
                    _ex_cfg_real.intermediate_size = intermediate_size
                    cur_modality_start_layer_idx = (
                        self.config.moe_layer_start_index[_i]
                        if isinstance(self.config.moe_layer_start_index, (tuple, list))
                        else self.config.moe_layer_start_index
                    )
                    cur_modality_end_layer_idx = (
                        self.config.moe_layer_end_index[_i]
                        if isinstance(self.config.moe_layer_end_index, (tuple, list))
                        else self.config.moe_layer_end_index
                    )
                    if (
                        layer_idx >= cur_modality_start_layer_idx
                        and layer_idx <= cur_modality_end_layer_idx
                    ):
                        if _i == 1:
                            with paddle.utils.unique_name.guard(
                                f"mm_expert_{layer_idx}_"
                            ):
                                fc.append((num_experts, fc_cls(_ex_cfg_real)))
                        else:
                            fc.append((num_experts, fc_cls(_ex_cfg_real)))
                    else:
                        logger.info(
                            f"moe multimodal experts use Identity layer_idx: {layer_idx}"
                        )
                        fc.append((num_experts, nn.Identity()))
            else:
                _ex_cfg.intermediate_size = _ex_cfg.moe_intermediate_size
                fc = [(_ex_cfg.moe_num_experts, fc_cls(_ex_cfg))]
        else:
            fc = [(_ex_cfg.moe_num_experts, fc_cls(_ex_cfg))]
        gate, experts, lm_gate, lm_experts = get_gate(
            self.config, fc, layer_idx, self.ipp
        )
        _sh_cfg = deepcopy(self.config)

        if _sh_cfg.moe_num_shared_experts > 0:
            if _sh_cfg.moe_intermediate_size:
                _sh_inter_size = (
                    _sh_cfg.moe_intermediate_size[0]
                    if isinstance(_sh_cfg.moe_intermediate_size, (tuple, list))
                    else _sh_cfg.moe_intermediate_size
                )
                _sh_cfg.intermediate_size = (
                    _sh_inter_size * _sh_cfg.moe_num_shared_experts
                )
            else:
                _sh_cfg.intermediate_size = (
                    _sh_cfg.intermediate_size * _sh_cfg.moe_num_shared_experts
                )
            _sh_cfg.disable_ffn_model_parallel = False
            shared_experts = ErnieMoeMLP(_sh_cfg, ipp)
        else:
            shared_experts = None

        is_moe_infer = self.config.get("is_moe_infer", False)
        if is_moe_infer:
            raise NotImplementedError
        elif self.config.moe_use_size_all2all:
            raise NotImplementedError
        else:
            logger.info(f"moe-logging:{self.config.moe_logging}")
            moe_cls = MOELayerAuto
            self.mlp = moe_cls(
                gate,
                experts,
                layer_idx=layer_idx,
                shared_experts=shared_experts,
                group=self.config.moe_group,
                recompute=self.config.use_recompute_moe,
                k=self.config.moe_k,
                enable_pbr=self.config.moe_use_bpr,
                all_to_all_dropout=self.config.moe_all_to_all_dropout,
                group_experts=self.config.moe_group_experts,
                config=self.config,
                ipp=self.ipp,
            )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        inbatch_pack_offset: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        output_gate_logits=True,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `cache` key value states are returned and can be used to speed up decoding
                (see `cache`).
            cache (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        (hidden_states, self_attn_weights, present_key_value, *router_loss_attn) = (
            self.self_attn(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=use_cache,
                inbatch_pack_offset=inbatch_pack_offset,
            )
        )

        if (
            self.config.tensor_parallel_degree > 1
            and self.config.hidden_dropout_prob > 0.0
        ):
            current_seed = (
                "local_seed" if self.config.sequence_parallel else "global_seed"
            )
            with get_rng_state_tracker().rng_state(current_seed):
                hidden_states = self.residual_add1(hidden_states, residual)
        else:
            hidden_states = self.residual_add1(hidden_states, residual)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if isinstance(
            self.mlp,
            (MOELayerAuto),
        ):

            hidden_states, _, router_loss, gate_logits = self.mlp(
                hidden_states, token_type_ids
            )
        else:
            if self.config.sequence_parallel:
                hidden_states = dist.reshard(
                    hidden_states,
                    get_mesh(self.ipp),
                    [dist.Shard(1), dist.Replicate()],
                )
            hidden_states = self.mlp(hidden_states)
            gate_logits = None

        if (
            self.config.tensor_parallel_degree > 1
            and self.config.hidden_dropout_prob > 0.0
        ):
            current_seed = (
                "local_seed" if self.config.sequence_parallel else "global_seed"
            )
            with get_rng_state_tracker().rng_state(current_seed):
                hidden_states = self.residual_add2(hidden_states, residual)
        else:
            hidden_states = self.residual_add2(hidden_states, residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if hasattr(self.config, "use_moe") and self.config.use_moe:
            if router_loss_attn:
                router_loss_attn = router_loss_attn[0]
                router_loss = router_loss + router_loss_attn

            if isinstance(self.mlp, (MOELayerAuto)):
                outputs += (router_loss,)
            else:
                outputs += (paddle.zeros([1], dtype=paddle.float32),)

            if output_gate_logits:
                outputs += (gate_logits,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]
        return outputs


class ErniePretrainedModelAuto(PretrainedModel):
    """
    ErniePretrainedModelAuto is a pretrained model class for Ernie model.
    It is composed of a encoder and a decoder.
    """

    config_class = ErnieMoEConfig
    base_model_prefix = "ernie"

    def init_weights(self, layer):
        """Initialization hook"""
        if self.config.tensor_parallel_degree > 1:
            rng_tracker = get_rng_state_tracker().rng_state
        else:
            rng_tracker = contextlib.nullcontext

        if isinstance(
            layer,
            (
                ErnieLMHead,
                nn.Embedding,
                nn.Linear,
                paddle.incubate.nn.FusedLinear,
            ),
        ):

            with rng_tracker():
                dtype = paddle.get_default_dtype()
                paddle.set_default_dtype("float32")
                if layer.weight._is_initialized():
                    if layer.weight.is_dist():
                        layer.weight._local_value().set_value(
                            paddle.randn(
                                layer.weight._local_shape, dtype=layer.weight.dtype
                            ).scale(self.config.initializer_range)
                        )
                    else:
                        layer.weight.set_value(
                            paddle.randn(
                                layer.weight.shape, dtype=layer.weight.dtype
                            ).scale(self.config.initializer_range)
                        )
                    paddle.set_default_dtype(dtype)
                    logger.info(
                        f"dist-init-fc: shape={layer.weight.shape}, "
                        f" range={self.config.initializer_range},"
                        f' type={type(layer)},norm={layer.weight.astype("float32").norm()}'
                    )

        elif isinstance(layer, Top2Gate):
            if not hasattr(layer, "weight"):
                return
            with rng_tracker("model_parallel_rng"):
                dtype = paddle.get_default_dtype()
                paddle.set_default_dtype("float32")
                if self.config.moe_group_experts:
                    if layer.weight._is_initialized():
                        layer.weight.set_value(
                            paddle.randn(
                                layer.weight.shape, dtype=layer.weight.dtype
                            ).scale(self.config.initializer_range)
                        )
                else:
                    if layer.weight._is_initialized():
                        granularity = (
                            1
                            if self.config.moe_intermediate_size == 0
                            else self.config.intermediate_size
                            // self.config.moe_intermediate_size
                        )
                        layer.weight.set_value(
                            paddle.randn(
                                [
                                    self.config.hidden_size,
                                    self.config.moe_num_experts // granularity,
                                ],
                                dtype="float32",
                            )
                            .scale(self.config.initializer_range)
                            .repeat_interleave(granularity, axis=-1)
                        )
                logger.info(
                    f"dist-init-moe_gate: shape={layer.weight.shape}, dtype={layer.weight.dtype} "
                    f"range={self.config.initializer_range},type={type(layer)}, "
                    f'norm={layer.weight.astype("float32").norm()}'
                )


@register_base_model
class ErnieModelAuto(ErniePretrainedModelAuto):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ErnieDecoderLayerAuto`]
    Args:
        config: ErnieMoEConfig
    """

    def __init__(self, config: ErnieMoEConfig):
        if hasattr(config, "use_moe") and config.use_moe:
            if config.moe_group in {"mp", "model", "tp", "mpdp"}:
                assert config.sequence_parallel
                logger.info(
                    f"disable FFN tensor model parallel, moe-group={config.moe_group}"
                )
                config.disable_ffn_model_parallel = True

            config.moe_group = _parse_moe_group(config.moe_group)
            if config.moe_group in fleet.auto.get_mesh().dim_names:
                config.moe_world_size = fleet.auto.get_mesh().get_dim_size(
                    config.moe_group
                )
                if config.moe_world_size < 0:
                    config.moe_world_size = 1
            else:
                config.moe_world_size = 1

        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.config = config

        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
        )

        if (
            self.config.tensor_parallel_degree > 1
            or self.config.pipeline_parallel_degree > 1
        ):
            if not in_auto_parallel_align_mode():
                self.embed_tokens.weight = dist.shard_tensor(
                    self.embed_tokens.weight,
                    get_mesh(),
                    [dist.Replicate(), dist.Shard(1)],
                )

        layers_list = []

        def get_layer_pp_info(ipp):
            mesh = fleet.auto.get_mesh()
            if is_pp_enable() is False:
                return None, False
            else:
                pp_degree = mesh.get_dim_size("pp")
                layer_num = (
                    config.num_hidden_layers - 1
                    if config.remove_tail_layer
                    else config.num_hidden_layers
                )
                layer_per_stage = math.ceil(layer_num / pp_degree)
                input_need_reshard = ipp % layer_per_stage == 0
                return ipp // layer_per_stage, input_need_reshard

        self.next_pp_stage_indexes = []
        for layer_idx in range(
            config.num_hidden_layers - 1
            if config.remove_tail_layer
            else config.num_hidden_layers
        ):
            pp_stage_id, input_need_reshard = get_layer_pp_info(layer_idx)
            layers_list.append(ErnieDecoderLayerAuto(config, layer_idx, pp_stage_id))
            if input_need_reshard:
                self.next_pp_stage_indexes.append(layer_idx)
        self.layers = nn.LayerList(layers_list)
        Norm = RMSNorm if config.use_rmsnorm else LayerNorm

        self.norm = Norm(config, -1)

        self.gradient_checkpointing = False

        self.placements = (
            [dist.Shard(1), dist.Shard(0)]
            if self.config.sequence_parallel
            else [dist.Shard(0), dist.Replicate()]
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @classmethod
    def _prepare_decoder_attention_mask(
        cls, attention_mask, input_shape, past_key_values_length, dtype
    ):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, past_key_values_length=past_key_values_length, dtype=dtype
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask, dtype, tgt_length=input_shape[-1]
            )
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )
        combined_attention_mask = paddle.maximum(
            combined_attention_mask.astype(dtype),
            paddle.to_tensor(float(finfo(dtype).min), dtype=dtype),
        )
        return combined_attention_mask

    def recompute_training(
        self,
        layer_module,
        hidden_states,
        attention_mask,
        position_ids,
        output_attentions,
        past_key_value,
        use_cache,
        inbatch_pack_offset,
        token_type_ids,
    ):

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, output_gate_logits=False)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            position_ids,
            output_attentions,
            past_key_value,
            use_cache,
            inbatch_pack_offset,
            token_type_ids,
            use_reentrant=False,
        )
        return hidden_states

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        inbatch_pack_offset=None,
        token_type_ids=None,
        **kwargs,
    ):
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        seq_length_with_past = seq_length
        cache_length = 0

        if past_key_values[0] is not None:
            cache_length = paddle.shape(past_key_values[0][0])[1]
            seq_length_with_past += cache_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).astype(
                self.embed_tokens.weight.dtype
            )

        global_mesh = global_mesh_starts_with_pp()
        if self.config.sequence_parallel:
            inputs_embeds = paddle.transpose(inputs_embeds, [1, 0, 2])

        if position_ids is not None:
            position_ids = dist.shard_tensor(
                position_ids,
                global_mesh,
                [dist.Replicate() for _ in range(len(global_mesh._shape))],
            )
        can_use_fa = self.config.use_flash_attn
        can_mem_eff_attn = (
            self.config.use_mem_eff_attn and inbatch_pack_offset is not None
        )
        if can_use_fa or can_mem_eff_attn:
            if attention_mask is not None:
                attention_mask = None

        elif attention_mask is None:
            attention_mask = paddle.ones(
                (batch_size, seq_length_with_past), dtype=paddle.bool
            )

        if attention_mask is not None:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                cache_length,
                inputs_embeds.dtype,
            )
            attention_mask = dist.shard_tensor(
                attention_mask,
                global_mesh,
                [dist.Replicate() for _ in range(len(global_mesh._shape))],
            )

        hidden_states = inputs_embeds
        if self.config.tensor_parallel_degree > 1:
            hidden_states = dist.reshard(hidden_states, get_mesh(0), self.placements)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        all_router_loss = None
        if hasattr(self.config, "use_moe") and self.config.use_moe:
            all_router_loss = paddle.to_tensor(0.0)
            all_router_loss = dist.shard_tensor(
                all_router_loss, get_mesh(0), dist.Replicate()
            )
        all_gate_logits = () if hasattr(self.config, "use_moe") else None
        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            has_gradient = not hidden_states.stop_gradient
            ipp = decoder_layer.ipp
            if not is_pp_enable():
                position_ids_input = position_ids
                attention_mask_input = attention_mask
                token_type_ids_input = token_type_ids
            else:
                if position_ids is not None:
                    position_ids_input = dist.reshard(
                        position_ids,
                        get_mesh(ipp),
                        [dist.Replicate(), dist.Replicate()],
                    )
                else:
                    position_ids_input = position_ids
                attention_mask_input = (
                    dist.reshard(
                        attention_mask,
                        get_mesh(ipp),
                        [dist.Replicate(), dist.Replicate()],
                    )
                    if attention_mask is not None
                    else None
                )
                token_type_ids_input = (
                    dist.reshard(
                        token_type_ids,
                        get_mesh(ipp),
                        [dist.Replicate(), dist.Replicate()],
                    )
                    if token_type_ids is not None
                    else None
                )

            if idx in self.next_pp_stage_indexes:
                hidden_states = dist.reshard(
                    hidden_states,
                    get_mesh(ipp),
                    self.placements,
                )
                if hasattr(self.config, "use_moe") and self.config.use_moe:
                    all_router_loss = dist.reshard(
                        all_router_loss,
                        get_mesh(ipp),
                        [dist.Replicate()],
                    )
            if self.config.use_recompute and has_gradient:
                layer_outputs = self.recompute_training(
                    decoder_layer,
                    hidden_states,
                    attention_mask_input,
                    position_ids_input,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    inbatch_pack_offset,
                    token_type_ids_input,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask_input,
                    position_ids_input,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    inbatch_pack_offset,
                    token_type_ids_input,
                )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if hasattr(self.config, "use_moe") and self.config.use_moe:
                if not (self.config.use_recompute and has_gradient):
                    layer_outputs, gate_logits = layer_outputs[:-1], layer_outputs[-1]
                    all_gate_logits = all_gate_logits + (gate_logits,)
                router_loss = layer_outputs[-1]
                all_router_loss += router_loss

        if use_cache and not (hasattr(self.config, "use_moe") and self.config.use_moe):
            hidden_states = paddle.unsqueeze(hidden_states[:, -1, :], 1)

        if self.config.pipeline_parallel_degree > 1:
            hidden_states = dist.reshard(
                hidden_states,
                get_mesh(-1),
                self.placements,
            )
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_loss,
                    all_gate_logits,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
            router_loss=all_router_loss,
            gate_logits=all_gate_logits,
        )


class ErniePretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for Ernie.
    It calculates the final loss.
    """

    def __init__(self, config, return_tuple=True):
        super(ErniePretrainingCriterion, self).__init__()
        self.ignored_index = getattr(config, "ignored_index", -100)
        self.config = config
        self.return_tuple = return_tuple
        self.enable_parallel_cross_entropy = (
            config.tensor_parallel_degree > 1 and config.tensor_parallel_output
        )

        self.loss_func = paddle.nn.CrossEntropyLoss(
            reduction="none",
        )

    def forward(self, prediction_scores, masked_lm_labels, router_loss=None):
        """
        calculates the final loss
        """
        res = self.forward_impl(prediction_scores, masked_lm_labels)
        if self.return_tuple:
            loss, loss_sum = res
        else:
            loss, loss_sum = res, None
        if router_loss is not None and not in_auto_parallel_align_mode():
            global_mesh = global_mesh_starts_with_pp()
            if self.config.pipeline_parallel_degree > 1:
                loss = dist.reshard(
                    loss,
                    global_mesh,
                    [dist.Replicate() for _ in range(len(global_mesh._shape))],
                )
                router_loss = dist.reshard(
                    router_loss,
                    global_mesh,
                    [dist.Replicate() for _ in range(len(global_mesh._shape))],
                )
            loss = loss + router_loss - router_loss.detach()
        return loss, loss_sum

    def loss_impl(self, prediction_scores, masked_lm_labels):
        """extract loss impl for subbatch"""
        masked_lm_loss = self.loss_func(
            prediction_scores.astype("float32"), masked_lm_labels.unsqueeze(-1)
        )
        return masked_lm_loss

    def forward_impl(self, prediction_scores, masked_lm_labels):

        with paddle.amp.auto_cast(False):
            if self.config.use_sparse_head_and_loss_fn and prediction_scores.shape[
                0
            ] > self.config.get("loss_subbatch_seqlen", 32768):
                sb_loss_func = subbatch(
                    self.loss_impl,
                    [0, 1],
                    [0, 0],
                    self.config.get("loss_subbatch_seqlen", 32768),
                    0,
                )
                masked_lm_loss = sb_loss_func(prediction_scores, masked_lm_labels)
            else:
                masked_lm_loss = self.loss_impl(prediction_scores, masked_lm_labels)
            lossmask = masked_lm_labels != self.ignored_index

            if (~lossmask).all():
                logger.warning(
                    f"encounter empty span when calculate loss, ignored_index={self.ignored_index}"
                )
                loss = paddle.mean(masked_lm_loss) * 0.0
                loss_sum = masked_lm_loss.sum().detach()
            else:
                lossmask_ = lossmask.reshape([-1]).cast(paddle.float32)
                masked_lm_loss_ = paddle.sum(
                    masked_lm_loss.cast(paddle.float32).reshape([-1]) * lossmask_
                )
                loss = masked_lm_loss_ / lossmask_.sum()
                loss_sum = masked_lm_loss_.sum().detach()

        if not self.return_tuple:
            if self.training:
                return loss
            return loss_sum
        return loss, loss_sum


class ErnieLMHead(nn.Layer):
    """
    ErnieLMHead is the linear layer used to project hidden state of decoder into word embeddings.
    """

    def __init__(self, config):
        super(ErnieLMHead, self).__init__()
        self.config = config
        vocab_size = config.vocab_size
        self.weight = self.create_parameter(
            shape=(
                [vocab_size, config.hidden_size]
                if config.tie_word_embeddings
                else [config.hidden_size, vocab_size]
            ),
            dtype=paddle.get_default_dtype(),
        )

        if (
            self.config.tensor_parallel_degree > 1
            or self.config.pipeline_parallel_degree > 1
        ):
            self.weight = dist.shard_tensor(
                self.weight,
                get_mesh(-1),
                [dist.Replicate(), dist.Shard(1)],
            )

        logger.info(
            f"output-weight:{self.weight.shape} config.tie_word_embeddings={config.tie_word_embeddings}"
        )
        if config.weight_share_add_bias and config.use_bias:
            self.bias = self.create_parameter(
                shape=[vocab_size],
                dtype=paddle.get_default_dtype(),
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.constant.Constant(0.0)
                ),
            )
            if (
                self.config.tensor_parallel_degree > 1
                or self.config.pipeline_parallel_degree > 1
            ):
                self.bias = dist.shard_tensor(
                    self.bias,
                    get_mesh(-1),
                    [dist.Replicate(), dist.Shard(0)],
                )
        else:
            self.bias = None

        self.weight.is_distributed = (
            True if (vocab_size != config.vocab_size) else False
        )
        if config.weight_share_add_bias and config.use_bias:
            self.bias.is_distributed = (
                True if (vocab_size != config.vocab_size) else False
            )

        if self.weight.is_distributed:
            self.weight.split_axis = 1
        if (
            config.weight_share_add_bias
            and config.use_bias
            and self.bias.is_distributed
        ):
            self.bias.split_axis = 0

        if self.config.use_recompute_loss_fn:
            logger.info(
                "Using recompute_loss_fn, the calculation of logits will be moved into "
                "loss_fn for memory optimization"
            )

    def forward(self, hidden_states, tensor_parallel_output=None):

        if self.config.use_recompute_loss_fn or self.config.use_sparse_head_and_loss_fn:
            out_tensors = (
                (hidden_states, self.weight, self.bias)
                if tensor_parallel_output is None
                else (hidden_states, self.weight, self.bias, tensor_parallel_output)
            )

            return out_tensors

        return calc_lm_head_logits(
            self.config,
            hidden_states,
            self.weight,
            self.bias,
            None,
            tensor_parallel_output,
        )


class ErnieForCausalLMAuto(ErniePretrainedModelAuto):
    """
    ErnieForCausalLMAuto is the model class for causal language modeling.
    """

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)

        if config.sequence_parallel:
            logger.info(f"using sequence_parallel, input seqlen={config.seqlen}")
            if config.using_dynamic_sequence_length:
                assert (
                    not config.micro_batch_size
                ), "sequence-parallel needs micro_batch_size setting when using dygramic_sequnence_length"
            else:
                assert config.seqlen is not None

            assert (
                config.tensor_parallel_degree > 1
            ), f"sequence-parallel needs mp>1, got mp={config.tensor_parallel_degree}"

        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(
            f"change initializer-range from {config.initializer_range} to {new_initializer_range}"
        )
        config.initializer_range = new_initializer_range
        self.config = config
        self.ernie = ErnieModelAuto(config)
        self.lm_head = ErnieLMHead(config)
        self.criterion = ErniePretrainingCriterion(config)

        self.tie_weights()

        if self.config.use_rmsnorm:
            if self.config.fuse_rms_norm:
                logger.info("Use fusedRMSNorm")
            else:
                logger.info("Use normal RMSNorm")
        else:
            logger.info("Use normal LayerNorm")

    def _post_init(self, original_init, *args, **kwargs):
        """
        Initialize weights and apply final processing
        """
        super()._post_init(self, original_init, *args, **kwargs)
        factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
        logger.info(f"using post init div: factor:{factor}")

        def scale_by_factor_if_valid(w):
            if w.is_dist() and w._is_initialized():
                w.scale_(factor)

        if hasattr(self.config, "use_moe") and self.config.use_moe:
            with paddle.no_grad():
                for left in self.ernie.layers:
                    if isinstance(
                        left.self_attn.o_proj,
                        (MOELayerAuto),
                    ):
                        for e in left.self_attn.o_proj.experts:
                            if isinstance(e, ErnieMoeMLP):
                                scale_by_factor_if_valid(e.weight)
                    else:
                        scale_by_factor_if_valid(left.self_attn.o_proj.weight)

                    if isinstance(
                        left.mlp,
                        (MOELayerAuto),
                    ):
                        for e in left.mlp.experts:
                            if isinstance(e, ErnieMoeMLP):
                                scale_by_factor_if_valid(e.down_proj.weight)
                    else:
                        scale_by_factor_if_valid(left.mlp.down_proj.weight)
        else:
            with paddle.no_grad():
                for left in self.ernie.layers:
                    scale_by_factor_if_valid(left.self_attn.o_proj.weight)
                    scale_by_factor_if_valid(left.mlp.down_proj.weight)

    def get_input_embeddings(self):

        return self.ernie.embed_tokens

    def set_input_embeddings(self, value):

        self.ernie.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.ernie = decoder

    def get_decoder(self):
        return self.ernie

    def prepare_inputs_for_generation(
        self,
        input_ids,
        use_cache=False,
        past_key_values=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        attention_mask = kwargs.get("attention_mask", None)

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
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids,
        labels=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
        ignored_index=0,
        inbatch_pack_offset=None,
        token_type_ids=None,
    ):
        if isinstance(input_ids, list):
            input_ids, labels = input_ids[:2]

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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.ernie(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            inbatch_pack_offset=inbatch_pack_offset,
            token_type_ids=token_type_ids,
        )

        hidden_states = outputs.last_hidden_state

        logits = self.lm_head(
            hidden_states,
        )

        if return_dict:
            if labels is not None:
                loss, _ = self.criterion(logits, labels)
            else:
                loss = None
            return CausalLMOutputWithCrossAttentionsAuto(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_loss=outputs.router_loss if self.config.use_moe else None,
            )

        assert labels is not None
        router_loss = (
            outputs.router_loss
            if hasattr(self.config, "use_moe") and self.config.use_moe
            else None
        )
        return self.criterion(logits, labels, router_loss)
