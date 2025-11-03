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
"""Paddle Ernie model"""
import math
import logging
from typing import Optional, Tuple


from copy import deepcopy
from dataclasses import dataclass
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.utils import recompute
from paddle.incubate.nn.layer.fused_dropout_add import FusedDropoutAdd
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker

from paddle.distributed.auto_parallel.intermediate.tensor_parallel import (
    PrepareLayerInput,
)
from paddleformers.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions as _BaseModelOutput,
)
from paddleformers.transformers.model_outputs import CausalLMOutputWithCrossAttentions

from paddleformers.transformers.model_utils import PretrainedModel

from models.configuration import ErnieMoEConfig


from paddle.nn.functional.flash_attention import flash_attention
from paddle.incubate.nn.functional import fused_rotary_position_embedding as fused_rope


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(_BaseModelOutput):
    router_loss: Optional[paddle.Tensor] = None
    gate_logits: Optional[Tuple[paddle.Tensor]] = None
    mtp_outputs: Optional[paddle.Tensor] = None


@dataclass
class CausalLMOutputWithCrossAttentionsErnie(CausalLMOutputWithCrossAttentions):
    router_loss: Optional[paddle.Tensor] = None


logger = logging.getLogger(__name__)


__all__ = [
    "ErnieForCausalLM",
]


def calc_lm_head_logits(
    config,
    hidden_states,
    weight,
    bias,
    sparse_label_idx=None,
):
    """the core function to calc lm head"""

    logits = paddle.matmul(
        hidden_states, weight, transpose_y=config.tie_word_embeddings
    )
    if bias is not None:
        logits += bias

    return logits


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def scaled_dot_product_attention(
    query_states,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    config,
    is_causal=True,
    inbatch_pack_offset=None,
    training=True,
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, num_key_value_heads, _ = value_states.shape

    can_use_fa = config.use_flash_attn

    if can_use_fa:
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
    else:
        if query_states.shape[-2] != key_states.shape[-2]:
            key_states = key_states.repeat_interleave(
                num_heads // num_key_value_heads, axis=-2
            )
        if query_states.shape[-2] != value_states.shape[-2]:
            value_states = value_states.repeat_interleave(
                num_heads // num_key_value_heads, axis=-2
            )
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
            attention_mask = F.get_triangle_upper_mask(attn_weights)

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
                    float(paddle.finfo(query_states.dtype).min),
                    dtype=query_states.dtype,
                ),
            )

            with paddle.amp.auto_cast(False):
                attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(
                    query_states.dtype
                )

        else:
            attn_weights = attn_weights.cast(paddle.float32)
            attention_mask = attention_mask.cast(paddle.float32)
            attn_weights = attn_weights.add_(attention_mask)
            attn_weights = F.softmax_(attn_weights, axis=-1).astype(query_states.dtype)

        if config.attention_probs_dropout_prob > 0.0:
            with get_rng_state_tracker().rng_state("local_seed"):
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

    mask = paddle.full((target_length, target_length), float(paddle.finfo(dtype).min))

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
        inverted_mask, inverted_mask.cast("bool"), float(paddle.finfo(dtype).min)
    )


class RMSNorm(nn.Layer):
    def __init__(self, config):
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
        with paddle.amp.auto_cast(False):
            variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
            hidden_states = (
                paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
            )

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return hidden_states * self.weight


class LayerNorm(nn.LayerNorm):

    def __init__(self, config):
        super().__init__(config.hidden_size, epsilon=config.rms_norm_eps)


class ErnieMLP(nn.Layer):
    def __init__(self, config, ipp=None, do_shard_tensor=True):
        super().__init__()
        self.config = config
        self.ipp = ipp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias_attr=config.use_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias_attr=config.use_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias_attr=config.use_bias
        )

        self.fuse_swiglu = config.fuse_swiglu

    def forward(self, x):
        from paddle.incubate.nn.functional import swiglu

        if self.fuse_swiglu:
            x = swiglu(self.gate_proj(x), self.up_proj(x))
        else:
            x = F.silu(self.gate_proj(x)) * self.up_proj(x)

        out = self.down_proj(x)
        return out


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


class RopeEmbedding(nn.Layer):
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


class ErnieAttention(nn.Layer):
    def __init__(self, config, ipp: Optional[int] = None):
        super().__init__()
        self.ipp = ipp
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.is_gqa = (
            config.num_key_value_heads is not None
            and config.num_key_value_heads != self.num_heads
        )
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

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias_attr=config.use_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size if not self.is_gqa else kv_hidden_size,
            bias_attr=config.use_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size if not self.is_gqa else kv_hidden_size,
            bias_attr=config.use_bias,
        )
        self.o_proj = nn.Linear(
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
            self.rotary_emb = RopeEmbedding(
                self.head_dim,
                compression_ratio=config.compression_ratio,
                base=config.rope_theta,
            )

        self.config = config
        self.reshard_row_and_col = ReshardLayer()

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

        attn_output = self.o_proj(attn_output)
        attn_output = self.reshard_row_and_col(attn_output)

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


class ErnieDecoderLayer(nn.Layer):
    """
    ErnieDecoderLayer is a decoder layer in Ernie model.
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
        self.self_attn = ErnieAttention(config, ipp)
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
            pp_degree = config.pipeline_parallel_degree
            chunk_size = (
                config.num_hidden_layers // pp_degree // config.virtual_pp_degree
            )
            stage_id = (layer_idx // chunk_size) % pp_degree
            self.create_moe_mlp_layer(layer_idx, stage_id)
        else:
            self.mlp = ErnieMLP(config, ipp)
        Norm = RMSNorm if config.use_rmsnorm else LayerNorm
        self.input_layernorm = Norm(config)
        self.post_attention_layernorm = Norm(config)
        self.residual_add1 = FusedDropoutAdd(
            config.hidden_dropout_prob, mode="upscale_in_train"
        )
        self.residual_add2 = FusedDropoutAdd(
            config.hidden_dropout_prob, mode="upscale_in_train"
        )
        self.reshard_col = ReshardLayer()
        self.reshard_replicate = ReshardLayer()

    def create_moe_mlp_layer(self, layer_idx, ipp):
        _ex_cfg = deepcopy(self.config)
        from .moe_layer import ErnieMoeMLP, MOELayer, ErnieMoeMLPFused, get_gate

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
        gate, experts, lm_gate, lm_experts, moe_statics = get_gate(
            self.config, fc, layer_idx, ipp
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

        logger.info(f"moe-logging:{self.config.moe_logging}")
        self.mlp = MOELayer(
            gate,
            experts,
            layer_idx,
            shared_experts,
            self.config.moe_group,
            self.config.use_recompute_moe,
            self.config.moe_k,
            self.config.moe_all_to_all_dropout,
            self.config.moe_group_experts,
            moe_statics,
            self.config,
            ipp,
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
                hidden_states,
                past_key_value,
                attention_mask,
                position_ids,
                output_attentions,
                use_cache,
                inbatch_pack_offset,
            )
        )

        with get_rng_state_tracker().rng_state("local_seed"):
            hidden_states = self.residual_add1(hidden_states, residual)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        from .moe_layer import MOELayer

        if isinstance(
            self.mlp,
            (MOELayer),
        ):
            hidden_states, _, router_loss, gate_logits = self.mlp(
                hidden_states, token_type_ids
            )
        else:
            hidden_states = self.reshard_col(hidden_states)
            hidden_states = self.mlp(hidden_states)
            gate_logits = None

        with get_rng_state_tracker().rng_state("local_seed"):
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

            if isinstance(self.mlp, (MOELayer)):
                outputs += (router_loss,)
            else:
                router_loss = self.reshard_replicate(
                    paddle.zeros([1], dtype=paddle.float32)
                )
                outputs += (router_loss,)

            if output_gate_logits:
                outputs += (gate_logits,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]
        return list(outputs)


class ErniePretrainedModel(PretrainedModel):
    """
    ErniePretrainedModel is a pretrained model class for Ernie model.
    It is composed of a encoder and a decoder.
    """

    config_class = ErnieMoEConfig
    base_model_prefix = "ernie"

    def init_weights(self, layer):
        """Initialization hook"""
        rng_tracker = get_rng_state_tracker().rng_state
        from .moe_layer import TopKGateFused

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
        elif isinstance(layer, TopKGateFused):
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


class ErnieModel(ErniePretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ErnieDecoderLayer`]
    Args:
        config: ErnieMoEConfig
    """

    def __init__(self, config: ErnieMoEConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
        )

        self.layers = nn.LayerList()
        for idx in range(
            config.num_hidden_layers - 1
            if config.remove_tail_layer
            else config.num_hidden_layers
        ):
            self.layers.append(ErnieDecoderLayer(config, idx))

        Norm = RMSNorm if config.use_rmsnorm else LayerNorm
        self.norm = Norm(config)
        self.lm_head = ErnieLMHead(config)

        self.gradient_checkpointing = False

        if self.config.multi_token_pred_depth > 0:
            Norm = RMSNorm if config.use_rmsnorm else LayerNorm
            self.mtp_block = nn.LayerList(
                [
                    ErnieDecoderLayer(config, layer_idx, -1)
                    for layer_idx in range(self.config.multi_token_pred_depth)
                ]
            )
            self.mtp_hidden_norm = nn.LayerList(
                [Norm(config) for _ in range(self.config.multi_token_pred_depth)]
            )
            self.mtp_emb_norm = nn.LayerList(
                [Norm(config) for _ in range(self.config.multi_token_pred_depth)]
            )

            LinearFN = (
                paddle.incubate.nn.FusedLinear
                if config.fuse_linear
                else paddle.nn.Linear
            )
            self.mtp_linear_proj = nn.LayerList(
                [
                    LinearFN(
                        self.config.hidden_size * 2,
                        self.config.hidden_size,
                        bias_attr=config.use_bias,
                    )
                    for _ in range(self.config.multi_token_pred_depth)
                ]
            )

        self.all_gate_logits = () if hasattr(self.config, "use_moe") else None
        self.inbatch_pack_offset = None
        self.token_type_ids = None
        self.past_key_values = None
        self.inbatch_pack_offset = None
        self.inputs_embeds = None
        self.all_hidden_states = None
        self.all_self_attns = None
        self.next_decoder_cache = None
        self.inputs_embeds_cur_depth_list = None
        self.reshard_replicate = ReshardLayer()

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
            paddle.to_tensor(float(paddle.finfo(dtype).min), dtype=dtype),
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
            use_reentrant=True,
        )
        return hidden_states

    def embed_inputs(self, input_ids, attention_mask, position_ids):
        inputs_embeds = self.inputs_embeds

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

        if self.past_key_values is None:
            past_key_values = tuple([None] * self.config.num_hidden_layers)

        seq_length -= self.config.multi_token_pred_depth
        seq_length_with_past = seq_length
        cache_length = 0

        if past_key_values[0] is not None:
            cache_length = paddle.shape(past_key_values[0][0])[1]
            seq_length_with_past += cache_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).astype(
                self.embed_tokens.weight.dtype
            )

        if self.config.multi_token_pred_depth > 0:
            inputs_embeds_extra = inputs_embeds[
                :, -self.config.multi_token_pred_depth :, :
            ]  # [B, S, D]
            inputs_embeds = inputs_embeds[:, : -self.config.multi_token_pred_depth, :]
            inputs_embeds_ori = inputs_embeds
            inputs_embeds_cur_depth_list = []
            for depth in range(self.config.multi_token_pred_depth):
                inputs_embeds_cur_depth = paddle.concat(
                    [
                        inputs_embeds_ori[:, (depth + 1) :, :],
                        inputs_embeds_extra[:, : (depth + 1), :],
                    ],
                    axis=1,
                )
                inputs_embeds_cur_depth_list.append(inputs_embeds_cur_depth)
                self.inputs_embeds_cur_depth_list = paddle.concat(
                    inputs_embeds_cur_depth_list
                )

        if position_ids is not None:
            position_ids = self.reshard_replicate(position_ids)
        can_use_fa = self.config.use_flash_attn and flash_attention is not None

        if can_use_fa:
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
            attention_mask = self.reshard_replicate(attention_mask)

        if self.config.multi_token_pred_depth > 0:
            return (
                inputs_embeds,
                attention_mask,
                position_ids,
                self.inputs_embeds_cur_depth_list,
            )
        else:
            return inputs_embeds, attention_mask, position_ids

    def decode_layer(
        self,
        decoder_layer,
        hidden_states,
        attention_mask,
        position_ids,
        all_router_loss=None,
    ):
        if self.config.output_hidden_states:
            self.all_hidden_states += (hidden_states,)
        has_gradient = not hidden_states.stop_gradient
        position_ids_input = position_ids
        attention_mask_input = attention_mask
        token_type_ids_input = self.token_type_ids

        if self.config.use_recompute and has_gradient:
            layer_outputs = self.recompute_training(
                decoder_layer,
                hidden_states,
                attention_mask_input,
                position_ids_input,
                self.config.output_attentions,
                self.past_key_values,
                self.config.use_cache,
                self.inbatch_pack_offset,
                token_type_ids_input,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask_input,
                position_ids_input,
                self.config.output_attentions,
                self.past_key_values,
                self.config.use_cache,
                self.inbatch_pack_offset,
                token_type_ids_input,
            )

        if isinstance(layer_outputs, (tuple, list)):
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs

        if self.config.use_cache:
            self.next_decoder_cache += (
                layer_outputs[2 if self.config.output_attentions else 1],
            )

        if self.config.output_attentions:
            self.all_self_attns += (layer_outputs[1],)
        if hasattr(self.config, "use_moe") and self.config.use_moe:
            if not (self.config.use_recompute and has_gradient):
                layer_outputs, gate_logits = layer_outputs[:-1], layer_outputs[-1]
                self.all_gate_logits = self.all_gate_logits + (gate_logits,)
            router_loss = self.reshard_replicate(layer_outputs[-1])
            if all_router_loss is not None:
                all_router_loss += router_loss
        return hidden_states, all_router_loss

    def mtp_layer(
        self, hidden_states, inputs_embeds_cur_depth_list, attention_mask, position_ids
    ):
        has_gradient = not hidden_states.stop_gradient
        mtp_outputs = []
        mtp_outputs.append(hidden_states)

        for depth in range(self.config.multi_token_pred_depth):

            inputs_embeds_cur_depth = inputs_embeds_cur_depth_list[depth]

            # Norm&Concat
            inputs_embeds_cur_depth_norm = self.mtp_emb_norm[depth](
                inputs_embeds_cur_depth
            )
            hidden_states_norm = self.mtp_hidden_norm[depth](hidden_states)
            inputs_embeds_cur_depth = self.mtp_linear_proj[depth](
                paddle.concat(
                    [inputs_embeds_cur_depth_norm, hidden_states_norm], axis=-1
                )
            )

            decoder_layer = self.mtp_block[depth]
            past_key_values = None
            layer_outputs = decoder_layer(
                inputs_embeds_cur_depth,
                attention_mask,
                position_ids,
                self.config.output_attentions,
                past_key_values,
                self.config.use_cache,
                self.inbatch_pack_offset,
                self.token_type_ids,
            )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if self.config.use_moe:
                if not (self.config.use_recompute and has_gradient):
                    layer_outputs, gate_logits = layer_outputs[:-1], layer_outputs[-1]
                    self.all_gate_logits = self.all_gate_logits + (gate_logits,)

            mtp_outputs.append(hidden_states)
        mtp_outputs = [self.norm(hidden_states) for hidden_states in mtp_outputs]
        return mtp_outputs

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
        self.inputs_embeds = inputs_embeds
        self.past_key_values = past_key_values
        self.inbatch_pack_offset = inbatch_pack_offset
        self.token_type_ids = token_type_ids
        self.inbatch_pack_offset = inbatch_pack_offset
        if use_cache is not None:
            self.config.use_cache = use_cache
        if return_dict is not None:
            self.config.return_dict = return_dict
        if output_hidden_states is not None:
            self.config.output_hidden_states = output_hidden_states
        if output_attentions is not None:
            self.config.output_attentions = output_attentions

        if self.config.multi_token_pred_depth > 0:
            (
                hidden_states,
                attention_mask,
                position_ids,
                inputs_embeds_cur_depth_list,
            ) = self.embed_inputs(input_ids, attention_mask, position_ids)
        else:
            hidden_states, attention_mask, position_ids = self.embed_inputs(
                input_ids, attention_mask, position_ids
            )

        self.all_hidden_states = () if output_hidden_states else None
        self.all_self_attns = () if output_attentions else None
        self.next_decoder_cache = () if use_cache else None

        all_router_loss = None
        if hasattr(self.config, "use_moe") and self.config.use_moe:
            all_router_loss = self.reshard_replicate(paddle.to_tensor(0.0))

        for idx, (decoder_layer) in enumerate(self.layers):
            hidden_states, all_router_loss = self.decode_layer(
                decoder_layer,
                hidden_states,
                attention_mask,
                position_ids,
                all_router_loss,
            )

        if use_cache and not (hasattr(self.config, "use_moe") and self.config.use_moe):
            hidden_states = paddle.unsqueeze(hidden_states[:, -1, :], 1)

        # Multi Token Prediction
        mtp_outputs = []
        if self.config.multi_token_pred_depth > 0:
            inputs_embeds_cur_depth_list = paddle.split(
                inputs_embeds_cur_depth_list, self.config.multi_token_pred_depth
            )
            mtp_outputs = self.mtp_layer(
                hidden_states,
                inputs_embeds_cur_depth_list,
                attention_mask,
                position_ids,
            )
            hidden_states, mtp_outputs = mtp_outputs[0], mtp_outputs[1:]
        else:
            hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            self.all_hidden_states += (hidden_states,)

        next_cache = self.next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    self.all_hidden_states,
                    self.all_self_attns,
                    all_router_loss,
                    self.all_gate_logits,
                    mtp_outputs,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=self.all_hidden_states,
            attentions=self.all_self_attns,
            cross_attentions=None,
            router_loss=all_router_loss,
            gate_logits=self.all_gate_logits,
            mtp_outputs=mtp_outputs,
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

        self.loss_func = paddle.nn.CrossEntropyLoss(
            reduction="none",
        )

    def forward(self, prediction_scores, masked_lm_labels, router_loss=None):
        """
        calculates the final loss
        """
        if self.config.multi_token_pred_depth > 0:
            # prediction_scores :[logits, mtp_logits]
            logits = paddle.split(
                prediction_scores, self.config.multi_token_pred_depth + 1
            )
            prediction_scores = logits[0]
            mtp_logits = logits[1:]
            masked_lm_labels_ori = masked_lm_labels
            masked_lm_labels = masked_lm_labels[
                :, : -self.config.multi_token_pred_depth
            ]
            seq_length = masked_lm_labels.shape[1]
        res = self.forward_impl(prediction_scores, masked_lm_labels)
        if self.config.multi_token_pred_depth > 0:
            mtp_loss_res = []
            for depth in range(self.config.multi_token_pred_depth):
                prediction_scores_cur_depth = mtp_logits[depth]
                masked_lm_labels_cur_depth = masked_lm_labels_ori[
                    :, (depth + 1) : (depth + 1 + seq_length)
                ]
                res_cur_depth = self.forward_impl(
                    prediction_scores_cur_depth,
                    masked_lm_labels_cur_depth,
                )
                mtp_loss_res.append(res_cur_depth)

        def add_loss(main_loss, loss):
            return main_loss + loss - loss.detach()

        if self.return_tuple:
            loss, loss_sum = res
            if self.config.multi_token_pred_depth > 0:
                loss = add_loss(
                    loss,
                    self.config.multi_token_pred_lambda
                    * sum([x[0] for x in mtp_loss_res])
                    / len(mtp_loss_res),
                )
                loss_sum = loss_sum + self.config.multi_token_pred_lambda * sum(
                    [x[1].detach() for x in mtp_loss_res]
                ) / len(mtp_loss_res)
        else:
            loss, loss_sum = res, None
            if self.config.multi_token_pred_depth > 0:
                loss = add_loss(
                    loss,
                    self.config.multi_token_pred_lambda
                    * sum(mtp_loss_res)
                    / len(mtp_loss_res),
                )
        if router_loss is not None:
            loss = loss + router_loss - router_loss.detach()
        if not self.return_tuple:
            return loss
        return loss, loss_sum

    def forward_impl(self, prediction_scores, masked_lm_labels):
        with paddle.amp.auto_cast(False):
            masked_lm_loss = self.loss_func(
                prediction_scores.astype("float32"), masked_lm_labels.unsqueeze(-1)
            )
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
        self.weight = self.create_parameter(
            shape=(
                [config.vocab_size, config.hidden_size]
                if config.tie_word_embeddings
                else [config.hidden_size, config.vocab_size]
            ),
            dtype=paddle.get_default_dtype(),
        )

        self.weight.is_distributed = False

        logger.info(
            f"output-weight:{self.weight.shape} config.tie_word_embeddings={config.tie_word_embeddings}"
        )
        if config.weight_share_add_bias and config.use_bias:
            self.bias = self.create_parameter(
                shape=[config.vocab_size],
                dtype=paddle.get_default_dtype(),
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.constant.Constant(0.0)
                ),
            )
            self.bias.is_distributed = False
        else:
            self.bias = None

        if self.config.use_recompute_loss_fn:
            logger.info(
                "Using recompute_loss_fn, the calculation of logits will be moved into "
                "loss_fn for memory optimization"
            )

    def forward(self, hidden_states):
        return calc_lm_head_logits(
            self.config,
            hidden_states,
            self.weight,
            self.bias,
            None,
        )


class ErnieForCausalLM(ErniePretrainedModel):
    """
    ErnieForCausalLM is the model class for causal language modeling.
    """

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        config.initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(f"Initializer-range is {config.initializer_range}")
        self.config = config
        self.criterion = ErniePretrainingCriterion(config, False)

        self.tie_weights()

        self.ernie = ErnieModel(config)
        self.lm_head = ErnieLMHead(config)

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

        from .moe_layer import MOELayer, ErnieMoeMLP

        layers = self.ernie.layers
        if hasattr(self.config, "use_moe") and self.config.use_moe:
            with paddle.no_grad():
                for left in layers:
                    if isinstance(
                        left.self_attn.o_proj,
                        (MOELayer),
                    ):
                        for e in left.self_attn.o_proj.experts:
                            if isinstance(e, ErnieMoeMLP):
                                scale_by_factor_if_valid(e.weight)
                    else:
                        scale_by_factor_if_valid(left.self_attn.o_proj.weight)

                    if isinstance(
                        left.mlp,
                        (MOELayer),
                    ):
                        for e in left.mlp.experts:
                            if isinstance(e, ErnieMoeMLP):
                                scale_by_factor_if_valid(e.down_proj.weight)
                    else:
                        scale_by_factor_if_valid(left.mlp.down_proj.weight)
        else:
            with paddle.no_grad():
                for left in layers:
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
        mtp_outputs = outputs.mtp_outputs
        logits = self.lm_head(hidden_states)

        mtp_logits = [logits]
        if len(mtp_outputs) > 0:
            for _hidden_states in mtp_outputs:
                mtp_logits.append(self.lm_head(_hidden_states))
            logits = paddle.concat(mtp_logits)

        if return_dict:
            if labels is not None:
                loss, _ = self.criterion(logits, labels)
            else:
                loss = None
            return CausalLMOutputWithCrossAttentionsErnie(
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

    def auto_dist_config(self, prefix=""):
        if prefix != "":
            assert prefix.endswith(".")

        config = {
            "sp_config": {
                "parallelize_plan": {
                    f"{prefix}ernie.embed_tokens": [
                        dist.ColWiseParallel(),
                        dist.SequenceParallelBegin(),
                    ],
                    f"{prefix}ernie.layers.*.self_attn": dist.SequenceParallelDisable(),
                    f"{prefix}ernie.layers.*.self_attn.q_proj": dist.ColWiseParallel(),
                    f"{prefix}ernie.layers.*.self_attn.k_proj": dist.ColWiseParallel(),
                    f"{prefix}ernie.layers.*.self_attn.v_proj": dist.ColWiseParallel(),
                    f"{prefix}ernie.layers.*.self_attn.o_proj": dist.RowWiseParallel(),
                    f"{prefix}ernie.layers.*.self_attn.reshard_row_and_col": PrepareLayerInput(
                        layer_input_reshard_row_and_col_hook
                    ),
                    f"{prefix}ernie.layers.*.reshard_col": PrepareLayerInput(
                        layer_input_reshard_col_hook
                    ),
                    f"{prefix}ernie.layers.*.reshard_replicate": PrepareLayerInput(
                        layer_input_reshard_replicate_hook
                    ),
                    f"{prefix}ernie.layers.*.mlp": dist.SequenceParallelDisable(
                        need_transpose=False
                    ),
                    f"{prefix}ernie.layers.*.mlp.gate_proj": dist.ColWiseParallel(),
                    f"{prefix}ernie.layers.*.mlp.up_proj": dist.ColWiseParallel(),
                    f"{prefix}ernie.layers.*.mlp.down_proj": dist.RowWiseParallel(),
                    f"{prefix}ernie.reshard_replicate": PrepareLayerInput(
                        layer_input_reshard_replicate_hook
                    ),
                    f"{prefix}lm_head.weight": dist.ColWiseParallel(),
                    f"{prefix}lm_head": dist.SequenceParallelEnd(),
                }
            },
            "mp_config": {
                "parallelize_plan": {
                    f"{prefix}ernie.embed_tokens": dist.ColWiseParallel(),
                    f"{prefix}ernie.layers.*.self_attn.q_proj": dist.ColWiseParallel(),
                    f"{prefix}ernie.layers.*.self_attn.k_proj": dist.ColWiseParallel(),
                    f"{prefix}ernie.layers.*.self_attn.v_proj": dist.ColWiseParallel(),
                    f"{prefix}ernie.layers.*.self_attn.o_proj": dist.RowWiseParallel(),
                    f"{prefix}ernie.layers.*.reshard_replicate": PrepareLayerInput(
                        layer_input_reshard_replicate_hook
                    ),
                    f"{prefix}ernie.layers.*.mlp.gate_proj": dist.ColWiseParallel(),
                    f"{prefix}ernie.layers.*.mlp.up_proj": dist.ColWiseParallel(),
                    f"{prefix}ernie.layers.*.mlp.down_proj": dist.RowWiseParallel(),
                    f"{prefix}ernie.reshard_replicate": PrepareLayerInput(
                        layer_input_reshard_replicate_hook
                    ),
                    f"{prefix}lm_head.weight": dist.ColWiseParallel(),
                }
            },
            "pp_config": {
                "split_spec": f"{prefix}ernie.layers",
            },
        }
        return config


class ReshardLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


def layer_input_reshard_row_and_col_hook(process_mesh):
    def hook(layer, inputs, output=None):
        res_inputs = []
        for input in inputs:
            if not input.is_dist():
                x = dist.shard_tensor(
                    input, process_mesh, [dist.Shard(0), dist.Shard(1)]
                )
                res_inputs.append(
                    dist.reshard(x, process_mesh, [dist.Shard(0), dist.Shard(1)])
                )
            else:
                res_inputs.append(
                    dist.reshard(input, process_mesh, [dist.Shard(0), dist.Shard(1)])
                )
        return tuple(res_inputs)

    return hook


def layer_input_reshard_col_hook(process_mesh):
    def hook(layer, inputs, output=None):
        res_inputs = []
        for input in inputs:
            if not input.is_dist():
                x = dist.shard_tensor(
                    input, process_mesh, [dist.Shard(1), dist.Replicate()]
                )
                res_inputs.append(
                    dist.reshard(x, process_mesh, [dist.Shard(1), dist.Replicate()])
                )
            else:
                res_inputs.append(
                    dist.reshard(input, process_mesh, [dist.Shard(1), dist.Replicate()])
                )
        return tuple(res_inputs)

    return hook


def layer_input_reshard_replicate_hook(process_mesh):
    def hook(layer, inputs, output=None):
        res_inputs = []
        for input in inputs:
            if not input.is_dist():
                x = dist.shard_tensor(
                    input,
                    process_mesh,
                    [dist.Replicate() for _ in range(len(process_mesh.shape))],
                )
                res_inputs.append(
                    dist.reshard(
                        x,
                        process_mesh,
                        [dist.Replicate() for _ in range(len(process_mesh.shape))],
                    )
                )
            else:
                res_inputs.append(
                    dist.reshard(
                        input,
                        process_mesh,
                        [dist.Replicate() for _ in range(len(process_mesh.shape))],
                    )
                )
        return tuple(res_inputs)

    return hook
