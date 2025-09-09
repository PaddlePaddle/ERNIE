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


import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle import nn
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker

from .modeling import (
    RMSNorm,
    ErniePretrainedModel,
    ErnieModel,
    ErniePretrainingCriterion,
    ReshardLayer,
    ErnieLMHead,
    ErnieDecoderLayer,
    ErnieAttention,
    ErnieForCausalLM,
    ErnieMLP,
)


from models.moe_layer import MOELayer
from models.configuration import ErnieMoEConfig
from utils.training_utils import get_mesh

logger = logging.getLogger(__name__)


__all__ = [
    "ErnieForCausalLMVPP",
]


class LayerNorm(nn.LayerNorm):
    def __init__(self, config, ipp=0):
        super().__init__(config.hidden_size, epsilon=config.rms_norm_eps)
        if config.pipeline_parallel_degree > 1:
            self.weight = dist.shard_tensor(
                self.weight, get_mesh(ipp), [dist.Replicate(), dist.Replicate()]
            )
            self.bias = dist.shard_tensor(
                self.bias, get_mesh(ipp), [dist.Replicate(), dist.Replicate()]
            )


class ErnieMLPVPP(ErnieMLP):
    def __init__(self, config, ipp=None, do_shard_tensor=True):
        super().__init__(config, ipp, do_shard_tensor)
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

    def forward(self, x):
        out = super().forward(x)
        if self.config.sequence_parallel:
            out = dist.reshard(out, get_mesh(self.ipp), [dist.Shard(1), dist.Shard(0)])
        return out


class ErnieAttentionVPP(ErnieAttention):
    def __init__(self, config, ipp: Optional[int] = None):
        super().__init__(config, ipp)
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
            attn_output = self.o_proj(paddle.transpose(attn_output, [1, 0, 2]))
            attn_output = dist.reshard(
                attn_output, get_mesh(self.ipp), [dist.Shard(1), dist.Shard(0)]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class ErnieDecoderLayerVPP(ErnieDecoderLayer):
    """
    ErnieDecoderLayerVPP is ErnieDecoderLayer with sequence_parallel and tensor_parallel.
    """

    def __init__(self, config, layer_idx=0, ipp=0):
        super().__init__(config, layer_idx, ipp)
        self.self_attn = ErnieAttentionVPP(config, ipp)
        if isinstance(self.mlp, ErnieMLP):
            self.mlp = ErnieMLPVPP(config, ipp)

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
            (MOELayer),
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

            if isinstance(self.mlp, (MOELayer)):
                outputs += (router_loss,)
            else:
                outputs += (paddle.zeros([1], dtype=paddle.float32),)

            if output_gate_logits:
                outputs += (gate_logits,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]
        return outputs


class ErnieModelVPP(ErnieModel):
    """
    ErnieModelVPP is a variant of ErnieModel that support vpp schedule.
    """

    def __init__(self, config: ErnieMoEConfig, pp_layer_idx=None, ipp=0):
        super(ErniePretrainedModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.config = config
        self.layer = ErnieDecoderLayerVPP(config, pp_layer_idx, ipp)

        if pp_layer_idx == 0:
            self.vocab_size = config.vocab_size
            self.hidden_size = config.hidden_size
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )
            self.embed_tokens.weight = dist.shard_tensor(
                self.embed_tokens.weight,
                get_mesh(pp_idx=0),
                [dist.Replicate(), dist.Shard(1)],
            )
        if pp_layer_idx == self.config.num_hidden_layers - 1:
            Norm = RMSNorm if config.use_rmsnorm else LayerNorm
            self.norm = Norm(config)
            self.lm_head = ErnieLMHeadVPP(config)

        self.gradient_checkpointing = False

        self.placements = (
            [dist.Shard(1), dist.Shard(0)]
            if self.config.sequence_parallel
            else [dist.Shard(0), dist.Replicate()]
        )

        if self.config.multi_token_pred_depth > 0:
            Norm = RMSNorm if config.use_rmsnorm else LayerNorm
            self.mtp_block = nn.LayerList(
                [
                    ErnieDecoderLayerVPP(config, layer_idx, -1)
                    for layer_idx in range(self.config.multi_token_pred_depth)
                ]
            )
            self.mtp_hidden_norm = nn.LayerList(
                [Norm(config, -1) for _ in range(self.config.multi_token_pred_depth)]
            )
            self.mtp_emb_norm = nn.LayerList(
                [Norm(config, -1) for _ in range(self.config.multi_token_pred_depth)]
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

    def mtp_layer(
        self, hidden_states, inputs_embeds_cur_depth_list, attention_mask, position_ids
    ):
        has_gradient = not hidden_states.stop_gradient
        mtp_outputs = []
        mtp_outputs.append(hidden_states)

        for depth in range(self.config.multi_token_pred_depth):
            if self.config.sequence_parallel or self.config.submatrix_parallel:
                hidden_states = dist.reshard(
                    hidden_states,
                    get_mesh(-1),
                    [dist.Replicate(), dist.Replicate()],
                )
                hidden_states = paddle.transpose(hidden_states, [1, 0, 2])

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

            # scatter
            if self.config.sequence_parallel or self.config.submatrix_parallel:
                inputs_embeds_cur_depth = paddle.transpose(
                    inputs_embeds_cur_depth, [1, 0, 2]
                )
                inputs_embeds_cur_depth = dist.reshard(
                    inputs_embeds_cur_depth,
                    get_mesh(-1),
                    self.placements,
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

    def forward(self, args):
        attention_mask, position_ids = None, None
        if isinstance(args, tuple):
            hidden_states = args[0] if len(args) > 0 else args
            attention_mask = args[1] if len(args) > 1 else None
            position_ids = args[2] if len(args) > 2 else None

            if len(args) == 2 and self.config.multi_token_pred_depth > 0:
                hidden_states = args[0]
                inputs_embeds_cur_depth_list = args[1]
        else:
            hidden_states = args
        if self.layer.layer_idx == 0:
            if self.config.multi_token_pred_depth > 0:
                (
                    hidden_states,
                    attention_mask,
                    position_ids,
                    inputs_embeds_cur_depth_list,
                ) = self.embed_inputs(hidden_states, attention_mask, position_ids)
            else:
                hidden_states, attention_mask, position_ids = self.embed_inputs(
                    hidden_states, attention_mask, position_ids
                )
            global_mesh = get_mesh(pp_idx=None)
            if self.config.sequence_parallel:
                hidden_states = paddle.transpose(hidden_states, [1, 0, 2])

            if position_ids is not None:
                position_ids = dist.shard_tensor(
                    position_ids,
                    global_mesh,
                    [dist.Replicate() for _ in range(len(global_mesh._shape))],
                )
            if attention_mask is not None:
                attention_mask = dist.shard_tensor(
                    attention_mask,
                    global_mesh,
                    [dist.Replicate() for _ in range(len(global_mesh._shape))],
                )
            hidden_states = dist.reshard(hidden_states, get_mesh(0), self.placements)

        hidden_states, _ = self.decode_layer(
            self.layer, hidden_states, attention_mask, position_ids
        )
        if self.layer.layer_idx == self.config.num_hidden_layers - 1:
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
            logits = self.lm_head(hidden_states)

            if self.config.multi_token_pred_depth > 0:
                mtp_logits = [logits]
                for _hidden_states in mtp_outputs:
                    mtp_logits.append(self.lm_head(_hidden_states))
                logits = paddle.concat(mtp_logits)
            return logits
        else:
            if self.config.multi_token_pred_depth > 0:
                return hidden_states, inputs_embeds_cur_depth_list
            else:
                return hidden_states


class ErnieLMHeadVPP(ErnieLMHead):
    """
    ErnieLMHeadVPP is ErnieLMHead for vpp schedule with shard_tensor
    """

    def __init__(self, config):
        super().__init__(config)
        if (
            self.config.tensor_parallel_degree > 1
            or self.config.pipeline_parallel_degree > 1
        ):
            self.weight = dist.shard_tensor(
                self.weight,
                get_mesh(-1),
                [dist.Replicate(), dist.Shard(1)],
            )
        if self.bias:
            if (
                self.config.tensor_parallel_degree > 1
                or self.config.pipeline_parallel_degree > 1
            ):
                self.bias = dist.shard_tensor(
                    self.bias,
                    get_mesh(-1),
                    [dist.Replicate(), dist.Shard(0)],
                )

    def forward(self, hidden_states):
        if self.config.sequence_parallel:
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
            hidden_states = hidden_states.reshape(
                [-1, self.config.seqlen, hidden_states.shape[-1]]
            )
        return super().forward(hidden_states)


class ErnieForCausalLMVPP(ErnieForCausalLM):
    """
    ErnieForCausalLMVPP is the model class for causal language modeling for vpp pipeline schedule mode.
    """

    def __init__(self, config):
        super(ErniePretrainedModel, self).__init__(config)
        config.initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(f"Initializer-range is {config.initializer_range}")
        self.config = config
        self.criterion = ErniePretrainingCriterion(config, False)
        self.tie_weights()

        if config.pipeline_parallel_degree > 1:
            self.layers = nn.LayerList()
            pp_degree = config.pipeline_parallel_degree
            chunk_size = (
                config.num_hidden_layers // pp_degree // config.virtual_pp_degree
            )
            current_rank = (
                fleet.get_hybrid_communicate_group().get_pipe_parallel_group().rank
                % pp_degree
            )
            for idx in range(config.num_hidden_layers):
                target_stage = (idx // chunk_size) % pp_degree
                if target_stage == current_rank:
                    stage_id = (idx // chunk_size) % pp_degree
                    self.layers.append(ErnieModelVPP(config, idx, stage_id))
                else:
                    self.layers.append(nn.Identity())

    def _post_init(self, original_init, *args, **kwargs):
        decoder_layers = []
        for layer in self.layers:
            if isinstance(layer, ErnieModelVPP):
                decoder_layers.append(layer.layer)
        layers = decoder_layers
        self.ernie = type("ernie", (), {"layers": layers})()
        super()._post_init(self, original_init, *args, **kwargs)
