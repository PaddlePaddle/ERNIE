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


import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.pipelining.schedules import (
    Schedule1F1B,
    ScheduleFThenB,
    ScheduleVPP,
)
from paddle.distributed.auto_parallel.pipelining.stage import PipelineStage

from paddle.distributed.fleet.utils import recompute


from models.moe.moe_utils_auto import get_mesh

from .modeling_auto import (
    _parse_moe_group,
    ErnieDecoderLayerAuto,
    ErniePretrainedModelAuto,
    LayerNorm,
    RMSNorm,
    FusedLayerNorm,
    ErniePretrainingCriterion,
    ErnieLMHead,
)

from paddle.distributed import in_auto_parallel_align_mode


# Because param_name is generated based on the class name,
# when changes in distributed strategies result in class modifications,
# there may be mismatches during parameter loading.
# You can achieve class name changes by importing the following environment variables.
# Example: `export rowcol_parallel_linear_class_name_convert_map="tpsp->smp"`

logger = logging.getLogger(__name__)

try:
    from paddle.nn.functional.flash_attention import flash_attention

    logger.warning(
        "Use flash attention in scaled-dot-product. Attention mask is deprecated"
    )
except (ImportError, ModuleNotFoundError):
    flash_attention = None


__all__ = [
    "get_ernie_pp_schedule",
    "ErnieForCausalLMAutoPP",
]


def parse_args(args):
    hidden_states, attention_mask, position_ids = None, None, None
    if isinstance(args, tuple):
        if len(args) == 3:
            hidden_states, attention_mask, position_ids = args
        elif len(args) == 2:
            hidden_states, attention_mask = args
        elif len(args) == 1:
            hidden_states = args[0]
    else:
        hidden_states = args
    if position_ids is not None:
        position_ids.stop_gradient = True

    if attention_mask is not None:
        attention_mask.stop_gradient = True

    return hidden_states, attention_mask, position_ids


def return_args(hidden_states, attention_mask=None, position_ids=None):

    ret = (hidden_states,)

    if attention_mask is not None:
        ret += (attention_mask.clone(),)
    if position_ids is not None:
        ret += (position_ids.clone(),)
    if len(ret) == 1:
        ret = ret[0]

    return ret


def global_mesh_starts_with_pp():

    mesh = fleet.auto.get_mesh()
    if "pp" in mesh.dim_names:
        return mesh.get_mesh_with_dim("pp")
    else:
        return mesh


class ErnieChunk(nn.Layer):
    def __init__(self, layers=None, is_first=False):

        super(ErnieChunk, self).__init__()
        self.layers = layers
        self.is_first = is_first

    def forward(self, *args, **kwargs):
        """
            Forward function of the model.

        Args:
            *args (tuple, optional): Tuple containing input tensors. If is_first is True,
                input_ids, attention_mask and position_ids are required; otherwise,
                it should be a tuple of output tensors from previous layer. Default None.
            **kwargs (dict, optional): Dictionary containing input tensors. If is_first is False,
                input_ids, attention_mask and position_ids are required; otherwise, it should be
                an empty dictionary. Default None.

        Returns:
            tuple (list[Tensor], Tensor, Tensor): Tuple containing output tensors from each decoder layer.
            The first item is a list of output tensors from each decoder layer, the second item is the last
            hidden state of the encoder, and the third item is the last position encoding index.
        """
        if self.is_first:
            input_ids = kwargs.get("input_ids")
            attention_mask = kwargs.get("attention_mask")
            position_ids = kwargs.get("position_ids")
            outputs = tuple([input_ids, attention_mask, position_ids])
            # decoder layers
            for idx, (decoder_layer) in enumerate(self.layers):
                outputs = decoder_layer(outputs)
            return outputs
        else:
            outputs = args
            # decoder layers
            for idx, (decoder_layer) in enumerate(self.layers):
                outputs = decoder_layer(outputs)
        return outputs


def manual_model_split(model, stage_idx, group, mode, pp_degree):

    num_hidden_layers = model.config.num_hidden_layers
    virtual_pp_degree = model.config.virtual_pp_degree if mode == "VPP" else 1
    chunk_size = num_hidden_layers // virtual_pp_degree // pp_degree
    chunk_num = virtual_pp_degree * pp_degree
    layer_lists = None

    layer_lists = model.layers

    def _build_stage(model, stage_idx, group):
        new_model = None
        if stage_idx == 0:
            new_model = ErnieChunk(layer_lists[:chunk_size], is_first=True)
        else:
            new_model = ErnieChunk(
                layer_lists[stage_idx * chunk_size : (stage_idx + 1) * chunk_size],
                is_first=False,
            )
        stage = PipelineStage(new_model, stage_idx, chunk_num, group=group)
        return stage

    stages = []
    for i in range(virtual_pp_degree):
        stage = _build_stage(model, stage_idx + i * pp_degree, group)
        stages.append(stage)
    return stages


def get_ernie_pp_schedule(model, n_microbatches, loss_fn, mode, pp_degree, group):

    assert mode in ["VPP", "1F1B", "FThenB"]
    stages = manual_model_split(model, group.rank, group, mode, pp_degree)
    if mode == "VPP":
        schedule = ScheduleVPP(stages, n_microbatches=n_microbatches, loss_fn=loss_fn)
    elif mode == "1F1B":
        schedule = Schedule1F1B(
            stages[0], n_microbatches=n_microbatches, loss_fn=loss_fn
        )
    else:
        schedule = ScheduleFThenB(
            stages[0], n_microbatches=n_microbatches, loss_fn=loss_fn
        )
    return schedule


class ErnieDecoderLayerAutoPP(nn.Layer):
    def __init__(self, config, layer_idx=0, ipp=0):
        """
            Initializes the model.

        Args:
            config (ErnieConfig): The configuration of the model.
            layer_idx (int, optional): The index of the decoder layer. Defaults to 0.
            ipp (int, optional): The index of the inner parallelism dimension. Defaults to 0.

        Returns:
            None.
        """
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

        super().__init__()
        self.config = config

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

        self.layer_idx = layer_idx
        self.ipp = ipp
        self.placements = (
            [dist.Shard(1), dist.Shard(0)]
            if self.config.sequence_parallel
            else [dist.Shard(0), dist.Replicate()]
        )
        self.embed_tokens = None
        self.norm = None
        self.lm_head = None
        if layer_idx == 0:
            self.vocab_size = config.vocab_size
            self.hidden_size = config.hidden_size
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
        self.layer = ErnieDecoderLayerAuto(config, layer_idx, ipp)

        Norm = RMSNorm if config.use_rmsnorm else LayerNorm
        if not config.use_rmsnorm and config.fuse_ln:
            Norm = FusedLayerNorm
        if self.layer_idx == self.config.num_hidden_layers - 1:
            self.norm = Norm(config, -1)
            self.lm_head = ErnieLMHead(config)

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

    def forward(self, args):
        output_attentions = self.config.output_attentions
        use_cache = self.config.use_cache
        output_hidden_states = self.config.output_hidden_states
        return_dict = self.config.return_dict
        past_key_values = None
        past_key_value = None
        token_type_ids = None
        inbatch_pack_offset = None
        if self.embed_tokens is not None:

            input_ids, attention_mask, position_ids = parse_args(args)
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
            use_cache = use_cache if use_cache is not None else self.config.use_cache

            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )

            # retrieve input_ids and inputs_embeds
            if input_ids is not None:
                batch_size, seq_length = input_ids.shape
            else:
                raise ValueError(
                    "You have to specify either input_ids or inputs_embeds"
                )

            seq_length_with_past = seq_length
            cache_length = 0

            if past_key_values is not None:
                cache_length = paddle.shape(past_key_values[0])[1]
                seq_length_with_past += cache_length
            inputs_embeds = self.embed_tokens(input_ids).astype(
                self.embed_tokens.weight.dtype
            )

            if self.config.sequence_parallel:
                # [B, S, H] -> [S, B, H]
                inputs_embeds = paddle.transpose(inputs_embeds, [1, 0, 2])
                # if token_type_ids is not None:
                # token_type_ids = token_type_ids.reshape([-1, 1])
                # token_type_ids = dist.reshard(
                #     token_type_ids, global_mesh, [dist.Replicate() for _ in range(len(global_mesh._shape))]
                # )
                # token_type_ids = token_type_ids.reshape([-1])
            global_mesh = global_mesh_starts_with_pp()

            if position_ids is not None:
                position_ids = dist.shard_tensor(
                    position_ids,
                    global_mesh,
                    [dist.Replicate() for _ in range(len(global_mesh._shape))],
                )

            can_use_fa = self.config.use_flash_attn and flash_attention is not None
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
                hidden_states = dist.reshard(
                    hidden_states, get_mesh(0), self.placements
                )

            args = return_args(hidden_states, attention_mask, position_ids)

        # decoder layers
        hidden_states, attention_mask, position_ids = parse_args(args)

        all_hidden_states = () if output_hidden_states else None

        all_router_loss = None
        if hasattr(self.config, "use_moe") and self.config.use_moe:
            all_router_loss = paddle.to_tensor(0.0)
            all_router_loss = dist.shard_tensor(
                all_router_loss, get_mesh(0), dist.Replicate()
            )

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        has_gradient = not hidden_states.stop_gradient
        if position_ids is not None:
            position_ids_input = dist.reshard(
                position_ids,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Replicate()],
            )
        else:
            position_ids_input = position_ids
        attention_mask_input = (
            dist.reshard(
                attention_mask,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Replicate()],
            )
            if attention_mask is not None
            else None
        )
        token_type_ids_input = (
            dist.reshard(
                token_type_ids,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Replicate()],
            )
            if token_type_ids is not None
            else None
        )
        if self.config.use_recompute and has_gradient:
            layer_outputs = self.recompute_training(
                self.layer,
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
            layer_outputs = self.layer(
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

        ret_args = return_args(
            hidden_states,
            attention_mask,
            position_ids,
        )
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
            ret_args = return_args(
                logits,
            )

        return ret_args


class ErniePretrainingCriterionPP(ErniePretrainingCriterion):
    """
    Criterion for Ernie.
    It calculates the final loss.
    """

    def __init__(self, config):

        super().__init__(config)

    def forward(self, prediction_scores, masked_lm_labels, router_loss=None):
        """
        calculates the final loss
        """
        losses = super().forward(prediction_scores, masked_lm_labels)
        if losses is not None:
            loss = losses[0]
        else:
            print("err")
        return loss


class ErnieForCausalLMAutoPP(ErniePretrainedModelAuto):
    """
    ErnieForCausalLMAutoPP is the model class for causal language modeling.
    """

    def __init__(self, config):
        """
            Args:
            config (Config): Config object containing hyperparameters and other configuration details.

        Returns:
            None.

        Initializes the ErnieDecoder with the given config.
        """
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

        # initialize-trick for big model, see
        # https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/README.md#std-init
        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(
            f"change initializer-range from {config.initializer_range} to {new_initializer_range}"
        )
        config.initializer_range = new_initializer_range
        self.config = config
        self.criterion = ErniePretrainingCriterionPP(config)

        if self.config.use_rmsnorm:
            if self.config.fuse_rms_norm:
                logger.info("Use fusedRMSNorm")
            else:
                logger.info("Use normal RMSNorm")
        else:
            if self.config.fuse_ln:
                logger.info("Use fusedLN")
            else:
                logger.info("Use normal LayerNorm")

        decoder_layers = []

        def get_pp_stage_id(layer_id):
            pp_degree = global_mesh_starts_with_pp().shape[0]
            chunk_size = self.config.num_hidden_layers // (
                pp_degree * self.config.virtual_pp_degree
            )
            chunk_id = layer_id // chunk_size
            pp_stage_id = chunk_id % pp_degree
            return pp_stage_id

        for i in range(config.num_hidden_layers):
            pp_stage_id = get_pp_stage_id(i)
            decoder_layers.append(ErnieDecoderLayerAutoPP(config, i, pp_stage_id))
        self.layers = nn.LayerList(decoder_layers)

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
        outputs = return_args(input_ids, attention_mask, position_ids)

        for layer in self.layers:
            outputs = layer(outputs)

        return outputs[0]
