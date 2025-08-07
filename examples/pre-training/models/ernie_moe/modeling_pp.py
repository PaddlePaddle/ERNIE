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
docstring
"""

import copy
import math
import logging
import os

import numpy as np
import contextlib
from collections import deque

import paddle
import paddle.distributed.fleet as fleet
import paddle.distributed as dist
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
)
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker
from paddle.distributed.fleet.utils import recompute

from paddle.distributed.fleet.layers.mpu.mp_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

from paddleformers.transformers import PretrainedModel

from models.sequence_parallel_utils import (
    ColumnSequenceParallelLinear,
    RowSequenceParallelLinear,
)
from models.ernie.modeling_pp import create_skip_config_for_refined_recompute
from models.ernie import ErnieMoEConfig
from models.sequence_parallel_utils import (
    ScatterOp,
    mark_as_sequence_parallel_parameter,
)
from models.moe.moe_layer import MOELayer
from models.moe.moe_layer_uneven import MOELayer as MOELayerSizeAll2All

from models.moe.top2_gate import Top2Gate
from models.moe.round_robin_gate import RoundRobinGate

from models.sequence_parallel_utils import (
    GatherOp,
)

from models.ernie_moe.modeling import (
    ErnieDecoderLayer,
    ErnieMoELMHead,
    ErniePretrainedModel,
    ErniePretrainingCriterion,
    RotaryEmbedding,
    RMSNorm,
    LayerNorm,
    ErnieModel,
    _parse_moe_group,
    moe_statedict_upcycle,
    moe_statedict_local_id_to_global,
    moe_statedict_cherry_pick,
    moe_ep2mp,
    ErnieMLP,
)
from models.ernie.modeling import FusedLayerNorm
from models.utils import inplace_offload
from models.ernie.modeling import fp8_autocast, te_recompute, TEFP8Linear

try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}

from paddleformers.utils.tools import get_env_device

try:
    from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
        pipeline_bubble_hooks_,
    )
except ImportError:
    pipeline_bubble_hooks_ = None

try:
    from paddle.framework.recall_error import AADIFF_ERROR
except ImportError:
    AADIFF_ERROR = "CUDA error(1001)"


try:
    from paddle.distributed import in_auto_parallel_align_mode
except ImportError:

    def in_auto_parallel_align_mode():
        """
        hack for paddlenlp develop branch.
        """
        return False


input_ids_for_mtp = deque()
NativeLinear = nn.Linear
if get_env_device() == "xpu":
    try:
        from paddle_xpu.layers.nn import Linear as XPULinear
        from paddle_xpu.layers.nn import ColumnParallelLinear as XPUColumnParallelLinear
        from paddle_xpu.layers.nn import RowParallelLinear as XPURowParallelLinear

        NativeLinear = XPULinear
        ColumnParallelLinear = XPUColumnParallelLinear
        RowParallelLinear = XPURowParallelLinear
    except ImportError:
        print("[WARN] import paddle_xpu for mp error!")

    try:
        from paddle_xpu.layers.nn.sequence_parallel import GatherOp as XPUGatherOp
        from paddle_xpu.layers.nn.sequence_parallel import ScatterOp as XPUScatterOp

        GatherOp = XPUGatherOp
        ScatterOp = XPUScatterOp
    except ImportError:
        print("[WARN] import paddle_xpu for sp error!")

    if not os.getenv("XPU_MOE_USE_ALLGATHER", False):
        try:
            from paddle_xpu.layers.nn.sequence_parallel import (
                XPUColumnSequenceParallelLinear,
                XPURowSequenceParallelLinear,
            )

            ColumnSequenceParallelLinear = XPUColumnSequenceParallelLinear
            RowSequenceParallelLinear = XPURowSequenceParallelLinear
        except ImportError:
            print("[WARN] import paddle_xpu for sp error!")

logger = logging.getLogger(__name__)


def get_hcg():

    return fleet.get_hybrid_communicate_group()


class ErnieEmbeddingPipe(nn.Layer):
    """Extends ErnieEmbeddingPipe to forward attention_mask through the pipeline."""

    def __init__(self, config):
        self.sequence_parallel = config.sequence_parallel
        self.use_mem_eff_attn = config.use_mem_eff_attn
        self.config = config

        super(ErnieEmbeddingPipe, self).__init__()
        self.use_moe = config.use_moe
        if config.tensor_parallel_degree > 1 and not in_auto_parallel_align_mode():
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    @property
    def embedding_weight(self):
        """embedding_weight property"""
        return getattr(self.embed_tokens, "weight")

    def forward(self, args):
        """
        :param args: Tuple or Tensor, containing the inputs to the module.
                     If it's a tuple, then it should contain at least three elements,
                     i.e. input_ids, attention_mask and position_ids.
                     In this case, the first three elements will be used as the arguments for forward.
                     The order of these elements is defined by `forward` method.

        :return: A Tensor with shape [batch_size * beam_width, sequence_length, hidden_dim].
                  This is the output of the model when given the specified input.

        """
        if isinstance(args, tuple):
            if len(args) == 3:
                if self.config.skip_align_position_id and self.use_mem_eff_attn:
                    input_ids, position_ids, inbatch_pack_offset = args
                    attention_mask = None
                else:
                    input_ids, attention_mask, position_ids = args
                    inbatch_pack_offset = None
            elif len(args) == 2:
                if self.use_mem_eff_attn:
                    input_ids, inbatch_pack_offset = args
                    position_ids, attention_mask = None, None
                    inbatch_pack_offset.stop_gradient = True
                else:
                    input_ids, attention_mask = args
                    position_ids = None
                    inbatch_pack_offset = None

        else:
            input_ids = args
            attention_mask, position_ids, inbatch_pack_offset = None, None, None

        if position_ids is not None:
            position_ids.stop_gradient = True

        emb = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)

        if self.config.multi_token_pred_depth > 0:
            if self.config.enable_mtp_magic_send:
                emb = emb[:, : -self.config.multi_token_pred_depth, :]
                if self.sequence_parallel:
                    emb = emb.reshape([-1, emb.shape[-1]])
                    emb = ScatterOp.apply(emb)
            else:
                inputs_embeds_extra = emb[
                    :, -self.config.multi_token_pred_depth :, :
                ]  # [B, S, D]
                inputs_embeds = emb[:, : -self.config.multi_token_pred_depth, :]
                inputs_embeds_ori = inputs_embeds
                batch_size, seq_length, _ = inputs_embeds.shape

                if self.sequence_parallel:
                    inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
                    inputs_embeds = ScatterOp.apply(inputs_embeds)
                mtp_emb_res = [inputs_embeds]
                for depth in range(self.config.multi_token_pred_depth):
                    inputs_embeds_mtp = paddle.concat(
                        [
                            inputs_embeds_ori[:, (depth + 1) :, :],
                            inputs_embeds_extra[:, : (depth + 1), :],
                        ],
                        axis=1,
                    )
                    if self.sequence_parallel:
                        inputs_embeds_mtp = inputs_embeds_mtp.reshape(
                            [-1, inputs_embeds_mtp.shape[-1]]
                        )
                        inputs_embeds_mtp = ScatterOp.apply(inputs_embeds_mtp)
                    mtp_emb_res.append(inputs_embeds_mtp)
                res = paddle.concat(mtp_emb_res)
                return [res]
        else:
            if self.sequence_parallel:
                emb = emb.reshape([-1, emb.shape[-1]])
                emb = ScatterOp.apply(emb)

        if attention_mask is not None:
            batch_size, seq_length = input_ids.shape
            attention_mask = ErnieModel._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), 0, emb.dtype
            )
            attention_mask.stop_gradient = True

        ret = (emb,)
        if attention_mask is not None:
            ret += (attention_mask.clone(),)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if inbatch_pack_offset is not None:
            ret += (inbatch_pack_offset.clone(),)
        if (
            self.config.multi_token_pred_depth > 0
            and not self.config.enable_mtp_magic_send
        ):
            ret += (input_ids,)
            assert len(ret) == 2, "mtp only support one input which is input_ids"
        if len(ret) == 1:
            ret = ret[0]
        return ret


class MTPEmbeddingPipe(ErnieEmbeddingPipe):
    """Extends ErnieEmbeddingPipe to forward attention_mask through the pipeline."""

    def __init__(self, config):
        super(MTPEmbeddingPipe, self).__init__(config)

    @property
    def embedding_weight(self):
        return getattr(self.embed_tokens, "weight")

    def forward(self, args):
        """
        :param args: Tuple or Tensor, containing the inputs to the module.
                     If it's a tuple, then it should contain at least three elements,
                     i.e. input_ids, attention_mask and position_ids.
                     In this case, the first three elements will be used as the arguments for forward.
                     The order of these elements is defined by `forward` method.

        :return: A Tensor with shape [batch_size * beam_width, sequence_length, hidden_dim].
                  This is the output of the model when given the specified input.

        """
        assert (
            self.config.enable_mtp_magic_send
        ), "MTPEmbedding can only be added into model only support enable_mtp_magic_send=True"

        global input_ids_for_mtp
        assert len(input_ids_for_mtp) > 0, "input_ids for mtp is empty"
        hidden_states = args[0]
        input_ids = input_ids_for_mtp.popleft()
        input_embeds = self.embed_tokens(input_ids).astype(
            self.embed_tokens.weight.dtype
        )

        return (hidden_states, input_embeds)


class EmptyLayer(nn.Layer):
    """ "EmptyLayer"""

    def __init__(self):

        super().__init__()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor with shape [B, C, T, V].

        Returns:
            Tensor: Output tensor with the same size as input.
        """
        return x


class ErnieDecoderLayerPipe(ErnieDecoderLayer):
    """_summary_

    Args:
        ErnieDecoderLayer (_type_): _description_
    """

    def __init__(self, config, layer_idx, use_full_recompute=False):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.use_full_recompute = use_full_recompute
        logger.info(f"using pp full recompute={use_full_recompute}")
        self.use_mem_eff_attn = config.use_mem_eff_attn

    def forward(self, args):
        """
        Args:
            args (tuple or Tensor): The input arguments to the layer.

        Returns:
            Union[Tuple, Tensor]: A tuple of outputs from the layer.
                If `use_moe` is set to False, a single Tensor will be returned instead.

        """
        if (
            self.config.multi_token_pred_depth > 0
            and not self.config.enable_mtp_magic_send
        ):
            res = args[0]
            tensor_list = paddle.split(res, self.config.multi_token_pred_depth + 1)
            inputs_embeds = tensor_list[-self.config.multi_token_pred_depth :]
            args = tuple(tensor_list[: -self.config.multi_token_pred_depth])
        else:
            res = None

        if isinstance(args, tuple):
            if len(args) == 3:
                if self.config.skip_align_position_id and self.use_mem_eff_attn:
                    hidden_states, position_ids, inbatch_pack_offset = args
                    attention_mask = None
                else:
                    hidden_states, attention_mask, position_ids = args
                    inbatch_pack_offset = None
            elif len(args) == 2:
                if self.use_mem_eff_attn:
                    hidden_states, inbatch_pack_offset = args
                    position_ids, attention_mask = None, None
                    inbatch_pack_offset.stop_gradient = True
                else:
                    hidden_states, attention_mask = args
                    position_ids, inbatch_pack_offset = None, None
            elif len(args) == 1:
                (hidden_states,) = args
                attention_mask, position_ids, inbatch_pack_offset = None, None, None
        else:
            hidden_states = args
            attention_mask, position_ids, inbatch_pack_offset = None, None, None

        if position_ids is not None:
            position_ids.stop_gradient = True

        if attention_mask is not None:
            attention_mask.stop_gradient = True
        if self.training and self.use_full_recompute:
            decoderlayer_act_offload_settings = self.config.get(
                "decoderlayer_act_offload_settings", {"type": "", "value": ""}
            )
            setting_type = decoderlayer_act_offload_settings["type"]
            offload_value = decoderlayer_act_offload_settings["value"]
            offload_kwargs = {}
            if "mod" == setting_type:
                assert isinstance(offload_value, (list, tuple))
                v1, v2 = offload_value
                offload_kwargs["offload_indices"] = (
                    [0] if self.layer_idx % v1 == v2 else []
                )
            elif "layer_idxs" == setting_type:
                offload_kwargs["offload_indices"] = (
                    [0] if self.layer_idx in offload_value else []
                )

            if offload_kwargs.get("offload_indices", []) and res is not None:
                inplace_offload(res)

            recompute_func = te_recompute if self.config.use_fp8 else recompute
            ret = recompute_func(
                super().forward,
                hidden_states,
                attention_mask,
                position_ids,
                None,  # token-type
                False,  # output-attention
                None,  # past-kv-cache
                False,  # use-cache
                inbatch_pack_offset,
                False,  # output_gate_logits
                **offload_kwargs,
            )
        else:
            ret = super().forward(
                hidden_states,
                attention_mask,
                position_ids,
                None,  # token-type
                False,  # output-attention
                None,  # past-kv-cache
                False,  # use-cache
                inbatch_pack_offset,
                False,  # output_gate_logits
            )
        if isinstance(ret, paddle.Tensor):
            ret = (ret,)
        if attention_mask is not None:
            ret += (attention_mask.clone(),)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if inbatch_pack_offset is not None:
            ret += (inbatch_pack_offset.clone(),)
        if len(ret) == 1:
            (ret,) = ret
        if self.config.multi_token_pred_depth > 0:
            if self.config.enable_mtp_magic_send:
                ret = (ret,)
            else:
                ret = (paddle.concat([ret, *inputs_embeds]),)
        return ret


class RMSNormPipe(RMSNorm):
    """ "_summary_"

    Args:
        RMSNorm (_type_): _description_
    """

    def __init__(self, config):

        super().__init__(config)
        self.use_moe = config.use_moe
        mark_as_sequence_parallel_parameter(self.weight)

    def forward(self, args):
        """
        Args:
        - use_moe: bool, whether to use MoE in the model.

        Returns:
        - output: torch.Tensor. The output tensor of the model after processing the inputs.

        """
        if self.config.multi_token_pred_depth > 0:
            if self.config.enable_mtp_magic_send:
                assert (
                    len(args) == self.config.multi_token_pred_depth + 1
                ), "the length is not valid in mtp"
                mtp_outputs = []
                for hidden_states in args:
                    mtp_outputs.append(super().forward(hidden_states))
                return mtp_outputs
            else:
                tensor_list = paddle.split(
                    args[0], self.config.multi_token_pred_depth + 1
                )
                mtp_outputs = []
                for hidden_states in tensor_list:
                    mtp_outputs.append(super().forward(hidden_states))
                return mtp_outputs
        else:
            if self.use_moe:
                hidden_states = args[:1]
            if isinstance(args, tuple):
                if len(args) == 3:
                    hidden_states, attention_mask, position_ids = args
                elif len(args) == 2:
                    hidden_states, attention_mask = args
            else:
                hidden_states = args
            hidden_states = super().forward(hidden_states)
            return hidden_states


class LayerNormPipe(LayerNorm):
    """_summary_

    Args:
        LayerNorm (_type_): _description_
    """

    def __init__(self, config):

        super().__init__(config)
        self.use_moe = config.moe_num_experts > 0
        mark_as_sequence_parallel_parameter(self.weight)
        mark_as_sequence_parallel_parameter(self.bias)

    def forward(self, args):
        """

        def forward(self, *args):
        """
        if self.config.multi_token_pred_depth > 0:
            if self.config.enable_mtp_magic_send:
                assert (
                    len(args) == self.config.multi_token_pred_depth + 1
                ), "the length is not valid in mtp"
                mtp_outputs = []
                for hidden_states in args:
                    mtp_outputs.append(super().forward(hidden_states))
                return mtp_outputs
            else:
                tensor_list = paddle.split(
                    args[0], self.config.multi_token_pred_depth + 1
                )
                mtp_outputs = []
                for hidden_states in tensor_list:
                    mtp_outputs.append(super().forward(hidden_states))
                return mtp_outputs
        else:
            if self.use_moe:
                hidden_states = args[0]

            if isinstance(args, tuple):
                if len(args) == 3:
                    hidden_states, attention_mask, position_ids = args
                elif len(args) == 2:
                    hidden_states, attention_mask = args
                elif len(args) == 1:
                    hidden_states = args[0]
            else:
                hidden_states = args
            hidden_states = super().forward(hidden_states)
        return hidden_states


class ErnieMoELMHeadPipe(ErnieMoELMHead):

    @property
    def embedding_weight(self):
        """embedding_weight property"""
        return getattr(self, "weight")

    def forward(self, args):
        if self.config.multi_token_pred_depth > 0:
            logits = list()
            for _hidden_states in args:
                logits.append(super().forward(_hidden_states))
            return logits
        hidden_states = args
        logits = super().forward(hidden_states)
        return logits


class MTPLayer(nn.Layer):
    """_summary_

    Args:
        MTPLayer (_type_): _description_
    """

    def __init__(self, config):

        super().__init__()
        config = copy.deepcopy(config)
        self.config = config
        if self.config.use_recompute_mtp:
            self.config.use_recompute = False
        assert (
            self.config.multi_token_pred_depth > 0
        ), "Adding MTPLayer must assign value to multi_token_pred_depth"

        self.mtp_block = paddle.nn.LayerList(
            [
                ErnieDecoderLayer(config, layer_idx)
                for layer_idx in range(self.config.multi_token_pred_depth)
            ]
        )
        Norm = RMSNorm if config.use_rmsnorm else LayerNorm
        if not config.use_rmsnorm and config.fuse_ln:
            Norm = FusedLayerNorm
        self.mtp_hidden_norm = paddle.nn.LayerList(
            [Norm(config) for _ in range(self.config.multi_token_pred_depth)]
        )
        self.mtp_emb_norm = paddle.nn.LayerList(
            [Norm(config) for _ in range(self.config.multi_token_pred_depth)]
        )

        LinearFN = (
            paddle.incubate.nn.FusedLinear if config.fuse_linear else paddle.nn.Linear
        )
        self.mtp_linear_proj = paddle.nn.LayerList(
            [
                LinearFN(
                    self.config.hidden_size * 2,
                    self.config.hidden_size,
                    bias_attr=config.use_bias,
                )
                for _ in range(self.config.multi_token_pred_depth)
            ]
        )
        if config.sequence_parallel or config.submatrix_parallel:
            for mtp_linear in self.mtp_linear_proj:
                mark_as_sequence_parallel_parameter(mtp_linear.weight)
                if config.use_bias:
                    mark_as_sequence_parallel_parameter(mtp_linear.bias)

    def forward(self, args):
        """forward"""

        def custom_forward(*inputs):
            """custom_forward function"""
            return self.forward_impl(*inputs)

        if self.config.use_recompute_mtp:
            return recompute(custom_forward, *args)
        else:
            return custom_forward(*args)

    def forward_impl(self, *args):
        """forward_impl"""
        if self.config.enable_mtp_magic_send:
            assert isinstance(args, tuple), "Input for MTPLayer must be tuple"
            hidden_states, inputs_embeds = args
            inputs_embeds_extra = inputs_embeds[
                :, -self.config.multi_token_pred_depth :, :
            ]  # [B, S, D]
            inputs_embeds = inputs_embeds[:, : -self.config.multi_token_pred_depth, :]
            inputs_embeds_ori = inputs_embeds
        else:
            res = args[0]
            tensor_list = paddle.split(res, self.config.multi_token_pred_depth + 1)
            hidden_states = tensor_list[0]
            inputs_embeds_cur_depth_list = tensor_list[1:]

        output_list = [hidden_states]
        for depth in range(self.config.multi_token_pred_depth):
            if self.config.enable_mtp_magic_send:
                # 构建输入向量
                inputs_embeds_cur_depth = paddle.concat(
                    [
                        inputs_embeds_ori[:, (depth + 1) :, :],
                        inputs_embeds_extra[:, : (depth + 1), :],
                    ],
                    axis=1,
                )

                if self.config.sequence_parallel or self.config.submatrix_parallel:
                    inputs_embeds_cur_depth = inputs_embeds_cur_depth.reshape(
                        [-1, inputs_embeds_cur_depth.shape[-1]]
                    )
                    inputs_embeds_cur_depth = ScatterOp.apply(inputs_embeds_cur_depth)
            else:
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
            layer_outputs = decoder_layer(
                inputs_embeds_cur_depth,
                None,  # attention_mask
                None,  # position_ids
                None,  # token-type
                False,  # output-attention
                None,  # past-kv-cache
                False,  # use-cache
                None,  # inbatch_pack_offset,
                False,  # output_gate_logits
            )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            output_list.append(hidden_states)

        if self.config.enable_mtp_magic_send:
            return tuple(output_list)
        else:
            res = paddle.concat(output_list)
            return (res,)


class ErniePretrainingCriterionPipe(ErniePretrainingCriterion):
    """_summary_

    Args:
        ErniePretrainingCriterion (_type_): _description_
    """

    def __init__(self, config):
        super().__init__(config)

    def forward(self, logits, labels):
        """
        Args:
            logits (torch.Tensor | Tuple[torch.Tensor]): The output of classifier.
            labels (torch.LongTensor): Ground truth label of the image.

        Returns:
            torch.FloatTensor: Loss value.
            Dict[str, Union[float, List]]: A dictionary containing information about training process.
        """
        # self.info = {
        if self.config.multi_token_pred_depth > 0:
            mtp_logits = logits[1:]
            logits = logits[0]
            loss, loss_sum = super().forward(logits, labels, mtp_logits=mtp_logits)
            if not self.training:
                return loss_sum
            return loss
        loss, loss_sum = super().forward(logits, labels)
        if not self.training:
            return loss_sum
        return loss


class PipelinePretrainedModel(PretrainedModel):
    """_summary_

    Args:
        PretrainedModel (_type_): _description_
    """

    def __init__(self, config, *args, **kwargs):  # no call
        self.config = config
        super().__init__(config, *args, **kwargs)

    def init(self, config, *args, **kwargs):  # no call
        """Add a single layer to the sequential module."""
        self._sequential_layers = []
        self._pipeline_name_mapping = None
        self._pp_to_single_mapping = None

    def add_sequential_layer(self, layer_desc, name_prefix=""):

        self._sequential_layers.append(
            {"layer": layer_desc, "name_prefix": name_prefix}
        )

    def get_sequential_layers(self):

        return [x["layer"] for x in self._sequential_layers]

    def get_sequential_name_prefixs(self):

        return {
            str(index): x["name_prefix"]
            for index, x in enumerate(self._sequential_layers)
        }

    def get_shardlayer_prefix(self, name_splited):

        shared_layer_names = {
            s.layer_name for s in self._layers_desc if isinstance(s, SharedLayerDesc)
        }
        assert (
            name_splited[1] in shared_layer_names
        ), f"The shared layer name {name_splited[1]} must be in prefixes!"
        shared_layer_key = name_splited[1]
        for idx, layer in enumerate(self._layers_desc):
            if (
                isinstance(layer, SharedLayerDesc)
                and layer.layer_name == shared_layer_key
            ):
                if self.get_stage_from_index(idx) == self._stage_id:
                    return self.get_sequential_name_prefixs()[str(idx)]

        # the prefix must be in the current stage, else raise error
        raise ValueError(
            f"The shared layer {shared_layer_key} must be in the current stage!"
        )

    def _set_pipeline_name_mapping(self, mappings=None):

        if mappings is not None:
            self._pipeline_name_mapping = mappings
        else:
            single_to_pp_mapping = {}
            pp_to_single_mapping = {}

            state_dict_keys = list(super().state_dict().keys())
            first_key = ""
            for k in state_dict_keys:
                if "shared_layers" not in k:
                    first_key = k
                    break
            first_key = first_key.split(".")
            use_virtual_pp_degree = first_key[0].isdigit() and first_key[1].isdigit()

            prefixes = self.get_sequential_name_prefixs()
            for k in state_dict_keys:
                name_splited = k.split(".")
                if use_virtual_pp_degree:
                    if name_splited[0].isdigit():
                        if name_splited[1].isdigit():
                            idx = str(int(name_splited[0]) + int(name_splited[1]))
                            single_name = [prefixes[idx]]
                            single_name.extend(name_splited[2:])
                        else:
                            single_name = [prefixes[str(len(prefixes) - 1)]]
                            single_name.extend(name_splited[2:])
                            logger.warning(
                                f"Please check! we treat this key as last layer, get {k}, \
                                        set origin name as {'.'.join(single_name)}"
                            )
                    elif name_splited[0] == "shared_layers":
                        single_name = [self.get_shardlayer_prefix(name_splited)]
                        single_name.extend(name_splited[2:])
                    else:
                        single_to_pp_mapping[k] = k
                        pp_to_single_mapping[k] = k
                        continue
                else:
                    idx = name_splited[0]
                    # for normal pp layer
                    if idx.isdigit():
                        # allow empty prefix
                        single_name = [] if prefixes[idx] == "" else [prefixes[idx]]
                        single_name.extend(name_splited[1:])
                    elif idx == "shared_layers":
                        single_name = [self.get_shardlayer_prefix(name_splited)]
                        single_name.extend(name_splited[2:])
                    else:
                        single_to_pp_mapping[k] = k
                        pp_to_single_mapping[k] = k
                        continue

                single_to_pp_mapping[".".join(single_name)] = k
                pp_to_single_mapping[k] = ".".join(single_name)

            self._pipeline_name_mapping = single_to_pp_mapping
            self._pp_to_single_mapping = pp_to_single_mapping

        return self._pipeline_name_mapping

    def _check_shared_model_state(self):
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()

        super_state_dict = super().state_dict()
        structure_name_to_tensor = {}
        for k, v in super_state_dict.items():
            k = self._pp_to_single_mapping[k]
            if k not in structure_name_to_tensor:
                structure_name_to_tensor[k] = v
            else:
                old_v = structure_name_to_tensor[k]
                assert old_v is v, f"Shared tensor with different structure name: {k}"

        missing_shared_keys = {}
        for k, v in self._pp_to_single_mapping.items():
            mapped_k = self._pipeline_name_mapping[v]
            if k != mapped_k:
                missing_shared_keys[k] = mapped_k
        return missing_shared_keys

    def state_dict(self, *args, **kwargs):

        state_dict = super().state_dict(*args, **kwargs)

        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()
        # assert len(self._pipeline_name_mapping) > 0, "The pipeline stage must have parameters!"

        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            state_dict[self._pp_to_single_mapping[k]] = v

        return state_dict

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
                TEFP8Linear,
                ErnieMoELMHead,
                nn.Embedding,
                NativeLinear,
                paddle.incubate.nn.FusedLinear,
            ),
        ):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            # logger.info(f'initializing pp:{type(layer)}')

            if isinstance(layer, RoundRobinGate):
                return

            is_moe = getattr(layer.weight, "no_sync", False)
            with rng_tracker("local_seed" if is_moe else "model_parallel_rng"):
                dtype = paddle.get_default_dtype()  # layer.weight.dtype  #
                # dtype = str(dtype).replace("paddle.", "")
                paddle.set_default_dtype("float32")
                if isinstance(layer, TEFP8Linear):
                    layer.weight.set_value(
                        paddle.transpose(
                            paddle.randn(layer.weight.shape[::-1], dtype=dtype).scale(
                                self.config.initializer_range
                            ),
                            perm=[1, 0],
                        )
                    )
                else:
                    layer.weight.set_value(
                        paddle.randn(layer.weight.shape, dtype=dtype).scale(
                            self.config.initializer_range
                        )
                    )
                paddle.set_default_dtype(dtype)
                logger.info(
                    f"dist-init-fc: shape={layer.weight.shape}, range={self.config.initializer_range}, "
                    f"dtype={layer.weight.dtype} "
                    f'type={type(layer)},norm={layer.weight.astype("float32").norm().item()},is_moe={is_moe}'
                )

        elif isinstance(layer, Top2Gate):
            if not hasattr(
                layer, "weight"
            ):  # no weight to initialie : moe-round-robin-gate
                return
            with rng_tracker("model_parallel_rng"):
                dtype = paddle.get_default_dtype()  # layer.weight.dtype  #
                paddle.set_default_dtype("float32")
                moe_inter = self.config.moe_intermediate_size
                moe_num_experts = self.config.moe_num_experts
                if isinstance(moe_inter, (list, tuple)):
                    moe_inter = moe_inter[0]
                if isinstance(moe_num_experts, (list, tuple)):
                    moe_num_experts = moe_num_experts[0]
                if self.config.moe_group_experts:
                    layer.weight.set_value(
                        paddle.randn(
                            layer.weight.shape, dtype=layer.weight.dtype
                        ).scale(self.config.initializer_range)
                    )
                else:
                    layer.weight.set_value(
                        paddle.randn(
                            [self.config.hidden_size, moe_num_experts],
                            dtype="float32",
                        ).scale(self.config.initializer_range)
                    )
                logger.info(
                    f"dist-init-moe_gate: shape={layer.weight.shape}, dtype={layer.weight.dtype} "
                    f"range={self.config.initializer_range},type={type(layer)}, "
                    f'norm={layer.weight.astype("float32").norm().item()}'
                )
                if isinstance(self.config.moe_num_experts, (tuple, list)):
                    for i in range(1, len(self.config.moe_num_experts)):
                        layer_weight = getattr(layer, f"weight_{i}")
                        layer_weight.set_value(
                            paddle.randn(
                                layer_weight.shape, dtype=layer_weight.dtype
                            ).scale(self.config.initializer_range)
                        )
                        logger.info(
                            f"dist-init-moe_gate: shape={layer_weight.shape}, dtype={layer_weight.dtype} "
                            f"range={self.config.initializer_range},type={type(layer)}, "
                            f'norm={layer_weight.astype("float32").norm().item()}'
                        )
                paddle.set_default_dtype(dtype)

        elif isinstance(layer, RotaryEmbedding):
            head_dim = self.config.hidden_size // self.config.num_attention_heads
            inv_freq = 1.0 / (
                layer.base ** (np.arange(0, head_dim, 2).astype("float32") / head_dim)
            )
            # self.register_buffer("inv_freq", inv_freq.cast(dtype))

            # higher acc using float32
            t = np.arange(layer.max_position_embeddings, dtype="float32")
            freqs = np.einsum("i,j->ij", t, inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = np.concatenate([freqs, freqs], axis=-1)
            # [bs, seqlen, nhead, head_dim]
            cos_cached = np.cos(emb)[:, :]
            sin_cached = np.sin(emb)[:, :]

            layer.cos_cached.set_value(cos_cached)
            layer.sin_cached.set_value(sin_cached)


def get_pp_vp_split_layers(config):
    hcg = get_hcg()
    pp_size = max(hcg.get_pipe_parallel_world_size(), 1)
    vp_size = max(config.virtual_pp_degree, 1)
    layer_num = config.num_hidden_layers
    selective_no_recompute_num = config.selective_no_recompute_num

    no_recompute_layer_num = []
    if selective_no_recompute_num == 0:
        return set(no_recompute_layer_num)

    assert layer_num % (pp_size * vp_size) == 0, (
        "layer_num must be divisible by pp_size * vp_size,"
        " but got layer_num: {}, pp_size: {}, vp_size: {}".format(
            layer_num, pp_size, vp_size
        )
    )

    chunk_size = layer_num // (pp_size * vp_size)
    chunk_list = [
        list(range(i * chunk_size, (i + 1) * chunk_size))
        for i in range(pp_size * vp_size)
    ]

    stage_chunk_list = [[] for _ in range(pp_size)]
    for i in range(pp_size * vp_size):
        stage_chunk_list[i % pp_size].append(chunk_list[i])

    if config.use_recompute_attn:
        logger.error(
            "selective recompute only support full recompute now, please set use_recompute_attn to False"
        )

    for i in range(pp_size):
        no_recompute_layer_num.extend(stage_chunk_list[i][-selective_no_recompute_num:])

    # trick to convert to 1D list
    return set(sum(no_recompute_layer_num, []))


class ErnieMoEForCausalLMPipe(PipelinePretrainedModel, PipelineLayer):

    config_class = ErnieMoEConfig

    _get_tensor_parallel_mappings = ErniePretrainedModel._get_tensor_parallel_mappings

    ErnieEmbeddingPipeClass = ErnieEmbeddingPipe
    ErnieDecoderLayerPipeClass = ErnieDecoderLayerPipe
    MTPEmbeddingPipeClass = MTPEmbeddingPipe
    MTPLayerClass = MTPLayer
    RMSNormPipeClass = RMSNormPipe
    LayerNormPipeClass = LayerNormPipe
    ErnieMoELMHeadPipeClass = ErnieMoELMHeadPipe

    @classmethod
    def _prepare_pipeline_inputs_func(cls, data):

        inputs = tuple(
            [
                [d[k] for d in data]
                for k in [
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "inbatch_pack_offset",
                ]
                if k in data[0]
            ]
        )
        if len(inputs) == 1:
            inputs = inputs[0]
        # logger.debug([len(i) for i in inputs])
        labels = [d["labels"] for d in data]
        return inputs, labels

    def __init__(
        self,
        config,
    ):
        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(
            f"change initializer-range from {config.initializer_range} to {new_initializer_range}"
        )
        config.initializer_range = new_initializer_range

        if config.moe_group == "mp":
            assert config.sequence_parallel

        if config.moe_group in {"mp", "model", "tp", "mpdp", "ep"}:
            # assert config.sequence_parallel
            logger.info(
                f"disable FFN tensor model parallel, moe-group={config.moe_group}"
            )
            config.disable_ffn_model_parallel = True

        hcg = get_hcg()
        if config.moe_group != "ep":
            config.moe_group = _parse_moe_group(config.moe_group)
        else:
            config.moe_group = hcg.get_expert_parallel_group()
        config.moe_world_size = dist.get_world_size(config.moe_group)
        if config.moe_world_size < 0:
            config.moe_world_size = 1
        config.moe_rank = dist.get_rank(config.moe_group)

        self.config = config

        hcg = get_hcg()
        tensor_parallel_degree = max(hcg.get_model_parallel_world_size(), 1)
        tensor_parallel_rank = max(hcg.get_model_parallel_rank(), 0)
        logger.info(f"using vpp={config.virtual_pp_degree}")
        if config.sequence_parallel:
            logger.info(f"using sequence_parallel, input seqlen={config.seqlen}")
            assert config.seqlen is not None
            assert (
                config.tensor_parallel_degree > 1
            ), f"sequence-parallel needs mp>1, got mp={config.tensor_parallel_degree}"

        config.tensor_parallel_degree = tensor_parallel_degree
        config.tensor_parallel_rank = tensor_parallel_rank
        PipelinePretrainedModel.init(self, config=config)

        if config.pp_no_recompute_layer is not None:
            no_recompute_layers = config.pp_no_recompute_layer
        else:
            no_recompute_layers = get_pp_vp_split_layers(config)
        logger.info(f"use no_recompute_layers: {no_recompute_layers}")

        def _need_full_recompute(layer_idx):
            return layer_idx not in no_recompute_layers and config.use_recompute

        insert_empty_layer = config.insert_empty_layer
        if len(insert_empty_layer) > 0:
            assert (
                min(insert_empty_layer) >= 0
            ), "cannot insert empty layer as first layer of the model"
            assert (
                max(insert_empty_layer) < config.num_hidden_layers
            ), "empty layers location exceed the num layers"
        logger.info(f"use insert_empty_layer: {insert_empty_layer}")

        if config.multi_token_pred_depth == 0:
            if config.tie_word_embeddings:
                self.add_sequential_layer(
                    SharedLayerDesc(
                        key="embed_weight_share",
                        layer_func=self.ErnieEmbeddingPipeClass,
                        shared_weight_attr="embedding_weight",
                        config=config,
                    ),
                    "ernie",
                )
            else:
                self.add_sequential_layer(
                    LayerDesc(self.ErnieEmbeddingPipeClass, config=config), "ernie"
                )
        else:
            if config.enable_mtp_magic_send:
                self.add_sequential_layer(
                    SharedLayerDesc(
                        key="embed_weight_share",
                        layer_func=self.ErnieEmbeddingPipeClass,
                        shared_weight_attr="embedding_weight",
                        config=config,
                    ),
                    "ernie.embed",
                )
            else:
                self.add_sequential_layer(
                    LayerDesc(self.ErnieEmbeddingPipeClass, config=config), "ernie"
                )

        num_empty_layers = (
            config.remove_tail_layer if isinstance(config.remove_tail_layer, int) else 1
        )
        for i in range(config.num_hidden_layers - num_empty_layers):
            self.add_sequential_layer(
                LayerDesc(
                    self.ErnieDecoderLayerPipeClass,
                    config=create_skip_config_for_refined_recompute(i, config),
                    layer_idx=i,
                    use_full_recompute=_need_full_recompute(i),
                ),
                f"ernie.layers.{i}",
            )
            if i in insert_empty_layer:
                self.add_sequential_layer(
                    LayerDesc(
                        EmptyLayer,
                    ),
                    f"empty.layers.{i}",
                )

        if config.multi_token_pred_depth > 0:
            if config.enable_mtp_magic_send:
                self.add_sequential_layer(
                    SharedLayerDesc(
                        key="embed_weight_share",
                        layer_func=self.MTPEmbeddingPipeClass,
                        shared_weight_attr="embedding_weight",
                        config=config,
                    ),
                    "embed_share",
                )
            self.add_sequential_layer(
                LayerDesc(self.MTPLayerClass, config=config), "ernie"
            )
            num_empty_layers = num_empty_layers - config.multi_token_pred_depth

        if config.remove_tail_layer:
            for n in range(num_empty_layers):
                self.add_sequential_layer(
                    LayerDesc(
                        EmptyLayer,
                    ),
                    f"empty.layers.{n}",
                )
        else:
            for n in range(num_empty_layers):
                self.add_sequential_layer(
                    LayerDesc(
                        self.ErnieDecoderLayerPipeClass,
                        config=create_skip_config_for_refined_recompute(i, config),
                        layer_idx=i,
                        use_full_recompute=_need_full_recompute(i),
                    ),
                    f"ernie.layers.{n + config.num_hidden_layers - num_empty_layers}",
                )

        i = config.num_hidden_layers
        if i in insert_empty_layer:
            self.add_sequential_layer(
                LayerDesc(
                    EmptyLayer,
                ),
                f"empty.layers.{i}",
            )

        self.add_sequential_layer(
            LayerDesc(
                (
                    self.RMSNormPipeClass
                    if config.use_rmsnorm
                    else self.LayerNormPipeClass
                ),
                config=config,
            ),
            "ernie.norm",
        )

        if config.tie_word_embeddings:
            self.add_sequential_layer(
                SharedLayerDesc(
                    key="embed_weight_share",
                    layer_func=self.ErnieMoELMHeadPipeClass,
                    shared_weight_attr="embedding_weight",
                    config=config,
                ),
                "lm_head",
            )
        else:
            self.add_sequential_layer(
                LayerDesc(self.ErnieMoELMHeadPipeClass, config=config), "lm_head"
            )

        recompute_interval = 0

        seg_method = "layer:ErnieDecoderLayer|EmptyLayer|MTPLayer"
        if config.num_hidden_layers % get_hcg().topology().get_dim_size("pipe") != 0:
            seg_method = "uniform"
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
                "partition": False,
            },
            num_virtual_pipeline_stages=config.virtual_pp_degree,
        )

    def forward(self, *args, **kwargs):

        with fp8_autocast(self.config.use_fp8, self.config.fp8_configs["recipe"]):
            return super().forward(*args, **kwargs)

    def get_loss_fn(self, config):
        """get_loss_fn"""
        return ErniePretrainingCriterionPipe(config)

    def rename_model_params(self, func):
        if self.config.virtual_pp_degree == 1:
            _layers = iter(self.run_function)
        else:
            _layers = (cc for c in self._model_chunks for cc in c.run_function)
        func(self.config, _layers)

    def _post_init(self, original_init, *args, **kwargs):

        super()._post_init(self, original_init, *args, **kwargs)
        with paddle.no_grad():
            for i, layer in self._sub_layers.items():
                if isinstance(layer, ErnieDecoderLayer):
                    layer_id = int(i) - 1  # skip_vocab
                    factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
                    logger.info(
                        f"using post init div: layer[{layer_id}].factor={factor}"
                    )
                    if self.config.use_linear_residual_norm_recompute:
                        layer.fused_linear_add_norm.linear_weight.scale_(factor)
                    else:
                        layer.self_attn.o_proj.weight.scale_(factor)
                    if isinstance(
                        layer.mlp,
                        (
                            MOELayer,
                            MOELayerSizeAll2All,
                        ),
                    ):
                        for e in layer.mlp.experts:
                            if isinstance(e, ErnieMLP):
                                e.down_proj.weight.scale_(factor)
                    else:
                        layer.mlp.down_proj.weight.scale_(factor)

        if not self.config.disable_pipeline_warmup:
            self.pipeline_warmup()
        self.aadiff_check()

    def pipeline_warmup(self):
        """pipeline_warmup"""
        # warmup moe layer for pp
        ori_state = self.training
        try:
            input = None
            hcg = get_hcg()
            mp_size = hcg.get_model_parallel_world_size()
            with paddle.no_grad():
                self.eval()
                if hcg.get_stage_id() == 0:
                    input = paddle.randint(
                        0, self.config.vocab_size, [1, self.config.max_sequence_length]
                    )
                else:
                    input = paddle.randn(
                        [
                            self.config.max_sequence_length // mp_size,
                            self.config.hidden_size,
                        ]
                    )
                _ = self.forward(input, chunk_id=0)
            logger.info(f"warmup moe layer for pp: {input.shape} successfully")
        except Exception as e:
            logger.info(f"failed to warmup moe layer for pp: {e}")
        finally:
            if ori_state:
                self.train()
            if isinstance(global_training_logs, dict):
                global_training_logs.clear()
            else:
                global_training_logs.reset()

    def aadiff_check(self):
        """aadiff_check"""
        if pipeline_bubble_hooks_ is None:
            return

        snapshot = {}
        input = None

        def aadiff_check(bubble_id):
            nonlocal input
            hcg = get_hcg()
            mp_size = hcg.get_model_parallel_world_size()
            with paddle.no_grad():
                if bubble_id == 0:
                    if hcg.get_stage_id() == 0:
                        input = paddle.randint(
                            0,
                            self.config.vocab_size,
                            [1, self.config.max_sequence_length],
                        )
                    else:
                        input = paddle.randn(
                            [
                                self.config.max_sequence_length // mp_size,
                                self.config.hidden_size,
                            ]
                        )

                global_training_logs.take_snapshot()
                ori_state = self.training
                self.eval()
                output = self.forward(input, chunk_id=0)
                if ori_state:
                    self.train()
                global_training_logs.restore_snapshot()

            res = output[0] if isinstance(output, tuple) else output

            if bubble_id == 0:
                snapshot[bubble_id] = res
            elif bubble_id > 0:
                assert paddle.equal_all(
                    snapshot[0], res
                ), f"{AADIFF_ERROR}. Check AADiff Error at rank {paddle.distributed.get_rank()}."

        def register_hooks():
            hcg = get_hcg()
            pp_size = hcg.get_pipe_parallel_world_size()
            for bubble_id in range(pp_size * 2 - 2):
                pipeline_bubble_hooks_.register_hook(bubble_id, aadiff_check)

        if self.config.using_precision_check:
            register_hooks()
            logger.info("Open aadiff checker online successfully.")

    def state_dict(self, *args, **kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        state_dict = super().state_dict(*args, **kwargs)
        # moe state dict 转换成全局id.
        return moe_statedict_local_id_to_global(state_dict, self.config)

    def set_state_dict(self, state_dict, *args, **kwargs):
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()
        # assert len(self._pipeline_name_mapping) > 0, "The pipeline stage must have parameters!"

        layer_idxs = []
        if self.config.virtual_pp_degree == 1:
            _layers = iter(self.run_function)
        else:
            _layers = (cc for c in self._model_chunks for cc in c.run_function)

        for layer in _layers:
            if isinstance(layer, self.ErnieDecoderLayerPipeClass):
                layer_idxs.append(layer.layer_idx)
        logger.info(f"this pipeline stage has ErnieDecoderLayers: {layer_idxs}")
        if not self.parameters():  # all Empty layer
            logger.info("this pipe not need param, skip set state-dict")
            return {}, {}
        state_dict = moe_statedict_upcycle(
            state_dict,
            self.config,
            list(self.parameters())[0].dtype,
            self._get_tensor_parallel_mappings(self.config, is_split=False),
            self._get_tensor_parallel_mappings(self.config, is_split=True),
            layer_idxs,
        )
        state_dict = moe_ep2mp(
            state_dict,
            self.config,
            self._get_tensor_parallel_mappings(self.config, is_split=True),
        )
        state_dict = moe_statedict_cherry_pick(state_dict, self.config)
        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if k not in self._pipeline_name_mapping:
                continue
            state_dict[self._pipeline_name_mapping[k]] = v

        if not isinstance(self.config.moe_num_experts, (list, tuple)):
            logger.info("convert mm correction_bias to lm")
            state_dict = {
                k: v[:1] if "e_score_correction_bias" in k else v
                for k, v in state_dict.items()
            }

        missing_keys, mismatch_keys = super().set_state_dict(
            state_dict, *args, **kwargs
        )

        missing_shared_keys = self._check_shared_model_state()
        tmp_missing_keys = []
        for key in missing_keys:
            if (
                key in missing_shared_keys
                and missing_shared_keys[key] not in missing_keys
            ):
                continue
            tmp_missing_keys.append(key)
        missing_keys = tmp_missing_keys

        logger.info(f"moe_set_state_dict: {missing_keys}, {mismatch_keys}")
        return missing_keys, mismatch_keys
