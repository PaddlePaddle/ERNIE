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

import inspect
import logging
import numpy as np
from contextlib import contextmanager
from typing import Tuple, List, Optional
from functools import partial
from copy import deepcopy

import paddle
import paddle.nn.functional as F
import paddle.distributed as dist
from paddle import nn
from paddle.incubate.nn.functional import swiglu
from paddle.distributed.communication.group import Group
from paddle.distributed import fleet
from paddle import Tensor
from paddle.incubate.nn.functional import moe_combine, moe_gate_dispatch
from paddle.utils import unique_name
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker
from paddleformers.trainer.plugins.timer import get_timers
from paddleformers.transformers.moe_layer import dispatching, combining

from utils.training_utils import get_flatten_mesh, get_mesh, _reshard
from models.configuration import ErnieMoEConfig
from models.modeling import ErnieMLP

logger = logging.getLogger(__name__)


class MoEStatics(nn.Layer):
    """
    Stores MoE (Mixture of Experts) statistics
    and expert usage information.
    """

    def __init__(self, config, layer_idx):
        """
        Initialize MoE statistics tracking.
        Args:
            config: Model configuration containing MoE parameters
            layer_idx: Index of the MoE layer in the model
        """
        super().__init__()
        self._cast_to_low_precision = False
        self._cast_to_low_precison = False
        num_experts = (
            config.moe_num_experts[0]
            if config.multimodel_experts
            else config.moe_num_experts
        )
        if config.multimodel_experts:
            assert (
                len(set(config.moe_num_experts)) == 1
            ), f"assume expert group has same size, got: {config.moe_num_experts}"

        with paddle.utils.unique_name.guard(f"mm_layer_{layer_idx}_"):
            num_experts_groups = (
                len(config.moe_num_experts) if config.multimodel_experts else 1
            )
            p = self.create_parameter(
                shape=[num_experts_groups * num_experts],
                dtype="float32",
                is_bias=True,
                attr=paddle.ParamAttr(
                    name=paddle.utils.unique_name.generate("corr_bias")
                ),
            )
            p.stop_gradient = True
            self.e_score_correction_bias = p
            self.e_score_correction_bias.is_distributed = True
            p = paddle.zeros(
                shape=[num_experts_groups, num_experts],
                dtype="int64",
            )
            p.stop_gradient = True
            self.expert_usage = p


@contextmanager
def profile(name):
    """doc"""
    if get_timers() is not None:
        get_timers()(name).start()
    yield
    if get_timers() is not None:
        get_timers()(name).stop()


def combining_fused(x, combine_weights, scatter_index, hard_gate=False):
    """
    Args:
        x: Tensor[seq, dim]
        combine_weights: [s, k]
        scatter_index:  ** [k, s] **

    Returns:
        y: Tensor[s, dim]
    """
    if hard_gate:
        x_gatherd = F.embedding(scatter_index, x)
        return x_gatherd.squeeze(-2)
    ret = moe_combine(x, combine_weights, scatter_index)

    ret.stop_gradient = False
    return ret


class MOELayer(nn.Layer):
    def __init__(
        self,
        gate: nn.Layer,
        experts: List[nn.Layer],
        layer_idx,
        shared_experts: Optional[List[nn.Layer]] = None,
        group: Group = None,
        recompute=False,
        k=2,
        all_to_all_dropout=0,
        group_experts=False,
        moe_statics=None,
        config=None,
        ipp=0,
    ):
        super().__init__()
        self.config = config
        self.gate = gate
        self.layer_idx = layer_idx
        self.ipp = ipp
        self.recompute = recompute
        logger.info(f"using moe recompute={recompute}")
        for p in self.gate.parameters():
            p.is_gate = True
        if isinstance(experts, nn.LayerList):
            self.experts = experts
        else:
            logger.info(f"using fused experts, type={type(experts)}")
            self.experts = experts
        self.shared_experts = shared_experts

        self.group = group
        self.k = k
        self.all_to_all_dropout = all_to_all_dropout
        is_mp_moe = (
            hasattr(fleet.fleet, "_hcg")
            and group is fleet.get_hybrid_communicate_group().get_model_parallel_group()
        )
        is_dummy_moe = config.moe_world_size == 1

        for p in experts.parameters():
            p.expert = not (is_mp_moe or is_dummy_moe)
            p.no_sync = not (is_mp_moe or is_dummy_moe)
            logger.info(f"expert no-sync={p.no_sync}-{p.name}")
            if is_mp_moe or is_mp_moe:
                p.is_distributed = True

        self.world_size = config.moe_world_size
        if self.group in fleet.auto.get_mesh().dim_names:
            self.rank = fleet.auto.get_mesh().get_rank_by_dim_and_process_id(
                self.group, dist.get_rank()
            )
            if self.rank < 0:
                self.rank = 0
        else:
            self.rank = 0

        self.num_experts_per_group = len(self.experts)
        self.ep_group_num = config.moe_world_size
        self.num_local_experts = self.num_experts_per_group // self.ep_group_num

        self.moe_mesh_dim = 0 if config.moe_group == "dp" else 1
        self.dispatch_by_task = (
            hasattr(self.gate, "dispatch_by_task") and self.gate.dispatch_by_task
        )

        if self.dispatch_by_task:
            assert 0, "no supported, checkout earylier code"
            assert self.num_local_experts == 1

        self.input_preprocess = self.output_postprocess = None
        self.group_experts = group_experts
        self.use_correction_bias = moe_statics is not None
        self.moe_statics = moe_statics

    def fused_gate_logits_process(
        self, gate_logits, token_type_ids, offload_helper=None
    ):

        k = self.k
        experts_type_ids = self.gate.experts_type_ids
        use_hard_gate = self.config.moe_use_hard_gate
        max_prob = None

        if token_type_ids is not None and use_hard_gate:
            if offload_helper is None:
                offload_helper = dict()
                lm_mask = token_type_ids == 0
                is_lm = lm_mask.any()
                mm_mask = token_type_ids == 1
                is_mm = mm_mask.any()
                seq_lm = lm_mask.sum()
                seq_mm = mm_mask.sum()
                lm_mask = lm_mask.unsqueeze(1) & (experts_type_ids == 0).unsqueeze(0)
                mm_mask = mm_mask.unsqueeze(1) & (experts_type_ids == 1).unsqueeze(0)
                offload_helper["lm_mask"] = [lm_mask, is_lm, seq_lm]
                offload_helper["mm_mask"] = [mm_mask, is_mm, seq_mm]

            is_lm = offload_helper["lm_mask"][1]
            prob = paddle.zeros_like(gate_logits)
            if is_lm:
                lm_mask = offload_helper["lm_mask"][0]
                seq_lm_cpu = offload_helper["lm_mask"][2]
                lm_mask_nonzero = lm_mask.nonzero()
                lm_partial_gate_logits = gate_logits.gather_nd(lm_mask_nonzero).reshape(
                    [seq_lm_cpu, -1]
                )
                if self.group_experts:
                    lm_prob = self.gate.act(
                        lm_partial_gate_logits.reshape(
                            [lm_partial_gate_logits.shape[0], k, -1]
                        )
                    )
                    max_prob = lm_prob.max(-1, keepdim=True)
                    lm_prob /= max_prob
                else:
                    lm_prob = self.gate.act(lm_partial_gate_logits)
                prob = paddle.scatter_nd_add(prob, lm_mask_nonzero, lm_prob.flatten())
            is_mm = offload_helper["mm_mask"][1]
            if is_mm:
                mm_mask = offload_helper["mm_mask"][0]
                seq_mm_cpu = offload_helper["mm_mask"][2]
                mm_mask_nonzero = paddle.nonzero(mm_mask)
                mm_partial_gate_logits = gate_logits.gather_nd(mm_mask_nonzero).reshape(
                    [seq_mm_cpu, -1]
                )
                mm_prob = self.gate.act(mm_partial_gate_logits)
                prob = paddle.scatter_nd_add(prob, mm_mask_nonzero, mm_prob.flatten())
        else:
            if self.group_experts:
                prob = self.gate.act(gate_logits.reshape([gate_logits.shape[0], k, -1]))
                max_prob = prob.max(-1, keepdim=True)
                prob /= max_prob
                prob = prob.reshape([prob.shape[0], -1])
            else:
                prob = self.gate.act(gate_logits)
        return prob, max_prob

    def _calc_router_loss(
        self,
        dispatch_mask,
        gate_logits,
        gate_prob,
        num_experts,
        use_group,
        layer_idx,
        token_type=None,
        tokens_type_mask=None,
        dispatch_tokens_mask=None,
        prefix="",
    ):
        router_loss, l_aux, orthogonal_loss = 0.0, None, None
        if self.gate.config.moe_aux_loss_lambda:
            l_aux = self.gate._cal_aux_loss(
                gate_prob,
                dispatch_mask,
                num_experts,
                use_group,
                tokens_type_mask,
                dispatch_tokens_mask,
            )
            router_loss += self.gate.moe_aux_loss_lambda[token_type or 0] * l_aux
        else:
            router_loss += self.zero * gate_prob[0, 0]
        if self.gate.config.moe_orthogonal_loss_lambda:
            orthogonal_loss = self.gate._cal_orthogonal_loss(token_type, use_group)
            router_loss += (
                self.gate.moe_orthogonal_loss_lambda[token_type or 0] * orthogonal_loss
            )

        return router_loss

    def calc_router_loss_and_logging(
        self,
        router_loss,
        combine_weights,
        dispatch_mask,
        gate_logits,
        gate_prob,
        token_type_ids,
        dispatch_token_type_ids=None,
        offload_helper=None,
    ):

        use_fuse = isinstance(self.gate, (TopKGateFused))
        if use_fuse:
            assert gate_prob is not None
            if token_type_ids is not None and self.gate.config.moe_use_hard_gate:
                if not self.gate.weight.stop_gradient:
                    lm_tokens_mask = token_type_ids == 0
                    if offload_helper is not None:
                        is_lm = offload_helper["lm_mask"][1]
                    else:
                        is_lm = lm_tokens_mask.any()
                    if is_lm:
                        dispatch_tokens_mask = (
                            dispatch_token_type_ids == 0
                            if dispatch_token_type_ids is not None
                            else None
                        )
                        router_loss += self._calc_router_loss(
                            (
                                dispatch_mask[self.gate.experts_type_mask[0]]
                                if hasattr(self.gate, "experts_type_mask")
                                else dispatch_mask
                            ),
                            (
                                gate_logits[:, self.gate.experts_type_mask[0]]
                                if hasattr(self.gate, "experts_type_mask")
                                else gate_logits
                            ),
                            (
                                gate_prob[:, self.gate.experts_type_mask[0]]
                                if hasattr(self.gate, "experts_type_mask")
                                else gate_prob
                            ),
                            (
                                self.gate.num_experts_list[0]
                                if hasattr(self.gate, "num_experts_list")
                                else self.gate.num_experts_tensor
                            ),
                            self.group_experts,
                            self.layer_idx,
                            0,
                            lm_tokens_mask,
                            dispatch_tokens_mask,
                            prefix="lm",
                        )
                mm_tokens_mask = token_type_ids == 1
                if offload_helper is not None:
                    is_mm = offload_helper["mm_mask"][1]
                else:
                    is_mm = mm_tokens_mask.any()
                if is_mm:
                    dispatch_tokens_mask = (
                        dispatch_token_type_ids == 1
                        if dispatch_token_type_ids is not None
                        else None
                    )
                    router_loss += self._calc_router_loss(
                        dispatch_mask[self.gate.experts_type_mask[1]],
                        gate_logits[:, self.gate.experts_type_mask[1]],
                        gate_prob[:, self.gate.experts_type_mask[1]],
                        self.gate.num_experts_list[1],
                        False,
                        self.layer_idx,
                        1,
                        mm_tokens_mask,
                        dispatch_tokens_mask,
                        prefix="mm",
                    )

            else:
                router_loss += self._calc_router_loss(
                    dispatch_mask,
                    gate_logits,
                    gate_prob,
                    self.gate.num_experts_tensor,
                    self.group_experts,
                    self.layer_idx,
                )

        return router_loss

    def forward_experts(self, dispatched_input):
        """
        call experts sequently
        Args:
            dispatched_input: Tensor[num_experts, capacity, dim]
        Returns:
            expert_output: Tensor[num_experts, capacity, dim]
        """
        assert isinstance(self.experts, nn.LayerList)
        if self.config.moe_group == "mp":
            local_input_list = dist.auto_parallel.api.moe_sub_mesh_tensors(
                dispatched_input,
                get_mesh(self.ipp),
                self.moe_mesh_dim,
                [dist.Shard(2), dist.Shard(0)],
            )

            assert len(self.experts) % len(local_input_list) == 0, (
                "num of experts must be divided by num of ep_group, "
                f"but got {len(self.experts)} and {len(local_input_list)}"
            )
            expert_group_outputs = []
            for i_ep_group, local_input in enumerate(local_input_list):
                chunks = local_input.unbind(1)
                experts = self.experts[
                    i_ep_group
                    * self.num_local_experts : (i_ep_group + 1)
                    * self.num_local_experts
                ]
                ep_output = []
                assert len(experts) == len(
                    chunks
                ), f"num of experts must be equal to num of chunks, but got {len(experts)} and {len(chunks)}"
                for chunk_id, (chunk, expert) in enumerate(zip(chunks, experts)):
                    ep_output += [expert(chunk)]
                expert_group_outputs += [paddle.stack(ep_output, axis=1)]
            return expert_group_outputs
        else:
            chunks = dispatched_input.unbind(1)
            expert_outputs = []
            assert len(chunks) == len(self.experts), (len(chunks), len(self.experts))
            for chunk, expert in zip(chunks, self.experts):
                expert_outputs += [expert(chunk)]
            expert_output = paddle.stack(expert_outputs, axis=1)
            return expert_output

    def gate_and_distpach(self, input, token_type_ids):
        """
        calc gate and dispatch inputs (and do logging, optionaly)
        Args:
            input: Tensor[seq, dim], float
            token_type_ids: Tensor[seq], int
        Returns:
            dispatched_input: Tensor[num_experts, capacity, dim]
            combine_weights: [seq, k]
            scatter_index: [seq, k]
            router_loss: scalar
            gate_logits: [seq, num_experts]
        """
        with profile("moe-gate"):
            args = ()
            if token_type_ids is not None:
                token_type_ids = token_type_ids.reshape([-1])
                args = (token_type_ids,)
            use_fuse = isinstance(self.gate, (TopKGateFused))
            if use_fuse:
                (gate_logits, capacity, router_loss, local_capacity) = self.gate(
                    input, *args
                )
            else:
                (
                    capacity,
                    dispatch_mask,
                    combine_weights,
                    scatter_index,
                    router_loss,
                    gate_logits,
                ) = self.gate(input, *args)
                prob = None
            if self.input_preprocess is not None:
                input, gate_logits = self.input_preprocess(input, gate_logits, capacity)

        with profile("moe-dispatch"):
            if use_fuse:
                k = self.k
                prob, max_prob = self.fused_gate_logits_process(
                    gate_logits, token_type_ids
                )
                if "corr_bias" in inspect.signature(moe_gate_dispatch).parameters:
                    if self.use_correction_bias:
                        compat_args = (
                            _reshard(
                                self.moe_statics.e_score_correction_bias[0],
                                get_flatten_mesh(get_mesh(self.ipp)),
                                [dist.Replicate()],
                            ),
                        )
                    else:
                        compat_args = (None,)
                else:
                    assert (
                        not self.use_correction_bias
                    ), "correction bias not supported, rebuild moe-ops"
                    compat_args = ()
                (
                    dispatched_input,
                    combine_weights_unnorm,
                    scatter_index,
                    dispatch_mask,
                    _,
                ) = moe_gate_dispatch(
                    input, prob, *compat_args, k, local_capacity, True
                )
                dispatch_mask = paddle.diff(F.pad(dispatch_mask, (1, 0)))
                if self.use_correction_bias:
                    if self.gate.config.multimodel_experts:
                        for i in range(len(self.moe_statics.expert_usage)):
                            self.moe_statics.expert_usage[i] += dispatch_mask[
                                self.gate.experts_type_mask[i]
                            ].detach()
                    else:
                        reshard_dispatch_mask = _reshard(
                            dispatch_mask.detach(),
                            get_mesh(self.ipp),
                            [
                                dist.Replicate()
                                for _ in range(len(get_mesh(self.ipp).shape))
                            ],
                        )
                        self.moe_statics.expert_usage[0] += reshard_dispatch_mask
                dispatched_input.stop_gradient = False
                combine_weights_unnorm.stop_gradient = False
                dispatch_mask.stop_gradient = True

                scatter_index = scatter_index.transpose([1, 0])

                if self.group_experts:
                    if max_prob is not None:
                        if token_type_ids is not None:
                            p = paddle.ones_like(combine_weights_unnorm.unsqueeze(-1))
                            p = paddle.scatter_nd_add(
                                p, paddle.nonzero(token_type_ids == 0), -1 + max_prob
                            )
                        else:
                            p = max_prob
                        combine_weights_unnorm = (
                            combine_weights_unnorm.unsqueeze(-1) * p
                        ).squeeze(-1)
                        prob = (prob.reshape([p.shape[0], k, -1]) * p).reshape(
                            [p.shape[0], -1]
                        )
                combine_weights = combine_weights_unnorm / paddle.clip(
                    combine_weights_unnorm.sum(-1, keepdim=True), min=1e-12
                )
                combine_weights = combine_weights.cast(dispatched_input.dtype)
            else:
                dispatched_input = dispatching(
                    input,
                    dispatch_mask.unbind(1),
                    scatter_index.unbind(1),
                    num_experts=self.config.moe_num_experts,
                    capacity=capacity,
                )
        dispatch_mask.stop_gradient = True
        return (
            dispatched_input,
            combine_weights,
            dispatch_mask,
            scatter_index,
            router_loss,
            gate_logits,
            prob,
        )

    def combine_expert_output(self, expert_output, combine_weights, scatter_index):
        """
        Combine Expert output
        Args:
            expert_output: Tensor[num_experts, caapcity, dim]
            combine_weights:
        Returns:
            combined_output: Tensor[seqlen, dim]
        """
        with profile("moe-combine"):
            if self.config.moe_use_all2all and self.config.moe_group == "mp":
                expert_output = dist.auto_parallel.moe_utils._dist_reshape(
                    expert_output,
                    [-1, expert_output.shape[-1]],
                    get_flatten_mesh(get_mesh(self.ipp)),
                    [dist.Shard(0)],
                )
            else:
                expert_output = expert_output.reshape([-1, expert_output.shape[-1]])

            use_fuse = isinstance(self.gate, (TopKGateFused))
            combine_fn = combining_fused if use_fuse else combining
            combine_weights = (
                combine_weights if use_fuse else combine_weights.unsqueeze(1)
            )

            combined_output = combine_fn(expert_output, combine_weights, scatter_index)

            if self.output_postprocess is not None:
                combined_output = self.output_postprocess(combined_output)
        return combined_output

    def forward(
        self,
        input: Tensor,
        token_type_ids=None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        Args:
            input (`Tensor`): The input data with shape ``(s, d)``.
                Only one token is supported for now.
            token_type_ids (`Tensor`) int64 tensor with shape (s),
                if specified, rount tensor according to `token_type_ids`.
        Returns:
            output (`Tensor`): The final output tensor with shape ``(s, d)`` where ``m`` is the
                size of model parameters.
            combine_weights (`Tensor`, optional): A tensor with shape ``(s,)``, which represents weights
                for each expert in MoE.
            router_loss (`Tensor`, optional): A scalar tensor representing the loss of routing function.
        """
        if self.shared_experts is not None:
            shared_expert_input = dist.reshard(
                input,
                get_mesh(self.ipp),
                [dist.Shard(1), dist.Replicate()],
            )
        if input.ndim == 3:
            orig_shape = input.shape
            input = dist.reshard(
                input, get_mesh(self.ipp), [dist.Replicate(), dist.Shard(0)]
            )
            if self.config.moe_use_all2all:
                input = dist.auto_parallel.moe_utils._dist_reshape(
                    input,
                    [-1, input.shape[-1]],
                    get_flatten_mesh(get_mesh(self.ipp)),
                    [dist.Shard(0)],
                )
        else:
            orig_shape = None
        assert (
            len(input.shape) == 2
        ), f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"
        seqlen, d_model = input.shape

        if token_type_ids is not None:
            token_type_ids = token_type_ids.clone()[:, :-1]
            if self.config.sequence_parallel:
                token_type_ids = token_type_ids.reshape([-1])
                token_type_ids.stop_gradient = True

        assert self.gate is not None
        if hasattr(self, "rng") and self.rng.random() < self.all_to_all_dropout:
            orig_shape_2 = input.shape
            output = self.forward_experts(input)
            output += self.gate.weight.sum() * 0.0
            output = output.reshape(orig_shape or orig_shape_2)
            return output, None, 0

        (
            dispatched_input,
            combine_weights,
            dispatch_mask,
            scatter_index,
            router_loss,
            gate_logits,
            gate_prob,
        ) = self.gate_and_distpach(input, token_type_ids)
        if self.config.moe_use_all2all and self.config.moe_group == "mp":
            dispatched_input = _reshard(
                dispatched_input, get_mesh(self.ipp), [dist.Shard(1), dist.Shard(1)]
            )
        if self.config.moe_group == "mp":
            dispatched_input = dist.reshard(
                dispatched_input, get_mesh(self.ipp), [dist.Shard(1), dist.Shard(0)]
            )

        if self.shared_experts is not None:
            shared_out = self.shared_experts(shared_expert_input)
        dispatched_input = dispatched_input.reshape(
            [self.config.moe_world_size, self.num_local_experts, -1, d_model]
        )
        expert_out = self.forward_experts(dispatched_input)
        if self.config.moe_group == "mp":
            expert_out = dist.auto_parallel.api.moe_global_mesh_tensor(
                expert_out,
                get_mesh(self.ipp),
                [dist.Shard(2), dist.Shard(0)],
                self.moe_mesh_dim,
            )
            expert_out = dist.auto_parallel.moe_utils._dist_reshape(
                expert_out,
                [self.config.moe_world_size * self.num_local_experts, -1, d_model],
                get_mesh(self.ipp),
                [dist.Shard(1), dist.Shard(0)],
            )
            expert_out = dist.reshard(
                expert_out, get_mesh(self.ipp), [dist.Shard(1), dist.Shard(1)]
            )
        router_loss2 = self.calc_router_loss_and_logging(
            router_loss,
            combine_weights,
            dispatch_mask,
            gate_logits,
            gate_prob,
            token_type_ids,
        )

        combined_output = self.combine_expert_output(
            expert_out, combine_weights, scatter_index
        )

        if self.shared_experts is not None:
            shared_out = dist.auto_parallel.moe_utils._dist_reshape(
                shared_out,
                [-1, shared_out.shape[-1]],
                get_flatten_mesh(get_mesh(self.ipp)),
                [dist.Shard(0)],
            )
            combined_output += shared_out

        if orig_shape:
            if self.config.moe_use_all2all:
                combined_output = dist.auto_parallel.moe_utils._dist_reshape(
                    combined_output,
                    orig_shape[:-1] + [combined_output.shape[-1]],
                    get_mesh(self.ipp),
                    [dist.Shard(1), dist.Shard(0)],
                )
                router_loss2 = _reshard(
                    router_loss2,
                    get_mesh(self.ipp),
                    [dist.Replicate(), dist.Replicate()],
                )
        return combined_output, combine_weights, router_loss2, gate_logits


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
        gate = TopKGateFused(
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

    if config.moe_use_aux_free:
        moe_statics = MoEStatics(config, layer_idx)
    else:
        moe_statics = None
    return gate, experts, lm_gate, lm_experts, moe_statics


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
            x = swiglu(self.gate_proj(x), self.up_proj(x))
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

    def __len__(self):
        return self.num_local_experts

    def __iter__(self):
        return (self for _ in range(1))

    def forward(self, x):
        if self.fuse_swiglu:
            x = swiglu(self.up_gate_proj(x))
        else:
            gate, x = self.up_gate_proj(x).chunk(2, axis=-1)
            x = F.silu(gate) * x
        x = self.down_proj(x)
        return x


def cal_aux_loss_func(
    gate_prob,
    dispatch_mask,
    tokens_mask,
    dispatch_tokens_mask,
    num_experts,
    use_group,
    moe_k,
    global_aux_loss=False,
    rank=None,
    group=None,
):
    if tokens_mask is not None and tokens_mask.dtype != gate_prob.dtype:
        tokens_mask = tokens_mask.astype(gate_prob.dtype)

    scale = None
    if dispatch_tokens_mask is not None:
        seqlen_float = dispatch_tokens_mask.astype(gate_prob.dtype).sum()
        if (
            tokens_mask is not None
            and gate_prob.shape[0] != dispatch_tokens_mask.shape[0]
        ):
            scale = seqlen_float / paddle.clip(tokens_mask.sum(), min=1e-6)
    elif tokens_mask is not None:
        seqlen_float = tokens_mask.sum()
    else:
        seqlen_float = gate_prob.numel().astype(gate_prob.dtype) / num_experts
    seqlen_float = paddle.clip(seqlen_float, min=1e-6)

    if len(dispatch_mask.shape) == 2:
        dispatch_mask = dispatch_mask.sum(0)
    ce = dispatch_mask.astype(gate_prob.dtype).detach() / seqlen_float
    me = paddle.sum(gate_prob, axis=0) / seqlen_float
    if global_aux_loss:
        me_list, ce_list = [], []
        dist.all_gather(me_list, me, group=group)
        dist.all_gather(ce_list, ce, group=group)

        me_list[rank] = me
        ce_list[rank] = ce
        me = paddle.stack(me_list).mean(0)
        ce = paddle.stack(ce_list).mean(0)

    l_aux = paddle.sum(me * ce) * num_experts
    if use_group:
        l_aux = l_aux / moe_k

    if scale is not None:
        l_aux = l_aux + (scale - 1) * l_aux.detach()

    return l_aux


def gate_detach_matmul(x, weight, use_fake_gate=False):
    x = x.cast(paddle.float32) if x.dtype != paddle.float32 else x
    score = F.linear(x, weight)

    if use_fake_gate:
        score = paddle.randn(score.shape).astype(score.dtype) + score - score
    return score


class TopKGateFused(nn.Layer):

    def __init__(self, config, layer_idx: int, group, ipp=0) -> None:
        super().__init__()
        self.config = config
        assert not config.fuse_gate_detach_matmul, "matmul_bwd is not supported"

        self.use_fake_gate = config.use_fake_gate
        if self.use_fake_gate:
            logging.warning(
                "You are use fake_gate, which is just for test, not for real training."
            )

        self.model_dim = config.hidden_size
        self.num_experts = config.moe_num_experts
        self.num_experts_tensor = (
            sum(config.moe_num_experts)
            if config.multimodel_experts
            else config.moe_num_experts
        )

        self.cap = config.moe_capacity
        self.group = group

        self.layer_idx = layer_idx
        self.global_aux_loss = config.global_aux_loss
        if self.global_aux_loss:
            self.rank = dist.get_rank(self.group)

        self.use_token_type_bias = config.moe_use_token_type_bias
        self.use_correction_bias = config.moe_use_aux_free

        self.ipp = ipp

        if config.moe_gate_act == "softmax":
            self.act = partial(F.softmax, axis=-1)
        elif config.moe_gate_act == "sigmoid":
            self.act = F.sigmoid
        else:
            raise ValueError(f"{config.moe_gate_act} is not supported.")

        self.moe_aux_loss_lambda = paddle.to_tensor(
            config.moe_aux_loss_lambda, dtype="float32"
        )

        if self.moe_aux_loss_lambda.ndim == 0:
            self.moe_aux_loss_lambda = self.moe_aux_loss_lambda.unsqueeze(0)

        self.moe_orthogonal_loss_lambda = paddle.to_tensor(
            config.moe_orthogonal_loss_lambda, dtype="float32"
        )

        if self.moe_orthogonal_loss_lambda.ndim == 0:
            self.moe_orthogonal_loss_lambda = self.moe_orthogonal_loss_lambda.unsqueeze(
                0
            )

        self.experts_type_ids = None
        if config.moe_orthogonal_loss_lambda:
            if hasattr(fleet.fleet, "_user_defined_strategy"):
                strategy = fleet.fleet._user_defined_strategy
                sharding_configs = strategy.hybrid_configs["sharding_configs"]
                pp_config = strategy.hybrid_configs["pp_configs"]
                assert (
                    not sharding_configs.comm_overlap
                    and not pp_config.sharding_comm_overlap
                ), "orthogonal loss will cause twice gradient accumulate, will break pp/sharding overlap"

        self.eps = paddle.to_tensor([1e-12], dtype="float32")
        if config.multimodel_experts:
            if config.moe_use_hard_gate:
                self.num_experts_list = []
                self.experts_type_mask = []
                experts_ids = paddle.zeros(
                    [sum(self.num_experts)], dtype="int64"
                ).reshape([config.moe_world_size, -1])
                offset = 0
                for i, expert_num in enumerate(self.num_experts):
                    experts_ids[
                        :, offset : offset + expert_num // config.moe_world_size
                    ] = i
                    offset += expert_num // config.moe_world_size
                self.experts_type_ids = experts_ids.reshape([-1])
                logger.info(
                    f"use moe_use_hard_gate, experts_ids: {self.experts_type_ids}"
                )
                for i, expert_num in enumerate(self.num_experts):
                    self.experts_type_mask.append(
                        self.experts_type_ids == i,
                    )
                    self.num_experts_list.append(expert_num)
            else:
                assert (
                    not config.moe_group_experts
                ), "group_experts must use hard_gate when multimodel_experts is True"
        else:
            self.num_experts_list = [self.num_experts]
        self._create_gate_parameter()
        logger.info(
            f"{config.moe_gate}: w/ capacity: {self.cap} experts:{self.num_experts} "
            f"use_token_type_bias:{self.use_token_type_bias} gate_act:{config.moe_gate_act} "
        )

    def _create_gate_parameter(self):

        if self.config.multimodel_experts:

            self.moe_aux_loss_lambda = self.moe_aux_loss_lambda.expand(
                len(self.num_experts)
            )
            self.moe_orthogonal_loss_lambda = self.moe_orthogonal_loss_lambda.expand(
                len(self.num_experts)
            )

            for i, num_experts in enumerate(self.num_experts):
                if i == 1:
                    with paddle.utils.unique_name.guard(f"mm_gate_{self.layer_idx}_"):
                        p = self.create_parameter(
                            shape=[self.model_dim, num_experts],
                            dtype="float32",
                            attr=paddle.ParamAttr(
                                name=unique_name.generate("moe_gate")
                            ),
                        )
                else:
                    p = self.create_parameter(
                        shape=[self.model_dim, num_experts],
                        dtype="float32",
                        attr=paddle.ParamAttr(name=unique_name.generate("moe_gate")),
                    )
                p.expert_type = f"expert_type_{i}"
                self.add_parameter(
                    ("weight" if i == 0 else f"weight_{i}"),
                    p,
                )
        else:
            self.weight = self.create_parameter(
                shape=[self.model_dim, self.num_experts],
                dtype="float32",
                attr=paddle.ParamAttr(name=unique_name.generate("moe_gate")),
            )
            logger.info(f"moe-Gate, {self.weight}")

        if self.use_token_type_bias:
            if self.config.multimodel_experts:
                assert (
                    not self.config.moe_use_hard_gate
                ), "multimodel_experts with hard_gate is not support token_type_bias."
            num_experts = (
                sum(self.num_experts)
                if self.config.multimodel_experts
                else self.num_experts
            )
            bias_type_num = (
                len(self.num_experts) if self.config.multimodel_experts else 1
            )
            self.bias = self.create_parameter(
                shape=[bias_type_num, num_experts],
                dtype="float32",
                attr=paddle.ParamAttr(
                    name=unique_name.generate("moe_gate_bias"),
                    initializer=paddle.nn.initializer.Assign(
                        np.zeros([bias_type_num, num_experts])
                    ),
                ),
            )
            logger.info(f"using token type bias, bias: {self.bias},")
        self._cast_to_low_precision = False
        self._cast_to_low_precison = False

    def get_gate_weight(self, transform_weight):
        if not self.config.multimodel_experts:
            return self.weight
        if not transform_weight:
            return paddle.concat(
                [
                    getattr(self, "weight" if i == 0 else f"weight_{i}")
                    for i in range(len(self.num_experts))
                ],
                -1,
            )
        weight = paddle.zeros(
            [
                self.model_dim,
                self.config.moe_world_size,
                sum(self.num_experts) // self.config.moe_world_size,
            ],
            dtype="float32",
        )
        offset = 0
        for i, num_experts in enumerate(self.num_experts):
            weight[
                :, :, offset : offset + num_experts // self.config.moe_world_size
            ] = getattr(self, "weight" if i == 0 else f"weight_{i}").reshape(
                [self.model_dim, self.config.moe_world_size, -1]
            )
            offset += num_experts // self.config.moe_world_size
        weight = weight.reshape([self.model_dim, -1])

        return weight

    def _cal_aux_loss(
        self,
        gate_prob,
        dispatch_mask,
        num_experts=None,
        use_group=None,
        tokens_mask=None,
        dispatch_tokens_mask=None,
    ):

        if self.act is F.sigmoid:
            gate_prob = gate_prob / gate_prob.sum(-1, keepdim=True)

        if self.use_correction_bias:
            if tokens_mask is not None:
                gate_prob_this_modality = gate_prob[tokens_mask.astype("bool")]
                if gate_prob_this_modality.shape[0]:
                    _, top_idx = gate_prob_this_modality.topk(
                        k=self.config.moe_k, axis=-1
                    )
                    mask = paddle.zeros_like(gate_prob_this_modality).put_along_axis(
                        top_idx, paddle.to_tensor(1.0), axis=1
                    )
                    dispatch_mask = paddle.sum(mask.cast(paddle.int64), axis=0)
                else:
                    dispatch_mask = paddle.zeros(gate_prob.shape[-1], dtype="int64")
                dist.stream.all_reduce(
                    dispatch_mask,
                    group=self.group,
                    use_calc_stream=True,
                )
            else:
                _, top_idx = gate_prob.topk(k=self.config.moe_k, axis=-1)

                mask = paddle.zeros_like(gate_prob).put_along_axis(
                    top_idx, paddle.to_tensor(1.0), axis=1
                )
                dispatch_mask = paddle.sum(mask.cast(paddle.int64), axis=0)

        if num_experts is None:
            num_experts = self.num_experts_tensor
        if use_group is None:
            use_group = self.config.moe_group_experts

        return cal_aux_loss_func(
            gate_prob,
            dispatch_mask,
            tokens_mask,
            dispatch_tokens_mask,
            num_experts,
            use_group,
            self.config.moe_k,
            self.global_aux_loss,
            self.rank if self.global_aux_loss else None,
            self.group if self.global_aux_loss else None,
        )

    def forward(
        self,
        input: Tensor,
        token_type_ids=None,
        transform_weight=True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            input: paddle.Tensor, hidden-states of layer
        Retruns:
            paddle.Tensor [Seq, Expert, Capacity]: float32, combine weights
            paddle.Tensor [Seq, Expert, Capacity]: bool, dispatch mask
            Tuple[paddle.Tensor]: `GateOutput`
        """
        num_experts = (
            sum(self.num_experts)
            if self.config.multimodel_experts
            else self.num_experts
        )
        if self.training:
            cap = self.cap[0]
        elif input.shape[0] < num_experts:
            cap = self.cap[2]
        else:
            cap = self.cap[1]
        num_tokens = input.shape[0]
        global_capacity = int(cap * num_tokens // num_experts)
        local_num_tokens = input._local_shape[0]
        local_capacity = int(cap * local_num_tokens // num_experts)

        weight = self.get_gate_weight(transform_weight)
        with paddle.amp.auto_cast(False):
            input = _reshard(
                input, get_mesh(self.ipp), [dist.Replicate(), dist.Shard(0)]
            )
            logits = gate_detach_matmul(input, weight, self.use_fake_gate)
            logits = _reshard(
                logits, get_flatten_mesh(get_mesh(self.ipp)), [dist.Shard(0)]
            )
            if self.use_token_type_bias:
                assert token_type_ids is not None
                assert (
                    token_type_ids.max() < self.bias.shape[0]
                ), f"token_type_ids {token_type_ids.max()} >= bias shape {self.bias.shape[0]}"
                bias = self.bias[token_type_ids]
                logits = logits + bias
            router_loss = paddle.zeros([1], dtype="float32")
            router_loss.stop_gradient = False

        return logits, global_capacity, router_loss, local_capacity
