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

"""_summary_

Returns:
    _type_: _description_
"""
from typing import Tuple, List, Optional
import logging
from collections import namedtuple
from contextlib import contextmanager

import paddle
from paddle import nn
import paddle.nn.functional as F

from paddle.distributed.communication.group import Group
from paddle.distributed import fleet

import paddle.distributed as dist
from paddle import Tensor
from paddleformers.trainer.plugins.timer import get_timers

from models.moe.top2_gate_auto import TopKGateFusedAuto
from models.moe.moe_utils import get_flatten_mesh, get_mesh, _reshard
from models.moe.moe_layer import MOELayer

try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}

try:
    from paddle.distributed import in_auto_parallel_align_mode
except ImportError:

    def in_auto_parallel_align_mode():
        """
        hack for paddlenlp develop branch.
        """
        return False


logger = logging.getLogger(__name__)

try:
    import moe_ops
except ImportError:
    moe_ops = None
    logger.warning(
        "`moe-ops` not found, run "
        "`python3  src/ernie_core/ops/moe/setup.py  install` to install"
    )

try:
    import moe_ops_auto
except ImportError:
    moe_ops_auto = None
    logger.warning(
        "`moe_ops_auto` not found, run "
        "`python3  src/ernie_core/ops/moe/setup_auto.py  install` to install"
    )

try:
    import moe_combine_auto
except ImportError:
    moe_combine_auto = None
    logger.warning(
        "`moe_combine_auto` not found, run "
        "`python3  src/ernie_core/ops/moe/setup_auto.py  install` to install"
    )


GateOutput = namedtuple(
    "GateOutput",
    [
        "aux",
        "z",
        "logits",
    ],
)


@contextmanager
def profile(name):

    if get_timers() is not None:
        get_timers()(name).start()
    yield
    if get_timers() is not None:
        get_timers()(name).stop()


def combining_fused_auto(x, combine_weights, scatter_index, hard_gate=False):
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
    ret = moe_combine_auto.moe_combine_auto(x, combine_weights, scatter_index)
    ret.stop_gradient = False
    return ret


def dispatching(x, dispatch_mask, scatter_index, num_experts, capacity):

    output = None
    orig_dtype = x.dtype
    scatter_index = scatter_index.unbind(1)
    dispatch_mask = dispatch_mask.unbind(1)
    for i_scatter_index, i_dispatch_mask in zip(scatter_index, dispatch_mask):
        init_output = paddle.zeros(
            [num_experts * capacity, x.shape[-1]], dtype="float32"
        )
        updates = x * i_dispatch_mask.unsqueeze(-1).cast(x.dtype)
        if output is None:
            output = paddle.scatter(
                init_output,
                i_scatter_index,
                updates,
                overwrite=False,
            )
        else:
            output = output + paddle.scatter(
                init_output,
                i_scatter_index,
                updates,
                overwrite=False,
            )
        if output.dtype != orig_dtype:
            output = output.cast(orig_dtype)
    return output


class MOELayerAuto(MOELayer):

    def __init__(
        self,
        gate: nn.Layer,
        experts: List[nn.Layer],
        layer_idx,
        shared_experts: Optional[List[nn.Layer]] = None,
        group: Group = None,
        recompute=False,
        enable_logging: bool = False,
        k=2,
        all_to_all_dropout=0,
        group_experts=False,
        config=None,
        ipp=0,
    ):

        nn.Layer.__init__(self)
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
        self.enable_logging = enable_logging
        is_mp_moe = (
            hasattr(fleet.fleet, "_hcg")
            and group is fleet.get_hybrid_communicate_group().get_model_parallel_group()
        )
        is_dummy_moe = config.moe_world_size == 1

        for p in experts.parameters():
            p.expert = not (is_mp_moe or is_dummy_moe)  # type: ignore
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

    def _cal_multimodel_experts_prob(
        self, gate_logits, token_type_ids, group_experts, moe_k
    ):

        if not self.gate.experts_type_ids.is_dist():
            self.gate.experts_type_ids = dist.shard_tensor(
                self.gate.experts_type_ids,
                get_mesh(),
                [dist.Replicate(), dist.Replicate()],
            )
        return super()._cal_multimodel_experts_prob(
            gate_logits, token_type_ids, group_experts, moe_k
        )

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
                # logger.info(f'moe-fwd-expert: {chunk.shape}'
                # f'-> {expert_outputs[-1].shape}: {chunk.astype("float32").norm(axis=-1)}')
            expert_output = paddle.stack(expert_outputs, axis=1)  # [ecm]
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
            use_fuse = isinstance(self.gate, (TopKGateFusedAuto))
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
                # capacity no use
                k = self.k
                prob, max_prob = self.fused_gate_logits_process(
                    gate_logits, token_type_ids
                )
                (
                    dispatched_input,
                    combine_weights_unnorm,
                    scatter_index,
                    dispatch_mask,
                    _,
                ) = moe_ops_auto.moe_gate_dispatch_auto(
                    input, prob, k, local_capacity, True
                )
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
                        # gate_prob 进行还原
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
                    dispatch_mask,
                    scatter_index,
                    num_experts=self.config.moe_num_experts,
                    capacity=capacity,
                )
        dispatch_mask.stop_gradient = True
        scatter_index.stop_gradient = True
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

            if not self.config.moe_use_all2all:
                if self.config.moe_group == "mp":
                    expert_output = dist.reshard(
                        expert_output,
                        get_mesh(self.ipp),
                        [dist.Replicate(), dist.Replicate()],
                    )
                else:
                    expert_output = dist.reshard(
                        expert_output, get_mesh(), [dist.Shard(0), dist.Replicate()]
                    )
            assert isinstance(
                self.gate, (TopKGateFusedAuto)
            ), "Only TopKGateFusedAuto is supported here"
            combined_output = combining_fused_auto(
                expert_output, combine_weights, scatter_index
            )

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
                input = input.reshape([-1, input.shape[-1]])
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
                # token_type_ids = ScatterOp.apply(token_type_ids)
                token_type_ids.stop_gradient = True

        assert self.gate is not None
        if hasattr(self, "rng") and self.rng.random() < self.all_to_all_dropout:
            orig_shape_2 = input.shape
            output = self.forward_experts(input)
            output += self.gate.weight.sum() * 0.0  # hack for grad
            output = output.reshape(orig_shape or orig_shape_2)  # [e*1,c,m]
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
        if not in_auto_parallel_align_mode():
            router_loss2 = self.calc_router_loss_and_logging(
                router_loss,
                combine_weights,
                dispatch_mask,
                gate_logits,
                gate_prob,
                token_type_ids,
            )
        else:
            router_loss2 = router_loss
            router_loss2 = dist.shard_tensor(
                router_loss2, get_flatten_mesh(get_mesh(self.ipp)), [dist.Replicate()]
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
            else:
                combined_output = combined_output.reshape(
                    orig_shape[:-1] + [combined_output.shape[-1]]
                )
        return combined_output, combine_weights, router_loss2, gate_logits
