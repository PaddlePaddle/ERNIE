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


from typing import Tuple, List, Optional
import logging
import inspect
from collections import namedtuple
from contextlib import contextmanager
from functools import partial

import paddle
from paddle import framework
from paddle import nn
from paddle.distributed.communication import stream
import paddle.nn.functional as F
from paddle.distributed import in_auto_parallel_align_mode

from paddle.autograd import PyLayer
from paddle.distributed.communication.group import Group
from paddle.distributed.fleet.utils import recompute
from paddle.distributed import fleet

import paddle.distributed as dist
from paddle import Tensor
from paddleformers.trainer.plugins.timer import get_timers
from models.moe.top2_gate_auto import TopKGateFused, TopKGateFusedAuto
from models.sequence_parallel_utils_auto import ScatterOp
from models.utils_auto import (
    manual_backward,
)


from models.moe.moe_utils_auto import get_flatten_mesh, get_mesh, _reshard
from paddle.incubate.nn.functional import (
    moe_combine,
)


try:
    import moe_router_loss_ops
except ImportError:
    moe_router_loss_ops = None


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

try:
    from moe_combine import moe_combine_no_weight
except ImportError:
    moe_combine_no_weight = None

try:
    import moe_ops_fp8
except ImportError:
    moe_ops_fp8 = None
    logger.warning(
        "`moe-ops` not found, run "
        "`python3  src/ernie_core/ops/moe/setup_fp8.py  install` to install"
    )


@contextmanager
def profile(name):
    """doc"""
    if get_timers() is not None:
        get_timers()(name).start()
    yield
    if get_timers() is not None:
        get_timers()(name).stop()


GateOutput = namedtuple(
    "GateOutput",
    [
        "aux",
        "z",
        "logits",
    ],
)


class GateCombine(PyLayer):

    @staticmethod
    def forward(ctx, x, combine_weights, scatter_index):
        ctx.x = x
        ctx.combine_weights = combine_weights
        ctx.scatter_index = scatter_index
        assert moe_combine is not None
        ret = moe_combine.moe_combine(x, combine_weights, scatter_index)
        return ret

    @staticmethod
    def backward(ctx, grad_y, *_):
        assert moe_combine is not None
        grad_x, grad_combine_weight_helper = moe_combine.moe_combine_bwd(
            ctx.x, ctx.combine_weights, ctx.scatter_index, grad_y
        )

        grad_combine_weight = grad_combine_weight_helper.sum(-1)
        return grad_x, grad_combine_weight.reshape(ctx.combine_weights.shape), None


def combining_fused(x, combine_weights, scatter_index, hard_gate=False):

    if hard_gate:
        x_gatherd = F.embedding(scatter_index, x)
        return x_gatherd.squeeze(-2)
    ret = GateCombine.apply(x, combine_weights, scatter_index)
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


def combining(x, combine_weights, scatter_index):

    dim = x.shape[-1]
    scatter_index = scatter_index.reshape([-1])
    num_k = combine_weights.shape[-1]
    combine_weights = combine_weights.unsqueeze(1)
    x = paddle.gather(x, scatter_index).reshape([-1, num_k, dim])
    return paddle.matmul(combine_weights, x).squeeze(1)


class Fp8MoeGateDispatchAndQuant(paddle.autograd.PyLayer):

    @staticmethod
    def forward(
        ctx, x, gate_logtis, corr_bias, k, capacity, use_pad, use_pow2_scale=True
    ):
        (
            out_fp8,
            scale,
            combine_weights,
            scatter_index,
            expert_offset,
            expert_id,
        ) = moe_ops_fp8.moe_gate_dispatch_and_quant(
            x,
            gate_logtis,
            corr_bias=corr_bias,
            k=k,
            capacity=capacity,
            use_pad=use_pad,
            use_pow2_scale=use_pow2_scale,
        )
        assert out_fp8.shape[0] == scale.shape[0]

        out_fp8.stop_gradient = False
        combine_weights.stop_gradient = False
        scatter_index.stop_gradient = True
        expert_offset.stop_gradient = True
        expert_id.stop_gradient = True
        scale.stop_gradient = True

        ctx.k = k
        ctx.capacity = capacity
        ctx.use_pad = use_pad
        ctx.combine_weights = combine_weights
        ctx.scatter_index = scatter_index
        ctx.expert_id = expert_id
        ctx.has_corr_bias = corr_bias is not None

        return (
            out_fp8,
            combine_weights,
            scatter_index,
            expert_offset,
            expert_id,
            {
                "scale": scale,
            },
        )

    @staticmethod
    def backward(ctx, *grads):
        out_grad, combine_weights_grad = grads[0], grads[1]
        x_grad, gate_logits_grad = moe_ops.moe_gate_dispatch_bwd(
            ctx.combine_weights,
            ctx.scatter_index,
            ctx.expert_id,
            out_grad,
            combine_weights_grad,
            k=ctx.k,
            capacity=ctx.capacity,
            use_pad=ctx.use_pad,
        )
        if ctx.has_corr_bias:
            return x_grad, gate_logits_grad, None
        else:
            return x_grad, gate_logits_grad


class AlltoAll(PyLayer):

    @staticmethod
    def forward(ctx, x, group, sync_op=True):

        ctx.group = group
        if dist.get_world_size(group) <= 1:
            return x
        output = paddle.empty_like(x)
        output.stop_gradient = False
        task = stream.alltoall_single(
            output, x, None, None, group, sync_op=sync_op, use_calc_stream=sync_op
        )
        if not sync_op:
            return output, task
        else:
            return output

    @staticmethod
    def backward(ctx, *dx):
        return AlltoAll.apply(*dx, group=ctx.group)


class AlltoAllExpertOverlap(PyLayer):

    @staticmethod
    def forward(
        ctx, input, group, num_local_experts, forward_func_dict, is_first_fwd=False
    ):
        assert (
            dist.get_world_size(group) > 1
        ), "AlltoAllExpertOverlap is not supported for a world size less than or equal to 1."

        ctx.bw_funcs = {}
        ctx.group = group
        ctx.num_local_experts = num_local_experts

        assert isinstance(forward_func_dict, nn.LayerList)
        all2all_tasks = []
        all2all_ins = paddle.unbind(input, axis=0)
        for stage_id in range(1):
            stage_input = all2all_ins[stage_id]
            x_out, task = AlltoAll.apply(stage_input, group=group, sync_op=False)
            all2all_tasks.append((task, x_out))

        expert_outputs = []
        for stage_id in range(num_local_experts):
            if stage_id + 1 != num_local_experts:
                stage_input = all2all_ins[stage_id + 1]
                x_out, task = AlltoAll.apply(stage_input, group=group, sync_op=False)
                all2all_tasks.append((task, x_out))

            task, dispatched_input = all2all_tasks[stage_id]
            task.wait()
            bwf, (expert_outputs_cur_stage,) = manual_backward(
                forward_func_dict[stage_id], is_first_fwd, dispatched_input
            )
            ctx.bw_funcs[stage_id] = bwf
            expert_outputs.append(expert_outputs_cur_stage)

        expert_output = paddle.stack(expert_outputs, axis=1)
        return expert_output

    @staticmethod
    def backward(ctx, out_grad):
        all2all_tasks = []
        expert_outputs = []

        out_grad_list = paddle.split(
            out_grad, num_or_sections=out_grad.shape[1], axis=1
        )
        for stage_id in range(ctx.num_local_experts):
            (grad_cur_stage,) = ctx.bw_funcs[stage_id](out_grad_list[stage_id])

            x_out, task = AlltoAll.apply(grad_cur_stage, group=ctx.group, sync_op=False)
            all2all_tasks.append(task)
            expert_outputs.append(x_out)

        for task in all2all_tasks:
            task.wait()

        expert_output = paddle.stack(expert_outputs, axis=0)
        return expert_output


class AlltoAllAsync(PyLayer):

    @staticmethod
    def forward(ctx, x, *fn_args, group=None, fn=None, is_first_fwd=False):

        assert fn is not None, "use AlltoAll no async"
        ctx.group = group
        if dist.get_world_size(group) <= 1:
            ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
            return (x,) + fn_out
        x_out = paddle.empty_like(x)
        x_out.stop_gradient = False
        task = stream.alltoall_single(
            x_out,
            x,
            None,
            None,
            group,
            sync_op=False,
        )
        ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
        task.wait()
        return (x_out,) + fn_out

    @staticmethod
    def backward(ctx, dx_out, *fn_out_grads):
        if dist.get_world_size(ctx.group) <= 1:
            fn_args_grads = ctx.bwf(*fn_out_grads)
            return (dx_out,) + fn_args_grads

        dx = paddle.empty_like(dx_out)
        dx.stop_gradient = False
        task = stream.alltoall_single(
            dx,
            dx_out,
            None,
            None,
            ctx.group,
            sync_op=False,
        )
        fn_args_grads = ctx.bwf(*fn_out_grads)
        task.wait()
        return (dx,) + fn_args_grads


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
        enable_bpr: bool = False,
        all_to_all_dropout=0,
        group_experts=False,
        moe_statics=None,
    ):

        super().__init__()
        self.gate = gate
        self.layer_idx = layer_idx
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
        self.use_correction_bias = moe_statics is not None
        self.moe_statics = moe_statics
        if self.use_correction_bias:
            logger.info(
                f"using correction bias, aux-coef:{self.gate.config.moe_aux_loss_lambda}"
            )
            assert self.gate.config.moe_use_aux_free

        self.is_mp_moe = (
            hasattr(fleet.fleet, "_hcg")
            and group is fleet.get_hybrid_communicate_group().get_model_parallel_group()
        )
        self.is_ep_moe = (
            hasattr(fleet.fleet, "_hcg")
            and hasattr(
                fleet.get_hybrid_communicate_group(),
                "get_moe_sharding_parallel_world_size",
            )
            and fleet.get_hybrid_communicate_group().get_moe_sharding_parallel_world_size()
            > 0
        )
        is_dummy_moe = dist.get_world_size(group) == 1

        for p in experts.parameters():
            p.expert = not (self.is_mp_moe or is_dummy_moe)
            p.no_sync = not (self.is_mp_moe or is_dummy_moe)
            logger.info(f"expert no-sync={p.no_sync}-{p.name}")
            if self.is_mp_moe or self.is_ep_moe:
                p.is_distributed = True

        expert_color = None
        if self.is_ep_moe:
            moe_grad_group = (
                fleet.get_hybrid_communicate_group().get_moe_sharding_parallel_group()
            )
            expert_color = {"color": "moe_expert", "group": moe_grad_group}
        elif (
            self.config.offline_quant_expert_weight
            and self.config.clear_origin_weight_when_offline_quant
        ):
            expert_color = {"color": "moe_expert"}

        if expert_color is not None:
            for p in self.experts.parameters():
                setattr(p, "color", expert_color)

        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        if self.world_size < 1:
            self.world_size = 1
        if self.rank < 0:
            self.rank = 0

        self.num_local_experts = len(self.experts)
        self.dispatch_by_task = (
            hasattr(self.gate, "dispatch_by_task") and self.gate.dispatch_by_task
        )

        if self.dispatch_by_task:
            assert 0, "no supported, checkout earylier code"
            assert self.num_local_experts == 1

        self.input_preprocess = self.output_postprocess = None
        self.group_experts = group_experts
        self.config = self.gate.config
        self.zero = paddle.to_tensor(0, dtype=paddle.float32)

        self._rr_moe_gate_dispatch = None
        self._rr_moe_combine = None
        self.use_norm_gate_recompute = None

        if self.config.use_recompute and self.config.skip_recompute_ops.get(
            "moe_gate_dispatch", False
        ):
            self._rr_moe_gate_dispatch = None
        if self.config.use_recompute and self.config.skip_recompute_ops.get(
            "moe_combine", False
        ):
            self._rr_moe_combine = None
        if hasattr(fleet.fleet, "_hcg"):
            hcg = fleet.get_hybrid_communicate_group()
            if (
                hasattr(hcg, "get_moe_sharding_parallel_world_size")
                and hcg.get_moe_sharding_parallel_world_size() > 0
            ):
                moe_grad_group = hcg.get_moe_sharding_parallel_group()
                for p in self.experts.parameters():
                    setattr(
                        p, "color", {"color": "moe_expert", "group": moe_grad_group}
                    )

    def forward_experts(self, dispatched_input):

        with profile("fwd-expert"):
            dispatched_input = dispatched_input.reshape(
                [
                    self.world_size,
                    self.num_local_experts,
                    -1,
                    dispatched_input.shape[-1],
                ]
            )
            expert_outputs = []
            if isinstance(self.experts, nn.LayerList):

                chunks = dispatched_input.transpose([1, 0, 2, 3]).contiguous().unbind(0)
                assert len(chunks) == len(self.experts), (
                    len(chunks),
                    len(self.experts),
                )
                for chunk, expert in zip(chunks, self.experts):
                    expert_outputs += [expert(chunk)]

                expert_output = paddle.stack(expert_outputs, axis=1)

            else:
                dispatched_input = dispatched_input.transpose([1, 0, 2, 3])
                dispatched_input.contiguous()
                orig_shape = dispatched_input.shape
                chunks = dispatched_input.reshape([orig_shape[0], -1, orig_shape[-1]])
                chunks = self.experts(chunks)
                chunks = chunks.reshape(orig_shape[:-1] + [chunks.shape[-1]]).unbind(0)
                expert_outputs += chunks
                expert_output = paddle.stack(expert_outputs, axis=1)
        return expert_output

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

    def gate_distpach_and_quant(self, input, token_type_ids):

        assert isinstance(self.gate, (TopKGateFused)), "Only fused gate is supported."
        assert not self.config.use_ep_comm_overlap, "ep_comm_overlap is not supported"
        assert (
            self._rr_moe_gate_dispatch is None
        ), "rr_moe_gate_dispatch is not supported"
        assert moe_ops_fp8 is not None

        args = ()
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape([-1])
            args = (token_type_ids,)

        (
            gate_logits,
            capacity,
            router_loss,
        ) = self.gate(input, *args)

        if self.config.moe_multimodal_paired_experts:
            assert token_type_ids is not None
            input = paddle.concat(
                [input, token_type_ids.unsqueeze(-1).astype(input.dtype)], axis=-1
            )
        if self.input_preprocess is not None:
            input, gate_logits = self.input_preprocess(input, gate_logits, capacity)

        k = self.k
        prob, max_prob = self.fused_gate_logits_process(gate_logits, token_type_ids)

        with profile("dispatch_op"):
            corr_bias = (
                self.moe_statics.e_score_correction_bias[0].detach()
                if self.use_correction_bias
                else None
            )

            (
                dispatched_input,
                combine_weights_unnorm,
                scatter_index,
                dispatch_mask,
                _,
                fp8_dispatched_handle,
            ) = Fp8MoeGateDispatchAndQuant.apply(
                input, prob, corr_bias, k=k, capacity=capacity, use_pad=True
            )

        dispatch_mask = paddle.diff(F.pad(dispatch_mask, (1, 0)))
        if self.use_correction_bias:
            if self.gate.config.multimodel_experts:
                for i in range(len(self.moe_statics.expert_usage)):
                    self.moe_statics.expert_usage[i] += dispatch_mask[
                        self.gate.experts_type_mask[i]
                    ].detach()
            else:
                self.moe_statics.expert_usage[0] += dispatch_mask.detach()
        dispatched_input.stop_gradient = False
        combine_weights_unnorm.stop_gradient = False
        scatter_index.stop_gradient = True
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
                prob = (prob.reshape([p.shape[0], k, -1]) * p).reshape([p.shape[0], -1])
        if self.gate.norm_gate_logits:
            combine_weights = combine_weights_unnorm / paddle.clip(
                combine_weights_unnorm.sum(-1, keepdim=True), min=1e-12
            )
        else:
            combine_weights = combine_weights_unnorm
        combine_weights = combine_weights.cast("bfloat16")

        def reshape_for_a2a(tensor):
            return tensor.reshape(
                [
                    self.world_size * self.num_local_experts,
                    capacity,
                    -1,
                ]
            )

        dispatched_input = reshape_for_a2a(dispatched_input)
        fp8_dispatched_handle["scale"] = reshape_for_a2a(fp8_dispatched_handle["scale"])
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
            fp8_dispatched_handle,
        )

    def gate_and_distpach(self, input, token_type_ids):

        seqlen, d_model = input.shape
        args = ()
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape([-1])
            args = (token_type_ids,)

        use_fuse = isinstance(self.gate, (TopKGateFused))
        if use_fuse:
            if self.use_norm_gate_recompute:
                (
                    gate_logits,
                    capacity,
                    router_loss,
                    norm_res,
                ) = self.fused_norm_gate(input)
                input = norm_res
            else:
                (
                    gate_logits,
                    capacity,
                    router_loss,
                ) = self.gate(input, *args)
        else:
            (
                capacity,
                dispatch_mask,
                combine_weights,
                scatter_index,
                router_loss,
                gate_logits,
            ) = self.gate(
                input,
                *args,
                correction_bias=(
                    self.moe_statics.e_score_correction_bias[0]
                    if self.use_correction_bias
                    else None
                ),
            )
            prob = None
        if self.config.moe_multimodal_paired_experts:
            assert token_type_ids is not None
            input = paddle.concat(
                [input, token_type_ids.unsqueeze(-1).astype(input.dtype)], axis=-1
            )
        if self.input_preprocess is not None:
            input, gate_logits = self.input_preprocess(input, gate_logits, capacity)
        if use_fuse:
            k = self.k
            prob, max_prob = self.fused_gate_logits_process(gate_logits, token_type_ids)

            assert moe_ops is not None
            with profile("dispatch_op"):
                if (
                    "corr_bias"
                    in inspect.signature(moe_ops.moe_gate_dispatch).parameters
                ):
                    if self.use_correction_bias:
                        compat_args = (self.moe_statics.e_score_correction_bias[0],)
                    else:
                        compat_args = (None,)
                else:
                    assert (
                        not self.use_correction_bias
                    ), "correction bias not supported, rebuild moe-ops"
                    compat_args = ()
                if not self.config.use_ep_comm_overlap:
                    if self._rr_moe_gate_dispatch is None:
                        (
                            dispatched_input,
                            combine_weights_unnorm,
                            scatter_index,
                            dispatch_mask,
                            _,
                        ) = moe_ops.moe_gate_dispatch(
                            input,
                            prob,
                            *compat_args,
                            k=k,
                            capacity=capacity,
                            use_pad=True,
                        )
                    else:
                        (
                            dispatched_input,
                            combine_weights_unnorm,
                            scatter_index,
                            dispatch_mask,
                            _,
                        ) = self._rr_moe_gate_dispatch(
                            input,
                            prob,
                            compat_args,
                            k=k,
                            capacity=capacity,
                            use_pad=True,
                        )
                else:
                    (
                        dispatched_input,
                        combine_weights_unnorm,
                        scatter_index,
                        dispatch_mask,
                        _,
                    ) = moe_ops.moe_gate_dispatch_permute(
                        input,
                        prob,
                        *compat_args,
                        k=k,
                        capacity=capacity,
                        world_size=self.group.nranks,
                    )
            dispatch_mask = paddle.diff(F.pad(dispatch_mask, (1, 0)))
            if self.use_correction_bias and framework._dygraph_tracer()._has_grad:
                if self.gate.config.multimodel_experts:
                    for i in range(len(self.moe_statics.expert_usage)):
                        self.moe_statics.expert_usage[i] += dispatch_mask[
                            self.gate.experts_type_mask[i]
                        ].detach()
                else:
                    self.moe_statics.expert_usage[0] += dispatch_mask.detach()
            dispatched_input.stop_gradient = False
            combine_weights_unnorm.stop_gradient = False
            scatter_index.stop_gradient = True
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
            if self.gate.norm_gate_logits:
                combine_weights = combine_weights_unnorm / paddle.clip(
                    combine_weights_unnorm.sum(-1, keepdim=True), min=1e-12
                )
            else:
                combine_weights = combine_weights_unnorm
            combine_weights = combine_weights.cast(dispatched_input.dtype)
        else:
            dispatched_input = dispatching(
                input,
                dispatch_mask,
                scatter_index,
                num_experts=self.world_size * self.num_local_experts,
                capacity=capacity,
            )
            if self.use_correction_bias and framework._dygraph_tracer()._has_grad:
                usage = paddle.bincount(
                    scatter_index.reshape([-1]) // capacity,
                    minlength=self.world_size * self.num_local_experts,
                )
                assert (
                    not self.config.multimodel_experts
                ), "correction bias not supported, use top2-fused gate"
                self.moe_statics.expert_usage[0] += usage.detach()
        if not self.config.use_ep_comm_overlap:
            dispatched_input = dispatched_input.reshape(
                [
                    self.world_size * self.num_local_experts,
                    capacity,
                    (
                        d_model
                        if not self.config.moe_multimodal_paired_experts
                        else d_model + 1
                    ),
                ]
            )
        else:
            assert (
                len(dispatched_input.shape) == 4
                and dispatched_input.shape[1] == self.world_size
                and dispatched_input.shape[0] == self.num_local_experts
            ), (
                f"When using ep_comm_overlap, moe_gate_dispatch_permute is needed. "
                f"Expected dispatched_input to have shape[1] == {self.world_size} "
                f"and shape[0] == {self.num_local_experts}, "
                f"but got shape {dispatched_input.shape}"
            )
            dispatched_input = dispatched_input
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

    def combine_expert_output(self, expert_output, combine_weights, scatter_index):

        expert_output = expert_output.reshape([-1, expert_output.shape[-1]])
        use_fuse = isinstance(self.gate, (TopKGateFused))
        combine_fn = combining_fused if use_fuse else combining
        combined_output = combine_fn(expert_output, combine_weights, scatter_index)

        if self.output_postprocess is not None:
            combined_output = self.output_postprocess(combined_output)
        return combined_output

    def forward_single_stage(self, dispatched_input, stage_id):
        assert isinstance(self.experts, nn.LayerList)
        return self.experts[stage_id](dispatched_input)

    def forward(
        self,
        input: Tensor,
        token_type_ids=None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:

        if input.ndim == 3:
            orig_shape = input.shape
            input = input.reshape([-1, input.shape[-1]])
        else:
            orig_shape = None
        assert (
            len(input.shape) == 2
        ), f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"
        hidden_size = input.shape[1]
        if token_type_ids is not None:
            token_type_ids = token_type_ids.clone()[:, :-1]
            if self.config.sequence_parallel:
                token_type_ids = token_type_ids.reshape([-1])
                token_type_ids = ScatterOp.apply(token_type_ids)
                token_type_ids.stop_gradient = True

        assert self.gate is not None
        if hasattr(self, "rng") and self.rng.random() < self.all_to_all_dropout:
            orig_shape_2 = input.shape
            if self.config.moe_multimodal_paired_experts:
                assert token_type_ids is not None
                input = paddle.concat(
                    [input, token_type_ids.unsqueeze(-1).astype(input.dtype)], axis=-1
                )
            output = self.forward_experts(input)
            output += self.gate.weight.sum() * 0.0
            output = output.reshape(orig_shape or orig_shape_2)
            return output, None, 0

        is_first_fwd = not framework._dygraph_tracer()._has_grad
        use_async = self.shared_experts is not None
        if in_auto_parallel_align_mode():
            gate_input = paddle.assign(input)
        else:
            gate_input = input

        use_fp8_fuse_node = (
            self.config.use_combine_before_a2a and self.config.use_fp8_fuse_node
        )
        use_fp8_dispatch_a2a = self.config.use_fp8_dispatch_a2a and use_fp8_fuse_node

        with profile("fused_gate_and_dispatch"):
            fp8_dispatched_handle = None
            if use_fp8_dispatch_a2a:
                (
                    dispatched_input,
                    combine_weights,
                    dispatch_mask,
                    scatter_index,
                    router_loss,
                    gate_logits,
                    gate_prob,
                    fp8_dispatched_handle,
                ) = self.gate_distpach_and_quant(gate_input, token_type_ids)
            else:
                (
                    dispatched_input,
                    combine_weights,
                    dispatch_mask,
                    scatter_index,
                    router_loss,
                    gate_logits,
                    gate_prob,
                ) = self.gate_and_distpach(gate_input, token_type_ids)

        if self.config.use_combine_before_a2a:
            assert (
                not self.config.use_ep_comm_overlap
            ), "Dont support use_ep_comm_overlap"
            assert (
                moe_combine_no_weight is not None
            ), "use_combine_before_a2a can only use with moe_combine_no_weight op, please install it first."
            cw_shape = combine_weights.shape
            si_shape = scatter_index.shape
            scatter_index = scatter_index.reshape([-1])

            token_combine_weights = paddle.zeros(
                [cw_shape[0] * cw_shape[1]], dtype=combine_weights.dtype
            )
            token_combine_weights = paddle.scatter(
                token_combine_weights,
                scatter_index,
                combine_weights.reshape([-1]),
                overwrite=False,
            )

            token_combine_weights = token_combine_weights.reshape(
                [cw_shape[0], cw_shape[1], 1]
            )
            token_combine_weights = AlltoAll.apply(token_combine_weights, self.group)

        if not self.config.use_ep_comm_overlap:
            if use_fp8_dispatch_a2a:
                shared_out = (
                    self.shared_experts(input)
                    if self.shared_experts is not None
                    else None
                )
            else:
                with profile("moe_comm_and_shared_expert"):
                    if use_async:
                        dispatched_input, shared_out = AlltoAllAsync.apply(
                            dispatched_input,
                            input,
                            group=self.group,
                            fn=self.shared_experts,
                            is_first_fwd=is_first_fwd,
                        )
                    else:
                        dispatched_input = AlltoAll.apply(dispatched_input, self.group)

            expert_out = (
                recompute(self.forward_experts, dispatched_input)
                if self.recompute and self.training
                else self.forward_experts(dispatched_input)
            )

            if self.config.use_combine_before_a2a:
                token_combine_weights = token_combine_weights.clone().reshape(
                    expert_out.shape[:-1] + [1]
                )
                expert_out = expert_out * token_combine_weights
        else:
            assert (
                len(dispatched_input.shape) == 4
                and dispatched_input.shape[1] == self.world_size
                and dispatched_input.shape[0] == self.num_local_experts
            ), (
                f"When using ep_comm_overlap, moe_gate_dispatch_permute is needed. "
                f"Expected dispatched_input to have shape[1] == {self.world_size} "
                f"and shape[0] == {self.num_local_experts}, "
                f"but got shape {dispatched_input.shape}"
            )
            with profile("moe_comm_and_forward_expert"):
                expert_out = AlltoAllExpertOverlap.apply(
                    dispatched_input,
                    self.group,
                    self.num_local_experts,
                    self.experts,
                    is_first_fwd=is_first_fwd,
                )
                if self.shared_experts is not None:
                    shared_out = self.shared_experts(input)

        with profile("moe_comm_and_calc_routerloss"):
            expert_out, router_loss2 = AlltoAllAsync.apply(
                expert_out,
                router_loss,
                combine_weights,
                dispatch_mask,
                gate_logits,
                gate_prob,
                token_type_ids,
                group=self.group,
                fn=self.calc_router_loss_and_logging,
                is_first_fwd=is_first_fwd,
            )

        with profile("combine"):
            if self.config.use_combine_before_a2a:
                expert_out = expert_out.reshape([-1, hidden_size])

                scatter_index = scatter_index.reshape(si_shape)
                combined_output = moe_combine_no_weight(
                    expert_out, combine_weights, scatter_index, epsilon=1e-15
                )
            else:
                combined_output = self.combine_expert_output(
                    expert_out, combine_weights, scatter_index
                )

        if self.shared_experts is not None:
            combined_output += shared_out

        if orig_shape:
            combined_output = combined_output.clone().reshape(
                orig_shape[:-1] + [combined_output.shape[-1]]
            )
        return combined_output, combine_weights, router_loss2, gate_logits


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


def detach_and_requires_grad_(*args):
    """detach_and_requires_grad_"""
    ret = [a.detach() if a is not None else None for a in args]
    for r, a in zip(ret, args):
        if a is not None:
            r.stop_gradient = a.stop_gradient
    return ret


def bpr_preprocess(input, logits, capacity, buffer):
    """impletment bpr sorting"""
    assert input.ndim == 2, input.shape
    idx = paddle.argsort(logits.max(-1), axis=0, descending=True)
    input = input[idx]
    logits = logits[idx]
    buffer["idx"] = idx
    return input, logits


def bpr_postprocess(output, buffer):
    """bpr sorting"""
    idx = buffer.pop("idx")
    rev_idx = paddle.argsort(idx)
    output = output[rev_idx]
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
        k=2,
        enable_pbr: bool = False,
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

        if enable_pbr:
            logger.info("using BPR")
            prepost_process_buffer = {}
            self.input_preprocess = partial(
                bpr_preprocess, buffer=prepost_process_buffer
            )
            self.output_postprocess = partial(
                bpr_postprocess, buffer=prepost_process_buffer
            )
        else:
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
            use_fuse = isinstance(self.gate, (TopKGateFusedAuto))
            combine_fn = combining_fused_auto if use_fuse else combining
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
