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

"""_summary_

Returns:
    _type_: _description_
"""
from typing import Tuple, List, Optional
import logging
from collections import namedtuple
from functools import partial
import inspect

import paddle
from paddle import framework
from paddle import nn
from paddle.distributed.communication import stream
import paddle.nn.functional as F

from paddle.autograd import PyLayer
from paddle.distributed.communication.group import Group
from paddle.distributed.fleet.utils import recompute
from paddle.distributed import fleet
from paddle.distributed import in_auto_parallel_align_mode

import paddle.distributed as dist
from paddle import Tensor
from paddleformers.utils.tools import get_env_device

from models.moe.top2_gate_auto_auto import (
    TopKGateFused,
    cast_if_needed,
)
from models.sequence_parallel_utils_auto import ScatterOp
from models.utils import (
    global_training_logs_enabled,
    manual_backward,
)

from models.comm_utils import profile

from models.moe.moe_utils import MOEAllGatherDispatcher


from paddle.incubate.nn.functional import (
    moe_combine,
)

try:
    from paddle.incubate.nn.functional import (
        moe_gate_dispatch_and_quant,
    )
except ImportError:
    moe_gate_dispatch_and_quant = None

try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {} 
try:
    import moe_router_loss_ops
except ImportError:
    moe_router_loss_ops = None


try:
    from paddle import scatter_add_
except ImportError:
    scatter_add_ = None

try:
    from bincount_ops import int_bincount
except ImportError:
    int_bincount = None

logger = logging.getLogger(__name__)

if get_env_device() == "xpu":
    try:
        from paddle_xpu_nn import moe_gate_dispatch as xpu_moe_gate_dispatch
    except ImportError:
        xpu_moe_gate_dispatch = None
        logger.warning("`xpu moe dispatch` not found")
else:
    try:
        import moe_ops
    except ImportError:
        moe_ops = None
        logger.warning(
            "`moe-ops` not found, run "
            "`python3  src/ernie_core/ops/moe/setup.py  install` to install"
        )

    try:
        import moe_ops_fp8
    except ImportError:
        moe_ops_fp8 = None
        logger.warning(
            "`moe-ops` not found, run "
            "`python3  src/ernie_core/ops/moe/setup_fp8.py  install` to install"
        )

try:
    from moe_combine import moe_combine_no_weight
except ImportError:
    moe_combine_no_weight = None


try:
    import fused_ln as fused
except ImportError:
    logger.warning(
        "fused-ln not found, run `python src/ops/fused_ln_setup.py install` to build fused ln"
    )
    fused = None

try:
    from custom_setup_ops import matmul_bwd
except ImportError:
    matmul_bwd = None


GateOutput = namedtuple(
    "GateOutput",
    [
        "aux",
        "z",
        "logits",
    ],
)


class GateCombine_ori(PyLayer):
    """GateCombine_ori"""

    @staticmethod
    def forward(ctx, x, combine_weights, scatter_index):
        """
        Input:
            x:  [seqlen * k, hidden_size]
            combine_weights: [seqlen, k]
            scatter_index: [seqlen, k]
        Output:
            y: [seqlen, hidden_size]
        """
        ctx.x = x
        ctx.combine_weights = combine_weights
        ctx.scatter_index = scatter_index
        assert moe_combine is not None
        ret = moe_combine.moe_combine(x, combine_weights, scatter_index)
        return ret

    @staticmethod
    def backward(ctx, grad_y, *_):
        """
        Input:
            grad_y:  [seqlen, hidden_size]
            combine_weights: [seqlen, k]
            scatter_index: [seqlen, k]
        Output:
            grad_x: [seqlen * k, hidden_size]
            grad_combine_weight: [seqlen, k]

        """

        assert moe_combine is not None
        grad_x, grad_combine_weight_helper = moe_combine.moe_combine_bwd(
            ctx.x, ctx.combine_weights, ctx.scatter_index, grad_y
        )

        grad_combine_weight = grad_combine_weight_helper.sum(-1)
        return grad_x, grad_combine_weight.reshape(ctx.combine_weights.shape), None


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
        x_gatherd = F.embedding(scatter_index, x)  # [s,k,dim]
        return x_gatherd.squeeze(-2)
    ret = GateCombine_ori.apply(x, combine_weights, scatter_index)
    ret.stop_gradient = False
    return ret


def recompute_fwd_gate_up_func(config, layer_idx):

    if "recompute_fwd_gate_up" in config.fp8_mem_configs:
        if isinstance(config.fp8_mem_configs["recompute_fwd_gate_up"], bool):
            return config.fp8_mem_configs["recompute_fwd_gate_up"]
        if isinstance(config.fp8_mem_configs["recompute_fwd_gate_up"], list):
            return layer_idx in config.fp8_mem_configs["recompute_fwd_gate_up"]

    return False


class MoEStatics(nn.Layer):

    def __init__(self, config, layer_idx):
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
            ), "assume expert group has same size, got: {config.moe_num_experts}"

        with paddle.utils.unique_name.guard(f"mm_layer_{layer_idx}_"):
            num_experts_groups = (
                len(config.moe_num_experts) if config.multimodel_experts else 1
            )
            p = self.create_parameter(
                shape=[num_experts_groups, num_experts],
                dtype="float32",
                is_bias=True,
                attr=paddle.ParamAttr(
                    name=paddle.utils.unique_name.generate("corr_bias")
                ),
            )

            p.stop_gradient = False
            self.e_score_correction_bias = p
            self.e_score_correction_bias.is_distributed = True
            self.e_score_correction_bias.unused_param = True
            if getattr(config, "build_skip_comm_buffer", False):
                self.e_score_correction_bias.color = {
                    "color": "skip_comm",
                    "group": paddle.distributed.new_group(
                        [paddle.distributed.get_rank()]
                    ),
                }
            p = paddle.zeros(
                shape=[num_experts_groups, num_experts],
                dtype="int64",
            )
            p.stop_gradient = True
            self.expert_usage = p
            # self.expert_usage.is_distributed = True


def dispatching(x, dispatch_mask, scatter_index, num_experts, capacity):

    output = None
    # init_output = paddle.zeros([num_experts * capacity, x.shape[-1]], dtype='float32')
    # output = init_output + 0. * x.sum()
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
    # num_k = 2
    x = paddle.gather(x, scatter_index).reshape([-1, num_k, dim])  # [seq,2,dim]
    return paddle.matmul(combine_weights, x).squeeze(
        1
    )  # [seq,1,2] @ [seq,2,dim] -> [seq,1,dim]


def fuse_logging(gate_logits, combine_weights, token_type_ids):
    """fuse_logging"""
    with paddle.no_grad():
        gate_expert_per_token_type_0, gate_expert_per_token_type_1 = None, None
        gate_experts_per_token = None
        ce = moe_router_loss_ops.cal_cross_entropy_info(gate_logits).mean(0)
        if token_type_ids is not None:
            (
                gate_expert_per_token_type_0,
                gate_expert_per_token_type_1,
                gate_experts_per_token,
            ) = moe_router_loss_ops.cal_gate_experts_per_token_info(
                combine_weights, token_type_ids
            )
        else:
            gate_experts_per_token = paddle.count_nonzero(combine_weights) / (
                gate_logits.shape[0]
            )

        return (
            gate_expert_per_token_type_0,
            gate_expert_per_token_type_1,
            gate_experts_per_token,
            ce,
        )


class GateCombine(PyLayer):
    @staticmethod
    def forward(ctx, x, combine_weights, scatter_index):
        ctx.x = x
        ctx.combine_weights = combine_weights
        ctx.scatter_index = scatter_index
        ret = moe_combine(x, combine_weights, scatter_index)
        return ret

    @staticmethod
    def backward(ctx, grad_y, *_):
        # assert moe_combine is not None
        grad_x, grad_combine_weight_helper = paddle._C_ops.moe_combine_grad(
            ctx.x, ctx.combine_weights, ctx.scatter_index, grad_y
        )
        grad_combine_weight = grad_combine_weight_helper.sum(-1)
        return grad_x, grad_combine_weight.reshape(ctx.combine_weights.shape), None


class Fp8MoeGateDispatchAndQuant(paddle.autograd.PyLayer):
    """Fp8MoeGateDispatchAndQuant"""

    @staticmethod
    def forward(
        ctx, x, gate_logtis, corr_bias, k, capacity, use_pad, use_pow2_scale=True
    ):
        """forward"""
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
        """backward"""
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
    """
    AlltoAll w/ backward
    """

    @staticmethod
    def forward(ctx, x, group, sync_op=True):
        """
        All-to-all communication in the group.
        """
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
        """backward"""
        return AlltoAll.apply(*dx, group=ctx.group)


class AlltoAllExpertOverlap(PyLayer):
    """
    AlltoAllExpertOverlap w/ backward
    """

    @staticmethod
    def forward(
        ctx, input, group, num_local_experts, forward_func_dict, is_first_fwd=False
    ):
        """forward"""
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
        """backward"""
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
    """
    AlltoAll async w/ backward
    """

    @staticmethod
    def forward(ctx, x, *fn_args, group=None, fn=None, is_first_fwd=False):
        """
        All-to-all communication in the group.
        Args:
            x: Tensor
            args: List[Any], argument(s) to `fn`
            group: ProcessGroup
            fn: callable, called while doing alltoall
            is_first_fwd: if using recompute, don't record bacward when first forward
        Returns:
            x: Tensor
            fn_out: List[Tensor]
        """
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
        """backward"""
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


class FusedNormGateFunc(paddle.autograd.PyLayer):
    """recompute of postnorm and gate"""

    @staticmethod
    def forward(ctx, x, rms_norm_weight, moe_gate_weight, eps):
        """doc"""
        ctx.dtype = paddle.float32
        norm_output, invar = fused.fused_rms_norm(x, rms_norm_weight, eps)
        with paddle.amp.auto_cast(False):
            gate_logits = F.linear(
                cast_if_needed(norm_output, ctx.dtype),
                cast_if_needed(moe_gate_weight, ctx.dtype),
            )

        ctx.save_for_backward(x, rms_norm_weight, moe_gate_weight, eps)
        return gate_logits, norm_output

    @staticmethod
    def backward(ctx, d_gate_logits, d_norm_output):
        """doc"""
        x, rms_norm_weight, moe_gate_weight, eps = ctx.saved_tensor()
        # recompute rmsnorm
        norm_output, invar = fused.fused_rms_norm(x, rms_norm_weight, eps)
        # with paddle.amp.auto_cast(False):
        d_norm_output_linear, d_moe_gate_weight = matmul_bwd(
            cast_if_needed(norm_output, ctx.dtype),
            cast_if_needed(moe_gate_weight, ctx.dtype),
            d_gate_logits,
            False,
            False,
        )
        d_norm_output_linear, d_moe_gate_weight = cast_if_needed(
            d_norm_output_linear, norm_output.dtype
        ), cast_if_needed(d_moe_gate_weight, moe_gate_weight.dtype)
        d_norm_output = d_norm_output + d_norm_output_linear
        dx, d_rms_norm_weight = fused.fused_rms_norm_grad_func(
            x, rms_norm_weight, invar, d_norm_output, eps
        )

        return dx, d_rms_norm_weight, d_moe_gate_weight


class FusedNormGateMoe(paddle.nn.Layer):
    """recompute of postnorm and gate"""

    def __init__(self, gate, rms_norm_weight, eps) -> None:
        """doc"""
        super().__init__()
        self.rms_norm_weight = rms_norm_weight
        self.gate = gate
        self.eps = eps

    def forward(self, x):
        """doc"""
        moe_gate_weight = self.gate.get_gate_weight(True)
        capacity = self.gate.get_capacity(x.shape[0])

        router_loss = paddle.zeros([1], dtype="float32")
        router_loss.stop_gradient = False

        gate_logits, norm_output = FusedNormGateFunc.apply(
            x, self.rms_norm_weight, moe_gate_weight, self.eps
        )
        return gate_logits, capacity, router_loss, norm_output


class MOELayer(nn.Layer):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)

        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (paddle.nn.Layer):
            gate network
        expert (paddle.nn.LayerList):
            expert network, LayerList 长度是 per_device 上的 expert 数。
        group (paddle.ProgressGroup)
        recompute: 启用MOE内recomupte
    Returns:
        output
        combine_weight
        router-loss
    """

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
        self.enable_logging = enable_logging
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
            p.expert = not (self.is_mp_moe or is_dummy_moe)  # type: ignore
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
        # assert self.world_size > 1, f'moe-group not found, world_size {self.world_size}'
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

        if enable_bpr:
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

    def add_gate_recompute_func(self, post_norm_weight, post_norm_eps):
        """Add FusedNormGateMoe recompute function"""
        self.config.use_norm_gate_recompute = True
        self.fused_norm_gate = FusedNormGateMoe(
            self.gate, post_norm_weight, post_norm_eps
        )

    def forward_experts(self, dispatched_input):
        """
        call experts sequently
        Args:
            dispatched_input: Tensor[num_experts, capacity, dim]
        Returns:
            expert_output: Tensor[num_experts, capacity, dim]
        """
        with profile("fwd-expert"):
            dispatched_input = dispatched_input.reshape(
                [
                    self.world_size,
                    self.num_local_experts,
                    -1,
                    dispatched_input.shape[-1],
                ]
            )  # [e,1,c,m]
            expert_outputs = []
            if isinstance(self.experts, nn.LayerList):

                chunks = dispatched_input.transpose([1, 0, 2, 3]).contiguous().unbind(0)
                assert len(chunks) == len(self.experts), (
                    len(chunks),
                    len(self.experts),
                )
                for chunk, expert in zip(chunks, self.experts):
                    expert_outputs += [expert(chunk)]
                    # logger.info(
                    #     f"moe-fwd-expert: {chunk.shape}"
                    #     f'-> {expert_outputs[-1].shape}: {chunk.astype("float32").norm(axis=-1)}'
                    # )
                expert_output = paddle.stack(expert_outputs, axis=1)  # [ecm]

            else:
                dispatched_input = dispatched_input.transpose([1, 0, 2, 3])
                dispatched_input.contiguous()
                orig_shape = dispatched_input.shape
                chunks = dispatched_input.reshape([orig_shape[0], -1, orig_shape[-1]])
                chunks = self.experts(chunks)
                chunks = chunks.reshape(orig_shape[:-1] + [chunks.shape[-1]]).unbind(0)
                expert_outputs += chunks
                expert_output = paddle.stack(expert_outputs, axis=1)  # [ecm]
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
            # 处理 lm_prob
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
                    max_prob = lm_prob.max(-1, keepdim=True)  # [s_l, k, 1]
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
        """
        gate_distpach_and_quant
        """
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

        # TODO(zhangyuqin): 把这些代码封装起来, 增强代码复用
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

        scatter_index = scatter_index.transpose([1, 0])  # [k,s] ->[s,k]
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
            # capacity no use
            k = self.k
            prob, max_prob = self.fused_gate_logits_process(gate_logits, token_type_ids)
            if get_env_device() == "xpu":
                assert xpu_moe_gate_dispatch is not None
                (
                    dispatched_input,
                    combine_weights_unnorm,
                    scatter_index,
                    dispatch_mask,
                    _,
                ) = xpu_moe_gate_dispatch(input, prob, k, capacity, True)
            else:
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

            scatter_index = scatter_index.transpose([1, 0])  # [k,s] ->[s,k]
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
            )  # .clone()
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
            dispatched_input = dispatched_input  # .clone()
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
        log = {}
        router_loss, l_aux, orthogonal_loss, zloss = 0.0, None, None, None
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
            router_loss += (
                self.zero * gate_prob[0, 0]
            )  # must use gate prob to avoid zero pointer
        if self.gate.config.moe_orthogonal_loss_lambda:
            orthogonal_loss = self.gate._cal_orthogonal_loss(token_type, use_group)
            router_loss += (
                self.gate.moe_orthogonal_loss_lambda[token_type or 0] * orthogonal_loss
            )
        if self.gate.config.moe_z_loss_lambda and not in_auto_parallel_align_mode():
            zloss = self.gate._cal_z_loss(gate_logits, tokens_type_mask)
            router_loss += self.gate.moe_z_loss_lambda[token_type or 0] * zloss

        tracer = framework._dygraph_tracer()
        if self.enable_logging and global_training_logs_enabled() and tracer._has_grad:
            if l_aux is not None:
                log[f"aux_loss_layer_{self.layer_idx}"] = l_aux

            if orthogonal_loss is not None:
                log[f"orthogonal_loss_layer_{self.layer_idx}"] = orthogonal_loss

            if zloss is not None:
                log[f"zloss_layer_{self.layer_idx}"] = zloss

            global_training_logs.update(
                **log,
                **{
                    k.replace(f"_layer_{self.layer_idx}", ""): v for k, v in log.items()
                },
            )
            global_training_logs.update(
                **{
                    prefix + "_" + k.replace(f"_layer_{self.layer_idx}", ""): v
                    for k, v in log.items()
                }
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

            if self.enable_logging and global_training_logs_enabled():
                seqlen = gate_logits.shape[0]
                num_active = paddle.count_nonzero(combine_weights)
                gate_experts_per_token = num_active.item() / seqlen

                if token_type_ids is not None:
                    token_type_ids = token_type_ids.reshape([-1])
                    combine_weights_type_0 = combine_weights[token_type_ids == 0]
                    if combine_weights_type_0.size:
                        gate_expert_per_token_type_0 = (
                            paddle.count_nonzero(combine_weights_type_0).item()
                            / combine_weights_type_0.shape[0]
                        )
                        global_training_logs.update(
                            experts_per_token_text=gate_expert_per_token_type_0,
                        )

                    combine_weights_type_1 = combine_weights[token_type_ids == 1]
                    if combine_weights_type_1.size:
                        gate_expert_per_token_type_1 = (
                            paddle.count_nonzero(combine_weights_type_1).item()
                            / combine_weights_type_1.shape[0]
                        )
                        global_training_logs.update(
                            experts_per_token_image=gate_expert_per_token_type_1,
                        )

                ce = (
                    (-F.softmax(gate_logits, -1) * F.log_softmax(gate_logits, -1))
                    .sum(-1)
                    .mean(0)
                )
                _log = {
                    f"gate_prob_ce_layer_{self.layer_idx}": ce.item(),
                    f"experts_per_token_layer_{self.layer_idx}": gate_experts_per_token,
                }
                global_training_logs.update(
                    **_log,
                    **{
                        k.replace(f"_layer_{self.layer_idx}", ""): v
                        for k, v in _log.items()
                    },
                )
        else:
            seqlen = dispatch_mask.shape[0]
            dispatch_mask = dispatch_mask.unbind(-1)
            top1_gate_experts_per_token = (
                paddle.cast(dispatch_mask[0], dtype="float32").sum() / seqlen
            )
            if (
                self.enable_logging
                and global_training_logs_enabled()
                and len(dispatch_mask) == 2
            ):
                top2_gate_experts_per_token = (
                    paddle.cast(dispatch_mask[1], dtype="float32").sum() / seqlen
                )
                leakage_experts_per_token = (
                    paddle.cast(
                        (~dispatch_mask[0]) & (~dispatch_mask[1]), dtype="float32"
                    ).sum()
                    / seqlen
                )
                experts_per_token = (
                    top1_gate_experts_per_token + top2_gate_experts_per_token
                )
                global_training_logs.update(
                    experts_per_token=experts_per_token.detach(),
                    top1_experts_per_token=top1_gate_experts_per_token.detach(),
                    top2_experts_per_token=top2_gate_experts_per_token.detach(),
                    leakage_experts_per_token=leakage_experts_per_token.detach(),
                )
            elif (
                self.enable_logging
                and global_training_logs_enabled()
                and len(dispatch_mask) == 1
            ):
                experts_per_token = top1_gate_experts_per_token
                leakage_experts_per_token = (
                    paddle.cast(~dispatch_mask[0], dtype="float32").sum() / seqlen
                )
                global_training_logs.update(
                    experts_per_token=experts_per_token.detach(),
                    top1_experts_per_token=top1_gate_experts_per_token.detach(),
                    leakage_experts_per_token=leakage_experts_per_token.detach(),
                )

        return router_loss

    def combine_expert_output(self, expert_output, combine_weights, scatter_index):
        """
        Combine Expert output
        Args:
            expert_output: Tensor[num_experts, caapcity, dim]
            combine_weights:
        Returns:
            combined_output: Tensor[seqlen, dim]
        """
        expert_output = expert_output.reshape(
            [-1, expert_output.shape[-1]]
        )  # [e*1,c,m]
        use_fuse = isinstance(self.gate, (TopKGateFused))
        combine_fn = combining_fused if use_fuse else combining
        combined_output = combine_fn(expert_output, combine_weights, scatter_index)

        if self.output_postprocess is not None:
            combined_output = self.output_postprocess(combined_output)
        return combined_output

    def forward_single_stage(self, dispatched_input, stage_id):
        """forward_single_stage"""
        assert isinstance(self.experts, nn.LayerList)
        return self.experts[stage_id](dispatched_input)

    def all2all_expert_overlap(self, x, group):
        """all2all_expert_overlap"""
        all2all_tasks = []
        all2all_ins = paddle.unbind(x, axis=0)
        for stage_id in range(1):
            stage_input = all2all_ins[stage_id]
            x_out, task = AlltoAll.apply(stage_input, group=self.group, sync_op=False)
            all2all_tasks.append((task, x_out))

        expert_outputs = []
        for stage_id in range(self.num_local_experts):
            if stage_id + 1 != self.num_local_experts:
                stage_input = all2all_ins[stage_id + 1]
                x_out, task = AlltoAll.apply(
                    stage_input, group=self.group, sync_op=False
                )
                all2all_tasks.append((task, x_out))

            task, dispatched_input = all2all_tasks[stage_id]
            task.wait()
            expert_outputs_cur_stage = (
                recompute(self.forward_single_stage, dispatched_input, stage_id)
                if self.recompute and self.training
                else self.forward_single_stage(dispatched_input, stage_id)
            )
            expert_outputs.append(expert_outputs_cur_stage)

        expert_output = paddle.stack(expert_outputs, axis=1)
        return expert_output

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
            output += self.gate.weight.sum() * 0.0  # hack for grad
            output = output.reshape(orig_shape or orig_shape_2)  # [e*1,c,m]
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

        # TODO(shenliang03): to fuse one kernel to optimize
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


class MOEInferLayer(nn.Layer):

    def __init__(
        self,
        gate: nn.Layer,
        experts: List[nn.Layer],
        group: Group = None,
        recompute=False,
    ) -> None:

        super().__init__()
        self.gate = gate
        self.recompute = recompute
        logger.info(f"using infer moe recompute={recompute}")
        for p in self.gate.parameters():
            p.is_gate = True
        if type(experts) == nn.LayerList:
            self.experts = experts
        else:
            self.experts = nn.LayerList([experts])
        self.group = group
        for p in experts.parameters():
            p.expert = True  # type: ignore
            p.no_sync = True

        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)

        if self.world_size < 1:
            self.world_size = 1
        if self.rank < 0:
            self.rank = 0
        self.num_local_experts = len(self.experts)

    def forward(
        self,
        input: Tensor,
        token_type_ids=None,
    ) -> Tensor:
        """_summary_

        Args:
            input (Tensor): _description_

        Returns:
            Tensor: _description_
        """
        # assert len(input) == 1, "only single input Tensor supported"
        if input.ndim == 3:
            orig_shape = input.shape
            input = input.reshape([-1, input.shape[-1]])
        else:
            orig_shape = None
        assert (
            len(input.shape) == 2
        ), f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"

        # Implement Algorithm 2 from GShard paper.
        seqlen, d_model = input.shape

        # Reshape into S tokens by dropping sequence dimension.
        # reshaped_input = input.reshape(-1, d_model)
        # assert reshaped_input.shape[0] % len(self.experts) == 0,
        # f'num tokens must be order of number of local experts, {input[0].shape[0]} vs {len(self.experts)}'
        def fwdfn(dispatched_input):
            chunks = dispatched_input.unbind(1)
            expert_outputs = []
            for chunk, expert in zip(chunks, self.experts):
                expert_outputs += [expert(chunk)]
            expert_output = paddle.stack(expert_outputs, axis=1)  # [ecm]
            return expert_output

        assert self.gate is not None
        (
            capacity,
            dispatch_mask,
            combine_weights,
            scatter_index,
            router_loss,
        ) = self.gate(input)

        dispatched_input = dispatching(
            input,
            dispatch_mask,
            scatter_index,
            num_experts=self.world_size * self.num_local_experts,
            capacity=capacity,
        )
        dispatched_input = dispatched_input.reshape(
            [self.world_size * self.num_local_experts, capacity, d_model]
        )
        #  dispatched_input = _AllToAll.apply(dispatched_input, self.group) #[ecm]
        dispatched_input = dispatched_input.reshape(
            [self.world_size, self.num_local_experts, -1, d_model]
        )  # [e,1,c,m]
        dispatched_input = dispatched_input[
            self.rank : (self.rank + 1)
        ]  # [1, local_experts, c, m]

        expert_output = (
            recompute(fwdfn, dispatched_input)
            if self.recompute and self.training
            else fwdfn(dispatched_input)
        )
        # expert_output = fwdfn(dispatched_input)
        #  expert_output = _AllToAll.apply(expert_output, self.group) #[ecm]
        if self.world_size > 1:
            tmp = []
            dist.all_gather(tmp, expert_output, group=self.group)
            expert_output = paddle.concat(tmp, axis=0)

        expert_output = expert_output.reshape(
            [self.world_size * self.num_local_experts * capacity, d_model]
        )  # [e*1,c,m]
        combined_output = combining(expert_output, combine_weights, scatter_index)

        # combined_output = paddle.einsum("sec,ecm->sm", combine_weights, expert_output)
        if orig_shape:
            combined_output = combined_output.reshape(orig_shape)
        top1_gate_experts_per_token = (
            paddle.cast(dispatch_mask[0], dtype="float32").sum() / seqlen
        )
        top2_gate_experts_per_token = (
            paddle.cast(dispatch_mask[1], dtype="float32").sum() / seqlen
        )
        leakage_experts_per_token = (
            paddle.cast(
                (~dispatch_mask[0]) & (~dispatch_mask[1]), dtype="float32"
            ).sum()
            / seqlen
        )

        experts_per_token = top1_gate_experts_per_token + top2_gate_experts_per_token
        global_training_logs.update(
            experts_per_token=experts_per_token.detach(),
            top1_experts_per_token=top1_gate_experts_per_token.detach(),
            top2_experts_per_token=top2_gate_experts_per_token.detach(),
            leakage_experts_per_token=leakage_experts_per_token.detach(),
        )
        return combined_output, combine_weights, router_loss, None


class MOELayerWithAllGatherDispatcher(MOELayer):
    """
    MOELayer with allgather dispatcher.
    """

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
        enable_bpr: bool = False,
        all_to_all_dropout=0,
        group_experts=False,
    ):
        super(MOELayerWithAllGatherDispatcher, self).__init__(
            gate=gate,
            experts=experts,
            layer_idx=layer_idx,
            shared_experts=shared_experts,
            group=group,
            recompute=recompute,
            enable_logging=enable_logging,
            k=k,
            enable_bpr=enable_bpr,
            all_to_all_dropout=all_to_all_dropout,
            group_experts=group_experts,
        )
        logger.info("Using MOELayerWithAllGatherDispatcher")
        assert get_env_device() == "xpu"
        assert isinstance(self.gate, TopKGateFused)
        assert self.shared_experts is not None
        local_expert_indices_offset = self.rank * self.num_local_experts
        self.expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]

    def gate_and_distpach(self, input, token_type_ids):
        """
        gate and dispatch
        """
        args = ()

        gate_logits, capacity, router_loss = self.gate(input, *args)

        if self.input_preprocess is not None:
            input, gate_logits = self.input_preprocess(input, gate_logits, capacity)

        moe_allgather_dispatcher_return = MOEAllGatherDispatcher.token_dispatcher(
            input,
            gate_logits,
            self.k,
            self.expert_indices,
            self.num_local_experts * self.world_size,
            self.num_local_experts,
        )
        global_hidden_states = moe_allgather_dispatcher_return.global_hidden_states
        dispatched_input = moe_allgather_dispatcher_return.dispatched_input
        combine_weights = moe_allgather_dispatcher_return.combine_weights
        scatter_index = moe_allgather_dispatcher_return.scatter_index
        gather_scatter_mask = moe_allgather_dispatcher_return.gather_scatter_mask
        dispatch_mask = moe_allgather_dispatcher_return.dispatch_mask
        tokens_per_expert = moe_allgather_dispatcher_return.tokens_per_expert

        dispatched_input.stop_gradient = False
        combine_weights.stop_gradient = False
        scatter_index.stop_gradient = True
        gather_scatter_mask.stop_gradient = True
        dispatch_mask.stop_gradient = True

        return (
            dispatched_input,
            combine_weights,
            gather_scatter_mask,
            dispatch_mask,
            scatter_index,
            router_loss,
            gate_logits,
            global_hidden_states,
            tokens_per_expert,
        )

    def forward_experts(
        self, dispatched_input, global_hidden_states, tokens_per_expert
    ):
        """
        call moe experts and share experts
        """
        tokens_per_expert_no_zero = list(
            filter(lambda x: x != 0, tokens_per_expert.tolist())
        )
        chunks_per_expert = paddle.split(
            dispatched_input, tokens_per_expert_no_zero, axis=0
        )
        assert len(chunks_per_expert) <= len(self.experts)
        moe_output = []
        offset = 0
        for index, cur_tokens in enumerate(tokens_per_expert.tolist()):
            if cur_tokens == 0:
                offset += 1
            else:
                cur_expert = self.experts[index]
                cur_chunk = chunks_per_expert[index - offset]
                moe_output.append(cur_expert(cur_chunk))
        hidden_states = paddle.concat(moe_output, axis=0)
        shared_expert_out = self.shared_experts(global_hidden_states)
        return hidden_states, shared_expert_out

    def forward(self, input, token_type_ids):
        """
        forward function
        """
        assert (
            len(input.shape) == 2
        ), f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"
        orig_shape = input.shape
        global_shape = [orig_shape[0] * self.world_size, orig_shape[1]]
        if token_type_ids is not None:
            token_type_ids.stop_gradient = True
        assert self.gate is not None

        (
            dispatched_input,
            combine_weights,
            gather_scatter_mask,
            dispatch_mask,
            scatter_index,
            router_loss,
            gate_logits,
            global_hidden_states,
            tokens_per_expert,
        ) = self.gate_and_distpach(input, token_type_ids)

        expert_out, shared_out = (
            recompute(
                self.forward_experts,
                dispatched_input,
                global_hidden_states,
                tokens_per_expert,
            )
            if self.recompute and self.training
            else self.forward_experts(
                dispatched_input, global_hidden_states, tokens_per_expert
            )
        )
        combined_output = MOEAllGatherDispatcher.token_combine(
            expert_out,
            shared_out,
            combine_weights,
            scatter_index,
            gather_scatter_mask,
            global_shape,
        )
        if self.shared_experts.down_proj.bias is not None:
            combined_output = combined_output + self.shared_experts.down_proj.bias
        router_loss2 = self.calc_router_loss_and_logging(
            router_loss, combine_weights, dispatch_mask, gate_logits, token_type_ids
        )

        return combined_output, combine_weights, router_loss2, gate_logits
