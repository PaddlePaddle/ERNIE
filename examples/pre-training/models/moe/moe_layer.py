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
import numpy as np

import paddle
from paddle import framework
from paddle import nn
from paddle.distributed.communication import stream
import paddle.nn.functional as F

from paddle.autograd import PyLayer
from paddle.distributed.communication.group import Group
from paddle.distributed.fleet.utils import recompute
from paddle.distributed import fleet

import paddle.distributed as dist
from paddle import Tensor
from paddleformers.utils.tools import get_env_device

from models.moe.top2_gate import (
    TopKGateFused,
    DeepEPTop2Gate,
    cast_if_needed,
)
from models.sequence_parallel_utils import ScatterOp
from models.utils import (
    global_training_logs_enabled,
    manual_backward,
    FakeGather,
    FusedUnpermutation,
)

from models.comm_utils import profile

from models.moe.moe_utils import MOEAllGatherDispatcher

from models.moe.token_dispatcher.token_dispatcher import MoEFlexTokenDispatcher
from models.moe.token_dispatcher.fp8_utils import (
    has_config,
    FP8_ALIGN,
    tilewise_quant,
    ExpertsGroupGemmNode,
    ExpertsGroupGemmContiguousNode,
    ExpertsGroupGemmWLCHNode,
)
from models.moe.token_dispatcher.moe_utils import (
    UnZipNode,
    ZipNode,
    topk_to_permuted_indices_single,
    inplace_offload_if_needed,
    tokens_zip_unique_add_with_subbatch,
    merge_subbatch_cast,
)
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
    global_training_logs = {}  # 没有erniebot的环境下无法打印 debug 量
try:
    import moe_router_loss_ops
except ImportError:
    moe_router_loss_ops = None
try:
    from paddle.distributed import in_auto_parallel_align_mode
except:

    def in_auto_parallel_align_mode():
        """
        hack for paddlenlp develop branch.
        """
        return False


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
    import TokenDispatcherUtils as TDU
except ImportError:
    TDU = None


try:
    import FusedQuantOps as FQO
except ImportError:
    FQO = None

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

try:
    import FusedQuantOps
except ImportError:
    pass

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
        if get_env_device() == "xpu":
            assert xpu_moe_combine is not None
            return xpu_moe_combine(x, combine_weights, scatter_index)
        else:
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

        if get_env_device() == "xpu":
            assert xpu_moe_combine_bwd is not None
            grad_x, grad_combine_weight_helper = xpu_moe_combine_bwd(
                ctx.x, ctx.combine_weights, ctx.scatter_index, grad_y
            )
        else:
            assert moe_combine is not None
            grad_x, grad_combine_weight_helper = moe_combine.moe_combine_bwd(
                ctx.x, ctx.combine_weights, ctx.scatter_index, grad_y
            )
        # grad_combine_weight_helper is the same shape with grad x [seqlen * K, dim]
        # reduce the hidden shape
        # TODO: implement reduce in cuda ops
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

class Fp8MoeGateDispatchAndQuant(paddle.autograd.PyLayer):
    """Fp8MoeGateDispatchAndQuant"""

    @staticmethod
    def forward(
        ctx,
        x,
        gate_logtis,
        corr_bias,
        k,
        capacity,
        use_pad,
        use_pow2_scale=True,
    ):
        """forward"""
        assert moe_gate_dispatch_and_quant is not None, "Please use new version Paddle."
        with paddle.amp.auto_cast(enable=False):
            (
                out_fp8,
                scale,
                combine_weights,
                scatter_index,
                expert_offset,
                expert_id,
            ) = moe_gate_dispatch_and_quant(
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
        x_grad, gate_logits_grad = paddle._C_ops.moe_gate_dispatch_grad(
            ctx.combine_weights,
            ctx.scatter_index,
            ctx.expert_id,
            out_grad,
            combine_weights_grad,
            ctx.k,
            ctx.capacity,
            ctx.use_pad,
        )
        if ctx.has_corr_bias:
            return x_grad, gate_logits_grad, None
        else:
            return x_grad, gate_logits_grad


def recompute_fwd_gate_up_func(config, layer_idx):
    """
    根据配置和层索引确定是否重计算。
    """
    if "recompute_fwd_gate_up" in config.fp8_mem_configs:
        if isinstance(config.fp8_mem_configs["recompute_fwd_gate_up"], bool):
            return config.fp8_mem_configs["recompute_fwd_gate_up"]
        if isinstance(config.fp8_mem_configs["recompute_fwd_gate_up"], list):
            return layer_idx in config.fp8_mem_configs["recompute_fwd_gate_up"]

    return False


class MoEStatics(nn.Layer):
    """
    存放 MoE 统计信息
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self._cast_to_low_precision = False  # 兼容develop分支paddle
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
            # NOTE: 由于ZCC适配问题, e_score_correction_bias一定要stop_gradient=False
            # 但实际上e_score_correction_bias是在callback里更新的, 不需要有梯度。所以在使用
            # e_score_correction_bias时要使用detach(), 避免产生梯度重复更新
            # 因为 e_score_correction_bias 不会产生梯度，所以通过设置build_skip_comm_buffer
            # 跳过梯度通信，以支持sharding_overlap
            p.stop_gradient = False
            self.e_score_correction_bias = p
            self.e_score_correction_bias.is_distributed = True
            self.e_score_correction_bias.unused_param = True
            if getattr(config, "build_skip_comm_buffer", False):
                # e_score_correction_bias单独一个buffer和通信组，reduce时会跳过该buffer
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
    """
    根据 gate 结果重排 `x`,  按照 capacity 截断、padding

    Args:
        x (Tensor)[Seq, Dim]: 输入张量。
        dispatch_mask Tensor[Seq, 2]: 分发掩码列表。
        scatter_index Tensor[Seq, 2]: 分布索引列表。
        num_experts (int): 专家数量。
        capacity (int): 容量大小。

    Returns:
        Tensor [Expert*Capacity, Dim]: 分派后的输出张量。

    """
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
    """
    对输入的矩阵进行组合和聚合操作

    Args:
        x: Tensor[num_experts * capacity, dim] 待处理的输入矩阵，最后一维表示特征数量。
        combine_weights: Tensor[seq, 2] 包含每个特征的组合权重列表。
        scatter_index:   Tensor[seq, 2]: 表示要被聚合的索引元组，第一个元素为行索引，第二个元素为列索引。

    Returns:
        Tensor: 经过组合和聚合后的输出矩阵，形状为[n, dim * num_features]，其中n是输入矩阵中的样本数目。
    """
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


class FusionFP8Expert(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, hidden_states, custom_map):
        ctx.node = ExpertsGroupGemmNode(None, custom_map)

        t1 = hidden_states.transpose([1, 0, 2, 3]).contiguous()
        expert_num = t1.shape[0]
        tokens_num = t1.shape[1] * t1.shape[2]
        tokens_per_expert = paddle.full(
            [expert_num], fill_value=tokens_num, dtype="int32"
        )

        t1 = t1.reshape([-1, hidden_states.shape[-1]])
        out = ctx.node.forward_no_prob(t1, tokens_per_expert)

        expert_output = (
            out.reshape(hidden_states.shape).transpose([1, 0, 2, 3]).contiguous()
        )

        ctx.save_for_backward(tokens_per_expert)
        return expert_output

    @staticmethod
    def backward(ctx, output_grad):
        (tokens_per_expert,) = ctx.saved_tensor()

        t1 = output_grad.transpose([1, 0, 2, 3]).contiguous()
        t1 = t1.reshape([-1, output_grad.shape[-1]])

        dx = ctx.node.backward_no_prob(t1, tokens_per_expert)
        dx = dx.reshape(output_grad.shape).transpose([1, 0, 2, 3]).contiguous()
        return dx


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


class FusionFP8Expert(paddle.autograd.PyLayer):
    """FusionFP8Expert"""

    @staticmethod
    def forward(ctx, hidden_states, custom_map):
        """
            前向传播函数，将输入的隐藏状态转换为特定的形式，并进行计算。
        该函数接收以下参数：
            hidden_states (Tensor): 输入的隐藏状态，形状为（N, num_experts, sequence_length, hidden_size）。
            custom_map (Dict[str, int]): 包含自定义映射的字典，用于指定每个专家对应的权重矩阵。
        返回值是一个 Tensor，形状为（N, num_experts, sequence_length, hidden_size），表示经过计算后的输出。
        """
        ctx.node = ExpertsGroupGemmNode(None, custom_map)

        t1 = hidden_states.transpose([1, 0, 2, 3]).contiguous()
        expert_num = t1.shape[0]
        tokens_num = t1.shape[1] * t1.shape[2]
        tokens_per_expert = paddle.full(
            [expert_num], fill_value=tokens_num, dtype="int32"
        )

        t1 = t1.reshape([-1, hidden_states.shape[-1]])
        out = ctx.node.forward_no_prob(t1, tokens_per_expert)

        expert_output = (
            out.reshape(hidden_states.shape).transpose([1, 0, 2, 3]).contiguous()
        )

        ctx.save_for_backward(tokens_per_expert)
        return expert_output

    @staticmethod
    def backward(ctx, output_grad):
        """
            计算反向传播，返回输出的梯度。
        参数：
            ctx (Context): 上下文对象，包含了需要的信息。
            output_grad (Tensor): 输出的梯度张量，形状为（N, num_expert, seq_len, hidden_size）。
        返回值：
            dx (Tensor, Tensor): 返回一个元组，第一个元素是dx，形状为（N, num_expert, seq_len, hidden_size），表示输入的梯度；第二个元素是None。
            其中，dx = dE / dX，dE / dX = dL / dX，dL / dX = output_grad。
        """
        (tokens_per_expert,) = ctx.saved_tensor()

        t1 = output_grad.transpose([1, 0, 2, 3]).contiguous()
        t1 = t1.reshape([-1, output_grad.shape[-1]])

        dx = ctx.node.backward_no_prob(t1, tokens_per_expert)
        dx = dx.reshape(output_grad.shape).transpose([1, 0, 2, 3]).contiguous()
        return dx


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
        """
        初始化MoE层。

        Args:
            gate (nn.Layer): 智能门控层，用于选择需要使用的专家。
            experts (List[nn.Layer]): 需要使用的专家列表。
            layer_idx (int): 当前MoE层的索引。
            group (Group): 分布式通信组。默认值为None。
            recompute (bool): 是否在每个训练迭代中重新计算MoE输出。默认值为False。
        """
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
                if self.config.use_fp8_fuse_node:
                    expert_output = FusionFP8Expert.apply(dispatched_input, self)
                else:
                    chunks = (
                        dispatched_input.transpose([1, 0, 2, 3]).contiguous().unbind(0)
                    )
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
        """处理并合并 gate logits

        Args:
            gate_logits (_type_): _description_
            token_type_ids (_type_): _description_

        Returns:
            _type_: _description_
        """
        k = self.k
        moe_num_experts = gate_logits.shape[-1]
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
            # 处理 mm_prob
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
            # 处理非硬门和不需要token_type_ids的情况
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
        assert isinstance(
            self.gate, (TopKGateFused)
        ), "Only fused gate is supported."
        assert not self.config.use_ep_comm_overlap, "ep_comm_overlap is not supported"
        assert (
            self._rr_moe_gate_dispatch is None
        ), "rr_moe_gate_dispatch is not supported"
        assert moe_ops_fp8 is not None

        seqlen, d_model = input.shape
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
        # 目前只有 `SinkHornGate` aka Top1 gate 支持输入 token type ids
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape([-1])
            args = (token_type_ids,)

        use_fuse = isinstance(
            self.gate, (TopKGateFused)
        )
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
            # 避免recompute的时候重复统计
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
            # 避免recompute的时候重复统计
            if self.use_correction_bias and framework._dygraph_tracer()._has_grad:
                usage = paddle.bincount(
                    scatter_index.reshape([-1]) // capacity,
                    minlength=self.world_size * self.num_local_experts,
                )
                assert (
                    not self.config.multimodel_experts
                ), "correction bias not supported, use top2-fused gate"
                self.moe_statics.expert_usage[0] += usage.detach()
        # clone 保平安
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
        """
        在fused expert 的情况下，计算辅助 loss (Aux-loss, 正交 loss, z-loss) 并 打印 log
        """
        use_fuse = isinstance(
            self.gate, (TopKGateFused)
        )
        if use_fuse:
            assert gate_prob is not None
            if token_type_ids is not None and self.gate.config.moe_use_hard_gate:
                if not self.gate.weight.stop_gradient:
                    # 文参数训练时才计算。
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
                cap_factor = (
                    self.gate.cap[0]
                    if isinstance(self.gate.cap, (tuple, list))
                    else self.gate.cap
                )
                capacity = (
                    cap_factor * combine_weights.shape[0] // gate_logits.shape[-1]
                )
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
        use_fuse = isinstance(
            self.gate, (TopKGateFused)
        )
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
        # assert len(input) == 1, "only single input Tensor supported"
        if input.ndim == 3:
            orig_shape = input.shape
            # clone 保平安
            input = input.reshape([-1, input.shape[-1]])
        else:
            orig_shape = None
        assert (
            len(input.shape) == 2
        ), f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"
        sequence_len = input.shape[0]
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

        # 依赖关系: use_combine_before_a2a <- use_fp8_fuse_node <- use_fp8_dispatch_a2a
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

            # TODO(shenliang03): to async alltoall in the furture

            token_combine_weights = token_combine_weights.reshape(
                [cw_shape[0], cw_shape[1], 1]
            )
            token_combine_weights = AlltoAll.apply(token_combine_weights, self.group)
            # scatter_index = scatter_index.reshape(si_shape)

        if not self.config.use_ep_comm_overlap:
            if use_fp8_dispatch_a2a:
                # 为了反向做overlap, 在FP8FusedWLCHFunc内部做a2a, 外面就不用做a2a了。但是不要漏了shared_experts的计算
                shared_out = (
                    self.shared_experts(input)
                    if self.shared_experts is not None
                    else None
                )
            else:
                # 常规通信
                with profile("moe_comm_and_shared_expert"):
                    if use_async:
                        dispatched_input, shared_out = AlltoAllAsync.apply(
                            dispatched_input,
                            input,  # args to shared-experts
                            group=self.group,
                            fn=self.shared_experts,
                            is_first_fwd=is_first_fwd,
                        )
                    else:
                        dispatched_input = AlltoAll.apply(dispatched_input, self.group)

            if use_fp8_fuse_node:
                expert_out = FP8FusedWLCHFunc.apply(
                    dispatched_input,
                    token_combine_weights,
                    self,
                    recompute_fwd_gate_up=recompute_fwd_gate_up_func(
                        self.config, self.layer_idx
                    ),
                    dequant_input=("dequant_input" in self.config.fp8_mem_configs)
                    and self.config.fp8_mem_configs["dequant_input"],
                    fp8_dispatch_a2a=use_fp8_dispatch_a2a,
                    is_first_fwd=not framework._dygraph_tracer()._has_grad,
                    group=self.group,
                    fp8_dispatched_handle=fp8_dispatched_handle,
                )
            else:
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


class MlpNode:
    """
    The FusedMoeLayer class includes operations for unzipping, expert computation, and zipping.
    """

    def __init__(
        self,
        custom_map,
        max_topk,
        recompute_fwd_gate_up=False,
        dequant_input=False,
        use_expert_subbatch=False,
        recompute_unzipped=False,
        tokens_zip_unique_add_subbatch_rows=None,
        backward_subbatch_rows=None,
    ):
        """
        Constructor
        """
        self.token_dispatcher = custom_map.dispatcher
        self.use_expert_subbatch = use_expert_subbatch
        self.experts = custom_map.experts
        if recompute_unzipped:
            assert (
                use_expert_subbatch
            ), "use_expert_subbatch must be enabled when recompute_unzipped = True"
            assert (
                recompute_fwd_gate_up
            ), "recompute_fwd_gate_up must be enabled when recompute_unzipped = True"
            assert (
                dequant_input
            ), "dequant_input must be enabled with recompute_unzipped = True"
        self.recompute_unzipped = recompute_unzipped
        self.tokens_zip_unique_add_subbatch_rows = tokens_zip_unique_add_subbatch_rows

        if self.use_expert_subbatch:
            self.experts_group_gemm_node = [
                ExpertsGroupGemmContiguousNode(
                    custom_map,
                    recompute_fwd_gate_up=recompute_fwd_gate_up,
                    dequant_input=dequant_input,
                    expert_id=expert_id,
                    backward_subbatch_rows=backward_subbatch_rows,
                )
                for expert_id in range(len(custom_map.experts))
            ]
        else:
            self.experts_group_gemm_node = ExpertsGroupGemmContiguousNode(
                custom_map,
                recompute_fwd_gate_up=recompute_fwd_gate_up,
                dequant_input=dequant_input,
                backward_subbatch_rows=backward_subbatch_rows,
            )
        self.unzip_node = UnZipNode(self.token_dispatcher)
        self.zip_node = ZipNode(self.token_dispatcher)
        self.hs_2d_dispatched_fp8 = None
        self.hs_2d_dispatched_scale = None
        self.dispatched_indices = None
        self.dispatched_probs = None
        self.unzipped_probs = None
        self.tokens_per_expert = (
            self.token_dispatcher._comm_manager.tokens_per_expert_list
        )
        self.padding_token_per_experts = [
            (x + FP8_ALIGN - 1) // FP8_ALIGN * FP8_ALIGN for x in self.tokens_per_expert
        ]
        self.token_offsets = [0]
        for padding_token in self.padding_token_per_experts:
            self.token_offsets.append(self.token_offsets[-1] + padding_token)
        self.router_topk = max_topk

    def cached_tensors(self):
        """
        cached tensors
        """
        if self.experts_group_gemm_node is not None:
            if self.use_expert_subbatch:
                gemm_node_tensors = []
                for gemm_node in self.experts_group_gemm_node:
                    gemm_node_tensors.extend(gemm_node.cached_tensors())
            else:
                gemm_node_tensors = self.experts_group_gemm_node.cached_tensors()
        else:
            gemm_node_tensors = []

        return (
            gemm_node_tensors
            + self.unzip_node.cached_tensors()
            + self.zip_node.cached_tensors()
            + [
                self.hs_2d_dispatched_fp8,
                self.hs_2d_dispatched_scale,
                self.dispatched_indices,
                self.dispatched_probs,
                self.unzipped_probs,
                self.tokens_per_expert,
                self.router_topk,
            ]
        )

    def set_cached_tensors(self, tensors):
        """
        set_cached_tensors
        """
        idx = 0
        if self.experts_group_gemm_node is not None:
            if self.use_expert_subbatch:
                for expert_id, gemm_node in enumerate(self.experts_group_gemm_node):
                    num = len(gemm_node.cached_tensors())
                    gemm_node.set_cached_tensors(tensors[idx : idx + num])
                    idx += num
            else:
                num = len(self.experts_group_gemm_node.cached_tensors())
                self.experts_group_gemm_node.set_cached_tensors(
                    tensors[idx : idx + num]
                )
                idx += num

        num = len(self.unzip_node.cached_tensors())
        self.unzip_node.set_cached_tensors(tensors[idx : idx + num])
        idx += num

        num = len(self.zip_node.cached_tensors())
        self.zip_node.set_cached_tensors(tensors[idx : idx + num])
        idx += num

        (
            self.hs_2d_dispatched_fp8,
            self.hs_2d_dispatched_scale,
            self.dispatched_indices,
            self.dispatched_probs,
            self.unzipped_probs,
            self.tokens_per_expert,
            self.router_topk,
        ) = tensors[idx:]

    def clear_cached_tensors(self):
        """
        clear_cached_tensors
        """
        self.set_cached_tensors([None] * len(self.cached_tensors()))

    def reset_statue(self):
        """
        重置所有状态变量。

        Args:
            无。

        Returns:
            无。

        """
        self.dispatched_indices = None
        self.dispatched_probs = None
        self.unzipped_probs = None
        self.tokens_per_expert = None
        self.padding_token_per_experts = None
        self.router_topk = None
        self.release_mem()

    def release_mem(self):
        """
            释放内存，将变量置为None。
        这个函数应该在程序结束时调用，以便释放不再需要的资源。

        Args:
            无参数。

        Returns:
            无返回值，直接修改了类实例中的变量。
        """
        if self.use_expert_subbatch:
            for node in self.experts_group_gemm_node:
                node.reset_statue()
        else:
            self.experts_group_gemm_node.reset_statue()
        self.experts_group_gemm_node = None

    def subbatch_unzip_and_prepare_gemm_node(
        self, hs_2d_dispatched, zipped_expertwise_rowmap, expert_id
    ):
        """
        subbatch_unzip_and_prepare_gemm_node
        """
        hs_2d_dispatched, hs_2d_dispatched_scale = hs_2d_dispatched
        expert_out, expert_out_scale, expert_unzipped_idx = TDU.tokens_unzip_gather(
            hs_2d_dispatched,
            hs_2d_dispatched_scale,
            zipped_expertwise_rowmap,
            expert_id=expert_id,
            tokens_per_expert=self.tokens_per_expert,
            padding_multiplex=FP8_ALIGN,
        )
        gemm_node = self.experts_group_gemm_node[expert_id]
        gemm_node.input_fp8 = expert_out
        gemm_node.input_scale = expert_out_scale
        return expert_unzipped_idx

    @paddle.no_grad()
    def forward(self, hs_2d_dispatched, dispatched_indices, dispatched_probs):
        """
        对输入数据进行前向传播计算。

        Args:
            hs_2d_dispatched (Tensor): 表示被分派到各个专家的输入数据。
            dispatched_indices (Tensor):表示输入数据被分派到的专家索引。
            dispatched_probs (Tensor): 表示输入数据被分派到各个专家的概率。

        Returns:
            Tensor: 经过前向传播计算后的输出数据。

        """
        use_quant_before_a2a = isinstance(hs_2d_dispatched, tuple)

        num_experts = len(self.tokens_per_expert)
        # 1 unzip
        self.dispatched_indices = dispatched_indices.to(paddle.int32)
        (unzipped_tokens, zipped_expertwise_rowmap, unzipped_probs, unzipped_scale) = (
            self.unzip_node.forward(
                hs_2d_dispatched,
                self.dispatched_indices,
                dispatched_probs,
                topk=self.router_topk,
                num_experts=num_experts,
                tokens_per_expert=self.tokens_per_expert,
                fill_output=not self.use_expert_subbatch,
            )
        )
        self.unzipped_probs = unzipped_probs
        if self.use_expert_subbatch:
            unzipped_tokens = None

        if use_quant_before_a2a:
            total_zipped_tokens = hs_2d_dispatched[0].shape[0]
            hidden_size = hs_2d_dispatched[0].shape[-1]
            hs_2d_dispatched[0]._record_stream()
            hs_2d_dispatched[1]._record_stream()
        else:
            total_zipped_tokens = hs_2d_dispatched.shape[0]
            hidden_size = hs_2d_dispatched.shape[-1]
            hs_2d_dispatched._record_stream()
        dispatched_indices._record_stream()
        dispatched_probs._record_stream()
        if self.dispatched_indices.dtype is not dispatched_indices:
            dispatched_indices._clear_to_zero_allocation()

        if self.use_expert_subbatch:
            if use_quant_before_a2a:
                hs_2d_dispatched_fp8, hs_2d_dispatched_scale = hs_2d_dispatched
            else:
                hs_2d_dispatched_fp8, hs_2d_dispatched_scale = tilewise_quant(
                    hs_2d_dispatched
                )
                hs_2d_dispatched._clear_to_zero_allocation()

            if self.recompute_unzipped:
                self.hs_2d_dispatched_fp8 = hs_2d_dispatched_fp8
                self.hs_2d_dispatched_scale = hs_2d_dispatched_scale

            output = paddle.empty([0, hidden_size], dtype=paddle.float32)
            for expert_id, tokens_per_expert in enumerate(self.tokens_per_expert):
                expert_unzipped_idx = self.subbatch_unzip_and_prepare_gemm_node(
                    (hs_2d_dispatched_fp8, hs_2d_dispatched_scale),
                    zipped_expertwise_rowmap,
                    expert_id,
                )
                gemm_node = self.experts_group_gemm_node[expert_id]
                expert_out = gemm_node.forward(
                    None,
                    unzipped_probs[
                        self.token_offsets[expert_id] : self.token_offsets[
                            expert_id + 1
                        ]
                    ],
                    [self.padding_token_per_experts[expert_id]],
                    self.tokens_per_expert[expert_id],
                )
                if self.recompute_unzipped:
                    gemm_node.input_fp8 = None
                    gemm_node.input_scale = None

                output = tokens_zip_unique_add_with_subbatch(
                    output,
                    expert_out,
                    expert_unzipped_idx,
                    zipped_rows=total_zipped_tokens,
                    subbatch_rows=self.tokens_zip_unique_add_subbatch_rows,
                )
                del expert_out
                del expert_unzipped_idx

            if use_quant_before_a2a:
                expected_output_dtype = paddle.bfloat16
            else:
                expected_output_dtype = hs_2d_dispatched.dtype

            expert_out = merge_subbatch_cast(output, expected_output_dtype)
            del output
        else:
            if not use_quant_before_a2a:
                hs_2d_dispatched._clear_to_zero_allocation()
            # 2 experts
            expert_out = self.experts_group_gemm_node.forward(
                unzipped_tokens,
                unzipped_probs,
                self.padding_token_per_experts,
                self.tokens_per_expert,
                output=unzipped_tokens,
                scale=unzipped_scale,  # maybe None
            )

            # 3 zip
            expert_out = expert_out.reshape([-1, expert_out.shape[-1]])

            expert_out = self.zip_node.forward(
                expert_out,
                zipped_expertwise_rowmap,
                self.dispatched_indices,
                unzipped_probs,
                total_zipped_tokens=total_zipped_tokens,
                num_experts=num_experts,
            )

        self.dispatched_probs = dispatched_probs
        expert_out.stop_gradient = False

        return expert_out

    @paddle.no_grad()
    def backward(self, hidden_states_out_grad):
        """
        反向传播函数。

        Args:
            hidden_states_out_grad (Tensor): 隐藏状态梯度。

        Returns:
            Tuple[Tensor, Tensor]: 包含两个元素，分别为hs_fp8_dispatched_grad和dispatched_probs_grad。
                - hs_fp8_dispatched_grad (Tensor): 解压后的隐藏状态梯度。
                - dispatched_probs_grad (Tensor): 分发概率梯度。

        """
        # zip_grad
        hidden_states_out_grad_shape = hidden_states_out_grad.shape
        unzipped_grad = self.zip_node.backward(
            hidden_states_out_grad,
            self.dispatched_indices,
            self.dispatched_probs,
            top_k=self.router_topk,
            num_experts=len(self.tokens_per_expert),
            tokens_per_expert=self.tokens_per_expert,
            fill_output=not self.use_expert_subbatch,
        )
        hidden_states_out_grad._record_stream()

        if self.use_expert_subbatch:
            output = paddle.empty(
                [0, hidden_states_out_grad_shape[-1]], dtype=paddle.float32
            )
            probs_grad_list = []

            for expert_id, tokens_per_expert in enumerate(self.tokens_per_expert):
                unzipped_grad, _, unzipped_grad_idx = TDU.tokens_unzip_gather(
                    hidden_states_out_grad,
                    None,
                    self.unzip_node.zipped_expertwise_rowmap,
                    expert_id=expert_id,
                    tokens_per_expert=self.tokens_per_expert,
                    padding_multiplex=FP8_ALIGN,
                )
                if self.recompute_unzipped:
                    self.subbatch_unzip_and_prepare_gemm_node(
                        (self.hs_2d_dispatched_fp8, self.hs_2d_dispatched_scale),
                        self.unzip_node.zipped_expertwise_rowmap,
                        expert_id,
                    )
                unzipped_grad, unzipped_probs_grad = self.experts_group_gemm_node[
                    expert_id
                ].backward(
                    unzipped_grad,
                    self.unzipped_probs[
                        self.token_offsets[expert_id] : self.token_offsets[
                            expert_id + 1
                        ]
                    ],
                )
                output = tokens_zip_unique_add_with_subbatch(
                    output,
                    unzipped_grad,
                    unzipped_grad_idx,
                    zipped_rows=hidden_states_out_grad_shape[0],
                    subbatch_rows=self.tokens_zip_unique_add_subbatch_rows,
                )
                if len(unzipped_probs_grad.shape) > 1:
                    unzipped_probs_grad = unzipped_probs_grad.squeeze(-1)
                assert len(unzipped_probs_grad.shape) == 1, unzipped_probs_grad.shape
                probs_grad_list.append(unzipped_probs_grad)
                del unzipped_grad
                del unzipped_grad_idx
                del unzipped_probs_grad

            hidden_states_out_grad._clear_to_zero_allocation()
            hs_fp8_dispatched_grad = merge_subbatch_cast(
                output, hidden_states_out_grad.dtype
            )
            del output
            dispatched_probs_grad = TDU.tokens_zip_prob(
                probs_grad_list,
                self.unzip_node.zipped_expertwise_rowmap,
                self.dispatched_indices,
            )
        else:
            hidden_states_out_grad._clear_to_zero_allocation()

            # expert_grad
            expert_out, probs_grad = self.experts_group_gemm_node.backward(
                unzipped_grad
            )
            del unzipped_grad

            hs_fp8_dispatched_grad, dispatched_probs_grad = self.unzip_node.backward(
                expert_out,
                hidden_states_out_grad_shape,
                probs_grad,
                self.dispatched_indices,
                num_experts=len(self.tokens_per_expert),
            )
        self.reset_statue()
        return hs_fp8_dispatched_grad, dispatched_probs_grad


class FP8FusedWLCHFunc(paddle.autograd.PyLayer):
    """FP8FusedWLCHFunc"""

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        dispatched_probs,
        custom_map,
        recompute_fwd_gate_up=False,
        dequant_input=False,
        fp8_dispatch_a2a=False,
        is_first_fwd=False,
        group=None,
        fp8_dispatched_handle=None,
    ):
        """
        根据给定的参数执行前向传播操作。

        Args:
            hidden_states (tensor): 输入的隐藏状态张量。
            dispatched_probs (tensor): 分派概率张量。

        Returns:
            tensor: 前向传播的结果张量。
        """

        ctx.node = ExpertsGroupGemmWLCHNode(
            custom_map,
            recompute_fwd_gate_up=recompute_fwd_gate_up,
            dequant_input=dequant_input,
            group=group,
        )
        ctx.group = group
        ctx.fp8_dispatch_a2a = fp8_dispatch_a2a
        world_size = custom_map.world_size
        num_local_experts = custom_map.num_local_experts

        def a2a_fn(input_fp8, input_scale):
            return AlltoAll.apply(input_fp8, group), AlltoAll.apply(input_scale, group)

        if fp8_dispatch_a2a:
            assert fp8_dispatched_handle is not None
            assert hidden_states.dtype == paddle.float8_e4m3fn
            hidden_states, scale = a2a_fn(hidden_states, fp8_dispatched_handle["scale"])
            scale = scale.reshape([-1, scale.shape[-1]])
        else:
            scale = None
        hidden_states = hidden_states.reshape([-1, hidden_states.shape[-1]])
        dispatched_probs = dispatched_probs.reshape([-1, dispatched_probs.shape[-1]])
        tokens_per_expert = [
            np.prod(hidden_states.shape[:-1]) // num_local_experts
        ] * num_local_experts

        out = ctx.node.forward(
            hidden_states,
            dispatched_probs,
            tokens_per_expert,
            tokens_per_expert,
            scale=scale,
        )

        if is_first_fwd:
            ctx.node.reset_statue()

        return out

    @staticmethod
    def backward(ctx, output_grad):
        """
        计算反向传播梯度。

        Args:
            output_grad (Tensor): 输出梯度张量。

        Returns:
            Tuple[Tensor, Tensor]: 返回两个梯度张量，前两个分别是隐藏状态和派发概率的梯度，

        """

        def a2a_async_fn(input):
            return AlltoAll.apply(input, ctx.group, sync_op=False)

        if ctx.fp8_dispatch_a2a:
            return ctx.node.backward(output_grad, a2a_async_fn=a2a_async_fn)
        else:
            return ctx.node.backward(output_grad, a2a_async_fn=None)


class Fp8FusedMoeFunc(paddle.autograd.PyLayer):
    """
    The Fp8FusedMoeFunc class includes operations for unzipping, expert computation, and zipping.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        dispatched_probs,
        dispatched_indices,
        custom_map,
        max_topk,
        recompute_fwd_gate_up=False,
        dequant_input=False,
        use_expert_subbatch=False,
        use_recompute_unzipped=False,
        tokens_zip_unique_add_subbatch_rows=None,
        backward_subbatch_rows=None,
        is_first_fwd=False,
        fp8_dispatched_handle=None,
    ):
        """
        根据给定的参数执行前向传播操作。

        Args:
            hidden_states (tensor): 输入的隐藏状态张量。
            dispatched_probs (tensor): 分派概率张量。
            dispatched_indices (tensor): 分派索引张量。
            max_topk (int): topk。

        Returns:
            tensor: 前向传播的结果张量。
        """
        ctx.node = MlpNode(
            custom_map,
            max_topk,
            recompute_fwd_gate_up=recompute_fwd_gate_up,
            dequant_input=dequant_input,
            use_expert_subbatch=use_expert_subbatch,
            recompute_unzipped=use_recompute_unzipped,
            tokens_zip_unique_add_subbatch_rows=tokens_zip_unique_add_subbatch_rows,
            backward_subbatch_rows=backward_subbatch_rows,
        )

        if fp8_dispatched_handle is not None:
            assert hidden_states.dtype == paddle.float8_e4m3fn
            scale = fp8_dispatched_handle["scale"]
            hidden_states = (hidden_states, scale)

        out = ctx.node.forward(hidden_states, dispatched_indices, dispatched_probs)

        if is_first_fwd:
            ctx.node.release_mem()

        cached_tensors = ctx.node.cached_tensors()
        ctx.save_for_backward(cached_tensors)
        ctx.node.clear_cached_tensors()
        return out

    @staticmethod
    def backward(ctx, output_grad):
        """
        计算反向传播梯度。

        Args:
            output_grad (Tensor): 输出梯度张量。

        Returns:
            Tuple[Tensor, Tensor, None]: 返回三个梯度张量，前两个分别是隐藏状态和派发概率的梯度，
                                            第三个为None，表示没有需要传递给更前向节点的梯度。

        """
        (cached_tensors,) = ctx.saved_tensor()
        ctx.node.set_cached_tensors(cached_tensors)
        hidden_states_grad, dispatched_probs_grad = ctx.node.backward(output_grad)
        return hidden_states_grad, dispatched_probs_grad, None


class DeepEPMOELayer(nn.Layer):
    """
    MoE 层，使用 DeepEP 进行通信
    """

    def __init__(
        self,
        gate: nn.Layer,
        experts: List[nn.Layer],
        layer_idx,
        shared_experts: Optional[List[nn.Layer]] = None,
        group: Group = None,
        recompute=False,
        enable_logging=False,
        k=2,
        enable_bpr=False,
        all_to_all_dropout=False,
        group_experts=False,
        moe_statics=None,
    ):
        """
        初始化MoE层。

        Args:
            gate (nn.Layer): 智能门控层，用于选择需要使用的专家。
            experts (List[nn.Layer]): 需要使用的专家列表。
            layer_idx (int): 当前MoE层的索引。
            group (Group): 分布式通信组。默认值为None。
            recompute (bool): 是否在每个训练迭代中重新计算MoE输出。默认值为False。
        """
        super().__init__()
        self.gate = gate
        self.config = self.gate.config
        self.zero = paddle.to_tensor(0, dtype=paddle.float32)
        self.zero.stop_gradient = True

        if isinstance(self.config.moe_num_experts, (list, tuple)):
            self.global_num_experts = sum(self.config.moe_num_experts)
        else:
            self.global_num_experts = self.config.moe_num_experts
        self.scaling_factor = self.config.scaling_factor  # router_scaling_factor
        logger.info(f"using router_scaling_factor={self.scaling_factor}")

        assert not enable_bpr, "enable bpr is not supported now."
        assert not all_to_all_dropout, "all_to_all_dropout is not supported now."

        self.enable_logging = enable_logging
        self.enable_bpr = enable_bpr
        self.all_to_all_dropout = all_to_all_dropout

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
        for p in experts.parameters():
            # p.no_sync 如果设True, 初始化权重时会使用local_seed(处处不一样); 如果设False, 初始化权重时会使用model_parallel_rng(在DP间一样)
            # 所以这里的逻辑是:
            #   如果是tp-moe, 应使用model_parallel_rng, 保证权重在DP间一样(框架保证)
            #   如果是dp-moe, 应使用local_seed, 保证权重处处不一样
            #   如果是ep-moe, 应保证权重在moe-dp组内相同(框架保证), 在ep组内不同
            # 此外, 框架中, 如果参数的no_sync为True, 在sync_params_buffers时会跳过。
            p.no_sync = not self.is_mp_moe
            # 在框架broadcast_moe_sharding_parameter或broadcast_moe_dp_parameter中,
            # 如果参数的expert=True, 会强制在moe_sharding_group/moe_dp_group同步参数
            p.expert = not self.is_mp_moe
            logger.info(f"expert no-sync={p.no_sync}-{p.name}")
            if self.is_mp_moe or self.is_ep_moe:
                p.is_distributed = True

        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        if self.world_size < 1:
            self.world_size = 1
        if self.rank < 0:
            self.rank = 0

        self.num_local_experts = len(self.experts)

        self.dispatcher = MoEFlexTokenDispatcher(
            self.num_local_experts, self.global_num_experts, self.group
        )

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

    def fp8_quant_weight(self):
        """fp8_quant_weight"""

        def quant_weight(weight):
            fp8_weight_w, fp8_scale_w = FusedQuantOps.fused_stack_quant([weight])
            setattr(weight, "fp8_weight_stacked", fp8_weight_w)
            setattr(weight, "fp8_scale_stacked", fp8_scale_w)

            fp8_weight_w_t, fp8_scale_w_t = FusedQuantOps.fused_stack_transpose_quant(
                [weight]
            )
            setattr(weight, "fp8_weight_stacked_transpose", fp8_weight_w_t)
            setattr(weight, "fp8_scale_stacked_transpose", fp8_scale_w_t)

        for expert in self.experts:
            quant_weight(expert.up_gate_proj.weight)
            quant_weight(expert.down_proj.weight)

    def cal_norm_combine_weights(self, combine_weights_unnorm):
        """
        normalize combine weights
        """
        if self.gate.norm_gate_logits:
            combine_weights = combine_weights_unnorm / paddle.clip(
                combine_weights_unnorm.sum(-1, keepdim=True), min=1e-12
            )
        else:
            combine_weights = combine_weights_unnorm
        return combine_weights

    def calc_and_log_moe_summary(self, key, summary_data, is_numpy=False):
        """算max min等并记录"""
        if not is_numpy:
            summary_data = summary_data.numpy()

        max_value = max(summary_data)
        min_value = min(summary_data)
        var_value = np.var(summary_data)
        median_value = np.median(summary_data)
        mean_value = np.mean(summary_data)
        if mean_value == 0:
            assert max_value == 0, max_value
            assert min_value == 0, min_value

        prefix = f"{key}_layer_{self.layer_idx}"

        _log = {
            f"{prefix}_max": max_value,
            f"{prefix}_min": min_value,
            f"{prefix}_var": var_value,
            f"{prefix}_median": median_value,
            f"{prefix}_mean": mean_value,
            f"{prefix}_max_mean_ratio": (
                max_value / mean_value if mean_value != 0 else 1.0
            ),
            f"{prefix}_min_mean_ratio": (
                min_value / mean_value if mean_value != 0 else 1.0
            ),
        }
        global_training_logs.update(**_log)

    def moe_tokens_per_experts_indicator(self, key, summary_data, count):
        """统计每个专家处理的token数相关指标"""
        dist.all_reduce(summary_data, group=self.group)

        hcg = fleet.get_hybrid_communicate_group()
        tp_world_size = hcg.get_model_parallel_world_size()
        # if tp_group not equal to expert_group, add count inter batch
        if tp_world_size != self.world_size:
            dist.all_reduce(count, group=self.group)
            count /= tp_world_size

        count = paddle.ones_like(count) if count.item() == 0 else count
        summary_data_avg = paddle.cast(summary_data, dtype=count.dtype) / count

        # avg-summary
        self.calc_and_log_moe_summary(key + "_avg", summary_data_avg)

        # origin
        self.calc_and_log_moe_summary(key, summary_data)

    def node_limit_routing(self, gate_probs):
        """
        将所有专家分组, 只在topk_group个group内选择专家
        """
        assert len(gate_probs.shape) == 2
        seq_length, n_experts = gate_probs.shape
        assert (
            n_experts % self.config.n_group == 0
        ), "n_experts must be divisible by n_groups"

        group_scores = (
            gate_probs.reshape([seq_length, self.config.n_group, -1])
            .topk(2, axis=-1)[0]
            .sum(axis=-1)
        )  # [n, n_group]
        group_idx = paddle.topk(
            group_scores, k=self.config.topk_group, axis=-1, sorted=True
        )[
            1
        ]  # [n, top_k_group]
        group_mask = paddle.zeros_like(group_scores).put_along_axis(
            group_idx, paddle.ones([], dtype="float32"), axis=-1
        )
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand([seq_length, self.config.n_group, n_experts // self.config.n_group])
            .reshape([seq_length, -1])
        )  # [n, e]
        gate_probs = gate_probs.masked_fill(
            ~score_mask.astype(paddle.bool), float("-inf")
        )
        return gate_probs

    def gate_score(
        self, input, global_gate_mask=None, is_diff_expert_num=False, input_ids=None
    ):
        """
        Gating function, compute the probs of each token to
        be dispatched to each expert

        Args:
            input: Tensor[S, H]
            global_gate_mask: Tensor[E]
        Returns:
            gate_logits: Tensor[S, E]
            gate_probs: Tensor[S, E]
            topk_normed_probs: Tensor[S, K]
            topk_indices: Tensor[S, K]
            router_loss: Tensor[1]
        """
        assert len(input.shape) == 2
        assert isinstance(self.gate, DeepEPTop2Gate)

        (
            gate_logits,
            router_loss,
        ) = self.gate(input, global_gate_mask=global_gate_mask)

        gate_probs = self.gate.act(gate_logits)

        if input_ids is not None and self.config.gate_force_zero_padding_grad:
            assert (
                input_ids.shape[0] == gate_logits.shape[0]
            ), f"check input_ids shape {input_ids.shape}"
            valid_mask = (input_ids != 0).astype(paddle.float32).unsqueeze(-1)
            gate_logits = gate_logits * valid_mask
            gate_probs = gate_probs * valid_mask

        if self.use_correction_bias:
            # NOTE: e_score_correction_bias只能影响topk选择的indices, 而不能影响auxloss、gate_probs等数值
            assert (
                self.moe_statics.e_score_correction_bias is not None
            ), "e_score_correction_bias is None"
            if is_diff_expert_num:
                inf_mask = paddle.isinf(global_gate_mask) & (
                    global_gate_mask < 0
                )  # shape [E], dtype: bool
                correction_bias = self.moe_statics.e_score_correction_bias[0].detach()
                fixed_correction_bias = paddle.where(
                    inf_mask, paddle.zeros_like(correction_bias), correction_bias
                )
                probs_for_choice = gate_probs + fixed_correction_bias

            else:
                probs_for_choice = (
                    gate_probs + self.moe_statics.e_score_correction_bias[0].detach()
                )
        else:
            probs_for_choice = gate_probs

        if self.config.n_group != 0 and self.config.topk_group != 0:
            probs_for_choice = self.node_limit_routing(probs_for_choice)

        return gate_logits, gate_probs, probs_for_choice, router_loss

    def topk(
        self,
        moe_k,
        gate_logits,
        gate_probs,
        probs_for_choice,
        is_diff_topk=False,
        is_diff_expert_num=False,
        input_ids=None,
        is_pure_text_line=None,
    ):
        """topk_and_norm"""
        # NOTE: e_score_correction_bias只能影响topk选择的indices, 而不能影响auxloss、gate_probs等数值
        # 所以有gate_probs和probs_for_choice的区别
        _, topk_indices = paddle.topk(probs_for_choice, moe_k, axis=-1)
        topk_probs = paddle.take_along_axis(gate_probs, topk_indices, axis=-1)
        topk_indices.stop_gradient = True

        topk_normed_probs = self.cal_norm_combine_weights(topk_probs)

        # if int_bincount is not None:
        #     dispatch_mask = int_bincount(topk_indices, 0, gate_probs.shape[-1], paddle.int64)
        # else:
        routing_map = (
            paddle.zeros_like(gate_probs)
            .put_along_axis(topk_indices, paddle.full([], fill_value=1.0), axis=1)
            .astype("bool")
        )
        if input_ids is not None:
            # has_padding = (input_ids == 0).any()
            valid_mask = input_ids != 0
            valid_mask = valid_mask.unsqueeze(-1)
            routing_map = routing_map * valid_mask.cast(routing_map.dtype)
            # -1 means neither participates in routing nor expert calculation
            topk_indices = topk_indices.masked_fill(~valid_mask, -1)

        if is_pure_text_line is not None:
            routing_map = routing_map * is_pure_text_line.cast(routing_map.dtype)
            # -1 means neither participates in routing nor expert calculation
            topk_indices = topk_indices.masked_fill(
                ~is_pure_text_line.astype(paddle.bool), -1
            )

        dispatch_mask = paddle.sum(routing_map.cast(paddle.int64), axis=0)
        if self.use_correction_bias and not (is_diff_topk or is_diff_expert_num):
            # 避免recompute的时候重复统计
            if framework._dygraph_tracer()._has_grad:
                self.moe_statics.expert_usage[0] += dispatch_mask.detach()

        if self.scaling_factor:
            # align with ds-v3
            # Ref to https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py#L471
            topk_normed_probs = topk_normed_probs * self.scaling_factor

        return topk_normed_probs, topk_indices, routing_map, dispatch_mask

    def pad_for_elastic(self, max_topk, topk_probs, topk_indices):
        """pad_for_elastic"""
        assert len(topk_probs.shape) == 2
        assert len(topk_indices.shape) == 2
        assert topk_probs.shape[-1] == topk_indices.shape[-1]

        if topk_probs.shape[-1] == max_topk:
            return topk_probs, topk_indices

        num_to_pad = max_topk - topk_probs.shape[-1]
        padded_topk_probs = F.pad(
            topk_probs, (0, 0, 0, num_to_pad), value=0, mode="constant"
        )
        padded_topk_indices = F.pad(
            topk_indices, (0, 0, 0, num_to_pad), value=-1, mode="constant"
        )
        return padded_topk_probs, padded_topk_indices

    def forward(
        self,
        input: Tensor,
        input_ids: Tensor,
        token_type_ids=None,  # not used
        elastic_topk_value=None,
        global_gate_mask=None,
        is_diff_expert_num=False,
        is_diff_topk=False,
        max_topk=None,  # 保证ep组内topk相等,弹性topk时需padding到max_topk
        origin_input_ids=None,
        is_pure_text_line=None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        Args:
            input (`Tensor`): The input data with shape ``(s, d)``.
                Only one token is supported for now.
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
        assert (
            int(self.all_to_all_dropout) == 0
        ), "all_to_all_dropout is not supported yet."
        assert token_type_ids is None, "token_type_ids is not supported yet."
        assert self.gate is not None

        # NOTE: Make sure that the data in each rank within the moe groups is different.

        # gate
        gate_logits, gate_probs, probs_for_choice, router_loss = self.gate_score(
            input,
            global_gate_mask=global_gate_mask,
            is_diff_expert_num=is_diff_expert_num,
            input_ids=input_ids,
        )
        if elastic_topk_value is None:
            self.elastic_topk_value = self.k
        else:
            self.elastic_topk_value = elastic_topk_value

        if self.config.sequence_parallel or self.config.submatrix_parallel:
            self.bsz = (
                input.shape[0]
                * self.config.tensor_parallel_degree
                // self.config.seqlen
            )
        else:
            self.bsz = input.shape[0]

        no_elastic_summary, elastic_topk_summary, elastic_expert_summary = (
            paddle.zeros(self.global_num_experts, dtype=paddle.int64),
            paddle.zeros(self.global_num_experts, dtype=paddle.int64),
            paddle.zeros(self.global_num_experts, dtype=paddle.int64),
        )
        no_elastic_num, elastic_topk_num, elastic_expert_num = (
            paddle.zeros(1, dtype=paddle.int64),
            paddle.zeros(1, dtype=paddle.int64),
            paddle.zeros(1, dtype=paddle.int64),
        )

        ori_topk_probs, ori_topk_indices, routing_map, dispatch_mask = self.topk(
            self.elastic_topk_value,
            gate_logits,
            gate_probs,
            probs_for_choice,
            input_ids=input_ids,
            is_diff_topk=is_diff_topk,
            is_diff_expert_num=is_diff_expert_num,
            is_pure_text_line=is_pure_text_line,
        )

        if max_topk is not None:
            assert (
                max_topk <= 32
            ), f"max_topk {max_topk} exceeds 32, which is not supported yet."
            topk_probs, topk_indices = self.pad_for_elastic(
                max_topk, ori_topk_probs, ori_topk_indices
            )
        else:
            topk_probs, topk_indices = ori_topk_probs, ori_topk_indices
        topk_indices.stop_gradient = True

        if not (is_diff_topk or is_diff_expert_num):
            no_elastic_summary += dispatch_mask
            no_elastic_num += self.bsz
        if is_diff_topk:
            elastic_topk_summary += dispatch_mask
            elastic_topk_num += self.bsz
        if is_diff_expert_num:
            elastic_expert_summary += dispatch_mask
            elastic_expert_num += self.bsz

        if self.config.use_ep_comm_overlap:
            dispatch_overlap_handle = {
                "fn": self.calc_router_loss_and_logging,
                "fn_args": (
                    router_loss,
                    gate_logits,
                    gate_probs,
                    routing_map,
                    dispatch_mask,
                    input_ids,
                    paddle.full(shape=[1], dtype=bool, fill_value=is_diff_topk),
                    paddle.full(shape=[1], dtype=bool, fill_value=is_diff_expert_num),
                    origin_input_ids,
                    is_pure_text_line,
                ),
            }
        else:
            router_loss2 = self.calc_router_loss_and_logging(
                router_loss,
                gate_logits=gate_logits,
                gate_probs=gate_probs.clone(),  # 为了确保ep_comm_overlap开和关时, gate_probs梯度累加顺序相同
                routing_map=routing_map,
                dispatch_mask=dispatch_mask,
                input_ids=input_ids,
                is_diff_topk=is_diff_topk,
                is_diff_expert_num=is_diff_expert_num,
                origin_input_ids=origin_input_ids,
                is_pure_text_line=is_pure_text_line,
            )
            dispatch_overlap_handle = None

        if self.shared_experts is not None and self.config.use_ep_comm_overlap:
            combine_overlap_handle = {"fn": self.shared_experts, "fn_args": (input,)}
        else:
            combine_overlap_handle = None

        # calc and log summary
        is_first_fwd = framework._dygraph_tracer()._has_grad
        if self.enable_logging and global_training_logs_enabled() and is_first_fwd:
            self.moe_tokens_per_experts_indicator(
                "no_elastic", no_elastic_summary, no_elastic_num
            )
            self.moe_tokens_per_experts_indicator(
                "elastic_topk", elastic_topk_summary, elastic_topk_num
            )
            self.moe_tokens_per_experts_indicator(
                "elastic_expert", elastic_expert_summary, elastic_expert_num
            )

        if max_topk is not None:
            dispatch_topk = max_topk
        else:
            dispatch_topk = self.elastic_topk_value

        if self.config.use_fp8_fuse_node:
            with profile("dispatch"):
                # fp8_dispatched_handle里装的是scale, 为了避免污染计算图, 用dict形式的handle进行传递
                (
                    dispatched_hidden_states,
                    dispatched_indices,
                    dispatched_probs,
                    fp8_dispatched_handle,
                ) = self.dispatcher._comm_manager.dispatch(
                    input,
                    topk_indices,
                    topk_probs,
                    fp8_dispatch_a2a=self.config.use_fp8_dispatch_a2a,
                    inner_layer_overlap_handle=dispatch_overlap_handle,
                )

            with profile("fusion_mlp"):
                hidden_states_tmp = Fp8FusedMoeFunc.apply(
                    dispatched_hidden_states,
                    dispatched_probs,
                    dispatched_indices,
                    self,
                    dispatch_topk,
                    recompute_fwd_gate_up=recompute_fwd_gate_up_func(
                        self.config, self.layer_idx
                    ),
                    dequant_input=("dequant_input" in self.config.fp8_mem_configs)
                    and self.config.fp8_mem_configs["dequant_input"],
                    use_expert_subbatch=has_config(
                        self.config.fp8_mem_configs, "use_expert_subbatch"
                    ),
                    use_recompute_unzipped=has_config(
                        self.config.fp8_mem_configs, "use_recompute_unzipped"
                    ),
                    tokens_zip_unique_add_subbatch_rows=self.config.fp8_mem_configs.get(
                        "tokens_zip_unique_add_subbatch_rows"
                    ),
                    backward_subbatch_rows=self.config.fp8_mem_configs.get(
                        "backward_subbatch_rows"
                    ),
                    is_first_fwd=not framework._dygraph_tracer()._has_grad,
                    fp8_dispatched_handle=fp8_dispatched_handle,
                )

            with profile("combine"):
                combined_output = self.dispatcher._comm_manager.combine(
                    hidden_states_tmp, inner_layer_overlap_handle=combine_overlap_handle
                )
            del hidden_states_tmp
        elif self.config.deepep_fine_grained:
            assert (
                not self.config.use_fp8_fuse_node
            ), "Deepep_fine_grained is not supported for fp8 yet."
            # global dispatch
            dispatched_output, dispatched_indices, dispatched_probs, _ = (
                self.dispatcher._comm_manager.dispatch(
                    input,
                    topk_indices,
                    topk_probs,
                    inner_layer_overlap_handle=dispatch_overlap_handle,
                )
            )

            # local dispatch & forward_experts & local combine
            output_tokens = self.fine_grained_forward_experts(
                dispatched_output, dispatched_probs, dispatched_indices, dispatch_topk
            )

            # global combine
            combined_output = self.dispatcher._comm_manager.combine(
                output_tokens, inner_layer_overlap_handle=combine_overlap_handle
            )
        else:
            # dispatch
            (
                dispatched_input,
                token_permuted_indices,
                prob_permuted_indices,
                dispatched_probs,
            ) = self.dispatcher.token_permutation(
                input,
                topk_indices,
                topk_probs,
                dispatch_topk,
                inner_layer_overlap_handle=dispatch_overlap_handle,
            )

            # ffn
            expert_out = self.forward_experts(dispatched_input)

            # combine
            combined_output = self.dispatcher.token_unpermutation(
                expert_out,
                token_permuted_indices,
                prob_permuted_indices,
                dispatched_probs,
                deepep_use_fused=self.config.deepep_use_fused,
                inner_layer_overlap_handle=combine_overlap_handle,
            )

        if self.config.use_ep_comm_overlap:
            router_loss2 = dispatch_overlap_handle["fn_out"][0]

        # shared_experts
        if self.shared_experts is not None:
            if self.config.use_ep_comm_overlap:
                shared_out = combine_overlap_handle["fn_out"][0]
            elif self.config.use_decoder_chunk_recompute:
                shared_out = recompute(self.shared_experts, input)
            else:
                shared_out = self.shared_experts(input)
            combined_output += shared_out

        if self.enable_logging and global_training_logs_enabled() and is_first_fwd:
            num_tokens_per_expert = self.get_num_tokens_per_expert()
            total_num_tokens_per_expert = paddle.full(
                [], fill_value=sum(num_tokens_per_expert)
            )
            total_list = paddle.empty(
                [self.group.world_size], dtype=total_num_tokens_per_expert.dtype
            )
            dist.stream.all_gather(
                total_list, total_num_tokens_per_expert, group=self.group
            )
            self.calc_and_log_moe_summary("local_tokens_per_card", total_list)

        if orig_shape:
            combined_output = combined_output.clone().reshape(
                orig_shape[:-1] + [combined_output.shape[-1]]
            )
        return combined_output, topk_probs, router_loss2, gate_logits

    def get_num_tokens_per_expert(self):
        """
        获取每个专家处理的 token 数量。
        """
        assert len(self.dispatcher._comm_manager.tokens_per_expert_list) == len(
            self.experts
        )
        return self.dispatcher._comm_manager.tokens_per_expert_list

    def maybe_split_subbatch_data(
        self, permuted_tokens, token_permuted_indices, prob_permuted_indices
    ):
        """maybe_split_subbatch_data"""

        def split_subbatch_data(data, tokens_per_subbatch):
            total_token_num = data.shape[0]

            full_batch_num, remainder = divmod(total_token_num, tokens_per_subbatch)
            num_or_sections = [tokens_per_subbatch] * full_batch_num
            if remainder:
                num_or_sections.append(remainder)

            assert (
                sum(num_or_sections) == total_token_num
            ), f"get_subbatch_data fail, {sum(num_or_sections)}, {total_token_num}"
            # when data is 0-size tensor, we need to compute it and construct the right backward graph.
            if total_token_num == 0:
                return [data]
            return paddle.split(data, num_or_sections=num_or_sections, axis=0)

        if self.config.deepep_tokens_per_subbatch > 0:
            assert (
                permuted_tokens.shape[0] == token_permuted_indices.shape[0]
            ), f"Shape mismatch between {permuted_tokens.shape[0]} and {token_permuted_indices.shape[0]}"
            assert (
                permuted_tokens.shape[0] == prob_permuted_indices.shape[0]
            ), f"Shape mismatch between {permuted_tokens.shape[0]} and {prob_permuted_indices.shape[0]}"
            permuted_tokens_list = split_subbatch_data(
                permuted_tokens, self.config.deepep_tokens_per_subbatch
            )
            token_permuted_indices_list = split_subbatch_data(
                token_permuted_indices, self.config.deepep_tokens_per_subbatch
            )
            prob_permuted_indices_list = split_subbatch_data(
                prob_permuted_indices, self.config.deepep_tokens_per_subbatch
            )
        else:
            permuted_tokens_list = [permuted_tokens]
            token_permuted_indices_list = [token_permuted_indices]
            prob_permuted_indices_list = [prob_permuted_indices]
        return (
            permuted_tokens_list,
            token_permuted_indices_list,
            prob_permuted_indices_list,
        )

    def fine_grained_forward_experts(
        self, dispatched_output, dispatched_probs, dispatched_indices, dispatch_topk
    ):
        """fine_grained_forward_experts"""
        output_tokens = paddle.zeros(dispatched_output.shape, dispatched_probs.dtype)

        for expert_id, num_tokens in enumerate(self.get_num_tokens_per_expert()):
            # local dispatch
            token_permuted_indices, prob_permuted_indices = (
                topk_to_permuted_indices_single(
                    dispatched_indices, num_tokens, expert_id, dispatch_topk
                )
            )
            permuted_tokens = FakeGather.apply(
                dispatched_output, token_permuted_indices
            )
            inplace_offload_if_needed(permuted_tokens)

            # If deepep_tokens_per_subbatch > 0, the data is split into multiple subbatches.
            (
                permuted_tokens_list,
                token_permuted_indices_list,
                prob_permuted_indices_list,
            ) = self.maybe_split_subbatch_data(
                permuted_tokens, token_permuted_indices, prob_permuted_indices
            )

            for (
                permuted_tokens_,
                token_permuted_indices_,
                prob_permuted_indices_,
            ) in zip(
                permuted_tokens_list,
                token_permuted_indices_list,
                prob_permuted_indices_list,
            ):
                # ffn
                permuted_tokens_ = self.experts[expert_id](permuted_tokens_)

                # local combine
                if self.config.deepep_use_fused and dispatched_probs is not None:
                    output_tokens = FusedUnpermutation.apply(
                        output_tokens,
                        permuted_tokens_,
                        token_permuted_indices_,
                        dispatched_probs.flatten(),
                        prob_permuted_indices_,
                    )
                else:
                    if dispatched_probs is not None:
                        permuted_probs = FakeGather.apply(
                            dispatched_probs.flatten(), prob_permuted_indices_
                        )
                        if permuted_tokens_.dtype != permuted_probs.dtype:
                            new_permuted_tokens = permuted_tokens_.astype(
                                permuted_probs.dtype
                            )
                        else:
                            new_permuted_tokens = permuted_tokens_
                        inplace_offload_if_needed(new_permuted_tokens)
                        permuted_tokens_ = (
                            new_permuted_tokens * permuted_probs.unsqueeze(-1)
                        )
                    if scatter_add_ is not None:
                        scatter_add_(
                            output_tokens, token_permuted_indices_, permuted_tokens_
                        )
                    else:
                        output_tokens.scatter_(
                            index=token_permuted_indices_,
                            updates=permuted_tokens_,
                            overwrite=False,
                        )

        return output_tokens.astype(dispatched_output.dtype)

    def forward_experts(self, dispatched_input):
        """
        each expert gets a chunk of input and runs forward
        """
        num_tokens_per_expert = self.get_num_tokens_per_expert()
        outputs = []
        chunks = paddle.split(
            dispatched_input, num_or_sections=num_tokens_per_expert, axis=0
        )
        assert len(chunks) == len(self.experts), (len(chunks), len(self.experts))

        for chunk, expert in zip(chunks, self.experts):
            chunk = chunk.contiguous()
            outputs += [expert(chunk)]
        return paddle.concat(outputs, axis=0)

    def calc_router_loss_and_logging(
        self,
        router_loss,
        gate_logits,
        gate_probs,
        routing_map,
        dispatch_mask,
        input_ids=None,
        is_diff_topk=False,
        is_diff_expert_num=False,
        origin_input_ids=None,
        is_pure_text_line=None,
    ):
        """calc_router_loss_and_logging"""
        assert isinstance(self.gate, DeepEPTop2Gate)
        if isinstance(is_diff_topk, paddle.Tensor):
            is_diff_topk = is_diff_topk.item()
            assert isinstance(
                is_diff_topk, bool
            ), f"check is_diff_topk {type(is_diff_topk)}"
        if isinstance(is_diff_expert_num, paddle.Tensor):
            is_diff_expert_num = is_diff_expert_num.item()
            assert isinstance(
                is_diff_expert_num, bool
            ), f"check is_diff_topk {type(is_diff_expert_num)}"
        l_aux, orthogonal_loss, zloss = None, None, None
        if self.gate.config.moe_aux_loss_lambda and not (
            is_diff_topk or is_diff_expert_num
        ):
            if self.gate.act is F.sigmoid:
                gate_probs = self.cal_norm_combine_weights(gate_probs)

            if self.config.sequence_parallel:
                hcg = fleet.get_hybrid_communicate_group()
                sequence_partition_group = hcg.get_model_parallel_group()
            else:
                sequence_partition_group = None

            if is_pure_text_line is not None:
                gate_probs *= is_pure_text_line

            if self.config.aux_loss_type == "seq_aux_loss":
                l_aux = self.gate._cal_seq_aux_loss(
                    gate_probs,
                    routing_map,
                    self.bsz,
                    self.config.seqlen,
                    self.elastic_topk_value,
                    sequence_partition_group,
                    input_ids,
                    origin_input_ids,
                )
            elif self.config.aux_loss_type == "switch_aux_loss":
                l_aux = self.gate._cal_switch_aux_loss(
                    gate_probs,
                    dispatch_mask,
                    self.elastic_topk_value,
                    sequence_partition_group,
                    input_ids,
                )
            else:
                l_aux = self.gate._cal_aux_loss(gate_probs, dispatch_mask, input_ids)
            router_loss += self.gate.moe_aux_loss_lambda[0] * l_aux
        else:
            router_loss += (
                self.zero * gate_probs[0, 0]
            )  # must use gate prob to avoid zero pointer
        if self.gate.config.moe_orthogonal_loss_lambda:
            orthogonal_loss = self.gate._cal_orthogonal_loss()
            router_loss += self.gate.moe_orthogonal_loss_lambda[0] * orthogonal_loss
        if self.gate.config.moe_z_loss_lambda and not (
            is_diff_topk or is_diff_expert_num
        ):
            if is_pure_text_line is not None:
                gate_logits *= is_pure_text_line
            zloss = self.gate._cal_z_loss(gate_logits, input_ids, origin_input_ids)
            router_loss += self.gate.moe_z_loss_lambda[0] * zloss

        # 开启重计算的话只需要logging一次就行
        tracer = framework._dygraph_tracer()
        is_first_fwd = tracer._has_grad
        if self.enable_logging and global_training_logs_enabled() and is_first_fwd:
            log = {}
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

        return router_loss


class DeepEPDropTokenMOELayer(MOELayer):
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
        """
        初始化MoE层。

        Args:
            gate (nn.Layer): 智能门控层，用于选择需要使用的专家。
            experts (List[nn.Layer]): 需要使用的专家列表。
            layer_idx (int): 当前MoE层的索引。
            group (Group): 分布式通信组。默认值为None。
            recompute (bool): 是否在每个训练迭代中重新计算MoE输出。默认值为False。
        """
        super(DeepEPDropTokenMOELayer, self).__init__(
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
            moe_statics=moe_statics,
        )
        logger.info("Using DeepEPDropTokenMOELayer")

        if isinstance(self.config.moe_num_experts, (list, tuple)):
            self.global_num_experts = sum(self.config.moe_num_experts)
        else:
            self.global_num_experts = self.config.moe_num_experts
        self.dispatcher = MoEFlexTokenDispatcher(
            self.num_local_experts, self.global_num_experts, self.group
        )

    def gate_and_drop(self, input, token_type_ids):
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
        # 目前只有 `SinkHornGate` aka Top1 gate 支持输入 token type ids
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape([-1])
            args = (token_type_ids,)

        use_fuse = isinstance(
            self.gate, (TopKGateFused)
        )
        assert use_fuse
        (
            gate_logits,
            capacity,
            router_loss,
        ) = self.gate(input, *args)

        if self.input_preprocess is not None:
            input, gate_logits = self.input_preprocess(input, gate_logits, capacity)
        # capacity no use
        k = self.k
        prob, max_prob = self.fused_gate_logits_process(gate_logits, token_type_ids)

        assert moe_ops is not None

        if "corr_bias" in inspect.signature(moe_ops.moe_gate_dispatch).parameters:
            if self.use_correction_bias:
                compat_args = (self.moe_statics.e_score_correction_bias[0],)
            else:
                compat_args = (None,)
        else:
            assert (
                not self.use_correction_bias
            ), "correction bias not supported, rebuild moe-ops"

            compat_args = ()
        with paddle.no_grad():
            temp_input = paddle.zeros([input.shape[0], 64], input.dtype)
            (
                temp_output,
                prob_drop,
                _,
                dispatch_mask,
                topk_indices,
            ) = moe_ops.moe_gate_dispatch(
                temp_input, prob, *compat_args, k=k, capacity=capacity, use_pad=True
            )
            del temp_input
            del temp_output
            drop_mask = prob_drop != 0.0

        prob_drop.stop_gradient = True
        dispatch_mask.stop_gradient = True
        topk_indices.stop_gradient = True

        combine_weights_unnorm = paddle.take_along_axis(prob, topk_indices, axis=-1)

        combine_weights_unnorm = combine_weights_unnorm * (drop_mask.cast("float32"))

        topk_indices = topk_indices.cast(paddle.int64)

        # drop token
        drop_mask_int = drop_mask
        pad_token = self.config.deepep_drop_padding
        if pad_token:
            un_pad_mask = paddle.uniform(drop_mask_int.shape, min=0.0, max=1.0) > 0.1

            fill_mask = (~drop_mask_int) * un_pad_mask
            pad_mask = (~drop_mask_int) * (~un_pad_mask)

            topk_indices = topk_indices.masked_fill(
                fill_mask, paddle.full([], fill_value=-1, dtype=paddle.int64)
            )

            topk_indices = topk_indices.masked_fill(
                pad_mask,
                paddle.full(
                    [], fill_value=dist.get_rank() * 8 % 64, dtype=paddle.int64
                ),
            )
        else:
            fill_mask = ~drop_mask_int

            topk_indices = topk_indices.masked_fill(
                fill_mask, paddle.full([], fill_value=-1, dtype=paddle.int64)
            )

        dispatch_mask = paddle.diff(F.pad(dispatch_mask, (1, 0)))
        # 避免recompute的时候重复统计
        if self.use_correction_bias and framework._dygraph_tracer()._has_grad:
            if self.gate.config.multimodel_experts:
                for i in range(len(self.moe_statics.expert_usage)):
                    self.moe_statics.expert_usage[i] += dispatch_mask[
                        self.gate.experts_type_mask[i]
                    ].detach()
            else:
                self.moe_statics.expert_usage[0] += dispatch_mask.detach()

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

        return (
            combine_weights,
            dispatch_mask,
            router_loss,
            gate_logits,
            topk_indices,
            topk_indices,
            prob,
        )

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
        # assert len(input) == 1, "only single input Tensor supported"
        if input.ndim == 3:
            orig_shape = input.shape
            # clone 保平安
            input = input.reshape([-1, input.shape[-1]])
        else:
            orig_shape = None
        assert (
            len(input.shape) == 2
        ), f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"
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

        (
            topk_probs,
            dispatch_mask,
            router_loss,
            gate_logits,
            topk_indices,
            topk_indices,
            gate_prob,
        ) = self.gate_and_drop(gate_input, token_type_ids)

        with profile("dispatch"):
            dispatched_hidden_states, dispatched_indices, dispatched_probs, _ = (
                self.dispatcher._comm_manager.dispatch(
                    input,
                    topk_indices,
                    topk_probs,
                    inner_layer_overlap_handle=None,
                )
            )

        with profile("fusion_mlp"):
            hidden_states_tmp = Fp8FusedMoeFunc.apply(
                dispatched_hidden_states,
                dispatched_probs,
                dispatched_indices,
                self,
                self.k,
                recompute_fwd_gate_up=(
                    "recompute_fwd_gate_up" in self.config.fp8_mem_configs
                )
                and self.config.fp8_mem_configs["recompute_fwd_gate_up"],
                dequant_input=("dequant_input" in self.config.fp8_mem_configs)
                and self.config.fp8_mem_configs["dequant_input"],
                is_first_fwd=not framework._dygraph_tracer()._has_grad,
            )

        with profile("combine"):
            combined_output = self.dispatcher._comm_manager.combine(
                hidden_states_tmp, inner_layer_overlap_handle=None
            )

        router_loss2 = self.calc_router_loss_and_logging(
            router_loss,
            topk_probs,
            dispatch_mask,
            gate_logits,
            gate_prob,
            token_type_ids,
        )
        if orig_shape:
            combined_output = combined_output.clone().reshape(
                orig_shape[:-1] + [combined_output.shape[-1]]
            )

        return combined_output, topk_probs, router_loss2, gate_logits


class MOEInferLayer(nn.Layer):
    """MOEInferLayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        moe = MOEInferLayer(gate, expert)
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
    """

    def __init__(
        self,
        gate: nn.Layer,
        experts: List[nn.Layer],
        group: Group = None,
        recompute=False,
    ) -> None:
        """
        初始化 MoE 模型

        Args:
            gate (nn.Layer): 分流门的网络层
            experts (List[nn.Layer]): 专家网络层列表，可以是 `nn.Layer` 或 `nn.LayerList` 的实例
            group (Group, optional): 分布式通信组，默认使用 `None`。如果传入该参数，则会将模型放入指定的分布式通信组中。
            recompute (bool, optional): 是否启用在推理阶段进行重计算（即每次前向传播时都更新所有专家模型的参数），默认为 `False`，表示禁用重计算。

        Attributes:
            gate (nn.Layer): 分流门的网络层
            recompute (bool): 是否启用在推理阶段进行重计算
            experts (nn.LayerList): 专家网络层列表
            group (Group): 分布式通信组
            world_size (int): 当前分布式任务中的总节点数
            rank (int): 当前节点在当前分布式任务中的编号
            num_local_experts (int): 当前节点所属的本地专家数目

        """
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
