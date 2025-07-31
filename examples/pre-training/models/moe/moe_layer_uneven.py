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

"""
moe
"""

from typing import Any, Tuple, List
import logging
import inspect
from collections import namedtuple

import numpy as np
import paddle
from paddle import nn
from paddle.distributed.communication import stream

from paddle.autograd import PyLayer
from paddle.distributed.communication.group import Group
from paddle.distributed.fleet.utils import recompute
import paddle.distributed as dist
from paddle import Tensor
from paddle.nn import functional as F
from paddleformers.trainer.plugins.timer import get_timers
from paddle.distributed import fleet


# from models.moe.moe_layer import _AllToAll
from models.moe.top2_gate import TopKGateFused
from models.moe.sinkhorn_gate import SinkHornGateFused
from models.moe.round_robin_gate import RoundRobinGateFused

try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}  # 没有erniebot的环境下无法打印 debug 量

logger = logging.getLogger(__name__)

GateOutput = namedtuple(
    "GateOutput",
    [
        "aux",
        "z",
        "logits",
    ],
)

from paddleformers.utils.tools import get_env_device

if get_env_device() == "xpu":
    try:
        from paddle_xpu_nn import moe_combine as xpu_moe_combine
        from paddle_xpu_nn import moe_combine_bwd as xpu_moe_combine_bwd
    except ImportError:
        xpu_moe_combine = None
        xpu_moe_combine_bwd = None
        logger.warning("`xpu moe combine` not found")
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
        import moe_combine
    except ImportError:
        moe_combine = None
        logger.warning(
            "`moe-combine` not found, run "
            "`python3  src/ernie_core/ops/moe/setup.py  install` to install"
        )


try:
    import moe_ops_no_softmaxtopk
except ImportError:
    moe_ops_no_softmaxtopk = None
    logger.warning(
        "moe-ops-no-softmaxtopk` not found, run "
        "`python3  src/ernie_core/ops/moe/setup.py  install` to install"
    )


def average_grad(x, y, dy, eps=1e-12):
    """
    TODO: fuse 这坨 shit
    y=x/x.sum(-1, keepdim=True) 的反向过程
    """
    s, k = x.shape
    xsum = x.sum(axis=-1, keepdim=True)  # [s,1]
    maskpos = (xsum == 0.0).expand_as(x)

    xsum_square = xsum.square()  # [s,1]
    left = paddle.triu(
        paddle.tril((1 / xsum).unsqueeze(-1).expand([s, k, k]))
    )  # aka diag-emb [s,k,k]
    right = (-x / xsum_square).unsqueeze(-1).expand([s, k, k])
    dydx = left + right
    dx = paddle.matmul(dy.unsqueeze(-2).cast(dydx.dtype), dydx).squeeze(
        -2
    )  # [s,1,k] @[s,k,k] -> [s,1,k]
    dx = paddle.where(maskpos, paddle.zeros_like(dx), dx)
    return dx


mask = paddle.to_tensor(
    [
        [1, -1],
        [-1, 1],
    ]
).unsqueeze(0)


def average_grad_bi(x, y, dy, eps=1e-12):
    """
    y=x/x.sum(-1, keepdim=True)
    k=2 下面的反向过程，精度会更准一些:
        dx1 = (y2 *dy1 - y2*dy2)/(y1+y2)**2
        dx2 = (y1 *dy2 - y1*dy1)/(y1+y2)**2
    """
    s, k = x.shape
    assert k == 2, k
    xsum = paddle.clip(x.sum(axis=-1, keepdim=True), min=eps)  # [s,1]
    dydx = (
        x.flip(axis=1).unsqueeze(-2).tile([1, 2, 1])
        * mask.cast(x.dtype)
        / xsum.square().unsqueeze(-1)
    )
    dx = paddle.matmul(dy.unsqueeze(-2).cast(dydx.dtype), dydx).squeeze(
        -2
    )  # [s,1,k] @[s,k,k] -> [s,1,k]
    return dx


def topk_grad(x, dy, indicies):
    """
    TODO: fuse 这坨 shit
    y=gather(topk(x)) 的反向过程
    x:  [s,e]
    dy: [s,k]
    """
    s, e = x.shape
    _, k = dy.shape
    dx = paddle.scatter_nd(
        paddle.stack(
            [
                paddle.arange(s).repeat_interleave(k).cast(indicies.dtype),
                indicies.reshape([-1]),
            ],
            -1,
        ),
        dy.reshape([-1]),
        shape=[s, e],
    )  # [s,k] -> [s,e]
    return dx  # dx 保持高精度


class GateDispatch(PyLayer):
    """doc"""

    @staticmethod
    def forward(ctx, x, gate_prob, k, capacity, use_pad, eps=1e-12):
        """
        对`gate_prob` 进行 softmax 并根据结果选取 topk 路由expert。 最后根据 expert 号对 `x` 进行重排。
        Args:
            x: [s, d] 输入的 activateion
            gate_prob: [s, e]
        k: int
            capacity: int #no use
        Returns:
            y: [s*k, d] 将所有 `x` 根据其路由的 `expert-id` 升序的排序，融合到 s 维度。
                    当截断发生时 s 会比输入 s 小。
            combine_weights: [s, k], float： 每个 token 第 k 选择的 expert 的权重。
                    当截断发生时 s 会比输入 s 小。
            scatter_index: [k, s] ： 每个 token 第 k 次选择对应到 `y` 中的位置。
            expert_offset: [e]： `y`中每个 expert-id 的分割位置。
            expert_id: [s] `x` 中激活的 expert 号
        """
        ctx.k = k
        ctx.eps = eps
        ctx.capacity = capacity
        ctx.gate_prob = gate_prob
        if "corr_bias" in inspect.signature(moe_ops.moe_gate_dispatch).parameters:
            compat_args = (None,)
        else:
            compat_args = ()
        y, combine_weights, scatter_index, expert_offset, expert_id = (
            moe_ops.moe_gate_dispatch(
                x, gate_prob, *compat_args, k=k, capacity=capacity, use_pad=use_pad
            )
        )
        ctx.combine_weights = combine_weights
        scatter_index = scatter_index.transpose([1, 0])  # [k,s] ->[s,k]
        ctx.scatter_index = scatter_index
        ctx.expert_id = expert_id
        num_experts = gate_prob.shape[-1]

        ctx.num_experts = num_experts
        ctx.seqlen = gate_prob.shape[0]

        return y, combine_weights, scatter_index, expert_offset, expert_id

    @staticmethod
    def backward(ctx, dy, dw, *_):
        """
        TODO: 这坨代码可以 fuse 一手。
        关于 softmax 对 logits 的导数，参考：
        https://stats.stackexchange.com/questions/215521/
        how-to-find-derivative-of-softmax-function-for-the-purpose-of-gradient-descent/328095#328095
        """
        s, k = ctx.combine_weights.shape
        grad = F.embedding(ctx.scatter_index, dy)  # [s, k,d]
        mask = (ctx.combine_weights > 0.0).astype(grad.dtype)  # [s,k]
        dx = paddle.matmul(mask.unsqueeze(1), grad).squeeze(
            1
        )  # [s,1,k] @ [s,k,d] -> [s,1,d]
        if ctx.gate_prob.stop_gradient:
            return dx, None

        combine_weights_unnorm = ctx.combine_weights
        dw = dw.astype(combine_weights_unnorm.dtype)
        d_prob = topk_grad(ctx.gate_prob, dw, ctx.expert_id)
        return dx, d_prob


class HardGateDispatch(PyLayer):
    """专门为 MM 场景定制的 hard gate 分发"""

    @staticmethod
    def forward(ctx, x, expert_id, k, capacity, num_experts, eps=1e-12):
        """
        对`gate_logits` 进行 softmax 并根据结果选取 topk 路由expert。 最后根据 expert 号对 `x` 进行重排。
        Args:
            x: [s, d] 输入的 activateion
            gate_logits: [s, e]
        k: int
            capacity: int #no use
        Returns:
            y: [s*k, d] 将所有 `x` 根据其路由的 `expert-id` 升序的排序，融合到 s 维度。
                    当截断发生时 s 会比输入 s 小。
            combine_weights: [s, k], float： 每个 token 第 k 选择的 expert 的权重。
                    当截断发生时 s 会比输入 s 小。
            scatter_index: [k, s] ： 每个 token 第 k 次选择对应到 `y` 中的位置。
            expert_offset: [e]： `y`中每个 expert-id 的分割位置。
            expert_id: [s] `x` 中激活的 expert 号
        """
        ctx.k = k
        assert k == 1, f"k must be 1 in hard gate, got k={k}"
        ctx.eps = eps
        ctx.capacity = capacity
        y, scatter_index, expert_offset = (
            moe_ops_no_softmaxtopk.moe_gate_dispatch_no_softmax_topk(
                x, expert_id, k, capacity, num_experts, False
            )
        )
        scatter_index = scatter_index.transpose([1, 0])  # [k,s] ->[s,k]
        ctx.scatter_index = scatter_index
        ctx.expert_id = expert_id
        ctx.num_experts = num_experts

        return y, scatter_index, expert_offset

    @staticmethod
    def backward(ctx, dy, dw, *_):
        """
        TODO: 这坨代码可以 fuse 一手。
        关于 softmax 对 logits 的导数，参考：
        https://stats.stackexchange.com/questions/215521/
        how-to-find-derivative-of-softmax-function-for-the-purpose-of-gradient-descent/328095#328095
        """
        grad = F.embedding(ctx.scatter_index, dy)  # [s, k,d]
        assert grad.shape[1] == 1, f"k must == 1, got {grad.shape}"
        dx = grad  # [s,1,k] @ [s,k,d] -> [s,1,d]
        return dx, None


class GateCombine(PyLayer):
    """GateCombine"""

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


def rr_combining_wrapper(x, combine_weights, scatter_index, rr_func, hard_gate=False):
    """
    Args:
        x: Tensor[seq, dim]
        combine_weights: [s, k]
        scatter_index:  ** [k, s] **
        rr_func: refined recompute function for moe_combine
    Returns:
        y: Tensor[s, dim]
    """
    if hard_gate:
        x_gatherd = F.embedding(scatter_index, x)  # [s,k,dim]
        return x_gatherd.squeeze(-2)
    ret = rr_func(x, combine_weights, scatter_index)
    ret.stop_gradient = False
    return ret


def combining(x, combine_weights, scatter_index, hard_gate=False):
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
    ret = GateCombine.apply(x, combine_weights, scatter_index)
    ret.stop_gradient = False
    return ret


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAllSized(PyLayer):
    """doc"""

    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        input_indices: Tensor,
        group: Group,
    ) -> Tensor:  # type: ignore
        """
        实现 All-to-All 操作，将 input 和 output 根据给定的 indices 进行分组并交换数据。

        Args:
            ctx (Any): 运行上下文对象。
            input (Tensor): 输入的 Tensor 数据。
            input_indices (Tensor): 每台设备需要聚合的原始 indices。
            group (Group): 计算拓扑对应的 Group。

        Returns:
            Tuple[Tensor, Tensor]: 返回一个元组，第一个元素为输出的 Tensor 数据，第二个元素为输出的indices。

        """
        ctx.group = group
        # return input
        assert input.ndim == 2, input.shape

        if get_timers() is not None:
            get_timers()("moe-all2all").start()
        input_indices = input_indices.astype("int64")

        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        expert_per_device = len(input_indices) // world_size
        input_indices = input_indices.reshape([-1, expert_per_device])

        # >>>> 第一次 all2all <<<<
        output_indices = paddle.zeros_like(input_indices)
        stream.alltoall_single(
            output_indices, input_indices, None, None, group, True, True
        )  # .wait()

        input_indices.stop_gradient = True
        output_indices.stop_gradient = True
        ctx.input_indices = input_indices
        ctx.output_indices = output_indices

        # has_empty_input = input_indices.sum(-1)
        # # logger.info(f'has_empty_input: {has_empty_input}')
        # stream.all_reduce(has_empty_input, op=dist.ReduceOp.MIN, group=group, sync_op=True, use_calc_stream=True)
        # # logger.info(f'has_empty_input: {has_empty_input}')
        # has_empty_input = (has_empty_input.numpy() == 0).any()

        # gathered_input_indices =
        # paddle.empty([input_indices.shape[0] * world_size, input_indices.shape[-1]], dtype=input_indices.dtype)
        # stream.all_gather(gathered_input_indices, input_indices, group=group, use_calc_stream=True)
        # gathered_input_indices =
        # gathered_input_indices.reshape([world_size, world_size, -1]) #[world, world, #local_experts]
        # output_indices = gathered_input_indices[rank]
        # >>>> 第一次 all2all <<<<

        # # 处理空 send 情况
        # input_sum_per_rank = input_indices.sum(-1).numpy()
        # output_indices_orig = output_indices
        # # has_empty_input = (gathered_input_indices.sum(-1).numpy() == 0).any()
        # # logger.info(f'has_empty_input: {has_empty_input}')
        # # if (input_sum_per_rank == 0).any():
        # if has_empty_input:
        #     # input_indices = input_indices.numpy()
        #     # logger.warning(f"This rank send empty data, before pad: {input_indices} #input={len(input)}")
        #     padded = []
        #     cnt = 0
        #     for rank, span_len in enumerate(input_sum_per_rank):
        #         if span_len != 0:
        #             span = input[cnt : cnt + span_len]
        #         else:
        #             input_indices[rank, 0] = 1
        #             span = paddle.zeros([1, input.shape[-1]], dtype=input.dtype)
        #         cnt += span_len
        #         padded.append(span)
        #     # input_indices = paddle.to_tensor(input_indices).clone()
        #     # fill_mask = input_indices[:,0] == 0
        #     input = paddle.concat(padded, 0).clone()
        #     # logger.warning(f"This rank send empty data, after pad: {input_indices} #input={len(input)}")
        #     # del padded
        #     input = input.clone()
        #     assert input_indices.sum().item() == len(input), (input_indices, input.shape)

        # if has_empty_input:
        #     # >>>> 额外一次 all2all <<<<
        #     output_indices = paddle.zeros_like(input_indices)
        #     stream.alltoall_single(output_indices, input_indices, None, None, group, True, True)  # .wait()
        #     # logger.info(f'extra a2a:  input:{input_indices} -> output:{output_indices}')
        #     # >>>> 额外一次 all2all <<<<

        # >>>> 第二次 all2all <<<<
        output = paddle.empty(
            [output_indices.sum().item()] + input.shape[1:], dtype=input.dtype
        )

        # logger.info(
        #     f"before a2a:rank-{rank}: in:{input_indices} out:{output_indices}"
        #     f" in-shape:{input.shape} out-shape:{output.shape}"
        #     f" dtype:{input.dtype} {output.dtype}"
        # )

        stream.alltoall_single(
            output,
            input,
            output_indices.sum(-1).tolist(),
            input_indices.sum(-1).tolist(),
            group,
            True,
            True,
        )  # .wait()
        # logger.info(f'after a2a:rank-{rank}: out-shape:{output.shape}')
        # >>>> 第二次 all2all <<<<

        # 处理空 recv 情况
        # output_sum_per_rank_orig = output_indices_orig.sum(-1).numpy()
        # output_sum_per_rank = output_indices.sum(-1).numpy()

        # if has_empty_input:
        #     # logger.warning(f"This rank recv empty data, before unpad: {output_indices_orig}")
        #     cnt, buf = 0, []
        #     for i, (span_len_orig, span_len) in enumerate(zip(output_indices_orig, output_indices)):
        #         if span_len_orig == 0:
        #             if span_len == 0:  # nonzero local experts
        #                 pass
        #             assert span_len == 1, span_len
        #             cnt += span_len
        #         else:
        #             assert span_len == span_len_orig
        #             buf.append(output[cnt : cnt + span_len])
        #             cnt += span_len
        #     if not buf:
        #         output = paddle.zeros([0, output.shape[-1]], dtype=output.dtype)
        #     else:
        #         output = paddle.concat(buf, 0)
        #     del buf
        #     # logger.warning(f"This rank recv empty data, after unpad: {output.shape}")

        # dist.alltoall(output, input, group=group)
        if get_timers() is not None:
            get_timers()("moe-all2all").stop()
        return output, output_indices.reshape([-1]).cast("float32")

    @staticmethod
    def backward(ctx: Any, dy: Tensor, _) -> Tuple[Tensor]:
        """
        将输入的Tensor分发到多个设备上，并将分发后的结果收集起来。

        Args:
            ctx (Any): 上下文信息，包含了分发信息和分发结果的缓存信息。
            grad_output (Tensor): 分布在多个设备上的梯度输出。
            _: 不使用。

        Returns:
            Tuple[Tensor]: 分发后的梯度输出，形状为 [ctx.input_indices.sum().item()] + grad_output.shape[1:]。
            在 ctx.output_indices 中指定的位置上，返回对应设备上的梯度输出。

        """
        # return grad_output
        # logger.info('begin a2a bwd')
        # output = paddle.empty([ctx.input_indices.sum().item()] + grad_output.shape[1:], dtype=grad_output.dtype)
        # stream.alltoall_single(
        #     output,
        #     grad_output,
        #     ctx.input_indices.sum(-1).tolist(),
        #     ctx.output_indices.sum(-1).tolist(),
        #     ctx.group,
        #     True,
        #     True,
        # )  # .wait()

        dx, _ = _AllToAllSized.apply(
            dy,
            ctx.output_indices.reshape([-1]),
            ctx.group,
        )
        # logger.info('done a2a bwd')

        return dx, None


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
    """

    def __init__(
        self,
        gate: nn.Layer,
        experts: List[nn.Layer],
        layer_idx,
        group: Group = None,
        recompute=False,
        enable_logging=False,
    ) -> None:
        """
        初始化方法，用于创建一个多层专家的模块。

        Args:
        - `gate`: `torch.nn.Module`类型的门控网络。
        - `experts`: 包含多个专家网络的列表或 `torch.nn.ModuleList`。
        -  layer_idx (int): 当前MoE层的索引。
        - `group (Optional)`: 分组对象。默认值为 `None`，表示不使用分组。
        - `recompute (Optional)`: 是否进行模型重计算。默认值为 `False`，表示不进行重计算。

        Returns:
        - `None`: 该方法没有返回值。
        """
        super().__init__()
        self.gate = gate
        if gate is not None:
            assert isinstance(
                gate, (TopKGateFused, SinkHornGateFused, RoundRobinGateFused)
            ), type(gate)
            for p in self.gate.parameters():
                p.is_gate = True
        self.recompute = recompute
        self.layer_idx = layer_idx
        self.enable_logging = enable_logging

        logger.info(f"using moe recompute={recompute}")

        if type(experts) == nn.LayerList:
            self.experts = experts
        else:
            self.experts = nn.LayerList([experts])
        self.group = group
        is_mp_moe = (
            hasattr(fleet.fleet, "_hcg")
            and group is fleet.get_hybrid_communicate_group().get_model_parallel_group()
        )
        is_dummy_moe = dist.get_world_size(group) == 1

        for p in experts.parameters():
            p.expert = not (is_mp_moe or is_dummy_moe)  # type: ignore
            p.no_sync = not (is_mp_moe or is_dummy_moe)
            if is_mp_moe or is_mp_moe:
                p.is_distributed = True
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.num_local_experts = len(self.experts)
        self.use_hard_gate = gate is None

    def forward(
        self,
        input: Tensor,
        token_type_ids=None,
    ) -> Tensor:
        """
        Args:
            input: Tensor, shape [batch, s, d]

        Returns:
            combined_output: Tensor, shape [batch, s, d]
            combine_weights: Tensor, shape [batch, 2*n, n+1]
            l_aux: float
            l_zloss: float
        """
        if (
            self.use_hard_gate
            and (not self.training)
            and paddle.all(token_type_ids == 0).numpy()
        ):
            # 在token type hard gate 情况下, 且所有 token_type 为0，全部走文本expert
            # 去除掉所有dispatch的开销
            return (
                self.experts[0](input),
                None,
                paddle.zeros([1], dtype="float32"),
                None,
            )

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
        # assert reshaped_input.shape[0] % len(self.experts) == 0, \
        #       f'num tokens must be order of number of local experts, {input[0].shape[0]} vs {len(self.experts)}'
        def fwdfn(dispatched_input, expert_count_out):
            """
            运行专家模型并合并结果。

            Args:
                dispatched_input (paddle.Tensor): [S, dim] 被分发到专家的输入数据
                expert_count_out (list): 每个阶段需要运行专家数量

            Returns:
                paddle.Tensor: [S, dim] 由所有专家模型产生的输出
            """
            # dispatched_input: [S, dim]
            expert_outputs = []  # 如果由empty初始化，则训练会出inf
            expert_count_out = expert_count_out.astype("int64")
            assert (len(expert_count_out) // self.world_size) == len(self.experts), (
                expert_count_out,
                len(self.experts),
            )
            # [worldsize, local_experts]
            expert_offset_r = expert_count_out.cumsum(0).numpy()
            # logger.info(f'expert_offset_r:{expert_offset_r}')
            expert_offset_l = np.pad(expert_offset_r, (1, 0), constant_values=0)[:-1]
            expert_offset = np.stack([expert_offset_l, expert_offset_r], axis=-1)
            # logger.info(f'expert_offset before transpose: {expert_offset}')

            expert_offset = np.transpose(
                expert_offset.reshape([self.world_size, len(self.experts), 2]),
                [1, 0, 2],
            )
            # logger.info(f'expert_offset: {expert_offset}')

            for iexpert, (expert, off_list) in enumerate(
                zip(self.experts, expert_offset)
            ):
                # logger.info(f'off_list:{off_list}')
                ins = []
                ins_offset, cnt = [], 0
                for l, r in off_list:
                    ins.append(dispatched_input[l:r])
                    ins_offset.append((cnt, cnt + (r - l)))
                    cnt += r - l
                if not sum(map(len, ins)):
                    # logger.warning(f"local-expert:{iexpert} does not process data, we do not call expert")
                    # ins = paddle.zeros([1, dispatched_input.shape[-1]], dtype=dispatched_input.dtype)
                    expert_outputs.append([None for _ in ins_offset])
                    continue

                ins = paddle.concat(ins, 0)
                out = expert(ins)
                out = [out[l:r] for l, r in ins_offset]
                # logger.info(f'expert:{iexpert} input-shape: {ins.shape} input-dtype:{ins.dtype}')
                expert_outputs.append(out)
                # debug_buffer[_mask] = paddle.zeros_like(out)
            # logger.info(f'output-shape before tranpose: {[ [oo.shape for oo in o]for o in expert_outputs]}')
            expert_outputs = list(zip(*expert_outputs))  # transpose
            # logger.info(f'output-shape after transpose: {[ [oo.shape for oo in o]for o in expert_outputs]}')
            expert_outputs = [
                o for outputs in expert_outputs for o in outputs if o is not None
            ]

            if not expert_outputs:
                expert_outputs = paddle.zeros_like(dispatched_input)
                expert_outputs.stop_gradient = False
            else:
                expert_outputs = paddle.concat(expert_outputs, 0)

            # logger.info(f'after fwd:{expert_outputs.astype("float32").mean(axis=-1)}')
            # assert (
            #     expert_outputs.shape == dispatched_input.shape
            # ), f"output-shape: {expert_outputs.shape} vs {dispatched_input.shape}"
            return expert_outputs

        if get_timers() is not None:
            get_timers()("moe-gate").start()
        gate_args = ()
        if self.use_hard_gate:
            assert token_type_ids is not None

        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape([-1])
            gate_args = (token_type_ids,)

        if self.use_hard_gate:
            logits, capacity, routerloss = (
                None,
                None,
                paddle.zeros([1], dtype="float32"),
            )
        else:
            logits, capacity, routerloss = self.gate(input, *gate_args)

        # capacity no use
        if self.use_hard_gate:
            dispatched_input, scatter_index, expert_offset = HardGateDispatch.apply(
                input,
                token_type_ids.astype("int32").unsqueeze(-1),
                k=1,
                capacity=input.shape[0],
                num_experts=len(self.experts),
            )
            combine_weights = None
            routerloss = paddle.zeros([1], dtype="float32")
        else:
            if get_timers() is not None:
                get_timers()("moe-gate").stop()
            k = 1 if isinstance(self.gate, SinkHornGateFused) else 2
            gates = F.softmax(logits, axis=-1)
            (
                dispatched_input,
                combine_weights,
                scatter_index,
                expert_offset,
                expert_id,
            ) = GateDispatch.apply(
                input,
                gates,
                k=k,
                capacity=capacity,
                use_pad=False,
            )

            if get_timers() is not None:
                get_timers()("moe-gate").start()
            gates = F.softmax(logits, axis=-1)
            expert_mask = expert_id[:, 0]
            valid_pos = expert_mask < self.gate.num_experts
            expert_mask[~valid_pos] = 0
            mask1 = F.one_hot(
                expert_mask, num_classes=self.gate.num_experts
            ) * valid_pos.astype("float32").unsqueeze(-1)
            l_aux = self.gate._cal_aux_loss(gates, mask1)
            routerloss = routerloss + self.gate.config.moe_aux_loss_lambda * l_aux
            if self.enable_logging:
                _log = {
                    f"aux_loss_layer_{self.layer_idx}": l_aux.item(),
                }
                global_training_logs.update(
                    **_log,
                    **{
                        k.replace(f"_layer_{self.layer_idx}", ""): v
                        for k, v in _log.items()
                    },
                )
            if get_timers() is not None:
                get_timers()("moe-gate").stop()

        expert_offset_pad = F.pad(expert_offset, [1, 0], value=0)
        expert_count = (
            paddle.diff(expert_offset_pad).astype(expert_offset.dtype).detach()
        )

        # expert_count = expert_offset_pad - expert_offset_pad.roll(1, axis=0)
        # expert_count = expert_count[1:]
        # logger.info(f'before a2a: {dispatched_input.shape} \n  expert_count {expert_count}')
        if self.world_size > 1:
            dispatched_input, expert_count_out = _AllToAllSized.apply(
                dispatched_input, expert_count, self.group
            )  # [ecm]
        else:
            expert_count_out = expert_count

        # logger.info(f'a2a: expert_count_out:{expert_count_out}')
        expert_output = (
            recompute(fwdfn, dispatched_input, expert_count_out)
            if self.recompute and self.training
            else fwdfn(dispatched_input, expert_count_out)
        )

        if self.world_size > 1:
            expert_output, expert_count_rec = _AllToAllSized.apply(
                expert_output, expert_count_out, self.group
            )  # [ecm]
        else:
            expert_count_rec = expert_count_out
        # logger.info(f'after a2a: '
        # f'{expert_output.astype("float32").mean(axis=-1)} \nexpert_count_rec {expert_count_rec}')
        combined_output = combining(
            expert_output, combine_weights, scatter_index, hard_gate=self.use_hard_gate
        )
        # logger.info("[A2A]: " + "; ".join([f"id:{i}={j}" for i, j in list(enumerate(expert_count.tolist()))]))
        if orig_shape:
            combined_output = combined_output.reshape(
                orig_shape[:-1] + [combined_output.shape[-1]]
            )
        return combined_output, combine_weights, routerloss, logits
