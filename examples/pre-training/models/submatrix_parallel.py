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

""" SubMatrixParallel implemention """

import types
import numpy as np
import paddle
from paddle.autograd import PyLayer

import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.base import topology as tp
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker


__all__ = [
    "smp_row_col_col_linear",
    "smp_col_row_row_linear",
    "smp_row_parallel_linear",
    "smp_column_parallel_linear",
    "SMPRowParallelLinear",
    "SMPColumnParallelLinear",
    "smp_split",
    "smp_all_gather",
]


def smp_split(tensor, axis=1):
    """split"""
    hcg = fleet.get_hybrid_communicate_group()
    group = hcg.get_model_parallel_group()
    return _Split.apply(tensor, axis, group)


def smp_all_gather(tensor, axis=1):
    """
    Gathers tensors from the whole group in a list.

    Arguments:
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple([Tensor]): Output of the collective.

    """
    hcg = fleet.get_hybrid_communicate_group()
    group = hcg.get_model_parallel_group()
    tensor_list = AllGather.apply(tensor, group)
    output = paddle.concat(tensor_list, axis=axis)
    return output


def all_gather(input, group, sync_op=True):
    """all gather"""
    parallelism = group.nranks
    output_shape = input.shape
    output_shape[0] = output_shape[0] * parallelism
    output = paddle.empty(shape=output_shape, dtype=input.dtype)
    task = group.process_group.all_gather(output, input, sync_op)
    return output, task


def reduce_scatter(input, group, sync_op=True):
    """reduce scatter"""
    parallelism = group.nranks
    output_shape = input.shape
    assert (
        input.shape[0] % parallelism == 0
    ), "Input sequence length {} can't be divided exactly by smp parallelism {}".format(
        input.shape[0], parallelism
    )
    output_shape[0] = output_shape[0] // parallelism
    output = paddle.empty(shape=output_shape, dtype=input.dtype)
    task = dist.stream.reduce_scatter(
        output, input, op=dist.ReduceOp.SUM, group=group, sync_op=sync_op
    )
    return output, task


class _Split(PyLayer):
    @staticmethod
    def forward(ctx, tensor, axis, group):
        """ """
        ctx.group = group
        ctx.axis = axis

        rank = dist.get_rank()
        src_rank_in_group = group.get_group_rank(rank)
        out = paddle.split(tensor, group.nranks, axis=axis)[src_rank_in_group]

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """ """
        tensor_list = []
        dist.all_gather(tensor_list, grad_output, group=ctx.group)
        grad = paddle.concat(tensor_list, axis=ctx.axis)
        return grad


class ReduceScatter(PyLayer):
    """ """

    @staticmethod
    def forward(ctx, tensor, group, *input_tensor_list):
        """ """
        ctx.group = group
        dist.reduce_scatter(tensor, list(input_tensor_list), group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """ """
        return (None) + AllGather.apply(grad_output, ctx.group)


class AllGather(PyLayer):
    """ """

    @staticmethod
    def forward(ctx, tensor, group):
        """ """
        ctx.group = group

        out_tensor_list = []
        dist.all_gather(out_tensor_list, tensor, group=group)
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """ """
        rank = dist.get_rank()
        src_rank_in_group = ctx.group.get_group_rank(rank)
        gx = paddle.empty_like(grad_outputs[src_rank_in_group])
        ReduceScatter.apply(gx, ctx.group, *grad_outputs)
        return gx


def init_model_parallel_ring_group():
    """init smp communication group based on model parallel"""
    _mp_ring_comm_group = {}

    hcg = fleet.get_hybrid_communicate_group()
    mp_group = hcg.get_model_parallel_group()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    degrees = [hcg._dp_degree, hcg._pp_degree, hcg._sharding_degree, hcg._mp_degree]
    group_arr = np.arange(0, world_size).reshape(degrees)
    degree = hcg._mp_degree
    arr = group_arr.reshape((-1, degree))

    for i in range(world_size // degree):
        ranks = arr[i].tolist()
        ranks = ranks + [ranks[0]]
        for idx in range(len(ranks) - 1):
            p2p_ranks = [ranks[idx], ranks[idx + 1]]
            group = dist.new_group(p2p_ranks)
            if p2p_ranks[0] in mp_group.ranks or p2p_ranks[1] in mp_group.ranks:
                src_rank = mp_group.get_group_rank(p2p_ranks[0])
                dst_rank = mp_group.get_group_rank(p2p_ranks[1])
                _mp_ring_comm_group[f"mp_{src_rank}to{dst_rank}"] = group

    # register attr and method to hcg instance
    setattr(hcg, "_mp_ring_comm_group", _mp_ring_comm_group)

    def get_model_parallel_ring_group(self):
        return self._mp_ring_comm_group

    hcg.get_model_parallel_ring_group = types.MethodType(
        get_model_parallel_ring_group, hcg
    )


def get_smp_communication_info():
    """get smp communication info"""
    hcg = fleet.get_hybrid_communicate_group()
    mp_rank = hcg.get_model_parallel_rank()
    mp_ranks = hcg.get_model_parallel_world_size()
    if not hasattr(hcg, "_mp_ring_comm_group"):
        init_model_parallel_ring_group()
    mp_ring_comm_group = hcg.get_model_parallel_ring_group()
    mp_group = hcg.get_model_parallel_group()

    next_mp_rank = (mp_rank + 1) % len(mp_group.ranks)
    prev_mp_rank = (mp_rank - 1 + len(mp_group.ranks)) % len(mp_group.ranks)
    send_group = mp_ring_comm_group[f"mp_{mp_rank}to{next_mp_rank}"]
    recv_group = mp_ring_comm_group[f"mp_{prev_mp_rank}to{mp_rank}"]
    send_dst = mp_group.ranks[next_mp_rank]
    recv_src = mp_group.ranks[prev_mp_rank]

    return mp_rank, mp_ranks, mp_group, send_group, recv_group, send_dst, recv_src


def smp_row_col_col_linear(
    x, weight, bias=None, transpose_y=False, return_x=False, name=None
):
    """
    y = x * weight + b = matmul(x, weight) + b
    """

    mp_rank, mp_ranks, mp_group, send_group, recv_group, send_dst, recv_src = (
        get_smp_communication_info()
    )

    hidden_size = x.shape[-1]
    assert (
        hidden_size % mp_ranks == 0
    ), f"hidden_size {hidden_size} must be divided by mp_ranks {mp_ranks}"
    micro_size = hidden_size // mp_ranks
    # reverse order [mp_ranks-1, ..., 1, 0]
    cal_index = list(range(mp_ranks - 1, -1, -1))
    # shift
    shift = mp_ranks - mp_rank - 1
    cal_index = cal_index[shift:] + cal_index[:shift]

    xi = x
    y = []

    if return_x:
        x_recvs = [x]
    else:
        x_recvs = []

    for idx, t in enumerate(cal_index):
        start = t * micro_size
        end = start + micro_size

        # launch async send and recv
        if idx < mp_ranks - 1:
            if mp_rank % 2 == 0:
                task_send = dist.isend(xi, dst=send_dst, group=send_group)
            else:
                if return_x:
                    x_recvs.append(paddle.empty_like(xi))
                    x_recv = x_recvs[idx + 1]
                else:
                    if len(x_recvs) < 2:
                        x_recvs.append(paddle.empty_like(xi))
                    x_recv = x_recvs[idx % 2]
                task_recv = dist.irecv(x_recv, src=recv_src, group=recv_group)

            if mp_rank % 2 == 0:
                if return_x:
                    x_recvs.append(paddle.empty_like(xi))
                    x_recv = x_recvs[idx + 1]
                else:
                    if len(x_recvs) < 2:
                        x_recvs.append(paddle.empty_like(xi))
                    x_recv = x_recvs[idx % 2]
                task_recv = dist.irecv(x_recv, src=recv_src, group=recv_group)
            else:
                task_send = dist.isend(xi, dst=send_dst, group=send_group)

        yi = paddle.matmul(xi, weight, transpose_y=transpose_y)

        y.append(yi)
        # we need to sync and get received xi
        if idx < mp_ranks - 1:
            task_send.wait()
            task_recv.wait()
            xi = x_recv

    # shift results
    shift = mp_rank + 1
    y = y[shift:] + y[:shift]
    y = y[::-1]
    y = paddle.concat(y, axis=-2)

    if return_x:
        x_recvs = x_recvs[shift:] + x_recvs[:shift]
        x_recvs = x_recvs[::-1]
        x = paddle.concat(x_recvs, axis=-2)

    if bias is not None:
        y = y + bias

    if return_x:
        return y, x
    else:
        return y


def smp_col_row_row_linear(x, weight, bias=None, transpose_y=False, name=None):
    """
    y = x * weight + b = matmul(x, weight) + b
    """

    mp_rank, mp_ranks, mp_group, send_group, recv_group, send_dst, recv_src = (
        get_smp_communication_info()
    )

    size = x.shape[-2]
    assert size % mp_ranks == 0, f"size {size} must be divided by mp_ranks {mp_ranks}"
    micro_size = size // mp_ranks
    # reverse order [mp_ranks-1, ..., 1, 0]
    cal_index = list(range(mp_ranks - 1, -1, -1))
    # shift
    shift = mp_ranks - mp_rank
    cal_index = cal_index[shift:] + cal_index[:shift]

    y = None
    task_send = None
    task_recv = None
    y_recv = None

    for idx, t in enumerate(cal_index):
        start = t * micro_size
        end = start + micro_size
        # slice and calculate matmul

        if False:  # len(x.shape) == 2:
            xi = x._slice(start, end)
        else:
            xi = paddle.slice(x, axes=[-2], starts=[start], ends=[end])

        yi = paddle.matmul(xi, weight, transpose_y=transpose_y)
        # we need to sync and get received
        if idx > 0:
            task_send.wait()
            task_recv.wait()
            yi_send = yi + y_recv
        else:
            yi_send = yi
            y_recv = paddle.empty_like(yi)
        # launch async send and recv
        if idx < mp_ranks - 1:
            if mp_rank % 2 == 0:
                task_send = dist.isend(yi_send, dst=send_dst, group=send_group)
            else:
                task_recv = dist.irecv(y_recv, src=recv_src, group=recv_group)

            if mp_rank % 2 == 0:
                task_recv = dist.irecv(y_recv, src=recv_src, group=recv_group)
            else:
                task_send = dist.isend(yi_send, dst=send_dst, group=send_group)

    y = yi_send
    if bias is not None:
        y = y + bias

    return y


def smp_row_col_col_cal_dw(x, dy, y=None, name=None):
    """
    y = x^T * weight + b = matmul(x, weight, transpose_x=True) + b
    """

    mp_rank, mp_ranks, mp_group, send_group, recv_group, send_dst, recv_src = (
        get_smp_communication_info()
    )

    size = dy.shape[-2]
    assert size % mp_ranks == 0, f"size {size} must be divided by mp_ranks {mp_ranks}"
    micro_size = size // mp_ranks

    # reverse order [mp_ranks-1, ..., 1, 0]
    cal_index = list(range(mp_ranks - 1, -1, -1))
    # shift
    shift = mp_ranks - mp_rank - 1
    cal_index = cal_index[shift:] + cal_index[:shift]

    multi_precision = False
    if y is not None and y.dtype == paddle.float32:
        multi_precision = True
    xi = x
    x_recvs = []
    for idx, t in enumerate(cal_index):
        start = t * micro_size
        end = start + micro_size
        # launch async send and recv
        if idx < mp_ranks - 1:
            if mp_rank % 2 == 0:
                task_send = dist.isend(xi, dst=send_dst, group=send_group)
            else:
                if len(x_recvs) < 2:
                    x_recvs.append(paddle.empty_like(xi))
                x_recv = x_recvs[idx % 2]
                task_recv = dist.irecv(x_recv, src=recv_src, group=recv_group)

            if mp_rank % 2 == 0:
                if len(x_recvs) < 2:
                    x_recvs.append(paddle.empty_like(xi))
                x_recv = x_recvs[idx % 2]
                task_recv = dist.irecv(x_recv, src=recv_src, group=recv_group)
            else:
                task_send = dist.isend(xi, dst=send_dst, group=send_group)

        # slice and calculate matmul
        if False:  # len(x.shape) == 2:
            dyi = dy._slice(start, end)
        else:
            dyi = paddle.slice(dy, axes=[-2], starts=[start], ends=[end])

        # TODO: fuse d_bias
        y, _ = paddle._C_ops.fused_linear_param_grad_add(
            xi, dyi, y, None, multi_precision, False
        )

        # we need to sync and get received
        if idx < mp_ranks - 1:
            task_send.wait()
            task_recv.wait()
            xi = x_recv

    return y


def smp_col_row_row_cal_dw(x, dy, return_dy=False, dw=None, name=None):
    """cal dw for smp_col_row_row_cal"""

    mp_rank, mp_ranks, mp_group, send_group, recv_group, send_dst, recv_src = (
        get_smp_communication_info()
    )

    hidden_size = x.shape[-2]
    assert (
        hidden_size % mp_ranks == 0
    ), f"hidden_size {hidden_size} must be divided by mp_ranks {mp_ranks}"
    micro_hidden_size = hidden_size // mp_ranks

    # reverse order [mp_ranks-1, ..., 1, 0]
    cal_index = list(range(mp_ranks - 1, -1, -1))
    # shift
    shift = mp_ranks - mp_rank - 1
    cal_index = cal_index[shift:] + cal_index[:shift]

    dyi = dy

    if return_dy:
        dyi_recvs = [dy]
    else:
        dyi_recvs = []

    multi_precision = False
    if dw is not None and dw.dtype == paddle.float32:
        multi_precision = True

    for idx, t in enumerate(cal_index):
        start = t * micro_hidden_size
        end = start + micro_hidden_size

        # launch async send and recv
        if idx < mp_ranks - 1:
            if mp_rank % 2 == 0:
                task_send = dist.isend(dyi, dst=send_dst, group=send_group)
            else:
                if return_dy:
                    dyi_recvs.append(paddle.empty_like(dyi))
                    dyi_recv = dyi_recvs[idx + 1]
                else:
                    if len(dyi_recvs) < 2:
                        dyi_recvs.append(paddle.empty_like(dyi))
                    dyi_recv = dyi_recvs[idx % 2]
                task_recv = dist.irecv(dyi_recv, src=recv_src, group=recv_group)

            if mp_rank % 2 == 0:
                if return_dy:
                    dyi_recvs.append(paddle.empty_like(dyi))
                    dyi_recv = dyi_recvs[idx + 1]
                else:
                    if len(dyi_recvs) < 2:
                        dyi_recvs.append(paddle.empty_like(dyi))
                    dyi_recv = dyi_recvs[idx % 2]
                task_recv = dist.irecv(dyi_recv, src=recv_src, group=recv_group)
            else:
                task_send = dist.isend(dyi, dst=send_dst, group=send_group)

        # slice and calculate matmul
        if False:  # len(x.shape) == 2:
            xi = x._slice(start, end)
        else:
            xi = paddle.slice(x, axes=[-2], starts=[start], ends=[end])
        dw, _ = paddle._C_ops.fused_linear_param_grad_add(
            xi, dyi, dw, None, multi_precision, False
        )

        # we need to sync and get received xi
        if idx < mp_ranks - 1:
            task_send.wait()
            task_recv.wait()
            dyi = dyi_recv

    if return_dy:
        shift = mp_rank + 1
        dyi_recvs = dyi_recvs[shift:] + dyi_recvs[:shift]
        dyi_recvs = dyi_recvs[::-1]
        dy = paddle.concat(dyi_recvs, axis=-2)
        return dw, dy
    else:
        return dw


def smp_row_col_col_linear_grad(dy, x, weight, bias=None, name=None):
    """cal grad for smp_row_col_col_linear"""
    mp_rank, mp_ranks, mp_group, send_group, recv_group, send_dst, recv_src = (
        get_smp_communication_info()
    )

    dx = paddle.matmul(dy, weight, transpose_y=True)
    dx, task_dx = reduce_scatter(dx, mp_group, sync_op=False)

    if x.shape[0] != dy.shape[0]:
        if hasattr(weight, "main_grad"):
            weight.main_grad = smp_row_col_col_cal_dw(x, dy, y=weight.main_grad)
            dw = None
        else:
            dw = smp_row_col_col_cal_dw(x, dy)
        if bias is not None:
            db = paddle.sum(dy, axis=list(range(len(dy.shape) - 1)))
        else:
            db = None
    else:
        if bias is not None:
            if hasattr(weight, "main_grad") and hasattr(bias, "main_grad"):
                weight.main_grad, bias.main_grad = (
                    paddle._C_ops.fused_linear_param_grad_add(
                        x, dy, weight.main_grad, bias.main_grad, True, True
                    )
                )
                dw = None
                db = None
            else:
                dw, db = paddle._C_ops.fused_linear_param_grad_add(
                    x, dy, None, None, False, True
                )
            # db = paddle.sum(dy, axis=list(range(len(dy.shape)-1)))
        else:
            dw = paddle.matmul(x, dy, transpose_x=True)
            db = None

    task_dx.wait()

    return dx, dw, db


def smp_col_row_row_linear_grad(dy, x, weight, bias=None, low_memory=True, name=None):
    """cal grad for smp_col_row_row_linear"""
    mp_rank, mp_ranks, mp_group, send_group, recv_group, send_dst, recv_src = (
        get_smp_communication_info()
    )

    if bias is not None:
        db = paddle.sum(dy, axis=list(range(len(dy.shape) - 1)))
        task_bias_grad = dist.all_reduce(db, group=mp_group, sync_op=False)

    dw = None
    if hasattr(weight, "main_grad"):
        dw = weight.main_grad

    if low_memory:
        dw, dy = smp_col_row_row_cal_dw(x, dy, return_dy=True, dw=dw)
        dx = paddle.matmul(dy, weight, transpose_y=True)
    else:
        dw = smp_col_row_row_cal_dw(x, dy, dw=dw)
        dx = smp_row_col_col_linear(dy, weight, transpose_y=True)

    if bias is not None:
        task_bias_grad.wait()
    else:
        db = None

    if hasattr(weight, "main_grad"):
        return dx, None, db

    return dx, dw, db


class SMPRowShardedLinearFunction(PyLayer):
    """ """

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        split_x=False,
        split_axis=0,
        gather_y=False,
        gather_axis=0,
        low_memory=True,
        name=None,
    ):
        """ """
        # Note(GuoxiaWang): save input dtype to recover grad dtype for amp
        ctx.x_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        ctx.name = name
        ctx.has_bias = bias is not None
        if bias is not None:
            ctx.bias_dtype = bias.dtype

        if split_x:
            world_size = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            mp_rank = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()
            x = paddle.split(x, world_size, axis=split_axis)[mp_rank]

        if ctx.has_bias:
            ctx.save_for_backward(x, weight, bias)
        else:
            ctx.save_for_backward(x, weight)
        ctx.split_x = split_x
        ctx.gather_y = gather_y
        ctx.split_axis = split_axis
        ctx.gather_axis = gather_axis
        ctx.low_memory = low_memory

        # Note(GuoxiaWang): it will auto cast dtype when enabling amp
        y = smp_col_row_row_linear(x, weight, bias)

        if gather_y:
            mp_group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            tensor_list = []
            dist.all_gather(tensor_list, y, group=mp_group)
            y = paddle.concat(tensor_list, axis=gather_axis)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """ """
        if ctx.has_bias:
            x, weight, bias = ctx.saved_tensor()
        else:
            x, weight = ctx.saved_tensor()
            bias = None
        if ctx.gather_y:
            world_size = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            mp_rank = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()
            grad_output = paddle.split(grad_output, world_size, axis=ctx.gather_axis)[
                mp_rank
            ]

        # Note(GuoxiaWang): it needs to be manually converted to the type when enabling the amp.
        if x.dtype != grad_output.dtype:
            x = x.astype(grad_output.dtype)
        if weight.dtype != grad_output.dtype:
            weight = weight.astype(grad_output.dtype)
        if bias is not None and bias.dtype != grad_output.dtype:
            bias = bias.astype(grad_output.dtype)

        x_grad, weight_grad, bias_grad = smp_col_row_row_linear_grad(
            grad_output, x, weight, bias, ctx.low_memory
        )

        if ctx.split_x:
            mp_group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            tensor_list = []
            dist.all_gather(tensor_list, x_grad, group=mp_group)
            x_grad = paddle.concat(tensor_list, axis=ctx.split_axis)

        # Note(GuoxiaWang): recover the grad dtype
        if x_grad.dtype != ctx.x_dtype:
            x_grad = x_grad.astype(ctx.x_dtype)
        if weight_grad is not None and weight_grad.dtype != ctx.weight_dtype:
            weight_grad = weight_grad.astype(ctx.weight_dtype)
        if bias is not None and bias_grad.dtype != ctx.bias_dtype:
            bias_grad = bias_grad.astype(ctx.bias_dtype)

        if bias is not None:
            return x_grad, weight_grad, bias_grad
        else:
            return x_grad, weight_grad


class SMPColumnShardedLinearFunction(PyLayer):
    """ """

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        split_x=False,
        split_axis=0,
        gather_y=False,
        gather_axis=0,
        low_memory=True,
        name=None,
    ):
        """ """
        # Note(GuoxiaWang): save input dtype to recover grad dtype for amp
        ctx.x_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        ctx.name = name
        ctx.has_bias = bias is not None
        if bias is not None:
            ctx.bias_dtype = bias.dtype

        if split_x:
            world_size = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            mp_rank = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()
            x = paddle.split(x, world_size, axis=split_axis)[mp_rank]

        ctx.split_x = split_x
        ctx.gather_y = gather_y
        ctx.split_axis = split_axis
        ctx.gather_axis = gather_axis
        ctx.low_memory = low_memory

        # Note(GuoxiaWang): it will auto cast dtype when enable amp
        if low_memory or x.stop_gradient:
            y = smp_row_col_col_linear(x, weight, bias)
        else:
            y, x = smp_row_col_col_linear(x, weight, bias, return_x=True)

        if ctx.has_bias:
            ctx.save_for_backward(x, weight, bias)
        else:
            ctx.save_for_backward(x, weight)

        if gather_y:
            mp_group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            tensor_list = []
            dist.all_gather(tensor_list, y, group=mp_group)
            y = paddle.concat(tensor_list, axis=gather_axis)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """ """
        if ctx.has_bias:
            x, weight, bias = ctx.saved_tensor()
        else:
            x, weight = ctx.saved_tensor()
            bias = None
        if ctx.gather_y:
            world_size = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            mp_rank = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()
            grad_output = paddle.split(grad_output, world_size, axis=ctx.gather_axis)[
                mp_rank
            ]

        # Note(GuoxiaWang): it needs to be manually converted to the type when enabling the amp.
        if x.dtype != grad_output.dtype:
            x = x.astype(grad_output.dtype)
        if weight.dtype != grad_output.dtype:
            weight = weight.astype(grad_output.dtype)
        if bias is not None and bias.dtype != grad_output.dtype:
            bias = bias.astype(grad_output.dtype)

        x_grad, weight_grad, bias_grad = smp_row_col_col_linear_grad(
            grad_output, x, weight, bias
        )

        if ctx.split_x:
            mp_group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            tensor_list = []
            dist.all_gather(tensor_list, x_grad, group=mp_group)
            x_grad = paddle.concat(tensor_list, axis=ctx.split_axis)

        # Note(GuoxiaWang): recover the grad dtype
        if x_grad.dtype != ctx.x_dtype:
            x_grad = x_grad.astype(ctx.x_dtype)
        if weight_grad is not None and weight_grad.dtype != ctx.weight_dtype:
            weight_grad = weight_grad.astype(ctx.weight_dtype)
        if bias is not None and bias_grad.dtype != ctx.bias_dtype:
            bias_grad = bias_grad.astype(ctx.bias_dtype)

        if bias is not None:
            return x_grad, weight_grad, bias_grad
        else:
            return x_grad, weight_grad


def smp_row_parallel_linear(
    x,
    weight,
    bias=None,
    split_x=False,
    split_axis=0,
    gather_y=False,
    gather_axis=0,
    low_memory=True,
    name=None,
):
    """smp row parallel api"""
    return SMPRowShardedLinearFunction.apply(
        x,
        weight,
        bias=bias,
        split_x=split_x,
        split_axis=split_axis,
        gather_y=gather_y,
        gather_axis=gather_axis,
        low_memory=low_memory,
        name=name,
    )


def smp_column_parallel_linear(
    x,
    weight,
    bias=None,
    split_x=False,
    split_axis=0,
    gather_y=False,
    gather_axis=0,
    low_memory=True,
    name=None,
):
    """smp column parallel api"""
    return SMPColumnShardedLinearFunction.apply(
        x,
        weight,
        bias=bias,
        split_x=split_x,
        split_axis=split_axis,
        gather_y=gather_y,
        gather_axis=gather_axis,
        low_memory=low_memory,
        name=name,
    )


class SMPRowParallelLinear(paddle.nn.Layer):
    """Class for SMP row parallel"""

    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        bias_attr=None,
        input_split=False,
        input_split_axis=0,
        gather_output=False,
        gather_axis=0,
        mp_group=None,
        low_memory=True,
        name=None,
    ):
        """ """
        super().__init__()

        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            if mp_group is None
            else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            if mp_group is None
            else mp_group.nranks
        )
        self._name = name
        self.is_mp = self.world_size > 1
        self.input_split = input_split
        self.input_split_axis = input_split_axis
        self.gather_output = gather_output
        self.gather_axis = gather_axis
        self.low_memory = low_memory

        assert in_features % self.world_size == 0, (
            "Number of rows of the weight for linear ({}) must be"
            " divisible by model parallel size ({})".format(
                in_features, self.world_size
            )
        )

        self.input_size_per_partition = in_features // self.world_size

        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()

        if self.is_mp and paddle.in_dynamic_mode():
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[self.input_size_per_partition, out_features],
                    attr=self._weight_attr,
                    dtype=self._dtype,
                    is_bias=False,
                )
        else:
            self.weight = self.create_parameter(
                shape=[self.input_size_per_partition, out_features],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )

        self.weight.is_distributed = True if self.is_mp else False

        if self.weight.is_distributed:
            self.weight.split_axis = 0

        if bias_attr:
            self.bias = self.create_parameter(
                shape=[out_features],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype,
                is_bias=True,
            )
        else:
            self.bias = None

        self.linear = smp_row_parallel_linear

    def forward(self, x):
        """ """
        output = self.linear(
            x,
            self.weight,
            self.bias,
            split_x=self.input_split and self.is_mp,
            split_axis=self.input_split_axis,
            gather_y=self.gather_output and self.is_mp,
            gather_axis=self.gather_axis,
            low_memory=self.low_memory,
            name=self._name,
        )
        return output


class SMPColumnParallelLinear(paddle.nn.Layer):
    """Class for SMP column parallel"""

    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        bias_attr=None,
        input_split=False,
        input_split_axis=0,
        gather_output=False,
        gather_axis=0,
        mp_group=None,
        low_memory=True,
        name=None,
    ):
        """ """
        super().__init__()

        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            if mp_group is None
            else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            if mp_group is None
            else mp_group.nranks
        )
        self._name = name
        self.is_mp = self.world_size > 1
        self.input_split = input_split
        self.input_split_axis = input_split_axis
        self.gather_output = gather_output
        self.gather_axis = gather_axis
        self.low_memory = low_memory

        assert out_features % self.world_size == 0, (
            "Number of column of the weight for linear ({}) must be"
            " divisible by model parallel size ({})".format(
                out_features, self.world_size
            )
        )
        self.output_size_per_partition = out_features // self.world_size

        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()

        if self.is_mp and paddle.in_dynamic_mode():
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[in_features, self.output_size_per_partition],
                    attr=self._weight_attr,
                    dtype=self._dtype,
                    is_bias=False,
                )
        else:
            self.weight = self.create_parameter(
                shape=[in_features, self.output_size_per_partition],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )

        self.weight.is_distributed = True if self.is_mp else False

        if self.weight.is_distributed:
            self.weight.split_axis = 1

        if bias_attr:
            self.bias = self.create_parameter(
                shape=[self.output_size_per_partition],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype,
                is_bias=True,
            )
            self.bias.is_distributed = True if self.is_mp else False
        else:
            self.bias = None

        self.linear = smp_column_parallel_linear

    def forward(self, x):
        """ """
        output = self.linear(
            x,
            self.weight,
            self.bias,
            split_x=self.input_split and self.is_mp,
            split_axis=self.input_split_axis,
            gather_y=self.gather_output and self.is_mp,
            gather_axis=self.gather_axis,
            low_memory=self.low_memory,
            name=self._name,
        )
        return output
