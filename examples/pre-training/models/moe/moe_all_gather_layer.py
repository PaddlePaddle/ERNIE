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
@author: kebo
@contact: kebo01@baidu.com

@version: 1.0
@file: moe_layer_all_gather.py
@time: 2024/09/21 15:11:10
@Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved

这一行开始写关于本文件的说明与解释


"""
from typing import Tuple, List, Dict, Optional, Callable
import logging
import contextlib
import numpy as np
import inspect

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle import framework
import paddle.nn.functional as F
from paddle import nn
from paddle.autograd import PyLayer
from paddle.distributed.communication.group import _get_global_group
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.communication.group import Group

from models.moe.sinkhorn_gate import SinkHornGateFused
from models.moe.round_robin_gate import RoundRobinGateFused
from models.moe.top2_gate import TopKGateFused
from models.comm_utils import reduce_scatter, all_gather
from models.sequence_parallel_utils import (
    GatherOp,  # 进入同步区，从此以后 tp 间必须执行一样的操作
    AllGatherOp,  # 进入异步区，tp间的操作可以不一样
    ReduceScatterOp,
    ScatterOp,
    get_async_loader,
    hack_offload_wait,
)
from models.utils import global_training_logs_enabled, manual_backward
from paddle.incubate.tensor.manipulation import async_offload

from .moe_layer import MOELayer, fuse_logging
from paddleformers.utils.tools import get_env_device
from models.utils import get_global_training_logs

global_training_logs = (
    get_global_training_logs()
)  # 没有erniebot的环境下无法打印 debug 量
try:
    import moe_router_loss_ops
except ImportError:
    moe_router_loss_ops = None

try:
    from paddle import scatter_add_
except ImportError:
    scatter_add_ = None


def profile(_):
    """dumy profile"""
    return contextlib.nullcontext()


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
        import moe_ops_partial
    except ImportError:
        moe_ops_partial = None
        logger.warning(
            "`moe-ops-partial` not found, run "
            "`python3  src/ernie_core/ops/moe/setup.py  install` to install"
        )
    try:
        import moe_ops_partial_nosoftmaxtopk
    except ImportError:
        moe_ops_partial_nosoftmaxtopk = None
        logger.warning(
            "`moe-ops-partial-nosoftmaxtopk` not found, run "
            "`python3  src/ernie_core/ops/moe/setup.py  install` to install"
        )

    try:
        import moe_utils
    except ImportError:
        moe_utils = None
        logger.warning(
            "`moe_utils` not found, run "
            "`python3  src/ernie_core/ops/moe/setup.py  install` to install"
        )


@paddle.no_grad()
def all_to_all_unpadding(input, output, input_sizes, output_sizes, group=None):
    """helper function for unbalaced alltoall, Feat. Liyurui"""
    # `alltoall_single` 支持发送 0 tensor 之后，可以直接调用。
    if group.nranks <= 1:
        # output.copy_(input, True)
        output[:] = input[:]
        return None
    task = dist.stream.alltoall_single(
        output,
        input,
        output_sizes,
        input_sizes,
        group=group,
        sync_op=False,
        use_calc_stream=False,
    )
    return task
    if input is not None and output is not None:
        assert input.dtype == output.dtype, (input.dtype, output.dtype)

    if group is None:
        group = _get_global_group()
    world_size = dist.get_world_size(group)
    assert (
        len(input_sizes) == len(output_sizes) == world_size
    ), f"#input-offset={len(input_sizes)}, #output-offset={len(output_sizes)}, world-size={world_size}"
    op_list = []
    send_ptr = recv_ptr = 0
    for j, (out_size, in_size) in enumerate(zip(output_sizes, input_sizes)):
        if in_size > 0:
            send_partial_tensor = input.slice((0,), send_ptr, send_ptr + in_size)
            # logger.info(f"[send--send] {dist.get_rank()}->{group.ranks[j]}, len={send_partial_tensor.shape}")
            op_list.append(
                dist.P2POp(dist.isend, send_partial_tensor, group.ranks[j], group)
            )
            send_ptr += in_size
        if out_size > 0:
            recv_partial_tensor = output.slice((0,), recv_ptr, recv_ptr + out_size)
            recv_ptr += out_size
            # logger.info(f"[recv--recv] {dist.get_rank()}<-{group.ranks[j]}, len={recv_partial_tensor.shape}")
            op_list.append(
                dist.P2POp(dist.irecv, recv_partial_tensor, group.ranks[j], group)
            )
    return op_list
    # if len(op_list):
    #     dist.batch_isend_irecv(op_list)
    #     paddle.device.synchronize()
    # return


def allgather_async(input, group=None):
    """
    allgather async
    """
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    if parallelism == 1:
        return input.clone(), None
    output_shape = input.shape
    output_shape[0] = output_shape[0] * parallelism
    output = paddle.empty(shape=output_shape, dtype=input.dtype)
    task = dist.stream.all_gather(
        output, input, group=group, use_calc_stream=False, sync_op=False
    )
    return output, task


def reduce_scatter_async(input, group=None):
    """reduce_scatter"""
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    if parallelism == 1:
        return input.clone(), None
    output_shape = input.shape
    assert (
        input.shape[0] % parallelism == 0
    ), "Input sequence length {0} can't be divided exactly by sequence parallelism {1}".format(
        input.shape[0], parallelism
    )
    output_shape[0] = output_shape[0] // parallelism
    output = paddle.empty(shape=output_shape, dtype=input.dtype)
    task = dist.stream.reduce_scatter(
        output,
        input,
        op=dist.ReduceOp.SUM,
        group=group,
        use_calc_stream=False,
        sync_op=False,
    )
    return output, task


class AllGatherAsync(PyLayer):
    """
    input shape: [s/n, b, h], n is mp parallelism
    after forward shape: [s, b, h]
    """

    @staticmethod
    def forward(ctx, input, *fn_args, group=None, fn=None, is_first_fwd=False):
        """fwd"""
        ctx.group = group
        if dist.get_world_size(group) <= 1:
            ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
            return (input,) + fn_out
        out, task = allgather_async(input, group=group)
        ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
        task and task.wait()
        return (out,) + fn_out

    # grad shape: [s, b, h], n is mp parallelism
    # after forward shape: [s/n, b, h]
    @staticmethod
    def backward(ctx, grad, *fn_out_grads):
        """bwd"""
        if dist.get_world_size(ctx.group) <= 1:
            fn_args_grads = ctx.bwf(*fn_out_grads)
            return (grad,) + fn_args_grads

        grad, task = reduce_scatter_async(grad, group=ctx.group)
        fn_args_grads = ctx.bwf(*fn_out_grads)
        task and task.wait()
        return (grad,) + fn_args_grads


class ReshardCombineWeight(PyLayer):
    """
    Combine weight(shape==[Seq,k]) 有 2 种切片方式：
        * expert 切片：指向非本机 expert 的值为 0
        * seq 切片：在 `Seq`维度切片
    从`expert 切片` 到`seq` 切片的操作是 reduce -> scatter
    反操作是 all_gather -> mask_select
    """

    @staticmethod
    def forward(ctx, input, group=None):
        """fwd"""
        ctx.mask = input == 0.0
        ctx.group = group
        return reduce_scatter(input, group=group)

    @staticmethod
    def backward(ctx, grad):
        """bwd"""
        gathered = all_gather(grad, group=ctx.group)
        return gathered.masked_fill(
            ctx.mask,
            0.0,
        )


class AlltoAllSmart(paddle.autograd.PyLayer):
    """
    支持不均匀（包括 不发）send/recv 的 Alltoall.
    """

    @staticmethod
    def forward(
        ctx,
        *inputs,
        router_loss_fn: Callable = None,
        forward_func_dict: Dict[int, Callable] = None,
        local_expert_id=None,
        send_rank_global=None,
        recv_rank_global=None,
        num_local_experts=None,
        capacity=None,
        use_padding=True,
        expert_num_global=None,
        is_first_fwd=None,
        group=None,
        recv_size=None,
        send_counts=None,
        recv_counts=None,
        send_counts_num=None,
        recv_counts_num=None,
    ):
        """
        批量点到点通信
        Args:
            input*: Tensor[*] permuted hidden before expert-fn
            router_fn_args [*]: args to `router_loss_fn`
            forward_func_dict: Optional[Dict[callable]],
                expert forward fn dict, do expert/alltoall overlap if specified.
            router_loss_fn:
                func to perform router loss calc
            send_rank: Tensor[int]: 发送 rank 列表， group 内必须保持一致。 use local-rank!
            recv_rank: Tensor[int]: 接受 rank 列表， group 内必须保持一致。 use local-rank!
            int: num_local_experts
            group: process group
        Returns:
            output: Tensor[*]
            send_counts: Tensor[local_expertnum, world_size]
            recv_counts: Tensor[local_expertnum, world_size]
        """
        if group is None:
            group = _get_global_group()
        # ctx.send_rank = send_rank
        # ctx.recv_rank = recv_rank
        router_loss_args = inputs[num_local_experts:]
        inputs = inputs[:num_local_experts]

        ctx.group = group
        ctx.use_padding = use_padding
        ctx.num_local_experts = num_local_experts
        ctx.input_shape = [i.shape if i is not None else None for i in inputs]

        this_rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        capacity = len(send_rank_global) // world_size // num_local_experts
        ctx.capacity = capacity
        assert len(local_expert_id) == len(recv_rank_global), (
            len(local_expert_id),
            len(recv_rank_global),
        )

        for i in inputs:
            if i is not None:
                input_dtype = i.dtype
                input_shape = i.shape
                break
        else:
            raise RuntimeError("all inputs are None")

        output = paddle.zeros([recv_size] + input_shape[1:], dtype=input_dtype)
        # send_counts, recv_counts = [], []
        # assert len(local_expert_nums) == num_local_experts, (local_expert_nums, num_local_experts)
        output_ptr = 0

        tasks = []
        dummy_input = paddle.empty([0] + input_shape[1:], dtype=input_dtype)
        ctx.dummy_input = dummy_input
        ctx.bw_funcs = {}

        for i_local_expert in range(num_local_experts):
            send_count = send_counts[i_local_expert]  # remove final worldsize
            recv_count = recv_counts[i_local_expert]
            assert len(recv_count) == len(send_count) == (world_size), (
                len(recv_count),
                len(send_count),
            )

            # sanity check
            # recv_rank_this_rank = recv_rank_global[(local_expert_id == i_local_expert) &
            # (send_rank_global == this_rank)& (recv_rank_global!=world_size)]

            # # logger.info(f'recv_rank_this_rank:{recv_rank_this_rank.shape}
            # # send_rank_this_rank:{send_rank_this_rank.shape}')
            # if sum(send_count) > 0:
            #     if len(recv_rank_this_rank) > 1:
            #         logger.info(f'recv_rank_this_rank:{recv_rank_this_rank}')
            #         assert (
            #             paddle.diff(recv_rank_this_rank) >= 0
            #         ).all(), f"recv_rank_this_rank: must in ascend order, got: {recv_rank_this_rank.tolist()}"
            if send_counts_num[i_local_expert] > 0:
                input_local_expert = inputs[i_local_expert].slice(
                    (0,), 0, send_counts_num[i_local_expert]
                )
                if forward_func_dict is not None:
                    input_local_expert.stop_gradient = False
                    bwf, (input_local_expert,) = manual_backward(
                        forward_func_dict[i_local_expert],
                        is_first_fwd,
                        input_local_expert,
                    )
                    ctx.bw_funcs[i_local_expert] = bwf

                if input_local_expert is None:
                    input_local_expert = dummy_input
                input_local_expert.stop_gradient = True
            else:
                input_local_expert = dummy_input
            if recv_counts_num[i_local_expert] > 0:
                output_local_expert = output.slice(
                    (0,), output_ptr, output_ptr + recv_counts_num[i_local_expert]
                )
            else:
                output_local_expert = dummy_input

            output_ptr += recv_counts_num[i_local_expert]
            # logger.info(
            #     f"alltoall[{i_local_expert}]: send:{input_local_expert.shape}/recv:{output_local_expert.shape} "
            #     f"send-cnt:{send_count} / recv-cnt:{recv_count} "
            # )
            # logger.info(f"send_recv_count_global:{send_recv_count_global[i_local_expert]}")

            tasks.append(
                all_to_all_unpadding(
                    input_local_expert,
                    output_local_expert,
                    send_count,
                    recv_count,
                    group=group,
                )
            )
        ctx.router_loss_bwfn, (router_loss,) = manual_backward(
            router_loss_fn, is_first_fwd, *router_loss_args
        )
        # `expert_out_to_combine` 相比 `global_experts_out` 缩短了，所以需要更改 `local_scatter_index`
        with paddle.no_grad():
            recv_mask = (recv_rank_global == this_rank).astype(send_rank_global.dtype)
            with profile("alltoall-prepare2"):
                if ctx.use_padding:
                    recv_mask_alltoall_out = (
                        recv_mask.reshape([-1, num_local_experts, capacity])
                        .transpose([1, 0, 2])
                        .reshape([-1])
                    )
                    distributed_input_to_alltoall_out = paddle.maximum(
                        recv_mask_alltoall_out.cumsum() - 1,
                        paddle.zeros([1], dtype=recv_mask_alltoall_out.dtype),
                    )
                    distributed_input_to_alltoall_out = (
                        distributed_input_to_alltoall_out.view(
                            [num_local_experts, -1, capacity]
                        )
                        .transpose([1, 0, 2])
                        .reshape([-1])
                    )
                else:
                    recv_mask_alltoall_out = recv_mask.split(
                        expert_num_global
                    )  # h->d copy break overlap
                    recv_mask_alltoall_out = [
                        recv_mask_alltoall_out[
                            (iexpert % world_size) * num_local_experts
                            + (iexpert // world_size)
                        ]
                        for iexpert in range(world_size * num_local_experts)
                    ]
                    alltoall_shape = [i.shape[0] for i in recv_mask_alltoall_out]

                    recv_mask_alltoall_out = paddle.concat(recv_mask_alltoall_out, 0)
                    distributed_input_to_alltoall_out = paddle.maximum(
                        recv_mask_alltoall_out.cumsum() - 1,
                        paddle.zeros([1], dtype=recv_mask_alltoall_out.dtype),
                    )
                    distributed_input_to_alltoall_out = (
                        distributed_input_to_alltoall_out.split(alltoall_shape)
                    )

                    distributed_input_to_alltoall_out = paddle.concat(
                        [
                            distributed_input_to_alltoall_out[
                                (iexpert % num_local_experts) * world_size
                                + (iexpert // num_local_experts)
                            ]
                            for iexpert in range(world_size * num_local_experts)
                        ],
                        0,
                    )

        distributed_input_to_alltoall_out.stop_gradient = True
        for t in tasks:
            t and t.wait()
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        return output, router_loss, distributed_input_to_alltoall_out

    @staticmethod
    def backward(
        ctx,
        out_grad,
        d_routerloss,
        _,  # scatter-idx no grad
    ):
        """
        backward
        """
        grads = [
            paddle.zeros(s, dtype=out_grad.dtype) if s is not None else None
            for s in ctx.input_shape
        ]
        assert len(grads) == ctx.num_local_experts
        out_ptr = 0
        tasks = []
        tmp_g = []
        send_counts_num = ctx.send_counts.sum(-1)
        recv_counts_num = ctx.recv_counts.sum(-1)
        out_grad = out_grad.contiguous()
        for i_local_expert in range(ctx.num_local_experts):
            send_count = ctx.send_counts[i_local_expert]
            recv_count = ctx.recv_counts[i_local_expert]
            if recv_counts_num[i_local_expert] > 0:
                out_g = out_grad.slice(
                    (0,), out_ptr, out_ptr + recv_counts_num[i_local_expert]
                )
            else:
                out_g = (
                    ctx.dummy_input
                )  # paddle.empty([0,]+out_grad.shape[1:], dtype=out_grad.dtype)
            if send_counts_num[i_local_expert] > 0:
                g = grads[i_local_expert].slice(
                    (0,), 0, send_counts_num[i_local_expert]
                )
            else:
                g = ctx.dummy_input
            tmp_g.append(g)
            out_ptr += recv_counts_num[i_local_expert]
            task = all_to_all_unpadding(
                out_g, g, recv_count, send_count, group=ctx.group
            )
            tasks.append(task)
        router_fn_args_grad = ctx.router_loss_bwfn(d_routerloss)
        # logger.info(f'd_routerloss:{d_routerloss} router_fn_args_grad:{router_fn_args_grad}')

        for i_local_expert, t in enumerate(tasks):
            t and t.wait()
            send_cnt = send_counts_num[i_local_expert]
            if send_cnt > 0 and ctx.bw_funcs:
                # logger.info(f'before bw:[{i_local_expert}]: {grads[i_local_expert]}')
                (g,) = ctx.bw_funcs[i_local_expert](tmp_g[i_local_expert])
                grads[i_local_expert][:send_cnt] = g
                # logger.info(f'after bw:[{i_local_expert}]: {grads[i_local_expert]}')

        grads = [g for g in grads if g is not None]
        # router_fn_args_grad = [g for g in router_fn_args_grad if g is not None]
        return tuple(grads) + tuple(router_fn_args_grad)


class MOEAllGatherLayer(MOELayer):
    """_summary_

    Args:
        MOELayer (_type_): _description_
    """

    def __init__(
        self,
        gate: nn.Layer,
        experts: List[nn.Layer],
        layer_idx,
        shared_experts: Optional[List[nn.Layer]] = None,
        dense_experts: Optional[List[nn.Layer]] = None,  # no use
        group: Group = None,
        recompute=False,
        enable_logging: bool = False,
        k=2,
        enable_bpr: bool = False,
        all_to_all_dropout=0,
        group_experts=False,
        moe_statics=None,
    ):

        super().__init__(
            gate,
            experts,
            layer_idx,
            shared_experts,
            group,
            recompute,
            enable_logging,
            k,
            enable_bpr,
            all_to_all_dropout,
            group_experts,
            moe_statics,
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
                [self.num_local_experts, -1, dispatched_input.shape[-1]]
            )  # [e,1,c,m]
            expert_outputs = []
            assert isinstance(self.experts, nn.LayerList), type(self.experts)
            chunks = dispatched_input.unbind(0)
            assert len(chunks) == len(self.experts), (len(chunks), len(self.experts))
            for chunk, expert in zip(chunks, self.experts):
                chunk = chunk.contiguous()
                expert_outputs += [expert(chunk)]
                # logger.info(
                #     f"moe-fwd-expert allgather: {chunk.shape}"
                #     f'-> {expert_outputs[-1].shape}: {chunk.astype("float32").norm(axis=-1)}'
                # )
            expert_output = paddle.stack(expert_outputs, axis=0)  # [ecm]
            return expert_output

    def forward(
        self,
        input: paddle.Tensor,
        token_type_ids=None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """_summary_

        Args:
            input (paddle.Tensor): _description_
            token_type_ids (_type_, optional): _description_. Defaults to None.

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]: _description_
        """
        use_fuse = isinstance(
            self.gate, (RoundRobinGateFused, SinkHornGateFused, TopKGateFused)
        )
        assert use_fuse

        if input.ndim == 3:
            orig_shape = input.shape
            # clone 保平安
            input = input.reshape([-1, input.shape[-1]])
        else:
            orig_shape = None
        assert (
            len(input.shape) == 2
        ), f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"
        dispatch_token_type_ids = None
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, :-1].reshape([-1])
            dispatch_token_type_ids = token_type_ids
            if self.config.sequence_parallel:
                hcg = fleet.get_hybrid_communicate_group()
                rank = hcg.get_model_parallel_rank()
                interval = (
                    token_type_ids.shape[0] // hcg.get_model_parallel_world_size()
                )
                token_type_ids = token_type_ids.slice(
                    [0], rank * interval, (rank + 1) * interval
                )
                token_type_ids.stop_gradient = True
            if hasattr(fleet.fleet, "_hcg"):
                hcg = fleet.get_hybrid_communicate_group()
                if self.config.moe_group is hcg.get_data_parallel_group():
                    dispatch_token_type_ids = all_gather(
                        token_type_ids, axis=0, group=self.config.moe_group
                    )
            elif self.config.moe_group is _get_global_group():
                # DP-MoE
                dispatch_token_type_ids = all_gather(
                    token_type_ids, axis=0, group=self.config.moe_group
                )
        assert self.gate is not None
        if hasattr(self, "rng") and self.rng.random() < self.all_to_all_dropout:
            orig_shape_2 = input.shape
            output = self.forward_experts(input)
            output += self.gate.weight.sum() * 0.0  # hack for grad
            output = output.reshape(orig_shape or orig_shape_2)  # [e*1,c,m]
            return output, None, 0
        with profile("fused_gate_and_dispatch"):
            (
                dispatched_input,
                combine_weights,
                dispatch_mask,
                scatter_index,
                router_loss,
                gate_logits,
                gate_prob,
                offload_helper,
            ) = self.fused_gate_and_dispatch(input, token_type_ids)
        # allgather 算完dispatch后 进行Scatter 留下本机experts需要计算的tokens.
        dispatched_input = ScatterOp.apply(
            dispatched_input, group=self.config.moe_group
        )

        expert_out = (
            recompute(self.forward_experts, dispatched_input)
            if self.recompute and self.training
            else self.forward_experts(dispatched_input)
        )
        local_combine_weights = ScatterOp.apply(
            combine_weights, group=self.config.moe_group
        )
        local_scatter_index = ScatterOp.apply(
            scatter_index, group=self.config.moe_group
        )
        with profile("moe_comm"):
            global_experts_out = AllGatherOp.apply(
                expert_out, group=self.config.moe_group
            )

        with profile("combine"):
            combined_output = self.combine_expert_output(
                global_experts_out, local_combine_weights, local_scatter_index
            )
        # global_experts_out = GatherOp.apply(expert_out)
        # combined_output = self.combine_expert_output(global_experts_out, combine_weights, scatter_index)
        # combined_output = ScatterOp.apply(combined_output)
        with profile("shared-expert"):
            if self.shared_experts is not None:
                # 这里shared experts里面会再做一次allgather TODO 去掉
                shared_out = self.shared_experts(input)
                combined_output += shared_out
        if orig_shape:
            combined_output = combined_output.clone().reshape(
                orig_shape[:-1] + [combined_output.shape[-1]]
            )
        # local_dispatch_mask = ScatterOp.apply(dispatch_mask)
        with profile("calc_router_loss_and_logging"):
            router_loss2 = self.calc_router_loss_and_logging(
                router_loss,
                local_combine_weights,
                dispatch_mask,
                gate_logits,
                gate_prob,
                token_type_ids,
                dispatch_token_type_ids,
                offload_helper,
            )
        return combined_output, local_combine_weights, router_loss2, gate_logits
        # allgather

    def fused_gate_and_dispatch(self, input, token_type_ids):
        """_summary_

        Args:
            input (_type_): _description_
            token_type_ids (_type_): _description_

        Returns:
            _type_: _description_
        """
        seqlen, d_model = input.shape
        args = ()
        # 目前只有 `SinkHornGate` aka Top1 gate 支持输入 token type ids

        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape([-1])
            args = (token_type_ids,)

        offload_helper = None
        tasks = []
        if token_type_ids is not None and self.config.moe_use_hard_gate:
            offload_helper = dict()
            lm_mask = token_type_ids == 0
            is_lm = lm_mask.any()
            mm_mask = token_type_ids == 1
            is_mm = mm_mask.any()
            seq_lm = lm_mask.sum()
            seq_mm = mm_mask.sum()

            experts_type_ids = self.gate.experts_type_ids
            lm_mask = lm_mask.unsqueeze(1) & (experts_type_ids == 0).unsqueeze(0)
            mm_mask = mm_mask.unsqueeze(1) & (experts_type_ids == 1).unsqueeze(0)

            if get_env_device() == "xpu":
                is_lm_cpu = is_lm.to("cpu")
                is_mm_cpu = is_mm.to("cpu")
                seq_lm_cpu = seq_lm.to("cpu")
                seq_mm_cpu = seq_mm.to("cpu")
            else:
                async_loader = get_async_loader()
                is_lm_cpu, is_lm_task = async_offload(is_lm, async_loader)
                is_mm_cpu, is_mm_task = async_offload(is_mm, async_loader)
                seq_lm_cpu, seq_lm_task = async_offload(seq_lm, async_loader)
                seq_mm_cpu, seq_mm_task = async_offload(seq_mm, async_loader)
                tasks.extend([is_lm_task, is_mm_task, seq_lm_task, seq_mm_task])
            offload_helper["lm_mask"] = [lm_mask, is_lm_cpu, seq_lm_cpu]
            offload_helper["mm_mask"] = [mm_mask, is_mm_cpu, seq_mm_cpu]

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

        for task in tasks:
            hack_offload_wait(task)
        top_k = 1 if isinstance(self.gate, SinkHornGateFused) else self.k
        prob, max_prob = self.fused_gate_logits_process(
            gate_logits, token_type_ids, offload_helper
        )

        global_prob = GatherOp.apply(prob, group=self.config.moe_group)
        global_hidden_states = GatherOp.apply(input, group=self.config.moe_group)
        if self.group_experts:
            if max_prob is not None:
                if token_type_ids is not None:
                    global_max_prob = paddle.ones([seqlen, top_k, 1])
                    global_max_prob = paddle.scatter_nd_add(
                        global_max_prob,
                        paddle.nonzero(token_type_ids == 0),
                        -1 + max_prob,
                    )
                else:
                    global_max_prob = max_prob
            else:
                global_max_prob = paddle.ones([seqlen, top_k, 1])
            global_max_prob = GatherOp.apply(
                global_max_prob, group=self.config.moe_group
            )

        with profile("dispatch_op"):
            if "corr_bias" in inspect.signature(moe_ops.moe_gate_dispatch).parameters:
                if self.use_correction_bias:
                    compat_args = (
                        (self.moe_statics.e_score_correction_bias.reshape([-1]),)
                        if self.config.multimodel_experts
                        else (self.moe_statics.e_score_correction_bias[0],)
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
            ) = moe_ops.moe_gate_dispatch(
                global_hidden_states,
                global_prob,
                *compat_args,
                k=top_k,
                capacity=capacity * self.world_size,
                use_pad=True,
            )
        dispatch_mask = paddle.diff(F.pad(dispatch_mask, (1, 0)))
        dispatched_input.stop_gradient = False
        combine_weights_unnorm.stop_gradient = False
        scatter_index.stop_gradient = True
        dispatch_mask.stop_gradient = True

        scatter_index = scatter_index.transpose([1, 0])  # [k,s] ->[s,k]
        if self.group_experts:
            combine_weights_unnorm = (
                combine_weights_unnorm.unsqueeze(-1) * global_max_prob
            ).squeeze(-1)
            # prob 进行还原
            global_prob = (
                global_prob.reshape([global_max_prob.shape[0], top_k, -1])
                * global_max_prob
            ).reshape([global_max_prob.shape[0], -1])
        if self.gate.norm_gate_logits:
            combine_weights = combine_weights_unnorm / paddle.clip(
                combine_weights_unnorm.sum(-1, keepdim=True), min=1e-12
            )
        else:
            combine_weights = combine_weights_unnorm
        combine_weights = combine_weights.cast(dispatched_input.dtype)
        dispatched_input = dispatched_input.reshape(
            [
                self.world_size * self.num_local_experts,
                capacity * self.world_size,
                (
                    d_model
                    if not self.config.moe_multimodal_paired_experts
                    else d_model + 1
                ),
            ]
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
            global_prob,
            offload_helper,
        )


class MOEAllGatherLayerV2(MOEAllGatherLayer):
    """_summary_

    Args:
        MOELayer (_type_): _description_
    """

    def __init__(
        self,
        gate: nn.Layer,
        experts: List[nn.Layer],
        layer_idx,
        shared_experts: Optional[List[nn.Layer]] = None,
        dense_experts: Optional[List[nn.Layer]] = None,
        group: Group = None,
        recompute=False,
        enable_logging: bool = False,
        k=2,
        enable_bpr: bool = False,
        enable_reverse_token_drop=False,
        all_to_all_dropout=0,
        group_experts=False,
        use_expert_out_alltoall=True,  #
        use_expert_alltoall_overlap=False,
        use_padding=True,
        dense_token_type=3,  # considerd as dense tokens (no moe)
        moe_statics=None,
    ):
        super().__init__(
            gate,
            experts,
            layer_idx,
            shared_experts,
            dense_experts,
            group,
            recompute,
            enable_logging,
            k,
            enable_bpr,
            all_to_all_dropout,
            group_experts,
            moe_statics,
        )
        self.enable_reverse_token_drop = enable_reverse_token_drop
        self.is_allgather_moe_layer = True
        # assert self.gate.config.sequence_parallel
        self.use_padding = use_padding

        # 全局 gate gather
        self.send_rank = None
        self.local_expert_id = None
        self.dense_token_type = dense_token_type
        self.dense_experts = dense_experts
        self.capacity_tensor = None
        self.use_expert_out_alltoall = use_expert_out_alltoall
        self.use_expert_alltoall_overlap = use_expert_alltoall_overlap
        logger.info(
            f"uisng MOEAllGatherLayerV2, use_expert_out_alltoall={use_expert_out_alltoall}, "
            f"use_padding={use_padding}, use_expert_alltoall_overlap={use_expert_alltoall_overlap} "
            f"enable_reverse_token_drop={self.enable_reverse_token_drop}"
        )
        self.two = paddle.to_tensor(2, dtype=paddle.float32)
        self.zero = paddle.to_tensor(0, dtype=paddle.float32)

    def forward(
        self,
        input: paddle.Tensor,
        token_type_ids=None,
        use_dense_expert=False,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """_summary_

        Args:
            input (paddle.Tensor): _description_
            token_type_ids (_type_, optional): _description_. Defaults to None.

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]: _description_
        """
        use_fuse = isinstance(
            self.gate, (RoundRobinGateFused, SinkHornGateFused, TopKGateFused)
        )
        assert use_fuse
        if input.ndim == 3:
            orig_shape = input.shape
            # clone 保平安
            input = input.reshape([-1, input.shape[-1]])
        else:
            orig_shape = None

        assert (
            len(input.shape) == 2
        ), f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"
        dispatch_token_type_ids = None
        # 587 us
        global_dense_expert_mask = None
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, :-1].reshape([-1])
            dispatch_token_type_ids = token_type_ids
            if self.config.sequence_parallel:
                hcg = fleet.get_hybrid_communicate_group()
                rank = hcg.get_model_parallel_rank()
                interval = (
                    token_type_ids.shape[0] // hcg.get_model_parallel_world_size()
                )
                token_type_ids = token_type_ids.slice(
                    [0], rank * interval, (rank + 1) * interval
                )
                token_type_ids.stop_gradient = True
            if hasattr(fleet.fleet, "_hcg"):
                hcg = fleet.get_hybrid_communicate_group()
                if self.config.moe_group is hcg.get_data_parallel_group():
                    dispatch_token_type_ids = all_gather(
                        token_type_ids, axis=0, group=self.config.moe_group
                    )
            elif (
                self.config.moe_world_size > 1
                and self.config.moe_group is _get_global_group()
            ):
                # DP-MoE
                dispatch_token_type_ids = all_gather(
                    token_type_ids, axis=0, group=self.config.moe_group
                )

            if use_dense_expert:
                global_dense_expert_mask = (
                    dispatch_token_type_ids == self.dense_token_type
                )
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
        with profile("fused_gate_and_dispatch"):
            (
                dispatched_input,
                global_hidden_states,
                local_combine_weights,  # dispatched-expert间聚合后重新在 seq 维度切片
                expert_num_global_no_token_drop,  # 不考虑截断！
                expert_num_global,
                expert_num_global_list,
                local_scatter_index,  # dispatched-expert间聚合后重新在 seq 维度切片
                scatter_index_rev,
                router_loss,
                (gate_logits, gate_prob),
                (gate_logits_mm, gate_prob_mm),
                expert_num_local,
            ) = self.fused_gate_and_dispatch(
                input, token_type_ids, global_dense_expert_mask
            )
        seqlen_this_mp = input.shape[0]
        if len(scatter_index_rev):
            recv_rank_local = scatter_index_rev // seqlen_this_mp
        else:
            recv_rank_local = scatter_index_rev

        if self.use_padding:
            with profile("alltoall-prepare"):
                if self.send_rank is None:
                    capacity = self.gate.get_capacity(
                        input.shape[0] * self.config.moe_world_size
                    )
                    self.send_rank = (
                        paddle.arange(self.config.moe_world_size)
                        .repeat_interleave(capacity * self.num_local_experts)
                        .astype("int32")  # cap
                    )
                    self.local_expert_id = (
                        paddle.arange(self.num_local_experts)
                        .repeat_interleave(capacity)
                        .tile(self.config.moe_world_size)
                        .astype(self.send_rank.dtype)
                    )
                recv_rank, recv_rank_task = allgather_async(
                    recv_rank_local, group=self.config.moe_group
                )
                send_rank = self.send_rank
                local_expert_id = self.local_expert_id

        else:
            with profile("alltoall-prepare"):
                all_expert_num = sum(expert_num_global_list)
                # 非常慢
                if self.config.moe_group.nranks > 1:
                    recv_rank = paddle.empty(
                        [all_expert_num], dtype=recv_rank_local.dtype
                    )
                    recv_rank_task = dist.stream.alltoall_single(
                        recv_rank,
                        recv_rank_local.tile(self.config.moe_world_size),
                        [
                            sum(
                                expert_num_global_list[
                                    i
                                    * self.num_local_experts : (i + 1)
                                    * self.num_local_experts
                                ]
                            )
                            for i in range(self.config.moe_world_size)
                        ],  # output-size
                        [len(recv_rank_local)]
                        * self.config.moe_world_size,  # input-size
                        group=self.config.moe_group,
                        sync_op=False,
                        use_calc_stream=False,
                    )
                else:
                    recv_rank_task = None
                    recv_rank = recv_rank_local.tile(self.config.moe_world_size)
                # 尝试过用 self.id_buffer + slice,显然会更慢, 干脆直接在 np 上造完整个 id，后续写一手 C
                if moe_utils is None:
                    send_rank_cpu = np.concatenate(  # TOO SLOW!!! break every thing
                        [
                            np.full([j], i // self.num_local_experts, dtype="int32")
                            for i, j in enumerate(expert_num_global_list)
                        ],
                        0,
                    )
                    local_expert_id_cpu = np.concatenate(
                        [
                            np.full([j], i % self.num_local_experts, dtype="int32")
                            for i, j in enumerate(expert_num_global_list)
                        ],
                        0,
                    )
                    gpu_ids = paddle.to_tensor(
                        np.stack([send_rank_cpu, local_expert_id_cpu], 0), place="gpu"
                    )
                    send_rank, local_expert_id = gpu_ids.unbind(0)
                else:
                    send_rank, local_expert_id = (
                        moe_utils.build_src_rank_and_local_expert_id(
                            expert_num_global,
                            expert_num_global_list,
                            self.num_local_experts,
                        )
                    )

        with profile("moe_comm"):
            if not self.use_expert_out_alltoall:
                expert_outs = (
                    recompute(self.forward_experts, *dispatched_input)
                    if self.recompute and self.training
                    else self.forward_experts(*dispatched_input)
                )
                expert_outs = paddle.concat(
                    [e for e in expert_outs if e is not None], axis=0
                )  # [e*c,m]
                expert_out_to_combine = AllGatherOp.apply(
                    expert_outs, group=self.config.moe_group
                )  # for test
                router_loss2 = self.calc_router_loss_and_logging(
                    router_loss,
                    gate_logits,
                    gate_prob,
                    gate_logits_mm,
                    gate_prob_mm,
                    local_combine_weights,
                    expert_num_global_no_token_drop,
                    token_type_ids,
                    dispatch_token_type_ids,
                )
            else:
                if self.enable_logging and global_training_logs_enabled():
                    capacity = self.gate.get_capacity(
                        input.shape[0] * self.config.moe_world_size
                    )
                    _log = {}
                    valid_usage = [e for e in expert_num_global_list if e > 0]
                    if valid_usage:
                        max_usage = max(valid_usage)
                        min_usage = min(valid_usage)
                        _log[f"expert_min_usage_layer_{self.layer_idx}"] = (
                            min_usage / capacity
                        )
                        _log[f"expert_max_usage_layer_{self.layer_idx}"] = (
                            max_usage / capacity
                        )

                    global_training_logs = get_global_training_logs()
                    global_training_logs.update(**_log)

                recv_rank_task and recv_rank_task.wait()  # wait for recv_rank

                world_size = dist.get_world_size(self.config.moe_group)
                this_rank = dist.get_rank(self.config.moe_group)

                recv_size = paddle.count_nonzero(
                    recv_rank == dist.get_rank(self.config.moe_group)
                )
                recv_size = paddle.maximum(
                    recv_size, paddle.ones([], dtype=recv_size.dtype)
                )  # at least 1, make paddle happy
                if get_env_device() == "xpu":
                    recv_size_cpu = recv_size.to("cpu")
                    recv_size_task = None
                else:
                    recv_size_cpu, recv_size_task = async_offload(
                        recv_size, get_async_loader()
                    )

                send_rank_this_rank = paddle.count_nonzero(send_rank == this_rank)
                if get_env_device() == "xpu":
                    send_rank_this_rank_cpu = send_rank_this_rank.to("cpu")
                    send_rank_this_rank_task = None
                else:
                    send_rank_this_rank_cpu, send_rank_this_rank_task = async_offload(
                        send_rank_this_rank, get_async_loader()
                    )

                with profile("scatter_nd"):
                    recv_rank[recv_rank == -1] = world_size
                    send_recv_count_global = paddle.scatter_nd_add(
                        paddle.zeros(
                            [self.num_local_experts, world_size + 1, world_size + 1],
                            dtype="int32",
                        ),
                        paddle.stack([local_expert_id, send_rank, recv_rank], -1),
                        paddle.ones([len(send_rank)], dtype="int32"),
                    )  # [num_local_experts, world_size + 1 , world_size + 1]
                    send_counts_cpu = send_recv_count_global[:, this_rank, :-1].numpy()
                    recv_counts_cpu = send_recv_count_global[:, :-1, this_rank].numpy()
                    send_counts_num_cpu = send_counts_cpu.sum(-1)
                    recv_counts_num_cpu = recv_counts_cpu.sum(-1)

                # paddle.save(dict(
                #     expert_num_global=expert_num_global,
                #     expert_out=expert_out.shape,
                #     local_scatter_index=local_scatter_index,
                #     local_expert_id=local_expert_id,
                #     send_rank_global=send_rank,
                #     recv_rank_global=recv_rank,
                #     num_local_experts=self.num_local_experts,
                #     capacity=expert_out.shape[1],
                # ), f'test.deubg.dump.{dist.get_rank()}')
                if self.use_expert_alltoall_overlap:
                    forward_func_dict = {
                        i: lambda x: ex(x.contiguous())
                        for i, ex in enumerate(self.experts)
                    }
                else:
                    forward_func_dict = None
                    dispatched_input = self.forward_experts(*dispatched_input)

                if recv_size_task is not None:
                    recv_size_task.cpu_wait()
                if send_rank_this_rank_task is not None:
                    send_rank_this_rank_task.cpu_wait()

                input_size = sum(
                    [len(i) if i is not None else 0 for i in dispatched_input]
                )
                if (
                    self.use_padding or input_size > 1
                ):  # input=1时可能对应空输入，暂不校验
                    assert send_rank_this_rank_cpu.item() == input_size, (
                        send_rank,
                        [len(i) if i is not None else 0 for i in dispatched_input],
                    )

                (
                    expert_out_to_combine,
                    router_loss2,
                    distributed_input_to_alltoall_out,
                ) = AlltoAllSmart.apply(
                    *dispatched_input,
                    router_loss,
                    gate_logits,
                    gate_prob,
                    gate_logits_mm,
                    gate_prob_mm,
                    local_combine_weights,
                    expert_num_global_no_token_drop,
                    token_type_ids,
                    dispatch_token_type_ids,
                    forward_func_dict=forward_func_dict,
                    router_loss_fn=self.calc_router_loss_and_logging,
                    local_expert_id=local_expert_id,
                    send_rank_global=send_rank,
                    recv_rank_global=recv_rank,
                    num_local_experts=self.num_local_experts,
                    capacity=dispatched_input[0].shape[1] if self.use_padding else None,
                    use_padding=self.use_padding,
                    expert_num_global=expert_num_global_list,
                    is_first_fwd=not framework._dygraph_tracer()._has_grad,
                    group=self.config.moe_group,
                    recv_size=recv_size_cpu,
                    send_counts=send_counts_cpu,
                    recv_counts=recv_counts_cpu,
                    send_counts_num=send_counts_num_cpu,
                    recv_counts_num=recv_counts_num_cpu,
                )
                # /origin input -> distributed input/ => /origin-input -> alltoall out -input/
                local_scatter_index = distributed_input_to_alltoall_out[
                    local_scatter_index
                ]
                local_scatter_index.stop_gradient = True
            # global -> local
            with profile("combine"):
                # debug
                # combined_output2 = combined_output.clone()
                combined_output = self.combine_expert_output(
                    expert_out_to_combine, local_combine_weights, local_scatter_index
                )
                # debug
                # combined_output = combined_output - combined_output2.detach() + combined_output2
                # combined_output = combined_output2
        with profile("dense-expert"):
            if use_dense_expert:
                dense_input = global_hidden_states[global_dense_expert_mask.squeeze(-1)]
                dense_out = self.dense_experts(dense_input)
                # dense_out_padded = paddle.zeros_like(global_hidden_states)
                # dense_out_padded[global_dense_expert_mask.squeeze(-1)] = dense_out
                dense_out_padded = paddle.scatter_nd(
                    paddle.where(global_dense_expert_mask.squeeze(-1)),
                    dense_out,
                    global_hidden_states.shape,
                )

                dense_out_padded = ScatterOp.apply(dense_out_padded)
                combined_output += dense_out_padded

        with profile("shared-expert"):
            if self.shared_experts is not None:
                # globa -> local
                if self.is_mp_moe:
                    shared_out = self.shared_experts(
                        global_hidden_states, use_comm=False
                    )
                else:
                    shared_out = self.shared_experts(input)
                combined_output += shared_out

        if orig_shape:
            combined_output = combined_output.reshape(
                orig_shape[:-1] + [combined_output.shape[-1]]
            )

        return combined_output, local_combine_weights, router_loss2, gate_logits
        # allgather

    def fused_gate_logits_process_fused(
        self, gate_logits_lm, gate_logits_mm, token_type_ids
    ):
        """process gatelogits w/ moe utils"""
        top_k = 1 if isinstance(self.gate, SinkHornGateFused) else self.k
        num_expert_per_rank_per_modality = (
            gate_logits_lm.shape[-1] // self.config.moe_world_size
        )
        group_size = gate_logits_lm.shape[-1] // top_k
        if self.group_experts:
            assert not self.use_correction_bias
            gate_logits_lm = gate_logits_lm.reshape(
                [gate_logits_lm.shape[0], top_k, -1]
            )
            prob_lm = self.gate.act(gate_logits_lm)
            prob_lm_ = prob_lm
            weight_lm, expert_id_lm = prob_lm_.topk(k=1, axis=-1)
            weight_lm = weight_lm.reshape([gate_logits_lm.shape[0], -1])
            group_size = gate_logits_lm.shape[-1]
            expert_id_lm = expert_id_lm.squeeze(-1)
        else:
            prob_lm = self.gate.act(gate_logits_lm)
            if self.use_correction_bias:
                prob_lm_ = (
                    prob_lm + self.moe_statics.e_score_correction_bias[0].detach()
                )
            else:
                prob_lm_ = prob_lm
            weight_lm, expert_id_lm = prob_lm_.topk(k=top_k, axis=-1)
        assert (expert_id_lm >= 0).all(), f"lm prob has nan: {prob_lm_}"

        if self.use_correction_bias:
            batch_idx = (
                paddle.arange(prob_lm_.shape[0]).unsqueeze(-1).expand_as(expert_id_lm)
            )
            weight_lm = prob_lm[batch_idx, expert_id_lm]  # use correct bias

        # num_expert_per_modality == 0 时只执行 group-expert expand，不执行 multimodal-expand
        expert_id_lm = moe_utils.expand_modality_expert_id(
            expert_id_lm,
            num_expert_per_modality=(
                num_expert_per_rank_per_modality
                if (token_type_ids is not None and gate_logits_mm is not None)
                else 0
            ),
            group_size=group_size,
            modality_offset=0,
            is_group_expert=self.group_experts,
        )
        expert_id_lm = expert_id_lm.reshape(weight_lm.shape)
        lm_weight_and_expert_id = paddle.concat(
            [weight_lm, expert_id_lm.astype("float32")], -1
        )
        if token_type_ids is None or gate_logits_mm is None:
            return (
                lm_weight_and_expert_id,
                prob_lm.reshape([prob_lm.shape[0], -1]),
                None,
            )

        prob_mm = self.gate.act(gate_logits_mm)
        if self.use_correction_bias:
            prob_mm_ = prob_mm + self.moe_statics.e_score_correction_bias[1].detach()
        else:
            prob_mm_ = prob_mm
        weight_mm, expert_id_mm = prob_mm_.topk(k=top_k, axis=-1)
        assert (expert_id_mm >= 0).all(), f"lm prob has nan: {prob_mm_}"

        if self.use_correction_bias:
            batch_idx = (
                paddle.arange(prob_lm_.shape[0]).unsqueeze(-1).expand_as(expert_id_lm)
            )
            weight_mm = prob_mm[batch_idx, expert_id_mm]  # use correct bias

        expert_id_mm = moe_utils.expand_modality_expert_id(
            expert_id_mm,
            num_expert_per_modality=num_expert_per_rank_per_modality,
            group_size=group_size,
            modality_offset=1,
            is_group_expert=False,
        )
        expert_id_mm = expert_id_mm.reshape(weight_mm.shape)
        mm_weight_and_expert_id = paddle.concat(
            [weight_mm, expert_id_mm.astype("float32")], -1
        )
        weight_and_expert = paddle.where(
            (token_type_ids == 0).unsqueeze(-1),
            lm_weight_and_expert_id,
            mm_weight_and_expert_id,
        )
        return weight_and_expert, prob_lm.reshape([prob_lm.shape[0], -1]), prob_mm

    def fused_gate_and_dispatch(self, input, token_type_ids, global_dense_expert_mask):
        """_summary_

        Args:
            input (_type_): _description_
            token_type_ids (_type_): _description_
            global_dense_expert_mask: used in audio experts

        Returns:
            _type_: _description_
        """
        seqlen, d_model = input.shape
        args = ()
        # 目前只有 `SinkHornGate` aka Top1 gate 支持输入 token type ids
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape([-1])
            args = (token_type_ids,)

        router_loss = paddle.zeros([1], dtype="float32")
        router_loss.stop_gradient = False
        top_k = 1 if isinstance(self.gate, SinkHornGateFused) else self.k

        def build_weights_and_expert_id(input):
            nonlocal token_type_ids, args
            logits, capacity, router_loss = self.gate(
                input, *args, transform_weight=False
            )
            if self.config.multimodel_experts:
                gate_logits_lm, gate_logits_mm = logits.chunk(2, axis=-1)
            else:
                gate_logits_lm, gate_logits_mm = logits, None

            weigth_and_expert, gate_prob_lm, gate_prob_mm = (
                self.fused_gate_logits_process_fused(
                    gate_logits_lm,
                    gate_logits_mm,
                    (
                        token_type_ids if global_dense_expert_mask is None else None
                    ),  # HARD code: assume dense-expert == 纯音
                )
            )
            weigth_and_expert = AllGatherOp.apply(
                weigth_and_expert, group=self.config.moe_group
            )
            return (
                weigth_and_expert,
                gate_logits_lm,
                gate_logits_mm,
                gate_prob_lm,
                gate_prob_mm,
            )

        capacity = self.gate.get_capacity(input.shape[0] * self.world_size)
        # global_hidden_states = AllGatherOp.apply(input, group=self.config.moe_group)
        expert_input = input
        if self.config.moe_multimodal_paired_experts:
            assert token_type_ids is not None
            expert_input = paddle.concat(
                [expert_input, token_type_ids.unsqueeze(-1).astype(expert_input.dtype)],
                axis=-1,
            )
        (
            global_hidden_states,
            combine_weights_and_expert_id,
            gate_logits_lm,
            gate_logits_mm,
            gate_prob_lm,
            gate_prob_mm,
        ) = AllGatherAsync.apply(
            expert_input,
            input,
            fn=build_weights_and_expert_id,
            group=self.config.moe_group,
            is_first_fwd=not framework._dygraph_tracer()._has_grad,
        )
        combine_weights_unnorm, expert_id = combine_weights_and_expert_id.chunk(
            2, axis=-1
        )
        expert_id = expert_id.cast("int32")
        expert_id.stop_gradient = True
        num_experts = (
            sum(self.config.moe_num_experts)
            if isinstance(self.config.moe_num_experts, (tuple, list))
            else self.config.moe_num_experts
        )  # all-experts = 96
        if global_dense_expert_mask is not None:
            combine_weights_unnorm[global_dense_expert_mask] = 0.0
            expert_id[global_dense_expert_mask] = num_experts
            num_experts += 1

        with profile("dispatch_op"):
            if (
                "reverse_token_drop"
                in inspect.signature(
                    moe_ops_partial_nosoftmaxtopk.moe_gate_dispatch_partial_nosoftmaxtopk
                ).parameters
            ):
                compat_kwargs = {"reverse_token_drop": self.enable_reverse_token_drop}
            else:
                compat_kwargs = {}

            (
                dispatched_input,
                combine_weights_unnorm,
                scatter_index,  # input -> dispatched_input
                scatter_index_rev,  # dispatch-input -> input
                expert_num_global,  # global 不考虑截断！！
                expert_num_local,
            ) = moe_ops_partial_nosoftmaxtopk.moe_gate_dispatch_partial_nosoftmaxtopk(
                global_hidden_states,
                combine_weights_unnorm,
                expert_id,
                top_k,
                capacity,
                num_experts,
                self.use_padding,
                expert_start_index=self.num_local_experts * self.config.moe_rank,
                expert_end_index=self.num_local_experts * (self.config.moe_rank + 1),
                **compat_kwargs,
            )

        if self.use_correction_bias:
            if self.gate.config.multimodel_experts:
                for i in range(len(self.moe_statics.expert_usage)):
                    self.moe_statics.expert_usage[i] += expert_num_local[
                        self.gate.experts_type_mask[i]
                    ].detach()
            else:
                self.moe_statics.expert_usage[0] += expert_num_local.detach()

        if (
            scatter_index_rev.ndim == 0
        ):  # unpad 时, `moe_ops_partial` 中的空输出用 `scatter_index_rev==[]`表示。
            assert not self.use_padding
            scatter_index_rev = paddle.empty([0], dtype=scatter_index_rev.dtype)

        dispatched_input.stop_gradient = False
        combine_weights_unnorm.stop_gradient = False
        scatter_index.stop_gradient = True
        expert_num_global.stop_gradient = True
        expert_num_global_notrunc = expert_num_global
        self.capacity_tensor = paddle.to_tensor(capacity, dtype=expert_num_global.dtype)
        expert_num_global = paddle.minimum(expert_num_global, self.capacity_tensor)

        if global_dense_expert_mask is not None:  # 去掉dense expert
            expert_num_global = expert_num_global[:-1]
            expert_num_local = expert_num_local[:-1]
            expert_num_global_notrunc = expert_num_global_notrunc[:-1]

        scatter_index = scatter_index.transpose([1, 0])  # [k,s] ->[s,k]

        last_local_expert = self.num_local_experts * self.config.moe_rank
        expert_offset_global = expert_num_global.cumsum()
        if get_env_device() == "xpu":
            expert_num_global_list = expert_num_global.to("cpu")
            offload_task = None
        else:
            loader = get_async_loader()
            expert_num_global_list, offload_task = async_offload(
                expert_num_global, loader
            )
        if self.use_padding:
            offset = last_local_expert * capacity
        else:
            offset = (
                expert_offset_global[last_local_expert - 1]
                if self.config.moe_rank > 0
                else 0
            )
        local_combine_weights_unnorm = ReshardCombineWeight.apply(
            combine_weights_unnorm.contiguous(), group=self.config.moe_group
        )
        local_scatter_index = ReduceScatterOp.apply(
            paddle.where(
                combine_weights_unnorm > 0.0,
                scatter_index + offset,
                scatter_index,
            ),
            group=self.config.moe_group,
        )
        if self.gate.norm_gate_logits:
            local_combine_weights = local_combine_weights_unnorm / paddle.clip(
                local_combine_weights_unnorm.sum(-1, keepdim=True), min=1e-12
            )
        else:
            local_combine_weights = local_combine_weights_unnorm
        local_combine_weights = local_combine_weights.cast(dispatched_input.dtype)
        if self.use_padding:
            dispatched_input = dispatched_input.reshape(
                [
                    self.num_local_experts,
                    -1,
                    (
                        d_model
                        if not self.config.moe_multimodal_paired_experts
                        else d_model + 1
                    ),
                ]
            )
            dispatched_input = dispatched_input.unbind(0)
        else:
            s = self.num_local_experts * self.config.moe_rank
            e = self.num_local_experts * (self.config.moe_rank + 1)
            expert_num_local = expert_num_local.tolist()[s:e]
            expert_num_local_valid = [i for i in expert_num_local if i > 0]
            valid_pos = [j for j, i in enumerate(expert_num_local) if i > 0]
            if expert_num_local_valid:
                dispatched_input_list = dispatched_input.split(expert_num_local_valid)
                dispatched_input = [None] * len(expert_num_local)
                for p, t in zip(valid_pos, dispatched_input_list):
                    dispatched_input[p] = t
            else:
                dispatched_input = [dispatched_input] + (
                    [None] * (len(expert_num_local) - 1)
                )

        scatter_index.stop_gradient = True
        scatter_index_rev.stop_gradient = True
        if offload_task is not None:
            hack_offload_wait(offload_task)
        expert_num_global_list = expert_num_global_list.tolist()

        # logger.info(f'global-expert-usage:{expert_len}')
        # logger.info(f'scatter_index_rev: {self.num_local_experts} {capacity}')
        if self.config.moe_multimodal_paired_experts:
            global_hidden_states, _ = paddle.split(
                global_hidden_states, [d_model, 1], axis=-1
            )
        return (
            dispatched_input,
            global_hidden_states,
            local_combine_weights,
            expert_num_global_notrunc,  # 不考虑截断，为了计算auxloss
            expert_num_global,
            expert_num_global_list,
            local_scatter_index,
            scatter_index_rev,
            router_loss,
            (gate_logits_lm, gate_prob_lm),
            (gate_logits_mm, gate_prob_mm),
            expert_num_local,
        )

    def forward_experts(self, *dispatched_input):
        """
        call experts sequently
        forward_experts: List of Tensors, len equal to `num_local_experts`
        """
        with profile("fwd-expert"):
            expert_outputs = []
            assert isinstance(self.experts, nn.LayerList), type(self.experts)
            assert len(dispatched_input) == len(self.experts), (
                len(dispatched_input),
                len(self.experts),
            )

            no_tokens_expert_outputs = []
            for iexpert, chunk in enumerate(dispatched_input):
                if chunk is None:
                    if self.config.moe_layer_feed_fake_token:
                        chunk = paddle.zeros(
                            [1, self.experts[iexpert].up_gate_proj.weight.shape[0]],
                            dtype=self.experts[iexpert].up_gate_proj.weight.dtype,
                        )
                        if self.experts[iexpert].training:
                            chunk.stop_gradient = False
                        logger.warning(
                            f"local-expert: {iexpert} does not process data, we give a zero input to expert"
                        )
                        expert_out = self.experts[iexpert](chunk.contiguous())
                        no_tokens_expert_outputs.append(
                            expert_out * 0.0
                        )  # mutiply 0.0 to zero out and grad
                    expert_outputs.append(None)
                    continue

                expert_out = self.experts[iexpert](chunk.contiguous())
                expert_outputs.append(expert_out)
                # logger.info(
                #     f"moe-fwd-expert allgather: {chunk.shape}"
                #     f'-> {expert_outputs[-1].shape}: {chunk.astype("float32").norm(axis=-1)}'
                # )

            if (
                self.config.moe_layer_feed_fake_token
                and len(no_tokens_expert_outputs) > 0
            ):
                first_has_tokens_idx = 0
                for idx, expert_out in enumerate(expert_outputs):
                    if expert_out is not None:
                        first_has_tokens_idx = idx
                        break
                for idx, expert_out in enumerate(no_tokens_expert_outputs):
                    expert_outputs[first_has_tokens_idx] += expert_out

            return expert_outputs

    def calc_router_loss_and_logging(
        self,
        router_loss,
        gate_logits,
        gate_prob,
        gate_logits_mm,
        gate_prob_mm,
        combine_weights,
        dispatch_mask,
        token_type_ids,
        dispatch_token_type_ids,
    ):
        """
        分不同模态 gate prob(mm/lm) 进行 aux_loss 计算。
        """
        dispatch_mask_3d = dispatch_mask.reshape([self.config.moe_world_size, -1])
        if token_type_ids is not None and self.gate.config.moe_use_hard_gate:
            if not self.gate.weight.stop_gradient:
                # 文参数训练时才计算。
                dispatch_tokens_mask = (
                    dispatch_token_type_ids == 0
                    if dispatch_token_type_ids is not None
                    else None
                )
                lm_tokens_mask = (token_type_ids == 0).astype(gate_prob.dtype)
                # hard code
                lm_experts = (
                    self.gate.num_experts[0]
                    if isinstance(self.gate.num_experts, (tuple, list))
                    else self.gate.num_experts
                )
                dispatch_mask_lm = dispatch_mask_3d[
                    :, : lm_experts // self.config.moe_world_size
                ].reshape([-1])
                router_loss += self._calc_router_loss(
                    dispatch_mask_lm,
                    gate_logits * lm_tokens_mask.unsqueeze(-1),
                    gate_prob * lm_tokens_mask.unsqueeze(-1),
                    self.gate.num_experts_list[0],
                    self.group_experts,
                    self.layer_idx,
                    0,  # ortholoss
                    lm_tokens_mask,
                    dispatch_tokens_mask,
                    prefix="lm",
                )
            else:
                router_loss += self.zero * gate_logits[0, 0] * gate_prob[0, 0]
            if gate_prob_mm is not None:
                mm_tokens_mask = (token_type_ids == 1).astype(gate_prob_mm.dtype)
                dispatch_tokens_mask = (
                    dispatch_token_type_ids == 1
                    if dispatch_token_type_ids is not None
                    else None
                )
                dispatch_mask_mm = dispatch_mask_3d[
                    :, self.gate.num_experts[0] // self.config.moe_world_size :
                ].reshape([-1])

                router_loss += self._calc_router_loss(
                    dispatch_mask_mm,
                    gate_logits_mm * mm_tokens_mask.unsqueeze(-1),
                    gate_prob_mm * mm_tokens_mask.unsqueeze(-1),
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
                0,
                paddle.ones([gate_prob.shape[0]], "bool"),  # 半gate prob
                paddle.ones(
                    [self.gate.config.moe_world_size * gate_prob.shape[0]], "bool"
                ),
                prefix="lm",
            )

        tracer = framework._dygraph_tracer()
        global_training_logs = get_global_training_logs()
        if self.enable_logging and global_training_logs_enabled() and tracer._has_grad:
            if moe_router_loss_ops is not None and get_env_device() != "xpu":
                (
                    gate_expert_per_token_type_0,
                    gate_expert_per_token_type_1,
                    gate_experts_per_token,
                    ce,
                ) = fuse_logging(gate_logits, combine_weights, token_type_ids)

                if token_type_ids is not None:
                    global_training_logs.update(
                        experts_per_token_text=gate_expert_per_token_type_0,
                    )
                    global_training_logs.update(
                        experts_per_token_image=gate_expert_per_token_type_1,
                    )

            else:
                seqlen = gate_logits.shape[0]
                num_active = paddle.count_nonzero(combine_weights)
                gate_experts_per_token = num_active / seqlen
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.reshape([-1])
                    combine_weights_type_0 = combine_weights[token_type_ids == 0]
                    if combine_weights_type_0.size:
                        gate_expert_per_token_type_0 = (
                            paddle.count_nonzero(combine_weights_type_0)
                            / combine_weights_type_0.shape[0]
                        )
                        global_training_logs.update(
                            experts_per_token_text=gate_expert_per_token_type_0,
                        )

                    combine_weights_type_1 = combine_weights[token_type_ids == 1]
                    if combine_weights_type_1.size:
                        gate_expert_per_token_type_1 = (
                            paddle.count_nonzero(combine_weights_type_1)
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
                f"gate_prob_ce_layer_{self.layer_idx}": ce,
                f"experts_per_token_layer_{self.layer_idx}": gate_experts_per_token,
            }
            global_training_logs.update(
                **_log,
                **{
                    k.replace(f"_layer_{self.layer_idx}", ""): v
                    for k, v in _log.items()
                },
            )
        return router_loss
