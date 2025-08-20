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

""" moe utils for allgather dispatcher """
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
import paddle.nn.functional as F
from paddle import nn
from paddle.autograd import PyLayer

from models.sequence_parallel_utils import (
    AllGatherOp,
    ReduceScatterOp,
)


class MOEGather(PyLayer):
    """
    MOE Gather
    """

    @staticmethod
    def forward(ctx, input_, map_):
        """
        MOE Gather forward
        """
        ctx.input_shape = input_.shape
        ctx.map = map_
        return paddle.take_along_axis(input_, map_, 0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        MOE Gather backward
        """
        input_shape = ctx.input_shape
        map_ = ctx.map

        output = paddle.zeros(input_shape, dtype=grad_output.dtype)
        return paddle.put_along_axis(output, map_, grad_output, 0), None


class MOEScatter(PyLayer):
    """
    MOE Scatter
    """

    @staticmethod
    def forward(ctx, input_, map_, output_size=None):
        """
        MOE Scatter forward
        """
        ctx.map = map_

        if output_size is not None:
            output = paddle.zeros(output_size, dtype=input_.dtype)
        else:
            output = paddle.zeros_like(input_)

        return paddle.put_along_axis(output, map_, input_, 0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        MOE Scatter backward
        """
        map_ = ctx.map
        return paddle.take_along_axis(grad_output, map_, 0), None


class AllgatherDispatcherReturn(object):
    """
    MOE allgather dispatcher return value
    """

    def __init__(
        self,
        global_hidden_states,
        dispatched_input,
        combine_weights,
        scatter_index,
        gather_scatter_mask,
        dispatch_mask,
        tokens_per_expert,
    ):
        self.global_hidden_states = global_hidden_states
        self.dispatched_input = dispatched_input
        self.combine_weights = combine_weights
        self.scatter_index = scatter_index
        self.gather_scatter_mask = gather_scatter_mask
        self.dispatch_mask = dispatch_mask
        self.tokens_per_expert = tokens_per_expert


class MOEAllGatherDispatcher(nn.Layer):
    """
    MOE with allgather dispatcher.
    Contains two static methos.
    MOEAllGatherDispatcher.token_dispatcher
    MOEAllGatherDispatcher.token_combine
    """

    @staticmethod
    def token_dispatcher(
        hidden_states,
        local_gate_logits,
        top_k,
        local_expert_indices,
        num_moe_experts,
        num_local_experts,
    ):
        """
        MOE token dispatcher with allgather
        """
        seq_len = local_gate_logits.shape[0]
        num_experts = local_gate_logits.shape[-1]
        prob = F.softmax(local_gate_logits.reshape([seq_len, top_k, -1]), axis=-1)
        max_prob = prob.max(-1, keepdim=True)
        prob /= max_prob
        prob = prob.reshape([-1, num_experts])

        probs, scatter_index = paddle.topk(prob, top_k, axis=-1)
        dispatch_mask = paddle.cumsum(
            paddle.histogram(scatter_index.flatten(), bins=num_experts)
        )

        # dispatch
        with paddle.no_grad():
            global_indices = AllGatherOp.apply(scatter_index)
            global_local_mask = (global_indices >= local_expert_indices[0]) & (
                global_indices <= local_expert_indices[-1]
            )
            local_indices = global_indices.masked_select(global_local_mask)

        global_hidden_states = AllGatherOp.apply(hidden_states)
        global_probs = AllGatherOp.apply(probs)

        # get local hidden states
        combine_weights = global_probs.masked_select(global_local_mask).cast(
            dtype=hidden_states.dtype
        )
        gather_scatter_mask = global_local_mask.nonzero()[:, 0]
        gather_scatter_mask = paddle.reshape(gather_scatter_mask, shape=[-1, 1])
        gather_scatter_mask = paddle.expand(
            gather_scatter_mask, shape=[-1, hidden_states.shape[-1]]
        )
        local_hidden_states = MOEGather.apply(global_hidden_states, gather_scatter_mask)

        with paddle.no_grad():
            # The indices of local_indices that give its sorted order along dim 0.
            scatter_index = paddle.argsort(local_indices, axis=0)
            tokens_per_expert = paddle.bincount(
                paddle.reshape(local_indices, [-1]), minlength=num_moe_experts
            )
            if num_local_experts < num_moe_experts:
                start = local_expert_indices[0]
                end = local_expert_indices[-1] + 1
                tokens_per_expert = tokens_per_expert[start:end]

        scatter_index = paddle.reshape(scatter_index, shape=[-1, 1])
        scatter_index = paddle.expand(
            scatter_index, shape=[-1, hidden_states.shape[-1]]
        )

        dispatched_input = MOEGather.apply(local_hidden_states, scatter_index)

        return AllgatherDispatcherReturn(
            global_hidden_states,
            dispatched_input,
            combine_weights,
            scatter_index,
            gather_scatter_mask,
            dispatch_mask,
            tokens_per_expert,
        )

    @staticmethod
    def token_combine(
        expert_out,
        shared_out,
        combine_weights,
        scatter_index,
        gather_scatter_mask,
        global_shape,
    ):
        """
        MOE token combine with reduce scatter
        """
        expert_out = MOEScatter.apply(expert_out, scatter_index)
        expert_out = expert_out * paddle.reshape(combine_weights, shape=[-1, 1])
        expert_out = MOEScatter.apply(expert_out, gather_scatter_mask, global_shape)
        combine_out = expert_out + shared_out
        combine_out = ReduceScatterOp.apply(combine_out)
        return combine_out


def get_flatten_mesh(mesh):

    return dist.ProcessMesh(mesh.process_ids)


def get_mesh(pp_idx=0):

    mesh = fleet.auto.get_mesh()
    if "pp" in mesh.dim_names:
        mesh = mesh.get_mesh_with_dim("pp", pp_idx)
    return mesh


def _reshard(tensor, mesh, placements):

    dst_tensor = dist.auto_parallel.moe_utils._dist_reshape(
        tensor, tensor.shape, mesh, placements
    )
    return dst_tensor
