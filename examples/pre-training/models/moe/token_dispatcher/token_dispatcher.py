# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 DeepSeek
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
"""Flex token dispatcher using DeepEP"""

from typing import Optional, Tuple

import paddle
from paddle.distributed.communication.group import Group

from .fused_a2a import fused_combine, fused_dispatch
from .moe_utils import permute, topk_to_permuted_indices, unpermute


class _DeepepManager:
    """
    A manager class to handle fused all-to-all communication processes for MoE models using
    DeepEP backend. See https://github.com/deepseek-ai/deepep for more details.

    The workflow of the DeepEP dispatcher is:
    (1) dispatch():
        - Use fused kernel to permute tokens and perform all-to-all communication in single step
    (2) get_permuted_hidden_states_by_instances():
        - Convert routing map and probabilities to multihot format
        - Permute tokens using fused kernel
    (3) get_restored_hidden_states_by_instances():
        - Reverse permutation using fused kernel
    (4) combine():
        - Reverse process using fused kernel to unpermute and perform all-to-all in single step

    This implementation uses fused communication kernels (fused_dispatch/fused_combine) that
    combine permutation and communication operations for improved efficiency compared to
    separate permute+alltoall steps.
    """

    def __init__(
        self,
        group: Group,
        num_experts: int = None,
        num_local_experts: int = None,
    ):
        self.group = group
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts

        self.handle = None

    def dispatch(
        self,
        hidden_states,
        token_indices,
        token_probs,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
        fp8_dispatch_a2a=False,
        inner_layer_overlap_handle=None,
    ) -> paddle.Tensor:
        """
        call deepep dispatch, dispatch the tokens to experts
        """
        if fused_dispatch is None:
            raise ImportError(
                "DeepEP is not supported in your paddlepaddle whl package."
            )

        hidden_states, dispatched_probs, states, fp8_dispatched_handle = fused_dispatch(
            hidden_states,
            token_indices,
            token_probs,
            self.num_experts,
            self.group,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            fp8_dispatch_a2a=fp8_dispatch_a2a,
            inner_layer_overlap_handle=inner_layer_overlap_handle,
        )
        self.handle = states["handle"]
        self.tokens_per_expert_list = states["tokens_per_expert"]
        dispatched_indices = states["dispatched_indices"]

        return (
            hidden_states,
            dispatched_indices,
            dispatched_probs,
            fp8_dispatched_handle,
        )

    def _indices_to_multihot(self, indices, probs):
        """
        Converts a tensor of indices to a multihot vector.

        Args:
            indices (paddle.Tensor): [num_tokens, topk] token indices, where -1 means masked out.
            probs (paddle.Tensor): [num_tokens, topk] token probabilities.

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]:
                - routing_map: Multihot vector.
                - probs: Multihot probabilities.
        """
        batch_size = indices.shape[0]
        multihot_routing_map = paddle.zeros(
            (batch_size, self.num_local_experts), dtype=paddle.int64
        )

        multihot_probs = paddle.zeros(
            (batch_size, self.num_local_experts), dtype=paddle.float32
        )

        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = paddle.arange(batch_size).repeat_interleave(mask.sum(axis=1))
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.cast(paddle.bool), multihot_probs

    def combine(
        self,
        hidden_states,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
        inner_layer_overlap_handle=None,
    ) -> paddle.Tensor:
        """
        call deepep combine, combine the outputs from experts
        """
        hidden_states = fused_combine(
            hidden_states,
            self.group,
            self.handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            inner_layer_overlap_handle=inner_layer_overlap_handle,
        )
        return hidden_states

    def get_permuted_hidden_states_by_experts(
        self,
        hidden_states: paddle.Tensor,
        dispatched_indices: paddle.Tensor,
        moe_k: int,
    ) -> paddle.Tensor:
        """
        permute the tokens to the experts on current rank
        """
        self.hidden_shape_before_permute = hidden_states.shape
        token_permuted_indices, prob_permuted_indices = topk_to_permuted_indices(
            dispatched_indices, self.tokens_per_expert_list, moe_k
        )
        hidden_states = permute(hidden_states, token_permuted_indices)
        return hidden_states, token_permuted_indices, prob_permuted_indices

    def get_restored_hidden_states_by_experts(
        self,
        hidden_states: paddle.Tensor,
        token_permuted_indices: paddle.Tensor,
        prob_permuted_indices: paddle.Tensor,
        dispatched_probs: paddle.Tensor,
        deepep_use_fused: bool = False,
    ) -> paddle.Tensor:
        """
        combine the outputs from experts on current rank
        """
        input_dtype = hidden_states.dtype
        assert (
            dispatched_probs.dtype == paddle.float32
        ), "DeepEP only supports float32 probs"
        hidden_states = unpermute(
            permuted_tokens=hidden_states,
            token_permuted_indices=token_permuted_indices,
            prob_permuted_indices=prob_permuted_indices,
            restore_shape=self.hidden_shape_before_permute,
            probs=dispatched_probs,
            deepep_use_fused=deepep_use_fused,
        )
        return hidden_states.to(input_dtype)


class MoEFlexTokenDispatcher:
    """
    Flexible token dispatcher for MoE models with Efficient-A2A communication kernels.
    """

    def __init__(self, num_local_experts: int, num_moe_experts: int, ep_group: Group):
        self._ep_group = ep_group

        self.num_local_experts = num_local_experts
        assert self.ep_size > 1, "Flex token dispatcher requires EP > 1"
        self._comm_manager = _DeepepManager(
            group=self.ep_group,
            num_experts=num_moe_experts,
            num_local_experts=self.num_local_experts,
        )

    @property
    def ep_group(self):
        """Get expert model parallel group."""
        return self._ep_group

    @property
    def ep_size(self):
        """Get expert model parallel world_size."""
        return self.ep_group.world_size

    def cal_final_topk(self, probs, moe_k, num_tokens):
        """
        pre-process of dispatch
        """
        probs = probs.reshape([num_tokens, self._comm_manager.num_experts])
        # Convert the format of routing map from multihot to indices.
        token_probs, token_indices = paddle.topk(probs, moe_k, axis=-1)
        return token_indices, token_probs

    def token_permutation(
        self,
        hidden_states,
        token_indices,
        token_probs,
        moe_k,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
        inner_layer_overlap_handle=None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        dispatch tokens to experts
        """
        hidden_states, dispatched_indices, dispatched_probs, _ = (
            self._comm_manager.dispatch(
                hidden_states,
                token_indices,
                token_probs,
                previous_event=previous_event,
                async_finish=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
                inner_layer_overlap_handle=inner_layer_overlap_handle,
            )
        )
        (
            global_input_tokens,
            token_permuted_indices,
            prob_permuted_indices,
        ) = self._comm_manager.get_permuted_hidden_states_by_experts(
            hidden_states, dispatched_indices, moe_k
        )

        return (
            global_input_tokens,
            token_permuted_indices,
            prob_permuted_indices,
            dispatched_probs,
        )

    def token_unpermutation(
        self,
        hidden_states,
        token_permuted_indices,
        prob_permuted_indices,
        dispatched_probs,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
        deepep_use_fused=False,
        inner_layer_overlap_handle=None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
        """
        combine outputs from experts
        """
        hidden_states = self._comm_manager.get_restored_hidden_states_by_experts(
            hidden_states,
            token_permuted_indices,
            prob_permuted_indices,
            dispatched_probs,
            deepep_use_fused=deepep_use_fused,
        )
        hidden_states = self._comm_manager.combine(
            hidden_states,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            inner_layer_overlap_handle=inner_layer_overlap_handle,
        )
        return hidden_states
