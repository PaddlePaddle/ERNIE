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

import warnings
import inspect
import numpy as np
import paddle
from paddle import framework

try:
    from paddleformers.trainer.utils.offload_optimizer import reload
except ImportError:
    reload = None

try:
    import TokenDispatcherUtils as TDU
except ImportError:
    TDU = None


try:
    import FusedQuantOps as FQO
except ImportError:
    FQO = None


from .fp8_utils import FP8_ALIGN


if not hasattr(paddle.Tensor, "_clear_to_zero_allocation"):

    def _clear_to_zero_allocation(self):
        """
        _clear_to_zero_allocation
        """
        old_shape = self.shape
        dst = paddle.empty([0], dtype=self.dtype)
        dst_t = dst.value().get_tensor()
        src_t = self.value().get_tensor()
        src_t._share_data_with(dst_t)
        src_t._set_dims(old_shape)

    setattr(paddle.Tensor, "_clear_to_zero_allocation", _clear_to_zero_allocation)


if not hasattr(paddle.Tensor, "_holder_size"):

    def _holder_size(self):
        """
        _holder_size
        """
        if self._is_initialized():
            return int(np.prod(self.shape)) * paddle.core.size_of_dtype(self.dtype)
        else:
            return 0

    setattr(paddle.Tensor, "_holder_size", _holder_size)


def has_argument(method, name):
    """
    has_argument
    """
    return name in inspect.getfullargspec(method).args


if TDU is not None:
    if not has_argument(TDU.tokens_unzip_stable, "fill_output"):
        origin_tokens_unzip_stable = TDU.tokens_unzip_stable

        def new_tokens_unzip_stable(
            x,
            x_scale,
            expert_routemap_topk,
            expert_prob_topk,
            topk,
            num_experts,
            tokens_per_expert,
            padding_multiplex,
            fill_output=True,
        ):
            """
            new_tokens_unzip_stable
            """
            assert fill_output, "fill_output should be True"
            return origin_tokens_unzip_stable(
                x,
                x_scale,
                expert_routemap_topk,
                expert_prob_topk,
                topk,
                num_experts,
                tokens_per_expert,
                padding_multiplex,
            )

        setattr(TDU, "tokens_unzip_stable", new_tokens_unzip_stable)


if FQO is not None:
    if hasattr(FQO, "fused_swiglu_probs_bwd") and (
        not has_argument(FQO.fused_swiglu_probs_bwd, "inplace")
    ):
        origin_fused_swiglu_probs_bwd = FQO.fused_swiglu_probs_bwd

        def new_fused_swiglu_probs_bwd(o1, do2_s, unzipped_probs, inplace=False):
            """
            new_fused_swiglu_probs_bwd
            """
            return origin_fused_swiglu_probs_bwd(o1, do2_s, unzipped_probs)

        setattr(FQO, "fused_swiglu_probs_bwd", new_fused_swiglu_probs_bwd)


def tokens_zip_unique_add_with_subbatch(
    zipped, unzipped, index_unzipped, zipped_rows, subbatch_rows=None
):
    """
    tokens_zip_unique_add_with_subbatch
    """
    if subbatch_rows is None or subbatch_rows <= 0 or zipped_rows <= 0:
        return TDU.tokens_zip_unique_add(zipped, unzipped, index_unzipped, zipped_rows)
    else:
        if isinstance(zipped, paddle.Tensor):
            num_split = (zipped_rows + subbatch_rows - 1) // subbatch_rows
            remainder = zipped_rows % subbatch_rows
            if remainder == 0:
                rows = [subbatch_rows] * num_split
            else:
                rows = [subbatch_rows] * (num_split - 1) + [remainder]

            if zipped.shape[0] == 0:
                dtype = zipped.dtype
                hidden_size = zipped.shape[1]
                zipped = [paddle.zeros([r, hidden_size], dtype=dtype) for r in rows]
            else:
                zipped = paddle.split(zipped, rows, axis=0)
        return TDU.tokens_zip_unique_add_subbatch(
            zipped, unzipped, index_unzipped, zipped_rows, subbatch_rows
        )


def merge_subbatch_cast(x, dtype):
    """
    merge_subbatch_cast
    """
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            x = x[0]
            return x.cast(dtype) if x.dtype != dtype else x
        else:
            return TDU.merge_subbatch_cast(x, dtype)
    else:
        return x.cast(dtype) if x.dtype != dtype else x


def get_training_step():
    """
    get_training_step
    """
    try:
        from ernie4.src.utils.misc import global_training_logs as gtl
    except ImportError:
        gtl = None

    if not isinstance(gtl, dict) and gtl.trainer is not None:
        return gtl.trainer.state.global_step + 1
    else:
        return None


def inplace_reload(x):
    """
    inplace_reload
    """
    reload(x)
    return x


def inplace_offload(x, use_pinned=False):
    """
    inplace offload
    """
    place = paddle.CUDAPinnedPlace() if use_pinned else paddle.CPUPlace()
    if not x.place._equals(place):
        y = x.pin_memory() if use_pinned else x.cpu()
        if y is not x:
            x_t = x.value().get_tensor()
            y_t = y.value().get_tensor()
            x_t._share_data_with(y_t)


def inplace_offload_if_needed(x, threshold=2 * 1024 * 1024 * 1024):
    """Conditionally offload tensor to CPU if it exceeds memory threshold.

    Args:
        x (paddle.Tensor): The tensor to potentially offload.
        threshold (int, optional): Memory threshold in bytes. Defaults to 2GB.

    Note:
        Only offloads tensors during gradient computation when memory usage exceeds threshold.
        Issues a warning when offloading occurs.
    """
    if not framework._dygraph_tracer()._has_grad:
        return

    memory_size = np.prod(x.shape) * paddle.core.size_of_dtype(x.dtype)
    if memory_size >= threshold:
        inplace_offload(x)
        warnings.warn(
            f"Offload tensor with shape: {x.shape}, dtype: {x.dtype}, memory size {memory_size}"
        )


def topk_to_permuted_indices_single(x, num_tokens, expert_id, topk):
    """Convert topk indices to permuted indices for a single expert.

    Args:
        x (paddle.Tensor): Input tensor containing expert assignments.
        num_tokens (int): Number of tokens assigned to this expert.
        expert_id (int): ID of the expert to filter for.
        topk (int): Number of experts selected per token (top-k value).

    Returns:
        tuple: (token_permuted_indices, prob_permuted_indices)
            - token_permuted_indices: Indices of tokens assigned to this expert
            - prob_permuted_indices: Indices of probabilities for the expert assignments
    """
    x = paddle.flatten(x)
    prob_permuted_indices = paddle.tensor.search._restrict_nonzero(
        x == expert_id, num_tokens
    ).flatten()
    token_permuted_indices = prob_permuted_indices // topk
    return token_permuted_indices, prob_permuted_indices


def topk_to_permuted_indices(x, num_tokens_per_expert_list, topk):
    """Convert topk indices to permuted indices for all experts.

    Args:
        x (paddle.Tensor): Input tensor containing expert assignments.
        num_tokens_per_expert_list (list[int]): List of token counts per expert.
        topk (int): Number of experts selected per token (top-k value).

    Returns:
        tuple: (token_permuted_indices, prob_permuted_indices)
            - token_permuted_indices: Indices of tokens assigned to experts
            - prob_permuted_indices: Indices of probabilities for all expert assignments
    """
    x = paddle.flatten(x)
    prob_permuted_indices = paddle.concat(
        [
            paddle.tensor.search._restrict_nonzero(x == i, total_true_num)
            for i, total_true_num in enumerate(num_tokens_per_expert_list)
        ]
    ).flatten()
    token_permuted_indices = prob_permuted_indices // topk
    return token_permuted_indices, prob_permuted_indices


def permute(
    tokens,
    token_permuted_indices,
    drop_and_pad: bool = False,
):
    """Permute tokens based on expert assignment indices.

    Args:
        tokens (paddle.Tensor): Input tokens to be permuted.
        token_permuted_indices (paddle.Tensor): Indices for permutation.
        drop_and_pad (bool, optional): Whether to drop and pad tokens. Not supported yet.

    Returns:
        paddle.Tensor: Permuted tokens.

    Raises:
        AssertionError: If drop_and_pad is True (not supported).
    """
    assert not drop_and_pad, "token-drop and pads is not supported"
    permuted_input = paddle.gather(tokens, token_permuted_indices)
    return permuted_input


def unpermute(
    permuted_tokens: paddle.Tensor,
    token_permuted_indices: paddle.Tensor,
    prob_permuted_indices: paddle.Tensor,
    restore_shape: paddle.shape,
    probs: paddle.Tensor = None,
    drop_and_pad: bool = False,
):
    """Restore original token order from permuted tokens.

    Args:
        permuted_tokens (paddle.Tensor): Permuted tokens to be restored.
        token_permuted_indices (paddle.Tensor): Original token positions.
        prob_permuted_indices (paddle.Tensor): Indices for probability values.
        restore_shape (paddle.shape): Original shape of the tensor.
        probs (paddle.Tensor, optional): Probability values for weighted restoration.
        drop_and_pad (bool, optional): Whether to drop and pad tokens. Not supported yet.

    Returns:
        paddle.Tensor: Restored tokens in original order.

    Raises:
        AssertionError: If drop_and_pad is True (not supported).
    """
    assert not drop_and_pad, "token-drop and pads is not supported"
    _, hidden = restore_shape
    if probs is not None:
        permuted_probs = paddle.gather(probs.flatten(), prob_permuted_indices)
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    output_tokens = paddle.zeros(restore_shape, dtype=permuted_tokens.dtype)
    output_tokens.scatter_(
        index=token_permuted_indices, updates=permuted_tokens, overwrite=False
    )
    return output_tokens


class UnZipNode:
    """Handles the unzipping (high performance permute) of tokens for expert processing in Mixture of Experts,
    in an efficient, deterministic manner.

    This class manages the process of expanding tokens assigned to experts, including:
    - Forward pass: Distributes tokens to experts
    - Backward pass: Collects gradients from experts

    Attributes:
        token_dispatcher: Reference to the parent token dispatcher.
        name (str): Identifier for this node.
        unzipped_probs (paddle.Tensor): Probability values after unzipping.
        zipped_expertwise_rowmap (paddle.Tensor): Mapping between original and expanded tokens.
    """

    def __init__(self, token_dispatcher, name="unzip"):
        """Initialize the UnZipNode.

        Args:
            token_dispatcher: Parent token dispatcher instance.
            name (str, optional): Name identifier. Defaults to "unzip".
        """
        self.token_dispatcher = token_dispatcher
        self.name = name
        self.unzipped_probs = None
        self.zipped_expertwise_rowmap = None

    def reset_status(self):
        """Reset internal state between forward/backward passes."""
        self.unzipped_probs = None
        self.zipped_expertwise_rowmap = None

    def cached_tensors(self):
        """
        cached_tensors
        """
        return [self.unzipped_probs, self.zipped_expertwise_rowmap]

    def set_cached_tensors(self, tensors):
        """
        set_cached_tensors
        """
        self.unzipped_probs, self.zipped_expertwise_rowmap = tensors

    def clear_cached_tensors(self):
        """
        clear_cached_tensors
        """
        self.set_cached_tensors([None] * len(self.cached_tensors()))

    @paddle.no_grad()
    def forward(
        self,
        hs_2d_dispatched,
        dispatched_indices,
        dispatched_probs,
        topk,
        num_experts,
        tokens_per_expert,
        fill_output=True,
    ):
        """Forward pass - distribute tokens to experts.

        Args:
            hs_2d_dispatched (paddle.Tensor): Dispatched hidden states (2D).
            dispatched_indices (paddle.Tensor): Indices of expert assignments.
            dispatched_probs (paddle.Tensor): Routing probabilities.
            topk (int): Number of experts selected per token.
            num_experts (int): Total number of experts.
            tokens_per_expert (int): Tokens allocated per expert.

        Returns:
            tuple: (unzipped_tokens, zipped_expertwise_rowmap, unzipped_probs)
                - unzipped_tokens: Expanded tokens for expert processing
                - zipped_expertwise_rowmap: Mapping between original and expanded tokens
                - unzipped_probs: Expanded routing probabilities
        """
        if isinstance(hs_2d_dispatched, tuple):
            assert (
                len(hs_2d_dispatched) == 2
            ), f"hs_2d_dispatched should has at most 2 tensors, but bot {len(hs_2d_dispatched)}"
            hidden_states, scale = hs_2d_dispatched
        else:
            hidden_states, scale = hs_2d_dispatched, None

        (
            unzipped_tokens,
            zipped_expertwise_rowmap,
            unzipped_probs,
            unzipped_scale,
        ) = TDU.tokens_unzip_stable(
            hidden_states,
            scale,
            dispatched_indices,
            dispatched_probs,
            topk=topk,
            num_experts=num_experts,
            tokens_per_expert=tokens_per_expert,
            padding_multiplex=FP8_ALIGN,
            fill_output=fill_output,
        )

        self.unzipped_probs = unzipped_probs
        self.zipped_expertwise_rowmap = zipped_expertwise_rowmap
        return (
            unzipped_tokens,
            zipped_expertwise_rowmap,
            unzipped_probs,
            unzipped_scale,
        )

    @paddle.no_grad()
    def backward(
        self, dx, hidden_states_out_grad, probs_grad, dispatched_indices, num_experts
    ):
        """Backward pass - collect gradients from experts.

        Args:
            dx (paddle.Tensor): Gradient from experts.
            hidden_states_out_grad (paddle.Tensor): Gradient of output hidden states.
            probs_grad (paddle.Tensor): Gradient of routing probabilities.
            dispatched_indices (paddle.Tensor): Original expert assignment indices.
            num_experts (int): Total number of experts.

        Returns:
            tuple: (weighted_zipped_tokens, probs_grad_zipped)
                - weighted_zipped_tokens: Compressed gradients from experts
                - probs_grad_zipped: Compressed probability gradients
        """
        with paddle.amp.auto_cast(False):
            weighted_zipped_tokens, probs_grad_zipped = (
                paddle.nn.functional.moe_unpermute(
                    dx,
                    self.zipped_expertwise_rowmap,
                    dispatched_indices,
                    probs_grad,
                    total_zipped_tokens=hidden_states_out_grad.shape[0],
                    num_experts=num_experts,
                )
            )
        self.reset_status()
        return weighted_zipped_tokens, probs_grad_zipped


class ZipNode:
    """Handles the zipping (high performance unpermute) of expert outputs in Mixture of Experts,
    in an efficient, deterministic manner.

    This class manages the process of combining expert outputs, including:
    - Forward pass: Combines expert outputs
    - Backward pass: Distributes gradients to experts

    Attributes:
        token_dispatcher: Reference to the parent token dispatcher.
        name (str): Identifier for this node.
    """

    def __init__(self, token_dispatcher, name="zip"):
        """Initialize the ZipNode.

        Args:
            token_dispatcher: Parent token dispatcher instance.
            name (str, optional): Name identifier. Defaults to "zip".
        """
        self.token_dispatcher = token_dispatcher
        self.name = name

    def cached_tensors(self):
        """
        cached_tensors
        """
        return []

    def set_cached_tensors(self, tensors):
        """
        set_cached_tensors
        """
        assert len(tensors) == 0

    def clear_cached_tensors(self):
        """
        clear_cached_tensors
        """
        pass

    @paddle.no_grad()
    def forward(
        self,
        expert_out,
        zipped_expertwise_rowmap,
        routemap_topk,
        unzipped_probs,
        total_zipped_tokens,
        num_experts,
    ):
        """Forward pass - combine expert outputs.

        Args:
            expert_out (paddle.Tensor): Outputs from all experts.
            zipped_expertwise_rowmap (paddle.Tensor): Mapping between original and expanded tokens.
            routemap_topk (paddle.Tensor): Top-k routing information.
            unzipped_probs (paddle.Tensor): Expanded routing probabilities.
            total_zipped_tokens (int): Total number of original tokens.
            num_experts (int): Total number of experts.

        Returns:
            paddle.Tensor: Combined expert outputs.
        """
        with paddle.amp.auto_cast(False):
            expert_out_zipped, zipped_probs_topk = paddle.nn.functional.moe_unpermute(
                expert_out,
                zipped_expertwise_rowmap,
                routemap_topk,
                unzipped_probs,
                total_zipped_tokens,
                num_experts,
            )
        return expert_out_zipped

    @paddle.no_grad()
    def backward(
        self,
        grad_output,
        dispatched_indices,
        dispatched_probs,
        top_k,
        num_experts,
        tokens_per_expert,
    ):
        """Backward pass - distribute gradients to experts.

        Args:
            grad_output (paddle.Tensor): Gradient of the combined output.
            dispatched_indices (paddle.Tensor): Original expert assignment indices.
            dispatched_probs (paddle.Tensor): Original routing probabilities.
            top_k (int): Number of experts selected per token.
            num_experts (int): Total number of experts.
            tokens_per_expert (int): Tokens allocated per expert.

        Returns:
            paddle.Tensor: Expanded gradients to be sent to experts.
        """
        with paddle.amp.auto_cast(False):
            (
                unzipped_grad,
                zipped_expertwise_rowmap_grad,
                unzipped_probs_grad,
                _,
            ) = paddle.nn.functional.moe_permute(
                grad_output,
                None,
                dispatched_indices,
                dispatched_probs,
                num_experts,
                tokens_per_expert,
                padding_alignment=128,
            )

        return unzipped_grad
