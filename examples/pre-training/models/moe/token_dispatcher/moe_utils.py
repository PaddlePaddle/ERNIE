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
"""Token Dispatcher Utils"""

import inspect
import numpy as np
import paddle
from paddle import framework
from paddleformers.trainer.trainer import logger

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

try:
    from paddle import scatter_add_
except ImportError:
    scatter_add_ = None

from models.utils import (
    FusedUnpermutation,
)

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
    """
    inplace offload if needed
    """
    if not framework._dygraph_tracer()._has_grad:
        return

    memory_size = x._holder_size()
    if memory_size >= threshold:
        inplace_offload(x)
        step = get_training_step()
        logger.warning(
            "Offload tensor with step: {}, shape: {}, dtype: {}, memory size {}".format(
                step, x.shape, x.dtype, memory_size
            )
        )
        return True
    else:
        return False


def topk_to_permuted_indices_single(x, num_tokens, expert_id, topk):
    """
    Convert the topk indices to permuted indices.
    """
    x = paddle.flatten(x)
    prob_permuted_indices = paddle.tensor.search._restrict_nonzero(
        x == expert_id, num_tokens
    ).flatten()
    token_permuted_indices = prob_permuted_indices // topk
    return token_permuted_indices, prob_permuted_indices


def topk_to_permuted_indices(x, num_tokens_per_expert_list, topk):
    """
    Convert the topk indices to permuted indices.
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
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    Args:
        tokens (paddle.Tensor): The input token tensor, [num_tokens, hidden].
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.
    """
    assert not drop_and_pad, "token-drop and pads is not supported"
    permuted_input = paddle.gather(tokens, token_permuted_indices)
    # permuted_input = tokens.index_select(axis=0, index=token_permuted_indices)
    return permuted_input


def unpermute(
    permuted_tokens: paddle.Tensor,
    token_permuted_indices: paddle.Tensor,
    prob_permuted_indices: paddle.Tensor,
    restore_shape: paddle.shape,
    probs: paddle.Tensor = None,
    drop_and_pad: bool = False,
    deepep_use_fused: bool = False,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    Args:
        permuted_tokens (paddle.Tensor): The permuted token tensor.
        token_permuted_indices (paddle.Tensor): The indices used to sort the tokens.
        restore_shape (paddle.shape): The shape of the unpermuted tensor.
        probs (paddle.Tensor, optional): The unpermuted probs tensor,
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.

    Returns:
        paddle.Tensor: The tokens restored to their original order.
    """
    assert not drop_and_pad, "token-drop and pads is not supported"
    _, hidden = restore_shape

    if deepep_use_fused and probs is not None:
        # Create an output tensor filled with zeros
        output_tokens = paddle.zeros(restore_shape, dtype=probs.dtype)

        output_tokens = FusedUnpermutation.apply(
            output_tokens,
            permuted_tokens,
            token_permuted_indices,
            probs.flatten(),
            prob_permuted_indices,
        )
        return output_tokens

    if probs is not None:
        permuted_probs = paddle.gather(probs.flatten(), prob_permuted_indices)
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = paddle.zeros(restore_shape, dtype=permuted_tokens.dtype)
    # Scatter add the permuted_input back to the original positions
    if scatter_add_ is not None:
        scatter_add_(output_tokens, token_permuted_indices, permuted_tokens)
    else:
        output_tokens.scatter_(
            index=token_permuted_indices, updates=permuted_tokens, overwrite=False
        )
    return output_tokens


class UnZipNode:
    """
    UnZipNode 类用于对输入的token 矩阵根据分发索引进行解压操作,得到专家需要处理的 token。
    """

    def __init__(self, token_dispatcher, name="unzip"):
        self.token_dispatcher = token_dispatcher
        self.name = name
        self.unzipped_probs = None
        self.zipped_expertwise_rowmap = None

    def reset_statue(self):
        """
        重置模型的状态。

        Args:
            无

        Returns:
            无

        """
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
        """
        前向传播函数，用于解压输入的张量。

        Args:
            hs_2d_dispatched: 原始输入的token。
            dispatched_indices: 分发索引。
            dispatched_probs: 分发概率。

        Returns:
            tuple: 返回解压后的令牌、压缩后的专家行映射、解压后的概率。
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
        self,
        dx,
        hidden_states_out_grad_shape,
        probs_grad,
        dispatched_indices,
        num_experts,
    ):
        """
        反向传播函数。
        """
        weighted_zipped_tokens, probs_grad_zipped = TDU.tokens_zip(
            dx,
            self.zipped_expertwise_rowmap,
            dispatched_indices,
            probs_grad,
            total_zipped_tokens=hidden_states_out_grad_shape[0],
            num_experts=num_experts,
        )
        self.reset_statue()
        return weighted_zipped_tokens, probs_grad_zipped


class ZipNode:
    """
    与 UnzipNode 相反，类用将解压后的 token 张量压缩回原始状态。
    """

    def __init__(self, token_dispatcher, name="zip"):
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
        """
        前向传播函数，用于压缩输入的张量。
        """
        expert_out_zipped, zipped_probs_topk = TDU.tokens_zip(
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
        fill_output=True,
    ):
        """
        用于反向传播函数。
        """
        (
            unzipped_grad,
            zipped_expertwise_rowmap_grad,
            unzipped_probs_grad,
            _,
        ) = TDU.tokens_unzip_stable(
            grad_output,
            None,
            dispatched_indices,
            dispatched_probs,
            top_k,
            num_experts,
            tokens_per_expert,
            padding_multiplex=FP8_ALIGN,
            fill_output=fill_output,
        )

        return unzipped_grad
