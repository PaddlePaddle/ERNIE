# !/usr/bin/env python3
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import os
import queue
import logging
from paddle import framework
from paddle import _C_ops
from paddle.autograd import PyLayer
from models.refined_recompute.queue_check import global_rr_queue_log

logger = logging.getLogger(__name__)

try:
    from custom_setup_ops import flash_attn_bwd
except ImportError:
    logger.error(
        "custom_setup_ops not found, run `python third_party/ernie-core/src/ernie_core/ops/setup.py install` to build custom_setup_ops"
    )
    flash_attn_bwd = None

try:
    from custom_setup_ops import flash_attn_v1_bwd
except ImportError:
    logger.error("flash_attn v1 is not supported")
    flash_attn_v1_bwd = None

try:
    from custom_setup_ops import flashmask_attn_bwd
except ImportError:
    logger.error(
        "custom_setup_ops not found, "
        "run `python third_party/ernie-core/src/ernie_core/ops/setup.py install` to build custom_setup_ops"
    )
    flashmask_attn_bwd = None

g_use_flash_attn_v1 = os.getenv(
    "FLAGS_flash_attn_version", "v2"
).strip().lower() == "v1" and hasattr(_C_ops, "flash_attn_v1")


def flashattn_auto_cast(q, k, v, dtype=paddle.bfloat16):
    if q.dtype != dtype:
        q = q.astype(dtype)
    if k.dtype != dtype:
        k = k.astype(dtype)
    if v.dtype != dtype:
        v = v.astype(dtype)
    return q, k, v


class FlashAttnFunctor(PyLayer):
    """
    FlashAttnFunctor is a fake layer to make the backward of FlashAttention work.
    """

    @staticmethod
    def forward(ctx, q, k, v, hold_tensors):
        result_attention = hold_tensors["result_attention"]
        softmax_lse = hold_tensors["softmax_lse"]
        seed_offset = hold_tensors["seed_offset"]
        dropout = hold_tensors["dropout"]
        ctx.save_for_backward(
            q, k, v, result_attention, softmax_lse, seed_offset, dropout
        )
        return result_attention

    @staticmethod
    def backward(ctx, grad):
        q, k, v, result_attention, softmax_lse, seed_offset, dropout = (
            ctx.saved_tensor()
        )

        if g_use_flash_attn_v1:
            q_grad, k_grad, v_grad = flash_attn_v1_bwd(
                q.detach(),
                k.detach(),
                v.detach(),
                result_attention,
                softmax_lse,
                seed_offset,
                grad,
                dropout=dropout,
                causal=True,
            )
        else:
            q_grad, k_grad, v_grad = flash_attn_bwd(
                q.detach(),
                k.detach(),
                v.detach(),
                result_attention,
                softmax_lse,
                seed_offset,
                grad,
                dropout=dropout,
                causal=True,
            )

        # release memory
        result_attention._clear_dataptr()
        softmax_lse._clear_dataptr()
        seed_offset._clear_dataptr()

        return q_grad, k_grad, v_grad


class RefinedRcomputeFlashAttention(object):
    def __init__(self):
        self._hold_tensors_queue = queue.Queue()
        global_rr_queue_log.update(self._hold_tensors_queue, "flash_attention")

    def forward(
        self,
        query_states,
        key_states,
        value_states,
        dropout=0.0,
        causal=True,
        return_softmax=False,
        training=True,
    ):
        if not framework._dygraph_tracer()._has_grad:
            attn_output, attn_weights = self._first_fwd(
                query_states,
                key_states,
                value_states,
                dropout=dropout,
                causal=causal,
                return_softmax=return_softmax,
                training=training,
            )
        else:
            assert not self._hold_tensors_queue.empty(), "queue should not be empty"
            attn_output, attn_weights = self._second_fwd(
                query_states, key_states, value_states
            )

        return attn_output, attn_weights

    @paddle.no_grad()
    def _first_fwd(
        self,
        query_states,
        key_states,
        value_states,
        dropout=0.0,
        causal=True,
        return_softmax=False,
        training=True,
    ):

        query_states, key_states, value_states = flashattn_auto_cast(
            query_states, key_states, value_states
        )

        if g_use_flash_attn_v1:
            (result_attention, result_softmax, softmax_lse, seed_offset) = (
                _C_ops.flash_attn_v1(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    causal,
                    return_softmax,
                    not training,
                )
            )
        else:
            (result_attention, result_softmax, softmax_lse, seed_offset) = (
                _C_ops.flash_attn(
                    query_states,
                    key_states,
                    value_states,
                    None,
                    None,
                    dropout,
                    causal,
                    return_softmax,
                    not training,
                    "",
                )
            )

        hold_tensors = {
            "result_attention": result_attention,
            "softmax_lse": softmax_lse,
            "seed_offset": seed_offset,
            "result_softmax": result_softmax,
            "dropout": dropout,
        }

        self._hold_tensors_queue.put(hold_tensors)
        return result_attention, result_softmax if return_softmax else None

    def _second_fwd(self, query_states, key_states, value_states):
        hold_tensors = self._hold_tensors_queue.get()
        query_states, key_states, value_states = flashattn_auto_cast(
            query_states, key_states, value_states
        )
        output = FlashAttnFunctor.apply(
            query_states, key_states, value_states, hold_tensors
        )
        return output, hold_tensors["result_softmax"]

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)


class FlashMaskAttnFunctor(PyLayer):
    """
    FlashAttnFunctor is a fake layer to make the backward of FlashAttention work.
    """

    @staticmethod
    def forward(ctx, q, k, v, startend_row_indices, hold_tensors):
        """forward"""
        result_attention = hold_tensors["result_attention"]
        softmax_lse = hold_tensors["softmax_lse"]
        seed_offset = hold_tensors["seed_offset"]
        dropout = hold_tensors["dropout"]
        causal = hold_tensors["causal"]

        ctx.save_for_backward(
            q,
            k,
            v,
            startend_row_indices,
            result_attention,
            softmax_lse,
            seed_offset,
            dropout,
            causal,
        )
        return result_attention

    @staticmethod
    def backward(ctx, grad):
        """backward"""
        (
            q,
            k,
            v,
            startend_row_indices,
            result_attention,
            softmax_lse,
            seed_offset,
            dropout,
            causal,
        ) = ctx.saved_tensor()
        assert (
            flashmask_attn_bwd is not None
        ), "flashmask_attn_bwd is None, please install it first."
        q_grad, k_grad, v_grad = flashmask_attn_bwd(
            q.detach(),
            k.detach(),
            v.detach(),
            startend_row_indices,
            result_attention,
            softmax_lse,
            seed_offset,
            grad,
            dropout=dropout,
            causal=causal,
        )

        # release memory
        result_attention._clear_dataptr()
        softmax_lse._clear_dataptr()
        seed_offset._clear_dataptr()

        return q_grad, k_grad, v_grad


class RefinedRcomputeFlashMaskAttention(object):
    """RefinedRcomputeFlashMaskAttention"""

    def __init__(self):
        self._hold_tensors_queue = queue.Queue()
        global_rr_queue_log.update(self._hold_tensors_queue, "flashmask_attention")

    def forward(
        self,
        query_states,
        key_states,
        value_states,
        startend_row_indices,
        dropout=0.0,
        causal=True,
        return_softmax=False,
        training=True,
    ):
        """forward"""

        if not framework._dygraph_tracer()._has_grad:
            attn_output = self._first_fwd(
                query_states,
                key_states,
                value_states,
                startend_row_indices,
                dropout=dropout,
                causal=causal,
                return_softmax=return_softmax,
                training=training,
            )
        else:
            assert not self._hold_tensors_queue.empty(), "queue should not be empty"
            attn_output = self._second_fwd(
                query_states, key_states, value_states, startend_row_indices
            )

        return attn_output

    @paddle.no_grad()
    def _first_fwd(
        self,
        query_states,
        key_states,
        value_states,
        startend_row_indices,
        dropout=0.0,
        causal=True,
        return_softmax=False,
        training=True,
    ):
        """_first_fwd"""

        query_states, key_states, value_states = flashattn_auto_cast(
            query_states, key_states, value_states
        )

        (result_attention, result_softmax, softmax_lse, seed_offset) = (
            _C_ops.flashmask_attention(
                query_states,
                key_states,
                value_states,
                startend_row_indices,
                None,
                dropout,
                causal,
                return_softmax,
                not training,
                "",
            )
        )

        hold_tensors = {
            "result_attention": result_attention,
            "softmax_lse": softmax_lse,
            "seed_offset": seed_offset,
            "result_softmax": result_softmax,
            "dropout": dropout,
            "causal": causal,
        }

        self._hold_tensors_queue.put(hold_tensors)
        return result_attention

    def _second_fwd(self, query_states, key_states, value_states, startend_row_indices):
        """_second_fwd"""
        hold_tensors = self._hold_tensors_queue.get()
        query_states, key_states, value_states = flashattn_auto_cast(
            query_states, key_states, value_states
        )
        output = FlashMaskAttnFunctor.apply(
            query_states, key_states, value_states, startend_row_indices, hold_tensors
        )
        return output

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
