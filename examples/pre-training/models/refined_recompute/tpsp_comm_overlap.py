""" row_comm_ln & column_comm_ln setup """

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

import queue
import logging

import paddle
from paddle import framework
from paddle.autograd import PyLayer

logger = logging.getLogger(__name__)

from models.refined_recompute.queue_check import global_rr_queue_log
from models.comm_utils import all_gather

try:
    from paddle.nn.functional import gemm_reduce_scatter, all_gather_gemm
except ImportError:
    gemm_reduce_scatter = None
    all_gather_gemm = None


@paddle.no_grad()
def ag_gemm_bwd_(y_grad, x_parallel, weight, group, x_stop_gradient):
    """all gather gemm bwd"""
    assert (
        len(x_parallel.shape) == 2 and len(weight.shape) == 2
    ), f"x_parallel shape {x_parallel.shape}, weight shape {weight.shape}"
    if x_stop_gradient:
        x_grad = None
    else:
        x_grad = gemm_reduce_scatter(y_grad, weight, group)
    if weight.stop_gradient:
        weight_grad = None
    else:
        weight_grad = paddle.matmul(x_parallel, y_grad, transpose_x=True)
    return x_grad, weight_grad


@paddle.no_grad()
def gemm_rs_bwd_(y_grad, x, weight, group):
    """gemm reduce scatter bwd"""
    assert (
        len(x.shape) == 2 and len(weight.shape) == 2
    ), f"x shape {x.shape}, weight shape {weight.shape}"
    if x.stop_gradient and weight.stop_gradient:
        return None, None

    if x.stop_gradient:
        x_grad = None
        y_grad_parallel = None
    else:
        x_grad, y_grad_parallel = all_gather_gemm(
            y_grad, weight, group, deepcopy_input_parallel=False
        )

    if weight.stop_gradient:
        weight_grad = None
    else:
        if y_grad_parallel is None:
            y_grad_parallel = all_gather(y_grad, group=group)
        weight_grad = paddle.matmul(x, y_grad_parallel, transpose_x=True)
    return x_grad, weight_grad


class ColumnCommLNFunctor(PyLayer):
    """Helper for ColumnCommLNFunctor"""

    @staticmethod
    def forward(ctx, x, weight, hold_tensors, group):
        """forward"""
        fwd_output = hold_tensors["res_output"]
        x_parallel = all_gather(x, group=group)
        ctx.save_for_backward(x_parallel, weight)
        ctx.group = group
        ctx.x_stop_gradient = x.stop_gradient
        return fwd_output

    @staticmethod
    def backward(ctx, y_grad):
        """backward"""
        # backward for all_gather_gemm
        x_parallel, weight = ctx.saved_tensor()
        group = ctx.group
        x_stop_gradient = ctx.x_stop_gradient
        x_grad, weight_grad = ag_gemm_bwd_(
            y_grad, x_parallel, weight, group, x_stop_gradient
        )
        return x_grad, weight_grad


class RowCommLNFunctor(PyLayer):
    """Helper for RowCommLN"""

    @staticmethod
    def forward(ctx, x, weight, hold_tensors, group):
        """forward"""
        fwd_output = hold_tensors["res_output"]
        ctx.save_for_backward(x, weight)
        ctx.group = group
        return fwd_output

    @staticmethod
    def backward(ctx, y_grad):
        """backward"""
        # backward for gemm_reduce_scatter
        x, weight = ctx.saved_tensor()
        group = ctx.group
        x_grad, weight_grad = gemm_rs_bwd_(y_grad, x, weight, group)
        return x_grad, weight_grad


class ColumnCommLNRefinedRcompute(object):
    """Helper for ColumnCommLN in RefinedRcompute"""

    def __init__(self):
        self._hold_tensors_queue = queue.Queue()
        global_rr_queue_log.update(self._hold_tensors_queue, "ColumnCommLN")

    def forward(self, x, weight, group):
        """forward"""
        tracer = framework._dygraph_tracer()
        if not tracer._has_grad:
            fwd_output = self._first_fwd(x, weight, group)
            self._hold_tensors_queue.put(fwd_output)
            return fwd_output
        else:
            assert not self._hold_tensors_queue.empty(), "queue should not be empty"
            fwd_output = self._hold_tensors_queue.get()
            output = self._second_fwd(fwd_output, x, weight, group)
            return output

    @paddle.no_grad()
    def _first_fwd(self, x, weight, group):
        output, _ = all_gather_gemm(x, weight, group, deepcopy_input_parallel=False)
        return output

    def _second_fwd(self, fwd_output, x, weight, group):
        return ColumnCommLNFunctor.apply(x, weight, {"res_output": fwd_output}, group)

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)


class RowCommLNRefinedRcompute(object):
    """Helper for RowCommLN in RefinedRcompute"""

    def __init__(self):
        self._hold_tensors_queue = queue.Queue()
        global_rr_queue_log.update(self._hold_tensors_queue, "RowCommLN")

    def forward(self, x, weight, group):
        """forward"""
        tracer = framework._dygraph_tracer()
        if not tracer._has_grad:
            fwd_output = self._first_fwd(x, weight, group)
            self._hold_tensors_queue.put(fwd_output)
            return fwd_output
        else:
            assert not self._hold_tensors_queue.empty(), "queue should not be empty"
            fwd_output = self._hold_tensors_queue.get()
            output = self._second_fwd(fwd_output, x, weight, group)
            return output

    @paddle.no_grad()
    def _first_fwd(self, x, weight, group):
        output = gemm_reduce_scatter(x, weight, group)
        return output

    def _second_fwd(self, fwd_output, x, weight, group):
        return RowCommLNFunctor.apply(x, weight, {"res_output": fwd_output}, group)

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
