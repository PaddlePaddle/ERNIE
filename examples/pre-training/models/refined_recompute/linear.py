""" row_ln & column_ln setup """

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

import queue
import logging

import paddle
from paddle import framework
from paddle.autograd import PyLayer
from paddle.incubate.nn.functional import fused_linear
from paddle.nn.functional.common import linear

logger = logging.getLogger(__name__)
try:
    from custom_setup_ops import matmul_bwd, add_bwd, fused_gemm_epilogue_bwd
except ImportError:
    logger.error(
        'custom_setup_ops not found, "\
                 "run `python third_party/ernie-core/src/ernie_core/ops/setup.py install` "\
                 "to build custom_setup_ops'
    )
    matmul_bwd = None

from models.comm_utils import all_gather, reduce_scatter
from models.refined_recompute.queue_check import global_rr_queue_log


@paddle.no_grad()
def matmul_bwd_(x, weight, y_grad, trans_x, trans_y):
    """matmul_bwd_"""
    if not x.stop_gradient and not weight.stop_gradient:
        return matmul_bwd(x, weight, y_grad, trans_x, trans_y)

    dx, dweight = None, None
    assert trans_x is False
    assert trans_y is False
    assert (
        len(x.shape) == 2 and len(weight.shape) == 2
    ), f"x shape {x.shape}, weight shape {weight.shape}"
    if x.stop_gradient:
        dweight = paddle.matmul(x, y_grad, transpose_x=True)

    if weight.stop_gradient:
        dx = paddle.matmul(y_grad, weight, transpose_y=True)

    return dx, dweight


@paddle.no_grad()
def linear_grad(x, weight, bias, y_grad):
    """linear_grad"""
    if bias is not None:
        _, bias_grad = add_bwd(x, weight, bias, y_grad, -1)
        x_grad, weight_grad = matmul_bwd_(x, weight, y_grad, False, False)
        return x_grad, weight_grad, bias_grad
    else:
        x_grad, weight_grad = matmul_bwd_(x, weight, y_grad, False, False)
        return x_grad, weight_grad


@paddle.no_grad()
def fused_linear_grad(x, weight, bias, y_grad):
    """fused_linear_grad"""
    if bias is not None:
        try:
            x_grad, weight_grad, bias_grad = fused_gemm_epilogue_bwd(
                x,
                weight,
                y_grad,
                False,
                False,
                not x.stop_gradient,
                not weight.stop_gradient,
                not bias.stop_gradient,
            )
        except TypeError:
            x_grad, weight_grad, bias_grad = fused_gemm_epilogue_bwd(
                x, weight, y_grad, False, False
            )

        if x.stop_gradient:
            x_grad = None

        if weight.stop_gradient:
            weight_grad = None

        if bias.stop_gradient:
            bias_grad = None

        return x_grad, weight_grad, bias_grad
    else:
        x_grad, weight_grad = matmul_bwd_(x, weight, y_grad, False, False)
        return x_grad, weight_grad


class ColumnLNFunctor(PyLayer):
    """Helper for ColumnLN"""

    @staticmethod
    def forward(ctx, fwd_func, x, weight, bias, hold_tensors):
        """forward"""
        fwd_output = hold_tensors["res_output"]
        ctx.fwd_func = fwd_func
        ctx.save_for_backward(x, weight, bias)
        return fwd_output

    @staticmethod
    def backward(ctx, y_grad):
        """backward"""
        x, weight, bias = ctx.saved_tensor()
        # backward for linear/fused_linear
        if id(ctx.fwd_func) == id(linear):
            outputs = linear_grad(x, weight, bias, y_grad)
        elif id(ctx.fwd_func) == id(fused_linear):
            outputs = fused_linear_grad(
                x, weight, bias if bias is not None else None, y_grad
            )
        else:
            raise ValueError("not supported")
        return outputs


class RowLNFunctor(PyLayer):
    """Helper for RowLN"""

    @staticmethod
    def forward(ctx, fwd_func, x, weight, bias, hold_tensors):
        """forward"""
        fwd_output = hold_tensors["res_output"]
        ctx.fwd_func = fwd_func
        ctx.save_for_backward(x, weight, bias)
        return fwd_output

    @staticmethod
    def backward(ctx, y_grad):
        """backward"""
        x, weight, bias = ctx.saved_tensor()
        # backward for reduce_scatter
        y_grad = all_gather(y_grad)

        # backward for linear/fused_linear
        if id(ctx.fwd_func) == id(linear):
            outputs = linear_grad(x, weight, bias, y_grad)
        elif id(ctx.fwd_func) == id(fused_linear):
            outputs = fused_linear_grad(x, weight, bias, y_grad)
        else:
            raise ValueError("not supported")

        return outputs


class ColumnLNRefinedRcompute(object):
    """Helper for ColumnLN in RefinedRcompute"""

    def __init__(self):
        self._hold_tensors_queue = queue.Queue()
        global_rr_queue_log.update(self._hold_tensors_queue, "ColumnLN")

    def forward(self, fwd_func, x, weight, bias):
        """forward"""
        tracer = framework._dygraph_tracer()
        if not tracer._has_grad:
            fwd_output = self._first_fwd(fwd_func, x, weight, bias)
            self._hold_tensors_queue.put(fwd_output)
            return fwd_output
        else:
            assert not self._hold_tensors_queue.empty(), "queue should not be empty"
            fwd_output = self._hold_tensors_queue.get()
            output = self._second_fwd(fwd_func, fwd_output, x, weight, bias)
            return output

    @paddle.no_grad()
    def _first_fwd(self, fwd_func, x, weight, bias):
        output = fwd_func(x, weight, bias)
        return output

    def _second_fwd(self, fwd_func, fwd_output, x, weight, bias):
        return ColumnLNFunctor.apply(
            fwd_func, x, weight, bias, {"res_output": fwd_output}
        )

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)


class RowLNRefinedRcompute(object):
    """Helper for RowLN in RefinedRcompute"""

    def __init__(self):
        self._hold_tensors_queue = queue.Queue()
        global_rr_queue_log.update(self._hold_tensors_queue, "RowLN")

    def forward(self, fwd_func, x, weight, bias):
        """forward"""
        tracer = framework._dygraph_tracer()
        if not tracer._has_grad:
            fwd_output = self._first_fwd(fwd_func, x, weight, bias)
            self._hold_tensors_queue.put(fwd_output)
            return fwd_output
        else:
            assert not self._hold_tensors_queue.empty(), "queue should not be empty"
            fwd_output = self._hold_tensors_queue.get()
            output = self._second_fwd(fwd_func, fwd_output, x, weight, bias)
            return output

    @paddle.no_grad()
    def _first_fwd(self, fwd_func, x, weight, bias):
        output = fwd_func(x, weight, bias)
        output = reduce_scatter(output)
        return output

    def _second_fwd(self, fwd_func, fwd_output, x, weight, bias):
        return RowLNFunctor.apply(fwd_func, x, weight, bias, {"res_output": fwd_output})

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
