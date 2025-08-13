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

# !/usr/bin/env python3

import hashlib
import numpy as np
import logging

import paddle
from paddle import distributed as dist
from paddle.autograd import PyLayer
from paddle.distributed import fleet


from models.comm_utils import (
    scatter,
    all_gather,
    reduce_scatter,
)


from paddle.distributed import in_auto_parallel_align_mode


try:
    from paddle.nn.functional import gemm_reduce_scatter, all_gather_gemm
except ImportError:
    gemm_reduce_scatter = None
    all_gather_gemm = None
    flux = None

logger = logging.getLogger(__name__)

if not hasattr(paddle.Tensor, "contiguous"):

    def contiguous(self):

        return self

    setattr(paddle.Tensor, "contiguous", contiguous)


if not hasattr(paddle.Tensor, "_md5sum"):

    def _md5sum(self):
        numpy_array = np.array(self)
        array_bytes = numpy_array.tobytes()
        return hashlib.md5(array_bytes).hexdigest()

    setattr(paddle.Tensor, "_md5sum", _md5sum)


def get_hcg():
    return fleet.get_hybrid_communicate_group()


class ScatterOp(PyLayer):

    @staticmethod
    def forward(ctx, input, axis=0, group=None):
        ctx.axis = axis
        ctx.group = group
        return scatter(input, axis=axis, group=ctx.group)

    @staticmethod
    def backward(ctx, grad):
        return all_gather(grad, axis=ctx.axis, group=ctx.group)


class GatherOp(PyLayer):

    @staticmethod
    def forward(ctx, input, axis=0, group=None):
        ctx.axis = axis
        ctx.group = group
        return all_gather(input, axis=axis, group=group)

    @staticmethod
    def backward(ctx, grad):
        return scatter(grad, axis=ctx.axis, group=ctx.group)


class AllGatherOp(PyLayer):

    @staticmethod
    def forward(ctx, input, group=None):
        ctx.group = group
        return all_gather(input, group=group)

    @staticmethod
    def backward(ctx, grad):
        if in_auto_parallel_align_mode():
            group = ctx.group
            if group is None:
                group = get_hcg().get_model_parallel_group()
            pg = group.process_group
            pg.allreduce(grad).wait()
            return paddle.split(grad, group.nranks, axis=0)[group.rank]
        else:
            return reduce_scatter(grad, group=ctx.group)


class ReduceScatterOp(PyLayer):
    @staticmethod
    def forward(ctx, input, group=None):
        ctx.group = group
        return reduce_scatter(input, group=group)

    @staticmethod
    def backward(ctx, grad):
        return all_gather(grad, group=ctx.group)


class AllGatherVarlenOp(PyLayer):

    @staticmethod
    def forward(ctx, input, group=None):
        hcg = fleet.get_hybrid_communicate_group()
        if group is None:
            group = hcg.get_model_parallel_group()

        shape0 = paddle.to_tensor([input.shape[0]])
        shape0_all = paddle.empty(shape=[group.nranks], dtype=shape0.dtype)
        dist.stream.all_gather(shape0_all, shape0, group=group, use_calc_stream=True)
        shape0_all = shape0_all.numpy()
        max_shape0 = shape0_all.max()

        indices = []
        for idx, s in enumerate(shape0_all):
            offset = idx * max_shape0
            indices.append(list(range(offset, offset + s)))
        indices = np.concatenate(indices, axis=0)
        indices = indices.reshape([-1] + [1] * (len(input.shape) - 1))
        indices = paddle.to_tensor(indices, dtype=paddle.int32)

        padding = max_shape0 - input.shape[0]

        ctx.shape0 = input.shape[0]
        ctx.max_shape0 = max_shape0
        ctx.shape0_all = shape0_all
        ctx.padding = padding
        ctx.indices = indices
        ctx.group = group

        if padding > 0:
            input_shape = input.shape
            input_shape[0] = padding
            padding_tensor = paddle.empty(shape=input_shape, dtype=input.dtype)
            input = paddle.concat([input, padding_tensor], axis=0)
        output = all_gather(input, group)
        output = paddle.take_along_axis(output, indices, axis=0)

        return output

    @staticmethod
    def backward(ctx, grad):
        input_shape = grad.shape
        input_shape[0] = ctx.max_shape0 * ctx.shape0_all.shape[0]
        output = paddle.zeros(shape=input_shape, dtype=grad.dtype)

        grad = paddle.scatter(output, ctx.indices, grad)

        grad = scatter(grad, ctx.group)

        if ctx.padding > 0:
            grad = grad[: ctx.shape0]
        return grad


class GemmReduceScatterOp(PyLayer):

    @staticmethod
    def forward(ctx, input, weight, group):

        ctx.save_for_backward(input, weight)
        ctx.group = group
        output = gemm_reduce_scatter(input, weight, group)
        return output

    @staticmethod
    def backward(ctx, grad):
        input, weight = ctx.saved_tensor()
        group = ctx.group
        if input.stop_gradient and weight.stop_gradient:
            return None, None

        if input.stop_gradient:
            input_grad = None
            grad_parallel = None
        else:
            input_grad, grad_parallel = all_gather_gemm(
                grad, weight, group, deepcopy_input_parallel=False
            )

        if weight.stop_gradient:
            weight_grad = None
        else:
            if grad_parallel is None:
                grad_parallel = all_gather(grad)
            weight_grad = paddle.matmul(input, grad_parallel, transpose_x=True)
        return input_grad, weight_grad


def sequence_parallel_sparse_mask_labels(labels, ignore_label=-100):
    hcg = fleet.get_hybrid_communicate_group()
    group = hcg.get_model_parallel_group()
    labels = labels.flatten()
    labels_local = paddle.split(labels, group.nranks)[group.rank]

    tgt_index = paddle.nonzero(labels_local != ignore_label).squeeze()
    if tgt_index.numel() == 0:
        tgt_index = paddle.to_tensor([0])

    tgt_index = tgt_index.reshape([-1]).astype(paddle.int32)
    labels_local_gather = paddle.take_along_axis(labels_local, tgt_index, axis=0)
    labels_all_gather = AllGatherVarlenOp.apply(labels_local_gather)
    return labels_all_gather, tgt_index.reshape([-1, 1])
