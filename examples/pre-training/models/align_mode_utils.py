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

"""help to in do precision alignment for auto parallel"""

import paddle
from paddle.autograd import PyLayer
from paddle.distributed import fleet


class Concat(PyLayer):
    """
    拼接logits
    """

    @staticmethod
    def forward(ctx, inp, axis, group):
        inputs = []
        paddle.distributed.all_gather(inputs, inp, group=group)
        with paddle.no_grad():
            cat = paddle.concat(inputs, axis=axis)
        ctx.args_axis = axis
        ctx.args_group = group
        return cat

    @staticmethod
    def backward(ctx, grad):
        axis = ctx.args_axis
        group = ctx.args_group
        with paddle.no_grad():
            grads = paddle.split(
                grad, paddle.distributed.get_world_size(group), axis=axis
            )
        grad = grads[paddle.distributed.get_rank(group)]
        return grad


def concat_mp_with_grad(input):
    hcg = fleet.get_hybrid_communicate_group()
    mp_degree = hcg.get_model_parallel_world_size()
    if mp_degree <= 1:
        return input
    else:
        group = hcg.get_model_parallel_group()
        return Concat.apply(input, -1, group)
