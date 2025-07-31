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
        """
            分发送操作，将输入数据分割成多个部分并广播到所有进程中。
        然后将这些部分的数据合并起来，形成一个大的张量。

        Args:
            ctx (Context object): Context对象，用于存储和传递上下文信息。
            inp (Tensor): 输入张量，需要被分发和广播。
            axis (int, optional): 指定维度轴，默认为0。
            group (Group, optional): 指定分发和广播的进程组，默认为None。

        Returns:
            Tensor: 返回一个大的张量，包含了所有进程中的输入数据。

        Raises:
            None.
        """
        inputs = []
        paddle.distributed.all_gather(inputs, inp, group=group)
        with paddle.no_grad():
            cat = paddle.concat(inputs, axis=axis)
        ctx.args_axis = axis
        ctx.args_group = group
        return cat

    @staticmethod
    def backward(ctx, grad):
        """
            计算反向传播的结果，将输入的梯度分割成每个进程的梯度。
        参数：
            ctx (Context): 上下文对象，包含了需要传递给backward函数的参数。
            grad (Tensor): 输入的梯度张量，形状为（batch_size, ...）。
        返回值：
            Tensor: 返回一个形状为（batch_size, ...）的张量，表示每个进程的梯度。
            如果在反向传播时没有使用到梯度，则返回None。
        """
        axis = ctx.args_axis
        group = ctx.args_group
        with paddle.no_grad():
            grads = paddle.split(
                grad, paddle.distributed.get_world_size(group), axis=axis
            )
        grad = grads[paddle.distributed.get_rank(group)]
        return grad


def concat_mp_with_grad(input):
    """
    将模型并行（Model Parallel）的输入与梯度信息进行连接，返回一个新的张量。如果模型并行数量小于等于1，则直接返回原始输入；否则，将输入和梯度信息进行连接后返回。

    Args:
        input (Tensor, tuple[Tensor]): 要连接的张量或张量元组，可以是任意维度的张量。如果是张量元组，则会按照指定的轴进行连接。

    Returns:
        Tensor: 连接后的张量，维度与输入相同。如果输入为单个张量，则返回的张量与输入相同；如果输入为多个张量，则返回的张量与输入相同，但具有更大的维度。
    """
    hcg = fleet.get_hybrid_communicate_group()
    mp_degree = hcg.get_model_parallel_world_size()
    if mp_degree <= 1:
        return input
    else:
        group = hcg.get_model_parallel_group()
        return Concat.apply(input, -1, group)
