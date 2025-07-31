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

from typing import Tuple
import logging
import numpy as np

import paddle
from paddle import Tensor
import paddle.nn.functional as F
from models.moe.top2_gate import Top2Gate


logger = logging.getLogger(__name__)


class RoundRobinGate(Top2Gate):
    """
    测试用轮流分发gate
    """

    def __init__(self, config, layer_idx: int, group) -> None:
        """
        重载构造函数，用于初始化MoE层。

        Args:
        - config: 配置信息，包含MoE相关配置参数。
        - layer_idx: 当前层的索引。
        - group: 上一层的输出张量组成的列表。

        Returns:
        - None: 没有返回值。

        """
        super().__init__(config, layer_idx, group)
        cap = config.moe_capacity
        self.cap = cap[0] if isinstance(cap, (tuple, list)) else cap
        self.second_dispatch = 1

    def _create_gate_parameter(self):
        """
        创建门参数函数。

        Args:
            无参。

        Returns:
            无返回值。

        Raises:
            该方法不引发任何异常。
        """
        pass

    def forward(self, input, correction_bias=None):
        """
        计算moe的输出。

        Args:
            input：输入数据，为一个形状为[batch_size, max_seq_len]且值为整数的Tensor。

        Returns:
            tuple：包含capacity、dispatch_mask、combine_weights、scatter_index和router_loss这五个元素组成的tuple。

            - capacity：经过处理后的moe模型的capacity。
            - dispatch_mask：经过处理后的moe模型的dispatch mask。
            - combine_weights：经过处理后的moe模型的combination weights。
            - scatter_index：经过处理后的moe模型的scatter index。
            - router_loss：经过处理后的moe模型的router loss。

        """
        seqlen = input.shape[0]
        orig_dtype = input.dtype

        x = paddle.arange(seqlen).cast(paddle.int64)
        y = paddle.arange(seqlen).cast(paddle.int64)
        # x = paddle.zeros([seqlen]).cast(paddle.int64)
        # y = paddle.zeros([seqlen]).cast(paddle.int64)
        x = x % self.num_experts
        y = (y + self.second_dispatch) % self.num_experts
        x = F.one_hot(x, num_classes=self.num_experts)
        y = F.one_hot(y, num_classes=self.num_experts)
        # x[:x.shape[0],:] = x[:x.shape[0],:] *2
        # y[y.shape[0]:,:] = y[y.shape[0]:,:] *2
        x = 2 * x + y
        x[x <= 0.5] = -np.inf

        with paddle.amp.auto_cast(False):
            capacity, dispatch_mask, combine_weights, scatter_index, l_aux, l_zloss = (
                self.top2_gating(x, self.cap)
            )
        combine_weights = combine_weights.cast(orig_dtype)
        # router_loss = l_aux * self.config.moe_aux_loss_lambda + l_zloss * self.config.moe_z_loss_lambda
        combine_weights.stop_gradient = False
        router_loss = paddle.zeros([1], dtype="float32")
        router_loss.stop_gradient = False
        return capacity, dispatch_mask, combine_weights, scatter_index, router_loss, x

    def _cal_orthogonal_loss(self, weight=None, with_opt=True):
        return paddle.to_tensor(0)


class RoundRobinGateFused(RoundRobinGate):
    """
    doc
    """

    def forward(
        self,
        input: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        """
        Args:
            input: paddle.Tensor, hidden-states of layer
        Retruns:
            paddle.Tensor [Seq, Expert, Capacity]: float32, combine weights
            paddle.Tensor [Seq, Expert, Capacity]: bool, dispatch mask
            Tuple[paddle.Tensor]: `GateOutput`
        """

        seqlen = input.shape[0]
        orig_dtype = input.dtype

        # capacity = 2S/E
        capacity = int(self.cap * seqlen // self.num_experts)

        x = paddle.arange(seqlen).cast(paddle.int64)
        y = paddle.arange(seqlen).cast(paddle.int64)
        # x = paddle.zeros([seqlen]).cast(paddle.int64)
        # y = paddle.zeros([seqlen]).cast(paddle.int64)
        x = x % self.num_experts
        y = (y + self.second_dispatch) % self.num_experts
        x = F.one_hot(x, num_classes=self.num_experts)
        y = F.one_hot(y, num_classes=self.num_experts)
        # x[:x.shape[0],:] = x[:x.shape[0],:] *2
        # y[y.shape[0]:,:] = y[y.shape[0]:,:] *2
        x = 2 * x + y
        x[x <= 0.5] = -np.inf

        # router_loss = l_aux * self.config.moe_aux_loss_lambda + l_zloss * self.config.moe_z_loss_lambda
        x.stop_gradient = False
        router_loss = paddle.zeros([1], dtype="float32")
        router_loss.stop_gradient = False
        return x, capacity, router_loss
