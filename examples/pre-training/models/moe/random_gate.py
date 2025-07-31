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

"""
@version: 1.0
@file: random_gate.py
@time: 2023/07/26 12:06:42
@Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved

这一行开始写关于本文件的说明与解释


"""
import logging

import paddle
from models.moe.top2_gate import Top2Gate

logger = logging.getLogger(__name__)


class RandomGate(Top2Gate):
    """_summary_

    Args:
        Top2Gate (_type_): _description_
    """

    def __init__(self, config, layer_idx: int, group) -> None:
        """
        初始化函数，用于创建一个MoE层对象。

        Args:
            config: 配置参数，包含moe_num_experts和moe_capacity信息。
            layer_idx: 当前层的索引。
            group: 当前层的参数分组。

        Returns:
            None

        """
        super().__init__(config, layer_idx, group)
        self.num_experts = config.moe_num_experts
        cap = config.moe_capacity
        self.cap = cap[0] if isinstance(cap, (tuple, list)) else cap

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

    def forward(self, input):
        """
        执行前向传播，计算Capacity、dispatch_mask、combine_weights、scatter_index和router_loss。

        Args:
            input (paddle.Tensor): 形状为[seqlen, num_experts]的paddle张量，表示每个时间步和专家之间的权重。

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
            - capacity: 形状为[1,]的paddle张量，表示平均容量。
            - dispatch_mask: 形状为[seqlen, num_experts]的paddle张量，表示每个时间步和专家之间的掩码。
            - combine_weights: 形状为[seqlen, num_experts]的paddle张量，表示每个时间步和专家之间的组合权重。
            - scatter_index: 形状为[seqlen, num_experts]的paddle张量，表示每个时间步和专家之间的散布索引。
            - router_loss: 形状为[1,]的paddle张量，表示路由损失。

        """
        seqlen = input.shape[0]
        orig_dtype = input.dtype

        # x = paddle.arange(seqlen).cast(paddle.int64)
        # y = paddle.arange(seqlen).cast(paddle.int64)
        # x = x % self.num_experts
        # y = (y + 1) % self.num_experts
        # x = F.one_hot(x, num_classes=self.num_experts)
        # y = F.one_hot(y, num_classes=self.num_experts)
        # x = 2 * x + y
        x = paddle.rand([seqlen, self.num_experts])
        with paddle.amp.auto_cast(False):
            capacity, dispatch_mask, combine_weights, scatter_index, l_aux, l_zloss = (
                self.top2gating(x, self.cap)
            )
        router_loss = (
            l_aux * self.config.aux_loss_lambda + l_zloss * self.config.z_loss_lambda
        )
        combine_weights = combine_weights.cast(orig_dtype)
        return capacity, dispatch_mask, combine_weights, scatter_index, router_loss
