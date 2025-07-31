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
@author: kebo
@contact: kebo01@baidu.com

@version: 1.0
@file: task_gate.py
@time: 2023/10/19 17:57:06
@Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved

这一行开始写关于本文件的说明与解释


"""
import logging

import paddle
from paddle import Tensor
import paddle.nn.functional as F
from models.moe.top2_gate import Top2Gate


logger = logging.getLogger(__name__)


class TaskGate(Top2Gate):
    """
    Task Gate
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
        self.dispatch_by_task = True

    def forward(self, input: Tensor):
        """task gate forward

        Args:
            input (Tensor): _description_

        Returns:
            _type_: _description_
        """
        with paddle.amp.auto_cast(False):
            logits = F.linear(
                input.cast("float32").detach(), self.weight
            )  # [S,M] -> [S,E]
            if self.dispatch_by_task:
                router_loss = self._cal_task_loss(logits)
                return None, None, None, None, router_loss

    def _cal_task_loss(self, logits):
        """_summary_

        Args:
            logits (_type_): _description_
        """
        with paddle.amp.auto_cast(False):
            task_label = paddle.ones([logits.shape[0], 1], "int64") * self.rank
            task_loss = F.cross_entropy(logits, task_label)
        return task_loss
