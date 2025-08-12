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

"""
top2gate
"""


from typing import Tuple
import logging
from paddle import Tensor
import paddle.distributed as dist

# import paddle.nn.functional as F

logger = logging.getLogger(__name__)

from models.moe.top2_gate_auto_auto import TopKGateFused
from models.moe.moe_utils_auto import get_mesh, get_flatten_mesh


class TopKGateFusedAuto(TopKGateFused):
    """doc"""

    def __init__(self, config, layer_idx: int, group, gate_weight=None, ipp=0) -> None:
        super().__init__(config, layer_idx, group, gate_weight)
        self.ipp = ipp
        self.weight = dist.shard_tensor(
            self.weight, get_flatten_mesh(get_mesh(self.ipp)), [dist.Replicate()]
        )

    def forward(
        self,
        input: Tensor,
        token_type_ids=None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:  # type: ignore
        """
        Args:
            input: paddle.Tensor, hidden-states of layer
        Retruns:
            paddle.Tensor [Seq, Expert, Capacity]: float32, combine weights
            paddle.Tensor [Seq, Expert, Capacity]: bool, dispatch mask
            Tuple[paddle.Tensor]: `GateOutput`
        """
        num_experts = (
            sum(self.num_experts)
            if self.config.multimodel_experts
            else self.num_experts
        )
        if self.training:
            cap = self.cap[0]
        elif input.shape[0] < num_experts:  # seqlen < num_expert
            cap = self.cap[2]
        else:
            cap = self.cap[1]
        num_tokens = input.shape[0]
        # capacity = 2S/E
        global_capacity = int(cap * num_tokens // num_experts)
        local_num_tokens = input._local_shape[0]
        local_capacity = int(cap * local_num_tokens // num_experts)

        logits, _, router_loss = super().forward(input, token_type_ids)

        return logits, global_capacity, router_loss, local_capacity
