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

        super().__init__(config, layer_idx, group)
        self.num_experts = config.moe_num_experts
        cap = config.moe_capacity
        self.cap = cap[0] if isinstance(cap, (tuple, list)) else cap

    def _create_gate_parameter(self):

        pass

    def forward(self, input):

        seqlen = input.shape[0]
        orig_dtype = input.dtype

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
