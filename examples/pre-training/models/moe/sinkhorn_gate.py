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
import math
import paddle
from paddle import Tensor
import paddle.nn.functional as F

try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}

from models.moe.top2_gate import Top2Gate, compute_optimal_transport

try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}

logger = logging.getLogger(__name__)


class SinkHornGate(Top2Gate):
    """SinkHorn Gate"""

    def __init__(self, config, layer_idx: int, group) -> None:

        super().__init__(config, layer_idx, group)

        self.tol = 1e-7
        self.window_size = 0
        self.use_sinkhorn = config.sinkhorn_2gate
        self.cap = self.cap[0] if isinstance(self.cap, (tuple, list)) else self.cap
        self.enable_logging = config.moe_logging
        self.gate_scale = config.moe_gate_scale
        self.gate_detach = config.moe_gate_detach

    def forward(self, input, token_type_ids=None, correction_bias=None):

        orig_dtype = input.dtype

        with paddle.amp.auto_cast(False):
            if self.window_size:
                assert (
                    input.shape[0] % self.window_size == 0
                ), f"windos_size is {self.window_size}, input size is {input.shape}"
                indices = paddle.arange(0, input.shape[0], self.window_size)
                input = input[indices]
            input_32 = input.cast("float32")
            logits = F.linear(
                (1.0 - self.gate_detach) * input_32.detach()
                + self.gate_detach * input_32,
                self.weight,
            )  # [S,M] -> [S,E]
            if self.gate_scale:
                logits = logits / math.sqrt(self.weight.shape[0])  # cal temp

            num_tokens = logits.shape[0]
            num_experts = logits.shape[1]
            capacity = int(self.cap * num_tokens // num_experts)

            r = paddle.ones(num_tokens, "float32") / num_tokens
            c = paddle.ones(num_experts, "float32") / num_experts
            if self.training and (self.use_sinkhorn and self.config.moe_k == 1):
                assert (
                    token_type_ids is None
                ), "cannot use `token-type-ids` with sinkorn"
                pi, _ = compute_optimal_transport(
                    -logits.cast("float32"), r, c, lam=self.sinkhorn_temp
                )
            else:
                if token_type_ids is not None:
                    with paddle.no_grad():
                        assert (
                            token_type_ids.max().item() < logits.shape[-1]
                        ), f"`token-type-id`={token_type_ids} oov, num-expert={logits.shape[-1]}"
                        num_type = 2
                        scale = logits.shape[-1] // num_type
                        # [s] [0,1,2,3,4,5]
                        expert_id = paddle.arange(logits.shape[-1], dtype="int64")
                        # [s,e] [-2,-1,0,1,2,3] if token-type == 1
                        expert_id = expert_id.unsqueeze(0) - (
                            token_type_ids * scale
                        ).unsqueeze(-1)
                        mask = (expert_id < 0) | (expert_id >= scale)
                        logits[mask] = float("-inf")
                        # logger.info(f'using token-type-gate: {token_type_ids} {mask}  {logits}')
                if self.config.moe_k > 1:
                    scale = logits.shape[-1] // self.config.moe_k
                    # Reshape logits to combine the segments we want to apply argmax on
                    reshaped_logits = logits.reshape([-1, self.config.moe_k, scale])
                    # Apply argmax along the last axis (within each segment)
                    ma = paddle.argmax(reshaped_logits, axis=2)  # [s, k]
                    # Compute indices
                    offsets = paddle.arange(self.config.moe_k, dtype="int64") * scale
                    indices_s = ma + offsets.unsqueeze(0)
                    ma_one_hot = F.one_hot(ma, num_classes=scale).astype(paddle.int64)
                    mask1 = F.one_hot(indices_s, num_classes=num_experts).astype(
                        paddle.int64
                    )  # [s, k, e]
                    mask1 = mask1.reshape([-1, num_experts])  # [sk, e]
                else:
                    indices_s = paddle.argmax(logits, axis=1)
                    mask1 = F.one_hot(indices_s, num_classes=num_experts).cast(
                        paddle.int64
                    )  # [s, 1]

            l_zloss = self._cal_z_loss(logits.cast("float32"))

            if self.config.moe_k > 1:
                prob = F.softmax(
                    logits.reshape([-1, self.config.moe_k, scale]), axis=-1
                )
                l_aux = self._cal_aux_loss(prob, ma_one_hot) / self.config.moe_k
                prob = prob.reshape([-1, num_experts])
            else:
                prob = F.softmax(logits)
                l_aux = self._cal_aux_loss(prob, mask1)

            if self.window_size:
                mask1 = paddle.repeat_interleave(
                    mask1, repeats=self.window_size, axis=0
                )
                prob = paddle.repeat_interleave(prob, repeats=self.window_size, axis=0)
                capacity *= self.window_size

            locations1 = paddle.cumsum(mask1, axis=0) - 1  # [sk, e]
            mask1 *= paddle.cast(locations1 < capacity, dtype="int64")

            locations1_s = paddle.sum(locations1 * mask1, axis=1)  # [sk, 1]
            mask1_float = mask1.cast(paddle.float32)

            if self.config.moe_k > 1:
                locations1_s = locations1_s.reshape(shape=[-1, self.config.moe_k])
                mask1_float = mask1_float.reshape(
                    [-1, self.config.moe_k, num_experts]
                )  # [s, k ,e]
                scale = prob.shape[-1] // self.config.moe_k
                gates_s = (prob.unsqueeze(1) * mask1_float).sum(axis=-1)  # [s, k]
                gates_sum = paddle.sum(gates_s, axis=1, keepdim=True)  # [s, 1]
                gates = (gates_s / (gates_sum + 1e-8)).unsqueeze(
                    2
                ) * mask1_float  # [s, k, e]
                # Calculate combine_weight and scatter_index
                combine1_weight = paddle.max(gates, axis=-1)  # [s, k, 1]
                scatter1_index = (
                    paddle.argmax(gates, axis=-1) * capacity + locations1_s
                )  # [s, k]
            else:
                mask1_float = mask1.cast(paddle.float32)
                gates1 = mask1_float - prob.clone().detach() + prob

                expert1_index = paddle.argmax(gates1, -1)
                combine1_weight = paddle.max(gates1, -1, keepdim=True)
                scatter1_index = expert1_index * capacity + locations1_s

            dispatch1_mask = combine1_weight.cast(paddle.bool).detach()
            if self.enable_logging:
                global_training_logs.update(
                    **{
                        "top1_gate": (
                            combine1_weight.sum()
                            / (dispatch1_mask.cast("float32").sum() + 1e-9)
                        ).item(),
                        f"top1_gate_layer_{self.layer_idx}": (
                            combine1_weight.sum()
                            / (dispatch1_mask.cast("float32").sum() + 1e-9)
                        ).item(),
                    }
                )
                seqlen = logits.shape[0]
                top1_gate_experts_per_token = (
                    paddle.cast(dispatch1_mask, dtype="float32").sum() / seqlen
                )
                leakage_experts_per_token = (
                    paddle.cast(~dispatch1_mask, dtype="float32").sum() / seqlen
                )

                _log = {
                    f"experts_per_token_layer_{self.layer_idx}": top1_gate_experts_per_token.item(),
                    f"top1_experts_per_token_layer_{self.layer_idx}": top1_gate_experts_per_token.item(),
                    f"leakage_experts_per_token_layer_{self.layer_idx}": leakage_experts_per_token.item(),
                }
                global_training_logs.update(
                    **_log,
                    **{
                        k.replace(f"_layer_{self.layer_idx}", ""): v
                        for k, v in _log.items()
                    },
                )
            router_loss = (
                l_aux * self.config.moe_aux_loss_lambda
                + l_zloss * self.config.moe_z_loss_lambda
            )
            router_loss.stop_gradient = False
            if self.config.moe_k == 1:
                dispatch1_mask = dispatch1_mask.unsqueeze(0)
                combine1_weight = combine1_weight.cast(orig_dtype).unsqueeze(0)
                scatter1_index = scatter1_index.unsqueeze(0)
            else:
                combine1_weight = combine1_weight.cast(orig_dtype)
            return (
                capacity,
                dispatch1_mask,
                combine1_weight,
                scatter1_index,
                router_loss,
                logits,
            )


class SinkHornGateFused(SinkHornGate):

    def forward(
        self,
        input: Tensor,
        token_type_ids=None,
    ) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        """
        Args:
            input: paddle.Tensor, hidden-states of layer
        Retruns:
            paddle.Tensor [Seq, Expert, Capacity]: float32, combine weights
            paddle.Tensor [Seq, Expert, Capacity]: bool, dispatch mask
            Tuple[paddle.Tensor]: `GateOutput`
        """
        num_tokens = input.shape[0]
        # capacity = 2S/E
        capacity = int(self.cap * num_tokens // self.num_experts)

        with paddle.amp.auto_cast(False):
            input_32 = input.cast("float32")
            logits = F.linear(
                (1.0 - self.gate_detach) * input_32.detach()
                + self.gate_detach * input_32,
                self.weight,
            )  # [S,M] -> [S,E]
            if self.gate_scale:
                logits = logits / math.sqrt(self.weight.shape[0])  # cal temp

            if token_type_ids is not None:
                with paddle.no_grad():
                    assert (
                        token_type_ids.max().item() < logits.shape[-1]
                    ), f"`token-type-id`={token_type_ids} oov, num-expert={logits.shape[-1]}"
                    num_type = 2
                    scale = logits.shape[-1] // num_type
                    # [s] [0,1,2,3,4,5]
                    expert_id = paddle.arange(logits.shape[-1], dtype="int64")
                    # [s,e] [-2,-1,0,1,2,3] if token-type == 1
                    expert_id = expert_id.unsqueeze(0) - (
                        token_type_ids * scale
                    ).unsqueeze(-1)
                    mask = (expert_id < 0) | (expert_id >= scale)
                    logits[mask] = float("-inf")

            orthogonal_loss = self._cal_orthogonal_loss()

            router_loss = orthogonal_loss * self.config.moe_orthogonal_loss_lambda
            router_loss.stop_gradient = False
            if self.enable_logging:
                _log = {
                    # f"aux_loss_layer_{self.layer_idx}": l_aux.item(),
                    f"orthogonal_loss_layer_{self.layer_idx}": orthogonal_loss.item(),
                    # f"zloss_layer_{self.layer_idx}": l_zloss.item(),
                }
                global_training_logs.update(
                    **_log,
                    **{
                        k.replace(f"_layer_{self.layer_idx}", ""): v
                        for k, v in _log.items()
                    },
                )

        return logits, capacity, router_loss
