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
import paddle
from paddle import Tensor
import paddle.distributed as dist
import paddle.nn.functional as F
from paddleformers.utils.tools import get_env_device
from models.utils import global_training_logs_enabled

try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}  # 没有erniebot的环境下无法打印 debug 量
try:
    import moe_router_loss_ops
except ImportError:
    moe_router_loss_ops = None

try:
    from custom_setup_ops import matmul_bwd
except ImportError:
    matmul_bwd = None

try:
    from bincount_ops import int_bincount
except ImportError:
    int_bincount = None

logger = logging.getLogger(__name__)

from models.moe.top2_gate import (
    TopKGateFused,
    gate_detach_matmul,
    CalAuxLossFunctor,
    cal_aux_loss_func,
)


class ElasticTopKGateFused(TopKGateFused):
    """doc"""

    def forward(
        self,
        input: Tensor,
        cap_factor,
        token_type_ids=None,
        transform_weight=True,
        new_expert_num=None,
        global_gate_mask=None,
    ) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        """
        Args:
            input: paddle.Tensor, hidden-states of layer
            token_type_ids: paddle.Tensor[Seqw], token_type_ids of input
            transform_weight: bool, when using multimodal experts, perform `self.get_gate_weight` if specified
        Retruns:
            paddle.Tensor [Seq, Expert, Capacity]: float32, combine weights
            paddle.Tensor [Seq, Expert, Capacity]: bool, dispatch mask
            Tuple[paddle.Tensor]: `GateOutput`
        """
        capacity = self.get_capacity(input.shape[0], cap_factor=cap_factor)
        weight = self.get_gate_weight(transform_weight)
        with paddle.amp.auto_cast(False):
            if get_env_device() == "xpu" and self.xpu_matmul is not None:
                assert not self.fuse_gate_detach_matmul, "not supported on XPU"
                input_32 = input.cast("float32")
                logits = self.xpu_matmul(
                    input_32,
                    weight,
                    training=self.training,
                )
            else:
                logits = gate_detach_matmul(input, weight, self.fuse_gate_detach_matmul)
            if self.use_token_type_bias:
                assert token_type_ids is not None
                assert (
                    token_type_ids.max() < self.bias.shape[0]
                ), f"token_type_ids {token_type_ids.max()} >= bias shape {self.bias.shape[0]}"
                bias = self.bias[token_type_ids]  # [seq]
                logits = logits + bias
            if global_gate_mask is not None:
                logits = logits + global_gate_mask
            orthogonal_loss = None
            # 正交 loss 拿到 moe-layer 里去计算
            router_loss = paddle.zeros([1], dtype="float32")
            router_loss.stop_gradient = False
            if (
                self.enable_logging
                and global_training_logs_enabled()
                and orthogonal_loss is not None
            ):
                _log = {
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

    def _cal_aux_loss(
        self,
        gate_prob,
        dispatch_mask,
        num_experts=None,
        use_group=None,
        tokens_mask=None,
        dispatch_tokens_mask=None,
        moe_k=None,
        is_diff_topk=False,
        is_diff_expert_num=False,
    ):
        """
        计算辅助损失

        Args:
            gate_prob (paddle.Tensor[local_seq, num_experts]):
            dispatch_mask (paddle.Tensor[num_experts]): 每个 expert 被分配的 token 数（不考虑 token drop)
            tokens_mask (paddle.Tensor[Seq]): 每个 MP 内 token-type-id
            dispatch_tokens_mask (paddle.Tensor): AllGather 后的`tokens_mask`
        Returns:
            paddle.Tensor: 辅助损失值。

        """
        if self.act is F.sigmoid:
            gate_prob = gate_prob / gate_prob.sum(-1, keepdim=True)

        if self.use_correction_bias:
            if tokens_mask is not None:
                gate_prob_this_modality = gate_prob[tokens_mask.astype("bool")]
                if gate_prob_this_modality.shape[0]:
                    _, top_idx = gate_prob_this_modality.topk(k=moe_k.item(), axis=-1)
                    if int_bincount is not None:
                        dispatch_mask = int_bincount(
                            top_idx, 0, gate_prob.shape[-1], paddle.int64
                        )
                    else:
                        mask = paddle.zeros_like(
                            gate_prob_this_modality
                        ).put_along_axis(top_idx, paddle.to_tensor(1.0), axis=1)
                        dispatch_mask = paddle.sum(mask.cast(paddle.int64), axis=0)
                else:
                    dispatch_mask = paddle.zeros(gate_prob.shape[-1], dtype="int64")
                dist.stream.all_reduce(
                    dispatch_mask,
                    group=self.group,
                    use_calc_stream=True,
                )
            else:
                _, top_idx = gate_prob.topk(k=moe_k.item(), axis=-1)
                if int_bincount is not None:
                    dispatch_mask = int_bincount(
                        top_idx, 0, gate_prob.shape[-1], paddle.int64
                    )
                else:
                    mask = paddle.zeros_like(gate_prob).put_along_axis(
                        top_idx, paddle.to_tensor(1.0), axis=1
                    )
                    dispatch_mask = paddle.sum(mask.cast(paddle.int64), axis=0)

        if num_experts is None:
            num_experts = self.num_experts_tensor
        if use_group is None:
            use_group = self.config.moe_group_experts

        if (
            moe_router_loss_ops is not None
            and get_env_device() != "xpu"
            and (tokens_mask is None or len(tokens_mask.shape) == 1)
            and (tokens_mask is None or tokens_mask.shape[0] == gate_prob.shape[0])
            and (gate_prob.shape[0] >= gate_prob.shape[1])
            and (not self.global_aux_loss)
            and (gate_prob.dtype == paddle.float32)
        ):
            return CalAuxLossFunctor.apply(
                gate_prob,
                dispatch_mask,
                tokens_mask,
                dispatch_tokens_mask,
                num_experts,
                use_group,
                moe_k.item(),
                clip_min=1e-6,
            )
        else:
            return cal_aux_loss_func(
                gate_prob,
                dispatch_mask,
                tokens_mask,
                dispatch_tokens_mask,
                num_experts,
                use_group,
                moe_k.item(),
                self.global_aux_loss,
                self.rank if self.global_aux_loss else None,
                self.group if self.global_aux_loss else None,
            )
