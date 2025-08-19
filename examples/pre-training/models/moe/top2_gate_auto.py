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
from functools import partial
import logging
import numpy as np
import paddle
from paddle import Tensor
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.utils import unique_name
from paddle.distributed import fleet
from models.moe.moe_utils_auto import get_mesh, get_flatten_mesh

try:
    from custom_setup_ops import matmul_bwd
except ImportError:
    matmul_bwd = None


logger = logging.getLogger(__name__)


def cal_aux_loss_func(
    gate_prob,
    dispatch_mask,
    tokens_mask,
    dispatch_tokens_mask,
    num_experts,
    use_group,
    moe_k,
    global_aux_loss=False,
    rank=None,
    group=None,
):
    if tokens_mask is not None and tokens_mask.dtype != gate_prob.dtype:
        tokens_mask = tokens_mask.astype(gate_prob.dtype)

    scale = None
    if dispatch_tokens_mask is not None:
        seqlen_float = dispatch_tokens_mask.astype(gate_prob.dtype).sum()
        if (
            tokens_mask is not None
            and gate_prob.shape[0] != dispatch_tokens_mask.shape[0]
        ):
            scale = seqlen_float / paddle.clip(tokens_mask.sum(), min=1e-6)
    elif tokens_mask is not None:
        seqlen_float = tokens_mask.sum()
    else:
        seqlen_float = gate_prob.numel().astype(gate_prob.dtype) / num_experts
    seqlen_float = paddle.clip(seqlen_float, min=1e-6)

    if len(dispatch_mask.shape) == 2:
        dispatch_mask = dispatch_mask.sum(0)
    ce = dispatch_mask.astype(gate_prob.dtype).detach() / seqlen_float
    me = paddle.sum(gate_prob, axis=0) / seqlen_float
    if global_aux_loss:
        me_list, ce_list = [], []
        dist.all_gather(me_list, me, group=group)
        dist.all_gather(ce_list, ce, group=group)

        me_list[rank] = me
        ce_list[rank] = ce
        me = paddle.stack(me_list).mean(0)
        ce = paddle.stack(ce_list).mean(0)

    l_aux = paddle.sum(me * ce) * num_experts
    if use_group:
        l_aux = l_aux / moe_k

    if scale is not None:
        l_aux = l_aux + (scale - 1) * l_aux.detach()

    return l_aux


def cast_if_needed(x, dtype):

    return x.cast(dtype) if x.dtype != dtype else x


class FusedGateDetachMatmul(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, x, w):

        ctx.dtype = paddle.float32
        ctx.save_for_backward(x, w)
        return F.linear(cast_if_needed(x, ctx.dtype), cast_if_needed(w, ctx.dtype))

    @staticmethod
    def backward(ctx, y_grad):

        x, w = ctx.saved_tensor()
        assert ctx.dtype == y_grad.dtype, "dtype not match"
        x_g, w_g = matmul_bwd(
            cast_if_needed(x, ctx.dtype),
            cast_if_needed(w, ctx.dtype),
            y_grad,
            False,
            False,
        )
        return cast_if_needed(x_g, x.dtype), cast_if_needed(w_g, w.dtype)


def gate_detach_matmul(x, weight, use_fuse, use_fake_gate=False):

    if use_fuse:
        score = FusedGateDetachMatmul.apply(x, weight)
    else:
        x = cast_if_needed(x, paddle.float32)
        score = F.linear(x, weight)

    if use_fake_gate:
        score = paddle.randn(score.shape).astype(score.dtype) + score - score
    return score


class Top2Gate(nn.Layer):

    def __init__(self, config, layer_idx: int, group, gate_weight=None) -> None:

        super().__init__()
        self.config = config
        self.fuse_gate_detach_matmul = config.fuse_gate_detach_matmul
        if self.fuse_gate_detach_matmul:
            assert matmul_bwd is not None, "matmul_bwd is not supported"

        self.use_fake_gate = config.use_fake_gate
        if self.use_fake_gate:
            logging.warning(
                "You are use fake_gate, which is just for test, not for real training."
            )

        self.model_dim = config.hidden_size
        self.num_experts = config.moe_num_experts
        self.num_experts_tensor = (
            sum(config.moe_num_experts)
            if config.multimodel_experts
            else config.moe_num_experts
        )

        self.cap = config.moe_capacity
        self.group = group

        self.layer_idx = layer_idx
        self.global_aux_loss = config.global_aux_loss
        if self.global_aux_loss:
            self.rank = dist.get_rank(self.group)

        self.sinkhorn_2gate = config.sinkhorn_2gate
        self.sinkhorn_temp = config.sinkhorn_temp
        self.use_token_type_bias = config.moe_use_token_type_bias
        self.use_correction_bias = config.moe_use_aux_free

        if config.moe_gate_act == "softmax":
            self.act = partial(F.softmax, axis=-1)
        elif config.moe_gate_act == "sigmoid":
            self.act = F.sigmoid
        else:
            raise ValueError(f"{config.moe_gate_act} is not supported.")
        self.no_jitter = True
        self.expert_drop = False
        self.eye_matrix = None
        self.eye_matrix_size = None
        self.norm_gate_logits = config.moe_norm_gate_logits
        self.one = paddle.ones([], dtype="float32")

        self.moe_aux_loss_lambda = paddle.to_tensor(
            config.moe_aux_loss_lambda, dtype="float32"
        )

        self.moe_orthogonal_loss_lambda = paddle.to_tensor(
            config.moe_orthogonal_loss_lambda, dtype="float32"
        )
        if self.moe_aux_loss_lambda.ndim == 0:
            self.moe_aux_loss_lambda = self.moe_aux_loss_lambda.unsqueeze(0)

        if self.moe_orthogonal_loss_lambda.ndim == 0:
            self.moe_orthogonal_loss_lambda = self.moe_orthogonal_loss_lambda.unsqueeze(
                0
            )

        self.experts_type_ids = None
        if config.moe_orthogonal_loss_lambda:
            if hasattr(fleet.fleet, "_user_defined_strategy"):
                strategy = fleet.fleet._user_defined_strategy
                sharding_configs = strategy.hybrid_configs["sharding_configs"]
                pp_config = strategy.hybrid_configs["pp_configs"]
                assert (
                    not sharding_configs.comm_overlap
                    and not pp_config.sharding_comm_overlap
                ), "orthogonal loss will cause twice gradient accumulate, will break pp/sharding overlap"

        self.eps = paddle.to_tensor([1e-12], dtype="float32")
        if config.multimodel_experts:
            if config.moe_use_hard_gate:
                self.num_experts_list = []
                self.experts_type_mask = []
                experts_ids = paddle.zeros(
                    [sum(self.num_experts)], dtype="int64"
                ).reshape([config.moe_world_size, -1])
                offset = 0
                for i, expert_num in enumerate(self.num_experts):
                    experts_ids[
                        :, offset : offset + expert_num // config.moe_world_size
                    ] = i
                    offset += expert_num // config.moe_world_size
                self.experts_type_ids = experts_ids.reshape([-1])
                logger.info(
                    f"use moe_use_hard_gate, experts_ids: {self.experts_type_ids}"
                )
                for i, expert_num in enumerate(self.num_experts):
                    self.experts_type_mask.append(
                        self.experts_type_ids == i,
                    )
                    self.num_experts_list.append(expert_num)
            else:
                assert (
                    not config.moe_group_experts
                ), "group_experts must use hard_gate when multimodel_experts is True"
        else:
            self.num_experts_list = [self.num_experts]
        if gate_weight is not None:
            self.weight = gate_weight
            assert (
                not self.config.moe_use_token_type_bias
            ), "gate_weights is from outside, token_type_bias can't be used"
            logger.info("moe use gate_weight from outside")
            self._cast_to_low_precision = False
            self._cast_to_low_precison = False
        else:
            self._create_gate_parameter()
        logger.info(
            f"{config.moe_gate}: w/ capacity: {self.cap} experts:{self.num_experts} "
            f"use_token_type_bias:{self.use_token_type_bias} gate_act:{config.moe_gate_act} "
            f"norm_gate_logits={self.norm_gate_logits} use_correction_bias={self.use_correction_bias}"
        )

    def _create_gate_parameter(self):

        if self.config.multimodel_experts:

            self.moe_aux_loss_lambda = self.moe_aux_loss_lambda.expand(
                len(self.num_experts)
            )
            self.moe_orthogonal_loss_lambda = self.moe_orthogonal_loss_lambda.expand(
                len(self.num_experts)
            )

            for i, num_experts in enumerate(self.num_experts):
                if i == 1:
                    with paddle.utils.unique_name.guard(f"mm_gate_{self.layer_idx}_"):
                        p = self.create_parameter(
                            shape=[self.model_dim, num_experts],
                            dtype="float32",
                            attr=paddle.ParamAttr(
                                name=unique_name.generate("moe_gate")
                            ),
                        )
                else:
                    p = self.create_parameter(
                        shape=[self.model_dim, num_experts],
                        dtype="float32",
                        attr=paddle.ParamAttr(name=unique_name.generate("moe_gate")),
                    )
                p.expert_type = f"expert_type_{i}"
                self.add_parameter(
                    ("weight" if i == 0 else f"weight_{i}"),
                    p,
                )
        else:
            self.weight = self.create_parameter(
                shape=[self.model_dim, self.num_experts],
                dtype="float32",
                attr=paddle.ParamAttr(name=unique_name.generate("moe_gate")),
            )
            logger.info(f"moe-Gate, {self.weight}")

        if self.use_token_type_bias:
            if self.config.multimodel_experts:
                assert (
                    not self.config.moe_use_hard_gate
                ), "multimodel_experts with hard_gate is not support token_type_bias."
            num_experts = (
                sum(self.num_experts)
                if self.config.multimodel_experts
                else self.num_experts
            )
            bias_type_num = (
                len(self.num_experts) if self.config.multimodel_experts else 1
            )
            self.bias = self.create_parameter(
                shape=[bias_type_num, num_experts],
                dtype="float32",
                attr=paddle.ParamAttr(
                    name=unique_name.generate("moe_gate_bias"),
                    initializer=paddle.nn.initializer.Assign(
                        np.zeros([bias_type_num, num_experts])
                    ),
                ),
            )
            logger.info(f"using token type bias, bias: {self.bias},")
        self._cast_to_low_precision = False
        self._cast_to_low_precison = False

    def get_gate_weight(self, transform_weight):
        if not self.config.multimodel_experts:
            return self.weight
        if not transform_weight:
            return paddle.concat(
                [
                    getattr(self, "weight" if i == 0 else f"weight_{i}")
                    for i in range(len(self.num_experts))
                ],
                -1,
            )
        weight = paddle.zeros(
            [
                self.model_dim,
                self.config.moe_world_size,
                sum(self.num_experts) // self.config.moe_world_size,
            ],
            dtype="float32",
        )
        offset = 0
        for i, num_experts in enumerate(self.num_experts):
            weight[
                :, :, offset : offset + num_experts // self.config.moe_world_size
            ] = getattr(self, "weight" if i == 0 else f"weight_{i}").reshape(
                [self.model_dim, self.config.moe_world_size, -1]
            )
            offset += num_experts // self.config.moe_world_size
        weight = weight.reshape([self.model_dim, -1])

        return weight

    def forward(
        self,
        input: Tensor,
        token_type_ids: Tensor = None,
        transform_weight: bool = True,
        correction_bias: Tensor = None,
    ):
        pass

    def get_capacity(self, num_tokens, cap_factor=None):

        num_experts = (
            sum(self.num_experts)
            if self.config.multimodel_experts
            else self.num_experts
        )
        if cap_factor is not None:
            cap = cap_factor
        else:
            if self.training:
                cap = self.cap[0]
            elif num_tokens < num_experts:
                cap = self.cap[2]
            else:
                cap = self.cap[1]
        capacity = int(cap * num_tokens // num_experts)
        assert (
            capacity > 0
        ), f"requires capacity to >= 0. cap={cap}, num_tokens={num_tokens}"
        return capacity

    def _cal_aux_loss(
        self,
        gate_prob,
        dispatch_mask,
        num_experts=None,
        use_group=None,
        tokens_mask=None,
        dispatch_tokens_mask=None,
    ):

        if self.act is F.sigmoid:
            gate_prob = gate_prob / gate_prob.sum(-1, keepdim=True)

        if self.use_correction_bias:
            if tokens_mask is not None:
                gate_prob_this_modality = gate_prob[tokens_mask.astype("bool")]
                if gate_prob_this_modality.shape[0]:
                    _, top_idx = gate_prob_this_modality.topk(
                        k=self.config.moe_k, axis=-1
                    )
                    mask = paddle.zeros_like(gate_prob_this_modality).put_along_axis(
                        top_idx, paddle.to_tensor(1.0), axis=1
                    )
                    dispatch_mask = paddle.sum(mask.cast(paddle.int64), axis=0)
                else:
                    dispatch_mask = paddle.zeros(gate_prob.shape[-1], dtype="int64")
                dist.stream.all_reduce(
                    dispatch_mask,
                    group=self.group,
                    use_calc_stream=True,
                )
            else:
                _, top_idx = gate_prob.topk(k=self.config.moe_k, axis=-1)

                mask = paddle.zeros_like(gate_prob).put_along_axis(
                    top_idx, paddle.to_tensor(1.0), axis=1
                )
                dispatch_mask = paddle.sum(mask.cast(paddle.int64), axis=0)

        if num_experts is None:
            num_experts = self.num_experts_tensor
        if use_group is None:
            use_group = self.config.moe_group_experts

        return cal_aux_loss_func(
            gate_prob,
            dispatch_mask,
            tokens_mask,
            dispatch_tokens_mask,
            num_experts,
            use_group,
            self.config.moe_k,
            self.global_aux_loss,
            self.rank if self.global_aux_loss else None,
            self.group if self.global_aux_loss else None,
        )


class TopKGateFused(Top2Gate):

    def forward(
        self,
        input: Tensor,
        token_type_ids=None,
        transform_weight=True,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        capacity = self.get_capacity(input.shape[0])
        weight = self.get_gate_weight(transform_weight)
        with paddle.amp.auto_cast(False):

            logits = gate_detach_matmul(
                input, weight, self.fuse_gate_detach_matmul, self.use_fake_gate
            )
            if self.use_token_type_bias:
                assert token_type_ids is not None
                assert (
                    token_type_ids.max() < self.bias.shape[0]
                ), f"token_type_ids {token_type_ids.max()} >= bias shape {self.bias.shape[0]}"
                bias = self.bias[token_type_ids]
                logits = logits + bias
            router_loss = paddle.zeros([1], dtype="float32")
            router_loss.stop_gradient = False

        return logits, capacity, router_loss


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
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
        elif input.shape[0] < num_experts:
            cap = self.cap[2]
        else:
            cap = self.cap[1]
        num_tokens = input.shape[0]
        global_capacity = int(cap * num_tokens // num_experts)
        local_num_tokens = input._local_shape[0]
        local_capacity = int(cap * local_num_tokens // num_experts)

        logits, _, router_loss = super().forward(input, token_type_ids)

        return logits, global_capacity, router_loss, local_capacity
