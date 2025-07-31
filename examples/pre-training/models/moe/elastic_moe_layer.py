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

"""_summary_

Returns:
    _type_: _description_
"""
from typing import Tuple
import logging
from collections import namedtuple
import inspect

import paddle
from paddle import framework
import paddle.nn.functional as F

from paddle.distributed.fleet.utils import recompute
import paddle.distributed as dist

from paddle import Tensor
from paddleformers.utils.tools import get_env_device

from models.moe.sinkhorn_gate import SinkHornGateFused
from models.moe.round_robin_gate import RoundRobinGateFused
from models.moe.top2_gate import TopKGateFused
from models.sequence_parallel_utils import ScatterOp
from models.utils import global_training_logs_enabled

from models.comm_utils import profile
from models.sequence_parallel_utils import (
    GatherOp,
)

try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}  # 没有erniebot的环境下无法打印 debug 量
try:
    import moe_router_loss_ops
except ImportError:
    moe_router_loss_ops = None
try:
    from paddle.distributed import in_auto_parallel_align_mode
except:

    def in_auto_parallel_align_mode():
        """
        hack for paddlenlp develop branch.
        """
        return False


logger = logging.getLogger(__name__)

if get_env_device() == "xpu":
    try:
        from paddle_xpu_nn import moe_gate_dispatch as xpu_moe_gate_dispatch
    except ImportError:
        xpu_moe_gate_dispatch = None
        logger.warning("`xpu moe dispatch` not found")
else:
    try:
        import moe_ops
    except ImportError:
        moe_ops = None
        logger.warning(
            "`moe-ops` not found, run "
            "`python3  src/ernie_core/ops/moe/setup.py  install` to install"
        )

GateOutput = namedtuple(
    "GateOutput",
    [
        "aux",
        "z",
        "logits",
    ],
)

from models.moe.moe_layer import (
    dispatching,
    AlltoAll,
    AlltoAllAsync,
    AlltoAllExpertOverlap,
    MOELayer,
    DeepEPMOELayer,
    Fp8FusedMoeFunc,
    recompute_fwd_gate_up_func,
)


class ElasticMOELayer(MOELayer):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)

        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (paddle.nn.Layer):
            gate network
        expert (paddle.nn.LayerList):
            expert network, LayerList 长度是 per_device 上的 expert 数。
        group (paddle.ProgressGroup)
        recompute: 启用MOE内recomupte
    Returns:
        output
        combine_weight
        router-loss
    """

    def __init__(self, *args, **kwargs):
        """
        初始化弹性MoE层。
        """
        super().__init__(*args, **kwargs)
        self.k = None

    def fused_gate_logits_process(
        self, gate_logits, token_type_ids, moe_k, offload_helper=None
    ):
        """处理并合并 gate logits

        Args:
            gate_logits (_type_): _description_
            token_type_ids (_type_): _description_

        Returns:
            _type_: _description_
        """
        k = 1 if isinstance(self.gate, SinkHornGateFused) else moe_k
        moe_num_experts = gate_logits.shape[-1]
        experts_type_ids = self.gate.experts_type_ids
        use_hard_gate = self.config.moe_use_hard_gate
        max_prob = None

        if token_type_ids is not None and use_hard_gate:
            if offload_helper is None:
                offload_helper = dict()
                lm_mask = token_type_ids == 0
                is_lm = lm_mask.any()
                mm_mask = token_type_ids == 1
                is_mm = mm_mask.any()
                seq_lm = lm_mask.sum()
                seq_mm = mm_mask.sum()
                lm_mask = lm_mask.unsqueeze(1) & (experts_type_ids == 0).unsqueeze(0)
                mm_mask = mm_mask.unsqueeze(1) & (experts_type_ids == 1).unsqueeze(0)
                offload_helper["lm_mask"] = [lm_mask, is_lm, seq_lm]
                offload_helper["mm_mask"] = [mm_mask, is_mm, seq_mm]

            is_lm = offload_helper["lm_mask"][1]
            prob = paddle.zeros_like(gate_logits)
            # 处理 lm_prob
            if is_lm:
                lm_mask = offload_helper["lm_mask"][0]
                seq_lm_cpu = offload_helper["lm_mask"][2]
                lm_mask_nonzero = lm_mask.nonzero()
                lm_partial_gate_logits = gate_logits.gather_nd(lm_mask_nonzero).reshape(
                    [seq_lm_cpu, -1]
                )
                if self.group_experts:
                    lm_prob = self.gate.act(
                        lm_partial_gate_logits.reshape(
                            [lm_partial_gate_logits.shape[0], k, -1]
                        )
                    )
                    max_prob = lm_prob.max(-1, keepdim=True)  # [s_l, k, 1]
                    lm_prob /= max_prob
                else:
                    lm_prob = self.gate.act(lm_partial_gate_logits)
                prob = paddle.scatter_nd_add(prob, lm_mask_nonzero, lm_prob.flatten())
            # 处理 mm_prob
            is_mm = offload_helper["mm_mask"][1]
            if is_mm:
                mm_mask = offload_helper["mm_mask"][0]
                seq_mm_cpu = offload_helper["mm_mask"][2]
                mm_mask_nonzero = paddle.nonzero(mm_mask)
                mm_partial_gate_logits = gate_logits.gather_nd(mm_mask_nonzero).reshape(
                    [seq_mm_cpu, -1]
                )
                mm_prob = self.gate.act(mm_partial_gate_logits)
                prob = paddle.scatter_nd_add(prob, mm_mask_nonzero, mm_prob.flatten())
        else:
            # 处理非硬门和不需要token_type_ids的情况
            if self.group_experts:
                prob = self.gate.act(gate_logits.reshape([gate_logits.shape[0], k, -1]))
                max_prob = prob.max(-1, keepdim=True)
                prob /= max_prob
                prob = prob.reshape([prob.shape[0], -1])
            else:
                prob = self.gate.act(gate_logits)
        return prob, max_prob

    def gate_and_distpach(
        self,
        input,
        token_type_ids,
        moe_k,
        moe_capacity,
        new_expert_num=None,
        global_gate_mask=None,
        new_expert_rank_list=None,
        is_diff_topk=False,
        is_diff_expert_num=False,
    ):
        """
        calc gate and dispatch inputs (and do logging, optionaly)
        Args:
            input: Tensor[seq, dim], float
            token_type_ids: Tensor[seq], int
        Returns:
            dispatched_input: Tensor[num_experts, capacity, dim]
            combine_weights: [seq, k]
            scatter_index: [seq, k]
            router_loss: scalar
            gate_logits: [seq, num_experts]
        """
        seqlen, d_model = input.shape
        args = ()
        # 目前只有 `SinkHornGate` aka Top1 gate 支持输入 token type ids
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape([-1])
            args = (token_type_ids,)

        use_fuse = isinstance(
            self.gate, (RoundRobinGateFused, SinkHornGateFused, TopKGateFused)
        )
        if use_fuse:
            (
                gate_logits,
                capacity,
                router_loss,
            ) = self.gate(
                input,
                cap_factor=moe_capacity,
                new_expert_num=new_expert_num,
                global_gate_mask=global_gate_mask,
                *args,
            )
        else:
            (
                capacity,
                dispatch_mask,
                combine_weights,
                scatter_index,
                router_loss,
                gate_logits,
            ) = self.gate(
                input,
                cap_factor=moe_capacity,
                new_expert_num=new_expert_num,
                global_gate_mask=global_gate_mask,
                *args,
                correction_bias=(
                    self.moe_statics.e_score_correction_bias[0]
                    if self.use_correction_bias
                    else None
                ),
            )
            prob = None
        if self.input_preprocess is not None:
            input, gate_logits = self.input_preprocess(input, gate_logits, capacity)
        if use_fuse:
            # capacity no use
            k = 1 if isinstance(self.gate, SinkHornGateFused) else moe_k
            prob, max_prob = self.fused_gate_logits_process(
                gate_logits, token_type_ids, moe_k
            )
            if get_env_device() == "xpu":
                assert xpu_moe_gate_dispatch is not None
                (
                    dispatched_input,
                    combine_weights_unnorm,
                    scatter_index,
                    dispatch_mask,
                    _,
                ) = xpu_moe_gate_dispatch(input, prob, k, capacity, True)
            else:
                assert moe_ops is not None
                with profile("dispatch_op"):
                    if (
                        "corr_bias"
                        in inspect.signature(moe_ops.moe_gate_dispatch).parameters
                    ):
                        if self.use_correction_bias and not is_diff_expert_num:
                            compat_args = (self.moe_statics.e_score_correction_bias[0],)
                        else:
                            compat_args = (None,)
                    else:
                        assert (
                            not self.use_correction_bias
                        ), "correction bias not supported, rebuild moe-ops"
                        compat_args = ()
                    if not self.config.use_ep_comm_overlap:
                        if self._rr_moe_gate_dispatch is None:
                            (
                                dispatched_input,
                                combine_weights_unnorm,
                                scatter_index,
                                dispatch_mask,
                                _,
                            ) = moe_ops.moe_gate_dispatch(
                                input,
                                prob,
                                *compat_args,
                                k=k,
                                capacity=capacity,
                                use_pad=True,
                            )
                        else:
                            (
                                dispatched_input,
                                combine_weights_unnorm,
                                scatter_index,
                                dispatch_mask,
                                _,
                            ) = self._rr_moe_gate_dispatch(
                                input,
                                prob,
                                compat_args,
                                k=k,
                                capacity=capacity,
                                use_pad=True,
                            )
                    else:
                        (
                            dispatched_input,
                            combine_weights_unnorm,
                            scatter_index,
                            dispatch_mask,
                            _,
                        ) = moe_ops.moe_gate_dispatch_permute(
                            input,
                            prob,
                            *compat_args,
                            k=k,
                            capacity=capacity,
                            world_size=self.group.nranks,
                        )
            dispatch_mask = paddle.diff(F.pad(dispatch_mask, (1, 0)))
            if self.use_correction_bias and not (is_diff_topk or is_diff_expert_num):
                if self.gate.config.multimodel_experts:
                    for i in range(len(self.moe_statics.expert_usage)):
                        self.moe_statics.expert_usage[i] += dispatch_mask[
                            self.gate.experts_type_mask[i]
                        ].detach()
                else:
                    self.moe_statics.expert_usage[0] += dispatch_mask.detach()
            dispatched_input.stop_gradient = False
            combine_weights_unnorm.stop_gradient = False
            scatter_index.stop_gradient = True
            dispatch_mask.stop_gradient = True

            scatter_index = scatter_index.transpose([1, 0])  # [k,s] ->[s,k]
            if self.group_experts:
                if max_prob is not None:
                    if token_type_ids is not None:
                        p = paddle.ones_like(combine_weights_unnorm.unsqueeze(-1))
                        p = paddle.scatter_nd_add(
                            p, paddle.nonzero(token_type_ids == 0), -1 + max_prob
                        )
                    else:
                        p = max_prob
                    combine_weights_unnorm = (
                        combine_weights_unnorm.unsqueeze(-1) * p
                    ).squeeze(-1)
                    # gate_prob 进行还原
                    prob = (prob.reshape([p.shape[0], k, -1]) * p).reshape(
                        [p.shape[0], -1]
                    )
            if self.gate.norm_gate_logits:
                combine_weights = combine_weights_unnorm / paddle.clip(
                    combine_weights_unnorm.sum(-1, keepdim=True), min=1e-12
                )
            else:
                combine_weights = combine_weights_unnorm
            combine_weights = combine_weights.cast(dispatched_input.dtype)
        else:
            dispatched_input = dispatching(
                input,
                dispatch_mask,
                scatter_index,
                num_experts=self.world_size * self.num_local_experts,
                capacity=capacity,
            )
            if self.use_correction_bias and not (is_diff_topk or is_diff_expert_num):
                usage = paddle.bincount(
                    scatter_index.reshape([-1]) // capacity,
                    minlength=self.world_size * self.num_local_experts,
                )
                assert (
                    not self.config.multimodel_experts
                ), "correction bias not supported, use top2-fused gate"
                self.moe_statics.expert_usage[0] += usage.detach()
        # clone 保平安
        if not self.config.use_ep_comm_overlap:
            dispatched_input = dispatched_input.reshape(
                [self.world_size * self.num_local_experts, capacity, d_model]
            )  # .clone()
        else:
            assert (
                len(dispatched_input.shape) == 4
                and dispatched_input.shape[1] == self.world_size
                and dispatched_input.shape[0] == self.num_local_experts
            ), (
                f"When using ep_comm_overlap, moe_gate_dispatch_permute is needed. "
                f"Expected dispatched_input to have shape[1] == {self.world_size} "
                f"and shape[0] == {self.num_local_experts}, "
                f"but got shape {dispatched_input.shape}"
            )
            dispatched_input = dispatched_input  # .clone()
        dispatch_mask.stop_gradient = True
        scatter_index.stop_gradient = True
        return (
            dispatched_input,
            combine_weights,
            dispatch_mask,
            scatter_index,
            router_loss,
            gate_logits,
            prob,
        )

    def _calc_router_loss(
        self,
        dispatch_mask,
        gate_logits,
        gate_prob,
        num_experts,
        use_group,
        layer_idx,
        token_type=None,
        tokens_type_mask=None,
        dispatch_tokens_mask=None,
        prefix="",
        moe_k=None,
    ):
        log = {}
        router_loss, l_aux, orthogonal_loss, zloss = 0.0, None, None, None
        if self.gate.config.moe_aux_loss_lambda:
            l_aux = self.gate._cal_aux_loss(
                gate_prob,
                dispatch_mask,
                num_experts,
                use_group,
                tokens_type_mask,
                dispatch_tokens_mask,
                moe_k=moe_k,
            )
            router_loss += self.gate.moe_aux_loss_lambda[token_type or 0] * l_aux
        else:
            router_loss += (
                self.zero * gate_prob[0, 0]
            )  # must use gate prob to avoid zero pointer
        if self.gate.config.moe_orthogonal_loss_lambda:
            orthogonal_loss = self.gate._cal_orthogonal_loss(token_type, use_group)
            router_loss += (
                self.gate.moe_orthogonal_loss_lambda[token_type or 0] * orthogonal_loss
            )
        if self.gate.config.moe_z_loss_lambda and not in_auto_parallel_align_mode():
            zloss = self.gate._cal_z_loss(gate_logits, tokens_type_mask)
            router_loss += self.gate.moe_z_loss_lambda[token_type or 0] * zloss

        tracer = framework._dygraph_tracer()
        if self.enable_logging and global_training_logs_enabled() and tracer._has_grad:
            if l_aux is not None:
                log[f"aux_loss_layer_{self.layer_idx}"] = l_aux

            if orthogonal_loss is not None:
                log[f"orthogonal_loss_layer_{self.layer_idx}"] = orthogonal_loss

            if zloss is not None:
                log[f"zloss_layer_{self.layer_idx}"] = zloss

            global_training_logs.update(
                **log,
                **{
                    k.replace(f"_layer_{self.layer_idx}", ""): v for k, v in log.items()
                },
            )
            global_training_logs.update(
                **{
                    prefix + "_" + k.replace(f"_layer_{self.layer_idx}", ""): v
                    for k, v in log.items()
                }
            )
        return router_loss

    def calc_router_loss_and_logging(
        self,
        router_loss,
        combine_weights,
        dispatch_mask,
        gate_logits,
        gate_prob,
        token_type_ids,
        moe_k=None,
        dispatch_token_type_ids=None,
        offload_helper=None,
    ):
        """
        在fused expert 的情况下，计算辅助 loss (Aux-loss, 正交 loss, z-loss) 并 打印 log
        """
        use_fuse = isinstance(
            self.gate, (RoundRobinGateFused, SinkHornGateFused, TopKGateFused)
        )
        if use_fuse:
            assert gate_prob is not None
            if token_type_ids is not None and self.gate.config.moe_use_hard_gate:
                if not self.gate.weight.stop_gradient:
                    # 文参数训练时才计算。
                    lm_tokens_mask = token_type_ids == 0
                    if offload_helper is not None:
                        is_lm = offload_helper["lm_mask"][1]
                    else:
                        is_lm = lm_tokens_mask.any()
                    if is_lm:
                        dispatch_tokens_mask = (
                            dispatch_token_type_ids == 0
                            if dispatch_token_type_ids is not None
                            else None
                        )
                        router_loss += self._calc_router_loss(
                            (
                                dispatch_mask[self.gate.experts_type_mask[0]]
                                if hasattr(self.gate, "experts_type_mask")
                                else dispatch_mask
                            ),
                            (
                                gate_logits[:, self.gate.experts_type_mask[0]]
                                if hasattr(self.gate, "experts_type_mask")
                                else gate_logits
                            ),
                            (
                                gate_prob[:, self.gate.experts_type_mask[0]]
                                if hasattr(self.gate, "experts_type_mask")
                                else gate_prob
                            ),
                            (
                                self.gate.num_experts_list[0]
                                if hasattr(self.gate, "num_experts_list")
                                else self.gate.num_experts_tensor
                            ),
                            self.group_experts,
                            self.layer_idx,
                            0,
                            lm_tokens_mask,
                            dispatch_tokens_mask,
                            prefix="lm",
                            moe_k=moe_k,
                        )
                mm_tokens_mask = token_type_ids == 1
                if offload_helper is not None:
                    is_mm = offload_helper["mm_mask"][1]
                else:
                    is_mm = mm_tokens_mask.any()
                if is_mm:
                    dispatch_tokens_mask = (
                        dispatch_token_type_ids == 1
                        if dispatch_token_type_ids is not None
                        else None
                    )
                    router_loss += self._calc_router_loss(
                        dispatch_mask[self.gate.experts_type_mask[1]],
                        gate_logits[:, self.gate.experts_type_mask[1]],
                        gate_prob[:, self.gate.experts_type_mask[1]],
                        self.gate.num_experts_list[1],
                        False,
                        self.layer_idx,
                        1,
                        mm_tokens_mask,
                        dispatch_tokens_mask,
                        prefix="mm",
                        moe_k=moe_k,
                    )

            else:
                router_loss += self._calc_router_loss(
                    dispatch_mask,
                    gate_logits,
                    gate_prob,
                    self.gate.num_experts_tensor,
                    self.group_experts,
                    self.layer_idx,
                    moe_k=moe_k,
                )

            if self.enable_logging and global_training_logs_enabled():
                seqlen = gate_logits.shape[0]
                num_active = paddle.count_nonzero(combine_weights)
                gate_experts_per_token = num_active.item() / seqlen
                cap_factor = (
                    self.gate.cap[0]
                    if isinstance(self.gate.cap, (tuple, list))
                    else self.gate.cap
                )
                capacity = (
                    cap_factor * combine_weights.shape[0] // gate_logits.shape[-1]
                )
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.reshape([-1])
                    combine_weights_type_0 = combine_weights[token_type_ids == 0]
                    if combine_weights_type_0.size:
                        gate_expert_per_token_type_0 = (
                            paddle.count_nonzero(combine_weights_type_0).item()
                            / combine_weights_type_0.shape[0]
                        )
                        global_training_logs.update(
                            experts_per_token_text=gate_expert_per_token_type_0,
                        )

                    combine_weights_type_1 = combine_weights[token_type_ids == 1]
                    if combine_weights_type_1.size:
                        gate_expert_per_token_type_1 = (
                            paddle.count_nonzero(combine_weights_type_1).item()
                            / combine_weights_type_1.shape[0]
                        )
                        global_training_logs.update(
                            experts_per_token_image=gate_expert_per_token_type_1,
                        )

                ce = (
                    (-F.softmax(gate_logits, -1) * F.log_softmax(gate_logits, -1))
                    .sum(-1)
                    .mean(0)
                )
                _log = {
                    f"gate_prob_ce_layer_{self.layer_idx}": ce.item(),
                    f"experts_per_token_layer_{self.layer_idx}": gate_experts_per_token,
                }
                global_training_logs.update(
                    **_log,
                    **{
                        k.replace(f"_layer_{self.layer_idx}", ""): v
                        for k, v in _log.items()
                    },
                )
        else:
            seqlen = dispatch_mask.shape[0]
            dispatch_mask = dispatch_mask.unbind(-1)
            top1_gate_experts_per_token = (
                paddle.cast(dispatch_mask[0], dtype="float32").sum() / seqlen
            )
            if (
                self.enable_logging
                and global_training_logs_enabled()
                and len(dispatch_mask) == 2
            ):
                top2_gate_experts_per_token = (
                    paddle.cast(dispatch_mask[1], dtype="float32").sum() / seqlen
                )
                leakage_experts_per_token = (
                    paddle.cast(
                        (~dispatch_mask[0]) & (~dispatch_mask[1]), dtype="float32"
                    ).sum()
                    / seqlen
                )
                experts_per_token = (
                    top1_gate_experts_per_token + top2_gate_experts_per_token
                )
                global_training_logs.update(
                    experts_per_token=experts_per_token.detach(),
                    top1_experts_per_token=top1_gate_experts_per_token.detach(),
                    top2_experts_per_token=top2_gate_experts_per_token.detach(),
                    leakage_experts_per_token=leakage_experts_per_token.detach(),
                )
            elif (
                self.enable_logging
                and global_training_logs_enabled()
                and len(dispatch_mask) == 1
            ):
                experts_per_token = top1_gate_experts_per_token
                leakage_experts_per_token = (
                    paddle.cast(~dispatch_mask[0], dtype="float32").sum() / seqlen
                )
                global_training_logs.update(
                    experts_per_token=experts_per_token.detach(),
                    top1_experts_per_token=top1_gate_experts_per_token.detach(),
                    leakage_experts_per_token=leakage_experts_per_token.detach(),
                )

        return router_loss

    def forward(
        self,
        input: Tensor,
        token_type_ids=None,
        moe_k=None,
        moe_capacity=None,
        new_expert_num=None,
        global_gate_mask=None,
        new_expert_rank_list=None,
        is_diff_topk=False,
        is_diff_expert_num=False,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        Args:
            input (`Tensor`): The input data with shape ``(s, d)``.
                Only one token is supported for now.
            token_type_ids (`Tensor`) int64 tensor with shape (s),
                if specified, rount tensor according to `token_type_ids`.
        Returns:
            output (`Tensor`): The final output tensor with shape ``(s, d)`` where ``m`` is the
                size of model parameters.
            combine_weights (`Tensor`, optional): A tensor with shape ``(s,)``, which represents weights
                for each expert in MoE.
            router_loss (`Tensor`, optional): A scalar tensor representing the loss of routing function.
        """
        # assert len(input) == 1, "only single input Tensor supported"
        if input.ndim == 3:
            orig_shape = input.shape
            # clone 保平安
            input = input.reshape([-1, input.shape[-1]])
        else:
            orig_shape = None
        assert (
            len(input.shape) == 2
        ), f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"
        if token_type_ids is not None:
            token_type_ids = token_type_ids.clone()[:, :-1]
            if self.config.sequence_parallel:
                token_type_ids = token_type_ids.reshape([-1])
                token_type_ids = ScatterOp.apply(token_type_ids)
                token_type_ids.stop_gradient = True

        assert self.gate is not None
        if hasattr(self, "rng") and self.rng.random() < self.all_to_all_dropout:
            orig_shape_2 = input.shape
            output = self.forward_experts(input)
            output += self.gate.weight.sum() * 0.0  # hack for grad
            output = output.reshape(orig_shape or orig_shape_2)  # [e*1,c,m]
            return output, None, 0

        is_first_fwd = not framework._dygraph_tracer()._has_grad
        use_async = self.shared_experts is not None
        if in_auto_parallel_align_mode():
            gate_input = paddle.assign(input)
        else:
            gate_input = input

        with profile("fused_gate_and_dispatch"):
            (
                dispatched_input,
                combine_weights,
                dispatch_mask,
                scatter_index,
                router_loss,
                gate_logits,
                gate_prob,
            ) = self.gate_and_distpach(
                gate_input,
                token_type_ids,
                moe_k,
                moe_capacity,
                new_expert_num=new_expert_num,
                global_gate_mask=global_gate_mask,
                new_expert_rank_list=new_expert_rank_list,
                is_diff_topk=is_diff_topk,
                is_diff_expert_num=is_diff_expert_num,
            )

        if not self.config.use_ep_comm_overlap:
            with profile("moe_comm_and_shared_expert"):
                if use_async:
                    dispatched_input, shared_out = AlltoAllAsync.apply(
                        dispatched_input,
                        input,  # args to shared-experts
                        group=self.group,
                        fn=self.shared_experts,
                        is_first_fwd=is_first_fwd,
                    )
                else:
                    dispatched_input = AlltoAll.apply(dispatched_input, self.group)

            expert_out = (
                recompute(self.forward_experts, dispatched_input)
                if self.recompute and self.training
                else self.forward_experts(dispatched_input)
            )
        else:
            assert (
                len(dispatched_input.shape) == 4
                and dispatched_input.shape[1] == self.world_size
                and dispatched_input.shape[0] == self.num_local_experts
            ), (
                f"When using ep_comm_overlap, moe_gate_dispatch_permute is needed. "
                f"Expected dispatched_input to have shape[1] == {self.world_size} "
                f"and shape[0] == {self.num_local_experts}, "
                f"but got shape {dispatched_input.shape}"
            )
            with profile("moe_comm_and_forward_expert"):
                expert_out = AlltoAllExpertOverlap.apply(
                    dispatched_input,
                    self.group,
                    self.num_local_experts,
                    self.experts,
                    is_first_fwd=is_first_fwd,
                )

        with profile("moe_comm_and_calc_routerloss"):
            expert_out, router_loss2 = AlltoAllAsync.apply(
                expert_out,
                router_loss,
                combine_weights,
                dispatch_mask,
                gate_logits,
                gate_prob,
                token_type_ids,
                paddle.to_tensor(moe_k),
                group=self.group,
                fn=self.calc_router_loss_and_logging,
                is_first_fwd=is_first_fwd,
            )

        with profile("combine"):
            combined_output = self.combine_expert_output(
                expert_out, combine_weights, scatter_index
            )

        if self.shared_experts is not None:
            combined_output += shared_out

        if orig_shape:
            combined_output = combined_output.clone().reshape(
                orig_shape[:-1] + [combined_output.shape[-1]]
            )
        return combined_output, combine_weights, router_loss2, gate_logits


class DeepEPElasticMOELayer(DeepEPMOELayer):
    """
    DeepEPElasticMOELayer
    """

    def forward(
        self,
        hidden_states,
        input_ids,
        random_k_list,
        max_topk,
        output_gate_logits=True,
        global_gate_mask_list=None,
        is_diff_expert_num_list=None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        Args:
            hidden_states: Tensor[(B*S)/tp, H]
            random_k_list: list[B]
        Returns:
            combined_output: Tensor[(B*S)/tp, H]
            topk_normed_probs: None
            router_loss2: Tensor[1]
            gate_logits: Tensor[(B*S)/tp, E]
        """
        assert (
            len(hidden_states.shape) == 2
        ), f"input Tensor must have dimensions: (s)equence, (d)im, got:{hidden_states.shape}"
        assert (
            int(self.all_to_all_dropout) == 0
        ), "all_to_all_dropout is not supported yet."
        assert self.gate is not None
        assert self.config.sequence_parallel
        assert (
            max_topk <= 32
        ), f"max_topk {max_topk} exceeds 32, which is not supported yet."
        assert self.config.aux_loss_type != "seq_aux_loss"

        # for debug
        # random_k_list = [self.config.moe_k for _ in range(len(random_k_list))]

        bsz = len(random_k_list)
        seq_length = self.config.seqlen

        if self.config.sequence_parallel or self.config.submatrix_parallel:
            # SP对BSZ维度进行了切分，需要先合并，来支持Elastic从Bsz维度gather
            # [(B*S)/tp, H] -> [B*S, H] -> [B, S, H] -> [B, S/tp, H] -> [B*(S/tp), H]
            hidden_states = GatherOp.apply(hidden_states).reshape(
                [-1, seq_length, hidden_states.shape[-1]]
            )
            hidden_states = ScatterOp.apply(hidden_states, axis=1).reshape(
                [-1, hidden_states.shape[-1]]
            )

            input_ids = GatherOp.apply(input_ids).reshape([-1, seq_length])
            input_ids = ScatterOp.apply(input_ids, axis=1).reshape([-1])

        # topk ins
        hidden_states_list = hidden_states.reshape(
            [bsz, -1, hidden_states.shape[-1]]
        ).unbind(0)
        input_ids_list = input_ids.reshape([bsz, -1]).unbind(0)
        global_gate_mask_list = (
            global_gate_mask_list
            if global_gate_mask_list is not None
            else [None for _ in range(bsz)]
        )
        is_diff_expert_num_list = (
            is_diff_expert_num_list
            if is_diff_expert_num_list is not None
            else [False for _ in range(bsz)]
        )

        # topk outs
        padded_topk_normed_probs_list = []
        padded_topk_indices_list = []
        router_loss2_list = []
        no_elastic_summary, elastic_topk_summary, elastic_expert_summary = (
            paddle.zeros(self.global_num_experts, dtype=paddle.int64),
            paddle.zeros(self.global_num_experts, dtype=paddle.int64),
            paddle.zeros(self.global_num_experts, dtype=paddle.int64),
        )
        no_elastic_num, elastic_topk_num, elastic_expert_num = (
            paddle.zeros(1, dtype=paddle.int64),
            paddle.zeros(1, dtype=paddle.int64),
            paddle.zeros(1, dtype=paddle.int64),
        )

        for (
            moe_k,
            global_gate_mask,
            is_diff_expert_num,
            hidden_states_,
            input_ids_,
        ) in zip(
            random_k_list,
            global_gate_mask_list,
            is_diff_expert_num_list,
            hidden_states_list,
            input_ids_list,
        ):
            with profile("gate"):
                gate_logits, gate_probs, probs_for_choice, router_loss = (
                    self.gate_score(
                        hidden_states_,
                        global_gate_mask=global_gate_mask,
                        is_diff_expert_num=is_diff_expert_num,
                    )
                )
            if moe_k != self.config.moe_k:
                is_diff_topk = True
            else:
                is_diff_topk = False

            topk_probs, topk_indices, routing_map, dispatch_mask = self.topk(
                moe_k,
                gate_logits,
                gate_probs,
                probs_for_choice,
                is_diff_topk=is_diff_topk,
                is_diff_expert_num=is_diff_expert_num,
                input_ids=input_ids_,
            )

            if not (is_diff_topk or is_diff_expert_num):
                no_elastic_summary += dispatch_mask
                no_elastic_num += 1
            if is_diff_topk:
                elastic_topk_summary += dispatch_mask
                elastic_topk_num += 1
            if is_diff_expert_num:
                elastic_expert_summary += dispatch_mask
                elastic_expert_num += 1

            padded_topk_normed_probs, padded_topk_indices = self.pad_for_elastic(
                max_topk, topk_probs, topk_indices
            )

            with profile("calc_router_loss_and_logging"):
                router_loss2_ = self.calc_router_loss_and_logging(
                    router_loss,
                    gate_logits=gate_logits,
                    gate_probs=gate_probs,
                    routing_map=routing_map,
                    dispatch_mask=dispatch_mask,
                    input_ids=input_ids_,
                    is_diff_topk=is_diff_topk,
                    is_diff_expert_num=is_diff_expert_num,
                )

            padded_topk_normed_probs_list.append(padded_topk_normed_probs)
            padded_topk_indices_list.append(padded_topk_indices)
            router_loss2_list.append(router_loss2_)

        # calc and log summary
        is_first_fwd = framework._dygraph_tracer()._has_grad
        if self.enable_logging and global_training_logs_enabled() and is_first_fwd:
            self.moe_tokens_per_experts_indicator(
                "no_elastic", no_elastic_summary, no_elastic_num
            )
            self.moe_tokens_per_experts_indicator(
                "elastic_topk", elastic_topk_summary, elastic_topk_num
            )
            self.moe_tokens_per_experts_indicator(
                "elastic_expert", elastic_expert_summary, elastic_expert_num
            )

        # B * [S/tp, E] -> [B*(S/tp), E]
        topk_normed_probs = paddle.concat(padded_topk_normed_probs_list, axis=0)
        topk_indices = paddle.concat(padded_topk_indices_list, axis=0)
        router_loss2 = paddle.mean(paddle.stack(router_loss2_list), axis=0)
        topk_indices.stop_gradient = True

        if self.config.use_fp8_fuse_node:
            with profile("dispatch"):
                (
                    dispatched_hidden_states,
                    dispatched_indices,
                    dispatched_probs,
                    _,
                ) = self.dispatcher._comm_manager.dispatch(
                    hidden_states, topk_indices, topk_normed_probs
                )

            with profile("fusion_mlp"):
                hidden_states_tmp = Fp8FusedMoeFunc.apply(
                    dispatched_hidden_states,
                    dispatched_probs,
                    dispatched_indices,
                    self,
                    max_topk,
                    recompute_fwd_gate_up=recompute_fwd_gate_up_func(
                        self.config, self.layer_idx
                    ),
                    dequant_input=("dequant_input" in self.config.fp8_mem_configs)
                    and self.config.fp8_mem_configs["dequant_input"],
                    is_first_fwd=not framework._dygraph_tracer()._has_grad,
                )

            if self.shared_experts is not None and self.config.use_ep_comm_overlap:
                combine_overlap_handle = {
                    "fn": self.shared_experts,
                    "fn_args": (hidden_states,),
                }
            else:
                combine_overlap_handle = None

            with profile("combine"):
                combined_output = self.dispatcher._comm_manager.combine(
                    hidden_states_tmp, inner_layer_overlap_handle=combine_overlap_handle
                )
        else:
            with profile("permutation"):
                (
                    dispatched_input,
                    token_permuted_indices,
                    prob_permuted_indices,
                    dispatched_probs,
                ) = self.dispatcher.token_permutation(
                    hidden_states, topk_indices, topk_normed_probs, max_topk
                )

            with profile("forward_experts"):
                expert_out = self.forward_experts(dispatched_input)

            if self.shared_experts is not None and self.config.use_ep_comm_overlap:
                combine_overlap_handle = {
                    "fn": self.shared_experts,
                    "fn_args": (hidden_states,),
                }
            else:
                combine_overlap_handle = None

            if self.enable_logging and global_training_logs_enabled() and is_first_fwd:
                num_tokens_per_expert = self.get_num_tokens_per_expert()
                total_num_tokens_per_expert = paddle.to_tensor(
                    sum(num_tokens_per_expert)
                )
                total_list = paddle.empty(
                    [self.group.world_size], dtype=total_num_tokens_per_expert.dtype
                )
                dist.stream.all_gather(
                    total_list, total_num_tokens_per_expert, group=self.group
                )
                self.calc_and_log_moe_summary("local_tokens_per_card", total_list)

            with profile("unpermutation"):
                combined_output = self.dispatcher.token_unpermutation(
                    expert_out,
                    token_permuted_indices,
                    prob_permuted_indices,
                    dispatched_probs,
                    inner_layer_overlap_handle=combine_overlap_handle,
                )

        if self.shared_experts is not None:
            if self.config.use_ep_comm_overlap:
                shared_out = combine_overlap_handle["fn_out"][0]
            else:
                shared_out = self.shared_experts(hidden_states)
            combined_output += shared_out

        if self.config.sequence_parallel or self.config.submatrix_parallel:
            # [B*(S/tp), H] -> [B, S/tp, H] -> [B, S, H] -> [B*S, H] -> [(B*S)/tp, H]
            combined_output = GatherOp.apply(
                combined_output.reshape([bsz, -1, combined_output.shape[-1]]), axis=1
            )
            combined_output = ScatterOp.apply(
                combined_output.reshape([-1, combined_output.shape[-1]])
            )

            if output_gate_logits:
                gate_logits = GatherOp.apply(
                    gate_logits.reshape([bsz, -1, gate_logits.shape[-1]]), axis=1
                )
                gate_logits = ScatterOp.apply(
                    gate_logits.reshape([-1, gate_logits.shape[-1]])
                )
            else:
                gate_logits = None

        return combined_output, None, router_loss2, gate_logits
