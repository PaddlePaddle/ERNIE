# -*- coding: utf-8 -*-
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
elastic_moe_all_gather_layer.py
"""
from typing import Tuple
import logging
import contextlib
import numpy as np
import inspect

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle import framework
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute

from models.moe.sinkhorn_gate import SinkHornGateFused
from models.moe.round_robin_gate import RoundRobinGateFused
from models.moe.top2_gate import TopKGateFused
from models.sequence_parallel_utils import (
    AllGatherOp,  # 进入异步区，tp间的操作可以不一样
    ReduceScatterOp,
    ScatterOp,
    get_async_loader,
    hack_offload_wait,
)
from models.utils import global_training_logs_enabled
from paddle.incubate.tensor.manipulation import async_offload

from .moe_layer import fuse_logging
from paddleformers.utils.tools import get_env_device

try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}  # 没有erniebot的环境下无法打印 debug 量
try:
    import moe_router_loss_ops
except ImportError:
    moe_router_loss_ops = None


def profile(_):
    """dumy profile"""
    return contextlib.nullcontext()


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
    try:
        import moe_ops_partial
    except ImportError:
        moe_ops_partial = None
        logger.warning(
            "`moe-ops-partial` not found, run "
            "`python3  src/ernie_core/ops/moe/setup.py  install` to install"
        )
    try:
        import moe_ops_partial_nosoftmaxtopk
    except ImportError:
        moe_ops_partial_nosoftmaxtopk = None
        logger.warning(
            "`moe-ops-partial-nosoftmaxtopk` not found, run "
            "`python3  src/ernie_core/ops/moe/setup.py  install` to install"
        )

    try:
        import moe_utils
    except ImportError:
        moe_utils = None
        logger.warning(
            "`moe_utils` not found, run "
            "`python3  src/ernie_core/ops/moe/setup.py  install` to install"
        )

from models.moe.moe_all_gather_layer import (
    MOEAllGatherLayerV2,
    AllGatherAsync,
    ReshardCombineWeight,
    allgather_async,
    AlltoAllSmart,
)

from models.moe.elastic_moe_layer import in_auto_parallel_align_mode


class ElasticMOEAllGatherLayerV2(MOEAllGatherLayerV2):
    """弹性激活ElasticMOEAllGatherLayerV2"""

    def __init__(self, *args, **kwargs):
        """
        初始化弹性MoE层。
        """
        super().__init__(*args, **kwargs)
        self.k = None

    def fused_gate_and_dispatch(
        self,
        input,
        token_type_ids,
        global_dense_expert_mask,
        moe_k,
        moe_capacity,
        new_expert_num=None,
        global_gate_mask=None,
        new_expert_rank_list=None,
        is_diff_topk=False,
        is_diff_expert_num=False,
    ):
        """_summary_

        Args:
            input (_type_): _description_
            token_type_ids (_type_): _description_
            global_dense_expert_mask: used in audio experts

        Returns:
            _type_: _description_
        """
        seqlen, d_model = input.shape
        args = ()
        # 目前只有 `SinkHornGate` aka Top1 gate 支持输入 token type ids
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape([-1])
            args = (token_type_ids,)

        router_loss = paddle.zeros([1], dtype="float32")
        router_loss.stop_gradient = False
        top_k = 1 if isinstance(self.gate, SinkHornGateFused) else moe_k

        def build_weights_and_expert_id(input):
            nonlocal token_type_ids, args
            logits, capacity, router_loss = self.gate(
                input,
                moe_capacity,
                *args,
                transform_weight=False,
                new_expert_num=new_expert_num,
                global_gate_mask=global_gate_mask,
            )
            if self.config.multimodel_experts:
                gate_logits_lm, gate_logits_mm = logits.chunk(2, axis=-1)
            else:
                gate_logits_lm, gate_logits_mm = logits, None

            weigth_and_expert, gate_prob_lm, gate_prob_mm = (
                self.fused_gate_logits_process_fused(
                    gate_logits_lm,
                    gate_logits_mm,
                    (
                        token_type_ids if global_dense_expert_mask is None else None
                    ),  # HARD code: assume dense-expert == 纯音
                    moe_k=moe_k,
                    is_diff_topk=is_diff_topk,
                    is_diff_expert_num=is_diff_expert_num,
                )
            )
            weigth_and_expert = AllGatherOp.apply(
                weigth_and_expert, group=self.config.moe_group
            )
            return (
                weigth_and_expert,
                gate_logits_lm,
                gate_logits_mm,
                gate_prob_lm,
                gate_prob_mm,
            )

        capacity = (
            self.gate.get_capacity(input.shape[0], cap_factor=moe_capacity)
            * self.world_size
        )
        # global_hidden_states = AllGatherOp.apply(input, group=self.config.moe_group)
        (
            global_hidden_states,
            combine_weights_and_expert_id,
            gate_logits_lm,
            gate_logits_mm,
            gate_prob_lm,
            gate_prob_mm,
        ) = AllGatherAsync.apply(
            input,
            input,
            fn=build_weights_and_expert_id,
            group=self.config.moe_group,
            is_first_fwd=not framework._dygraph_tracer()._has_grad,
        )
        combine_weights_unnorm, expert_id = combine_weights_and_expert_id.chunk(
            2, axis=-1
        )
        expert_id = expert_id.cast("int32")
        expert_id.stop_gradient = True
        num_experts = (
            sum(self.config.moe_num_experts)
            if isinstance(self.config.moe_num_experts, (tuple, list))
            else self.config.moe_num_experts
        )  # all-experts = 96
        if global_dense_expert_mask is not None:
            combine_weights_unnorm[global_dense_expert_mask] = 0.0
            expert_id[global_dense_expert_mask] = num_experts
            num_experts += 1

        with profile("dispatch_op"):
            if (
                "reverse_token_drop"
                in inspect.signature(
                    moe_ops_partial_nosoftmaxtopk.moe_gate_dispatch_partial_nosoftmaxtopk
                ).parameters
            ):
                compat_kwargs = {"reverse_token_drop": self.enable_reverse_token_drop}
            else:
                compat_kwargs = {}

            (
                dispatched_input,
                combine_weights_unnorm,
                scatter_index,  # input -> dispatched_input
                scatter_index_rev,  # dispatch-input -> input
                expert_num_global,  # global 不考虑截断！！所有rank下的expert的使用数量
                expert_num_local,  # 当前rank下的expert的使用数量,其余为0
            ) = moe_ops_partial_nosoftmaxtopk.moe_gate_dispatch_partial_nosoftmaxtopk(
                global_hidden_states,
                combine_weights_unnorm,
                expert_id,
                top_k,
                capacity,
                num_experts,
                self.use_padding,
                expert_start_index=self.num_local_experts * self.config.moe_rank,
                expert_end_index=self.num_local_experts * (self.config.moe_rank + 1),
                **compat_kwargs,
            )

        if self.use_correction_bias and not (is_diff_topk or is_diff_expert_num):
            if self.gate.config.multimodel_experts:
                for i in range(len(self.moe_statics.expert_usage)):
                    self.moe_statics.expert_usage[i] += expert_num_local[
                        self.gate.experts_type_mask[i]
                    ].detach()
            else:
                self.moe_statics.expert_usage[0] += expert_num_local.detach()

        if (
            scatter_index_rev.ndim == 0
        ):  # unpad 时, `moe_ops_partial` 中的空输出用 `scatter_index_rev==[]`表示。
            assert not self.use_padding
            scatter_index_rev = paddle.empty([0], dtype=scatter_index_rev.dtype)

        dispatched_input.stop_gradient = False
        combine_weights_unnorm.stop_gradient = False
        scatter_index.stop_gradient = True
        expert_num_global.stop_gradient = True
        expert_num_global_notrunc = expert_num_global
        self.capacity_tensor = paddle.to_tensor(capacity, dtype=expert_num_global.dtype)
        expert_num_global = paddle.minimum(expert_num_global, self.capacity_tensor)

        if global_dense_expert_mask is not None:  # 去掉dense expert
            expert_num_global = expert_num_global[:-1]
            expert_num_local = expert_num_local[:-1]
            expert_num_global_notrunc = expert_num_global_notrunc[:-1]

        scatter_index = scatter_index.transpose([1, 0])  # [k,s] ->[s,k]

        last_local_expert = self.num_local_experts * self.config.moe_rank
        expert_offset_global = expert_num_global.cumsum()
        if get_env_device() == "xpu":
            expert_num_global_list = expert_num_global.to("cpu")
            offload_task = None
        else:
            loader = get_async_loader()
            expert_num_global_list, offload_task = async_offload(
                expert_num_global, loader
            )
        if self.use_padding:
            offset = last_local_expert * capacity
        else:
            offset = (
                expert_offset_global[last_local_expert - 1]
                if self.config.moe_rank > 0
                else 0
            )
        local_combine_weights_unnorm = ReshardCombineWeight.apply(
            combine_weights_unnorm.contiguous(), group=self.config.moe_group
        )
        local_scatter_index = ReduceScatterOp.apply(
            paddle.where(
                combine_weights_unnorm > 0.0,
                scatter_index + offset,
                scatter_index,
            ),
            group=self.config.moe_group,
        )
        if self.gate.norm_gate_logits:
            local_combine_weights = local_combine_weights_unnorm / paddle.clip(
                local_combine_weights_unnorm.sum(-1, keepdim=True), min=1e-12
            )
        else:
            local_combine_weights = local_combine_weights_unnorm
        local_combine_weights = local_combine_weights.cast(dispatched_input.dtype)
        if self.use_padding:
            dispatched_input = dispatched_input.reshape(
                [self.num_local_experts, -1, d_model]
            )
            dispatched_input = dispatched_input.unbind(0)
        else:
            s = self.num_local_experts * self.config.moe_rank
            e = self.num_local_experts * (self.config.moe_rank + 1)
            expert_num_local = expert_num_local.tolist()[s:e]
            expert_num_local_valid = [i for i in expert_num_local if i > 0]
            valid_pos = [j for j, i in enumerate(expert_num_local) if i > 0]
            if expert_num_local_valid:
                dispatched_input_list = dispatched_input.split(expert_num_local_valid)
                dispatched_input = [None] * len(expert_num_local)
                for p, t in zip(valid_pos, dispatched_input_list):
                    dispatched_input[p] = t
            else:
                dispatched_input = [dispatched_input] + (
                    [None] * (len(expert_num_local) - 1)
                )

        scatter_index.stop_gradient = True
        scatter_index_rev.stop_gradient = True
        if offload_task is not None:
            hack_offload_wait(offload_task)
        expert_num_global_list = expert_num_global_list.tolist()

        # logger.info(f'global-expert-usage:{expert_len}')
        # logger.info(f'scatter_index_rev: {self.num_local_experts} {capacity}')
        return (
            dispatched_input,
            global_hidden_states,
            local_combine_weights,
            expert_num_global_notrunc,  # 不考虑截断，为了计算auxloss
            expert_num_global,
            expert_num_global_list,
            local_scatter_index,
            scatter_index_rev,
            router_loss,
            (gate_logits_lm, gate_prob_lm),
            (gate_logits_mm, gate_prob_mm),
            expert_num_local,
        )

    def fused_gate_logits_process_fused(
        self,
        gate_logits_lm,
        gate_logits_mm,
        token_type_ids,
        moe_k,
        is_diff_topk=False,
        is_diff_expert_num=False,
    ):
        """process gatelogits w/ moe utils"""
        top_k = 1 if isinstance(self.gate, SinkHornGateFused) else moe_k
        num_expert_per_rank_per_modality = (
            gate_logits_lm.shape[-1] // self.config.moe_world_size
        )
        group_size = gate_logits_lm.shape[-1] // top_k
        if self.group_experts:
            assert not self.use_correction_bias
            gate_logits_lm = gate_logits_lm.reshape(
                [gate_logits_lm.shape[0], top_k, -1]
            )
            prob_lm = self.gate.act(gate_logits_lm)
            prob_lm_ = prob_lm
            weight_lm, expert_id_lm = prob_lm_.topk(k=1, axis=-1)
            weight_lm = weight_lm.reshape([gate_logits_lm.shape[0], -1])
            group_size = gate_logits_lm.shape[-1]
            expert_id_lm = expert_id_lm.squeeze(-1)
        else:
            prob_lm = self.gate.act(gate_logits_lm)
            if self.use_correction_bias and not is_diff_expert_num:
                prob_lm_ = (
                    prob_lm + self.moe_statics.e_score_correction_bias[0].detach()
                )
            else:
                prob_lm_ = prob_lm
            weight_lm, expert_id_lm = prob_lm_.topk(k=top_k, axis=-1)

        if self.use_correction_bias and not is_diff_expert_num:
            batch_idx = (
                paddle.arange(prob_lm_.shape[0]).unsqueeze(-1).expand_as(expert_id_lm)
            )
            weight_lm = prob_lm[batch_idx, expert_id_lm]  # use correct bias

        # num_expert_per_modality == 0 时只执行 group-expert expand，不执行 multimodal-expand
        expert_id_lm = moe_utils.expand_modality_expert_id(
            expert_id_lm,
            num_expert_per_modality=(
                num_expert_per_rank_per_modality if token_type_ids is not None else 0
            ),
            group_size=group_size,
            modality_offset=0,
            is_group_expert=self.group_experts,
        )
        expert_id_lm = expert_id_lm.reshape(weight_lm.shape)
        lm_weight_and_expert_id = paddle.concat(
            [weight_lm, expert_id_lm.astype("float32")], -1
        )
        if token_type_ids is None or gate_logits_mm is None:
            return (
                lm_weight_and_expert_id,
                prob_lm.reshape([prob_lm.shape[0], -1]),
                None,
            )

        prob_mm = self.gate.act(gate_logits_mm)
        if self.use_correction_bias and not is_diff_expert_num:
            prob_mm_ = prob_mm + self.moe_statics.e_score_correction_bias[1].detach()
        else:
            prob_mm_ = prob_mm
        weight_mm, expert_id_mm = prob_mm_.topk(k=top_k, axis=-1)
        if self.use_correction_bias and not is_diff_expert_num:
            batch_idx = (
                paddle.arange(prob_lm_.shape[0]).unsqueeze(-1).expand_as(expert_id_lm)
            )
            weight_mm = prob_mm[batch_idx, expert_id_mm]  # use correct bias

        expert_id_mm = moe_utils.expand_modality_expert_id(
            expert_id_mm,
            num_expert_per_modality=num_expert_per_rank_per_modality,
            group_size=group_size,
            modality_offset=1,
            is_group_expert=False,
        )
        expert_id_mm = expert_id_mm.reshape(weight_mm.shape)
        mm_weight_and_expert_id = paddle.concat(
            [weight_mm, expert_id_mm.astype("float32")], -1
        )
        weight_and_expert = paddle.where(
            (token_type_ids == 0).unsqueeze(-1),
            lm_weight_and_expert_id,
            mm_weight_and_expert_id,
        )
        return weight_and_expert, prob_lm.reshape([prob_lm.shape[0], -1]), prob_mm

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
        is_diff_topk=False,
        is_diff_expert_num=False,
    ):
        log = {}
        router_loss, l_aux, orthogonal_loss, zloss = 0.0, None, None, None
        if self.gate.config.moe_aux_loss_lambda and not (
            is_diff_topk or is_diff_expert_num
        ):
            l_aux = self.gate._cal_aux_loss(
                gate_prob,
                dispatch_mask,
                num_experts,
                use_group,
                tokens_type_mask,
                dispatch_tokens_mask,
                moe_k=moe_k,
                is_diff_topk=is_diff_topk,
                is_diff_expert_num=is_diff_expert_num,
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
        gate_logits,
        gate_prob,
        gate_logits_mm,
        gate_prob_mm,
        combine_weights,
        dispatch_mask,
        token_type_ids,
        dispatch_token_type_ids,
        moe_k,
        is_diff_topk=False,
        is_diff_expert_num=False,
    ):
        """
        分不同模态 gate prob(mm/lm) 进行 aux_loss 计算。
        """
        top_k = 1 if isinstance(self.gate, SinkHornGateFused) else moe_k
        dispatch_mask_3d = dispatch_mask.reshape([self.config.moe_world_size, -1])
        if token_type_ids is not None and self.gate.config.moe_use_hard_gate:
            if not self.gate.weight.stop_gradient:
                # 文参数训练时才计算。
                dispatch_tokens_mask = (
                    dispatch_token_type_ids == 0
                    if dispatch_token_type_ids is not None
                    else None
                )
                lm_tokens_mask = (token_type_ids == 0).astype(gate_prob.dtype)
                # hard code
                lm_experts = (
                    self.gate.num_experts[0]
                    if isinstance(self.gate.num_experts, (tuple, list))
                    else self.gate.num_experts
                )
                dispatch_mask_lm = dispatch_mask_3d[
                    :, : lm_experts // self.config.moe_world_size
                ].reshape([-1])
                router_loss += self._calc_router_loss(
                    dispatch_mask_lm,
                    gate_logits * lm_tokens_mask.unsqueeze(-1),
                    gate_prob * lm_tokens_mask.unsqueeze(-1),
                    self.gate.num_experts_list[0],
                    self.group_experts,
                    self.layer_idx,
                    0,  # ortholoss
                    lm_tokens_mask,
                    dispatch_tokens_mask,
                    prefix="lm",
                    moe_k=moe_k,
                    is_diff_topk=is_diff_topk,
                    is_diff_expert_num=is_diff_expert_num,
                )
            else:
                router_loss += self.zero * gate_logits[0, 0] * gate_prob[0, 0]
            if gate_prob_mm is not None:
                mm_tokens_mask = (token_type_ids == 1).astype(gate_prob_mm.dtype)
                dispatch_tokens_mask = (
                    dispatch_token_type_ids == 1
                    if dispatch_token_type_ids is not None
                    else None
                )
                dispatch_mask_mm = dispatch_mask_3d[
                    :, self.gate.num_experts[0] // self.config.moe_world_size :
                ].reshape([-1])

                router_loss += self._calc_router_loss(
                    dispatch_mask_mm,
                    gate_logits_mm * mm_tokens_mask.unsqueeze(-1),
                    gate_prob_mm * mm_tokens_mask.unsqueeze(-1),
                    self.gate.num_experts_list[1],
                    False,
                    self.layer_idx,
                    1,
                    mm_tokens_mask,
                    dispatch_tokens_mask,
                    prefix="mm",
                    moe_k=moe_k,
                    is_diff_topk=is_diff_topk,
                    is_diff_expert_num=is_diff_expert_num,
                )

        else:
            router_loss += self._calc_router_loss(
                dispatch_mask,
                gate_logits,
                gate_prob,
                self.gate.num_experts_tensor,
                self.group_experts,
                self.layer_idx,
                0,
                paddle.ones([gate_prob.shape[0]], "bool"),  # 半gate prob
                paddle.ones(
                    [self.gate.config.moe_world_size * gate_prob.shape[0]], "bool"
                ),
                prefix="lm",
                moe_k=moe_k,
                is_diff_topk=is_diff_topk,
                is_diff_expert_num=is_diff_expert_num,
            )

        tracer = framework._dygraph_tracer()
        if self.enable_logging and global_training_logs_enabled() and tracer._has_grad:
            if moe_router_loss_ops is not None and get_env_device() != "xpu":
                (
                    gate_expert_per_token_type_0,
                    gate_expert_per_token_type_1,
                    gate_experts_per_token,
                    ce,
                ) = fuse_logging(gate_logits, combine_weights, token_type_ids)

                if token_type_ids is not None:
                    global_training_logs.update(
                        experts_per_token_text=gate_expert_per_token_type_0,
                    )
                    global_training_logs.update(
                        experts_per_token_image=gate_expert_per_token_type_1,
                    )

            else:
                seqlen = gate_logits.shape[0]
                num_active = paddle.count_nonzero(combine_weights)
                gate_experts_per_token = num_active / seqlen
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.reshape([-1])
                    combine_weights_type_0 = combine_weights[token_type_ids == 0]
                    if combine_weights_type_0.size:
                        gate_expert_per_token_type_0 = (
                            paddle.count_nonzero(combine_weights_type_0)
                            / combine_weights_type_0.shape[0]
                        )
                        global_training_logs.update(
                            experts_per_token_text=gate_expert_per_token_type_0,
                        )

                    combine_weights_type_1 = combine_weights[token_type_ids == 1]
                    if combine_weights_type_1.size:
                        gate_expert_per_token_type_1 = (
                            paddle.count_nonzero(combine_weights_type_1)
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
                f"gate_prob_ce_layer_{self.layer_idx}": ce,
                f"experts_per_token_layer_{self.layer_idx}": gate_experts_per_token,
            }
            global_training_logs.update(
                **_log,
                **{
                    k.replace(f"_layer_{self.layer_idx}", ""): v
                    for k, v in _log.items()
                },
            )
        return router_loss

    def forward(
        self,
        input: paddle.Tensor,
        token_type_ids=None,
        use_dense_expert=False,
        moe_k=None,
        moe_capacity=None,
        new_expert_num=None,
        global_gate_mask=None,
        new_expert_rank_list=None,
        is_diff_topk=False,
        is_diff_expert_num=False,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """_summary_

        Args:
            input (paddle.Tensor): _description_
            token_type_ids (_type_, optional): _description_. Defaults to None.

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]: _description_
        """
        use_fuse = isinstance(
            self.gate, (RoundRobinGateFused, SinkHornGateFused, TopKGateFused)
        )
        assert use_fuse
        if input.ndim == 3:
            orig_shape = input.shape
            # clone 保平安
            input = input.reshape([-1, input.shape[-1]])
        else:
            orig_shape = None

        assert (
            len(input.shape) == 2
        ), f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"
        dispatch_token_type_ids = None
        # 587 us
        global_dense_expert_mask = None
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, :-1].reshape([-1])
            dispatch_token_type_ids = token_type_ids
            if self.config.sequence_parallel:
                hcg = fleet.get_hybrid_communicate_group()
                rank = hcg.get_model_parallel_rank()
                interval = (
                    token_type_ids.shape[0] // hcg.get_model_parallel_world_size()
                )
                token_type_ids = token_type_ids.slice(
                    [0], rank * interval, (rank + 1) * interval
                )
                token_type_ids.stop_gradient = True

            if use_dense_expert:
                global_dense_expert_mask = (
                    dispatch_token_type_ids == self.dense_token_type
                )
        assert self.gate is not None
        if hasattr(self, "rng") and self.rng.random() < self.all_to_all_dropout:
            orig_shape_2 = input.shape
            output = self.forward_experts(input)
            output += self.gate.weight.sum() * 0.0  # hack for grad
            output = output.reshape(orig_shape or orig_shape_2)  # [e*1,c,m]
            return output, None, 0
        with profile("fused_gate_and_dispatch"):
            (
                dispatched_input,
                global_hidden_states,
                local_combine_weights,  # dispatched-expert间聚合后重新在 seq 维度切片
                expert_num_global_no_token_drop,  # 不考虑截断！
                expert_num_global,  # 考虑截断
                expert_num_global_list,
                local_scatter_index,  # dispatched-expert间聚合后重新在 seq 维度切片
                scatter_index_rev,
                router_loss,
                (gate_logits, gate_prob),
                (gate_logits_mm, gate_prob_mm),
                expert_num_local,
            ) = self.fused_gate_and_dispatch(
                input,
                token_type_ids,
                global_dense_expert_mask,
                moe_k=moe_k,
                moe_capacity=moe_capacity,
                new_expert_num=new_expert_num,
                global_gate_mask=global_gate_mask,
                new_expert_rank_list=new_expert_rank_list,
                is_diff_topk=is_diff_topk,
                is_diff_expert_num=is_diff_expert_num,
            )
        seqlen_this_mp = input.shape[0]
        if len(scatter_index_rev):
            recv_rank_local = scatter_index_rev // seqlen_this_mp
        else:
            recv_rank_local = scatter_index_rev

        if self.use_padding:
            with profile("alltoall-prepare"):
                # if self.send_rank is None:
                capacity = self.gate.get_capacity(
                    input.shape[0] * self.config.moe_world_size, cap_factor=moe_capacity
                )
                self.send_rank = (
                    paddle.arange(self.config.moe_world_size)
                    .repeat_interleave(capacity * self.num_local_experts)
                    .astype("int32")  # cap
                )
                self.local_expert_id = (
                    paddle.arange(self.num_local_experts)
                    .repeat_interleave(capacity)
                    .tile(self.config.moe_world_size)
                    .astype(self.send_rank.dtype)
                )
                recv_rank, recv_rank_task = allgather_async(
                    recv_rank_local, group=self.config.moe_group
                )
                send_rank = self.send_rank
                local_expert_id = self.local_expert_id

        else:
            with profile("alltoall-prepare"):
                all_expert_num = sum(expert_num_global_list)
                recv_rank = paddle.empty([all_expert_num], dtype=recv_rank_local.dtype)
                # 非常慢
                recv_rank_task = dist.stream.alltoall_single(
                    recv_rank,
                    recv_rank_local.tile(self.config.moe_world_size),
                    [
                        sum(
                            expert_num_global_list[
                                i
                                * self.num_local_experts : (i + 1)
                                * self.num_local_experts
                            ]
                        )
                        for i in range(self.config.moe_world_size)
                    ],  # output-size
                    [len(recv_rank_local)] * self.config.moe_world_size,  # input-size
                    group=self.config.moe_group,
                    sync_op=False,
                    use_calc_stream=False,
                )
                # 尝试过用 self.id_buffer + slice,显然会更慢, 干脆直接在 np 上造完整个 id，后续写一手 C
                if moe_utils is None:
                    send_rank_cpu = np.concatenate(  # TOO SLOW!!! break every thing
                        [
                            np.full([j], i // self.num_local_experts, dtype="int32")
                            for i, j in enumerate(expert_num_global_list)
                        ],
                        0,
                    )
                    local_expert_id_cpu = np.concatenate(
                        [
                            np.full([j], i % self.num_local_experts, dtype="int32")
                            for i, j in enumerate(expert_num_global_list)
                        ],
                        0,
                    )
                    gpu_ids = paddle.to_tensor(
                        np.stack([send_rank_cpu, local_expert_id_cpu], 0), place="gpu"
                    )
                    send_rank, local_expert_id = gpu_ids.unbind(0)
                else:
                    send_rank, local_expert_id = (
                        moe_utils.build_src_rank_and_local_expert_id(
                            expert_num_global,
                            expert_num_global_list,
                            self.num_local_experts,
                        )
                    )

        with profile("moe_comm"):
            if not self.use_expert_out_alltoall:
                expert_outs = (
                    recompute(self.forward_experts, *dispatched_input)
                    if self.recompute and self.training
                    else self.forward_experts(*dispatched_input)
                )
                expert_outs = paddle.concat(
                    [e for e in expert_outs if e is not None], axis=0
                )  # [e*c,m]
                expert_out_to_combine = AllGatherOp.apply(
                    expert_outs, group=self.config.moe_group
                )  # for test
                router_loss2 = self.calc_router_loss_and_logging(
                    router_loss,
                    gate_logits,
                    gate_prob,
                    gate_logits_mm,
                    gate_prob_mm,
                    local_combine_weights,
                    expert_num_global_no_token_drop,
                    token_type_ids,
                    dispatch_token_type_ids,
                    moe_k=moe_k,
                    is_diff_topk=is_diff_topk,
                    is_diff_expert_num=is_diff_expert_num,
                )
            else:
                if self.enable_logging and global_training_logs_enabled():
                    capacity = self.gate.get_capacity(
                        input.shape[0] * self.config.moe_world_size,
                        cap_factor=moe_capacity,
                    )
                    _log = {}
                    valid_usage = [e for e in expert_num_global_list if e > 0]
                    if valid_usage:
                        max_usage = max(valid_usage)
                        min_usage = min(valid_usage)
                        _log[f"expert_min_usage_layer_{self.layer_idx}"] = (
                            min_usage / capacity
                        )
                        _log[f"expert_max_usage_layer_{self.layer_idx}"] = (
                            max_usage / capacity
                        )

                    global_training_logs.update(**_log)

                recv_rank_task and recv_rank_task.wait()  # wait for recv_rank

                world_size = dist.get_world_size(self.config.moe_group)
                this_rank = dist.get_rank(self.config.moe_group)

                recv_size = paddle.count_nonzero(
                    recv_rank == dist.get_rank(self.config.moe_group)
                )
                recv_size = paddle.maximum(
                    recv_size, paddle.ones([], dtype=recv_size.dtype)
                )  # at least 1, make paddle happy
                if get_env_device() == "xpu":
                    recv_size_cpu = recv_size.to("cpu")
                    recv_size_task = None
                else:
                    recv_size_cpu, recv_size_task = async_offload(
                        recv_size, get_async_loader()
                    )

                send_rank_this_rank = paddle.count_nonzero(send_rank == this_rank)
                if get_env_device() == "xpu":
                    send_rank_this_rank_cpu = send_rank_this_rank.to("cpu")
                    send_rank_this_rank_task = None
                else:
                    send_rank_this_rank_cpu, send_rank_this_rank_task = async_offload(
                        send_rank_this_rank, get_async_loader()
                    )

                with profile("scatter_nd"):
                    recv_rank[recv_rank == -1] = world_size
                    send_recv_count_global = paddle.scatter_nd_add(
                        paddle.zeros(
                            [self.num_local_experts, world_size + 1, world_size + 1],
                            dtype="int32",
                        ),
                        paddle.stack([local_expert_id, send_rank, recv_rank], -1),
                        paddle.ones([len(send_rank)], dtype="int32"),
                    )  # [num_local_experts, world_size + 1 , world_size + 1]
                    send_counts_cpu = send_recv_count_global[:, this_rank, :-1].numpy()
                    recv_counts_cpu = send_recv_count_global[:, :-1, this_rank].numpy()
                    send_counts_num_cpu = send_counts_cpu.sum(-1)
                    recv_counts_num_cpu = recv_counts_cpu.sum(-1)

                # paddle.save(dict(
                #     expert_num_global=expert_num_global,
                #     expert_out=expert_out.shape,
                #     local_scatter_index=local_scatter_index,
                #     local_expert_id=local_expert_id,
                #     send_rank_global=send_rank,
                #     recv_rank_global=recv_rank,
                #     num_local_experts=self.num_local_experts,
                #     capacity=expert_out.shape[1],
                # ), f'test.deubg.dump.{dist.get_rank()}')
                if self.use_expert_alltoall_overlap:
                    forward_func_dict = {
                        i: lambda x: ex(x.contiguous())
                        for i, ex in enumerate(self.experts)
                    }
                else:
                    forward_func_dict = None
                    dispatched_input = self.forward_experts(*dispatched_input)

                if recv_size_task is not None:
                    recv_size_task.cpu_wait()
                if send_rank_this_rank_task is not None:
                    send_rank_this_rank_task.cpu_wait()

                input_size = sum(
                    [len(i) if i is not None else 0 for i in dispatched_input]
                )
                if (
                    self.use_padding or input_size > 1
                ):  # input=1时可能对应空输入，暂不校验
                    assert send_rank_this_rank_cpu.item() == input_size, (
                        send_rank,
                        [len(i) if i is not None else 0 for i in dispatched_input],
                    )

                (
                    expert_out_to_combine,
                    router_loss2,
                    distributed_input_to_alltoall_out,
                ) = AlltoAllSmart.apply(
                    *dispatched_input,
                    router_loss,
                    gate_logits,
                    gate_prob,
                    gate_logits_mm,
                    gate_prob_mm,
                    local_combine_weights,
                    expert_num_global_no_token_drop,
                    token_type_ids,
                    dispatch_token_type_ids,
                    paddle.to_tensor(moe_k),
                    forward_func_dict=forward_func_dict,
                    router_loss_fn=self.calc_router_loss_and_logging,
                    local_expert_id=local_expert_id,
                    send_rank_global=send_rank,
                    recv_rank_global=recv_rank,
                    num_local_experts=self.num_local_experts,
                    capacity=dispatched_input[0].shape[1] if self.use_padding else None,
                    use_padding=self.use_padding,
                    expert_num_global=expert_num_global_list,
                    is_first_fwd=not framework._dygraph_tracer()._has_grad,
                    group=self.config.moe_group,
                    recv_size=recv_size_cpu,
                    send_counts=send_counts_cpu,
                    recv_counts=recv_counts_cpu,
                    send_counts_num=send_counts_num_cpu,
                    recv_counts_num=recv_counts_num_cpu,
                )
                # /origin input -> distributed input/ => /origin-input -> alltoall out -input/
                local_scatter_index = distributed_input_to_alltoall_out[
                    local_scatter_index
                ]
                local_scatter_index.stop_gradient = True
            # global -> local
            with profile("combine"):
                # debug
                # combined_output2 = combined_output.clone()
                combined_output = self.combine_expert_output(
                    expert_out_to_combine, local_combine_weights, local_scatter_index
                )
                # debug
                # combined_output = combined_output - combined_output2.detach() + combined_output2
                # combined_output = combined_output2
        with profile("dense-expert"):
            if use_dense_expert:
                dense_input = global_hidden_states[global_dense_expert_mask.squeeze(-1)]
                dense_out = self.dense_experts(dense_input)
                # dense_out_padded = paddle.zeros_like(global_hidden_states)
                # dense_out_padded[global_dense_expert_mask.squeeze(-1)] = dense_out
                dense_out_padded = paddle.scatter_nd(
                    paddle.where(global_dense_expert_mask.squeeze(-1)),
                    dense_out,
                    global_hidden_states.shape,
                )

                dense_out_padded = ScatterOp.apply(dense_out_padded)
                combined_output += dense_out_padded

        with profile("shared-expert"):
            if self.shared_experts is not None:
                # globa -> local
                if self.is_mp_moe:
                    shared_out = self.shared_experts(
                        global_hidden_states, use_comm=False
                    )
                else:
                    shared_out = self.shared_experts(input)
                combined_output += shared_out

        if orig_shape:
            combined_output = combined_output.reshape(
                orig_shape[:-1] + [combined_output.shape[-1]]
            )

        return combined_output, local_combine_weights, router_loss2, gate_logits
        # allgather
