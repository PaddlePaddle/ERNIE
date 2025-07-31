# !/usr/bin/env python3

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Ernie model"""
import math
import os
import re
from functools import partial
import logging
from typing import Optional, Tuple, Union, Dict
import contextlib
from dataclasses import dataclass
import random
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.incubate.tensor.manipulation import async_offload

from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.communication.group import _get_global_group
from paddle.autograd import PyLayer
import paddle.distributed.communication.group

from copy import deepcopy

# from src/ops which is install in build_envs

from paddle.distributed.fleet.layers.mpu.mp_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

from paddleformers.transformers.conversion_utils import (
    StateDictNameMapping,
    init_name_mappings,
)

from paddleformers.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions as _BaseModelOutput,
)
from paddleformers.transformers.model_outputs import (
    CausalLMOutputWithCrossAttentions as _CausalLMOutput,
)


from paddleformers.transformers.model_utils import PretrainedModel, register_base_model

from models.moe.moe_layer import MoEStatics
from models.moe.moe_all_gather_layer import (
    MOEAllGatherLayer,
    MOEAllGatherLayerV2,
    DeepEPMOELayerMultiModal,
)
from models.moe.elastic_moe_all_gather_layer import (
    ElasticMOEAllGatherLayerV2,
)
from models.sequence_parallel_utils import (
    ScatterOp,
    GatherOp,
    ColumnSequenceParallelLinear,
    RowSequenceParallelLinear,
    mark_as_sequence_parallel_parameter,
    get_async_loader,
    hack_offload_wait,
)

from paddleformers.utils.tools import get_env_device
from models.aadiff_decorator import check_aadiff

from models.ernie.modeling import fp8_autocast, te_recompute, TEFP8Linear

from models.fp8_linear import Fp8FusedMlpFunc, MemEfficientFp8FusedMlpFunc

NativeLinear = nn.Linear
if get_env_device() == "xpu":
    try:
        from paddle_xpu.layers.nn import Linear as XPULinear
        from paddle_xpu.layers.nn import ColumnParallelLinear as XPUColumnParallelLinear
        from paddle_xpu.layers.nn import RowParallelLinear as XPURowParallelLinear

        NativeLinear = XPULinear
        ColumnParallelLinear = XPUColumnParallelLinear
        RowParallelLinear = XPURowParallelLinear
    except ImportError:
        print("[WARN] import paddle_xpu for mp error!")

    try:
        from paddle_xpu.layers.nn.sequence_parallel import GatherOp as XPUGatherOp
        from paddle_xpu.layers.nn.sequence_parallel import ScatterOp as XPUScatterOp

        GatherOp = XPUGatherOp
        ScatterOp = XPUScatterOp
    except ImportError:
        print("[WARN] import paddle_xpu for sp error!")

    if not os.getenv("XPU_MOE_USE_ALLGATHER", False):
        try:
            from paddle_xpu.layers.nn.sequence_parallel import (
                XPUColumnSequenceParallelLinear,
                XPURowSequenceParallelLinear,
            )

            ColumnSequenceParallelLinear = XPUColumnSequenceParallelLinear
            RowSequenceParallelLinear = XPURowSequenceParallelLinear
        except ImportError:
            print("[WARN] import paddle_xpu for sp error!")

from models.comm_utils import profile
from models.ernie.modeling import (
    ErnieAttention,
    ErnieMLP,
    RotaryEmbedding,
    FusedLayerNorm,
    LayerNorm,
    RMSNorm,
    ErnieLMHead,
    _expand_mask,
    _make_causal_mask,
    finfo,
)

from models.ernie.modeling import (
    ErniePretrainingCriterion as ErniePretrainingCriterionBase,
)
from models.ernie.modeling import FusedDropoutImpl
from models.moe.moe_layer import (
    MOELayer,
    MOELayerWithAllGatherDispatcher,
    MOEInferLayer,
    DeepEPMOELayer,
    DeepEPDropTokenMOELayer,
)

from models.moe.elastic_moe_layer import ElasticMOELayer, DeepEPElasticMOELayer

from models.moe.moe_layer_uneven import MOELayer as MOELayerSizeAll2All


try:
    from paddle.distributed import in_auto_parallel_align_mode
except:

    def in_auto_parallel_align_mode():
        """
        hack for paddlenlp develop branch.
        """
        return False


from models.moe.top2_gate import Top2Gate, TopKGateFused, DeepEPTop2Gate
from models.moe.elastic_top2_gate import ElasticTopKGateFused
from models.moe.round_robin_gate import RoundRobinGate, RoundRobinGateFused
from models.moe.random_gate import RandomGate
from models.moe.sinkhorn_gate import SinkHornGate, SinkHornGateFused
from models.moe.task_gate import TaskGate
from .configuration import ErnieMoEConfig

try:
    from paddle.incubate.nn.functional import swiglu as fused_swiglu
except (ImportError, ModuleNotFoundError):
    fused_swiglu = None

from models.elastic_utils import get_topk_random_uniform

logger = logging.getLogger(__name__)

# HACK: paddle.fluid.libpaddle.ProcessGroupNCCL 不支持 deepcopy protocol, 此举实属无奈。
paddle.distributed.communication.group.Group.__deepcopy__ = lambda self, _: self
# HACK: paddle.fluid.libpaddle.ProcessGroupNCCL 不支持 json 序列化 protocol, 此举实属无奈。
paddle.distributed.communication.group.Group.to_json = lambda self: repr(self)


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(_BaseModelOutput):
    """doc"""

    router_loss: Optional[paddle.Tensor] = None
    gate_logits: Optional[Tuple[paddle.Tensor]] = None
    mtp_outputs: Optional[paddle.Tensor] = None


@dataclass
class CausalLMOutputWithCrossAttentions(_CausalLMOutput):
    """doc"""

    router_loss: Optional[paddle.Tensor] = None


try:
    from paddle.nn.functional.flash_attention import flash_attention

    logger.warning(
        "Use flash attention in scaled-dot-product. Attention mask is deprecated"
    )
except ImportError:
    flash_attention = None

try:
    import fused_ln as fused
except ImportError:
    logger.warning(
        "fused-ln not found, run `python src/ops/fused_ln_setup.py install` to build fused ln"
    )
    fused = None

from models.utils import get_global_training_logs

global_training_logs = (
    get_global_training_logs()
)  # 没有erniebot的环境下无法打印 debug 量

ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST = []

__all__ = [
    "ErnieMoEForCausalLM",
    "ErniePretrainingCriterion",
    "CausalLMOutputWithCrossAttentions",
]

gate_class = dict(
    round_robin=RoundRobinGate,
    round_robin_fused=RoundRobinGateFused,
    random=RandomGate,
    top2=Top2Gate,
    top2_fused=TopKGateFused,
    sinkhorn=SinkHornGate,
    sinkhorn_fused=SinkHornGateFused,
    task_gate=TaskGate,
    elastic_top2_fused=ElasticTopKGateFused,
    deepep_top2_fused=DeepEPTop2Gate,
)


def get_gate(
    config: ErnieMoEConfig,
    expert: Tuple[Tuple[int, nn.Layer]],
    layer_idx: int,
) -> Tuple[nn.Layer, nn.LayerList]:
    """get moe gate

    Args:
        config (ErnieMoEConfig): _description_
        expert (Tuple[Tuple[int, nn.Layer]]): tuple of (num_experts, expert prototype), each represents lm, mm, etc...
        layer_idx (int): _description_

    Returns:
        Tuple[nn.Layer, nn.LayerList]: _description_
    """
    f"""
    自动构建gate, 目前支持 {gate_class.keys()}
    """
    # if config.moe_group != "dummy":
    #     moe_world_size = min(1, dist.get_world_size(config.moe_group))
    # else:
    #     moe_world_size = 1
    moe_num_experts = (
        sum(config.moe_num_experts)
        if config.multimodel_experts
        else config.moe_num_experts
    )
    assert (
        moe_num_experts >= config.moe_world_size
    ), f"expert moe_num_experts={moe_num_experts} >= moe_world_size={config.moe_world_size}"
    assert (
        moe_num_experts % config.moe_world_size == 0
    ), f"expert moe_num_experts={moe_num_experts} % moe_world_size={config.moe_world_size} == 0"
    moe_num_experts_per_device = moe_num_experts // config.moe_world_size
    if not config.moe_fuse_experts:
        experts = nn.LayerList([])
        for expert_id, (experts_num, fc) in enumerate(expert):
            assert experts_num % config.moe_world_size == 0
            num_experts_per_device = experts_num // config.moe_world_size
            experts_to_append = []
            if not hasattr(fc, "__len__"):
                experts_to_append.append(fc)
                if expert_id == 1:
                    with paddle.utils.unique_name.guard("_mm_deepcopy"):
                        for _ in range(num_experts_per_device - 1):
                            experts_to_append.append(deepcopy(fc))
                else:
                    for _ in range(num_experts_per_device - 1):
                        experts_to_append.append(deepcopy(fc))
            else:
                experts_to_append = fc
            for ex in experts_to_append:
                for p in ex.parameters():
                    p.expert_type = f"expert_type_{expert_id}"  # 不同的 `expert_type` 可以有不同的 intermediate-size
            experts.extend(experts_to_append)
        assert (
            len(experts) == moe_num_experts_per_device
        ), f"experts.len={len(experts)} != moe_num_experts_per_device={moe_num_experts_per_device}"
    else:
        assert expert[0][0] == 1, "experts are fused and must be one"
        experts = deepcopy(expert[0][1])

    logger.info(
        f"using moe-world-size: {config.moe_world_size} "
        f"expert-per-device: {moe_num_experts_per_device} "
    )
    if config.moe_use_hard_gate and moe_num_experts <= 2:
        # TODO 3种类型，hard-gate问题
        gate = None
        logger.info("MOE-GATE:-hard-gate")
    else:
        logger.info(f"MOE-GATE:-{config.moe_gate}")
        gate = gate_class[config.moe_gate.lower()](
            config, layer_idx=layer_idx, group=config.moe_group
        )

    if config.multimodel_experts and config.moe_use_hard_gate and moe_num_experts > 2:
        lm_experts = experts[: config.moe_num_experts[0] // config.moe_world_size]
        lm_cfg = deepcopy(config)
        lm_cfg.moe_num_experts = config.moe_num_experts[0]
        lm_gate = gate_class[config.moe_gate.lower()](
            lm_cfg, layer_idx=layer_idx, group=config.moe_group, gate_weight=gate.weight
        )
    else:
        lm_gate, lm_experts = gate, experts
    logger.info(f"LM-experts-{lm_experts} -- experts-{experts}")
    return gate, experts, lm_gate, lm_experts


def build_mpdp_group():
    "mp-dp-group = mp x dp group"
    hcg = fleet.get_hybrid_communicate_group()
    mp_world_size = hcg.get_model_parallel_world_size()
    dp_world_size = hcg.get_data_parallel_world_size()
    sharding_world_size = hcg.get_sharding_parallel_world_size()
    pp_world_size = hcg.get_pipe_parallel_world_size()

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    topo = np.arange(world_size).reshape(
        [pp_world_size, sharding_world_size, dp_world_size, mp_world_size]
    )
    this_group = None
    for i in range(pp_world_size):
        for j in range(sharding_world_size):
            ranks = topo[i, j, :, :].reshape([-1]).tolist()
            group = dist.new_group(ranks)
            if rank in ranks:
                logger.info(f"building mpdp group, this group has rank: {ranks}")
                this_group = group
    return this_group


def _parse_moe_group(
    moe_group: str,
) -> Union[str, paddle.distributed.communication.group.Group]:
    """
    Args:
        moe_group (str): 目前支持："sharding|data|dp|tp|model|mp|dummy"

    Returns:
        Union[str, paddle.distributed.communication.group.Group]: 除了 dummy-moe 以外，返回对应的 progcess-group
    """
    moe_group = moe_group.lower()
    assert moe_group in {
        "sharding",
        "data",
        "dp",
        "mp",
        "tp",
        "model",
        "dummy",
        "none",
        "world",
        "all",
        "mpdp",
        "ep",
    }, f"moe-group not supported, got: {moe_group}"
    logger.info(f"using moe-group: {moe_group}")
    if not hasattr(fleet.fleet, "_hcg"):
        assert moe_group in {
            "dummy",
            "none",
            "world",
            "data",
        }, "only support dummy gate in `single-model`"
    if moe_group == "sharding":
        moe_group = fleet.get_hybrid_communicate_group().get_sharding_parallel_group()
    elif moe_group in {"data", "dp"}:
        if hasattr(fleet.fleet, "_hcg"):
            moe_group = fleet.get_hybrid_communicate_group().get_data_parallel_group()
        else:
            moe_group = _get_global_group()
    elif moe_group in {"mp", "model", "tp"}:
        moe_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
    elif moe_group in {"dummy"}:
        dummy_group = paddle.distributed.communication.group.Group(0, None, [0])
        moe_group = dummy_group
    elif moe_group in {"mpdp"}:
        moe_group = build_mpdp_group()
    elif moe_group in {"ep"}:
        moe_group = fleet.get_hybrid_communicate_group().get_expert_parallel_group()
    else:
        moe_group = _get_global_group()  # None 为全局通信组

    return moe_group


def moe_ep2mp(
    state_dict: Dict[str, paddle.Tensor], config: ErnieMoEConfig, split_actions
):
    """Auto ep2mp

    Args:
        sd (Dict[str, paddle.Tensor]): _description_
        config (ErnieMoEConfig): _description_
        split_actions (_type_): _description_

    Returns:
        _type_: _description_
    """
    if config.tensor_parallel_degree <= 1 or dist.get_world_size(config.moe_group) > 1:
        # dummy moe 且mp大于0, 才进行ep2mp
        return state_dict
    if isinstance(config.moe_num_experts, (list, tuple)):
        num_lm_experts, num_mm_experts = config.moe_num_experts
        num_experts = sum(config.moe_num_experts)
    else:
        num_lm_experts, num_mm_experts = config.moe_num_experts, 0
        num_experts = config.moe_num_experts
    expert_ids = [
        int(re.search(r"mlp\.experts\.(\d+)", k).group(1))
        for k in state_dict.keys()
        if "mlp.experts" in k
    ]
    if expert_ids and max(expert_ids) == num_experts - 1:
        # 参数本身为dummy moe 参数。
        return state_dict

    logger.info("auto ep2mp")
    hcg = fleet.get_hybrid_communicate_group()
    mp_group = hcg.get_model_parallel_group()
    world_size = dist.get_world_size(mp_group)
    num_lm_local_experts = num_lm_experts // world_size
    num_mm_local_experts = num_mm_experts // world_size

    new_sd = {}

    actual_keys = []
    for k in state_dict.keys():
        actual_keys.append(k)
    actual_keys_sorted = sorted(actual_keys)

    for k in actual_keys_sorted:
        if "mlp.experts" in k:
            expert_id = int(re.search(r"mlp\.experts\.(\d+)", k).group(1))
            gathered_experts = []
            tensor = paddle.to_tensor(state_dict[k])  # to gpu tensor
            dist.all_gather(gathered_experts, tensor, group=mp_group)
            for rank in range(len(gathered_experts)):
                if expert_id < num_lm_local_experts:
                    real_id = expert_id + rank * num_lm_local_experts
                else:
                    if num_mm_experts > 0:
                        real_id = (
                            num_lm_experts
                            + (expert_id - num_lm_local_experts)
                            + rank * num_mm_local_experts
                        )
                    else:
                        continue
                new_k = k.replace(f"mlp.experts.{expert_id}", f"mlp.experts.{real_id}")
                logger.info(
                    f"auto ep2mp: {k}->{new_k}, expert_id: {expert_id}, real_id: {real_id}"
                )
                new_sd[new_k] = split_actions[new_k.replace("ernie.", "")](
                    gathered_experts[rank]
                )
        else:
            new_sd[k] = state_dict[k]
    return new_sd


def moe_statedict_local_id_to_global(
    state_dict: Dict[str, paddle.Tensor], config: ErnieMoEConfig
):
    """
    将本地模型中的参数转换为全局模型的参数
    """
    if not config.moe_state_dict_use_global_expert_id:
        return state_dict
    if config.moe_world_size <= 1:
        return state_dict
    moe_num_experts = (
        sum(config.moe_num_experts)
        if isinstance(config.moe_num_experts, (list, tuple))
        else config.moe_num_experts
    )
    if moe_num_experts <= 1:
        return state_dict
    moe_world_size = config.moe_world_size
    if moe_world_size <= 1:
        moe_world_size = 1
    moe_world_size_per_device = moe_num_experts // moe_world_size
    is_multimodal = isinstance(config.moe_num_experts, (list, tuple))
    out_sd = {}
    for key in list(state_dict.keys()):
        if "mlp.experts" in key:
            imoe = int(re.search(r"mlp\.experts\.(\d+)", key).group(1))
            if is_multimodal:
                if imoe < moe_world_size_per_device // 2:  # lm experts
                    global_expert_id = (
                        config.moe_rank * moe_world_size_per_device // 2 + imoe
                    )
                else:
                    global_expert_id = (
                        config.moe_num_experts[0]
                        + config.moe_rank * moe_world_size_per_device // 2
                        + (imoe - moe_world_size_per_device // 2)
                    )
            else:
                global_expert_id = config.moe_rank * moe_world_size_per_device + imoe
            new_k = key.replace(
                f"mlp.experts.{imoe}", f"mlp.experts.{global_expert_id}"
            )
            if new_k != key:
                logger.info(
                    f"[moe-expert_id-local-to-global], auto changed state-dict {key} -> {new_k}"
                )
            out_sd[new_k] = state_dict.pop(key)
        else:
            out_sd[key] = state_dict.pop(key)
    del state_dict
    return out_sd


def moe_statedict_cherry_pick(
    state_dict: Dict[str, paddle.Tensor], config: ErnieMoEConfig
):
    """
    从state_dict中挑选出需要的参数
    """
    if not config.moe_state_dict_use_global_expert_id:
        return state_dict
    moe_num_experts = (
        sum(config.moe_num_experts)
        if isinstance(config.moe_num_experts, (list, tuple))
        else config.moe_num_experts
    )
    if moe_num_experts <= 1:
        return state_dict
    moe_world_size = config.moe_world_size
    if moe_world_size <= 1:
        moe_world_size = 1
    moe_world_size_per_device = moe_num_experts // moe_world_size
    is_multimodal = isinstance(config.moe_num_experts, (list, tuple))
    out_sd = {}
    for key in list(state_dict.keys()):
        if "mlp.experts" in key:
            imoe = int(re.search(r"mlp\.experts\.(\d+)", key).group(1))
            if is_multimodal:
                if imoe >= config.moe_num_experts[0]:
                    local_rank = (imoe - config.moe_num_experts[0]) // (
                        moe_world_size_per_device // 2
                    )
                    local_id = (imoe - config.moe_num_experts[0]) % (
                        moe_world_size_per_device // 2
                    ) + (moe_world_size_per_device // 2)
                else:  # lm experts
                    local_rank = imoe // (moe_world_size_per_device // 2)
                    local_id = imoe % (moe_world_size_per_device // 2)
            else:
                local_rank = imoe // moe_world_size_per_device
                local_id = imoe % moe_world_size_per_device
            if local_rank != config.moe_rank:
                logger.info(
                    f"IGNORE---{key}---{imoe}---{local_rank}--{config.moe_rank}"
                )
                state_dict.pop(key)
                continue
            local_moe_name = key.replace(
                f"mlp.experts.{imoe}", f"mlp.experts.{local_id}"
            )
            # if local_moe_name != key:
            logger.info(f"[moe-cherry-pick] state-dict using {key} as {local_moe_name}")
            out_sd[local_moe_name] = state_dict.pop(key)
        else:
            out_sd[key] = state_dict.pop(key)
    del state_dict
    return out_sd


def moe_statedict_upcycle(
    state_dict: Dict[str, paddle.Tensor],
    config: ErnieMoEConfig,
    dtype,
    merge_actions,
    split_actions,
    layer_idxs=None,
):
    """
    state-dict Upcycle 考虑：
        1. 细粒度 MoE (shared experts)
        2. Attention MoE
    """
    if not isinstance(config.moe_intermediate_size, int):
        logger.warning("moe upcycle only supports single modality expand !")
        return state_dict
    # upcycling目前只处理单mo

    moe_layer_start_index = (
        min(config.moe_layer_start_index)
        if isinstance(config.moe_layer_start_index, (tuple, list))
        else config.moe_layer_start_index
    )
    moe_layer_end_index = (
        max(config.moe_layer_end_index)
        if isinstance(config.moe_layer_end_index, (tuple, list))
        else config.moe_layer_end_index
    )

    if config.moe_num_experts > 0:
        moe_world_size = config.moe_world_size
        if moe_world_size <= 1:
            moe_world_size = 1
        moe_world_size_per_device = config.moe_num_experts // moe_world_size

        granularity = (
            1
            if config.moe_intermediate_size == 0
            else config.intermediate_size // config.moe_intermediate_size
        )

        def slice_granularity(
            w, global_expert_id, column=True, shuffle=False, group_experts=False
        ):
            """细粒度 expert 切分"""
            if group_experts:
                part_id = global_expert_id // (config.moe_num_experts // config.moe_k)
            else:
                part_id = global_expert_id % config.moe_k
            part_id = part_id % granularity
            if shuffle:
                rng = random.Random(
                    global_expert_id // config.moe_k
                )  # 用w.shape[-1] 作为shuffle的seed。
                if column:
                    idx = np.arange(w.shape[-1])
                    rng.shuffle(idx)
                    w = w.index_select(paddle.to_tensor(idx), axis=-1)
                else:
                    idx = np.arange(w.shape[0])
                    rng.shuffle(idx)
                    w = w.index_select(paddle.to_tensor(idx), axis=0)
            if granularity == 1:
                return w
            # part_id = global_expert_id % config.moe_k
            if column:
                per_expert = w.shape[-1] // granularity
                return w[..., part_id * per_expert : (part_id + 1) * per_expert]
            per_expert = w.shape[0] // granularity
            w *= config.moe_k  # tricky hack
            return w[part_id * per_expert : (part_id + 1) * per_expert, ...]

        def slice_granularity_shared(w, column=True):
            if column:
                per_expert = w.shape[-1] // granularity
                return w[..., -(per_expert * config.moe_num_shared_experts) :]
            per_expert = w.shape[0] // granularity
            return w[-(per_expert * config.moe_num_shared_experts) :, ...]

        def _chunk(t):
            return (
                t.chunk(2, axis=-1)
                if isinstance(w, paddle.Tensor)
                else np.split(w, 2, axis=-1)
            )

        def _cat(t):
            return (
                paddle.concat(t, -1)
                if isinstance(t[0], paddle.Tensor)
                else np.concatenate(t, -1)
            )

        granularity = (
            1
            if config.moe_intermediate_size == 0
            else config.intermediate_size // config.moe_intermediate_size
        )
        is_mp_moe = (
            hasattr(fleet.fleet, "_hcg")
            and config.moe_group
            is fleet.get_hybrid_communicate_group().get_model_parallel_group()
        )
        logger.info(f"UPCYCLE-IS_MP_MOE: {is_mp_moe}")
        if (
            is_mp_moe
            and fleet.get_hybrid_communicate_group().get_model_parallel_world_size() > 1
        ):
            mp_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
        else:
            mp_group = None

        for ilayer in range(config.num_hidden_layers):
            if layer_idxs and ilayer not in layer_idxs:
                continue
            if ilayer < moe_layer_start_index or ilayer > moe_layer_end_index:
                continue
            if (ilayer + 1) % config.moe_layer_interval == 0:  # use moe
                for k in ["up_proj", "gate_proj", "down_proj", "up_gate_proj"]:
                    for tail in ["weight", "bias"]:
                        non_moe_key = f"ernie.layers.{ilayer}.mlp.{k}.{tail}"
                        gate_key = f"ernie.layers.{ilayer}.mlp.gate.weight"
                        if non_moe_key in state_dict:
                            w = state_dict[non_moe_key]
                            if mp_group is not None and not (
                                k == "down_proj" and tail == "bias"
                            ):
                                w = paddle.to_tensor(w).to(get_env_device())
                                gathered_w = []
                                logger.info(
                                    f"all_gather {non_moe_key} for moe upcycling"
                                )
                                dist.all_gather(gathered_w, w, group=mp_group)
                                w = w.cpu()
                                gathered_w = [v.cpu() for v in gathered_w]
                                gathered_w = merge_actions[
                                    non_moe_key.replace("ernie.", "")
                                ](gathered_w)
                                logger.info(
                                    f"gathered w is {gathered_w.shape}, type {gathered_w.dtype}"
                                )
                                w = gathered_w
                        for imoe in range(moe_world_size_per_device):
                            moe_name = (
                                f"ernie.layers.{ilayer}.mlp.experts.{imoe}.{k}.{tail}"
                            )
                            if moe_name not in state_dict and non_moe_key in state_dict:
                                if k == "up_gate_proj":
                                    w_ = _cat(
                                        [
                                            slice_granularity(
                                                ww,
                                                config.moe_rank
                                                * moe_world_size_per_device
                                                + imoe,
                                                column=True,
                                                group_experts=config.moe_group_experts,
                                            )
                                            for ww in _chunk(w)
                                        ]
                                    )
                                elif k == "down_proj" and tail == "bias":
                                    w_ = deepcopy(w)
                                else:
                                    w_ = slice_granularity(
                                        w,
                                        config.moe_rank * moe_world_size_per_device
                                        + imoe,
                                        column=k
                                        in {"up_proj", "gate_proj", "up_gate_proj"},
                                        group_experts=config.moe_group_experts,
                                    )
                                    logger.info(
                                        f"before slice: {w.shape} -> {w_.shape}"
                                    )
                                logger.info(
                                    f"moe auto expand state-dict, ffn name G={granularity}: "
                                    f"{moe_name} {w_.shape} {w_.dtype} {dtype}"
                                )
                                if isinstance(w_, np.ndarray):
                                    w_ = paddle.to_tensor(w_)
                                if w_.dtype == dtype:
                                    state_dict[moe_name] = w_
                                else:
                                    state_dict[moe_name] = w_.cast(dtype)

                        if config.moe_num_shared_experts > 0:
                            moe_name = (
                                f"ernie.layers.{ilayer}.mlp.shared_experts.{k}.{tail}"
                            )
                            if moe_name not in state_dict and non_moe_key in state_dict:
                                if k == "up_gate_proj":
                                    w_ = _cat(
                                        [
                                            slice_granularity_shared(ww, column=True)
                                            for ww in _chunk(w)
                                        ]
                                    )
                                    if mp_group is not None:
                                        w_ = split_actions[
                                            non_moe_key.replace("ernie.", "")
                                        ](w_)
                                elif k == "down_proj" and tail == "bias":
                                    w_ = deepcopy(w)
                                else:
                                    w_ = slice_granularity_shared(
                                        w,
                                        column=k
                                        in {"up_proj", "gate_proj", "up_gate_proj"},
                                    )
                                    logger.info(f"W_ {k}-{w.shape}--shape-{w_.shape}")
                                    if mp_group is not None:
                                        w_ = split_actions[
                                            non_moe_key.replace("ernie.", "")
                                        ](w_)
                                logger.info(
                                    f"moe auto expand state-dict, shared experts, ffn name G={granularity}: "
                                    f"{moe_name} {w_.shape} {w_.dtype}"
                                )
                                if isinstance(w_, np.ndarray):
                                    w_ = paddle.to_tensor(w_)
                                if w_.dtype == dtype:
                                    state_dict[moe_name] = w_
                                else:
                                    state_dict[moe_name] = w_.cast(dtype)

    if config.moe_num_attn_experts > 0:
        moe_world_size = config.moe_world_size
        if moe_world_size <= 1:
            moe_world_size = 1
        moe_world_size_per_device = config.moe_num_attn_experts // moe_world_size
        for ilayer in range(config.num_hidden_layers):
            if layer_idxs and ilayer not in layer_idxs:
                continue
            if ilayer < moe_layer_start_index or ilayer > moe_layer_end_index:
                continue
            if (ilayer + 1) % config.moe_layer_interval == 0:  # use moe
                for k in ["q_proj", "k_proj", "v_proj", "qkv_proj"]:
                    for tail in ["weight", "bias"]:
                        non_moe_key = f"ernie.layers.{ilayer}.self_attn.{k}.{tail}"
                        if non_moe_key in state_dict:
                            w_ = state_dict[non_moe_key]
                            for imoe in range(moe_world_size_per_device):
                                moe_name = f"ernie.layers.{ilayer}.self_attn.{k}.experts.{imoe}.{tail}"
                                if moe_name not in state_dict:
                                    logger.info(
                                        "moe auto expand state-dict, hacking qkv name "
                                        f"{moe_name} {w_.shape} {w_.dtype}"
                                    )
                                    if isinstance(w_, np.ndarray):
                                        w_ = paddle.to_tensor(w_)
                                    if w_.dtype == dtype:
                                        state_dict[moe_name] = w_
                                    else:
                                        state_dict[moe_name] = w_.cast(dtype)
    return state_dict


class ErnieMoeMLP(ErnieMLP):
    """_summary_

    Args:
        ErnieMoeMLP (_type_): _description_
    """

    def __init__(self, config, is_shared_expert=False):
        """
        MOE experts，在 MP-moe 下，后才用FusedLinear 实现，其他情况下采用Col/Row Linear.
        """
        if getattr(config, "disable_ffn_model_parallel", False):
            # assert config.moe_group == "mp", f"when using mp_moe, expect moe-group == mp, but get {config.moe_group}"
            config = deepcopy(config)
            config.tensor_parallel_degree = 1
        super().__init__(config)
        self.moe_dropout_prob = config.moe_dropout_prob
        self.fuse_swiglu = config.fuse_swiglu
        if self.fuse_swiglu:
            assert fused_swiglu is not None, "fused_swiglu operator is not found."
        self.is_shared_expert = is_shared_expert
        self.shared_expert_mem_efficient = self.config.fp8_mem_configs["shared_expert"]

    def forward(self, x, use_comm=True):
        """
        Args:
            x: Tensor [B,S,dim] or [S,dim]
            use_comm: Skip column-linear allgather if set to false
        Returns:
            same shape as `x`
        """
        if (
            self.config.use_fp8
            and self.config.fp8_configs["layers"]["mlp_tp_fc1_linear"]
            and not use_comm
        ):
            raise ValueError(
                "TEFP8Linear do not support use_comm=Flase when using Column Sequence Parallel"
            )

        if (
            self.config.tensor_parallel_degree <= 1
            and self.fuse_ffn
            and self.config.use_fp8_mlp
            and not self.config.use_bias
        ):
            if self.is_shared_expert and self.shared_expert_mem_efficient:
                return MemEfficientFp8FusedMlpFunc.apply(
                    x, self.up_gate_proj.weight, self.down_proj.weight
                )
            return Fp8FusedMlpFunc.apply(
                x, self.up_gate_proj.weight, self.down_proj.weight
            )

        if self.fuse_ffn:
            up_gate_proj = (
                partial(self.up_gate_proj, use_comm=use_comm)
                if (
                    isinstance(self.up_gate_proj, ColumnSequenceParallelLinear)
                    and get_env_device() != "xpu"
                )
                else self.up_gate_proj
            )
        else:
            gate_proj = (
                partial(self.gate_proj, use_comm=use_comm)
                if (
                    isinstance(self.gate_proj, ColumnSequenceParallelLinear)
                    and get_env_device() != "xpu"
                )
                else self.gate_proj
            )
            up_proj = (
                partial(self.up_proj, use_comm=use_comm)
                if (
                    isinstance(self.up_proj, ColumnSequenceParallelLinear)
                    and get_env_device() != "xpu"
                )
                else self.up_proj
            )

        if self.fuse_swiglu:
            if self.fuse_ffn:
                if self.config.use_fp8 and self.config.fp8_configs["smooth_swiglu"]:
                    x, gate = up_gate_proj(x).chunk(2, axis=-1)

                    with paddle.no_grad():
                        scale = paddle.clip(gate.abs().max(axis=-1, keepdim=True), 1e-8)

                    gate = gate / scale
                    if self.config.sequence_parallel or self.config.submatrix_parallel:
                        scale = ScatterOp.apply(scale)

                    x = paddle.concat([x, gate], axis=-1)
                else:
                    x = up_gate_proj(x)
                x = fused_swiglu(x)
            else:
                x = fused_swiglu(gate_proj(x), up_proj(x))
        else:
            if self.fuse_ffn:
                x, gate = up_gate_proj(x).chunk(2, axis=-1)
                x = F.silu(x) * gate
            else:
                x = F.silu(gate_proj(x)) * up_proj(x)
        if self.moe_dropout_prob > 0:
            with get_rng_state_tracker().rng_state("local_seed"):
                x = F.dropout(x=x, p=self.moe_dropout_prob)
        if self.config.use_fp8 and self.config.fp8_configs["smooth_swiglu"]:
            return self.down_proj(x) * scale
        ret = self.down_proj(x)
        return ret


class ErnieMoEMultiModalPairedMLP(ErnieMoeMLP):
    """_summary_

    Args:
        nn (_type_): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    def __init__(self, config):
        """ """
        cfg = deepcopy(config)
        cfg.intermediate_size = config.moe_intermediate_size[0]
        super().__init__(cfg)
        sub_experts_list = []
        for inter in config.moe_intermediate_size[1:]:
            if inter > 0:
                cfg = deepcopy(config)
                cfg.intermediate_size = inter
                sub_experts_list.append(ErnieMoeMLP(cfg))
            else:
                sub_experts_list.append(None)
        self.sub_experts_list = nn.LayerList(sub_experts_list)

    def forward(self, hidden_state):
        """_summary_

        Args:
            hidden_state (_type_): _description_
            token_type_ids (_type_): _description_

        Returns:
            _type_: _description_
        """
        d_model = hidden_state.shape[-1] - 1
        hidden_state, token_type_ids = paddle.split(hidden_state, [d_model, 1], axis=-1)
        out = None
        for token_type, expert in enumerate([super().forward, *self.sub_experts_list]):
            if expert is None:
                continue
            if out is not None:
                out += expert(hidden_state) * (token_type_ids == token_type).astype(
                    hidden_state.dtype
                )
            else:
                out = expert(hidden_state) * (token_type_ids == token_type).astype(
                    hidden_state.dtype
                )
        return out


class ErnieMoeDenseExpert(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        LinearFN = paddle.incubate.nn.FusedLinear if config.fuse_linear else nn.Linear
        mp_degree = max(1, config.tensor_parallel_degree)
        self.is_mp = mp_degree > 1
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fuse_ffn = config.fuse_attn_ffn

        if config.fuse_attn_ffn:
            self.up_gate_proj = LinearFN(
                self.hidden_size,
                self.intermediate_size * 2 // mp_degree,
                bias_attr=config.use_bias,
            )
            self.up_gate_proj.weight.is_distributed = self.is_mp
            if config.use_bias:
                self.up_gate_proj.bias.is_distributed = self.is_mp
        else:
            self.gate_proj = LinearFN(
                self.hidden_size,
                self.intermediate_size // mp_degree,
                bias_attr=config.use_bias,
            )
            self.up_proj = LinearFN(
                self.hidden_size,
                self.intermediate_size // mp_degree,
                bias_attr=config.use_bias,
            )
            self.gate_proj.weight.is_distributed = self.is_mp
            self.up_proj.weight.is_distributed = self.is_mp
            if config.use_bias:
                self.gate_proj.bias.is_distributed = self.is_mp
                self.up_proj.bias.is_distributed = self.is_mp
        self.down_proj = LinearFN(
            self.intermediate_size // mp_degree,
            self.hidden_size,
            bias_attr=config.use_bias,
        )
        self.down_proj.weight.is_distributed = self.is_mp

        self.fuse_swiglu = config.fuse_swiglu
        if self.fuse_swiglu:
            assert fused_swiglu is not None, "fused_swiglu operator is not found."
        if self.is_mp:
            self.mp_group = (
                fleet.get_hybrid_communicate_group().get_model_parallel_group()
            )

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.fuse_swiglu:
            if self.fuse_ffn:
                x = fused_swiglu(self.up_gate_proj(x))
            else:
                x = fused_swiglu(self.gate_proj(x), self.up_proj(x))
        else:
            if self.fuse_ffn:
                x, gate = self.up_gate_proj(x).chunk(2, axis=-1)
                x = F.silu(x) * gate
            else:
                x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        if self.is_mp:
            x = F.linear(x, self.down_proj.weight)
            output_ = mp_ops._mp_allreduce(
                x,
                group=self.mp_group,
                use_calc_stream=True,
                use_model_parallel=True,
            )
            output = output_ + self.down_proj.bias if self.config.use_bias else output_
        else:
            output = self.down_proj(x)

        return output


class BMMLinear(nn.Layer):
    """BMM 实现的 SwishGLU 层, 实现expert-fusion"""

    def __init__(self, experts, d_in, d_out, use_bias=False):
        super().__init__()
        self.weight = self.create_parameter(
            [experts, d_in, d_out], dtype=paddle.get_default_dtype()
        )
        if use_bias:
            self.bias = self.create_parameter(
                [experts, d_out], dtype=paddle.get_default_dtype(), is_bias=True
            )
        else:
            self.bias = None

    def forward(self, x):
        """x: [num_experts, Seq, dim]"""
        if self.bias is not None:
            return paddle.bmm(x, self.weight) + self.bias
        return paddle.bmm(x, self.weight)


class ErnieMoeMLPFused(nn.Layer):
    """Fused Implement of ErnieMoeMLP"""

    def __init__(self, config):
        """doc"""
        assert (
            hasattr(config, "disable_ffn_model_parallel")
            or config.tensor_parallel_degree == 1
        ), f"fused mlp only suport mp-moe, mp={config.tensor_parallel_degree}"  # 还不吃支持  expert 内的mp 通信。
        assert config.fuse_attn_ffn, "fused mlp only support fuse_attn_ffn"
        # config = deepcopy(config)
        # config.tensor_parallel_degree = 1
        super().__init__()
        self.moe_dropout_prob = config.moe_dropout_prob
        self.num_local_experts = config.moe_num_experts // config.moe_world_size
        logger.info(
            f"fused-expert-weight-shape: {[self.num_local_experts, config.hidden_size, config.intermediate_size]}"
        )

        self.up_gate_proj = BMMLinear(
            self.num_local_experts, config.hidden_size, config.intermediate_size * 2
        )
        self.down_proj = BMMLinear(
            self.num_local_experts, config.intermediate_size, config.hidden_size
        )
        self.fuse_swiglu = config.fuse_swiglu
        if self.fuse_swiglu:
            assert fused_swiglu is not None, "fused_swiglu operator is not found."

    def __len__(self):
        return self.num_local_experts

    def __iter__(self):
        return (self for _ in range(1))

    def forward(self, x):
        """x"""
        if self.fuse_swiglu:
            x = fused_swiglu(self.up_gate_proj(x))
        else:
            gate, x = self.up_gate_proj(x).chunk(2, axis=-1)
            x = F.silu(gate) * x
        x = self.down_proj(x)
        return x


class FusedLinearAddNormFunc(paddle.autograd.PyLayer):
    """
    Recompute linear+residual+rms_norm
    """

    @staticmethod
    def forward(ctx, x, residual, linear_weight, rms_norm_weight, eps):
        """
        计算前向传播的结果。

        Args:
            ctx (paddle.autograd.grad_context.GradContext): 自动微分上下文，用于保存前向传播中的中间变量。
            x (paddle.Tensor): 输入的tensor，形状为[batch_size, seq_len, hidden_size]。
            residual (paddle.Tensor): 残差连接，形状为[batch_size, seq_len, hidden_size]。
            linear_weight (paddle.Tensor): 线性变换的权重，形状为[hidden_size, hidden_size]。
            rms_norm_weight (paddle.Tensor): RMSNorm的权重，形状为[hidden_size]。
            eps (float): 防止除零的小正数。

        Returns:
            tuple: 包含两个paddle.Tensor的元组，分别为归一化后的输出和加法操作后的输出。
                - norm_out (paddle.Tensor): 归一化后的输出，形状为[batch_size, seq_len, hidden_size]。
                - add_out (paddle.Tensor): 加法操作后的输出，形状为[batch_size, seq_len, hidden_size]。

        """
        linear_out = paddle.matmul(x, linear_weight)
        add_out = linear_out + residual
        norm_out, invar = fused.fused_rms_norm(add_out, rms_norm_weight, eps)

        ctx.save_for_backward(x, residual, linear_weight, rms_norm_weight, eps)

        return norm_out, add_out

    @staticmethod
    def backward(ctx, d_rms_norm_out, d_residual_out):
        """
        对给定的输入进行反向传播计算梯度。

        Args:
            ctx (paddle.autograd.grad_context.GradContext): 上下文对象，保存了前向传播中的状态信息。
            d_rms_norm_out (paddle.Tensor): RMS 归一化输出的梯度。
            d_residual_out (paddle.Tensor): 残差输出的梯度。

        Returns:
            tuple: 包含四个梯度值，分别为输入 x 的梯度、残差 residual 的梯度、线性权重 linear_weight 的梯度、RMS 归一化权重 rms_norm_weight 的梯度。

        """
        x, residual, linear_weight, rms_norm_weight, eps = ctx.saved_tensor()

        linear_out = paddle.matmul(x, linear_weight)
        add_out = linear_out + residual

        rms_out, invar = fused.fused_rms_norm(add_out, rms_norm_weight, eps)

        d_add_out, d_rms_norm_weight = fused.fused_rms_norm_grad_func(
            add_out, rms_norm_weight, invar, d_rms_norm_out, eps
        )

        d_residual = d_add_out + d_residual_out
        d_linear_out = d_residual
        dx, d_linear_weight = paddle._C_ops.matmul_grad(
            x, linear_weight, d_linear_out, False, False
        )

        return dx, d_residual, d_linear_weight, d_rms_norm_weight


class FusedLinearAddNorm(paddle.nn.Layer):
    """
    Recompute linear+residual+rms_norm
    """

    def __init__(self, hidden_size, eps=1e-6) -> None:
        super().__init__()
        self._dtype = self._helper.get_default_dtype()

        self.linear_weight = self.create_parameter(
            shape=[hidden_size, hidden_size],
            dtype=self._dtype,
            is_bias=False,
        )

        self.rms_norm_weight = self.create_parameter(
            shape=[hidden_size],
            dtype=self._dtype,
            default_initializer=nn.initializer.Constant(1.0),
        )

        self.eps = eps

    def forward(self, x, residual):
        """
        Args:
            x (Tensor): 输入张量。
            residual (Tensor): 残差张量。

        Returns:
            Tensor: 经过 FusedLinearAddNormFunc 操作后的输出张量。

        """
        return FusedLinearAddNormFunc.apply(
            x, residual, self.linear_weight, self.rms_norm_weight, self.eps
        )


class FusedRMSLinearFunc(paddle.autograd.PyLayer):
    """
    Recompute FusedRMSNorm in FusedRMSNorm + Linear combination
    """

    @staticmethod
    def forward(ctx, x, rms_norm_weight, linear_weight, eps):
        """
        前向传播函数，用于计算模型输出。

        Args:
            ctx (Context): 上下文对象，用于保存计算过程中的中间结果，以便在反向传播时使用。
            x (Tensor): 输入数据，形状为 (batch_size, seq_len, embed_dim)。
            rms_norm_weight (Tensor): RMSNorm层的权重，形状为 (embed_dim,)。
            linear_weight (Tensor): 线性层的权重，形状为 (embed_dim, output_dim)。
            eps (float): RMSNorm层中用于防止除零错误的小正数。

        Returns:
            Tensor: 输出数据，形状为 (batch_size, seq_len, output_dim)。

        """

        hidden_states, invar = fused.fused_rms_norm(x, rms_norm_weight, eps)
        q = paddle.matmul(hidden_states, linear_weight)

        ctx.save_for_backward(x, rms_norm_weight, linear_weight, eps)
        return q

    @staticmethod
    def backward(ctx, d_qkv):
        """
        反向传播函数，用于计算反向梯度。

        Args:
            ctx (Context): 存储了前向传播过程中的一些变量和梯度。
            d_qkv (Tensor): 对应于qkv向量的梯度。

        Returns:
            Tuple[Tensor, Tensor, Tensor]: 包含三个张量，分别对应于输入x的梯度、rms_norm_weight的梯度以及linear_weight的梯度。

        """
        x, rms_norm_weight, linear_weight, eps = ctx.saved_tensor()
        hidden_states, invar = fused.fused_rms_norm(x, rms_norm_weight, eps)
        h_grad, d_linear_weight = paddle._C_ops.matmul_grad(
            hidden_states, linear_weight, d_qkv, False, False
        )

        dx, d_rms_norm_weight = fused.fused_rms_norm_grad_func(
            x, rms_norm_weight, invar, h_grad, eps
        )

        return dx, d_rms_norm_weight, d_linear_weight


class FusedRMSLinear(paddle.nn.Layer):
    """
    Recompute FusedRMSNorm in FusedRMSNorm + Linear combination
    """

    def __init__(
        self, hidden_size, eps=1e-6, num_heads=1, num_key_value_heads=1
    ) -> None:
        super().__init__()
        self._dtype = self._helper.get_default_dtype()

        self.rms_norm_weight = self.create_parameter(
            shape=[hidden_size],
            dtype=self._dtype,
            default_initializer=nn.initializer.Constant(1.0),
        )
        kv_hidden_size = hidden_size // num_heads * num_key_value_heads
        qkv_out = hidden_size + kv_hidden_size * 2

        self.linear_weight = self.create_parameter(
            shape=[hidden_size, qkv_out],
            dtype=self._dtype,
            is_bias=False,
        )
        self.eps = eps

    def forward(self, x):
        """
        计算给定输入 x 的前向传播结果。

        Args:
            x (Tensor): 输入张量

        Returns:
            Tensor: 返回经过 FusedRMSLinearFunc 操作后的输出张量，类型与输入 x 相同。

        """
        return FusedRMSLinearFunc.apply(
            x, self.rms_norm_weight, self.linear_weight, self.eps
        )


class ErnieMoEAttention(ErnieAttention):
    """
    moe attention 模块，支持 qkv attention。 rope 等逻辑复用 `ErnieAttention`
    """

    def __init__(self, config, layer_idx):
        # assert not config.moe_attn or config.fuse_attn_ffn, "`moe-attn` requires `fuse-attn-ffn`"
        if (
            config.moe_num_experts
            and config.moe_num_attn_experts
            and (layer_idx + 1) % config.moe_layer_interval == 0
        ):
            assert (
                not config.sequence_parallel
            ), "# moe-attn 下不能开启sequence-parallel，因为不知道其他 mp 的 token 是什么模态的。"

            assert not hasattr(
                config, "disable_ffn_model_parallel"
            ), '`moe_group == "mp"` 情况下还不支持 moe-attn'
            # config = deepcopy(config)
            # config.tensor_parallel_degree = 1
            self.use_moe_attn = True
        else:
            self.use_moe_attn = False

        super().__init__(config)

        self.use_linear_residual_norm_recompute = (
            config.use_linear_residual_norm_recompute
        )
        self.use_rms_qkv_recompute = config.use_rms_qkv_recompute
        self.use_norm_gate_recompute = config.use_norm_gate_recompute

        if config.use_rms_qkv_recompute is True:

            assert config.use_rmsnorm is True and config.fuse_rms_norm is True
            assert config.fuse_linear is True and config.use_bias is False

            assert self.use_moe_attn is False and self.fuse_attn is True

            if self.is_gqa:
                self.fused_rms_norm_linear = FusedRMSLinear(
                    self.hidden_size,
                    config.rms_norm_eps,
                    self.num_heads,
                    self.num_key_value_heads,
                )
            else:
                self.fused_rms_norm_linear = FusedRMSLinear(
                    self.hidden_size, config.rms_norm_eps
                )
            del self.qkv_proj

        if not self.use_moe_attn:
            return
        config = deepcopy(config)
        config.moe_num_experts = config.moe_num_attn_experts
        self.moe_use_allgather = get_env_device() == "xpu" and os.getenv(
            "XPU_MOE_USE_ALLGATHER", False
        )

        def _moefy(
            fc,
        ):
            gate, experts, _, _ = get_gate(
                config, [(config.moe_num_experts, fc)], layer_idx
            )
            if config.moe_use_size_all2all:
                if hasattr(fleet.fleet, "_hcg"):
                    assert (
                        config.moe_group
                        is fleet.get_hybrid_communicate_group().get_model_parallel_group()
                        or not config.sequence_parallel
                    ), "dp moe w/ size all2all not support sequence parallel, will hang"
                return MOELayerSizeAll2All(
                    gate,
                    experts,
                    layer_idx=layer_idx,
                    group=config.moe_group,
                    recompute=config.use_recompute_moe,
                    enable_logging=config.moe_logging,
                )
            MOELayerClass = (
                MOELayerWithAllGatherDispatcher if self.moe_use_allgather else MOELayer
            )
            return MOELayerClass(
                gate,
                experts,
                layer_idx=layer_idx,
                group=config.moe_group,
                recompute=config.use_recompute_moe,
                enable_logging=config.moe_logging,
                k=config.moe_k,
                all_to_all_dropout=config.moe_all_to_all_dropout,
                group_experts=config.moe_group_experts,
            )

        if config.fuse_attn_ffn:
            self.qkv_proj = _moefy(
                self.qkv_proj,
            )
        else:
            self.q_proj = _moefy(
                self.q_proj,
            )
            self.k_proj = _moefy(
                self.k_proj,
            )
            self.v_proj = _moefy(
                self.v_proj,
            )

    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        inbatch_pack_offset: Optional[Tuple[paddle.Tensor]] = None,
        token_type_ids: Optional[Tuple[paddle.Tensor]] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel
        token_type_ids: for sinkhorn gating.
        """
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, :-1]
        if self.config.sequence_parallel:
            if token_type_ids is not None:
                token_type_ids = token_type_ids.reshape([-1])
                token_type_ids = ScatterOp.apply(token_type_ids)
                token_type_ids.stop_gradient = True
            # 在SP下输入 hidden.shape = [bsz*seq/mp_size, dim]
            # logger.info(f'into attn shape={hidden_states.shape}')
            # bsz = hidden_states.shape[0] * self.config.tensor_parallel_degree // self.config.seqlen
            q_len = self.config.seqlen
        else:
            q_len = hidden_states.shape[-2]

        query_states = key_states = value_states = mix_layer = None
        router_loss = None

        if self.use_rms_qkv_recompute:
            mix_layer = self.fused_rms_norm_linear(hidden_states)
        else:
            if self.fuse_attn:
                if self.use_moe_attn:
                    mix_layer, _, router_loss_attn_qkv, _ = self.qkv_proj(
                        hidden_states, token_type_ids
                    )
                    router_loss = router_loss_attn_qkv
                else:
                    mix_layer = self.qkv_proj(
                        hidden_states,
                    )
            else:
                if self.use_moe_attn:
                    query_states, _, router_loss_attn_q, _ = self.q_proj(
                        hidden_states, token_type_ids
                    )
                    key_states, _, router_loss_attn_k, _ = self.k_proj(
                        hidden_states, token_type_ids
                    )
                    value_states, _, router_loss_attn_v, _ = self.v_proj(
                        hidden_states, token_type_ids
                    )
                    router_loss = (
                        router_loss_attn_q + router_loss_attn_k + router_loss_attn_v
                    )
                else:
                    query_states = self.q_proj(hidden_states)
                    key_states = self.k_proj(hidden_states)
                    value_states = self.v_proj(hidden_states)

        if self.fuse_attn:
            if self.is_gqa:
                query_states, key_states, value_states = paddle.split(
                    mix_layer.reshape(
                        [
                            -1,
                            q_len,
                            self.num_heads + 2 * self.num_key_value_heads,
                            self.head_dim,
                        ]
                    ),
                    [
                        self.num_heads,
                        self.num_key_value_heads,
                        self.num_key_value_heads,
                    ],
                    axis=2,
                )
                mix_layer = None
            else:
                mix_layer = mix_layer.reshape(
                    [-1, q_len, self.num_heads, 3 * self.head_dim]
                )

        else:
            query_states = query_states.reshape(
                shape=[-1, q_len, self.num_heads, self.head_dim]
            )
            key_states = key_states.reshape(
                shape=[
                    -1,
                    q_len,
                    self.num_key_value_heads if self.is_gqa else self.num_heads,
                    self.head_dim,
                ]
            )
            value_states = value_states.reshape(
                shape=[
                    -1,
                    q_len,
                    self.num_key_value_heads if self.is_gqa else self.num_heads,
                    self.head_dim,
                ]
            )
        if self.use_recompute_attn:
            assert past_key_value is None, "do not use kv cache in recompute"
            assert not use_cache
            attn_output, attn_weights, past_key_value = recompute(
                self.rope_attn,
                mix_layer,
                query_states,
                key_states,
                value_states,
                attention_mask,
                position_ids,
                output_attentions,
                past_key_value,
                use_cache,
                inbatch_pack_offset,
                use_reentrant=False,
            )
        else:
            attn_output, attn_weights, past_key_value = self.rope_attn(
                mix_layer=mix_layer,
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                past_key_value=past_key_value,
                use_cache=use_cache,
                inbatch_pack_offset=inbatch_pack_offset,
            )
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        # logger.info(f'before reduce-scatter: {attn_output.astype("float32").norm(axis=-1)}')

        # `o_proj` will run outside of attn when `fuse_attn_ffn` is True.
        if self.use_linear_residual_norm_recompute is False:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        if router_loss is not None:
            return attn_output, attn_weights, past_key_value, router_loss
        return attn_output, attn_weights, past_key_value


def set_pp_can_free(x):
    """
    Mark a Tensor that can be freed by the pipeline parallelism
    """
    assert isinstance(x, paddle.Tensor), type(x)
    try:
        setattr(x, "pp_can_free", True)
    except AttributeError:
        pass


class FakeMoERouterLoss(PyLayer):
    """The trick function of adding router loss,
    which includes the gradient of the router loss during backpropagation."""

    @staticmethod
    def forward(ctx, x, router_loss, num_acc_steps, enable_delay_scale_loss):
        """
        Args:
            x (Tensor): Input tensor of hidden_states.
            router_loss (Tensor): Router loss of moe.

        Returns:
            Tensor: Output tensor with the same size and data as input.
        """
        ctx.num_acc_steps = num_acc_steps
        ctx.loss_shape = router_loss.shape
        ctx.loss_dtype = router_loss.dtype
        ctx.enable_delay_scale_loss = enable_delay_scale_loss
        set_pp_can_free(x)
        return x

    @staticmethod
    def backward(ctx, out_grad):
        """
        Args:
            out_grad (Tensor): Output grad tensor of hidden_states.

        Returns:
            Tensor: Output tensor with the same size and data as input.
            Tensor: The start loss grad.
        """
        if ctx.enable_delay_scale_loss:
            router_loss_grad_value = 1.0
        else:
            router_loss_grad_value = 1.0 / ctx.num_acc_steps

        # 精度对齐模式下需要设置router_loss回传的梯度为0
        # 当前自动并行和动手需要在不带router_loss场景下才能做到精度对齐，否则会导致自动并行与动手精度无法对齐
        if in_auto_parallel_align_mode():
            router_loss_grad_value = 0.0

        return out_grad, paddle.full(
            ctx.loss_shape, router_loss_grad_value, dtype=ctx.loss_dtype
        )


class ErnieDecoderLayer(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, config, layer_idx):
        """
        def __init__(self, config, layer_idx):
        """
        super().__init__()
        self._training = True
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_moe_infer = config.get("is_moe_infer", False)
        self.moe_use_size_all2all = config.moe_use_size_all2all
        self.config = config
        self.use_moe = config.use_moe
        self.self_attn = ErnieMoEAttention(config, layer_idx)
        self.moe_use_allgather = get_env_device() == "xpu" and os.getenv(
            "XPU_MOE_USE_ALLGATHER", False
        )
        self.use_elastic_topk = config.use_elastic_topk
        self.use_elastic_topk_for_mbs = config.use_elastic_topk_for_mbs
        self.use_deepep = config.use_deepep
        self.drop_before_deepep = config.drop_before_deepep
        self.use_linear_residual_norm_recompute = (
            config.use_linear_residual_norm_recompute
        )
        self.use_rms_qkv_recompute = config.use_rms_qkv_recompute
        self.use_norm_gate_recompute = config.use_norm_gate_recompute

        moe_layer_start_index = (
            min(config.moe_layer_start_index)
            if isinstance(config.moe_layer_start_index, (tuple, list))
            else config.moe_layer_start_index
        )
        moe_layer_end_index = (
            max(config.moe_layer_end_index)
            if isinstance(config.moe_layer_end_index, (tuple, list))
            else config.moe_layer_end_index
        )

        if (
            self.use_moe
            and ((layer_idx + 1) % config.moe_layer_interval == 0)
            and layer_idx >= moe_layer_start_index
            and layer_idx <= moe_layer_end_index
        ):
            gate, experts, lm_gate, lm_experts, moe_statics = (
                self._init_gate_and_experts(layer_idx)
            )
            shared_experts = self._init_shared_experts()
            dense_experts = self._init_dense_experts(layer_idx)
            if self.is_moe_infer:
                self.mlp = MOEInferLayer(
                    gate,
                    experts,
                    group=config.moe_group,
                    recompute=config.use_recompute_moe,
                )
            elif self.moe_use_size_all2all:
                if hasattr(fleet.fleet, "_hcg"):
                    assert (
                        config.moe_group
                        is fleet.get_hybrid_communicate_group().get_model_parallel_group()
                        or not config.sequence_parallel
                    ), "dp moe w/ size all2all not support sequence parallel, will hang"
                logger.info("moe use size all2all")
                assert gate is None or isinstance(
                    gate, (RoundRobinGateFused, TopKGateFused, SinkHornGateFused)
                ), type(gate)
                self.mlp = MOELayerSizeAll2All(
                    gate,
                    experts,
                    layer_idx=layer_idx,
                    group=config.moe_group,
                    recompute=config.use_recompute_moe,
                    enable_logging=config.moe_logging,
                )
            else:
                if self.use_elastic_topk:
                    moe_cls = (
                        DeepEPElasticMOELayer if self.use_deepep else ElasticMOELayer
                    )
                elif self.use_elastic_topk_for_mbs:
                    assert (
                        self.use_deepep
                    ), "can't support use_elastic_topk_for_mbs for not deepep yet"
                    if self.use_deepep:
                        if self.drop_before_deepep:
                            moe_cls = DeepEPDropTokenMOELayer
                        else:
                            moe_cls = DeepEPMOELayer
                else:
                    if self.use_deepep:
                        if self.drop_before_deepep:
                            moe_cls = DeepEPDropTokenMOELayer
                        else:
                            moe_cls = DeepEPMOELayer
                    else:
                        moe_cls = MOELayer
                if config.moe_multimodal_dispatch_use_allgather:
                    if (
                        str(config.moe_multimodal_dispatch_use_allgather)
                        == "deepep_nodrop"
                    ):
                        moe_cls = partial(
                            DeepEPMOELayerMultiModal,
                            dense_experts=dense_experts,
                            dense_token_type=config.moe_dense_experts_token_type_id,
                        )
                    elif str(config.moe_multimodal_dispatch_use_allgather).startswith(
                        "v2"
                    ):
                        if self.use_elastic_topk:
                            moe_cls = partial(
                                ElasticMOEAllGatherLayerV2,
                                use_expert_out_alltoall="alltoall"
                                in config.moe_multimodal_dispatch_use_allgather,
                                use_padding="unpad"
                                not in config.moe_multimodal_dispatch_use_allgather,
                                enable_reverse_token_drop=config.moe_reverse_token_drop,
                                dense_experts=dense_experts,
                                dense_token_type=config.moe_dense_experts_token_type_id,
                            )
                        else:
                            moe_cls = partial(
                                MOEAllGatherLayerV2,
                                use_expert_out_alltoall="alltoall"
                                in config.moe_multimodal_dispatch_use_allgather,
                                use_padding="unpad"
                                not in config.moe_multimodal_dispatch_use_allgather,
                                enable_reverse_token_drop=config.moe_reverse_token_drop,
                                dense_experts=dense_experts,
                                dense_token_type=config.moe_dense_experts_token_type_id,
                            )
                    else:
                        moe_cls = MOEAllGatherLayer

                    if dense_experts is not None:
                        assert str(
                            config.moe_multimodal_dispatch_use_allgather
                        ).startswith(
                            "v2"
                        ), "only `MOEAllGatherLayerV2` can process dense experts"
                else:
                    assert (
                        dense_experts is None
                    ), "only `MOEAllGatherLayerV2` can process dense experts"
                if self.moe_use_allgather:
                    moe_cls = MOELayerWithAllGatherDispatcher
                logger.info(
                    f"moe-logging:{config.moe_logging} {config.moe_multimodal_dispatch_use_allgather} moe_cls={moe_cls}"
                )
                self.mlp = moe_cls(
                    gate,
                    experts,
                    layer_idx=layer_idx,
                    shared_experts=shared_experts,
                    group=config.moe_group,
                    recompute=config.use_recompute_moe,
                    enable_logging=config.moe_logging,
                    k=config.moe_k,
                    enable_bpr=config.moe_use_bpr,
                    all_to_all_dropout=config.moe_all_to_all_dropout,
                    group_experts=config.moe_group_experts,
                    moe_statics=moe_statics,
                )
                if config.multimodel_experts and config.moe_use_hard_gate:
                    if (
                        str(config.moe_multimodal_dispatch_use_allgather)
                        == "deepep_nodrop"
                    ):
                        logger.info(
                            f"text expert using moe-allgather-layer: {type(self.mlp)}"
                        )
                        _mlp_text = DeepEPMOELayerMultiModal(
                            lm_gate,
                            lm_experts,
                            layer_idx=layer_idx,
                            shared_experts=shared_experts,
                            group=config.moe_group,
                            recompute=config.use_recompute_moe,
                            enable_logging=config.moe_logging,
                            k=config.moe_k,
                            enable_bpr=config.moe_use_bpr,
                            all_to_all_dropout=config.moe_all_to_all_dropout,
                            group_experts=config.moe_group_experts,
                            moe_statics=moe_statics,
                        )
                    elif "text" in str(config.moe_multimodal_dispatch_use_allgather):
                        logger.info(
                            f"text expert using moe-allgather-layer: {type(self.mlp)}"
                        )
                        _mlp_text = MOEAllGatherLayerV2(
                            lm_gate,
                            lm_experts,
                            layer_idx=layer_idx,
                            use_expert_out_alltoall=True,
                            use_padding=False,
                            shared_experts=shared_experts,
                            group=config.moe_group,
                            recompute=config.use_recompute_moe,
                            enable_logging=config.moe_logging,
                            k=config.moe_k,
                            enable_bpr=config.moe_use_bpr,
                            enable_reverse_token_drop=config.moe_reverse_token_drop,
                            all_to_all_dropout=config.moe_all_to_all_dropout,
                            group_experts=config.moe_group_experts,
                            moe_statics=moe_statics,
                        )
                    else:
                        if self.use_deepep:
                            if self.drop_before_deepep:
                                moe_cls = DeepEPDropTokenMOELayer
                            else:
                                moe_cls = DeepEPMOELayer
                            _mlp_text = moe_cls(
                                lm_gate,
                                lm_experts,
                                layer_idx=layer_idx,
                                shared_experts=shared_experts,
                                group=config.moe_group,
                                recompute=config.use_recompute_moe,
                                enable_logging=config.moe_logging,
                                k=config.moe_k,
                                enable_bpr=config.moe_use_bpr,
                                all_to_all_dropout=config.moe_all_to_all_dropout,
                                group_experts=config.moe_group_experts,
                                moe_statics=moe_statics,
                            )
                        else:
                            _mlp_text = MOELayer(
                                lm_gate,
                                lm_experts,
                                layer_idx=layer_idx,
                                shared_experts=shared_experts,
                                group=config.moe_group,
                                recompute=config.use_recompute_moe,
                                enable_logging=config.moe_logging,
                                k=config.moe_k,
                                enable_bpr=config.moe_use_bpr,
                                all_to_all_dropout=config.moe_all_to_all_dropout,
                                group_experts=config.moe_group_experts,
                                moe_statics=moe_statics,
                            )
                    self.mlp_text = (
                        lambda: _mlp_text
                    )  # 这个lambd防止 text部分参数被扫进state-dict
                if dense_experts is not None and config.moe_use_hard_gate:
                    _mlp_dense_experts = MOEAllGatherLayerV2(
                        lm_gate,
                        lm_experts,
                        use_expert_out_alltoall=True,
                        use_padding=False,
                        dense_experts=dense_experts,
                        layer_idx=layer_idx,
                        shared_experts=shared_experts,
                        group=config.moe_group,
                        recompute=config.use_recompute_moe,
                        enable_logging=config.moe_logging,
                        k=config.moe_k,
                        enable_bpr=config.moe_use_bpr,
                        enable_reverse_token_drop=config.moe_reverse_token_drop,
                        all_to_all_dropout=config.moe_all_to_all_dropout,
                        group_experts=config.moe_group_experts,
                        dense_token_type=config.moe_dense_experts_token_type_id,
                    )
                    self.mlp_dense_experts = (
                        lambda: _mlp_dense_experts
                    )  # 这个lambd防止 audio部分参数被扫进state-dict
            if (
                config.sequence_parallel
            ):  # `mp-moe` 下 gate 在 attn 中生效，处于同步区。
                for p in gate.parameters():
                    mark_as_sequence_parallel_parameter(p)
        else:
            self.mlp = ErnieMLP(config)

        Norm = RMSNorm if config.use_rmsnorm else LayerNorm
        if not config.use_rmsnorm and config.fuse_ln:
            Norm = FusedLayerNorm

        if self.use_rms_qkv_recompute is False:
            self.input_layernorm = Norm(config)

        if self.use_linear_residual_norm_recompute is True:
            assert config.hidden_dropout_prob == 0.0
            assert config.fuse_linear is True and config.use_bias is False
            assert config.use_rmsnorm is True and config.fuse_rms_norm is True
            self.fused_linear_add_norm = FusedLinearAddNorm(
                self.hidden_size, config.rms_norm_eps
            )
            del self.self_attn.o_proj
        else:
            self.residual_add1 = FusedDropoutImpl(
                config.hidden_dropout_prob, mode="upscale_in_train"
            )
            self.post_attention_layernorm = Norm(config)

        # check use postnorm_gate recompute
        if self.use_norm_gate_recompute is True and isinstance(self.mlp, MOELayer):
            self.mlp.add_gate_recompute_func(
                self.post_attention_layernorm.weight, config.rms_norm_eps
            )

        self.residual_add2 = FusedDropoutImpl(
            config.hidden_dropout_prob, mode="upscale_in_train"
        )
        # self.residual_add1 = FusedDropoutAdd(config.hidden_dropout_prob, mode="upscale_in_train")
        # self.residual_add2 = FusedDropoutAdd(config.hidden_dropout_prob, mode="upscale_in_train")

        if config.sequence_parallel:
            if self.use_linear_residual_norm_recompute is True:
                mark_as_sequence_parallel_parameter(
                    self.fused_linear_add_norm.rms_norm_weight
                )
            else:
                mark_as_sequence_parallel_parameter(
                    self.post_attention_layernorm.weight
                )
            # mp-moe下 bias, expert 内没有Column/RowLinear. 不用挂钩子.
            if not hasattr(config, "disable_ffn_model_parallel"):
                # mlp 相关bias的sequence_parallel hook 由RowLinear内部挂。
                if self.use_rms_qkv_recompute is True:
                    mark_as_sequence_parallel_parameter(
                        self.self_attn.fused_rms_norm_linear.rms_norm_weight
                    )
                else:
                    mark_as_sequence_parallel_parameter(self.input_layernorm.weight)

            if not config.use_rmsnorm:
                mark_as_sequence_parallel_parameter(self.post_attention_layernorm.bias)
                mark_as_sequence_parallel_parameter(self.input_layernorm.bias)

    @property
    def training(self):
        """overide training flag"""
        return self._training

    @training.setter
    def training(self, new):
        """overide training flag"""
        if hasattr(self, "mlp_text"):
            for c in self.mlp_text().sublayers():
                c.training = new
        self._training = new

    def _init_gate_and_experts(self, layer_idx):
        """init gate and experts

        Args:
            layer_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        cfg = deepcopy(self.config)
        fc_cls = (
            ErnieMoeMLPFused
            if cfg.moe_fuse_experts and not cfg.use_fp8_mlp
            else ErnieMoeMLP
        )
        if self.config.expert_mlp_use_bias is not None:
            cfg.use_bias = self.config.expert_mlp_use_bias

        if cfg.moe_intermediate_size:
            if isinstance(cfg.moe_intermediate_size, (tuple, list)):
                if cfg.moe_multimodal_paired_experts:
                    assert isinstance(
                        cfg.moe_num_experts, int
                    ), "cfg.moe_num_experts must be int when cfg.moe_multimodal_paired_experts is True"
                    fc_cls = ErnieMoEMultiModalPairedMLP
                    fc = [(cfg.moe_num_experts, fc_cls(cfg))]
                else:
                    assert isinstance(cfg.moe_num_experts, (tuple, list)) and len(
                        cfg.moe_num_experts
                    ) == len(cfg.moe_intermediate_size)
                    fc = []
                    for _i, (num_experts, intermediate_size) in enumerate(
                        zip(cfg.moe_num_experts, cfg.moe_intermediate_size)
                    ):
                        ex_cfg = deepcopy(cfg)
                        ex_cfg.intermediate_size = intermediate_size
                        cur_modality_start_layer_idx = (
                            cfg.moe_layer_start_index[_i]
                            if isinstance(cfg.moe_layer_start_index, (tuple, list))
                            else cfg.moe_layer_start_index
                        )
                        cur_modality_end_layer_idx = (
                            cfg.moe_layer_end_index[_i]
                            if isinstance(cfg.moe_layer_end_index, (tuple, list))
                            else cfg.moe_layer_end_index
                        )
                        if (
                            layer_idx >= cur_modality_start_layer_idx
                            and layer_idx <= cur_modality_end_layer_idx
                        ):
                            if _i == 1:
                                with paddle.utils.unique_name.guard(
                                    f"mm_expert_{layer_idx}_"
                                ):
                                    fc.append((num_experts, fc_cls(ex_cfg)))
                            else:
                                fc.append((num_experts, fc_cls(ex_cfg)))
                        else:
                            logger.info(
                                f"moe multimodal experts use Identity layer_idx: {layer_idx}"
                            )
                            fc.append((num_experts, nn.Identity()))
            else:
                cfg.intermediate_size = cfg.moe_intermediate_size
                if cfg.moe_fuse_experts:
                    fc = [(1, fc_cls(cfg))]
                else:
                    fc = [(cfg.moe_num_experts, fc_cls(cfg))]
        else:
            fc = [(cfg.moe_num_experts, fc_cls(cfg))]
        gate, experts, lm_gate, lm_experts = get_gate(self.config, fc, layer_idx)
        # for AuxLoss Free Router:
        if cfg.moe_use_aux_free:
            moe_statics = MoEStatics(cfg, layer_idx)
        else:
            moe_statics = None
        return gate, experts, lm_gate, lm_experts, moe_statics

    def _init_shared_experts(self):
        """init shared experts

        Returns:
            _type_: _description_
        """
        cfg = deepcopy(self.config)
        if cfg.moe_num_shared_experts > 0:
            if cfg.moe_intermediate_size:
                inter_size = (
                    cfg.moe_intermediate_size[0]
                    if isinstance(cfg.moe_intermediate_size, (tuple, list))
                    else cfg.moe_intermediate_size
                )
                cfg.intermediate_size = inter_size * cfg.moe_num_shared_experts
            else:
                cfg.intermediate_size = (
                    cfg.intermediate_size * cfg.moe_num_shared_experts
                )
            cfg.disable_ffn_model_parallel = False  # split shared epxert
            shared_experts = ErnieMoeMLP(cfg, True)
        else:
            shared_experts = None
        return shared_experts

    def _init_dense_experts(self, layer_idx):
        """init dense experts

        Returns:
            _type_: _description_
        """
        cfg = deepcopy(self.config)
        cfg.sequence_parallel = False  # dense experts 是全进全出。
        if cfg.moe_num_dense_experts > 0:
            logger.info("using dense experts")
            if cfg.moe_intermediate_size:
                # hack dense experts 用intermediate_size里面小的。
                inter_size = (
                    cfg.moe_intermediate_size[0]
                    if isinstance(cfg.moe_intermediate_size, (tuple, list))
                    else cfg.moe_intermediate_size
                )
                cfg.intermediate_size = inter_size * cfg.moe_num_dense_experts
                # TODO dense experts 不切mp，手动reduce
            else:
                cfg.intermediate_size = (
                    cfg.intermediate_size * cfg.moe_num_shared_experts
                )
            cfg.disable_ffn_model_parallel = False  # split shared epxert
            with paddle.utils.unique_name.guard(f"audio_expert_{layer_idx}_"):
                dense_experts = ErnieMoeDenseExpert(cfg)
            for p in dense_experts.parameters():
                p.expert_type = "expert_type_3"
                # p.name = unique_name.generate("shit")
        else:
            dense_experts = None
        return dense_experts

    @check_aadiff()
    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        inbatch_pack_offset: Optional[paddle.Tensor] = None,
        output_gate_logits=True,  # PP model should not output gate logits,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `cache` key value states are returned and can be used to speed up decoding
                (see `cache`).
            cache (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        if token_type_ids is not None:
            is_multimodel_token = token_type_ids.any()
            has_dense_experts_token = (
                token_type_ids == self.config.moe_dense_experts_token_type_id
            ).any()
            if get_env_device() == "xpu":
                is_multimodel_token_cpu = is_multimodel_token.to("cpu")
                has_dense_experts_token_cpu = has_dense_experts_token.to("cpu")
                is_multimodel_token_task = None
                has_dense_experts_token_task = None
            else:
                async_loader = get_async_loader()
                is_multimodel_token_cpu, is_multimodel_token_task = async_offload(
                    is_multimodel_token, async_loader
                )
                has_dense_experts_token_cpu, has_dense_experts_token_task = (
                    async_offload(has_dense_experts_token, async_loader)
                )
        else:
            is_multimodel_token_task = None
            is_multimodel_token_cpu = None
            has_dense_experts_token_task = None
            has_dense_experts_token_cpu = None

        # input_norm will run in `self_attn` when `use_rms_qkv_recompute` is True.
        if self.use_rms_qkv_recompute is False:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        (hidden_states, self_attn_weights, present_key_value, *router_loss_attn) = (
            self.self_attn(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=use_cache,
                inbatch_pack_offset=inbatch_pack_offset,
                token_type_ids=token_type_ids,
            )
        )
        if self.use_linear_residual_norm_recompute is True:
            hidden_states, residual = self.fused_linear_add_norm(
                hidden_states, residual
            )
        elif self.use_norm_gate_recompute is True and isinstance(self.mlp, MOELayer):
            with self.model_paralle_dropout():
                hidden_states = self.residual_add1(hidden_states, residual)
            # Fully Connected
            residual = hidden_states
        else:
            with self.model_paralle_dropout():
                hidden_states = self.residual_add1(hidden_states, residual)
            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)

        if isinstance(
            self.mlp,
            (
                MOELayer,
                MOEAllGatherLayer,
                MOELayerWithAllGatherDispatcher,
                MOELayerSizeAll2All,
                MOEInferLayer,
                DeepEPMOELayer,
                DeepEPDropTokenMOELayer,
            ),
        ):
            if is_multimodel_token_task is not None:
                hack_offload_wait(is_multimodel_token_task)
            if has_dense_experts_token_task is not None:
                hack_offload_wait(has_dense_experts_token_task)
            with profile("moe-mlp"):
                if (
                    self.config.multimodel_experts
                    and self.config.moe_use_hard_gate
                    and token_type_ids is not None
                    and not is_multimodel_token_cpu
                ):
                    # from models.comm_utils import md5
                    hidden_states, _, router_loss, gate_logits = self.mlp_text()(
                        hidden_states, None
                    )
                elif has_dense_experts_token_cpu:
                    (
                        hidden_states,
                        _,
                        router_loss,
                        gate_logits,
                    ) = self.mlp_dense_experts()(
                        hidden_states, token_type_ids, use_dense_expert=True
                    )
                else:
                    if self.use_elastic_topk:
                        random_k_list = get_topk_random_uniform(
                            topk_avg=self.config.moe_k,
                            data_num=hidden_states.shape[0],
                            max_topk=self.config.moe_k * 2 - 1,
                        )
                        # print(f"elastic info random_k_list is {random_k_list}")
                        hidden_states_list = paddle.split(
                            hidden_states, hidden_states.shape[0], axis=0
                        )
                        hidden_state_output_list = []
                        router_loss_list = []
                        gate_logits_list = []
                        for hidden_state_idx, one_hidden_state in enumerate(
                            hidden_states_list
                        ):
                            random_k = random_k_list[hidden_state_idx]
                            (
                                one_hidden_state_output,
                                _,
                                one_router_loss,
                                one_gate_logits,
                            ) = self.mlp(
                                one_hidden_state,
                                moe_k=random_k,
                                moe_capacity=random_k,
                                token_type_ids=token_type_ids,
                            )
                            hidden_state_output_list.append(one_hidden_state_output)
                            router_loss_list.append(one_router_loss)
                            gate_logits_list.append(one_gate_logits)
                        hidden_states = paddle.concat(hidden_state_output_list, axis=0)
                        router_loss = paddle.mean(
                            paddle.stack(router_loss_list), axis=0
                        )
                        gate_logits = paddle.concat(gate_logits_list, axis=0)
                    else:
                        if self.use_deepep:
                            if self.drop_before_deepep:
                                hidden_states, _, router_loss, gate_logits = self.mlp(
                                    hidden_states, token_type_ids=token_type_ids
                                )
                            else:
                                hidden_states, _, router_loss, gate_logits = self.mlp(
                                    hidden_states,
                                    input_ids=None,
                                    token_type_ids=token_type_ids,
                                )
                        else:
                            hidden_states, _, router_loss, gate_logits = self.mlp(
                                hidden_states, token_type_ids
                            )
        else:
            hidden_states = self.mlp(hidden_states)
            gate_logits = None

        with self.model_paralle_dropout():
            hidden_states = self.residual_add2(hidden_states, residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if self.use_moe:
            # 只有 `use_moe` 时为非空
            if router_loss_attn:
                router_loss_attn = router_loss_attn[0]
                if self.config.moe_logging:
                    global_training_logs = (
                        get_global_training_logs()
                    )  # 没有erniebot的环境下无法打印 debug 量
                    global_training_logs.update(
                        **{
                            f"router_loss_attn_{self.layer_idx}": router_loss_attn,
                            f"router_loss_{self.layer_idx}": router_loss,
                        }
                    )
                router_loss = router_loss + router_loss_attn

            # use-moe 时无论这一层有没有 moe layer，都会额外增加一个返回值
            if isinstance(
                self.mlp,
                (
                    MOELayer,
                    MOEAllGatherLayer,
                    MOELayerWithAllGatherDispatcher,
                    MOEInferLayer,
                    MOELayerSizeAll2All,
                    MOEAllGatherLayerV2,
                ),
            ):
                hidden_states = FakeMoERouterLoss.apply(
                    hidden_states,
                    router_loss,
                    self.config.num_acc_steps,
                    self.config.enable_delay_scale_loss,
                )
                if self.training:
                    hidden_states.stop_gradient = False

                outputs = (hidden_states,) + outputs[1:]

            if output_gate_logits:
                outputs += (gate_logits,)

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def model_paralle_dropout(self):
        """dropout with seed control"""
        if (
            self.config.tensor_parallel_degree > 1
            and self.config.hidden_dropout_prob > 0.0
        ):
            current_seed = (
                "local_seed" if self.config.sequence_parallel else "global_seed"
            )
            return get_rng_state_tracker().rng_state(current_seed)
        return contextlib.nullcontext()


class ErniePretrainedModel(PretrainedModel):
    """_summary_

    Args:
        PretrainedModel (_type_): _description_

    Returns:
        _type_: _description_
    """

    config_class = ErnieMoEConfig
    base_model_prefix = "ernie"

    @classmethod
    def _get_name_mappings(cls, config: ErnieMoEConfig) -> StateDictNameMapping:
        """
        返回用于映射ERNIE-M-O模型中状态字典名称的映射列表。

        Args:
            config (ErnieMoEConfig): 包含ERNIE-M-O模型配置信息的对象。

        Returns:
            List[StateDictNameMapping]: 包含用于映射ERNIE-M-O模型中的状态字典名称的映射对象的列表。

        """
        mappings: StateDictNameMapping = []
        model_mappings = [
            ["embed_tokens.weight"],
            ["norm.weight"],
        ]
        for layer_index in range(config.num_hidden_layers):
            if config.fuse_attn_ffn:
                layer_mappings = [
                    [
                        f"layers.{layer_index}.self_attn.qkv_proj.weight",
                        None,
                        "transpose",
                    ],
                    [
                        f"layers.{layer_index}.self_attn.o_proj.weight",
                        None,
                        "transpose",
                    ],
                    [f"layers.{layer_index}.self_attn.rotary_emb.inv_freq"],
                    [f"layers.{layer_index}.mlp.down_proj.weight", None, "transpose"],
                    [
                        f"layers.{layer_index}.mlp.up_gate_proj.weight",
                        None,
                        "transpose",
                    ],
                    [f"layers.{layer_index}.input_layernorm.weight"],
                    [f"layers.{layer_index}.post_attention_layernorm.weight"],
                ]
            else:
                layer_mappings = [
                    [
                        f"layers.{layer_index}.self_attn.q_proj.weight",
                        None,
                        "transpose",
                    ],
                    [
                        f"layers.{layer_index}.self_attn.k_proj.weight",
                        None,
                        "transpose",
                    ],
                    [
                        f"layers.{layer_index}.self_attn.v_proj.weight",
                        None,
                        "transpose",
                    ],
                    [
                        f"layers.{layer_index}.self_attn.o_proj.weight",
                        None,
                        "transpose",
                    ],
                    [f"layers.{layer_index}.self_attn.rotary_emb.inv_freq"],
                    [f"layers.{layer_index}.mlp.gate_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.mlp.down_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.mlp.up_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.input_layernorm.weight"],
                    [f"layers.{layer_index}.post_attention_layernorm.weight"],
                ]
            model_mappings.extend(layer_mappings)

        init_name_mappings(mappings=model_mappings)
        if "ErnieModel" not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "model." + mapping[0]
                mapping[1] = "ernie." + mapping[1]
            model_mappings.append(["lm_head.weight", "lm_head.weight", "transpose"])

        mappings = [
            StateDictNameMapping(*mapping, index=index)
            for index, mapping in enumerate(model_mappings)
        ]
        return mappings

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        """
        获取张量并行映射关系，返回一个字典。

        Args:
            config (TensorParallelConfig): TensorParallel配置对象，包含一些参数信息。
            is_split (bool): 是否为分割操作。默认为True。

        Returns:
            Dict[str, Callable[[Any], Any]]: 包含张量并行映射关系的字典，key为需要映射的张量名，value为相应的映射函数。

        """

        from paddleformers.transformers.conversion_utils import split_or_merge_func
        from models.ernie.modeling import gqa_qkv_split_func, gqa_qkv_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        if (
            config.num_key_value_heads is not None
            and config.num_key_value_heads != config.num_attention_heads
        ):
            if is_split:
                qkv_fn = partial(
                    gqa_qkv_split_func,
                    tensor_parallel_degree=config.tensor_parallel_degree,
                    tensor_parallel_rank=config.tensor_parallel_rank,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=config.hidden_size // config.num_attention_heads,
                )
            else:
                qkv_fn = partial(
                    gqa_qkv_merge_func,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=config.hidden_size // config.num_attention_heads,
                )
        else:
            qkv_fn = partial(fn, is_column=True)

        def get_tensor_parallel_split_mappings(num_layers):
            """
            目前仅支持对 dense 参数做自动 mp 切分。
            """
            # input is pySlice if load safetensors, cast to tensor here
            cast_to_tensor_if_needed = lambda x: (
                x if isinstance(x, (paddle.Tensor, np.ndarray)) else x[:]
            )
            final_actions = {}
            if config.fuse_attn_ffn:
                base_actions = {
                    # Column Linear
                    "layers.0.self_attn.qkv_proj.weight": qkv_fn,
                    "layers.0.mlp.up_gate_proj.weight": partial(
                        fn, is_column=True, is_naive_2fuse=True
                    ),
                    "lm_head.weight": partial(
                        fn, is_column=not config.tie_word_embeddings
                    ),
                    # Row Linear
                    "embed_tokens.weight": partial(fn, is_column=False),
                    "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                    "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
                }
                if config.use_bias:
                    base_actions.update(
                        {
                            # Column Linear
                            "layers.0.self_attn.qkv_proj.bias": qkv_fn,
                            "layers.0.mlp.up_gate_proj.bias": partial(
                                fn, is_column=True, is_naive_2fuse=True
                            ),
                            "layers.0.mlp.down_proj.bias": cast_to_tensor_if_needed,
                            "lm_head.bias": partial(fn, is_column=True),
                        }
                    )
            else:
                base_actions = {
                    # Column Linear
                    "layers.0.self_attn.q_proj.weight": partial(fn, is_column=True),
                    "layers.0.self_attn.k_proj.weight": partial(fn, is_column=True),
                    "layers.0.self_attn.v_proj.weight": partial(fn, is_column=True),
                    "layers.0.mlp.gate_proj.weight": partial(fn, is_column=True),
                    "layers.0.mlp.up_proj.weight": partial(fn, is_column=True),
                    "lm_head.weight": partial(
                        fn, is_column=not config.tie_word_embeddings
                    ),
                    # Row Linear
                    "embed_tokens.weight": partial(fn, is_column=False),
                    "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                    "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
                }
                if config.use_bias:
                    base_actions.update(
                        {
                            # Column Linear
                            "layers.0.self_attn.q_proj.bias": partial(
                                fn, is_column=True
                            ),
                            "layers.0.self_attn.k_proj.bias": partial(
                                fn, is_column=True
                            ),
                            "layers.0.self_attn.v_proj.bias": partial(
                                fn, is_column=True
                            ),
                            "layers.0.mlp.gate_proj.bias": partial(fn, is_column=True),
                            "layers.0.mlp.up_proj.bias": partial(fn, is_column=True),
                            "layers.0.mlp.down_proj.bias": cast_to_tensor_if_needed,
                            "lm_head.bias": partial(fn, is_column=True),
                        }
                    )
            moe_in_mp = (
                config.moe_group in {"mp", "model", "tp", "mpdp"}
                if isinstance(config.moe_group, str)
                else hasattr(fleet.fleet, "_hcg")
                and config.moe_group
                is fleet.get_hybrid_communicate_group().get_model_parallel_group()
            )

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        newkey = key.replace("layers.0.", f"layers.{i}.")
                        if config.moe_group in {"mpdp"}:
                            final_actions[newkey] = cast_to_tensor_if_needed
                        else:
                            final_actions[newkey] = action
                        if "mlp" in key and (i + 1) % config.moe_layer_interval == 0:
                            moe_num_experts = (
                                sum(config.moe_num_experts)
                                if config.multimodel_experts
                                else config.moe_num_experts
                            )
                            if moe_num_experts > 0:
                                for expert_id in range(moe_num_experts):
                                    _key = key.replace(
                                        "layers.0.mlp",
                                        f"layers.{i}.mlp.experts.{expert_id}",
                                    )
                                    if moe_in_mp:
                                        final_actions[_key] = cast_to_tensor_if_needed
                                    else:
                                        final_actions[_key] = action
                                for _ in range(config.moe_num_shared_experts):
                                    _key = key.replace(
                                        "layers.0.mlp", f"layers.{i}.mlp.shared_experts"
                                    )
                                    final_actions[_key] = action
                                for _ in range(config.moe_num_dense_experts):
                                    _key = key.replace(
                                        "layers.0.mlp", f"layers.{i}.mlp.dense_experts"
                                    )
                                    final_actions[_key] = action
                            else:
                                final_actions[
                                    key.replace("layers.0.", f"layers.{i}.")
                                ] = action

                        elif "self_attn" in key and (
                            "qkv_proj" in key
                            or "q_proj" in key
                            or "k_proj" in key
                            or "v_proj" in key
                        ):
                            if config.moe_num_attn_experts > 0:
                                for expert_id in range(config.moe_num_attn_experts):
                                    _key = key.replace(
                                        "layers.0.", f"layers.{i}."
                                    ).replace("_proj.", f"_proj.experts.{expert_id}.")
                                    if moe_in_mp:
                                        final_actions[_key] = cast_to_tensor_if_needed
                                    else:
                                        final_actions[_key] = action
                            else:
                                final_actions[
                                    key.replace("layers.0.", f"layers.{i}.")
                                ] = action
                        else:
                            final_actions[key.replace("layers.0.", f"layers.{i}.")] = (
                                action
                            )
                else:
                    final_actions[key] = action
            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)
        return mappings

    def _init_weights(self, layer):
        """Initialization hook"""
        if get_rng_state_tracker().states_:
            rng_tracker = get_rng_state_tracker().rng_state
        else:
            rng_tracker = contextlib.nullcontext

        if isinstance(
            layer,
            (
                ColumnParallelLinear,
                RowParallelLinear,
                ColumnSequenceParallelLinear,
                RowSequenceParallelLinear,
                VocabParallelEmbedding,
                TEFP8Linear,
                ErnieLMHead,
                nn.Embedding,
                BMMLinear,
                NativeLinear,
                paddle.incubate.nn.FusedLinear,
            ),
        ):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            # logger.info(f'initializing pp:{type(layer)}')
            if not hasattr(
                layer, "weight"
            ):  # no weight to initialie : moe-round-robin-gate
                return

            is_moe = getattr(layer.weight, "no_sync", False)
            is_distributed = getattr(layer.weight, "is_distributed")
            # 'model_parallel_rng' 在 DP 间一样(框架设置)，local-seed 是处处不一样
            with rng_tracker("local_seed" if is_moe else "model_parallel_rng"):
                dtype = paddle.get_default_dtype()  # layer.weight.dtype  #
                # dtype = str(dtype).replace("paddle.", "")
                paddle.set_default_dtype("float32")
                if isinstance(layer, TEFP8Linear):
                    layer.weight.set_value(
                        paddle.transpose(
                            paddle.randn(layer.weight.shape[::-1], dtype=dtype).scale(
                                self.config.initializer_range
                            ),
                            perm=[1, 0],
                        )
                    )
                else:
                    layer.weight.set_value(
                        paddle.randn(layer.weight.shape, dtype=dtype).scale(
                            self.config.initializer_range
                        )
                    )
                paddle.set_default_dtype(dtype)
                logger.info(
                    f"dist-init-fc: shape={layer.weight.shape}, dtype={layer.weight.dtype} "
                    f"range={self.config.initializer_range},type={type(layer)}, "
                    f'norm={layer.weight.astype("float32").norm().item()},is_moe={is_moe}'
                )
        elif isinstance(layer, Top2Gate):
            if not hasattr(
                layer, "weight"
            ):  # no weight to initialie : moe-round-robin-gate
                return
            with rng_tracker("model_parallel_rng"):
                dtype = paddle.get_default_dtype()  # layer.weight.dtype  #
                paddle.set_default_dtype("float32")
                layer.weight.set_value(
                    paddle.randn(layer.weight.shape, dtype=layer.weight.dtype).scale(
                        self.config.initializer_range
                    )
                )
                logger.info(
                    f"dist-init-moe_gate: shape={layer.weight.shape}, dtype={layer.weight.dtype} "
                    f"range={self.config.initializer_range},type={type(layer)}, "
                    f'norm={layer.weight.astype("float32").norm().item()}'
                )
                if isinstance(self.config.moe_num_experts, (tuple, list)):
                    for i in range(1, len(self.config.moe_num_experts)):
                        layer_weight = getattr(layer, f"weight_{i}")
                        layer_weight.set_value(
                            paddle.randn(
                                layer_weight.shape, dtype=layer_weight.dtype
                            ).scale(self.config.initializer_range)
                        )
                        logger.info(
                            f"dist-init-moe_gate: shape={layer_weight.shape}, dtype={layer_weight.dtype} "
                            f"range={self.config.initializer_range},type={type(layer)}, "
                            f'norm={layer_weight.astype("float32").norm().item()}'
                        )
                paddle.set_default_dtype(dtype)

        elif isinstance(layer, RotaryEmbedding):
            head_dim = self.config.hidden_size // self.config.num_attention_heads
            inv_freq = 1.0 / (
                layer.base ** (np.arange(0, head_dim, 2).astype("float32") / head_dim)
            )
            # self.register_buffer("inv_freq", inv_freq.cast(dtype))

            # higher acc using float32
            t = np.arange(layer.max_position_embeddings, dtype="float32")
            freqs = np.einsum("i,j->ij", t, inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = np.concatenate([freqs, freqs], axis=-1)
            # [bs, seqlen, nhead, head_dim]
            cos_cached = np.cos(emb)[:, :]  # .astype(dtype)
            sin_cached = np.sin(emb)[:, :]  # .astype(dtype)
            layer.cos_cached.set_value(cos_cached)  # model后续会被cast成half/bfloat16
            layer.sin_cached.set_value(sin_cached)


@register_base_model
class ErnieModel(ErniePretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ErnieDecoderLayer`]
    Args:
        config: ErnieMoEConfig
    """

    def __init__(self, config: ErnieMoEConfig):
        """
        初始化类并设置参数

        Args:
            config (ErnieMoEConfig): 模型的配置类，包含模型超参数

        Returns:
            None
        """
        if config.moe_group in {"mp", "model", "tp", "mpdp"}:
            # assert config.sequence_parallel
            logger.info(
                f"disable FFN tensor model parallel, moe-group={config.moe_group}"
            )
            config.disable_ffn_model_parallel = True

        config.moe_group = _parse_moe_group(config.moe_group)

        config.moe_world_size = dist.get_world_size(config.moe_group)
        if config.moe_world_size < 0:
            config.moe_world_size = 1
        config.moe_rank = dist.get_rank(config.moe_group)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.config = config

        if config.tensor_parallel_degree > 1 and not in_auto_parallel_align_mode():
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
            )
        else:
            # self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx)
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )

        self.layers = nn.LayerList(
            [ErnieDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        Norm = RMSNorm if config.use_rmsnorm else LayerNorm
        if not config.use_rmsnorm and config.fuse_ln:
            Norm = FusedLayerNorm
        self.norm = Norm(config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing

        if self.config.multi_token_pred_depth > 0:
            self.mtp_block = paddle.nn.LayerList(
                [
                    ErnieDecoderLayer(config, layer_idx)
                    for layer_idx in range(self.config.multi_token_pred_depth)
                ]
            )
            Norm = RMSNorm if config.use_rmsnorm else LayerNorm
            if not config.use_rmsnorm and config.fuse_ln:
                Norm = FusedLayerNorm
            self.mtp_hidden_norm = paddle.nn.LayerList(
                [Norm(config) for _ in range(self.config.multi_token_pred_depth)]
            )
            self.mtp_emb_norm = paddle.nn.LayerList(
                [Norm(config) for _ in range(self.config.multi_token_pred_depth)]
            )

            LinearFN = (
                paddle.incubate.nn.FusedLinear
                if config.fuse_linear
                else paddle.nn.Linear
            )
            self.mtp_linear_proj = paddle.nn.LayerList(
                [
                    LinearFN(
                        self.config.hidden_size * 2,
                        self.config.hidden_size,
                        bias_attr=config.use_bias,
                    )
                    for _ in range(self.config.multi_token_pred_depth)
                ]
            )
            if config.sequence_parallel or config.submatrix_parallel:
                for mtp_linear in self.mtp_linear_proj:
                    mark_as_sequence_parallel_parameter(mtp_linear.weight)
                    if config.use_bias:
                        mark_as_sequence_parallel_parameter(mtp_linear.bias)

    def get_input_embeddings(self):
        """
        获取输入嵌入

        Returns:
            nn.Embedding: 嵌入层对象，包含输入序列的嵌入表示

        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """设置模型输入嵌入矩阵。

        Args:
            value (`torch.nn.Embedding`): 包含模型输入嵌入的 Embedding 对象。

        Returns:
            None。

        """
        self.embed_tokens = value

    @classmethod
    def _prepare_decoder_attention_mask(
        cls, attention_mask, input_shape, past_key_values_length, dtype
    ):
        """
        根据输入的mask，将其转化为需要的类型。

        Args:
            attention_mask (Tensor[Bool]): 需要转换的attention mask。
            input_shape (Tuple[int]): 当前输入张量形状。
            past_key_values_length (int): 上一个时间步的密钥值长度。
            dtype (DType): 需要进行转换的目标数据类型。

        Returns:
            Tensor[Float]: 转换后的attention mask。

        """
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, past_key_values_length=past_key_values_length, dtype=dtype
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, dtype, tgt_length=input_shape[-1]
            )
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )
        combined_attention_mask = paddle.maximum(
            combined_attention_mask.astype(dtype),
            paddle.to_tensor(float(finfo(dtype).min), dtype=dtype),
        )
        return combined_attention_mask

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module,
        hidden_states,
        attention_mask,
        position_ids,
        token_type_ids,
        output_attentions,
        past_key_value,
        use_cache,
        inbatch_pack_offset,
    ):
        """
        Recompute the given layer with training parameters.

        Args:
            layer_module (nn.Layer): The model layer to be recomputed.
            hidden_states (Tensor): The input hidden states of shape [batch_size, seq_len, hidden_size].
            attention_mask (Tensor): The input attention mask of shape [batch_size, seq_len], where 1 indicates that
                the corresponding token should be attended while 0 means not.
            position_ids (Optional[Tensor]): The input position ids of shape [batch_size, seq_len] with the
                values in range [-1e9, 1e9].
            output_attentions (:obj:`bool`, `optional`): Whether or not to return the attentions tensors of each
                layer.

        Returns:
            Tensor: The output tensor of recomputed layer, same size as hidden state.

        """

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, output_gate_logits=False)

            return custom_forward

        decoderlayer_act_offload_settings = self.config.get(
            "decoderlayer_act_offload_settings", {"type": "", "value": ""}
        )

        setting_type = decoderlayer_act_offload_settings["type"]
        offload_value = decoderlayer_act_offload_settings["value"]

        def get_offload_kwargs(layer_idx, setting_type, offload_value):
            offload_kwargs = {}
            if "mod" == setting_type:
                assert isinstance(offload_value, (list, tuple))
                v1, v2 = offload_value
                offload_kwargs["offload_indices"] = [0] if layer_idx % v1 == v2 else []
            elif "layer_idxs" == setting_type:
                offload_kwargs["offload_indices"] = (
                    [0] if layer_idx in offload_value else []
                )
            return offload_kwargs

        layer_idx = layer_module.layer_idx
        # NOTE: the first layer inputs will be used in mtp, so do not offload it
        if layer_idx == 0:
            offload_kwargs = {}
        else:
            offload_kwargs = get_offload_kwargs(layer_idx, setting_type, offload_value)

        recompute_func = te_recompute if self.config.use_fp8 else recompute
        # Fleety11 下 recompute w/ use_cache=True 会出 core
        assert not use_cache, "should not use-recompute during infer(use-cache=True),"

        hidden_states = recompute_func(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            position_ids.clone() if position_ids is not None else None,
            token_type_ids.clone() if token_type_ids is not None else None,
            output_attentions,
            past_key_value,
            use_cache,
            inbatch_pack_offset,
            **offload_kwargs,
        )
        return hidden_states

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        inbatch_pack_offset=None,
        **kwargs,
    ):
        """
        forward
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        seq_length -= self.config.multi_token_pred_depth
        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = paddle.shape(past_key_values[0][0])[1]
            seq_length_with_past += cache_length
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.astype(self.embed_tokens.weight.dtype)

        if self.config.multi_token_pred_depth > 0:
            inputs_embeds_extra = inputs_embeds[
                :, -self.config.multi_token_pred_depth :, :
            ]  # [B, S, D]
            inputs_embeds = inputs_embeds[:, : -self.config.multi_token_pred_depth, :]
            inputs_embeds_ori = inputs_embeds

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        # embed positions
        can_use_fa = self.config.use_flash_attn and flash_attention is not None
        can_mem_eff_attn = (
            self.config.use_mem_eff_attn and inbatch_pack_offset is not None
        )
        if can_use_fa or can_mem_eff_attn:
            if attention_mask is not None:
                attention_mask = None
                # logger.warning(
                #     f"set attention_mask = None when (can_use_fa or can_mem_eff_attn) "
                #     f"and attention_mask is not None, "
                #     f"can_use_fa = {can_use_fa}, can_mem_eff_attn = {can_mem_eff_attn}, "
                #     f"attention_mask is not None = {attention_mask is not None}"
                # )
        elif attention_mask is None:
            attention_mask = paddle.ones(
                (batch_size, seq_length_with_past), dtype=paddle.bool
            )

        if attention_mask is not None:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                cache_length,
                inputs_embeds.dtype,
            )
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_router_loss = 0.0 if self.config.use_moe else None
        all_gate_logits = ()
        mtp_outputs = []

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )
            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.use_recompute
                and has_gradient
                and idx < self.config.recompute_num_layers
            ):
                layer_outputs = self.recompute_training(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    token_type_ids,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    inbatch_pack_offset,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    token_type_ids,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    inbatch_pack_offset,
                )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if self.config.use_moe:
                if not (self.config.use_recompute and has_gradient):
                    layer_outputs, gate_logits = layer_outputs[:-1], layer_outputs[-1]
                    all_gate_logits = all_gate_logits + (gate_logits,)

        # Multi Token Prediction
        if self.config.multi_token_pred_depth > 0:
            mtp_outputs.append(hidden_states)

            for depth in range(self.config.multi_token_pred_depth):
                # sp并行，先获取完整的上一层输出
                if self.config.sequence_parallel or self.config.submatrix_parallel:
                    hidden_states = GatherOp.apply(hidden_states)
                    hidden_states = hidden_states.reshape(
                        [-1, seq_length, hidden_states.shape[-1]]
                    )

                # 构建输入向量
                inputs_embeds_cur_depth = paddle.concat(
                    [
                        inputs_embeds_ori[:, (depth + 1) :, :],
                        inputs_embeds_extra[:, : (depth + 1), :],
                    ],
                    axis=1,
                )
                # inputs_embeds_cur_depth += hidden_states # sum

                # Norm&Concat
                inputs_embeds_cur_depth_norm = self.mtp_emb_norm[depth](
                    inputs_embeds_cur_depth
                )
                hidden_states_norm = self.mtp_hidden_norm[depth](hidden_states)

                inputs_embeds_cur_depth = self.mtp_linear_proj[depth](
                    paddle.concat(
                        [inputs_embeds_cur_depth_norm, hidden_states_norm], axis=-1
                    )
                )

                if self.config.sequence_parallel or self.config.submatrix_parallel:
                    inputs_embeds_cur_depth = inputs_embeds_cur_depth.reshape(
                        [-1, inputs_embeds_cur_depth.shape[-1]]
                    )
                    inputs_embeds_cur_depth = ScatterOp.apply(inputs_embeds_cur_depth)

                # 通过该层的decoder_layer进行预测
                decoder_layer = self.mtp_block[depth]
                past_key_value = None
                layer_outputs = decoder_layer(
                    inputs_embeds_cur_depth,
                    attention_mask,
                    position_ids,
                    token_type_ids,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    inbatch_pack_offset,
                )

                if isinstance(layer_outputs, (tuple, list)):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs

                if self.config.use_moe:
                    if not (self.config.use_recompute and has_gradient):
                        layer_outputs, gate_logits = (
                            layer_outputs[:-1],
                            layer_outputs[-1],
                        )
                        all_gate_logits = all_gate_logits + (gate_logits,)

                mtp_outputs.append(hidden_states)
            mtp_outputs = [self.norm(hidden_states) for hidden_states in mtp_outputs]
            hidden_states, mtp_outputs = mtp_outputs[0], mtp_outputs[1:]
        else:
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_loss,
                    all_gate_logits,
                    mtp_outputs,
                ]
                if v is not None
            )

        # assert all_router_loss is None, f'moe 不支持`return-dict`'
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
            router_loss=all_router_loss,
            gate_logits=all_gate_logits,
            mtp_outputs=mtp_outputs,
        )


ErnieMoELMHead = ErnieLMHead


class ErniePretrainingCriterion(ErniePretrainingCriterionBase):
    """
    Criterion for Ernie.
    It calculates the final loss.
    """

    def __init__(self, config, return_tuple=True):
        """
        用于ERNIE预训练任务的损失函数基类。

        Args:
            config (PretrainingConfig): ERNIE模型的配置对象。
            return_tuple (bool，可选): 如果为True，则返回损失函数的值为一个元组（loss），否则返回标量值。默认值为True。

        Returns:
            None
        """
        super(ErniePretrainingCriterion, self).__init__(
            config, return_tuple=return_tuple
        )
        self.ignored_index = getattr(config, "ignored_index", -100)
        self.config = config
        self.return_tuple = return_tuple
        self.enable_parallel_cross_entropy = (
            config.tensor_parallel_degree > 1
            and config.tensor_parallel_output
            and not in_auto_parallel_align_mode()
        )

        if (
            self.enable_parallel_cross_entropy
        ):  # and False: # and lm_head is distributed
            logger.info("using parallel cross entroy, take care")
            self.loss_func = fleet.meta_parallel.ParallelCrossEntropy()
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(
                reduction="none",
            )

    def forward(
        self, prediction_scores, masked_lm_labels, router_loss=None, mtp_logits=None
    ):
        """Loss Calculate"""
        if self.config.multi_token_pred_depth > 0:
            masked_lm_labels_ori = masked_lm_labels
            masked_lm_labels = masked_lm_labels[
                :, : -self.config.multi_token_pred_depth
            ]
            seq_length = masked_lm_labels.shape[1]
        res = super().forward(
            prediction_scores,
            masked_lm_labels,
        )
        global_training_logs = (
            get_global_training_logs()
        )  # 没有erniebot的环境下无法打印 debug 量

        if self.config.multi_token_pred_depth > 0:
            global_training_logs.update(mtp_depth_0_loss=res[0].clone().detach())
            mtp_loss_res = []
            for depth in range(self.config.multi_token_pred_depth):
                prediction_scores_cur_depth = mtp_logits[depth]
                masked_lm_labels_cur_depth = masked_lm_labels_ori[
                    :, (depth + 1) : (depth + 1 + seq_length)
                ]
                res_cur_depth = super().forward(
                    prediction_scores_cur_depth,
                    masked_lm_labels_cur_depth,
                )
                mtp_loss_res.append(res_cur_depth)
                global_training_logs.update(
                    **{f"mtp_depth_{depth + 1}_loss": res_cur_depth[0].clone().detach()}
                )

        def add_loss(main_loss, loss):
            return main_loss + loss - loss.detach()

        if self.return_tuple:
            loss, loss_sum = res
            if self.config.multi_token_pred_depth > 0:
                loss = add_loss(
                    loss,
                    self.config.multi_token_pred_lambda
                    * sum([x[0] for x in mtp_loss_res])
                    / len(mtp_loss_res),
                )
                loss_sum = loss_sum + self.config.multi_token_pred_lambda * sum(
                    [x[1].detach() for x in mtp_loss_res]
                ) / len(mtp_loss_res)
        else:
            loss, loss_sum = res, None
            if self.config.multi_token_pred_depth > 0:
                loss = add_loss(
                    loss,
                    self.config.multi_token_pred_lambda
                    * sum([x[0] for x in mtp_loss_res])
                    / len(mtp_loss_res),
                )

        global_training_logs.update(lm_loss=loss.clone().detach())
        if (
            router_loss is not None
            and isinstance(router_loss, paddle.Tensor)
            and not in_auto_parallel_align_mode()
        ):
            loss = loss + router_loss - router_loss.detach()
            if isinstance(router_loss, paddle.Tensor):
                global_training_logs.update(router_loss=router_loss.detach())
        return loss, loss_sum


class ErnieMoEForCausalLM(ErniePretrainedModel):
    """_summary_

    Args:
        ErniePretrainedModel (_type_): _description_

    Returns:
        _type_: _description_
    """

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        """
        初始化类，初始化ERNIE模型。

        Args:
            config (`obj:`dict`): 包含ERNIE模型配置信息的字典对象，包含以下参数：

                - ``use_ms_token`` (`bool`): 是否使用Microsoft Tokenizer进行预处理。默认值为False。
                - ``model_name_or_path`` (`str`): 指定用于预训练BERT或RoBERTa模型的文件路径或者名称。默认值为"bert-base-uncased"。
                - ``max_seq_length`` (`int`): BERT输入序列长度上限。默认值为512。
                - ``vocab_file`` (`str`): 指定词汇表文件路径。默认值为None。
                - ``num_labels`` (`int`): 标签数量。默认值为2。
                - ``hidden_dropout_prob`` (`float`): 隐含层Dropout概率。默认值为0.1。
                - ``attention_probs_dropout_prob`` (`float`): Attention Probability Dropout概率。默认值为0.1。
                - ``intermediate_size`` (`int`): 中间层大小。默认值为4096。
                - ``layer_num`` (`int`): Transformer块数。默认值为12。
                - ``use_rmsnorm`` (`bool`): 是否使用RMSNorm。默认值为True。
                - ``fuse_rms_norm`` (`bool`): 是否融合使用RMSNorm和LayerNorm两种方式。默认值为False。
                - ``fuse_ln`` (`bool`): 是否融合使用LayerNorm和GroupNorm两种方式。默认值为False。
                - ``initializer_range`` (`float`): 默认为0.02。
                - ``use_bfloat16`` (`bool`): 是否使用BFloat16精度计算。默认值为False。

        Returns:
            None
        """
        super().__init__(config)

        if config.sequence_parallel:
            logger.info(f"using sequence_parallel, input seqlen={config.seqlen}")
            assert config.seqlen is not None
            assert (
                config.tensor_parallel_degree > 1
            ), f"sequence-parallel needs mp>1, got mp={config.tensor_parallel_degree}"

        # initialize-trick for big model,
        # see https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/README.md#std-init
        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(
            f"change initializer-range from {config.initializer_range} to {new_initializer_range}"
        )
        config.initializer_range = new_initializer_range
        self.config = config
        self.ernie = ErnieModel(config)
        self.lm_head = ErnieMoELMHead(config)
        self.criterion = ErniePretrainingCriterion(config)

        self.tie_weights()  # maybe weight share

        if self.config.use_rmsnorm:
            if self.config.fuse_rms_norm:
                logger.info("Use fusedRMSNorm")
            else:
                logger.info("Use normal RMSNorm")
        else:
            if self.config.fuse_ln:
                logger.info("Use fusedLN")
            else:
                logger.info("Use normal LayerNorm")

    # Initialize weights and apply final processing
    def _post_init(self, original_init, *args, **kwargs):
        """
        初始化方法，用于在模型初始化之后执行一些操作。

        Args:
            self (LARKERForSequenceClassification): 当前的LARKERForSequenceClassification对象。
            original_init (callable): 原始的初始化方法。
            args (tuple): 参数列表。
            kwargs (dict): 参数字典。

        Returns:
            None: 不返回任何值。

        """
        super()._post_init(self, original_init, *args, **kwargs)
        factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
        logger.info(f"using post init div: factor:{factor}")
        with paddle.no_grad():
            for l in self.ernie.layers:
                if self.config.use_linear_residual_norm_recompute is True:
                    l.fused_linear_add_norm.linear_weight.scale_(factor)
                else:
                    if isinstance(
                        l.self_attn.o_proj,
                        (
                            MOELayer,
                            MOEAllGatherLayer,
                            MOEAllGatherLayerV2,
                            MOELayerWithAllGatherDispatcher,
                            MOEInferLayer,
                            MOELayerSizeAll2All,
                            DeepEPMOELayer,
                            DeepEPDropTokenMOELayer,
                        ),
                    ):
                        for e in l.self_attn.o_proj.experts:
                            e.weight.scale_(factor)
                        if hasattr(l.self_attn.o_proj, "dense_experts"):
                            l.self_attn.o_proj.dense_experts.down_proj.weight.scale_(
                                factor
                            )
                    else:
                        l.self_attn.o_proj.weight.scale_(factor)

                if isinstance(
                    l.mlp,
                    (
                        MOELayer,
                        MOEAllGatherLayer,
                        MOEAllGatherLayerV2,
                        MOELayerWithAllGatherDispatcher,
                        MOEInferLayer,
                        MOELayerSizeAll2All,
                        DeepEPMOELayer,
                        DeepEPDropTokenMOELayer,
                    ),
                ):
                    for e in l.mlp.experts:
                        if isinstance(e, ErnieMLP):
                            e.down_proj.weight.scale_(factor)
                    if getattr(l.mlp, "dense_experts", None) and isinstance(
                        l.mlp.dense_experts, ErnieMLP
                    ):
                        l.mlp.dense_experts.down_proj.weight.scale_(factor)
                else:
                    l.mlp.down_proj.weight.scale_(factor)

    def state_dict(self, *args, **kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        state_dict = super().state_dict(*args, **kwargs)
        # moe state dict 转换成全局id.
        return moe_statedict_local_id_to_global(state_dict, self.config)

    def set_state_dict(self, state_dict, *args, **kwargs):
        """
        自动加载 dense 网络的 state-dict
        """
        # 自动扩展dense ckpt
        state_dict = moe_statedict_upcycle(
            state_dict,
            self.config,
            self.lm_head.weight.dtype,
            self._get_tensor_parallel_mappings(self.config, is_split=False),
            self._get_tensor_parallel_mappings(self.config, is_split=True),
        )
        state_dict = moe_statedict_cherry_pick(state_dict, self.config)
        # 自动扩展ep2mp
        state_dict = moe_ep2mp(
            state_dict,
            self.config,
            self._get_tensor_parallel_mappings(self.config, is_split=True),
        )
        if not isinstance(self.config.moe_num_experts, (list, tuple)):
            logger.info("convert mm correction_bias to lm")
            state_dict = {
                k: v[:1] if "e_score_correction_bias" in k else v
                for k, v in state_dict.items()
            }
        ret = super().set_state_dict(state_dict, *args, **kwargs)
        logger.info(f"set_state_dict: {ret}")
        return ret

    def get_input_embeddings(self):
        """
        获取输入嵌入对象。

        Args:
            无。

        Returns:
            torch.nn.Module: 嵌入对象。

        """
        return self.ernie.embed_tokens

    def set_input_embeddings(self, value):
        """设置输入嵌入。

        Args:
            value (torch.nn.Embedding): 输入嵌入对象。

        Returns:
            None.

        """
        self.ernie.embed_tokens = value

    def get_output_embeddings(self):
        """
        获取输出嵌入

        Args:
            无参数

        Returns:
            nn.Module: 返回模型的 LM head 的输出嵌入层
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """设置输出嵌入（LM head）

        Args:
            new_embeddings (torch.nn.Embedding): 新的输出嵌入。

        Returns:
            None

        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        设置解码器

        Args:
            decoder (BertDecoder): 用于解码输入句子的 BertDecoder 对象

        Returns:
            None

        """
        self.ernie = decoder

    def get_decoder(self):
        """
        获取解码器

        Args:
            无

        Returns:
            Optional[nn.Module]: 返回解码器，如果未指定则返回None

        """
        return self.ernie

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        """
        根据给定的输入ID、填充Token ID和句末标记ID，生成对应需要的注意力掩码。

        Args:
            input_ids (`paddle.Tensor`): 输入ID，形状为[batch_size * seq_length]。
            pad_token_id (`paddle.Tensor`, optional): 填充Token ID，默认值为None。
            eos_token_id (`paddle.Tensor`, optional): 句末标记ID，默认值为None。

        Returns:
            `paddle.Tensor`: 生成的注意力掩码，形状为[batch_size * seq_length, seq_length]。

        """
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(
            input_ids == pad_token_id
        ).numpy().item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids != pad_token_id).astype("int64")
        else:
            attention_mask = paddle.ones_like(input_ids, dtype="int64")
        return attention_mask

    def prepare_inputs_for_generation(
        self,
        input_ids,
        use_cache=False,
        past_key_values=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """
        Prepare inputs for the decoder that will be used in the generation phase.

        Args:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`({0}, {1})`):
                Indices of input sequence tokens in the vocabulary. Padding will
                automatically be appended to the input as needed based on the length
                of the ``input_ids`` tensor.

                If :obj:`config.is_encoder_decoder=False`, :obj:`input_ids` should be a
                sequence of tokens that the BART model will make predictions for.
                In this case it is usually expected that :obj:`len(input_ids)` == :obj:`num_beams`.
                However, no validation check will be performed in this case because
                no other constraints are placed on generated sequences by the
                framework except for the aforementioned ``num_beams``.
            use_cache (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use cached hidden-states under :obj:`past_key_values` key of
                the dictionary returned by the model. Setting this option to :obj:`True` can
                save memory while speeding up generation, since the model does not have to
                store the intermediates activations required to generate generations
                from scratch.
            past_key_values (:obj:`Dict[str, torch.FloatTensor]`, `optional`):
                Dictionary with pre-computed hidden-states (key: str, value: Tensor) to speed up decoding.
                Should be set to speed up decoding through external memory (e.g.: beam search).
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, {1}, {2})`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids`
                indices into associated embeddings.
            kwargs (`optional`):
                Additional parameters passed along to the model.

        Returns:

            Dict:
                Contains all the model inputs and optional additional inputs/outputs of interest.
        """
        if past_key_values:
            input_ids = input_ids[:, -1:]

        attention_mask = kwargs.get("attention_mask", None)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": True,  # use_cache,
                "attention_mask": attention_mask,
                "return_dict": True,
            }
        )

        if self.config.rope_3d:
            model_inputs.update({"position_ids": kwargs["position_ids"]})

        return model_inputs

    def update_model_kwargs_for_generation(
        self, outputs, model_kwargs, is_encoder_decoder=False
    ):
        """
        更新模型参数以进行生成。

        Args:
            outputs (Any): 模型输出，可能是一个tuple或一个类实例。
            model_kwargs (Dict): 模型参数字典。
            is_encoder_decoder (bool, optional): 是否为编码器解码器模式，默认值为False。

        Returns:
            Dict: 更新后的模型参数字典。
        """
        # update cache
        if (
            isinstance(outputs, tuple)
            and len(outputs) > 1
            and not isinstance(outputs[1], paddle.Tensor)
        ):
            model_kwargs["past_key_values"] = outputs[1]

        if (
            isinstance(outputs, CausalLMOutputWithCrossAttentions)
            and "past_key_values" in outputs
        ):
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update token_type_ids with last value
        if (
            "token_type_ids" in model_kwargs
            and model_kwargs["token_type_ids"] is not None
        ):
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat(
                [token_type_ids, token_type_ids[:, -1:]], axis=-1
            )

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = paddle.concat(
                    [
                        attention_mask,
                        paddle.ones([attention_mask.shape[0], 1], dtype="int64"),
                    ],
                    axis=-1,
                )
        # update role_ids
        if "role_ids" in model_kwargs and model_kwargs["role_ids"] is not None:
            role_ids = model_kwargs["role_ids"]
            model_kwargs["role_ids"] = paddle.concat(
                [role_ids, role_ids[:, -1:]], axis=-1
            )

        if self.config.rope_3d:
            assert (
                "position_ids" in model_kwargs
            ), "position_ids must be provided if rope_3d is on"
            position_ids = model_kwargs["position_ids"]
            bsz = position_ids.shape[0]

            # becasue the model can only generate text.
            model_kwargs["position_ids"] = paddle.concat(
                [
                    position_ids,
                    position_ids.max(axis=(1, 2), keepdim=True).tile([1, 1, 3]) + 1,
                ],
                axis=1,
            )

        return model_kwargs

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        token_type_ids=None,  # 用于 moe 进行 token-type 路由。
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,  # decode 时需要return-dict, pretrain & eval时不需要。
        ignored_index=0,  # no use
        data_id=None,  # no use
        src_id=None,  # no use
        inbatch_pack_offset=None,
    ):
        """ """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        with fp8_autocast(self.config.use_fp8, self.config.fp8_configs["recipe"]):
            outputs = self.ernie(
                input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                inbatch_pack_offset=inbatch_pack_offset,
            )

        hidden_states = outputs.last_hidden_state
        mtp_outputs = outputs.mtp_outputs
        # if labels is None，means we need full output, instead of tensor_parallel_output
        # tensor_parallel_output is togather with ParallelCrossEntropy
        tensor_parallel_output = (
            self.config.tensor_parallel_output
            and labels is not None
            and self.config.tensor_parallel_degree > 1
        )

        logits = self.lm_head(hidden_states)
        mtp_logits = []
        if len(mtp_outputs) > 0:
            mtp_logits = [
                self.lm_head(_hidden_states) for _hidden_states in mtp_outputs
            ]

        if return_dict:  # aka Generate Decoding
            if labels is not None:
                loss, _ = self.criterion(logits, labels)
            else:
                loss = None
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_loss=outputs.router_loss if self.config.use_moe else None,
            )
        # Pretrain & Eval 必须有labels
        if self.config.use_moe:
            router_loss = outputs.router_loss
        else:
            router_loss = None
        assert labels is not None
        return self.criterion(logits, labels, router_loss, mtp_logits)
