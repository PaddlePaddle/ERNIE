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
import os
import copy
import math
from functools import partial
import logging
from typing import Optional, Tuple
import contextlib
import inspect

try:
    from fast_ln import fast_ln
except ImportError:
    fast_ln = None

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker
from paddle.incubate.nn.memory_efficient_attention import (
    memory_efficient_attention,
    BlockDiagonalCausalMask,
)
from paddle.distributed.fleet.layers.mpu import mp_ops


# paddlenlp import
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
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from paddleformers.transformers.model_utils import PretrainedModel, register_base_model
from paddleformers.utils.tools import get_env_device
from models.fp8_linear import MemEfficientFp8FusedMlpFunc

# end paddlenlp import


# ernie core import
from .configuration import ErnieConfig
from models.refined_recompute.flash_attn import (
    RefinedRcomputeFlashAttention,
    RefinedRcomputeFlashMaskAttention,
)
from models.comm_utils import ZLossOp, subbatch
from models.sequence_parallel_utils import (
    GatherOp,
    ScatterOp,
    ColumnSequenceParallelLinear,
    RowSequenceParallelLinear,
    mark_as_sequence_parallel_parameter,
    AllGatherVarlenOp,
    sequence_parallel_sparse_mask_labels,
)
from models.align_mode_utils import concat_mp_with_grad

from models.submatrix_parallel import (
    SMPRowParallelLinear,
    SMPColumnParallelLinear,
)
from models.utils import global_training_logs_enabled, get_global_training_logs


# end ernie core import

# try import

logger = logging.getLogger(__name__)

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
        logger.warning("[WARN] import paddle_xpu for mp error!")

    try:
        from paddle_xpu.layers.nn.sequence_parallel import GatherOp as XPUGatherOp
        from paddle_xpu.layers.nn.sequence_parallel import ScatterOp as XPUScatterOp

        GatherOp = XPUGatherOp
        ScatterOp = XPUScatterOp
    except ImportError:
        logger.warning("[WARN] import paddle_xpu for sp error!")

    if not os.getenv("XPU_MOE_USE_ALLGATHER", False):
        try:
            from paddle_xpu.layers.nn.sequence_parallel import (
                XPUColumnSequenceParallelLinear,
                XPURowSequenceParallelLinear,
            )

            ColumnSequenceParallelLinear = XPUColumnSequenceParallelLinear
            RowSequenceParallelLinear = XPURowSequenceParallelLinear
        except ImportError:
            logger.warning("[WARN] import paddle_xpu for sp error!")

    try:
        from paddle.incubate.nn.functional import (
            fused_rotary_position_embedding as xpu_fuse_rope,
        )
    except ImportError:
        xpu_fuse_rope = None


try:
    from paddle.nn.functional.flash_attention import flash_attention

    logger.warning(
        "Use flash attention in scaled-dot-product. Attention mask is deprecated"
    )
except (ImportError, ModuleNotFoundError):
    flash_attention = None

try:
    from paddle.nn.functional.flash_attention import flash_attention_with_mask
except (ImportError, ModuleNotFoundError):
    try:
        from paddle.nn.functional.flash_attention import (
            scaled_dot_product_attention as flash_attention_with_mask,
        )
    except (ImportError, ModuleNotFoundError):
        logger.warning(
            "flash_attention_with_mask not found. Use FleetY8.2 SFT instead."
        )
        flash_attention_with_mask = None

try:
    from paddle.nn.functional.flash_attention import flash_attention_with_sparse_mask
except (ImportError, ModuleNotFoundError):
    logger.warning("flash_attention_with_sparse_mask not found. Use FleetY8.9 instead.")
    flash_attention_with_sparse_mask = None

try:
    from paddle.nn.functional.flash_attention import flashmask_attention
except (ImportError, ModuleNotFoundError):
    logger.warning(
        "flashmask_attention not found. Use the latest version of FleetY10 instead."
    )
    flashmask_attention = None

if get_env_device() == "xpu":
    try:
        from paddle_xpu.layers.nn import varlen_flash_attention as varlen_fa
    except ImportError:
        logger.warning("var len not found. Use the latest version of xpu image.")
        varlen_fa = None

try:
    from to_block_diag_causal_mask import to_block_diag_causal_mask
except (ImportError, ModuleNotFoundError):
    logger.warning("to_block_diag_causal_mask not found. Use FleetY8.2 SFT instead.")
    to_block_diag_causal_mask = None

try:
    import fused_ln as fused
except ImportError:
    logger.warning(
        "fused-ln not found, run `python src/ops/fused_ln_setup.py install` to build fused ln"
    )
    fused = None

try:
    from paddle.incubate.nn.functional import (
        fused_rotary_position_embedding as fused_rope,
    )
except (ImportError, ModuleNotFoundError):
    logger.warning("fused_rotary_position_embedding not found")
    fused_rope = None

try:
    from paddle.incubate.nn.functional import swiglu as fused_swiglu
except (ImportError, ModuleNotFoundError):
    fused_swiglu = None


try:
    from paddle.distributed import in_auto_parallel_align_mode
except:

    def in_auto_parallel_align_mode():
        """
        hack for paddlenlp develop branch.
        """
        return False


try:
    import moe_router_loss_ops
except ImportError:
    moe_router_loss_ops = None

# end try import

# Because param_name is generated based on the class name,
# when changes in distributed strategies result in class modifications,
# there may be mismatches during parameter loading.
# You can achieve class name changes by importing the following environment variables.
# Example: `export rowcol_parallel_linear_class_name_convert_map="tpsp->smp"`

if "rowcol_parallel_linear_class_name_convert_map" in os.environ:
    rowcol_parallel_linear_class_name_convert_map = os.environ[
        "rowcol_parallel_linear_class_name_convert_map"
    ]
    assert rowcol_parallel_linear_class_name_convert_map in [
        "tp->tpsp",
        "tpsp->tp",
        "tp->smp",
        "smp->tp",
        "tpsp->smp",
        "smp->tpsp",
    ], (
        "rowcol_parallel_linear_class_name_convert_map must in "
        "['tp->tpsp', 'tpsp->tp', 'tp->smp', 'smp->tp', 'tpsp->smp', 'smp->tpsp'], "
        "but got {}".format(rowcol_parallel_linear_class_name_convert_map)
    )
    if rowcol_parallel_linear_class_name_convert_map == "tp->tpsp":
        ColumnSequenceParallelLinear.__name__ = "ColumnParallelLinear"
        RowSequenceParallelLinear.__name__ = "RowParallelLinear"
    elif rowcol_parallel_linear_class_name_convert_map == "tpsp->tp":
        ColumnParallelLinear.__name__ = "ColumnSequenceParallelLinear"
        RowParallelLinear.__name__ = "RowSequenceParallelLinear"
    elif rowcol_parallel_linear_class_name_convert_map == "tp->smp":
        SMPColumnParallelLinear.__name__ = "ColumnParallelLinear"
        SMPRowParallelLinear.__name__ = "RowParallelLinear"
    elif rowcol_parallel_linear_class_name_convert_map == "smp->tp":
        ColumnParallelLinear.__name__ = "SMPColumnParallelLinear"
        RowParallelLinear.__name__ = "SMPRowParallelLinear"
    elif rowcol_parallel_linear_class_name_convert_map == "tpsp->smp":
        SMPColumnParallelLinear.__name__ = "ColumnSequenceParallelLinear"
        SMPRowParallelLinear.__name__ = "RowSequenceParallelLinear"
    elif rowcol_parallel_linear_class_name_convert_map == "smp->tpsp":
        ColumnSequenceParallelLinear.__name__ = "SMPColumnParallelLinear"
        RowSequenceParallelLinear.__name__ = "SMPRowParallelLinear"


ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST = []

__all__ = [
    "ErnieModel",
    "ErniePretrainedModel",
    "ErnieForCausalLM",
]


# NOTE(zengjinle): do not use this method!
def is_fleety_func():
    """
    Check whether it is PaddlePaddle FleetY version.
    """
    if flash_attention_with_sparse_mask is None:
        return True

    args = inspect.getfullargspec(flash_attention_with_sparse_mask).args
    return "causal" in args


IS_FLEETY = is_fleety_func()

try:
    """doc"""
    from contextlib import contextmanager
    import transformer_engine.paddle as te
    from transformer_engine.paddle import recompute as te_recompute
    from transformer_engine.common.recipe import DelayedScaling, Format

    _IS_TRANSFORMER_ENGINE_INSTALLED = True

    class TEFP8Linear(te.Linear):
        """doc"""

        def __init__(
            self,
            in_features,
            out_features,
            weight_attr=None,
            bias_attr=None,
            parallel_mode=None,
            sequence_parallel=False,
            tp_group=None,
            fuse_wgrad_accumulation=False,
            gather_output=False,
            quant_weight_at_first_microbatch=False,
        ) -> None:
            super().__init__(
                in_features,
                out_features,
                weight_attr,
                bias_attr,
                parallel_mode,
                sequence_parallel,
                tp_group,
                fuse_wgrad_accumulation,
                gather_output,
            )
            strategy = fleet.fleet._user_defined_strategy
            self.num_steps = strategy.pipeline_configs["accumulate_steps"]
            self._micro_batch_id = 0
            self.quant_weight_at_first_microbatch = quant_weight_at_first_microbatch

        def forward(self, *args, **kwargs):
            """doc"""
            is_first_microbatch = None
            kwargs["micro_step"] = self._micro_batch_id

            if self.quant_weight_at_first_microbatch:
                is_first_microbatch = (self._micro_batch_id % self.num_steps) == 0

            if paddle.is_grad_enabled() and self.training:
                self._micro_batch_id = (self._micro_batch_id + 1) % self.num_steps

            kwargs["is_first_microbatch"] = is_first_microbatch
            return super().forward(*args, **kwargs)

except ModuleNotFoundError:
    te = None
    te_recompute = recompute
    _IS_TRANSFORMER_ENGINE_INSTALLED = False

    # fake TEFP8Linear for fix instance(layer, TEFP8Linear) error

    class TEFP8Linear(paddle.nn.Linear):
        """doc"""

        def __init__(
            self,
            in_features,
            out_features,
            weight_attr=None,
            bias_attr=None,
            parallel_mode=None,
            sequence_parallel=False,
            tp_group=None,
            fuse_wgrad_accumulation=False,
            gather_output=False,
            quant_weight_at_first_microbatch=False,
        ) -> None:
            super().__init__(in_features, out_features, weight_attr, bias_attr)
            pass


def is_te_installed():
    """doc"""
    return _IS_TRANSFORMER_ENGINE_INSTALLED


@contextmanager
def fp8_autocast(
    enabled=False,
    recipe={
        "format": "hybrid",
        "calibrating": True,
        "amax_history_len": 1024,
        "amax_compute_algo": "max",
    },
):
    """doc"""
    if is_te_installed():
        calibrating = recipe["calibrating"]
        fp8_dtype = recipe["format"]
        amax_history_len = recipe["amax_history_len"]
        amax_compute_algo = recipe["amax_compute_algo"]
        if fp8_dtype == "e4m3":
            fp8_format = (
                Format.E4M3
            )  # E4M3 during forward pass, E4M3 during backward pass
        elif fp8_dtype == "hybrid":
            fp8_format = (
                Format.HYBRID
            )  # E4M3 during forward pass, E5M2 during backward pass
        else:
            raise NotImplementedError(f"{fp8_dtype} not support")
        fp8_recipe = DelayedScaling(
            fp8_format=fp8_format,
            calibrating=calibrating,
            amax_history_len=amax_history_len,
            amax_compute_algo=amax_compute_algo,
        )
        try:
            # FIXME: in order to reduce communication overhead, we only sync amax among TP group
            hcg = fleet.get_hybrid_communicate_group()
            fp8_group = hcg.get_model_parallel_group()
        except:
            fp8_group = None
        with te.fp8_autocast(
            enabled=enabled,
            fp8_group=fp8_group,
            calibrating=calibrating,
            fp8_recipe=fp8_recipe,
        ):
            yield
    else:  # null context
        yield


def get_triangle_upper_mask(x, mask=None):
    if mask is not None:
        return mask
    # [bsz, n_head, q_len, kv_seq_len]
    shape = x.shape
    #  [bsz, 1, q_len, kv_seq_len]
    shape[1] = 1
    mask = paddle.full(shape, -np.inf, dtype=x.dtype)
    mask.stop_gradient = True
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


def gqa_qkv_split_func(
    weight,
    tensor_parallel_degree,
    tensor_parallel_rank,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
):
    """Split GQA QKV weight / bias in tensor parallel mode."""
    if not isinstance(weight, paddle.Tensor):
        weight = paddle.Tensor(weight[:], zero_copy=True, place=paddle.CPUPlace())
    q, k, v = paddle.split(
        weight,
        [
            num_attention_heads * head_dim,
            num_key_value_heads * head_dim,
            num_key_value_heads * head_dim,
        ],
        axis=-1,
    )
    if tensor_parallel_rank is None:
        q_list = paddle.split(q, tensor_parallel_degree, axis=-1)
        k_list = paddle.split(k, tensor_parallel_degree, axis=-1)
        v_list = paddle.split(v, tensor_parallel_degree, axis=-1)
        ret = [
            paddle.concat([q, k, v], axis=-1) for q, k, v in zip(q_list, k_list, v_list)
        ]
        return ret
    else:
        q = paddle.split(q, tensor_parallel_degree, axis=-1)[tensor_parallel_rank]
        k = paddle.split(k, tensor_parallel_degree, axis=-1)[tensor_parallel_rank]
        v = paddle.split(v, tensor_parallel_degree, axis=-1)[tensor_parallel_rank]
        return paddle.concat([q, k, v], axis=-1)


def gqa_qkv_merge_func(weight_list, num_attention_heads, num_key_value_heads, head_dim):
    """Merge GQA QKV weight / bias in tensor parallel mode."""
    tensor_parallel_degree = len(weight_list)
    num_attention_heads = num_attention_heads // tensor_parallel_degree
    num_key_value_heads = num_key_value_heads // tensor_parallel_degree
    q_list, k_list, v_list = [], [], []
    for weight in weight_list:
        q, k, v = paddle.split(
            weight,
            [
                num_attention_heads * head_dim,
                num_key_value_heads * head_dim,
                num_key_value_heads * head_dim,
            ],
            axis=-1,
        )
        q_list.append(q)
        k_list.append(k)
        v_list.append(v)
    return paddle.concat(q_list + k_list + v_list, axis=-1)


def naive_fuse_split_tp(
    weight,
    tensor_parallel_degree,
    tensor_parallel_rank=None,
    is_column=True,
    fuse_tensor_parts=2,
):
    """doc"""
    logging.info(f"spliting fused-ffn: {weight.shape}")
    axis = -1 if is_column else 0
    splited = np.split(weight, fuse_tensor_parts * tensor_parallel_degree, axis=axis)
    return np.concatenate(
        splited[tensor_parallel_rank::tensor_parallel_degree], axis=axis
    )


def parallel_matmul(
    x,
    y,
    bias=None,
    transpose_y=False,
    tensor_parallel_degree=1,
    tensor_parallel_output=True,
    fuse_linear=False,
    training=True,
):
    """doc"""
    if tensor_parallel_degree > 1:
        if isinstance(y, paddle.base.framework.EagerParamBase):
            assert y.is_distributed
        # if not running under distributed.launch, it will raise AttributeError: 'Fleet' object has no attribute '_hcg'
        if get_env_device() == "xpu":
            try:
                from paddle_xpu.layers.nn import parallel_matmul as xpu_parallel_matmul

                xpu_parallel_matmul = xpu_parallel_matmul()
                logits = xpu_parallel_matmul(
                    x,
                    y,
                    tensor_parallel_output=tensor_parallel_output,
                    training=training,
                )
                if bias is not None:
                    logits += bias
                return logits
            except ImportError:
                pass
        pg = (
            fleet.get_hybrid_communicate_group().get_model_parallel_group()
        )  # TODO:尽量避免从环境中拿变量（防止有卡内有多个模型时混淆），至少有的非MP、PP模型，不要需要做此拆分。
        input_parallel = paddle.distributed.collective._c_identity(x, group=pg)
        # TODO: should we _c_identity this bias?
        if transpose_y:
            logits = paddle.matmul(input_parallel, y, transpose_y=True)
            if bias is not None:
                logits += bias
        else:
            if fuse_linear:
                logits = paddle.incubate.nn.functional.fused_linear(
                    input_parallel, y, bias
                )  # hack for 逐位对齐
            else:
                logits = F.linear(input_parallel, y, bias)

        if tensor_parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(logits, group=pg)

    else:
        try:
            from paddle_xpu.layers.nn import xpu_matmul

            xpu_matmul = xpu_matmul()
        except ImportError:
            xpu_matmul = None
        if get_env_device() == "xpu" and xpu_matmul is not None:
            logits = xpu_matmul(x, y, transpose_y=transpose_y, training=training)
            if bias is not None:
                logits += bias
        else:
            if fuse_linear:
                logits = paddle.incubate.nn.functional.fused_linear(
                    x, y, bias, transpose_weight=transpose_y
                )
            else:
                logits = paddle.matmul(x, y, transpose_y=transpose_y)
                if bias is not None:
                    logits += bias
        return logits


def calc_lm_head_logits(
    config, hidden_states, weight, bias, tensor_parallel_output=None, training=True
):
    """the core function to calc lm head"""
    if config.sequence_parallel or config.submatrix_parallel:
        if config.use_sparse_head_and_loss_fn:
            pass  # Nothing needs to be done.
        else:
            lm_head_use_gather = getattr(config, "lm_head_use_gather", True)
            if lm_head_use_gather:
                hidden_states = GatherOp.apply(hidden_states)
            if not config.using_dynamic_sequence_length:
                hidden_states = hidden_states.reshape(
                    [-1, config.seqlen, hidden_states.shape[-1]]
                )
            else:
                assert (
                    config.micro_batch_size
                ), "micro_batch_size should be set when using dygramic sequence length."
                hidden_states = hidden_states.reshape(
                    [config.micro_batch_size, -1, hidden_states.shape[-1]]
                )

    # NOTE(shenliang03): for compatibility with old code, we need to support
    # hidden_states.ndim != 3 when tensor_parallel_degree == 1
    if config.tensor_parallel_degree == 1 and hidden_states.ndim != 3:
        if not config.using_dynamic_sequence_length:
            hidden_states = hidden_states.reshape(
                [-1, config.seqlen, hidden_states.shape[-1]]
            )
        else:
            assert (
                config.micro_batch_size
            ), "micro_batch_size should be set when using dygramic sequence length."
            hidden_states = hidden_states.reshape(
                [config.micro_batch_size, -1, hidden_states.shape[-1]]
            )

    if tensor_parallel_output is None:
        tensor_parallel_output = config.tensor_parallel_output
    logits = parallel_matmul(
        hidden_states,
        weight,
        bias=bias,
        transpose_y=config.tie_word_embeddings,
        tensor_parallel_degree=config.tensor_parallel_degree,
        tensor_parallel_output=tensor_parallel_output,
        fuse_linear=config.fuse_linear,
        training=training,
    )

    return logits


def finfo(dtype: paddle.dtype = None):
    if dtype is None:
        dtype = paddle.get_default_dtype()

    if dtype == paddle.bfloat16:
        # Numpy do not support `np.finfo(np.uint16)`, so try to construct a finfo object to fetch min value
        class BFloatFInfo:
            min = -3.3895313892515355e38

        return BFloatFInfo
    if dtype == paddle.float32:
        return np.finfo(np.float32)
    if dtype == paddle.float16:
        return np.finfo(np.float16)
    if dtype == paddle.float64:
        return np.finfo(np.float64)


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def mem_eff_attn(
    query, key, value, pack_offset, drop_prob=0.0, dtype=paddle.bfloat16, training=True
):
    """doc"""
    pack_offset = pack_offset.numpy()
    shape = pack_offset.shape
    assert len(shape) == 2, len(shape)
    assert shape[0] == 1, shape[0]
    n = pack_offset.size
    pack_offset = pack_offset.flatten()
    seqlens = []
    assert pack_offset[0] == 0, pack_offset[0]
    for i in range(1, n):
        if pack_offset[i] < 0:
            break
        cur = pack_offset[i] - pack_offset[i - 1]
        assert cur > 0
        seqlens.append(cur)

    assert drop_prob == 0.0, drop_prob
    assert dtype == paddle.bfloat16, dtype

    def cast(x):
        return x.astype(dtype) if x.dtype != dtype else x

    if len(seqlens) == 1:
        out, _ = flash_attention(
            query, key, value, drop_prob, causal=True, training=training
        )
    else:
        mask = BlockDiagonalCausalMask.from_seqlens(seqlens)
        out = memory_efficient_attention(
            cast(query),
            cast(key),
            cast(value),
            attn_bias=mask,
            p=drop_prob,
            training=training,
        )
    return out


def inbatch_pack_offset_to_attn_mask_start_row_indices(inbatch_pack_offset):
    """convert inbatch_pack_offset to attn_mask_start_row_indices"""
    inbatch_pack_offset = inbatch_pack_offset.numpy()
    attn_mask_row_start_indices = []
    min_start_row = np.inf
    for bidx in range(inbatch_pack_offset.shape[0]):
        item = inbatch_pack_offset[bidx]
        cumsum_item = item[item != -1]
        record_lens = cumsum_item[1:] - cumsum_item[0:-1]
        min_start_row = min(cumsum_item[1], min_start_row)
        row_start_indices = np.repeat(cumsum_item[1:], record_lens)
        attn_mask_row_start_indices.append(row_start_indices[None, None, ...])
    attn_mask_row_start_indices = np.concatenate(attn_mask_row_start_indices, axis=0)
    return paddle.to_tensor(attn_mask_row_start_indices, dtype=paddle.int32), int(
        min_start_row
    )


def inbatch_pack_offset_to_xpu_var_len_lod_tensor(inbatch_pack_offset):
    """convert inbatch_pack_offset to xpu_var_len_lod_tensor"""
    # inbatch_pack_offset: size of [max_seq_len], record the accumulated number of tokens of each batch
    # An example of inbatch_pack_offset is: [0, 111, 142, xxxx, 8192, -1, -1, xxxx, -1]
    # lod_tensor: size of [num of batch + 1],  record the accumulated number of tokens of each batch
    # lod_tensor is a length variable tensor, lod_tensor[0] indicates the length of this tensor
    # An example of lod_tensor is: [60, 0, 111, 142, xxxx, 8192]
    inbatch_pack_offset = inbatch_pack_offset.numpy()
    lod_tensor = np.setdiff1d(inbatch_pack_offset, [-1])
    lod_tensor = np.insert(lod_tensor, 0, lod_tensor.shape[0] + 1)
    return paddle.to_tensor(lod_tensor)


def scaled_dot_product_attention(
    query_states,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    config,
    is_causal=True,
    rr_flash_attn=None,
    inbatch_pack_offset=None,
    training=True,
    startend_row_indices=None,
    rr_flashmask_attn=None,
):
    """doc"""
    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, num_key_value_heads, _ = value_states.shape

    if startend_row_indices is not None:
        if rr_flashmask_attn is not None:
            flashmask_attention_func = rr_flashmask_attn
        else:
            flashmask_attention_func = flashmask_attention

        attn_output = flashmask_attention_func(
            query_states.astype(value_states.dtype),
            key_states.astype(value_states.dtype),
            value_states.astype(value_states.dtype),
            startend_row_indices=startend_row_indices,
            dropout=config.attention_probs_dropout_prob,
            causal=False,  # 不能设置死呀,哎,后续需要适配进来
        )
        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        return attn_output, None

    can_use_fa = config.use_flash_attn and flash_attention is not None
    can_use_fa_sparse_mask = (
        config.use_mem_eff_attn
        and inbatch_pack_offset is not None
        and flashmask_attention is not None
    )

    if not can_use_fa and not can_use_fa_sparse_mask:
        if query_states.shape[-2] != key_states.shape[-2]:
            key_states = key_states.repeat_interleave(
                num_heads // num_key_value_heads, axis=-2
            )
        if query_states.shape[-2] != value_states.shape[-2]:
            value_states = value_states.repeat_interleave(
                num_heads // num_key_value_heads, axis=-2
            )

    if can_use_fa:
        # Flash Attention now ignore attention mask
        # Current Flash Attention doesn't support attn maskt
        # Paddle Flash Attention input [ bz, seqlen, nhead, head_dim]
        # Torch Flash Attention input [ bz, nhead, seqlen, head_dim]
        # without past keys
        assert not (config.use_mem_eff_attn and inbatch_pack_offset is not None)
        if rr_flash_attn is not None:
            attn_output, attn_weights = rr_flash_attn(
                query_states,
                key_states,
                value_states,
                dropout=config.attention_probs_dropout_prob,
                causal=is_causal and query_states.shape[1] != 1,
                return_softmax=output_attentions,
            )
        else:
            attn_output, attn_weights = flash_attention(
                query_states,
                key_states,
                value_states,
                dropout=config.attention_probs_dropout_prob,
                causal=is_causal and query_states.shape[1] != 1,
                return_softmax=output_attentions,
            )

        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        return attn_output, attn_weights
    elif config.use_mem_eff_attn and inbatch_pack_offset is not None:
        assert (
            not output_attentions
        ), "output_attentions should be False when use_mem_eff_attn=True"
        if config.use_flash_attn_with_mask:
            if get_env_device() == "xpu":
                assert (
                    varlen_fa is not None
                ), "Please use the latest xpu image to use var len fa."
                var_len_lod = inbatch_pack_offset_to_xpu_var_len_lod_tensor(
                    inbatch_pack_offset
                )
                attn_output = varlen_fa(
                    query_states.astype(value_states.dtype),
                    key_states.astype(value_states.dtype),
                    value_states.astype(value_states.dtype),
                    var_len_lod,  # LoDSeqLenQ
                    var_len_lod,  # LoDSeqLenKV
                    None,  # FixedSeedOffset
                    None,  # AttentionMask
                    config.attention_probs_dropout_prob,  # dropout
                    True,  # is_causal
                    False,  # return_softmax
                )[0]
            elif flashmask_attention is not None:
                if len(inbatch_pack_offset.shape) == 2:
                    causal_mask_indices, attn_mask_min_start_row = (
                        inbatch_pack_offset_to_attn_mask_start_row_indices(
                            inbatch_pack_offset
                        )
                    )
                    startend_row_indices = causal_mask_indices.unsqueeze(-1)
                else:
                    # startend_row_indices (inbatch_pack_offset) shape: [batch_size, seq_len, {1, 2, 4}]
                    assert len(inbatch_pack_offset.shape) == 4, (
                        f"inbatch_pack_offset needs to be a 2 or 4-dimensional tensor when use flashmask attention, "
                        f"but got shape of {inbatch_pack_offset.shape}"
                    )
                    startend_row_indices = inbatch_pack_offset

                startend_row_indices = startend_row_indices.astype("int32")
                attn_output = flashmask_attention(
                    query_states.astype(value_states.dtype),
                    key_states.astype(value_states.dtype),
                    value_states.astype(value_states.dtype),
                    startend_row_indices=startend_row_indices,
                    dropout=config.attention_probs_dropout_prob,
                    causal=True,
                )
            else:
                attn_mask = to_block_diag_causal_mask(
                    inbatch_pack_offset, q_len, float("-inf"), "bfloat16"
                )
                attn_output = flash_attention_with_mask(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask,
                    config.attention_probs_dropout_prob,
                )
        else:
            attn_output = mem_eff_attn(
                query_states,
                key_states,
                value_states,
                inbatch_pack_offset,
                drop_prob=config.attention_probs_dropout_prob,
            )
        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        return attn_output, None
    else:

        query_states = paddle.transpose(query_states, [0, 2, 1, 3]) / math.sqrt(
            head_dim
        )
        # merge with the next tranpose
        key_states = paddle.transpose(key_states, [0, 2, 1, 3])
        value_states = paddle.transpose(value_states, [0, 2, 1, 3])

        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2]))

        if attn_weights.shape != [bsz, num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention weights should be of shape {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        # Pipeline 的Attention mask不能从外面传。
        if attention_mask is None:
            attention_mask = get_triangle_upper_mask(attn_weights)

        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])
        if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention mask should be of shape {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
            )
        # logger.info(attention_mask)
        if training:
            attn_weights = attention_mask + attn_weights
            attn_weights = paddle.maximum(
                attn_weights,
                paddle.to_tensor(
                    float(finfo(query_states.dtype).min), dtype=query_states.dtype
                ),
            )

            if paddle.in_dynamic_mode():
                with paddle.amp.auto_cast(False):
                    attn_weights = F.softmax(
                        attn_weights, axis=-1, dtype="float32"
                    ).astype(query_states.dtype)
            else:
                attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(
                    query_states.dtype
                )
        else:  # use inplace operation to save memory
            attn_weights = attn_weights.cast(paddle.float32)
            attention_mask = attention_mask.cast(paddle.float32)
            attn_weights = attn_weights.add_(attention_mask)
            attn_weights = F.softmax_(attn_weights, axis=-1).astype(query_states.dtype)

        if config.attention_probs_dropout_prob > 0.0:
            if config.tensor_parallel_degree > 1:
                with get_rng_state_tracker().rng_state("local_seed"):
                    attn_weights = F.dropout(
                        attn_weights,
                        config.attention_probs_dropout_prob,
                        training=training,
                        mode="upscale_in_train",
                    )
            else:
                attn_weights = F.dropout(
                    attn_weights,
                    config.attention_probs_dropout_prob,
                    training=training,
                    mode="upscale_in_train",
                )

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None


def _make_causal_mask(input_ids_shape, past_key_values_length, dtype):
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape

    mask = paddle.full((target_length, target_length), float(finfo(dtype).min))

    mask_cond = paddle.arange(mask.shape[-1])
    mask = masked_fill(
        mask, mask_cond < (mask_cond + 1).reshape([mask.shape[-1], 1]), 0
    )

    if past_key_values_length > 0:
        mask = paddle.concat(
            [paddle.zeros([target_length, past_key_values_length]), mask], axis=-1
        )

    return mask[None, None, :, :].expand(
        [batch_size, 1, target_length, target_length + past_key_values_length]
    )


def _expand_mask(mask, dtype, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    if mask.ndim == 4:
        expanded_mask = mask
    elif mask.ndim == 3:
        expanded_mask = mask[:, None, :, :]
    else:
        batch_size, src_length = mask.shape[0], mask.shape[-1]
        tgt_length = tgt_length if tgt_length is not None else src_length

        expanded_mask = mask[:, None, None, :].expand(
            [batch_size, 1, tgt_length, src_length]
        )

    inverted_mask = 1.0 - expanded_mask
    return masked_fill(
        inverted_mask, inverted_mask.cast("bool"), float(finfo(dtype).min)
    )


class FusedDropoutImpl(nn.Layer):
    """
    Fused dropout impl, supporting gpu and xpu
    """

    def __init__(self, prob, mode):
        super().__init__()
        self.prob = prob
        self.mode = mode

        # assert get_env_device() == "xpu"
        self.dropout = nn.Dropout(p=prob, mode=mode)

    def forward(self, x, y):
        """
        forward fn
        """
        if self.prob > 0:
            x = self.dropout(x)
        output = x + y

        return output


class RMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

        if config.sequence_parallel or config.submatrix_parallel:
            mark_as_sequence_parallel_parameter(self.weight)

    def forward(self, hidden_states):
        if get_env_device() == "xpu":
            try:
                import paddle_xpu_nn

                return paddle_xpu_nn.xpu_rms_norm(
                    hidden_states, self.weight, self.variance_epsilon
                )[0]
            except ImportError:
                pass
        if self.config.fuse_rms_norm:
            return fused.fused_rms_norm(
                hidden_states, self.weight, self.variance_epsilon
            )[0]
        if paddle.in_dynamic_mode():
            with paddle.amp.auto_cast(False):
                variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
                hidden_states = (
                    paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
                )
        else:
            variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
            hidden_states = (
                paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
            )

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return hidden_states * self.weight


class LayerNorm(nn.LayerNorm):
    def __init__(self, config):
        super().__init__(config.hidden_size, epsilon=config.rms_norm_eps)
        self.config = config
        if config.sequence_parallel or config.submatrix_parallel:
            mark_as_sequence_parallel_parameter(self.weight)
            mark_as_sequence_parallel_parameter(self.bias)

        self.use_fast_ln = config.use_fast_ln
        if self.use_fast_ln:
            assert fast_ln is not None

    def forward(self, hidden_states):
        """
        The layer normalization operator.
        """
        if self.use_fast_ln:
            return fast_ln(hidden_states, self.weight, self.bias, self._epsilon)[0]
        else:
            return super().forward(hidden_states)


class FusedLayerNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.bias = paddle.create_parameter(
            shape=[self.hidden_size], dtype=paddle.get_default_dtype(), is_bias=True
        )
        self.variance_epsilon = config.rms_norm_eps

    def forward(self, hidden_states):
        return fused.fused_ln(
            hidden_states, self.weight, self.bias, self.variance_epsilon
        )[0]


class RotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=4096, base=10000):
        """
        `self.sin_cached` 和`self.cos_cached` 的初始化最后会用numpy再做一遍！
        """
        super().__init__()
        # dtype = paddle.get_default_dtype()
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (
            base ** (paddle.cast(paddle.arange(0, dim, 2), dtype="float32") / dim)
        )

        # self.register_buffer("inv_freq", inv_freq.cast(dtype))

        # higher acc using float32
        t = paddle.arange(max_position_embeddings, dtype="float32")
        freqs = paddle.einsum("i,j->ij", t, inv_freq.cast("float32"))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = paddle.concat([freqs, freqs], axis=-1)

        # [bs, seqlen, nhead, head_dim]
        self.cos_cached = emb.cos()  # [None, :, None, :]  # .astype(dtype)
        self.sin_cached = emb.sin()  # [None, :, None, :]  # .astype(dtype)

        self._cast_to_low_precision = False  # 兼容develop分支paddle
        self._cast_to_low_precison = False

    def forward(self, x, seq_len=None):

        return (
            self.cos_cached[:seq_len, :],
            self.sin_cached[:seq_len, :],
        )

    @classmethod
    def rotate_half(cls, x):
        """Rotates half the hidden dims of the input."""

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return paddle.concat([-x2, x1], axis=-1)

    @classmethod
    def apply_rotary_pos_emb(cls, q, k, cos, sin, offset: int = 0, position_ids=None):
        """doc"""
        if position_ids is not None:
            # logger.info(f'applying pos:{position_ids}')
            assert offset == 0, offset
            cos = F.embedding(position_ids, cos)
            sin = F.embedding(position_ids, sin)
        else:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        cos = cos[:, offset : q.shape[1] + offset, None, :]
        sin = sin[:, offset : q.shape[1] + offset, None, :]

        # q_embed = (q * cos) + (rotate_half(q) * sin)
        # k_embed = (k * cos) + (rotate_half(k) * sin)
        # 不使用操作符，防止Paddle 不同精度之间的结合导致问题。
        q_embed = paddle.add(
            paddle.multiply(q, cos), paddle.multiply(cls.rotate_half(q), sin)
        )
        k_embed = paddle.add(
            paddle.multiply(k, cos), paddle.multiply(cls.rotate_half(k), sin)
        )
        q_embed = q_embed.astype(q.dtype)  # fp32->bf16
        k_embed = k_embed.astype(k.dtype)
        return q_embed, k_embed


class RopeEmbeddingLegacy(nn.Layer):
    """
    从`EB65B`代码搬运而来，在线算cos、sin。算出来的结果和上面有蛮大的diff，
    """

    def __init__(self, head_dim, compression_ratio=1.0, base=10000, freq_allocation=0):
        super().__init__()
        self.head_dim = head_dim
        self.compression_ratio = compression_ratio
        self.base = base

        # num of freq allocated to time
        self.freq_allocation = freq_allocation

    def forward(self, seq_length, position_ids=None):
        indices = paddle.arange(0, self.head_dim, 2, dtype="float32")
        indices = 1 / self.base ** (indices / self.head_dim)
        if position_ids is None:
            position_ids = paddle.arange(0, seq_length, 1, dtype="float32").unsqueeze(1)
            position_ids = position_ids / self.compression_ratio
            sinusoid_inp = position_ids * indices.unsqueeze(0)
        else:
            position_ids = position_ids / self.compression_ratio
            seq_length = position_ids.shape[-1]
            sinusoid_inp = position_ids.unsqueeze(-1).astype(
                "float32"
            ) * indices.unsqueeze(
                0
            )  # [b, s, 1] * [1, d/2] -> [b, s, d/2]
        pos_emb = paddle.concat(
            [paddle.sin(sinusoid_inp), paddle.cos(sinusoid_inp)], axis=-1
        )
        pos_emb = paddle.reshape(pos_emb, (-1, 1, seq_length, self.head_dim))
        pos_emb.stop_gradient = True
        return pos_emb

    def apply_rotary(self, rp, q, k):
        # sin [sequence_length, embed_size_per_head//2]
        # cos [sequence_length, embed_size_per_head//2]
        sin, cos = paddle.chunk(rp, 2, axis=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = paddle.reshape(paddle.stack([sin, sin], axis=-1), rp.shape)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = paddle.reshape(paddle.stack([cos, cos], axis=-1), rp.shape)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_q = paddle.reshape(
            paddle.stack([-q[:, :, :, 1::2], q[:, :, :, 0::2]], axis=-1),
            paddle.shape(q),
        )
        query = paddle.add(
            paddle.multiply(q.astype("float32"), cos_pos),
            paddle.multiply(rotate_half_q.astype("float32"), sin_pos),
        )
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_k = paddle.reshape(
            paddle.stack([-k[:, :, :, 1::2], k[:, :, :, 0::2]], axis=-1),
            paddle.shape(k),
        )
        key = paddle.add(
            paddle.multiply(k.astype("float32"), cos_pos),
            paddle.multiply(rotate_half_k.astype("float32"), sin_pos),
        )
        return query, key

    def apply_rotary_3d(self, rp, q, k, position_ids):
        """
        rope 3d rotary

        args:
            rp: [1, max_seqlen, 1, head_dim]
            q: [bsz, seqlen, head, head_dim]
            k: [bsz, seqlen, head, head_dim]
            position_ids: [bsz, seqlen, 3]
        """
        sin, cos = paddle.chunk(rp, 2, axis=-1)
        assert position_ids.shape[:1] == q.shape[:1]
        batch_indices = paddle.arange(end=position_ids.shape[0])
        batch_indices = batch_indices[..., None]
        sin = sin.tile([position_ids.shape[0], 1, 1, 1])
        cos = cos.tile([position_ids.shape[0], 1, 1, 1])

        assert self.freq_allocation != 0
        sin_t = sin[batch_indices, position_ids[..., 0], :, -self.freq_allocation :]
        sin_h = sin[
            batch_indices,
            position_ids[..., 1],
            :,
            : self.head_dim // 2 - self.freq_allocation : 2,
        ]
        sin_w = sin[
            batch_indices,
            position_ids[..., 2],
            :,
            1 : self.head_dim // 2 - self.freq_allocation : 2,
        ]
        sin_hw = paddle.stack([sin_h, sin_w], axis=-1).reshape(
            sin_h.shape[:-1] + [sin_h.shape[-1] * 2]
        )
        sin_thw = paddle.concat([sin_hw, sin_t], axis=-1)

        cos_t = cos[batch_indices, position_ids[..., 0], :, -self.freq_allocation :]
        cos_h = cos[
            batch_indices,
            position_ids[..., 1],
            :,
            : self.head_dim // 2 - self.freq_allocation : 2,
        ]
        cos_w = cos[
            batch_indices,
            position_ids[..., 2],
            :,
            1 : self.head_dim // 2 - self.freq_allocation : 2,
        ]
        cos_hw = paddle.stack([cos_h, cos_w], axis=-1).reshape(
            cos_h.shape[:-1] + [cos_h.shape[-1] * 2]
        )
        cos_thw = paddle.concat([cos_hw, cos_t], axis=-1)

        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = paddle.reshape(
            paddle.stack([sin_thw, sin_thw], axis=-1),
            sin_thw.shape[:3] + [sin_thw.shape[-1] * 2],
        )
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = paddle.reshape(
            paddle.stack([cos_thw, cos_thw], axis=-1),
            cos_thw.shape[:3] + [cos_thw.shape[-1] * 2],
        )

        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_q = paddle.reshape(
            paddle.stack([-q[:, :, :, 1::2], q[:, :, :, 0::2]], axis=-1),
            paddle.shape(q),
        )
        query = paddle.add(
            paddle.multiply(q.astype("float32"), cos_pos),
            paddle.multiply(rotate_half_q.astype("float32"), sin_pos),
        )
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_k = paddle.reshape(
            paddle.stack([-k[:, :, :, 1::2], k[:, :, :, 0::2]], axis=-1),
            paddle.shape(k),
        )
        key = paddle.add(
            paddle.multiply(k.astype("float32"), cos_pos),
            paddle.multiply(rotate_half_k.astype("float32"), sin_pos),
        )
        return query, key

    def forward_single(self, position_ids):
        batch_size, seq_length = position_ids.shape[:2]
        rope_emb = paddle.zeros(
            (2, batch_size, seq_length, 1, self.head_dim), dtype="float32"
        )
        inv_freq = self.base ** (
            -paddle.arange(0, self.head_dim, 2, dtype="float32") / self.head_dim
        )
        position_ids = position_ids.cast("float32")
        position_ids = position_ids / self.compression_ratio
        # shape: [B, S, D/2]
        freqs = paddle.einsum("ij,k->ijk", position_ids.cast("float32"), inv_freq)
        # shape: [B, S, D]
        emb = paddle.stack([freqs, freqs], axis=-1).reshape(
            (batch_size, seq_length, self.head_dim)
        )
        # shape: [B, S, 1, D]
        emb = paddle.unsqueeze(emb, 2)

        rope_emb[0] = paddle.cos(emb)
        rope_emb[1] = paddle.sin(emb)
        return rope_emb

    @staticmethod
    def apply_rotary_single(x, rope_emb):
        rotate_half_x = paddle.reshape(
            paddle.stack([-x[:, :, :, 1::2], x[:, :, :, 0::2]], axis=-1),
            paddle.shape(x),
        )
        return x * rope_emb[0] + rotate_half_x * rope_emb[1]


class RopeEmbeddingLegacy(nn.Layer):
    def __init__(self, head_dim, compression_ratio=1.0, base=10000, freq_allocation=0):
        super().__init__()
        self.head_dim = head_dim
        self.compression_ratio = compression_ratio
        self.base = base
        self.freq_allocation = freq_allocation

    def forward(self, seq_length, position_ids=None):
        indices = paddle.arange(0, self.head_dim, 2, dtype="float32")
        indices = 1 / self.base ** (indices / self.head_dim)
        if position_ids is None:
            position_ids = paddle.arange(0, seq_length, 1, dtype="float32").unsqueeze(1)
            position_ids = position_ids / self.compression_ratio
            sinusoid_inp = position_ids * indices.unsqueeze(0)
        else:
            position_ids = position_ids / self.compression_ratio
            seq_length = position_ids.shape[-1]
            sinusoid_inp = position_ids.unsqueeze(-1).astype(
                "float32"
            ) * indices.unsqueeze(0)
        pos_emb = paddle.concat(
            [paddle.sin(sinusoid_inp), paddle.cos(sinusoid_inp)], axis=-1
        )
        pos_emb = paddle.reshape(pos_emb, (-1, 1, seq_length, self.head_dim))
        pos_emb.stop_gradient = True
        return pos_emb

    def apply_rotary(self, rp, q, k):
        sin, cos = paddle.chunk(rp, 2, axis=-1)
        sin_pos = paddle.reshape(paddle.stack([sin, sin], axis=-1), rp.shape)
        cos_pos = paddle.reshape(paddle.stack([cos, cos], axis=-1), rp.shape)
        rotate_half_q = paddle.reshape(
            paddle.stack([-q[:, :, :, 1::2], q[:, :, :, 0::2]], axis=-1),
            paddle.shape(q),
        )
        query = paddle.add(
            paddle.multiply(q.astype("float32"), cos_pos),
            paddle.multiply(rotate_half_q.astype("float32"), sin_pos),
        )
        rotate_half_k = paddle.reshape(
            paddle.stack([-k[:, :, :, 1::2], k[:, :, :, 0::2]], axis=-1),
            paddle.shape(k),
        )
        key = paddle.add(
            paddle.multiply(k.astype("float32"), cos_pos),
            paddle.multiply(rotate_half_k.astype("float32"), sin_pos),
        )
        return query, key

    def apply_rotary_3d(self, rp, q, k, position_ids):
        sin, cos = paddle.chunk(rp, 2, axis=-1)
        assert position_ids.shape[:1] == q.shape[:1]
        batch_indices = paddle.arange(end=position_ids.shape[0])
        batch_indices = batch_indices[..., None]
        sin = sin.tile([position_ids.shape[0], 1, 1, 1])
        cos = cos.tile([position_ids.shape[0], 1, 1, 1])

        assert self.freq_allocation != 0
        sin_t = sin[batch_indices, position_ids[..., 0], :, -self.freq_allocation :]
        sin_h = sin[
            batch_indices,
            position_ids[..., 1],
            :,
            : self.head_dim // 2 - self.freq_allocation : 2,
        ]
        sin_w = sin[
            batch_indices,
            position_ids[..., 2],
            :,
            1 : self.head_dim // 2 - self.freq_allocation : 2,
        ]
        sin_hw = paddle.stack([sin_h, sin_w], axis=-1).reshape(
            sin_h.shape[:-1] + [sin_h.shape[-1] * 2]
        )
        sin_thw = paddle.concat([sin_hw, sin_t], axis=-1)

        cos_t = cos[batch_indices, position_ids[..., 0], :, -self.freq_allocation :]
        cos_h = cos[
            batch_indices,
            position_ids[..., 1],
            :,
            : self.head_dim // 2 - self.freq_allocation : 2,
        ]
        cos_w = cos[
            batch_indices,
            position_ids[..., 2],
            :,
            1 : self.head_dim // 2 - self.freq_allocation : 2,
        ]
        cos_hw = paddle.stack([cos_h, cos_w], axis=-1).reshape(
            cos_h.shape[:-1] + [cos_h.shape[-1] * 2]
        )
        cos_thw = paddle.concat([cos_hw, cos_t], axis=-1)

        sin_pos = paddle.reshape(
            paddle.stack([sin_thw, sin_thw], axis=-1),
            sin_thw.shape[:3] + [sin_thw.shape[-1] * 2],
        )
        cos_pos = paddle.reshape(
            paddle.stack([cos_thw, cos_thw], axis=-1),
            cos_thw.shape[:3] + [cos_thw.shape[-1] * 2],
        )

        rotate_half_q = paddle.reshape(
            paddle.stack([-q[:, :, :, 1::2], q[:, :, :, 0::2]], axis=-1),
            paddle.shape(q),
        )
        query = paddle.add(
            paddle.multiply(q.astype("float32"), cos_pos),
            paddle.multiply(rotate_half_q.astype("float32"), sin_pos),
        )
        rotate_half_k = paddle.reshape(
            paddle.stack([-k[:, :, :, 1::2], k[:, :, :, 0::2]], axis=-1),
            paddle.shape(k),
        )
        key = paddle.add(
            paddle.multiply(k.astype("float32"), cos_pos),
            paddle.multiply(rotate_half_k.astype("float32"), sin_pos),
        )
        return query, key

    def forward_single(self, position_ids):
        batch_size, seq_length = position_ids.shape[:2]
        rope_emb = paddle.zeros(
            (2, batch_size, seq_length, 1, self.head_dim), dtype="float32"
        )
        inv_freq = self.base ** (
            -paddle.arange(0, self.head_dim, 2, dtype="float32") / self.head_dim
        )
        position_ids = position_ids.cast("float32")
        position_ids = position_ids / self.compression_ratio
        freqs = paddle.einsum("ij,k->ijk", position_ids.cast("float32"), inv_freq)
        emb = paddle.stack([freqs, freqs], axis=-1).reshape(
            (batch_size, seq_length, self.head_dim)
        )
        emb = paddle.unsqueeze(emb, 2)

        rope_emb[0] = paddle.cos(emb)
        rope_emb[1] = paddle.sin(emb)
        return rope_emb

    @staticmethod
    def apply_rotary_single(x, rope_emb):
        rotate_half_x = paddle.reshape(
            paddle.stack([-x[:, :, :, 1::2], x[:, :, :, 0::2]], axis=-1),
            paddle.shape(x),
        )
        return x * rope_emb[0] + rotate_half_x * rope_emb[1]


class ErnieMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # self.fuse_ffn = config.tensor_parallel_degree > 1 and config.fuse_attn_ffn
        self.fuse_ffn = config.fuse_attn_ffn

        if config.tensor_parallel_degree > 1:
            if config.use_fp8 and config.fp8_configs["layers"]["mlp_tp_fc1_linear"]:
                if config.fuse_attn_ffn:
                    self.up_gate_proj = TEFP8Linear(
                        self.hidden_size,
                        self.intermediate_size * 2,
                        bias_attr=config.use_bias,
                        parallel_mode="column",
                        sequence_parallel=config.sequence_parallel,
                        quant_weight_at_first_microbatch=config.fp8_configs["recipe"][
                            "quant_weight_at_first_microbatch"
                        ],
                    )
                else:
                    self.gate_proj = TEFP8Linear(
                        self.hidden_size,
                        self.intermediate_size,
                        bias_attr=config.use_bias,
                        parallel_mode="column",
                        sequence_parallel=config.sequence_parallel,
                        quant_weight_at_first_microbatch=config.fp8_configs["recipe"][
                            "quant_weight_at_first_microbatch"
                        ],
                    )
                    self.up_proj = TEFP8Linear(
                        self.hidden_size,
                        self.intermediate_size,
                        bias_attr=config.use_bias,
                        parallel_mode="column",
                        sequence_parallel=config.sequence_parallel,
                        quant_weight_at_first_microbatch=config.fp8_configs["recipe"][
                            "quant_weight_at_first_microbatch"
                        ],
                    )
            elif config.submatrix_parallel:
                if config.fuse_attn_ffn:
                    self.up_gate_proj = SMPColumnParallelLinear(
                        self.hidden_size,
                        self.intermediate_size * 2,
                        bias_attr=config.use_bias,
                        low_memory=config.submatrix_parallel_low_memory,
                    )
                else:
                    self.gate_proj = SMPColumnParallelLinear(
                        self.hidden_size,
                        self.intermediate_size,
                        bias_attr=config.use_bias,
                        low_memory=config.submatrix_parallel_low_memory,
                    )
                    self.up_proj = SMPColumnParallelLinear(
                        self.hidden_size,
                        self.intermediate_size,
                        bias_attr=config.use_bias,
                        low_memory=config.submatrix_parallel_low_memory,
                    )
            else:
                ColumnLN = (
                    ColumnSequenceParallelLinear
                    if config.sequence_parallel
                    else ColumnParallelLinear
                )
                RowLN = (
                    RowSequenceParallelLinear
                    if config.sequence_parallel
                    else RowParallelLinear
                )

                self.use_comm = not (
                    get_env_device() == "xpu"
                    and os.getenv("XPU_MOE_USE_ALLGATHER", False)
                )

                column_ln_configs = (
                    {
                        "use_rr": config.use_recompute
                        and config.skip_recompute_ops.get("mlp_column_ln", False)
                    }
                    if config.sequence_parallel and get_env_device() == "gpu"
                    else {}
                )
                if not self.use_comm:
                    column_ln_configs["use_comm"] = self.use_comm
                if config.sequence_parallel and get_env_device() == "gpu":
                    column_ln_configs["use_tpsp_comm_overlap"] = (
                        config.use_tpsp_comm_overlap
                    )
                if config.fuse_attn_ffn:
                    self.up_gate_proj = ColumnLN(
                        self.hidden_size,
                        self.intermediate_size * 2,
                        gather_output=False,
                        has_bias=config.use_bias,
                        fuse_matmul_bias=config.fuse_linear,
                        **column_ln_configs,
                    )
                else:
                    self.gate_proj = ColumnLN(
                        self.hidden_size,
                        self.intermediate_size,
                        gather_output=False,
                        has_bias=config.use_bias,
                        fuse_matmul_bias=config.fuse_linear,
                        **column_ln_configs,
                    )
                    self.up_proj = ColumnLN(
                        self.hidden_size,
                        self.intermediate_size,
                        gather_output=False,
                        has_bias=config.use_bias,
                        fuse_matmul_bias=config.fuse_linear,
                        **column_ln_configs,
                    )
        else:
            LinearFN = (
                paddle.incubate.nn.FusedLinear if config.fuse_linear else NativeLinear
            )
            if config.use_fp8 and config.fp8_configs["layers"]["mlp_fc1_linear"]:
                LinearFN = partial(
                    TEFP8Linear,
                    quant_weight_at_first_microbatch=config.fp8_configs["recipe"][
                        "quant_weight_at_first_microbatch"
                    ],
                )
            if config.fuse_attn_ffn:
                self.up_gate_proj = LinearFN(
                    self.hidden_size,
                    self.intermediate_size * 2,
                    bias_attr=config.use_bias,
                )
            else:
                self.gate_proj = LinearFN(
                    self.hidden_size, self.intermediate_size, bias_attr=config.use_bias
                )
                self.up_proj = LinearFN(
                    self.hidden_size, self.intermediate_size, bias_attr=config.use_bias
                )

        if config.tensor_parallel_degree > 1:
            if config.use_fp8 and config.fp8_configs["layers"]["mlp_tp_fc2_linear"]:
                self.down_proj = TEFP8Linear(
                    self.intermediate_size,
                    self.hidden_size,
                    bias_attr=config.use_bias,
                    parallel_mode="row",
                    sequence_parallel=config.sequence_parallel,
                    quant_weight_at_first_microbatch=config.fp8_configs["recipe"][
                        "quant_weight_at_first_microbatch"
                    ],
                )
            elif config.submatrix_parallel:
                self.down_proj = SMPRowParallelLinear(
                    self.intermediate_size,
                    self.hidden_size,
                    bias_attr=config.use_bias,
                    low_memory=config.submatrix_parallel_low_memory,
                )
            else:
                row_ln_configs = (
                    {
                        "use_rr": config.use_recompute
                        and config.skip_recompute_ops.get("mlp_row_ln", False)
                    }
                    if config.sequence_parallel and get_env_device() == "gpu"
                    else {}
                )
                if not self.use_comm:
                    row_ln_configs["use_comm"] = self.use_comm
                if config.sequence_parallel and get_env_device() == "gpu":
                    row_ln_configs["use_tpsp_comm_overlap"] = (
                        config.use_tpsp_comm_overlap
                    )
                self.down_proj = RowLN(
                    self.intermediate_size,
                    self.hidden_size,
                    input_is_parallel=True,
                    has_bias=config.use_bias,
                    fuse_matmul_bias=config.fuse_linear,
                    **row_ln_configs,
                )
        else:
            LinearFN = (
                paddle.incubate.nn.FusedLinear if config.fuse_linear else NativeLinear
            )
            if config.use_fp8 and config.fp8_configs["layers"]["mlp_fc2_linear"]:
                LinearFN = partial(
                    TEFP8Linear,
                    quant_weight_at_first_microbatch=config.fp8_configs["recipe"][
                        "quant_weight_at_first_microbatch"
                    ],
                )
            self.down_proj = LinearFN(
                self.intermediate_size, self.hidden_size, bias_attr=config.use_bias
            )

        self.fuse_swiglu = config.fuse_swiglu
        if self.fuse_swiglu:
            assert fused_swiglu is not None, "fused_swiglu operator is not found."

    def forward(self, x):
        if (
            self.config.tensor_parallel_degree <= 1
            and self.fuse_ffn
            and self.config.use_fp8_mlp
            and not self.config.use_bias
        ):
            return MemEfficientFp8FusedMlpFunc.apply(
                x, self.up_gate_proj.weight, self.down_proj.weight
            )

        if self.fuse_swiglu:
            if self.fuse_ffn:
                if self.config.use_fp8 and self.config.fp8_configs["smooth_swiglu"]:
                    x, gate = self.up_gate_proj(x).chunk(2, axis=-1)
                    with paddle.no_grad():
                        scale = paddle.clip(gate.abs().max(axis=-1, keepdim=True), 1e-8)

                    gate = gate / scale
                    if self.config.sequence_parallel or self.config.submatrix_parallel:
                        scale = ScatterOp.apply(scale)

                    x = paddle.concat([x, gate], axis=-1)
                else:
                    x = self.up_gate_proj(x)
                x = fused_swiglu(x)
            else:
                x = fused_swiglu(self.gate_proj(x), self.up_proj(x))
        else:
            if self.fuse_ffn:
                x, gate = self.up_gate_proj(x).chunk(2, axis=-1)
                x = F.silu(x) * gate
            else:
                x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        if self.config.use_fp8 and self.config.fp8_configs["smooth_swiglu"]:
            return self.down_proj(x) * scale
        return self.down_proj(x)


class ErnieAttention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        if config.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = config.head_dim
        # self.fuse_attn = config.tensor_parallel_degree > 1 and config.fuse_attn_ffn
        self.fuse_attn = config.fuse_attn_ffn
        self.use_recompute_attn = config.use_recompute_attn  # aka recompute core-attn
        logger.info(f"using recompute attn={self.use_recompute_attn}")
        self.is_gqa = (
            config.num_key_value_heads is not None
            and config.num_key_value_heads != self.num_heads
        )
        if config.fuse_rope:
            assert fused_rope is not None, "fused_rope is not supported"
        self.fuse_rope = config.fuse_rope
        self.rope_3d = config.rope_3d
        if self.rope_3d:
            assert (
                not self.fuse_rope
            ), "does not support fuse rope when rope_3d is on for now."
            assert (
                not config.rope_reorder
            ), "does not support rope_reorder when rope_3d is on for now."
            assert (
                config.freq_allocation is not None
            ), "freq_allocation must be provided if rope_3d is on."

        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), f"num_heads: {self.num_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_heads = self.num_heads // config.tensor_parallel_degree
            if self.is_gqa:
                assert (
                    self.num_key_value_heads % config.tensor_parallel_degree == 0
                ), f"num_heads: {self.num_key_value_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
                self.num_key_value_heads = (
                    self.num_key_value_heads // config.tensor_parallel_degree
                )
        if self.is_gqa:
            logger.info(
                f"use GQA - num_heads: {self.num_heads}- num_key_value_heads: {self.num_key_value_heads}"
            )
            assert (
                self.num_heads % self.num_key_value_heads == 0
            ), f"num_heads: {self.num_heads}, num_key_value_heads: {self.num_key_value_heads}"
            q_hidden_size = self.head_dim * config.num_attention_heads
            kv_hidden_size = self.head_dim * config.num_key_value_heads

        else:
            q_hidden_size = kv_hidden_size = self.head_dim * config.num_attention_heads

        if config.tensor_parallel_degree > 1:
            if config.use_fp8 and config.fp8_configs["layers"]["attn_tp_fc1_linear"]:
                if config.fuse_attn_ffn:
                    self.qkv_proj = TEFP8Linear(
                        self.hidden_size,
                        (
                            self.hidden_size * 3
                            if not self.is_gqa
                            else self.hidden_size + kv_hidden_size * 2
                        ),
                        bias_attr=config.use_bias,
                        parallel_mode="column",
                        sequence_parallel=config.sequence_parallel,
                        quant_weight_at_first_microbatch=config.fp8_configs["recipe"][
                            "quant_weight_at_first_microbatch"
                        ],
                    )
                else:
                    self.q_proj = TEFP8Linear(
                        self.hidden_size,
                        self.hidden_size,
                        bias_attr=config.use_bias,
                        parallel_mode="column",
                        sequence_parallel=config.sequence_parallel,
                        quant_weight_at_first_microbatch=config.fp8_configs["recipe"][
                            "quant_weight_at_first_microbatch"
                        ],
                    )
                    self.k_proj = TEFP8Linear(
                        self.hidden_size,
                        self.hidden_size if not self.is_gqa else kv_hidden_size,
                        bias_attr=config.use_bias,
                        parallel_mode="column",
                        sequence_parallel=config.sequence_parallel,
                        quant_weight_at_first_microbatch=config.fp8_configs["recipe"][
                            "quant_weight_at_first_microbatch"
                        ],
                    )
                    self.v_proj = TEFP8Linear(
                        self.hidden_size,
                        self.hidden_size if not self.is_gqa else kv_hidden_size,
                        bias_attr=config.use_bias,
                        parallel_mode="column",
                        sequence_parallel=config.sequence_parallel,
                        quant_weight_at_first_microbatch=config.fp8_configs["recipe"][
                            "quant_weight_at_first_microbatch"
                        ],
                    )
            elif config.submatrix_parallel:
                if config.fuse_attn_ffn:
                    self.qkv_proj = SMPColumnParallelLinear(
                        self.hidden_size,
                        q_hidden_size + 2 * kv_hidden_size,
                        bias_attr=config.use_bias,
                        low_memory=config.submatrix_parallel_low_memory,
                    )
                else:
                    self.q_proj = SMPColumnParallelLinear(
                        self.hidden_size,
                        q_hidden_size,
                        bias_attr=config.use_bias,
                        low_memory=config.submatrix_parallel_low_memory,
                    )
                    self.k_proj = SMPColumnParallelLinear(
                        self.hidden_size,
                        kv_hidden_size,
                        bias_attr=config.use_bias,
                        low_memory=config.submatrix_parallel_low_memory,
                    )
                    self.v_proj = SMPColumnParallelLinear(
                        self.hidden_size,
                        kv_hidden_size,
                        bias_attr=config.use_bias,
                        low_memory=config.submatrix_parallel_low_memory,
                    )
            else:
                ColumnLN = (
                    ColumnSequenceParallelLinear
                    if config.sequence_parallel
                    else ColumnParallelLinear
                )
                RowLN = (
                    RowSequenceParallelLinear
                    if config.sequence_parallel
                    else RowParallelLinear
                )
                column_ln_configs = (
                    {
                        "use_rr": config.use_recompute
                        and config.skip_recompute_ops.get("attention_column_ln", False)
                    }
                    if config.sequence_parallel and get_env_device() == "gpu"
                    else {}
                )
                if config.sequence_parallel and get_env_device() == "gpu":
                    column_ln_configs["use_tpsp_comm_overlap"] = (
                        config.use_tpsp_comm_overlap
                    )

                if config.fuse_attn_ffn:
                    self.qkv_proj = ColumnLN(
                        self.hidden_size,
                        q_hidden_size + 2 * kv_hidden_size,
                        has_bias=config.use_bias,
                        gather_output=False,
                        fuse_matmul_bias=config.fuse_linear,
                        **column_ln_configs,
                    )
                else:
                    self.q_proj = ColumnLN(
                        self.hidden_size,
                        q_hidden_size,
                        has_bias=config.use_bias,
                        gather_output=False,
                        fuse_matmul_bias=config.fuse_linear,
                        **column_ln_configs,
                    )
                    self.k_proj = ColumnLN(
                        self.hidden_size,
                        kv_hidden_size,
                        has_bias=config.use_bias,
                        gather_output=False,
                        fuse_matmul_bias=config.fuse_linear,
                        **column_ln_configs,
                    )
                    self.v_proj = ColumnLN(
                        self.hidden_size,
                        kv_hidden_size,
                        has_bias=config.use_bias,
                        gather_output=False,
                        fuse_matmul_bias=config.fuse_linear,
                        **column_ln_configs,
                    )
        else:
            LinearFN = (
                paddle.incubate.nn.FusedLinear if config.fuse_linear else NativeLinear
            )
            if config.use_fp8 and config.fp8_configs["layers"]["attn_fc1_linear"]:
                LinearFN = partial(
                    TEFP8Linear,
                    quant_weight_at_first_microbatch=config.fp8_configs["recipe"][
                        "quant_weight_at_first_microbatch"
                    ],
                )
            if config.fuse_attn_ffn:
                self.qkv_proj = LinearFN(
                    self.hidden_size,
                    q_hidden_size + 2 * kv_hidden_size,
                    bias_attr=config.use_bias,
                )
            else:
                self.q_proj = LinearFN(
                    self.hidden_size,
                    q_hidden_size,
                    bias_attr=config.use_bias,
                )
                self.k_proj = LinearFN(
                    self.hidden_size,
                    kv_hidden_size,
                    bias_attr=config.use_bias,
                )
                self.v_proj = LinearFN(
                    self.hidden_size,
                    kv_hidden_size,
                    bias_attr=config.use_bias,
                )

        if config.tensor_parallel_degree > 1:
            if config.use_fp8 and config.fp8_configs["layers"]["attn_tp_fc2_linear"]:
                self.o_proj = TEFP8Linear(
                    q_hidden_size,
                    self.hidden_size,
                    bias_attr=config.use_bias,
                    parallel_mode="row",
                    sequence_parallel=config.sequence_parallel,
                    quant_weight_at_first_microbatch=config.fp8_configs["recipe"][
                        "quant_weight_at_first_microbatch"
                    ],
                )
            elif config.submatrix_parallel:
                self.o_proj = SMPRowParallelLinear(
                    q_hidden_size,
                    self.hidden_size,
                    bias_attr=config.use_bias,
                    low_memory=config.submatrix_parallel_low_memory,
                )
            else:
                row_ln_configs = (
                    {
                        "use_rr": config.use_recompute
                        and config.skip_recompute_ops.get("attention_row_ln", False)
                    }
                    if config.sequence_parallel and get_env_device() == "gpu"
                    else {}
                )
                if config.sequence_parallel and get_env_device() == "gpu":
                    row_ln_configs["use_tpsp_comm_overlap"] = (
                        config.use_tpsp_comm_overlap
                    )

                self.o_proj = RowLN(
                    q_hidden_size,
                    self.hidden_size,
                    has_bias=config.use_bias,
                    input_is_parallel=True,
                    fuse_matmul_bias=config.fuse_linear,
                    **row_ln_configs,
                )
        else:
            LinearFN = (
                paddle.incubate.nn.FusedLinear if config.fuse_linear else NativeLinear
            )
            if config.use_fp8 and config.fp8_configs["layers"]["attn_fc2_linear"]:
                LinearFN = partial(
                    TEFP8Linear,
                    quant_weight_at_first_microbatch=config.fp8_configs["recipe"][
                        "quant_weight_at_first_microbatch"
                    ],
                )
            self.o_proj = LinearFN(
                q_hidden_size,
                self.hidden_size,
                bias_attr=config.use_bias,
            )
        if config.rope_yarn_config:
            from paddleformers.transformers.long_sequence_strategies.embedding_strategies import (
                YaRNScalingRotaryEmbedding,
            )

            logger.info(f"using rope w/ yarn: {config.rope_yarn_config}")
            self.rotary_emb = YaRNScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                **config.rope_yarn_config,
            )
        elif config.rope_reorder:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        else:
            self.rotary_emb = RopeEmbeddingLegacy(
                self.head_dim,
                compression_ratio=config.compression_ratio,
                base=config.rope_theta,
                freq_allocation=config.freq_allocation,
            )
        self.config = config

        self._rr_flash_attn = None
        self._rr_flashmask_attn = None
        if config.use_recompute and config.skip_recompute_ops.get("flash_attn", False):
            self._rr_flash_attn = RefinedRcomputeFlashAttention()
            self._rr_flashmask_attn = RefinedRcomputeFlashMaskAttention()

        if self.config.use_qk_norm:
            logger.info(f"use_qk_norm, the head_dim is {self.head_dim}")
            Norm = RMSNorm if config.use_rmsnorm else LayerNorm
            if not config.use_rmsnorm and config.fuse_ln:
                Norm = FusedLayerNorm
            qk_norm_config = copy.deepcopy(config)
            qk_norm_config.hidden_size = self.head_dim
            self.q_norm = Norm(qk_norm_config)
            self.k_norm = Norm(qk_norm_config)

    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        inbatch_pack_offset: Optional[Tuple[paddle.Tensor]] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        if self.config.sequence_parallel or self.config.submatrix_parallel:
            # 在SP下输入 hidden.shape = [bsz*seq/mp_size, dim]
            # logger.info(f'into attn shape={hidden_states.shape}')
            if not self.config.using_dynamic_sequence_length:
                bsz = (
                    hidden_states.shape[0]
                    * self.config.tensor_parallel_degree
                    // self.config.seqlen
                )
                q_len = self.config.seqlen
            else:
                assert (
                    self.config.micro_batch_size
                ), "micro_batch_size should be set when using dygramic sequence length."

                bsz = self.config.micro_batch_size
                q_len = (
                    hidden_states.shape[0] * self.config.tensor_parallel_degree // bsz
                )
        else:
            bsz, q_len, _ = hidden_states.shape
        query_states = key_states = value_states = mix_layer = None
        if self.fuse_attn:
            mix_layer = self.qkv_proj(hidden_states)
            if self.is_gqa:
                query_states, key_states, value_states = paddle.split(
                    mix_layer.reshape([bsz, q_len, -1, self.head_dim]),
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
                    [bsz, q_len, self.num_heads, 3 * self.head_dim]
                )
        else:
            query_states = (
                self.q_proj(hidden_states).reshape(
                    shape=[bsz, q_len, self.num_heads, self.head_dim]
                )
                # .transpose([0, 2, 1, 3])
            )
            key_states = (
                self.k_proj(hidden_states).reshape(
                    shape=[
                        bsz,
                        q_len,
                        self.num_key_value_heads if self.is_gqa else self.num_heads,
                        self.head_dim,
                    ]
                )
                # .transpose([0, 2, 1, 3])
            )
            value_states = (
                self.v_proj(hidden_states).reshape(
                    shape=[
                        bsz,
                        q_len,
                        self.num_key_value_heads if self.is_gqa else self.num_heads,
                        self.head_dim,
                    ]
                )
                # .transpose([0, 2, 1, 3])
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
                use_reentrant=self.config.recompute_use_reentrant,
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

        if self.config.sequence_parallel or self.config.submatrix_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def rope_attn(
        self,
        mix_layer,
        query_states,
        key_states,
        value_states,
        attention_mask,
        position_ids,
        output_attentions=False,
        past_key_value=None,
        use_cache=False,
        inbatch_pack_offset=None,
    ):
        """
        在recompute区域内进行fuse-split， rope, matmul等操作。
        """
        if mix_layer is not None:
            query_states, key_states, value_states = paddle.split(mix_layer, 3, axis=-1)
        query_states_dtype = query_states.dtype

        # don't get confused, kv_seq_len is just used to retrieve correct cos_sin
        if self.rope_3d:
            assert position_ids is not None, "rope3d requires pos-id"
        kv_seq_len = (
            key_states.shape[-3] if not self.rope_3d else position_ids.max() + 1
        )
        offset = 0
        if past_key_value is not None:
            if not self.rope_3d:
                offset = past_key_value[0].shape[-3]
                kv_seq_len += offset
            else:
                # well, the model can only generate text.
                offset = position_ids.max()
                kv_seq_len = position_ids.max() + 1
                position_ids = position_ids[:, -1:, :]

        if self.config.rope_yarn_config:
            assert self.config.rope_reorder
            cos, sin = self.rotary_emb(kv_seq_len)
            query_states, key_states = RotaryEmbedding.apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                position_ids=position_ids,
                offset=offset if position_ids is None else 0,
            )

        elif self.config.rope_reorder:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

            query_states, key_states = self.rotary_emb.apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                position_ids=position_ids,
                offset=offset if position_ids is None else 0,
            )
        else:
            # 2.
            # cos_sin = self.rotary_emb.forward_single(paddle.arange(kv_seq_len).unsqueeze(0).astype('float32'))
            # query_states = self.rotary_emb.apply_rotary_single(query_states, cos_sin)
            # key_states = self.rotary_emb.apply_rotary_single(key_states, cos_sin)
            #
            # 3.
            if offset > 0 or position_ids is not None or not self.fuse_rope:
                if not self.rope_3d:
                    cos_sin = self.rotary_emb(kv_seq_len, position_ids).transpose(
                        [0, 2, 1, 3]
                    )  # [b,h,s,d]->[b,s,h,d]
                    if offset > 0 and position_ids is None:
                        # position_ids has been sliced in prepare_inputs_for_generation
                        cos_sin = cos_sin[:, offset:]
                    query_states, key_states = self.rotary_emb.apply_rotary(
                        cos_sin, query_states, key_states
                    )
                else:
                    cos_sin = self.rotary_emb(kv_seq_len).transpose(
                        [0, 2, 1, 3]
                    )  # [b,h,s,d]->[b,s,h,d]

                    if offset > 0 and position_ids is None:
                        cos_sin = cos_sin[:, offset:]

                    query_states, key_states = self.rotary_emb.apply_rotary_3d(
                        cos_sin, query_states, key_states, position_ids
                    )
            else:
                assert not self.rope_3d
                if get_env_device() == "xpu":
                    query_states, key_states, _ = xpu_fuse_rope(
                        query_states, key_states, None, None
                    )
                else:
                    bsz, q_len, num_heads, head_dim = query_states.shape
                    _, kv_seq_len, num_key_value_heads, _ = key_states.shape
                    if num_heads != num_key_value_heads:
                        query_states, _, _ = fused_rope(
                            query_states,
                            None,
                            None,
                            rotary_emb_base=self.config.rope_theta,
                        )
                        key_states, _, _ = fused_rope(
                            key_states,
                            None,
                            None,
                            rotary_emb_base=self.config.rope_theta,
                        )
                    else:
                        query_states, key_states, _ = fused_rope(
                            query_states,
                            key_states,
                            None,
                            rotary_emb_base=self.config.rope_theta,
                        )

        # [bsz, nh, t, hd]

        if use_cache:
            query_states = query_states.astype(query_states_dtype)
            key_states = key_states.astype(query_states_dtype)
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)

        # NOTE(for generation): use list instead of tuple to store the cache
        # tensors, so that we can clear the cache tensors for memory efficiency.
        past_key_value = [key_states, value_states] if use_cache else None

        if self.config.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        attn_output, attn_weights = scaled_dot_product_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            config=self.config,
            rr_flash_attn=self._rr_flash_attn if self.training else None,
            inbatch_pack_offset=inbatch_pack_offset,
            training=self.training,
        )
        return attn_output, attn_weights, past_key_value


class ErnieDecoderLayer(nn.Layer):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = ErnieAttention(config, layer_idx)
        self.mlp = ErnieMLP(config)
        Norm = RMSNorm if config.use_rmsnorm else LayerNorm
        if not config.use_rmsnorm and config.fuse_ln:
            Norm = FusedLayerNorm
        self.input_layernorm = Norm(config)
        self.post_attention_layernorm = Norm(config)
        # if get_env_device() == "xpu":
        self.residual_add1 = FusedDropoutImpl(
            config.hidden_dropout_prob, mode="upscale_in_train"
        )
        self.residual_add2 = FusedDropoutImpl(
            config.hidden_dropout_prob, mode="upscale_in_train"
        )
        # NOTE: FusedDropoutAdd may have problems when prob > 0. Disable it for now.
        # else:
        #     self.residual_add1 = FusedDropoutAdd(config.hidden_dropout_prob, mode="upscale_in_train")
        #     self.residual_add2 = FusedDropoutAdd(config.hidden_dropout_prob, mode="upscale_in_train")
        self.config = config

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        inbatch_pack_offset: Optional[paddle.Tensor] = None,
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
        # logger.debug(f'layer-hidden-dtype:{hidden_states.dtype}, norm={hidden_states.astype("float32").norm()}')
        hidden_states = self.input_layernorm(hidden_states)
        # logger.debug(f'after-norm-1:{hidden_states.dtype}, norm={hidden_states.astype("float32").norm()}')

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache,
            inbatch_pack_offset=inbatch_pack_offset,
        )

        if (
            self.config.tensor_parallel_degree > 1
            and self.config.hidden_dropout_prob > 0.0
        ):
            current_seed = (
                "local_seed" if self.config.sequence_parallel else "global_seed"
            )
            with get_rng_state_tracker().rng_state(current_seed):
                hidden_states = self.residual_add1(hidden_states, residual)
        else:
            hidden_states = self.residual_add1(hidden_states, residual)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if (
            self.config.tensor_parallel_degree > 1
            and self.config.hidden_dropout_prob > 0.0
        ):
            current_seed = (
                "local_seed" if self.config.sequence_parallel else "global_seed"
            )
            with get_rng_state_tracker().rng_state(current_seed):
                hidden_states = self.residual_add2(hidden_states, residual)
        else:
            hidden_states = self.residual_add2(hidden_states, residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class ErniePretrainedModel(PretrainedModel):
    config_class = ErnieConfig
    base_model_prefix = "ernie"

    @classmethod
    def _get_name_mappings(cls, config: ErnieConfig) -> StateDictNameMapping:
        mappings: StateDictNameMapping = []
        model_mappings = [
            ["embed_tokens.weight"],
            ["norm.weight"],
        ]
        for layer_index in range(
            config.num_hidden_layers
            if not config.remove_tail_layer
            else config.num_hidden_layers - 1
        ):
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
        根据配置和分割或合并的方式，获取张量并行映射关系。

        Args:
            config (TensorParallelConfig): TensorParallel的配置对象。
            is_split (bool, optional): 是否为分割操作。默认为True。

        Returns:
            Dict: 包含张量并行映射关系的字典。
        """

        from paddleformers.transformers.conversion_utils import split_or_merge_func

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
                            "lm_head.bias": partial(fn, is_column=True),
                        }
                    )
            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(
            config.num_hidden_layers
            if not config.remove_tail_layer
            else config.num_hidden_layers - 1
        )

        return mappings

    def _init_weights(self, layer):
        """Initialization hook"""
        if self.config.tensor_parallel_degree > 1:
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
                SMPRowParallelLinear,
                SMPColumnParallelLinear,
                VocabParallelEmbedding,
                TEFP8Linear,
                ErnieLMHead,
                nn.Embedding,
                NativeLinear,
                paddle.incubate.nn.FusedLinear,
            ),
        ):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            # logger.info(f'initializing pp:{type(layer)}')

            with rng_tracker():
                dtype = paddle.get_default_dtype()
                paddle.set_default_dtype("float32")
                # 在defualt dtype为bfloat16的情况下调用randn会报错，
                # 实际上randn已经支持bfloat16了，只是API检测层面的错误。
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
                    f"dist-init-fc: shape={layer.weight.shape}, "
                    f" range={self.config.initializer_range}, dtype={layer.weight.dtype} "
                    f' type={type(layer)},norm={layer.weight.astype("float32").norm().item()}'
                )

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
        config: ErnieConfig
    """

    def __init__(self, config: ErnieConfig):
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

        layers_list = [
            ErnieDecoderLayer(config, layer_idx)
            for layer_idx in range(
                config.num_hidden_layers - 1
                if config.remove_tail_layer
                else config.num_hidden_layers
            )
        ]

        self.layers = nn.LayerList(layers_list)
        Norm = RMSNorm if config.use_rmsnorm else LayerNorm
        if not config.use_rmsnorm and config.fuse_ln:
            Norm = FusedLayerNorm
        self.norm = Norm(config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @classmethod
    def _prepare_decoder_attention_mask(
        cls, attention_mask, input_shape, past_key_values_length, dtype
    ):
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
        output_attentions,
        past_key_value,
        use_cache,
        inbatch_pack_offset,
    ):
        """
        重新计算模型的训练过程。

        Args:
            layer_module (torch.nn.Module): 包含前馈神经网络和注意力机制的层模块。
            hidden_states (torch.Tensor): 形状为 (batch_size, sequence_length, hidden_size) 的隐藏状态张量。
            attention_mask (torch.Tensor): 形状为 (batch_size, sequence_length) 的注意力掩码张量。
            position_ids (torch.Tensor): 形状为 (batch_size, sequence_length) 的位置ID张量。
            output_attentions (bool): 如果为 True，则输出注意力权重。
            past_key_value (tuple): 包含过去键值对的元组。
            use_cache (bool): 如果为 True，则在前向传播中保存已缓存的张量。
            inbatch_pack_offset (int): 批次内的打包偏移量。

        Returns:
            torch.Tensor: 经过重新计算的隐藏状态张量。

        """

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        recompute_func = te_recompute if self.config.use_fp8 else recompute
        hidden_states = recompute_func(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            position_ids,
            output_attentions,
            past_key_value,
            use_cache,
            inbatch_pack_offset,
            use_reentrant=self.config.recompute_use_reentrant,
        )
        return hidden_states

    def forward(
        self,
        input_ids=None,
        position_ids=None,
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache

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

        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = paddle.shape(past_key_values[0][0])[1]
            seq_length_with_past += cache_length
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if self.embed_tokens is not None:
            inputs_embeds = inputs_embeds.astype(self.embed_tokens.weight.dtype)
        # logger.debug(f'emb out:{inputs_embeds.dtype} emb-weight:{self.embed_tokens.weight.dtype}"
        #              f" type={type(self.embed_tokens)}')

        if self.config.sequence_parallel or self.config.submatrix_parallel:
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
                logger.warning(
                    f"set attention_mask = None when (can_use_fa or can_mem_eff_attn) and attention_mask is not None, "
                    f"can_use_fa = {can_use_fa}, can_mem_eff_attn = {can_mem_eff_attn}, "
                    f"attention_mask is not None = {attention_mask is not None}"
                )
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

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            has_gradient = not hidden_states.stop_gradient
            if self.config.use_recompute and has_gradient:
                layer_outputs = self.recompute_training(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
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

        if use_cache:
            hidden_states = paddle.unsqueeze(hidden_states[:, -1, :], 1)
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
        )


class FusedHeadParallelCrossEntropy(PyLayer):
    """融合 Head 和 Loss 函数,在超长文上需要"""

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        weight,
        bias,
        labels,
        tensor_parallel_degree,
        mp_group=None,
        ignore_index=-100,
        seq_chunk_size=8192,
        transpose_y=False,
        fuse_linear=False,
        training=True,
    ):
        """doc"""

        ctx.tensor_parallel_degree = tensor_parallel_degree
        ctx.ignore_index = ignore_index
        ctx.seq_chunk_size = seq_chunk_size
        ctx.transpose_y = transpose_y
        ctx.fuse_linear = fuse_linear
        ctx.training = training

        ctx.hidden_states_shape = hidden_states.shape

        if ctx.tensor_parallel_degree > 1:
            ctx.mp_group = (
                fleet.get_hybrid_communicate_group().get_model_parallel_group()
                if mp_group is None
                else mp_group
            )
            ctx.rank = ctx.mp_group.rank
            ctx.world_size = ctx.mp_group.nranks
        else:
            ctx.mp_group = None
            ctx.rank = 0
            ctx.world_size = 1

        loss_all = []
        labels_all = []
        with paddle.no_grad():
            labels = labels.reshape_([-1])
            hidden_states = hidden_states.reshape_([-1, hidden_states.shape[-1]])

            num_tokens_per_rank = []
            if ctx.tensor_parallel_degree > 1:
                dist.stream.all_gather(
                    num_tokens_per_rank,
                    paddle.to_tensor(hidden_states.shape[0], dtype=paddle.int32),
                    group=ctx.mp_group,
                )
            ctx.num_tokens_per_rank = num_tokens_per_rank

            for idx in range(ctx.world_size):
                if idx == ctx.rank:
                    hidden_states_recv = hidden_states
                    labels_recv = labels
                else:
                    hidden_states_recv = paddle.empty(
                        [ctx.num_tokens_per_rank[idx], hidden_states.shape[-1]],
                        dtype=hidden_states.dtype,
                    )
                    labels_recv = paddle.empty(
                        [ctx.num_tokens_per_rank[idx]], dtype=labels.dtype
                    )

                if ctx.tensor_parallel_degree > 1:
                    dist.stream.broadcast(
                        hidden_states_recv,
                        src=ctx.mp_group.ranks[idx],
                        group=ctx.mp_group,
                    )
                    dist.stream.broadcast(
                        labels_recv, src=ctx.mp_group.ranks[idx], group=ctx.mp_group
                    )

                seq_len = hidden_states_recv.shape[0]
                num_chunk = (seq_len + ctx.seq_chunk_size - 1) // ctx.seq_chunk_size

                loss_chunk = []
                for chunk_idx in range(num_chunk):
                    start = chunk_idx * ctx.seq_chunk_size
                    end = min(start + ctx.seq_chunk_size, seq_len)
                    hidden_states_chunk = hidden_states_recv._slice(start, end)
                    labels_chunk = labels_recv._slice(start, end)

                    logits = parallel_matmul(
                        hidden_states_chunk,
                        weight,
                        bias=bias,
                        transpose_y=ctx.transpose_y,
                        tensor_parallel_degree=ctx.tensor_parallel_degree,
                        tensor_parallel_output=True,
                        fuse_linear=ctx.fuse_linear,
                        training=ctx.training,
                    )

                    with paddle.amp.auto_cast(False):
                        if ctx.tensor_parallel_degree > 1:
                            loss = mp_ops._c_softmax_with_cross_entropy(
                                logits.cast("float32"),
                                labels_chunk.unsqueeze(-1),
                                group=ctx.mp_group,
                                ignore_index=ctx.ignore_index,
                            )
                        else:
                            loss = paddle.nn.functional.softmax_with_cross_entropy(
                                logits.cast("float32"),
                                labels_chunk.unsqueeze(-1),
                                ignore_index=ctx.ignore_index,
                            )
                        loss_chunk.append(loss)
                loss_all.append(paddle.concat(loss_chunk, axis=0))
                labels_all.append(labels_recv)

            ctx.loss_concat_sections = [loss.shape[0] for loss in loss_all]
            loss_all = paddle.concat(loss_all, axis=0)
            labels_all = paddle.concat(labels_all, axis=0)

            tensor_inputs = [hidden_states, weight, bias, labels]
            ctx.save_for_backward(*tensor_inputs)

        return loss_all, labels_all

    @staticmethod
    def backward(ctx, loss_all_grad, labels_all_grad):
        """doc"""

        hidden_states, weight, bias, labels = ctx.saved_tensor()

        loss_all_grad_list = paddle.split(
            loss_all_grad, ctx.loss_concat_sections, axis=0
        )

        def detach_variable(inp):
            if inp is None:
                return None
            x = inp.detach()
            x.stop_gradient = inp.stop_gradient
            return x

        if weight.stop_gradient is False:
            weight_main_grad = paddle.zeros(weight.shape, dtype=paddle.float32)
        else:
            weight_main_grad = None
        if bias is not None and bias.stop_gradient is False:
            bias_main_grad = paddle.zeros(bias.shape, dtype=paddle.float32)
        else:
            bias_main_grad = None

        hidden_states = detach_variable(hidden_states)
        weight = detach_variable(weight)
        bias = detach_variable(bias)
        labels = detach_variable(labels)

        with paddle.base.dygraph.guard():
            tracer = paddle.base.framework._dygraph_tracer()
            tracer._has_grad = True

            for idx in range(ctx.world_size):
                if idx == ctx.rank:
                    hidden_states_recv = hidden_states
                    labels_recv = labels
                else:
                    hidden_states_recv = paddle.empty(
                        [ctx.num_tokens_per_rank[idx], hidden_states.shape[-1]],
                        dtype=hidden_states.dtype,
                    )
                    labels_recv = paddle.empty(
                        [ctx.num_tokens_per_rank[idx]], dtype=labels.dtype
                    )
                if ctx.tensor_parallel_degree > 1:
                    dist.stream.broadcast(
                        hidden_states_recv,
                        src=ctx.mp_group.ranks[idx],
                        group=ctx.mp_group,
                    )
                    dist.stream.broadcast(
                        labels_recv, src=ctx.mp_group.ranks[idx], group=ctx.mp_group
                    )
                hidden_states_recv.stop_gradient = False

                seq_len = hidden_states_recv.shape[0]
                num_chunk = (seq_len + ctx.seq_chunk_size - 1) // ctx.seq_chunk_size

                for chunk_idx in range(num_chunk):
                    start = chunk_idx * ctx.seq_chunk_size
                    end = min(start + ctx.seq_chunk_size, seq_len)
                    hidden_states_chunk = hidden_states_recv.slice(
                        axes=[0], starts=[start], ends=[end]
                    )
                    labels_chunk = labels_recv._slice(start, end)
                    loss_grad_chunk = loss_all_grad_list[idx]._slice(start, end)

                    logits = parallel_matmul(
                        hidden_states_chunk,
                        weight,
                        bias=bias,
                        transpose_y=ctx.transpose_y,
                        tensor_parallel_degree=ctx.tensor_parallel_degree,
                        tensor_parallel_output=True,
                        fuse_linear=ctx.fuse_linear,
                        training=ctx.training,
                    )

                    with paddle.amp.auto_cast(False):
                        if ctx.tensor_parallel_degree > 1:
                            loss_chunk = mp_ops._c_softmax_with_cross_entropy(
                                logits.cast("float32"),
                                labels_chunk.unsqueeze(-1),
                                group=ctx.mp_group,
                                ignore_index=ctx.ignore_index,
                            )
                        else:
                            loss_chunk = (
                                paddle.nn.functional.softmax_with_cross_entropy(
                                    logits.cast("float32"),
                                    labels_chunk.unsqueeze(-1),
                                    ignore_index=ctx.ignore_index,
                                )
                            )

                    with paddle.amp.auto_cast(enable=False):
                        paddle.autograd.backward(loss_chunk, loss_grad_chunk)

                    if weight_main_grad is not None:
                        weight_main_grad.add_(weight.grad.cast(paddle.float32))
                        weight.clear_gradient(True)
                    if bias_main_grad is not None:
                        bias_main_grad.add_(bias.grad.cast(paddle.float32))
                        bias.clear_gradient(True)

                if idx == ctx.rank:
                    hidden_states_grad = hidden_states_recv.grad
                    hidden_states_grad = hidden_states_grad.reshape(
                        ctx.hidden_states_shape
                    )

        if weight_main_grad is not None:
            weight_main_grad = weight_main_grad.astype(weight.dtype)
        if bias_main_grad is not None:
            bias_main_grad = bias_main_grad.astype(bias.dtype)

        if bias_main_grad is not None:
            return (
                hidden_states_grad,
                weight_main_grad,
                bias_main_grad,
            )
        else:
            return (
                hidden_states_grad,
                weight_main_grad,
            )


class ErniePretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for Ernie.
    It calculates the final loss.
    """

    def __init__(self, config, return_tuple=True):
        super(ErniePretrainingCriterion, self).__init__()
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
        self.token_balance_loss = config.token_balance_loss

    def forward(self, prediction_scores, masked_lm_labels):
        """forward"""
        if in_auto_parallel_align_mode():
            prediction_scores = concat_mp_with_grad(prediction_scores)

        if self.config.use_sparse_head_and_loss_fn:
            hidden_states, outlinear_weight, outlinear_bias = prediction_scores

            if self.config.sequence_parallel or self.config.submatrix_parallel:
                masked_lm_labels, sparse_label_idx = (
                    sequence_parallel_sparse_mask_labels(
                        masked_lm_labels, self.ignored_index
                    )
                )
                sparse_label_idx = sparse_label_idx.reshape([-1, 1])
                hidden_states = paddle.gather(hidden_states, sparse_label_idx, axis=0)
                hidden_states = AllGatherVarlenOp.apply(hidden_states)
            else:
                masked_lm_labels = masked_lm_labels.flatten()
                sparse_label_idx = paddle.nonzero(
                    masked_lm_labels != self.ignored_index
                ).flatten()
                masked_lm_labels = paddle.take_along_axis(
                    masked_lm_labels, sparse_label_idx, axis=0
                )

                hidden_states = hidden_states.reshape([-1, hidden_states.shape[-1]])
                hidden_states = paddle.take_along_axis(
                    hidden_states, sparse_label_idx.reshape([-1, 1]), axis=0
                )

            if self.config.use_recompute_loss_fn:
                offload_kwargs = {}
                if self.config.get("offload_lm_head", False):
                    offload_kwargs["offload_indices"] = [1]
                res = recompute(
                    self.forward_impl_with_calc_logits,
                    masked_lm_labels,
                    hidden_states,
                    outlinear_weight,
                    outlinear_bias,
                    **offload_kwargs,
                )
            else:
                logits = calc_lm_head_logits(
                    self.config,
                    hidden_states,
                    outlinear_weight,
                    outlinear_bias,
                    training=self.training,
                )
                res = self.forward_impl(logits, masked_lm_labels)
        elif self.config.use_recompute_loss_fn:
            if self.config.use_fused_head_loss_fn:
                if self.config.sequence_parallel or self.config.submatrix_parallel:
                    masked_lm_labels = ScatterOp.apply(masked_lm_labels, axis=-1)
                res = self.forward_impl_with_fused_head_loss_fn(
                    masked_lm_labels, *prediction_scores
                )
            else:
                assert isinstance(prediction_scores, tuple) and len(
                    prediction_scores
                ) in [3, 4], prediction_scores
                res = recompute(
                    self.forward_impl_with_calc_logits,
                    masked_lm_labels,
                    *prediction_scores,
                )
        else:
            res = self.forward_impl(prediction_scores, masked_lm_labels)

        return res

    def forward_impl_with_fused_head_loss_fn(
        self, masked_lm_labels, hidden_states, outlinear_weight, outlinear_bias
    ):
        masked_lm_labels.stop_gradient = True
        masked_lm_loss, masked_lm_labels_all = FusedHeadParallelCrossEntropy.apply(
            hidden_states,
            outlinear_weight,
            outlinear_bias,
            masked_lm_labels,
            self.config.tensor_parallel_degree,
            ignore_index=self.ignored_index,
            seq_chunk_size=self.config.get("loss_subbatch_seqlen", 32768),
            transpose_y=self.config.tie_word_embeddings,
            fuse_linear=self.config.fuse_linear,
            training=self.training,
        )
        lossmask = masked_lm_labels_all != self.ignored_index
        if (~lossmask).all():  # empty span
            logger.warning(
                f"encounter empty span when calculate loss, ignored_index={self.ignored_index}"
            )
            loss = paddle.mean(masked_lm_loss) * 0.0
            loss_sum = masked_lm_loss.sum().detach()
        else:
            lossmask = lossmask.reshape([-1]).cast(paddle.float32)
            # 逐位对齐, 全精度聚合
            masked_lm_loss = paddle.sum(
                masked_lm_loss.cast(paddle.float32).reshape([-1]) * lossmask
            )
            loss = masked_lm_loss / lossmask.sum()
            if self.token_balance_loss:
                _loss = masked_lm_loss / self.config.token_balance_seqlen
                get_global_training_logs().update(token_balance_loss=_loss.detach())
                loss = _loss - _loss.detach() + loss.detach()  # for 对线
            loss_sum = masked_lm_loss.sum().detach()
        if not self.return_tuple:  # only used in pp
            if self.training:
                return loss
            return loss_sum
        return loss, loss_sum

    def forward_impl_with_calc_logits(
        self, masked_lm_labels, hidden_states, outlinear_weight, outlinear_bias
    ):
        """doc"""

        logits = calc_lm_head_logits(
            self.config,
            hidden_states,
            outlinear_weight,
            outlinear_bias,
            training=self.training,
        )

        return self.forward_impl(logits, masked_lm_labels)

    def loss_impl(self, prediction_scores, masked_lm_labels):
        """extract loss impl for subbatch"""
        prediction_scores = prediction_scores.cast("float32")
        masked_lm_loss = self.loss_func(
            prediction_scores, masked_lm_labels.unsqueeze(-1)
        )

        if self.config.z_loss_lambda is not None:
            if self.enable_parallel_cross_entropy:
                pg = fleet.get_hybrid_communicate_group().get_model_parallel_group()
            else:
                pg = None
            # record logits when using z loss
            if moe_router_loss_ops is not None and hasattr(
                moe_router_loss_ops, "compute_sum_min_max"
            ):
                sum_logits, min_logits, max_logits = (
                    moe_router_loss_ops.compute_sum_min_max(
                        prediction_scores, axis=-1, keepdim=True
                    )
                )
            else:
                sum_logits = prediction_scores.sum(axis=-1, keepdim=True)
                min_logits = prediction_scores.min(axis=-1, keepdim=True)
                max_logits = prediction_scores.max(axis=-1, keepdim=True)

            if self.enable_parallel_cross_entropy:
                if global_training_logs_enabled():
                    dist.all_reduce(sum_logits, op=dist.ReduceOp.SUM, group=pg)
                    dist.all_reduce(min_logits, op=dist.ReduceOp.MIN, group=pg)
                dist.all_reduce(max_logits, op=dist.ReduceOp.MAX, group=pg)
            mean_logits = sum_logits / prediction_scores.shape[-1]
            if self.enable_parallel_cross_entropy:
                mean_logits = mean_logits / self.config.tensor_parallel_degree
            global_training_logs = get_global_training_logs()
            if global_training_logs_enabled():
                global_training_logs.update(
                    logits_mean=mean_logits.mean().detach(),
                    logits_min=min_logits.mean().detach(),
                    logits_max=max_logits.mean().detach(),
                    logits_gap_mean=(max_logits - min_logits).mean().detach(),
                    logits_gap_max=(max_logits - min_logits).max().detach(),
                )

            if self.config.z_loss_lambda != 0 or self.config.enable_entropy_logging:
                z_loss, log_z_global = ZLossOp.apply(
                    prediction_scores, max_logits, self.config.z_loss_lambda, group=pg
                )
                masked_lm_loss = masked_lm_loss + z_loss - z_loss.detach()
                if global_training_logs_enabled():
                    global_training_logs.update(z_loss=z_loss.mean().detach())

                    logp = prediction_scores - log_z_global
                    sum_prob_logits = (logp.exp_().multiply_(prediction_scores)).sum(
                        -1, keepdim=True
                    )
                    if self.enable_parallel_cross_entropy:
                        dist.all_reduce(sum_prob_logits, op=dist.ReduceOp.SUM, group=pg)
                    entropy = (log_z_global - sum_prob_logits).squeeze(
                        -1
                    )  # H(p) = LSE(x) - sum_i{p_i x_i}
                    """
                        entropy = (p*logp).sum()
                        =>(p*lo - plogZ).sum()
                        =>(p*lo).sum() - logZ*(p).sum()
                        =>(p*lo).sum() - logZ
                    """
                    global_training_logs.update(logits_entropy=entropy.mean().detach())
                    global_training_logs.update(
                        logits_entropy_tgt=entropy[masked_lm_labels >= 0]
                        .mean()
                        .detach()
                    )
        return masked_lm_loss

    def forward_impl(self, prediction_scores, masked_lm_labels):
        """include reduce loss"""
        if self.enable_parallel_cross_entropy:
            assert prediction_scores.shape[-1] != self.config.vocab_size, (
                f"enable_parallel_cross_entropy, the vocab_size should be splited:"
                f" {prediction_scores.shape[-1]}, {self.config.vocab_size}"
            )

        with paddle.amp.auto_cast(False):
            prediction_scores_dims = len(prediction_scores.shape)
            if prediction_scores_dims == 2 and prediction_scores.shape[
                0
            ] > self.config.get("loss_subbatch_seqlen", 32768):
                sb_loss_func = subbatch(
                    self.loss_impl,
                    [0, 1],
                    [0, 0],
                    self.config.get("loss_subbatch_seqlen", 32768),
                    0,
                )
                masked_lm_loss = sb_loss_func(prediction_scores, masked_lm_labels)
            elif prediction_scores_dims == 3 and prediction_scores.shape[
                1
            ] > self.config.get("loss_subbatch_seqlen", 32768):
                sb_loss_func = subbatch(
                    self.loss_impl,
                    [0, 1],
                    [1, 1],
                    self.config.get("loss_subbatch_seqlen", 32768),
                    1,
                )
                masked_lm_loss = sb_loss_func(prediction_scores, masked_lm_labels)
            else:
                masked_lm_loss = self.loss_impl(prediction_scores, masked_lm_labels)

            lossmask = masked_lm_labels != self.ignored_index
            if (~lossmask).all():  # empty span
                logger.warning(
                    f"encounter empty span when calculate loss, ignored_index={self.ignored_index}"
                )
                loss = paddle.mean(masked_lm_loss) * 0.0
                loss_sum = masked_lm_loss.sum().detach()
            else:
                lossmask = lossmask.reshape([-1]).cast(paddle.float32)
                # 逐位对齐, 全精度聚合
                masked_lm_loss = paddle.sum(
                    masked_lm_loss.cast(paddle.float32).reshape([-1]) * lossmask
                )
                loss = masked_lm_loss / lossmask.sum()
                if self.token_balance_loss:
                    _loss = masked_lm_loss / self.config.token_balance_seqlen
                    get_global_training_logs().update(token_balance_loss=_loss.detach())
                    loss = _loss - _loss.detach() + loss.detach()  # for 对线
                loss_sum = masked_lm_loss.sum().detach()
        if not self.return_tuple:  # only used in pp
            if self.training:
                return loss
            return loss_sum
        return loss, loss_sum


class ErnieLMHead(nn.Layer):
    def __init__(self, config):
        super(ErnieLMHead, self).__init__()
        self.config = config
        if config.tensor_parallel_degree > 1:
            vocab_size = config.vocab_size // config.tensor_parallel_degree
        else:
            vocab_size = config.vocab_size

        self.weight = self.create_parameter(
            shape=(
                [vocab_size, config.hidden_size]
                if config.tie_word_embeddings
                else [config.hidden_size, vocab_size]
            ),
            dtype=paddle.get_default_dtype(),
        )
        logger.info(
            f"output-weight:{self.weight.shape} config.tie_word_embeddings={config.tie_word_embeddings}"
        )
        if config.weight_share_add_bias and config.use_bias:
            self.bias = self.create_parameter(
                shape=[vocab_size],
                dtype=paddle.get_default_dtype(),
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.constant.Constant(0.0)
                ),
            )
        else:
            self.bias = None

        # Must set distributed attr for Tensor Parallel !
        self.weight.is_distributed = (
            True if (vocab_size != config.vocab_size) else False
        )
        if config.weight_share_add_bias and config.use_bias:
            self.bias.is_distributed = (
                True if (vocab_size != config.vocab_size) else False
            )

        if self.weight.is_distributed:
            self.weight.split_axis = 1
        if (
            config.weight_share_add_bias
            and config.use_bias
            and self.bias.is_distributed
        ):
            self.bias.split_axis = 0

        if self.config.use_recompute_loss_fn:
            logger.info(
                "Using recompute_loss_fn, the calculation of logits will be moved into "
                "loss_fn for memory optimization"
            )

    def forward(self, hidden_states, tensor_parallel_output=None):
        if self.config.use_recompute_loss_fn or self.config.use_sparse_head_and_loss_fn:
            out_tensors = (
                (hidden_states, self.weight, self.bias)
                if tensor_parallel_output is None
                else (hidden_states, self.weight, self.bias, tensor_parallel_output)
            )

            return out_tensors

        return calc_lm_head_logits(
            self.config,
            hidden_states,
            self.weight,
            self.bias,
            tensor_parallel_output,
            training=self.training,
        )


class ErnieForCausalLM(ErniePretrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)

        if config.sequence_parallel:
            logger.info(f"using sequence_parallel, input seqlen={config.seqlen}")
            if config.using_dynamic_sequence_length:
                assert (
                    not config.micro_batch_size
                ), "sequence-parallel needs micro_batch_size setting when using dygramic_sequnence_length"
            else:
                assert config.seqlen is not None

            assert (
                config.tensor_parallel_degree > 1
            ), f"sequence-parallel needs mp>1, got mp={config.tensor_parallel_degree}"

        if config.submatrix_parallel:
            assert config.seqlen is not None
            assert (
                config.tensor_parallel_degree > 1
            ), f"submatrix_parallel only works in mp, got mp={self.tensor_parallel_degree}"
            assert (
                config.sequence_parallel == 0
            ), "enable submatrix_parallel must disable sequence-parallel"

        # initialize-trick for big model, see
        # https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/README.md#std-init
        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(
            f"change initializer-range from {config.initializer_range} to {new_initializer_range}"
        )
        config.initializer_range = new_initializer_range
        self.config = config

        self.ernie = ErnieModel(config)
        self.lm_head = ErnieLMHead(config)
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
        super()._post_init(self, original_init, *args, **kwargs)
        factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
        logger.info(f"using post init div: factor:{factor}")
        with paddle.no_grad():
            for l in self.ernie.layers:
                l.self_attn.o_proj.weight.scale_(factor)
                l.mlp.down_proj.weight.scale_(factor)

    def get_input_embeddings(self):
        return self.ernie.embed_tokens

    def set_input_embeddings(self, value):
        self.ernie.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.ernie = decoder

    def get_decoder(self):
        return self.ernie

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
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
        update_model_kwargs_for_generation
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
        loss_mask=None,
    ):
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
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                inbatch_pack_offset=inbatch_pack_offset,
            )

        hidden_states = outputs[0]
        # if labels is None，means we need full output, instead of tensor_parallel_output
        # tensor_parallel_output is togather with ParallelCrossEntropy
        tensor_parallel_output = (
            self.config.tensor_parallel_output
            and labels is not None
            and self.config.tensor_parallel_degree > 1
        )

        logits = self.lm_head(
            hidden_states,
        )  # tensor_parallel_output=tensor_parallel_output)

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
            )
        # Pretrain & Eval 必须有labels
        assert labels is not None
        loss, loss_sum = self.criterion(logits, labels)
        return loss, loss_sum
