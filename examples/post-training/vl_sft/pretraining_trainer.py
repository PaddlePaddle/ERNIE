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

"""The trainer class used in training."""

__all__ = [
    "PretrainingTrainer",
]


import contextlib
import json
import logging
import math
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from types import MethodType
from typing import Optional

import numpy as np
import paddle
import paddle.amp.auto_cast as autocast
from paddle import framework, nn
from paddle.base import core
from paddle.distributed.fleet.utils import mix_precision_utils
from paddleformers.trainer import (
    Trainer,
    TrainingArguments,
    speed_metrics,
)
from paddleformers.utils.tools import get_env_device

try:
    from paddleformers.trainer import TRAINING_ARGS_NAME
except ImportError:
    TRAINING_ARGS_NAME = "training_args.bin"

# from paddlenlp.trainer.trainer import PREFIX_CHECKPOINT_DIR, SCHEDULER_NAME, TRAINER_STATE_NAME, OPTIMIZER_NAME
from paddleformers.utils.env import PADDLE_OPTIMIZER_NAME

OPTIMIZER_NAME = PADDLE_OPTIMIZER_NAME  # for compatibility

try:
    from paddleformers.trainer.trainer import PADDLE_WEIGHT_FILE_NAME as PADDLE_WEIGHTS_NAME
except ImportError:
    from paddleformers.utils.env import PADDLE_WEIGHTS_NAME
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
    HybridParallelOptimizer,
)
from paddle.distributed.fleet.utils.sequence_parallel_utils import register_sequence_parallel_allreduce_hooks
from paddleformers.trainer.trainer_callback import PrinterCallback
from paddleformers.trainer.trainer_utils import (
    ShardingOption,
)
from paddleformers.trainer.utils import add_start_docstrings
from paddleformers.transformers.model_utils import _add_variant, unwrap_model
from paddleformers.utils.log import logger

from ernie.callbacks import (
    ClipGradByAdaptiveNormCallback,
    GCCallback,
    LoggingCallback,
    OptimizerCallback,
    RefinedRecomputeCheckCallback,
    ReshardSaveExitCallback,
    SPGradSyncCallback,
    StopperCallback,
    TensorBoardCallback,
)

from ernie.callbacks.moe_logging_callback import MoeLoggingCallback
from ernie.lr_schedulers import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from ernie.utils.misc import global_training_logs
from ernie.utils.training_utils import reset_per_device_batch_size

DATATYPE_2_ID = {"mm": 0, "lm": 1, "audio": 2}


try:
    from paddleformers.trainer.trainer import is_dp_group_support_in_group_sharded_parallel
except Exception:

    def is_dp_group_support_in_group_sharded_parallel():
        """
        hack for paddlenlp develop branch.
        """
        return True


try:
    from paddle.distributed import in_auto_parallel_align_mode
except Exception:

    def in_auto_parallel_align_mode():
        """
        hack for paddle develop branch.
        """
        return False


def distributed_optimizer_maybe_hack(
    optimizer,
    use_moe,
):
    """rewrite the function of fleet.distributed_optimizer"""
    if use_moe:
        from src.trainers.dygraph_optimizer.hybrid_parallel_optimizer import (
            HybridParallelOptimizer as MoEHybridParallelOptimizer,
        )

        # moe下需要定制 `HybridParallelOptimizer` 函数以下重写了 `fleet.distributed_optimizer`
        fleet_env = fleet.fleet
        fleet_env.user_defined_optimizer = optimizer
        # TODO：sharding group 内做 moe
        hp_optim = MoEHybridParallelOptimizer(optimizer, fleet_env._hcg, fleet_env._user_defined_strategy)

        if fleet_env._user_defined_strategy.hybrid_configs["pp_configs"].dp_comm_overlap:
            hp_optim._dp_enable = False

        if fleet_env._user_defined_strategy.hybrid_configs["pp_configs"].sharding_comm_overlap:
            hp_optim._sharding_enable = False
        return hp_optim
    else:
        return fleet.distributed_optimizer(optimizer)


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class PreTrainingArguments(TrainingArguments):
    """pretraining arguments"""

    vocab_path: str = field(default=None, metadata={"help": "eb35 streaming data vocab"})
    task_need_convert: str = field(default=None, metadata={"help": "glm task id"})
    multimodal: bool = field(default=False, metadata={"help": "whether training with multimodal"})
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from "
            "https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
        },
    )
    vision_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from "
            "https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
        },
    )
    inception_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from "
            "https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
        },
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "global random seed factor."},
    )
    eval_iters: int = field(
        default=-1,
        metadata={"help": "eval iteration for every evaluation."},
    )
    num_consecutive: int = field(
        default=1,
        metadata={
            "help": "h5 file continuous sampling. For performance reason, read one ID at once."
        },
    )
    train_emb_only: int = field(
        default=0,
        metadata={"help": "train emb only flag"},
    )
    use_train_part_sharding: Optional[int] = field(
        default=1,
        metadata={"help": "according to file, cut data into pieces. Only used in pre-training."},
    )
    min_lr: float = field(
        default=0.0,
        metadata={"help": "minus learning rate"},
    )
    use_map_style_data: int = field(
        default=0,
        metadata={
            "help": "以为HF dataset为中心的 MapStyle SFT数据流（支持ShareGPT/DistillGPT)等数据",
        },
    )
    use_streaming_data: int = field(
        default=0,
        metadata={
            "help": "use streaming data",
        },
    )
    dataset: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    data_load_process_num: int = field(
        default=10, metadata={"help": "use multi process to speed up raw data reading"}
    )

    data_dir: str = field(default=None, metadata={"help": "data path (a dir)"})

    data_filelist: tuple = field(default=None, metadata={"help": "data file list"})
    data_weights: tuple = field(default=None, metadata={"help": "data weights"})

    dev_data: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )

    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    global_batch_size: int = field(
        default=-1,
        metadata={
            "help": "if `global_batch_size` and `per_device_train_batch_size` is provied, "
            "`gradient_accumulation_steps` will be ignored"
        },
    )
    init_global_batch_size: int = field(
        default=-1,
        metadata={
            "help": "开启动态Batching。必须提供`global_batch_size`, "
            "global_batch_size 会在 `batch_size_warumup_steps` 步内从 "
            "`init_global_batch_size` 提升到 `global_batch_size`, "
            "每次 `batchsize` 的提升量为`batch_size_warmup_increment`"
        },
    )
    batch_size_warmup_steps: int = field(
        default=-1,
        metadata={
            "help": "开启动态Batching。必须提供`global_batch_size`, "
            "global_batch_size 会在 `batch_size_warumup_steps` 步内从 "
            "`init_global_batch_size` 提升到 `global_batch_size`, "
            "每次 `batchsize` 的提升量为`batch_size_warmup_increment`"
        },
    )
    batch_size_warmup_increment: int = field(
        default=1,
        metadata={
            "help": "开启动态Batching。必须提供`global_batch_size`, "
            "global_batch_size 会在 `batch_size_warumup_steps` 步内从 "
            "`init_global_batch_size` 提升到 `global_batch_size`, "
            "每次 `batchsize` 的提升量为`batch_size_warmup_increment`"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    # tokenizer_name: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    # )
    init_ckpt: Optional[str] = field(
        default=None,
        metadata={},
    )
    sequence_parallel: Optional[int] = field(
        default=0,
        metadata={},
    )

    config_file: Optional[str] = field(
        default=None,
        metadata={"help": "config file (YAML) to update hyper-parameters"},
    )
    virtual_pp_degree: Optional[int] = field(
        default=1,
        metadata={
            "help": "vpp",
        },
    )
    from_scratch: Optional[int] = field(default=1, metadata={"help": "是否重头训练"})
    no_shuffle: Optional[int] = field(default=0, metadata={"help": "不要shuffle数据"})
    no_part_shuffle: Optional[int] = field(default=0, metadata={"help": "不进行part内数据shuffle"})
    record_optimizer_stat: Optional[bool] = field(default=False, metadata={"help": "是否记录优化器momentum信息"})
    skip_optimizer_badcases: Optional[bool] = field(
        default=False, metadata={"help": "是否跳过optimizer badcase很多的step"}
    )
    same_data: Optional[bool] = field(
        default=None, metadata={"help": "热启时，数据、配比、DP数是否完全一致, 支持续线"}
    )
    base_seq_length: Optional[int] = field(default=4096, metadata={"help": "reeao最小seq_length"})
    shuffle_consecutive: Optional[bool] = field(
        default=False,
        metadata={"help": "是否对num_consecutive片段进行shuffle, same_data=True热启时，该值需与上一次保持一致"},
    )
    global_shuffle_num_examples: Optional[int] = field(
        default=0,
        metadata={
            "help": "part间shuffle的num_example总数限制，默认不做限制, "
            "这个值与最小配比的积 必须大于1, 改变该值时，需要设置same_data=False"
        },
    )
    adaptive_norm_clip: Optional[bool] = field(
        default=False, metadata={"help": "是否启用 AdaptiveNormClip 梯度裁剪策略"}
    )
    adaptive_norm_clip_ratio: Optional[float] = field(
        default=1.03, metadata={"help": "AdaptiveNormClip 裁剪阈值, 大于设定的阈值才会启动裁剪"}
    )
    adaptive_norm_force_clear_state: Optional[bool] = field(
        default=False, metadata={"help": "AdaptiveNormClip 强制清空 state dict"}
    )
    adaptive_norm_shard_clip: Optional[bool] = field(
        default=False, metadata={"help": "AdaptiveNormClip 在切分参数上是否在局部clip"}
    )
    adaptive_norm_enable_record: Optional[bool] = field(
        default=False, metadata={"help": "AdaptiveNormClip 是否启用统计历史norm值"}
    )
    adaptive_norm_start_clip_steps: Optional[int] = field(
        default=100, metadata={"help": "AdaptiveNormClip 开始裁剪的step"}
    )
    adaptive_norm_enable_record_clip_history: Optional[bool] = field(
        default=False, metadata={"help": "AdaptiveNormClip 是否启用统计历史裁剪的记录"}
    )
    adaptive_norm_verbose: Optional[bool] = field(
        default=False, metadata={"help": "AdaptiveNormClip 是否开启裁剪日志打印"}
    )
    use_async_save: Optional[bool] = field(default=False, metadata={"help": "是否开启异步保存功能"})
    pre_alloc_memory: float = field(
        default=0.0,
        metadata={
            "help": "Pre-allocate one specific-capacity empty tensor "
            "and release it for avoiding memory fragmentation"
        },
    )
    enable_global_training_logs: bool = field(default=False, metadata={"help": "是否启用global_training_logs"})
    use_dummy_dataset: Optional[bool] = field(default=False, metadata={"help": "是否使用DummyDataSet, 仅用于Debug"})
    reshard_save_then_exit: Optional[bool] = field(default=False, metadata={"help": "是否在reshard后直接退出程序"})
    moe_group: Optional[str] = field(default="dp", metadata={"help": "moe 的通信组，目前支持“dp|sharding|mp|dummy”"})
    use_moe: Optional[bool] = field(default=False, metadata={"help": "expert parallel 临时替代"})
    log_global_grad_norm: Optional[bool] = field(
        default=False, metadata={"help": "打印全局grad-norm, 只有在开启`enable_global_training_logs`时生效"}
    )
    multi_token_pred_depth: Optional[int] = field(
        default=0,
        metadata={},
    )
    enable_mtp_magic_send: Optional[bool] = field(default=False, metadata={"help": ""})

    lr_scheduler: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use. suppor linear, cosine, constant, constant_with_warmup"},
    )
    decay_function: str = field(
        default="half_life",
        metadata={"help": "The decay function for WSD LR scheduler. support half_life(default), 1-sqrt"},
    )

    # image_token_len: int = field(default=64, metadata={"help": "number of images tokens from resampler per image"})
    freeze_config: str = field(
        default="",
        metadata={
            "help": (
                "Some additional config for freeze params, we provide some option to config it."
                "following config is support: freeze_vision,freeze_lm"
            )
        },
    )
    moe_gate_lr_ratio: float = field(
        default=None,
        metadata={"help": ("启用 moe 时，对 gate/router 的 LR 做特殊处理")},
    )
    vit_lr_ratio: float = field(
        default=None,
        metadata={"help": ("启用vit训练时，对 vit 的 LR 做特殊处理")},
    )
    visual_ld: float = field(
        default=None,
        metadata={"help": ("启用vit训练时，对 vit 的 LR 做特殊处理")},
    )
    modality_interleave: str = field(default="acc", metadata={"help": "acc"})
    modality_ratio: tuple = field(
        default=None,
        metadata={"help": "ratio of modality tokens to be masked out"},
    )

    pp_need_data_degree: int = field(
        default=0,
        metadata={"help": "pipline 并行中的机器也需要 fetch 数据，提升吞吐，搭配 `ErniemmMoEForCausalPipe` 使用"},
    )
    pp_need_data: bool = field(default=False, metadata={"help": "向前兼容"})
    balanced_image_preprocess: bool = field(default=False, metadata={"help": "向前兼容"})
    remote_inception_model_name_or_path: str = field(
        default=None,
        metadata={"help": "remote inception model name or path"},
    )
    remote_vision_model_name_or_path: str = field(
        default=None,
        metadata={"help": "remote vision model name or path"},
    )
    remote_freeze_expert_model_name_or_path_prefix: str = field(
        default=None,
        metadata={"help": "remote export model name or path"},
    )

    freeze_expert_model_name_or_path: str = field(
        default=None,
        metadata={"help": "local export model name or path"},
    )

    gc_interval: int = field(default=0, metadata={"help": "手动gc的间隔"})
    skip_load_data_seq_cache: bool = field(
        default=False, metadata={"help": "是否跳过加载数据序列缓存，跳过会导致初始化时间增加"}
    )
    vit_second_fwd_batch_size: int = field(default=None, metadata={"help": "vit second forward batch size"})
    use_sp_callback: bool = field(
        default=True, metadata={"help": "采用 SP callback 会跳过 SPHook的实现，避免梯度算重复"}
    )
    debug_reeao_dataset_world_size: int = field(default=0, metadata={"help": "debug reeao dataset world size"})
    moe_use_aux_free_update_coef: float = field(
        default=1.0e-3,
        metadata={"help": "moe aux free update coef, 只有 `model_config` 中启用了 `aux_free` 策略才生效"},
    )
    # variable_resolution: bool = field(default=True, metadata={"help": "是否使用变长vit"})

    use_fp8: bool = field(
        default=False,
        metadata={"help": "whether to use fp8 training"},
    )
    fp8_force_clear_state: bool = field(
        default=False,
        metadata={"help": "whether to force clear TE FP8 amax state when resume"},
    )
    enable_fp8_quantize_analysis: bool = field(
        default=False,
        metadata={"help": "whether to enable FP8 quantize analysis"},
    )
    disable_pipeline_warmup: bool = field(
        default=False,
        metadata={"help": "whether to disable pipeline warmup"},
    )
    global_logging_interval: int = field(
        default=1,
        metadata={"help": "the logging interval of global_training_logs"},
    )
    custom_data_status: str = field(default=None, metadata={"help": "load data status from custom trainer_state.json"})
    train_moe_only: int = field(default=None, metadata={"help": "train moe params only"})
    use_ortho_loss_callback: bool = field(default=False, metadata={"help": "是否开启正交loss callback"})
    use_doc_pack_atten: bool = field(default=False, metadata={"help": "是否开启Doc Pack Atten"})
    enable_flash_save_mode: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable Flash Save Mode"},
    )

    # @property
    # def use_moe(self):
    #     """_summary_

    #     Returns:
    #         _type_: _description_
    #     """
    #     # return getattr(self, "use_expert_parallel", self._use_moe)
    #     return True  # getattr(self, "use_expert_parallel", self._use_moe)

    # @use_moe.setter
    # def use_moe(self, value):
    #     """_summary_

    #     Args:
    #         value (_type_): _description_
    #     """
    #     self.use_expert_parallel = value
    #     self._use_moe = value

    @property
    def need_data(self):
        """
        判断是否需要加载数据。

        Args:
            无

        Returns:
            bool: 如果需要加载数据则返回True，否则返回False。

        """
        # mp0、pp0状态 卡才需要load数据
        if self.pp_need_data_degree:
            assert self.pipeline_parallel_degree > 1
            assert self.pp_need_data_degree >= 2 and self.pp_need_data_degree <= self.pipeline_parallel_degree, (
                self.pp_need_data_degree,
                self.pipeline_parallel_degree,
            )
            # shift by 1 to avoid last pp no nee data
            no_need_data_range = list(range(self.pp_need_data_degree - 1, self.pipeline_parallel_degree - 1))
            return self.tensor_parallel_rank == 0 and (self.pipeline_parallel_rank not in no_need_data_range)
        return self.pipeline_parallel_rank == 0 and self.tensor_parallel_rank == 0

    @property
    def combine_batch(self):
        """合并batch 用于增大seqlen

        Returns:
            _type_: _description_
        """
        return self.max_seq_length // self.base_seq_length

    @property
    def reeao_dataset_rank(self):
        """
        考虑 pp /sharding/ dp 总和的数据流 rank
        """
        if not self.pp_need_data_degree:
            return super().dataset_rank
        no_need_data_range = list(range(self.pp_need_data_degree - 1, self.pipeline_parallel_degree - 1))
        ranks = [i for i in range(self.pipeline_parallel_degree) if i not in no_need_data_range]
        if self.pipeline_parallel_rank not in ranks:
            return None
        reeao_pp_rank = ranks.index(self.pipeline_parallel_rank)
        pp_need_data_rank = self.pipeline_parallel_rank
        return (
            max(self.sharding_parallel_degree, 1) * max(self.pp_need_data_degree, 1) * self.data_parallel_rank
            + max(self.pp_need_data_degree, 1) * self.sharding_parallel_rank
            + reeao_pp_rank
        )

    @property
    def reeao_dataset_world_size(self):
        """
        考虑 pp /sharding/ dp 总和的数据流 worldsize
        """
        if not self.pp_need_data_degree:
            return super().dataset_world_size
        return (
            max(self.sharding_parallel_degree, 1)
            * max(self.pp_need_data_degree, 1)
            * max(self.data_parallel_degree, 1)
        )

    def __post_init__(self):
        super().__post_init__()
        # if self.sharding_parallel_degree > 1 and self.data_parallel_degree > 1:
        #     # MP/PP下， 当前框架不支持同时开启 sharding 和 DP
        #     assert (
        #         self.pipeline_parallel_degree <= 1 and self.tensor_parallel_degree <= 1
        #      ), f"when using mp/pp, `data_parallel_degree` should be 1 but receive {self.data_parallel_degree}"
        if in_auto_parallel_align_mode():
            self.adaptive_norm_clip = False
            self.adaptive_norm_clip_ratio = 0.0
            self.no_shuffle = 1
            self.no_part_shuffle = 1

        if self.global_batch_size > 0:
            micro_bsz, acc_steps = reset_per_device_batch_size(
                self.global_batch_size,
                self.per_device_train_batch_size,
                self.dataset_world_size,
            )
            logger.info(f"global_batch={self.global_batch_size} micro-bsz:{micro_bsz}, accumulate_steps:{acc_steps}")
            if (
                acc_steps != 1
                and self.gradient_accumulation_steps != 1
                and acc_steps != self.gradient_accumulation_steps
            ):
                raise ValueError(
                    f"global_accumulation_steps={self.gradient_accumulation_steps}"
                    f"& global_batch={self.global_batch_size} are both set"
                )
            self.per_device_train_batch_size, self.gradient_accumulation_steps = (
                micro_bsz,
                acc_steps,
            )

        if self.batch_size_warmup_steps > 0:
            assert self.global_batch_size > 0, self.global_batch_size
            assert self.init_global_batch_size > 0, self.init_global_batch_size
            self.max_gradient_accumulation_steps = self.gradient_accumulation_steps  # hack add new
            (
                self.per_device_train_batch_size,
                self.gradient_accumulation_steps,
            ) = reset_per_device_batch_size(
                self.init_global_batch_size,
                self.per_device_train_batch_size,
                self.dataset_world_size,
            )
            logger.info(
                f"using progressive batching, accumulate step will increese from {self.gradient_accumulation_steps}"
                f"to {self.max_gradient_accumulation_steps} in {self.batch_size_warmup_steps} steps"
            )
        else:
            self.max_gradient_accumulation_steps = self.gradient_accumulation_steps  # hack add new

        if self.pipeline_parallel_degree > 1:
            self.per_device_eval_batch_size = (
                self.per_device_train_batch_size * self.gradient_accumulation_steps
            )  # hack Eval for PP!
            logger.warning(f"eval_batch_size set to {self.per_device_eval_batch_size} in Pipeline Parallel!")
            user_defined_strategy = fleet.fleet._user_defined_strategy
            user_defined_strategy.strategy.pipeline_configs.accumulate_steps = self.gradient_accumulation_steps
            if self.pp_need_data and not self.pp_need_data_degree:
                self.pp_need_data_degree = self.pipeline_parallel_degree
            if self.pp_need_data_degree:
                assert self.gradient_accumulation_steps % self.pp_need_data_degree == 0, (
                    f"gradient_accumulation_steps[{self.gradient_accumulation_steps}] should be divisible by "
                    f"pp_need_data_degree[{self.pp_need_data_degree}]"
                )
                # pp_need_data_degree下，args的acc 需要//pp数量，欺骗 在prepare_inputs
                self.gradient_accumulation_steps = self.gradient_accumulation_steps // self.pp_need_data_degree
                logger.info(
                    f"pp-need-data hack args.gradient_accumulation_steps to - {self.gradient_accumulation_steps}"
                )
            self.max_gradient_accumulation_steps = self.gradient_accumulation_steps  # hack add new
            logger.info(f"fixing pp configs: {user_defined_strategy.pipeline_configs}")
        else:
            self.per_device_eval_batch_size = self.per_device_train_batch_size
            logger.warn(f"eval_batch_size set to {self.per_device_eval_batch_size}")

        if self.sharding_parallel_degree > 1:
            sharding_parallel_config = (
                set(self.sharding_parallel_config.split(" ")) if self.sharding_parallel_config else set()
            )
            sharding_comm_overlap_non_pp = (
                True
                if "shardingv1_comm_overlap" in sharding_parallel_config
                or "sharding_comm_overlap" in sharding_parallel_config
                else False
            )
            if sharding_comm_overlap_non_pp:
                # update grad acc steps
                assert hasattr(fleet.fleet, "_user_defined_strategy")
                user_defined_strategy = fleet.fleet._user_defined_strategy
                user_defined_strategy.hybrid_configs["sharding_configs"].accumulate_steps = (
                    self.gradient_accumulation_steps
                )

        # NOTE(shenliang03): Check sanity of `accumulate_steps` when using sharding comm overlap.
        if hasattr(fleet.fleet, "_user_defined_strategy"):
            user_defined_strategy = fleet.fleet._user_defined_strategy
            if (
                hasattr(user_defined_strategy, "hybrid_configs")
                and "sharding_configs" in user_defined_strategy.hybrid_configs
            ):
                sd_configs = user_defined_strategy.hybrid_configs["sharding_configs"]
                if sd_configs.comm_overlap:
                    assert self.global_batch_size % self.dataset_world_size == 0, (
                        f"global_batch_size[{self.global_batch_size}] should be divisible by "
                        f"dataset_world_size[{self.dataset_world_size}]"
                    )
                    lbs = self.global_batch_size // self.dataset_world_size
                    assert lbs % self.per_device_train_batch_size == 0, (
                        f"local_batch_size[{lbs}] should be divisible by "
                        f"per_device_train_batch_size[{self.per_device_train_batch_size}]"
                    )
                    assert lbs // self.per_device_train_batch_size == sd_configs.accumulate_steps, (
                        f"local_batch_size[{lbs}] should be equal to "
                        f"accumulate_steps[{sd_configs.accumulate_steps}] * "
                        f"per_device_train_batch_size[{self.per_device_train_batch_size}]"
                    )
        if self.vision_model_name_or_path is not None:
            self.multimodal = True
        if self.visual_ld and not self.vit_lr_ratio:
            self.vit_lr_ratio = self.visual_ld

        if ShardingOption.SHARD_GRAD_OP in self.sharding:
            logger.info("disabling `sp_callback` b/c using sharding stage2")
            self.use_sp_callback = False


class PretrainingTrainer(Trainer):
    """
    PretrainingTrainer 类继承自 Trainer 类，用于训练模型。
    """

    def __init__(self, _shit=None, args=None, model=None, callbacks=None, **kwargs):
        """
        初始化方法。

        Args:
            _shit (None, optional): 占位参数，用于确保其他参数使用关键字传递。默认为 None。
            args (None, optional): 初始化参数，包含训练过程中的各种配置信息。默认为 None。
            model (None, optional): 训练模型。默认为 None。
            callbacks (list, optional): 回调函数列表，用于在训练过程中执行特定操作。默认为空列表。
            **kwargs: 其他关键字参数，用于传递额外的初始化参数。

        Returns:
            无返回值。

        Raises:
            AssertionError: 如果 _shit 参数不为 None，则抛出异常。

        """
        assert _shit is None, "use key-ward argument"
        if callbacks is None:
            callbacks = []
        callbacks = [
            LoggingCallback(),  # ``LoggingCallback` 需要在`TensorBoardCallback` 前面, timmer会reset
            StopperCallback(),
            TensorBoardCallback(args, model=model, log_tokens_per_step=True, log_flops_per_step=False),
            RefinedRecomputeCheckCallback(),
            GCCallback(),
        ] + callbacks
        if args.reshard_save_then_exit:
            callbacks.append(ReshardSaveExitCallback(self))

        if args.adaptive_norm_clip:
            callbacks.append(
                ClipGradByAdaptiveNormCallback(),
            )
        # if args.use_fp8:
        #     callbacks.append(
        #         TEFP8Callback(),
        #     )
        # TODO: restore param.name in other ways
        args.use_async_save = args.use_async_save and args.save_sharded_model and args.load_sharded_model
        super().__init__(args=args, model=model, callbacks=callbacks, **kwargs)
        self.pop_callback(PrinterCallback)
        self.pp_data_buffer = []  # pp
        self._tokens_per_sec_per_card_buffer = []
        self._start_save_time = time.time()
        self._end_save_time = time.time()
        self._first_end_save_time = time.time()
        self.resume_global_step = -1
        self.first_skip_step = 5 if self.args.save_steps > 5 else self.args.save_steps / 2
        # self.return_value = paddle.zeros([]) #fake return value
        global_training_logs.enable_skip_zero([r".*aux_loss.*", r".*orthogonal_loss.*", r".*zloss.*"])
        global_training_logs.set_trainer_interval(self, self.args.global_logging_interval)

    def autocast_smart_context_manager(self):
        """
        需要精心处理精度上的黑白名单问题。
        """
        if self.enable_autocast_context_manager:
            black = [
                "reduce_sum",
                "c_softmax_with_cross_entropy",
                "elementwise_div",
                "sin",
                "cos",
            ]
            white = [
                "lookup_table",
                "lookup_table_v2",
                "flash_attn",
                "flash_attn_v1",
                "matmul",
                "matmul_v2",
                "fused_gemm_epilogue",
            ]
            if self.args.bf16 and self.args.fp16_opt_level == "O2":
                black.append("c_embedding")

            ctx_manager = autocast(
                True,
                custom_black_list=black,
                custom_white_list=white,
                level=self.args.fp16_opt_level,
                dtype=self.amp_dtype,
            )
        else:
            ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()
        return ctx_manager

    def _load_optimizer_state(self, checkpoint):
        """重写load_optimizer 方法，兼容moe optimizer merge 功能"""
        # def _load_moe_optimizer_state(checkpoint):
        #     opt_moe_suffix = re.sub(r"moe\d\d", "moe00", self.args.optimizer_name_suffix)
        #     return self._load_optimizer_state_of_one_shard(checkpoint, opt_moe_suffix)

        def _broadcast_moe_optimizer_state(state_dict):
            # boardcast_keys
            base_state_dict = {"master_weights": {}}
            buf = [
                {i: j.shape for i, j in state_dict.items() if i not in ["master_weights", "LR_Scheduler"]},
                {i: j.shape for i, j in state_dict["master_weights"].items()},
                {"LR_Scheduler": state_dict.get("LR_Scheduler", {})},
            ]

            if self.args.use_hybrid_parallel:
                hcg = fleet.get_hybrid_communicate_group()
                src_rank = hcg.get_data_parallel_group_src_rank()
                group = hcg.get_data_parallel_group()
            else:
                src_rank = 0
                group = None

            dist.broadcast_object_list(buf, src=src_rank, group=group)
            # logger.info(f"moe-optimizer-gather-keys{buf}")
            for k, s in buf[0].items():
                v = state_dict.get(k, paddle.zeros(s, "float32")).to(get_env_device())
                v.name = k
                # k = k.replace("_fp32_master_0", "")  # TODO 这一手replace待品
                dist.broadcast(v, src=src_rank, group=group)
                logger.info(f"broadcast moe optimizer {k} from {src_rank}")
                base_state_dict[k] = v.cpu()
            for k, s in buf[1].items():
                v = state_dict["master_weights"].get(k, paddle.zeros(s, "float32")).to(get_env_device())
                v.name = k
                dist.broadcast(v, src=src_rank, group=group)
                logger.info(f"broadcast moe optimizer-master_weights {k} from {src_rank}")
                base_state_dict["master_weights"][k] = v.cpu()
            base_state_dict.update(buf[2])
            return base_state_dict

        state_dict = super()._load_optimizer_state(checkpoint)

        if self.args.use_moe:
            base_state_dict = _broadcast_moe_optimizer_state(state_dict)
            # base_state_dict_1 = _load_moe_optimizer_state(checkpoint)
            if self.args.data_parallel_rank > 0:
                master_weight = state_dict.pop("master_weights", {})
                base_state_dict.update(state_dict)
                if master_weight:
                    if "master_weights" in base_state_dict:
                        base_state_dict["master_weights"].update(master_weight)
                    else:
                        base_state_dict["master_weights"] = master_weight
                state_dict = base_state_dict
                del base_state_dict
                # return base_state_dict
        return state_dict

    def _save_moe_weights(self, output_dir):
        """重写save_mow_weights 方法 进行参数分离存储"""
        optimizer_name = _add_variant(OPTIMIZER_NAME, self.args.optimizer_name_suffix)
        saved_signal_path = os.path.join(output_dir, f"saved_signal_{dist.get_rank()}")

        os.makedirs(output_dir, exist_ok=True)
        state_dict = self.model.state_dict()
        optimzier_state_dict = self.optimizer.state_dict()

        filtered_state_dict = OrderedDict()
        filter_optimzier_state_dict = OrderedDict()

        param_names_in_master_weights = list(optimzier_state_dict["master_weights"].keys()) if self.args.bf16 else []
        filter_optimzier_state_dict["master_weights"] = OrderedDict()

        for k, v in state_dict.items():
            if getattr(v, "no_sync", False):

                if v.name in param_names_in_master_weights:
                    filter_optimzier_state_dict["master_weights"][v.name] = optimzier_state_dict["master_weights"][
                        v.name
                    ]
                if not (
                    getattr(self.args, "should_save_sharding_stage1_model", False)
                    or getattr(self.args, "save_sharding_stage1_model", False)
                ):
                    filtered_state_dict[k] = v
                for op_k, op_v in optimzier_state_dict.items():
                    if op_k.startswith(v.name):
                        filter_optimzier_state_dict[op_k] = op_v

        if getattr(self.args, "should_save_sharding_stage1_model", False) or getattr(
            self.args, "save_sharding_stage1_model", False
        ):
            self._save(output_dir=output_dir)
        else:
            # 0 号sharding 保存模型
            if self.args.sharding_parallel_rank == 0:
                paddle.save(
                    filtered_state_dict,
                    os.path.join(output_dir, _add_variant(PADDLE_WEIGHTS_NAME, self.args.weight_name_suffix)),
                )
        paddle.save(filter_optimzier_state_dict, os.path.join(output_dir, optimizer_name))
        with open(saved_signal_path, mode="w+") as f:
            f.write("1")

    def _wrap_model(self, model, training=True):

        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        # Note: in paddle.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        # Mixed precision training
        if self.args.fp16 or self.args.bf16:
            model = paddle.amp.decorate(models=model, level=self.args.fp16_opt_level, dtype=self.amp_dtype)

        # Multi-gpu training
        if self.args.use_moe:
            # moe hack

            from src.trainers.data_parallel import DataParallel as MoEDDP

            logger.info("using moe ddp, hack Paddle")  # TODO move this into paddle
            paddle.DataParallel = MoEDDP

        if self.args.world_size > 1 and not self.args.use_hybrid_parallel:
            model = paddle.DataParallel(model)
            # Distributed training (should be after fp16 initialization)

        in_pipeline_parallel_mode = self.args.pipeline_parallel_degree > 1
        in_sharding_parallel_mode = self.sharding is not None
        in_tensor_parallel_model = self.args.tensor_parallel_degree > 1

        def enable_sequence_parallel(_model):
            if self.args.tensor_parallel_degree > 1 and self.args.sequence_parallel:
                if self.args.use_sp_callback:
                    self.add_callback(SPGradSyncCallback(_model._layers))
                else:
                    register_sequence_parallel_allreduce_hooks(_model)

        # Pipeline mode
        if in_pipeline_parallel_mode:
            if self.args.amp_master_grad:
                #  在开启overlap + TF32 的情况下需要 在TF32下开启 mater-weight，
                # 此时 `MixPrecisionLayer` 需要传送一个'float16' 才不会挂
                # `MixPrecisionLayer` 的返回值没有用。return value has no use
                # TODO: paddle在 TF32 下也应该允许使用main-grad
                mix_precision_utils.MixPrecisionLayer(
                    model, dtype=self.amp_dtype if hasattr(self, "amp_dtype") else "float16"
                )
            # hack for pipeline model mini batch to batch
            # need batter solution @ZHUI
            # make batch_fn compatible for fleet.distributed_model decorate.
            prepare_pipeline_inputs_func = (
                model._prepare_pipeline_inputs_func if hasattr(model, "_prepare_pipeline_inputs_func") else None
            )
            model = fleet.distributed_model(model)
            if prepare_pipeline_inputs_func is not None:
                model._prepare_pipeline_inputs_func = prepare_pipeline_inputs_func
            else:

                def _prepare_pipeline_inputs_func(inputs):
                    first_stage_keys = ["input_ids", "attention_mask", "position_ids"]
                    last_stage_keys = ["labels"]

                    def get_expected_keys(inputs, keys):
                        ret = tuple([inputs.pop(k) for k in keys if k in inputs])
                        if len(ret) == 1:
                            ret = ret[0]
                        return ret

                    if type(inputs) is dict:
                        return [
                            get_expected_keys(inputs, first_stage_keys),
                            get_expected_keys(inputs, last_stage_keys),
                        ]

                    keys = list(inputs[0].keys())
                    inputs_batch = {key: [data.pop(key) for data in inputs] for key in keys}
                    return [
                        get_expected_keys(inputs_batch, first_stage_keys),
                        get_expected_keys(inputs_batch, last_stage_keys),
                    ]

                logger.warning(
                    "Using default prepare pipeline inputs func, only support input_ids and labels as inputs."
                )
                model._prepare_pipeline_inputs_func = _prepare_pipeline_inputs_func

            enable_sequence_parallel(model)

            assert self.optimizer is not None, "Pipeline mode need decorate optimizer, pelease init optimizer."
            if self.args.amp_master_grad:
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
            self.optimizer = distributed_optimizer_maybe_hack(self.optimizer, self.args.use_moe)

        # No pipeline mode, sharding only
        if not in_pipeline_parallel_mode and in_sharding_parallel_mode:
            # Sharded DDP!
            if self.args.tensor_parallel_degree > 1:
                hcg = fleet.get_hybrid_communicate_group()
                assert (
                    ShardingOption.SHARD_GRAD_OP in self.args.sharding or ShardingOption.SHARD_OP in self.args.sharding
                ), "Only support tensor parallel + sharding stage1/stage2 hybrid parallel now."
                model = paddle.distributed.fleet.meta_parallel.TensorParallel(model, hcg, strategy=None)
                model.accumulate_steps = self.args.gradient_accumulation_steps
                enable_sequence_parallel(model)

            if ShardingOption.SHARD_OP in self.args.sharding:
                if self.args.amp_master_grad:
                    mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)  # return value has no use
                model = fleet.distributed_model(model)
                if self.args.amp_master_grad:
                    self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
                self.optimizer = distributed_optimizer_maybe_hack(self.optimizer, self.args.use_moe)

            else:
                # sync params (broadcast) buffers in dp group
                if (
                    not is_dp_group_support_in_group_sharded_parallel() or self.args.use_moe
                ) and self.args.data_parallel_degree > 1:
                    try:
                        from paddle.fluid.dygraph.parallel import sync_params_buffers
                    except ImportError:
                        # fix for new api in paddlepaddle v2.5
                        from paddle.distributed.parallel import sync_params_buffers

                    hcg = fleet.get_hybrid_communicate_group()
                    dp_group = hcg.get_data_parallel_group()
                    sync_params_buffers(model, comm_group=dp_group, src_rank=dp_group.ranks[0])

                cpu_offload = ShardingOption.OFFLOAD in self.args.sharding
                assert self.optimizer is not None, "optimizer is empty!"
                level = None
                if ShardingOption.SHARD_GRAD_OP in self.args.sharding:
                    level = "os_g"
                if ShardingOption.FULL_SHARD in self.args.sharding:
                    level = "p_g_os"

                from paddle.distributed.sharding import group_sharded_parallel

                # add dp_group and exclude_layer params
                # https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/
                # distributed/sharding/group_sharded_parallel_cn.html#group-sharded-parallel
                extra_kwargs = {}
                if is_dp_group_support_in_group_sharded_parallel() and not self.args.use_moe:
                    extra_kwargs["dp_group"] = self.dp_group
                    extra_kwargs["exclude_layer"] = ["GroupNorm"]

                model, optimizer, _ = group_sharded_parallel(
                    model,
                    self.optimizer,
                    level=level,
                    scaler=None,
                    group=self.sharding_group,
                    offload=cpu_offload,
                    **extra_kwargs,
                )
                self.optimizer = optimizer

        # pure tesnor parallel mode, no pipeline_parallel, no sharding.
        if not in_pipeline_parallel_mode and not in_sharding_parallel_mode and in_tensor_parallel_model:
            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)  # return value has no use

            model = fleet.distributed_model(model)
            model.accumulate_steps = self.args.gradient_accumulation_steps
            enable_sequence_parallel(model)
            assert self.optimizer is not None, "Tensor parallel mode need decorate optimizer, pelease init optimizer."
            if self.args.amp_master_grad:
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)

            self.optimizer = distributed_optimizer_maybe_hack(self.optimizer, self.args.use_moe)

        # Add optimizer callback
        if self.args.record_optimizer_stat:
            # Insert at the beginning of callbacks.
            self.callback_handler.callbacks.insert(0, OptimizerCallback(self.args, self.optimizer))
        if self.args.use_moe:
            self.callback_handler.callbacks.insert(0, MoeLoggingCallback(self.optimizer))

        try:
            from paddle.fluid.dygraph.parallel import sync_params_buffers
        except ImportError:
            # fix for new api in paddlepaddle v2.5
            from paddle.distributed.parallel import sync_params_buffers

        # hcg = fleet.get_hybrid_communicate_group()
        # dp_group = hcg.get_data_parallel_group()
        # sync_params_buffers(model, comm_group=dp_group, src_rank=dp_group.ranks[0])
        if (
            isinstance(self.optimizer, HybridParallelOptimizer)
            and self.args.log_global_grad_norm
            and self.args.max_grad_norm > 0
        ):
            logger.info("hacking `grad-clip`")
            gradclip = self.optimizer._inner_opt._grad_clip
            oldcomm = gradclip._comm_and_clip
            oldclip = gradclip._dygraph_clip
            hcg = fleet.get_hybrid_communicate_group()
            num_pp = hcg.get_pipe_parallel_world_size()

            @paddle.no_grad()
            def newcomm(self, params_grads, global_norm_var_dist, global_norm_var_not_dist, *args):
                if num_pp > 1:
                    for p, g in params_grads:
                        if not getattr(p, "pp_distributed", True):
                            g.scale_(np.sqrt(num_pp))
                ret = oldcomm(
                    params_grads, global_norm_var_dist, global_norm_var_not_dist, *args
                )  # `_comm_and_clip` 会做 inplace nccl.allreduce，在其返回之后算grad-norm可以获得全局值
                global_norm_var_fp32 = paddle.sqrt(global_norm_var_dist + global_norm_var_not_dist)
                # if global_training_logs_enabled():
                #    global_training_logs.update(global_grad_norm=global_norm_var_fp32.item())
                return ret

            @paddle.no_grad()
            def new_dygraph_clip(self, params_grads):
                """
                `_dygraph_clip` calls `_comm_and_clip`
                """
                # non_dist_pp = [g for p, g in params_grads if getattr(p, "pp_distributed", True)]
                # logger.info(f'vit-grad-norm-this-pp:{sum([g.norm()for g in non_dist_pp])}')
                if num_pp > 1:
                    for p, g in params_grads:
                        if not getattr(p, "pp_distributed", True):
                            g.scale_(1 / np.sqrt(num_pp))
                ret = oldclip(params_grads)
                # logger.info(f'[after-clip] vit-grad-norm-this-pp:{sum([g.norm()for g in non_dist_pp])}')
                return ret

            self.optimizer._inner_opt._grad_clip._comm_and_clip = MethodType(
                newcomm, self.optimizer._inner_opt._grad_clip
            )
            self.optimizer._inner_opt._grad_clip._dygraph_clip = MethodType(
                new_dygraph_clip, self.optimizer._inner_opt._grad_clip
            )

        return model

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """doc"""
        self.model_wrapped.accumulate_steps = self.args.gradient_accumulation_steps
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        start_time = time.time()
        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        eval_loop = self.evaluation_loop

        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if compute_metrics is None else None,
            ignore_keys=ignore_keys,
            # Only evaluate max_eval_iters
            max_eval_iters=self.args.eval_iters,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        return output.metrics

    def prediction_pipeline_step(self, model, inputs, prediction_loss_only, ignore_keys):
        """doc"""
        loss, _, labels = super().prediction_pipeline_step(
            model, inputs, prediction_loss_only, ignore_keys
        )  # ERNIE-PP模型的loss其实是loss-sum。
        num_tokens = (labels != self.tokenizer.ignored_index).sum().item()
        loss_avg = loss * self.model_wrapped.accumulate_steps / num_tokens
        return loss_avg, loss, labels

    def restore_dataloader_status(self):
        """
        restore dataloader status
        """
        # same_data is set to None and modifed here by default, but can be set to True/False explicitly
        if self.args.same_data is None or self.args.same_data == "":
            if self.args.resume_from_checkpoint is not None:
                train_bin_file = os.path.join(self.args.resume_from_checkpoint, TRAINING_ARGS_NAME)
                assert os.path.exists(train_bin_file), f"{train_bin_file} not found."
                train_bin = paddle.load(train_bin_file)
                old_data_filelist = train_bin.data_filelist
                old_data_weights = train_bin.data_weights
                old_sharding_degree = train_bin.sharding_parallel_degree
                old_data_parallel_degree = train_bin.data_parallel_degree
                old_modality_interleave = getattr(train_bin, "modality_interleave", None)
                old_modality_ratio = getattr(train_bin, "modality_ratio", None)
                old_reeao_data_world_size = getattr(train_bin, "reeao_data_world_size", None)
                new_data_filelist = self.args.data_filelist
                new_data_weights = self.args.data_weights
                new_sharding_degree = self.args.sharding_parallel_degree
                new_data_parallel_degree = self.args.data_parallel_degree
                self.args.same_data = (
                    (old_data_filelist == new_data_filelist)
                    and (old_data_weights == new_data_weights)
                    and (old_sharding_degree == new_sharding_degree)
                    and (old_data_parallel_degree == new_data_parallel_degree)
                    and (
                        not self.args.multimodal
                        or (
                            old_modality_interleave == self.args.modality_interleave
                            and old_modality_ratio == self.args.modality_ratio
                        )
                    )
                    and (
                        old_reeao_data_world_size is None
                        or old_reeao_data_world_size == self.args.reeao_data_world_size
                    )
                )
                logger.info(f"Automatically setting same_data value: {self.args.same_data}")
            else:
                self.args.same_data = False
                logger.info(f"Training from scratch, setting same_data value: {self.args.same_data}")
        else:
            logger.info(f"User has defined same_data value: {self.args.same_data}")

        if self.args.same_data:
            logger.warning(
                "same_data has been set to True. \
                            Carefully check whether the data, population proportion, "
                "and DP count are completely consistent with those before."
            )
        else:
            logger.warning(
                "same_data has been set to False. \
                            which will regenerate the global shuffle domain."
            )

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs):
        flag_log = self.control.should_log
        if self.control.should_log:
            logs = {}
            # all_gather + mean() to get average loss over all processes
            tr_loss_single_dp_scalar = tr_loss.item()
            dist.all_reduce(
                tr_loss, dist.ReduceOp.SUM
            )  # 3级并行时，每个pp下的loss会广播，全局reduce-mean的时候，分子分母都会乘以pp_world_size，结果会被约掉
            tr_loss_scalar = tr_loss.item() / dist.get_world_size()
            tr_loss.zero_()

            # reset tr_loss to zero
            logs["loss"] = tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged)
            logs["loss_cur_dp"] = tr_loss_single_dp_scalar / (self.state.global_step - self._globalstep_last_logged)
            logs["learning_rate"] = float(self._get_learning_rate())
            logs["global_step"] = int(self.state.global_step)

            divisor = 2**30

            current_device = framework._current_expected_place_()
            device_id = current_device.get_device_id()
            current_memory_allocated = core.device_memory_stat_current_value("Allocated", device_id)
            current_memory_reserved = core.device_memory_stat_current_value("Reserved", device_id)
            max_memory_allocated = core.device_memory_stat_peak_value("Allocated", device_id)
            max_memory_reserved = core.device_memory_stat_peak_value("Reserved", device_id)
            logs["mem_allocated_gb"] = current_memory_allocated / divisor
            logs["max_mem_allocated_gb"] = max_memory_allocated / divisor
            logs["mem_reserved_gb"] = current_memory_reserved / divisor
            logs["max_mem_reserved_gb"] = max_memory_reserved / divisor

            # if not self.args.enable_global_training_logs:
            #     # 开启任何全局同步都会导致 9000卡降速 5%
            #     global_training_logs.global_meters_keys = []  # ["data_not_valid", "experts_per_token"]

            if get_env_device() == "gpu":
                info_callback = global_training_logs.dict(use_async=True)

            if hasattr(self, "scaler"):
                logs["loss_scale"] = float(f"{self.scaler._scale.item():.3e}")

            total_train_batch_size = (
                self.args.train_batch_size * self.args.gradient_accumulation_steps * self.args.reeao_dataset_world_size
            )
            num_steps = self.state.global_step - self._globalstep_last_logged
            logs.update(
                speed_metrics(
                    "global",
                    self._globalstep_last_start_time,
                    num_samples=total_train_batch_size * num_steps,
                    num_steps=num_steps,
                )
            )
            if not hasattr(self, "model_numel"):
                model_numel = sum(
                    p.numel().item()
                    for n, p in model.named_parameters()
                    if not p.stop_gradient and "embeddings" not in n and "embed_tokens" not in n
                )
                numel_tensor = paddle.to_tensor(model_numel)
                dist.all_reduce(numel_tensor)
                self.model_numel = numel_tensor.item() // self.args.dataset_world_size

            tokens_per_steps = self.args.max_seq_length * total_train_batch_size
            logs["tokens_trained_current_step"] = tokens_per_steps
            # get time in microseconds for efficiency screen
            logs["timestamp"] = int(time.time() * 1000)
            logs["TFLOPS_per_sec_per_card"] = round(
                6
                * tokens_per_steps
                * self.model_numel
                * logs["global_steps_per_second"]
                / 1e12
                / self.args.world_size,
                3,
            )
            logs["tokens_per_sec_per_card"] = round(
                tokens_per_steps * logs["global_steps_per_second"] / self.args.world_size, 1
            )
            self._tokens_per_sec_per_card_buffer.append(logs["tokens_per_sec_per_card"])
            logs["tokens_per_sec_per_card_average"] = round(np.mean(self._tokens_per_sec_per_card_buffer), 1)
            if self.resume_global_step == -1:
                self.resume_global_step = self.state.global_step - 1
            if self.state.global_step <= self.resume_global_step + self.first_skip_step:
                self._tokens_per_sec_per_card_buffer = []
                self._end_save_time = time.time()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self._globalstep_last_start_time = time.time()

            # if get_env_device() == "xpu":
            #     info, gathered_info = global_training_logs.dict(use_async=False)
            # else:
            #     info, gathered_info = info_callback()
            info, gathered_info = info_callback()
            global_training_logs.reset()
            logs.update({f"{k}_cur_dp": v for k, v in info.items()})
            logs.update(gathered_info)
            # if self.args.enable_global_training_logs:
            #     info_list = []
            #     dist.all_gather_object(info_list, info)
            #     logs.update(
            #         {
            #             k: np.mean([v[k] for v in info_list if k in v])
            #             for k in {key for item in info_list for key in item.keys()}
            #         }
            #     )

            self.log(logs, **kwargs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)

        if self.control.should_save:
            if hasattr(self.args, "flash_save_steps") and self.args.flash_save_steps > 0:
                is_persistent_ckpt = 1 if self.state.global_step % self.args.save_steps == 0 else 0
            else:
                is_persistent_ckpt = 1

            if is_persistent_ckpt:
                self._start_save_time = time.time()
            else:
                zcc_start_save_time = time.time()
            self._save_checkpoint(model, metrics=metrics)
            paddle.distributed.barrier()
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            if flag_log:
                logs = {"is_persistent_ckpt": is_persistent_ckpt}
                tbk = self._start_save_time - self._end_save_time
                if (self.state.global_step == self.resume_global_step + self.args.save_steps) or (
                    hasattr(self.args, "flash_save_steps")
                    and (self.state.global_step == self.resume_global_step + self.args.flash_save_steps)
                ):
                    actual_tbk = self._start_save_time - self._first_end_save_time
                    actual_avg_speed_step = self.args.save_steps * tokens_per_steps / actual_tbk / self.args.world_size
                    tbk = tbk / (self.args.save_steps - self.first_skip_step) * self.args.save_steps
                if is_persistent_ckpt:
                    ts = time.time() - self._start_save_time
                else:
                    ts = time.time() - zcc_start_save_time
                logs["save_ckpt_time_sec"] = ts
                logs["global_save_step"] = self.state.global_step
                if is_persistent_ckpt:
                    tokens_per_steps = self.args.max_seq_length * total_train_batch_size
                    avg_speed_step = self.args.save_steps * tokens_per_steps / tbk / self.args.world_size
                    logs["train_time_sec_without_save"] = tbk
                    logs["average_tokens_per_sec_per_card_without_save"] = round(avg_speed_step, 1)
                    logs["average_tokens_per_sec_per_card_with_save"] = round(
                        self.args.save_steps * tokens_per_steps / (tbk + ts) / self.args.world_size, 2
                    )
                    if self.state.global_step == self.resume_global_step + self.args.save_steps:
                        logs["actual_average_tokens_per_sec_per_card_without_save"] = round(actual_avg_speed_step, 1)
                        logs["actual_average_tokens_per_sec_per_card_with_save"] = round(
                            self.args.save_steps * tokens_per_steps / (actual_tbk + ts) / self.args.world_size, 2
                        )
                    logs["one_day_billion_tokens_without_save"] = round(
                        0.0000864 * self.args.save_steps * tokens_per_steps / tbk, 2
                    )
                    logs["one_day_billion_tokens_with_save"] = round(
                        0.0000864 * self.args.save_steps * tokens_per_steps / (tbk + ts), 2
                    )
                self.log(logs, **kwargs)
                if is_persistent_ckpt:
                    self._globalstep_last_start_time = time.time()
                    self._tokens_per_sec_per_card_buffer = []
            if is_persistent_ckpt:
                self._end_save_time = time.time()

    def create_scheduler(self, num_training_steps):
        """
        创建一个学习率调度器。

        Args:
            num_training_steps (int): 训练总步数。

        Returns:
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器。

        """
        if self.args.warmup_steps > 0:
            warmup = self.args.warmup_steps
        else:
            warmup = int(self.args.warmup_ratio * num_training_steps)
        if self.args.lr_scheduler.startswith("wsd"):
            scheduler = self.args.lr_scheduler.split(":")
            if len(scheduler) == 2:
                num_steady_steps = int(scheduler[1])
            else:
                num_steady_steps = None
            logger.info(f"using wsd lr scheduler, num_steady_steps={num_steady_steps}")
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                self.args.learning_rate,
                warmup,
                self.args.max_steps,
                decay_function=self.args.decay_function,
                min_lr=self.args.min_lr if self.args.min_lr else 0.0,
                num_steady_steps=num_steady_steps,
            )
        else:
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.args.learning_rate,
                warmup,
                self.args.max_steps,
                min_lr=self.args.min_lr if self.args.min_lr else 0.0,
            )
        return self.lr_scheduler

    def create_optimizer(self, lr_scheduler=None):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        optimizer_params = (
            [p for n, p in self.model.named_parameters() if "embeddings" in n]
            if self.args.train_emb_only
            else [p for n, p in self.model.named_parameters() if p.stop_gradient is False]
        )
        if self.args.train_emb_only:
            logger.info(f"using `train-emb-only`, #embedding params={len(optimizer_params)}")
        elif self.args.train_moe_only:
            optimizer_params = (
                [p for n, p in self.model.named_parameters() if "mlp.experts" in n or "mlp.gate" in n]
                if self.args.train_moe_only
                else [p for n, p in self.model.named_parameters() if p.stop_gradient is False]
            )
            logger.info(f"using `train_moe-only`, #moe params={len(optimizer_params)}")
        elif len(optimizer_params) < len(self.model.parameters()):
            logger.info(
                f"some params are not optimized, #totally={len(self.model.parameters())}, \
                  #optimized={len(optimizer_params)}"
            )
        if self.optimizer is None:
            decay_parameters = [
                p.name for n, p in self.model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])
            ]

            def apply_decay_param_fun(x):
                return x in decay_parameters

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            grad_clip = nn.ClipGradByGlobalNorm(self.args.max_grad_norm) if self.args.max_grad_norm > 0 else None

            self.static_name_to_dyg_name = {p.name: n for n, p in self.model.state_dict().items()}
            gate_pattern = re.compile(r"ernie\.layers\.0\.mlp\.gate\.weight")  # TODO 换成更加通配的方法
            vit_pattern = re.compile(r"vision_model\.(cls_token|pos_embed|patch_embed|blocks)")
            vit_blocks_pattern = re.compile(r"vision_model\.blocks\.(\d+)\.")

            def lr_ratio_fn(param):
                name = self.static_name_to_dyg_name[param.name]
                # logger.info(f'search {param.name} -> {name}')
                if self.args.moe_gate_lr_ratio is not None and gate_pattern.match(name):
                    logger.info(f"apply moe_gate_lr_ratio to {name}, ratio={self.args.moe_gate_lr_ratio}")
                    return float(self.args.moe_gate_lr_ratio)
                elif self.args.vit_lr_ratio is not None and vit_pattern.match(name):
                    if hasattr(self.model.config.vision_config, "layers"):
                        n_layers = self.model.config.vision_config.layers
                    else:
                        n_layers = self.model.config.vision_config.depth
                    if vit_blocks_pattern.match(name):
                        layer_id = int(vit_blocks_pattern.match(name).group(1))
                    else:
                        layer_id = 0
                    lr_ratio = self.args.vit_lr_ratio ** (n_layers - 1 - layer_id)
                    logger.info(f"apply vit lr_ratio to {name}, ratio={lr_ratio}")
                    return float(lr_ratio)
                return 1.0

            self.optimizer = optimizer_cls(
                learning_rate=self.lr_scheduler if lr_scheduler is None else lr_scheduler,
                apply_decay_param_fun=apply_decay_param_fun,
                parameters=optimizer_params,
                weight_decay=self.args.weight_decay,
                grad_clip=grad_clip,
                multi_precision=True,
                lr_ratio=(
                    lr_ratio_fn
                    if (self.args.moe_gate_lr_ratio is not None or self.args.vit_lr_ratio is not None)
                    else None
                ),
                **optimizer_kwargs,
            )

        return self.optimizer

    def save_model(self, output_dir=None):
        """
        保存模型及相关的配置文件到指定的目录。

        Args:
            output_dir (str, optional): 保存模型的目录。默认为None。

        Returns:
            None

        Raises:
            None

        """
        super().save_model(output_dir)
        if self.args.should_save:
            with open(os.path.join(output_dir, "static_name_to_dyg_name.json"), "w") as of:
                of.write(json.dumps(self.static_name_to_dyg_name))

    def _load_rng_state(self, checkpoint):
        # 预训练环节并不需要由框架控制 `rng`
        pass
