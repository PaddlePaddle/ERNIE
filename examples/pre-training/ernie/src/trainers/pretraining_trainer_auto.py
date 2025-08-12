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

"""AutoPretrainingTrainer"""

__all__ = [
    "AutoPretrainingTrainer",
]


import sys
import re
import os
import json
import pickle
import contextlib
from typing import Optional, List
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
import random
import time
import math
import logging
from functools import partial

import numpy as np

import paddle
import paddle.nn as nn
from paddle import framework
from paddle.base import core
from paddle.io import DataLoader
import paddle.amp.auto_cast as autocast
from paddle.distributed.communication.group import _get_global_group

from paddleformers.utils.tools import get_env_device
from paddleformers.trainer import (
    speed_metrics,
)

from paddleformers.trainer.auto_trainer import AutoTrainer

try:
    from paddleformers.utils.env import (
        PREFIX_CHECKPOINT_DIR,
        SCHEDULER_NAME,
        TRAINER_STATE_NAME,
        PADDLE_OPTIMIZER_NAME,
    )
except ImportError:
    from paddleformers.trainer.trainer import (
        OPTIMIZER_NAME,
    )

    PADDLE_OPTIMIZER_NAME = OPTIMIZER_NAME
from paddleformers.utils.batch_sampler import (
    DistributedBatchSampler as PaddleNLPDistributedBatchSampler,
)

try:
    from paddleformers.trainer.trainer import (
        PADDLE_WEIGHT_FILE_NAME as PADDLE_WEIGHTS_NAME,
    )
except ImportError:
    from paddleformers.utils.env import PADDLE_WEIGHTS_NAME
from paddleformers.trainer.utils import add_start_docstrings
from paddleformers.trainer.trainer_callback import PrinterCallback
from paddle.distributed import fleet
import paddle.distributed as dist
from paddleformers.datasets import MapDataset

from paddleformers.trainer.trainer_utils import (
    _exec_mode_guard,
)
from paddleformers.transformers.model_utils import _add_variant

from src.lr_schedulers import get_cosine_schedule_with_warmup
from src.utils.training_utils import (
    reset_per_device_batch_size,
)
from src.callbacks import (
    TensorBoardCallback,
    LoggingCallback,
    StopperCallback,
    ClipGradByAdaptiveNormCallback,
    ReshardSaveExitCallback,
)
from src.datasets import (
    DistDataLoaderAuto,
    ExampleSet,
    ExampleSetSingleDataSource,
)
from src.utils.misc import global_training_logs
from src.clip import ClipGradByAdaptiveNorm, ClipGradForMOEByGlobalNorm
from src.trainers.pretraining_trainer import DummySampler

try:
    from paddleformers.trainer.trainer import (
        is_dp_group_support_in_group_sharded_parallel,
    )
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


logger = logging.getLogger(__name__)

try:
    from paddleformers.trainer import AutoTrainingArguments
except ImportError:
    from paddleformers.trainer import TrainingArguments as AutoTrainingArguments

    logger.warning("paddlenlp.trainer.AutoTrainingArguments CANNOT import!")
    logger.warning("Use TrainingArguments as an alternative but will lose some args!")


def distributed_optimizer_maybe_hack(
    optimizer,
    use_moe,
):
    """重写了 fleet.distributed_optimizer，在 moe 情况下会 hack Optimizer 的 Wrapper"""
    if use_moe:
        from src.trainers.dygraph_optimizer.hybrid_parallel_optimizer import (
            HybridParallelOptimizer as MoEHybridParallelOptimizer,
        )

        # moe下需要定制 `HybridParallelOptimizer` 函数以下重写了 `fleet.distributed_optimizer`
        fleet_env = fleet.fleet
        fleet_env.user_defined_optimizer = optimizer
        # TODO：sharding group 内做 moe
        hp_optim = MoEHybridParallelOptimizer(
            optimizer, fleet_env._hcg, fleet_env._user_defined_strategy
        )

        if fleet_env._user_defined_strategy.hybrid_configs[
            "pp_configs"
        ].dp_comm_overlap:
            hp_optim._dp_enable = False

        if fleet_env._user_defined_strategy.hybrid_configs[
            "pp_configs"
        ].sharding_comm_overlap:
            hp_optim._sharding_enable = False
        return hp_optim
    else:
        return fleet.distributed_optimizer(optimizer)


DATATYPE_2_ID = {"mm": 0, "lm": 1, "audio": 2}

@dataclass
@add_start_docstrings(AutoTrainingArguments.__doc__)
class AutoPreTrainingArguments(AutoTrainingArguments):
    """
    预训练相关参数配置
    """

    vocab_path: str = field(
        default=None, metadata={"help": "eb35 streaming data vocab"}
    )
    task_need_convert: str = field(default=None, metadata={"help": "glm task id"})
    multimodal: bool = field(
        default=False, metadata={"help": "whether training with multimodal"}
    )
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
            "help": "H5文件连续采样。为了保证AFS性能，在读取AFS H5文件的时候需要尽量读取一片ID"
            "，这个参数指定了一次连续读取的`样本`大小"
        },
    )
    train_emb_only: int = field(
        default=0,
        metadata={"help": "是否只训练embedding，通常用于热启换词表"},
    )
    use_train_part_sharding: Optional[int] = field(
        default=1,
        metadata={"help": "根据file进行数据切片，只在预训练时候使用。否则会很慢"},
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
            "help": "标准线上明文数据流",
        },
    )
    dataset: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    data_load_process_num: int = field(
        default=10,
        metadata={
            "help": "是否使用多进程加速原始数据读取,与DataLoader的num_workers意义不同"
        },
    )

    data_dir: str = field(default=None, metadata={"help": "数据路径（指向一个目录）"})

    data_filelist: str = field(
        default=None, metadata={"help": "数据文件列表，与`args.data_dir`互斥"}
    )
    data_weights: str = field(default=None, metadata={"help": "数据配比权重"})

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
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
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
    no_part_shuffle: Optional[int] = field(
        default=0, metadata={"help": "不进行part内数据shuffle"}
    )
    record_optimizer_stat: Optional[bool] = field(
        default=False, metadata={"help": "是否记录优化器momentum信息"}
    )
    skip_optimizer_badcases: Optional[bool] = field(
        default=False, metadata={"help": "是否跳过optimizer badcase很多的step"}
    )
    same_data: Optional[bool] = field(
        default=False,
        metadata={"help": "热启时，数据、配比、DP数是否完全一致, 支持续线"},
    )
    base_seq_length: Optional[int] = field(
        default=4096, metadata={"help": "reeao最小seq_length"}
    )
    shuffle_consecutive: Optional[bool] = field(
        default=False,
        metadata={
            "help": "是否对num_consecutive片段进行shuffle, same_data=True热启时，该值需与上一次保持一致"
        },
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
        default=1.03,
        metadata={"help": "AdaptiveNormClip 裁剪阈值, 大于设定的阈值才会启动裁剪"},
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
    use_async_save: Optional[bool] = field(
        default=False, metadata={"help": "是否开启异步保存功能"}
    )
    pre_alloc_memory: float = field(
        default=0.0,
        metadata={
            "help": "Pre-allocate one specific-capacity empty tensor "
            "and release it for avoiding memory fragmentation"
        },
    )
    enable_global_training_logs: bool = field(
        default=False, metadata={"help": "是否启用global_training_logs"}
    )
    use_dummy_dataset: Optional[bool] = field(
        default=False, metadata={"help": "是否使用DummyDataSet, 仅用于Debug"}
    )
    reshard_save_then_exit: Optional[bool] = field(
        default=False, metadata={"help": "是否在reshard后直接退出程序"}
    )
    moe_group: Optional[str] = field(
        default="dp", metadata={"help": "moe 的通信组，目前支持“dp|sharding|mp|dummy”"}
    )
    use_moe: Optional[bool] = field(
        default=False, metadata={"help": "expert parallel 临时替代"}
    )
    moe_use_all2all: Optional[bool] = field(
        default=False, metadata={"help": "是否使用all2all通信方式"}
    )
    log_global_grad_norm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "打印全局grad-norm, 只有在开启`enable_global_training_logs`时生效"
        },
    )

    lr_scheduler: str = field(
        default="cosine",
        metadata={
            "help": "The scheduler type to use. suppor linear, cosine, constant, constant_with_warmup"
        },
    )
    image_token_len: int = field(
        default=64,
        metadata={"help": "number of images tokens from resampler per image"},
    )
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
    modality_interleave: str = field(default="acc", metadata={"help": "acc"})
    modality_ratio: tuple = field(
        default=None,
        metadata={"help": "ratio of modality tokens to be masked out"},
    )
    bos_retry_max_time: int = field(
        default=0, metadata={"help": "when bos download failed, #retry times"}
    )
    bos_retry_interval: float = field(
        default=1, metadata={"help": "when bos download failed, interval between retry"}
    )

    pipeline_schedule_mode: str = field(
        default="1F1B",
        metadata={"help": "The pipeline schedule mode, support 1F1B and VPP"},
    )
    virtual_pipeline_seg_method: str = field(
        default="ErnieDecoderLayerAuto",
        metadata={"help": "The seg method of spliting pp layer for virtual pipeline."},
    )
    pp_need_data_degree: int = field(
        default=0,
        metadata={
            "help": "pipline 并行中的机器也需要 fetch 数据，提升吞吐，搭配 `ErniemmMoEForCausalPipe` 使用"
        },
    )
    pp_need_data: bool = field(default=False, metadata={"help": "向前兼容"})
    custom_data_status: str = field(
        default=None,
        metadata={"help": "load data status from custom trainer_state.json"},
    )
    model_type: Optional[str] = field(
        default="ernie",
        metadata={"help": "Only support for ernie pre-training for now."},
    )
    n_microbatches: int = field(
        default=1,
        metadata={"help": "Control the num of microbatches in one pp step."},
    )

    @property
    def use_moe(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return getattr(self, "use_expert_parallel", self._use_moe)

    @use_moe.setter
    def use_moe(self, value):
        """_summary_

        Args:
            value (_type_): _description_
        """
        self.use_expert_parallel = value
        self._use_moe = value

    @property
    def need_data(self):
        """
        判断当前进程是否需要加载数据。

        Args:
            无。

        Returns:
            bool: 如果当前进程为mp0和pp0状态（即主进程），则返回True，表示需要加载数据；
                  否则返回False，表示不需要加载数据。

        """
        # mp0、pp0状态 卡才需要load数据
        if self.pp_need_data_degree:
            assert self.pipeline_parallel_degree > 1
            assert (
                self.pp_need_data_degree >= 2
                and self.pp_need_data_degree <= self.pipeline_parallel_degree
            ), (
                self.pp_need_data_degree,
                self.pipeline_parallel_degree,
            )
            # shift by 1 to avoid last pp no nee data
            no_need_data_range = list(
                range(self.pp_need_data_degree - 1, self.pipeline_parallel_degree - 1)
            )
            return self.tensor_parallel_rank == 0 and (
                self.pipeline_parallel_rank not in no_need_data_range
            )
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
        no_need_data_range = list(
            range(self.pp_need_data_degree - 1, self.pipeline_parallel_degree - 1)
        )
        ranks = [
            i
            for i in range(self.pipeline_parallel_degree)
            if i not in no_need_data_range
        ]
        if self.pipeline_parallel_rank not in ranks:
            return None
        reeao_pp_rank = ranks.index(self.pipeline_parallel_rank)
        pp_need_data_rank = self.pipeline_parallel_rank

        assert not (self.sharding_parallel_degree > 1 and self.data_parallel_rank > 1)
        return (
            max(self.pp_need_data_degree, 1) * self.sharding_parallel_rank
            + reeao_pp_rank
        )

    @property
    def reeao_dataset_world_size(self):
        """
        考虑 pp /sharding/ dp 总和的数据流 worldsize
        """
        if not self.pp_need_data:
            return super().dataset_world_size
        return (
            max(self.sharding_parallel_degree, 1)
            * max(self.data_parallel_degree, 1)
            * max(self.pipeline_parallel_degree, 1)
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

        assert (
            self.global_batch_size
            == self.per_device_train_batch_size
            * self.gradient_accumulation_steps
            * max(self.sharding_parallel_degree, 1)
            * max(self.data_parallel_degree, 1)
        ), (
            f"`gbs` should be equal to `lbs * acc * (dp_degree or sd_degree)`, "
            f"but got gbs={self.global_batch_size}, "
            f"lbs={self.per_device_train_batch_size}, "
            f"acc={self.gradient_accumulation_steps}, "
            f"dp_degree={max(self.data_parallel_degree, 1)}, "
            f"sd_degree={max(self.sharding_parallel_degree, 1)}"
        )

        if self.global_batch_size > 0:
            micro_bsz, acc_steps = reset_per_device_batch_size(
                self.global_batch_size,
                self.per_device_train_batch_size,
                self.dataset_world_size,
            )
            logger.info(
                f"global_batch={self.global_batch_size} micro-bsz:{micro_bsz}, accumulate_steps:{acc_steps}"
            )
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
            self.max_gradient_accumulation_steps = (
                self.gradient_accumulation_steps
            )  # hack add new
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
            self.max_gradient_accumulation_steps = (
                self.gradient_accumulation_steps
            )  # hack add new

        if self.pipeline_parallel_degree > 1:
            self.per_device_eval_batch_size = (
                self.per_device_train_batch_size * self.gradient_accumulation_steps
            )  # hack Eval for PP!
            logger.warn(
                f"eval_batch_size set to {self.per_device_eval_batch_size} in Pipeline Parallel!"
            )
            user_defined_strategy = fleet.fleet._user_defined_strategy
            user_defined_strategy.strategy.pipeline_configs.accumulate_steps = (
                self.gradient_accumulation_steps
            )
            if self.pp_need_data and not self.pp_need_data_degree:
                self.pp_need_data_degree = self.pipeline_parallel_degree
            if self.pp_need_data_degree:
                assert (
                    self.gradient_accumulation_steps % self.pp_need_data_degree == 0
                ), (
                    f"gradient_accumulation_steps[{self.gradient_accumulation_steps}] should be divisible by "
                    f"pp_need_data_degree[{self.pp_need_data_degree}]"
                )
                # pp_need_data_degree下，args的acc 需要//pp数量，欺骗 在prepare_inputs
                self.gradient_accumulation_steps = (
                    self.gradient_accumulation_steps // self.pp_need_data_degree
                )
                logger.info(
                    f"pp-need-data hack args.gradient_accumulation_steps to - {self.gradient_accumulation_steps}"
                )
            self.max_gradient_accumulation_steps = (
                self.gradient_accumulation_steps
            )  # hack add new
            logger.info(f"fixing pp configs: {user_defined_strategy.pipeline_configs}")
        else:
            self.per_device_eval_batch_size = self.per_device_train_batch_size
            logger.warn(f"eval_batch_size set to {self.per_device_eval_batch_size}")

        if self.sharding_parallel_degree > 1:
            sharding_parallel_config = (
                set(self.sharding_parallel_config.split(" "))
                if self.sharding_parallel_config
                else set()
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
                user_defined_strategy.hybrid_configs[
                    "sharding_configs"
                ].accumulate_steps = self.gradient_accumulation_steps

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
                    assert (
                        lbs // self.per_device_train_batch_size
                        == sd_configs.accumulate_steps
                    ), (
                        f"local_batch_size[{lbs}] should be equal to "
                        f"accumulate_steps[{sd_configs.accumulate_steps}] * "
                        f"per_device_train_batch_size[{self.per_device_train_batch_size}]"
                    )
        if self.vision_model_name_or_path is not None:
            self.multimodal = True


class WeightedDistributedSamplerAuto(PaddleNLPDistributedBatchSampler):
    """
    业务自定义DistributedSampler
    """

    def __init__(
        self,
        dataset,
        batch_size,
        output_dir,
        dp_rank,
        dp_size,
        num_consecutive=1,
        seed=0,
        batch_size_warmup_steps=-1,  # 用来做热启 resumed ckpt热启
        gradient_accumulation_steps=None,  # 用于在 `Progressive-Batching` 情况下做热启动
        max_gradient_accumulation_steps=None,  # 用于在 `Progressive-Batching` 情况下做热启动
        per_device_train_batch_size=None,  # 用于在 `Progressive-Batching` 情况下做热启动
        batch_size_warmup_increment=None,  # 用于在 `Progressive-Batching` 情况下做热启动
        combine_batch: int = 1,
        shuffle_consecutive: bool = False,
        global_shuffle_num_examples: int = 0,
        same_data: bool = False,
        modality_ratio: tuple = None,  # 数据流模态配比 ('text_ratio', ‘mm_ratio’)
        modality_interleave: int = 1,
        **kwargs,
    ):
        self.num_consecutive = num_consecutive
        self.seed = seed
        super().__init__(dataset, batch_size, **kwargs)
        self.weights = None
        self.batch_size = batch_size  # per-device-micro-batchsize
        self.output_dir = output_dir
        self.rng = random.Random(self.seed + self.epoch)
        self.dp_rank = dp_rank
        # 并不根据这个ID做切片！切片逻辑用self.local_rank, self.nranks控制，这个变量只是用来打log。
        # 在按照part切分数据时，`WeightedDistributedSampler`并不切片。所以local_rank/nranksw=0/1
        self.dp_size = dp_size
        self.batch_size_warmup_steps = batch_size_warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_gradient_accumulation_steps = max_gradient_accumulation_steps
        self.per_device_train_batch_size = per_device_train_batch_size
        self.batch_size_warmup_increment = batch_size_warmup_increment
        self.combine_batch = combine_batch
        self.shuffle_consecutive = shuffle_consecutive
        self.global_shuffle_seed = 0
        self.global_shuffle_num_examples = global_shuffle_num_examples
        self.same_data = same_data
        self.load_data_seq = False
        self.modality_ratio = modality_ratio
        self.modality_interleave = modality_interleave
        if self.modality_ratio is not None:
            print("[my debug] modality_ratio:", modality_ratio)
            logger.info(f"modality ratio set to {self.modality_ratio}")
            assert sum(modality_ratio) == 1.0, "modality ratio should sum to 1"
            assert (
                self.modality_interleave * self.modality_ratio[0] % 1 == 0
                if len(self.modality_ratio) >= 1
                else True
            ), "modality_interleave * modality_ratio[0] should be integer"
            assert (
                self.modality_interleave * self.modality_ratio[1] % 1 == 0
                if len(self.modality_ratio) >= 2
                else True
            ), "modality_interleave * modality_ratio[1] should be integer"
            assert (
                self.modality_interleave * self.modality_ratio[2] % 1 == 0
                if len(self.modality_ratio) >= 3
                else True
            ), "modality_interleave * modality_ratio[1] should be integer"
        if isinstance(self.dataset, MapDataset):
            self.inner_dataset = self.dataset.data
        else:
            self.inner_dataset = self.dataset
        assert self.inner_dataset._load

        self.max_part_id = self.inner_dataset.global_max_part_id

        self.set_epoch(0)

    def load_data_status(self, data_status: List[int], global_shuffle_seed: int = 0):
        """load data_status 用于 DataTraceCallback 调用"""
        # 强制data_status整除combine_batch
        self.global_shuffle_seed = global_shuffle_seed
        if not hasattr(self.inner_dataset.exs[0], "data_status"):
            logger.warn(
                "Inner Datasource has no attribute data_status, ignore load_data_status"
            )
            return
        data_status = [
            math.ceil(i / self.combine_batch) * self.combine_batch for i in data_status
        ]
        for ex in self.inner_dataset.exs:
            if ex.part < len(data_status):
                ex.data_status = data_status[ex.part]
        logger.debug(
            f"dp-[{self.dp_rank}/{self.dp_size}]-loaded_data_status--[{data_status[:10]}]"
        )

    def gen_data_seq(self):
        """
        生成随机采样序列。在给定seed + epoch 的情况下，序列结果稳定可复现
        """
        total = []
        for ex in self.inner_dataset.exs:
            total.extend([(ex.part, 0, i) for i in range(ex.data_status, len(ex))])
        assert (
            len(total) > self.num_consecutive
        ), f"total={total} < num_consecutive={self.num_consecutive}"
        indices = np.array_split(np.array(total), len(total) // self.num_consecutive)
        if self.shuffle:
            self.rng.shuffle(indices)
        indices = np.concatenate(indices)
        indices = self.roundup_and_shard(indices)
        logger.debug(indices[:10])
        return indices

    def load_data_seq_from_cache(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        indices_file = os.path.join(
            self.output_dir,
            f"data_seq.epoch{self.epoch}.dp_{self.dp_rank}_of_{self.dp_size}"
            f"_shard_{self.local_rank}_of_{self.nranks}.pth",
        )
        if self.same_data and os.path.exists(indices_file):
            logger.info(f"load data seq from file - {indices_file}")
            self.load_data_seq = True
            with open(indices_file, "rb") as of:
                return pickle.load(of)
        return None

    def gen_data_seq_weighted_multimodal(
        self, lm_num_examples, mm_num_examples, audio_num_examples
    ):
        """multimodal data seq"""
        assert self.modality_ratio is not None
        logger.info(f"LM-num_examples -- {lm_num_examples}")
        lm_indices = (
            self.gen_data_seq_weighted(lm_num_examples, DATATYPE_2_ID["lm"])
            if lm_num_examples > 0
            else None
        )
        mm_indices = (
            self.gen_data_seq_weighted(mm_num_examples, DATATYPE_2_ID["mm"])
            if mm_num_examples > 0
            else None
        )
        audio_indices = (
            self.gen_data_seq_weighted(audio_num_examples, DATATYPE_2_ID["audio"])
            if audio_num_examples > 0
            else None
        )

        lm_base = (
            int(
                int(self.modality_ratio[0] * self.modality_interleave)
                * self.combine_batch
                * self.per_device_train_batch_size
            )
            if len(self.modality_ratio) >= 1
            else 0
        )
        mm_base = (
            int(
                int(self.modality_ratio[1] * self.modality_interleave)
                * self.combine_batch
                * self.per_device_train_batch_size
            )
            if len(self.modality_ratio) >= 2
            else 0
        )
        audio_base = (
            int(
                int(self.modality_ratio[2] * self.modality_interleave)
                * self.combine_batch
                * self.per_device_train_batch_size
            )
            if len(self.modality_ratio) >= 3
            else 0
        )

        num_batches = math.inf
        if lm_indices is not None and lm_base > 0:
            # lm_indices == None 时，表示该模态样本数为0，则不需要生成该模态data_seq
            # lm_base == 0 时，表示该模态训练step 占比为0，不需要生成该模态data_seq
            num_batches = min(lm_indices.shape[0] // lm_base, num_batches)
        if mm_indices is not None and mm_base > 0:
            # 定义同lm_indices 和 lm_base
            num_batches = min(mm_indices.shape[0] // mm_base, num_batches)
        if audio_indices is not None and audio_base > 0:
            # 定义同lm_indices 和 lm_base
            num_batches = min(audio_indices.shape[0] // audio_base, num_batches)

        all_indices = []
        if lm_indices is not None and lm_base > 0:
            lm_indices = lm_indices[: num_batches * lm_base, :].reshape(
                num_batches, lm_base, -1
            )
            all_indices.append(lm_indices)
        if mm_indices is not None and mm_base > 0:
            mm_indices = mm_indices[: num_batches * mm_base, :].reshape(
                num_batches, mm_base, -1
            )
            all_indices.append(mm_indices)
        if audio_indices is not None and audio_base > 0:
            audio_indices = audio_indices[: num_batches * audio_base, :].reshape(
                num_batches, audio_base, -1
            )
            all_indices.append(audio_indices)

        assert len(all_indices) > 0
        indices = np.concatenate(all_indices, axis=1).reshape(
            -1, all_indices[0].shape[-1]
        )
        logger.debug(
            f"multimodal_data_seq={len(indices)}, example={indices[:10]}, "
            f"modality_interleave={self.modality_interleave}, lm-{lm_base}, mm-{mm_base}, audio-{audio_base}"
        )
        return indices

    def gen_data_seq_weighted(self, num_examples, data_type=None):
        """
        根据数据源权重，生成随机采样序列。在给定seed + epoch 的情况下，序列结果稳定可复现
        ]
        """
        assert (
            self.load_data_seq is False
        ), "需要保证所有epoch的data_seq都从文件加载，否则下次删data_seq无法控住随机性"
        logger.debug(
            f"generating data sequence... #non_consecutive_data_chunks={num_examples},"
            f" num_consecutive={self.num_consecutive}"
        )

        if num_examples > 1e5:
            logger.debug(
                "generating data sequence for very large data, consider use large `num_consecutive`"
            )
        # part内保序全局shuffle
        # 每个part维护一个generator
        if data_type is not None:
            weights = [
                ex.weights for ex in self.inner_dataset.exs if ex.data_type == data_type
            ]
            exs = [ex for ex in self.inner_dataset.exs if ex.data_type == data_type]
        else:
            weights = [ex.weights for ex in self.inner_dataset.exs]
            exs = self.inner_dataset.exs
        assert len(exs) > 0, f"data_type={data_type}, no data found"
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        logger.info(
            f"using weighted sampler, num_consecutive={self.num_consecutive}:\n"
            + "\n".join(["%-100s...%.3e" % (e.path, w) for w, e in zip(weights, exs)])
        )

        part_indices_gen = {}
        indices = []
        for i, ex in enumerate(exs):
            sample_size = int(weights[i] * num_examples)
            logger.debug(
                f"part_data_pre_sampling--[part-{ex.part}]-[sampler-size-{sample_size}]"
            )
            assert ex.combine_batch == self.combine_batch
            part_indices_gen[ex.part] = ex.sampler()
            indices.extend([ex.part] * sample_size)

        logger.debug(
            f"shuffle part placeholder index, size={len(indices)}, exmaple={indices[0]}"
        )
        if self.shuffle:
            self.rng.shuffle(indices)
        logger.debug("shuffle done")
        indices_ret = []
        # 2. part内顺序进行恢复, 构造index时，保证consecutive内恢复, pop(0) 时间复杂度O(n), pop() 时间复杂度O(1)
        logger.debug("build_index from shuffled placeholder")

        for part_id in indices:
            epoch, _index = next(part_indices_gen[part_id])
            # combine_batch = max_seqlen (8k) / base_seqlen (1k)
            if len(_index) % self.combine_batch != 0:
                # part_id 对应padding idx = -1(part_id + 1)
                _index += [-1] * (self.combine_batch - len(_index) % self.combine_batch)
            indices_ret += [(part_id, epoch, i) for i in _index]

        if self.shuffle_consecutive and self.combine_batch >= 1:
            # 将num_consecutive 按 combine_batch 进行 chunk、shuffle 再合并
            part_data_gen = defaultdict(lambda: [])
            logger.debug("consecutive placeholder 2 shuffle")
            for item in indices_ret:
                part_data_gen[item[0]].append(item)
            logger.debug("consecutive placeholder 2 shuffle...")
            part_data_gen_iter = {}
            for key in part_data_gen.keys():
                part_data_gen_iter[key] = iter(part_data_gen[key])
            logger.debug("consecutive placeholder 2 shuffle......")
            # 将part id placeholder 拿出来，拆开consecutive 再按combine_batch 切碎进行shuffle
            placeholder_indices = [i[0] for i in indices_ret]
            placeholder_indices = [
                placeholder_indices[i : i + self.combine_batch]
                for i in range(0, len(placeholder_indices), self.combine_batch)
            ]
            logger.debug("consecutive placeholder 2 shuffle..........")
            self.rng.shuffle(placeholder_indices)
            logger.debug("consecutive placeholder 2 shuffle.............")
            placeholder_indices = [
                item for sublist in placeholder_indices for item in sublist
            ]
            logger.debug("consecutive placeholder 2 shuffle................")
            indices_ret = [next(part_data_gen_iter[i]) for i in placeholder_indices]
            logger.debug("consecutive placeholder 2 shuffle done")

        logger.debug("build index done")
        indices = np.array(indices_ret)
        del indices_ret
        logger.debug(f"num_data_seq={len(indices)}, example={indices[:10]}")
        indices = self.roundup_and_shard(indices)
        return indices

    def roundup_and_shard(self, indices):
        # add extra samples to make it evenly divisible
        if self.nranks == 1:
            logger.info("use use_train_part_sharding, skip padding")
            # TODO 非train part sharding 也需要保证padding 被combine batch 整除
            return indices

        padding_size = self.total_size - len(indices)
        logger.info(
            f"padding-size={padding_size}, total_size={self.total_size} shard={self.local_rank}/{self.nranks}"
        )
        if padding_size < 0:
            indices = indices[:padding_size]
        else:
            indices = np.concatenate(
                [
                    indices,
                    np.tile(indices, math.ceil(padding_size / len(indices)))[
                        :padding_size
                    ],
                ]
            )

        assert len(indices) == self.total_size, (len(indices), self.total_size)

        # subsample
        indices = indices[self.local_rank : self.total_size : self.nranks]
        assert len(indices) == self.num_samples
        return indices

    def __len__(self):
        # PaddleNLP expect TypeError for infinite datasets:
        # https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/trainer/trainer_utils.py#L515
        raise TypeError

    def __iter__(self):
        # deterministically shuffle based on epoch and seed
        self.rng = random.Random(self.seed + self.epoch + self.global_shuffle_seed)
        logger.info(f"seed={self.seed + self.epoch + self.global_shuffle_seed}")
        weights = [e.weights for e in self.inner_dataset.exs]
        if any([w is None for w in weights]) or sum(weights) == 0.0:
            logger.info(f"using normal sampler, num_consecutive={self.num_consecutive}")
            indices = self.gen_data_seq()
            self.weights = None
        else:
            self.weights = weights
            num_examples = sum([ex.num_examples for ex in self.inner_dataset.exs])
            if self.modality_ratio is not None:
                lm_num_examples = sum(
                    [
                        ex.num_examples
                        for ex in self.inner_dataset.exs
                        if ex.data_type == DATATYPE_2_ID["lm"]
                    ]
                )
                mm_num_examples = sum(
                    [
                        ex.num_examples
                        for ex in self.inner_dataset.exs
                        if ex.data_type == DATATYPE_2_ID["mm"]
                    ]
                )
                audio_num_examples = sum(
                    [
                        ex.num_examples
                        for ex in self.inner_dataset.exs
                        if ex.data_type == DATATYPE_2_ID["audio"]
                    ]
                )
            if self.global_shuffle_num_examples > 0:
                num_examples = min([self.global_shuffle_num_examples, num_examples])
                if self.modality_ratio is not None:
                    lm_num_examples = min(
                        [self.global_shuffle_num_examples, lm_num_examples]
                    )
                    mm_num_examples = min(
                        [self.global_shuffle_num_examples, mm_num_examples]
                    )
                    audio_num_examples = min(
                        [self.global_shuffle_num_examples, audio_num_examples]
                    )
                logger.debug(
                    f"using global shuffle num examples: {self.global_shuffle_num_examples}"
                )
            # TODO: 提前造好所有idx，在数据量特别大的时候会爆*内存*，后续fix下。
            indices = self.load_data_seq_from_cache()
            if indices is None:
                indices = (
                    self.gen_data_seq_weighted_multimodal(
                        lm_num_examples, mm_num_examples, audio_num_examples
                    )
                    if self.modality_ratio is not None
                    else self.gen_data_seq_weighted(num_examples)
                )

        if self.output_dir:
            with open(
                os.path.join(
                    self.output_dir,
                    f"data_seq.epoch{self.epoch}.dp_{self.dp_rank}_of_{self.dp_size}"
                    f"_shard_{self.local_rank}_of_{self.nranks}.pth",
                ),
                "wb",
            ) as of:
                pickle.dump(indices, of, protocol=4)

        def ret():  # 无穷长reader。
            # info = paddle.io.get_worker_info()
            nonlocal indices
            buf = []
            logger.info(f"start training sequence, data-sequence: {indices[:10]}")
            while 1:
                if self.consumed_samples >= len(indices):
                    self.consumed_samples -= len(indices)
                else:
                    for i in range(self.consumed_samples, len(indices)):
                        if len(buf) == self.batch_size:
                            yield buf
                            buf = []
                        buf.append(indices[i].tolist())
                    self.consumed_samples = 0
                self.epoch += 1
                logger.info(
                    f"epoch done, #data={self.total_size}, reshuffle-sequence: epoch={self.epoch}"
                )

                self.rng = random.Random(self.seed + self.epoch)
                if self.weights:
                    indices = self.load_data_seq_from_cache()
                    if indices is None:
                        indices = (
                            self.gen_data_seq_weighted_multimodal(
                                lm_num_examples, mm_num_examples, audio_num_examples
                            )
                            if self.modality_ratio is not None
                            else self.gen_data_seq_weighted(num_examples)
                        )
                else:
                    indices = self.gen_data_seq()
                if self.output_dir:
                    with open(
                        os.path.join(
                            self.output_dir,
                            f"data_seq.epoch{self.epoch}.dp_{self.dp_rank}_of_{self.dp_size}"
                            f"_shard_{self.local_rank}_of_{self.nranks}.pth",
                        ),
                        "wb",
                    ) as of:
                        pickle.dump(indices, of, protocol=4)

        return ret()

    def set_epoch(self, epoch=0, consumed_samples=0):  # PaddleNLP只会调用1次
        """
        设置当前训练轮次和已消耗样本数。

        Args:
            epoch (int, optional): 当前训练轮次。默认为0。
            consumed_samples (int, optional): 当前轮次已消耗的样本数（全局视角）。
                在PaddleNLP中，该参数为多机*全局视角的*已消耗样本数，
                这里会将其转换为单机视角的已消耗样本数。默认为0。

        Returns:
            None

        Note:
            PaddleNLP只会调用此函数一次。
        """
        consumed_samples = consumed_samples // self.dp_size  # per-device-micro-examples
        # paddle-NLP提供的consumed-example是多机*全局视角的*, 这里给还原成单机视角的 consumed_samples
        logger.debug(f"set consumed samples={consumed_samples}, epoch={epoch}")
        super().set_epoch(epoch, consumed_samples)

        if isinstance(self.inner_dataset, ExampleSet):
            for ex in self.inner_dataset.exs:
                if isinstance(ex, ExampleSetSingleDataSource):
                    ex.epoch = epoch


class AutoPretrainingTrainer(AutoTrainer):
    """
    自动并行训练器，当前许多功能仍在测试适配中。
    """

    def __init__(self, _shit=None, args=None, model=None, callbacks=[], **kwargs):
        assert _shit is None, "use key-ward argument"
        callbacks = [
            LoggingCallback(),  # ``LoggingCallback` 需要在`TensorBoardCallback` 前面, timmer会reset
            StopperCallback(),
            TensorBoardCallback(
                args, model=model, log_tokens_per_step=True, log_flops_per_step=False
            ),
        ] + callbacks
        if args.reshard_save_then_exit:
            callbacks.append(ReshardSaveExitCallback(self))

        if args.adaptive_norm_clip:
            callbacks.append(
                ClipGradByAdaptiveNormCallback(),
            )
        # TODO: restore param.name in other ways
        args.use_async_save = (
            args.use_async_save and args.save_sharded_model and args.load_sharded_model
        )
        super().__init__(args=args, model=model, callbacks=callbacks, **kwargs)

        def get_numel_item(p):
            item = p.numel().item()
            return item if item else 0

        model_numel = sum(
            get_numel_item(p)
            for n, p in model.named_parameters()
            if not p.stop_gradient and "embeddings" not in n and "embed_tokens" not in n
        )
        numel_tensor = paddle.to_tensor(model_numel)
        dist.all_reduce(numel_tensor)
        self.model_numel = numel_tensor.item() // self.args.dataset_world_size

        self.pop_callback(PrinterCallback)
        self.pp_data_buffer = []  # pp
        self._tokens_per_sec_per_card_buffer = []
        self._start_save_time = time.time()
        self._end_save_time = time.time()
        self._first_end_save_time = time.time()
        self.resume_global_step = -1
        self.first_skip_step = (
            5 if self.args.save_steps > 5 else self.args.save_steps / 2
        )
        if args.same_data:
            logger.warning(
                "You have set same_data=True. \
                            Carefully check whether the data, population proportion, "
                "and DP count are completely consistent with those before."
            )
        else:
            logger.warning(
                "You have set same_data=False. \
                            which will regenerate the global shuffle domain."
            )
        # self.return_value = paddle.zeros([]) #fake return value

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
            ctx_manager = (
                contextlib.nullcontext()
                if sys.version_info >= (3, 7)
                else contextlib.suppress()
            )

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
                {
                    i: j.shape
                    for i, j in state_dict.items()
                    if i not in ["master_weights", "LR_Scheduler"]
                },
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
                v = state_dict.get(k, paddle.zeros(s, "float32")).cuda()
                v.name = k
                # k = k.replace("_fp32_master_0", "")  # TODO 这一手replace待品
                dist.broadcast(v, src=src_rank, group=group)
                logger.info(f"broadcast moe optimizer {k} from {src_rank}")
                base_state_dict[k] = v.cpu()
            for k, s in buf[1].items():
                v = (
                    state_dict["master_weights"]
                    .get(k, paddle.zeros(s, "float32"))
                    .cuda()
                )
                v.name = k
                dist.broadcast(v, src=src_rank, group=group)
                logger.info(
                    f"broadcast moe optimizer-master_weights {k} from {src_rank}"
                )
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
        optimizer_name = _add_variant(
            PADDLE_OPTIMIZER_NAME, self.args.optimizer_name_suffix
        )
        saved_signal_path = os.path.join(output_dir, f"saved_signal_{dist.get_rank()}")

        os.makedirs(output_dir, exist_ok=True)
        state_dict = self.model.state_dict()
        optimzier_state_dict = self.optimizer.state_dict()

        filtered_state_dict = OrderedDict()
        filter_optimzier_state_dict = OrderedDict()

        param_names_in_master_weights = (
            list(optimzier_state_dict["master_weights"].keys())
            if self.args.bf16
            else []
        )
        filter_optimzier_state_dict["master_weights"] = OrderedDict()

        for k, v in state_dict.items():
            if getattr(v, "no_sync", False):

                if v.name in param_names_in_master_weights:
                    filter_optimzier_state_dict["master_weights"][v.name] = (
                        optimzier_state_dict["master_weights"][v.name]
                    )
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
                    os.path.join(
                        output_dir,
                        _add_variant(PADDLE_WEIGHTS_NAME, self.args.weight_name_suffix),
                    ),
                )
        paddle.save(
            filter_optimzier_state_dict, os.path.join(output_dir, optimizer_name)
        )
        with open(saved_signal_path, mode="w+") as f:
            f.write("1")

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"
    ):
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

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )
        return output.metrics

    def prediction_pipeline_step(
        self, model, inputs, prediction_loss_only, ignore_keys
    ):
        """doc"""
        loss, _, labels = super().prediction_pipeline_step(
            model, inputs, prediction_loss_only, ignore_keys
        )  # ERNIE-PP模型的loss其实是loss-sum。
        num_tokens = (labels != self.tokenizer.ignored_index).sum().item()
        loss_avg = loss * self.model_wrapped.accumulate_steps / num_tokens
        return loss_avg, loss, labels

    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        # only used in Map-Styled Dataset.
        if self.args.use_dummy_dataset:
            return DummySampler(
                self.train_dataset,
                self.args.per_device_train_batch_size * self.args.combine_batch,
            )
        if (
            self.args.use_train_part_sharding
        ):  # 走了train-part-sharding后暂时不需要做进程内切片。
            num_replicas = 1
            rank = 0
        else:
            num_replicas = self.args.reeao_dataset_world_size
            rank = self.args.reeao_dataset_rank
        # NOTE: batch_size没有乘以dp_world_size，是因为dataset层已经将数据按照dp_rank切片了
        batch_size = self.args.per_device_train_batch_size * self.args.combine_batch
        batch_size *= self.args.gradient_accumulation_steps
        batch_sampler = WeightedDistributedSamplerAuto(
            self.train_dataset,
            batch_size,
            self.args.output_dir,
            dp_rank=self.args.reeao_dataset_rank,
            dp_size=self.args.reeao_dataset_world_size,
            num_replicas=num_replicas,
            rank=rank,
            seed=self.args.seed,
            batch_size_warmup_steps=self.args.batch_size_warmup_steps,  # used to reesume from ckpt
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            max_gradient_accumulation_steps=self.args.max_gradient_accumulation_steps,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            batch_size_warmup_increment=self.args.batch_size_warmup_increment,
            shuffle=not self.args.no_shuffle,
            drop_last=False,
            num_consecutive=self.args.num_consecutive,
            combine_batch=self.args.combine_batch,
            shuffle_consecutive=self.args.shuffle_consecutive,
            global_shuffle_num_examples=self.args.global_shuffle_num_examples,
            same_data=self.args.same_data,
            modality_ratio=self.args.modality_ratio,
            modality_interleave=(
                self.args.modality_interleave * self.args.combine_batch
                if self.args.modality_interleave
                else None
            ),
        )
        return batch_sampler

    def get_train_dataloader(self):
        """
        获取训练数据的 DataLoader。

        Args:
            无。

        Returns:
            DataLoader: 训练数据的 DataLoader 对象。

        Raises:
            ValueError: 如果 `self.args.need_data` 为 True 但 `self.train_dataset` 为 None 时，抛出该异常。

        """
        if self.args.need_data and self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        # NOTE: 纯dp也可以用此dataloader, 无需区分
        _DataLoader = partial(
            DistDataLoaderAuto,
            need_data=self.args.need_data,
            pp_broadcast=not self.args.pp_need_data,
        )
        # _DataLoader = (
        #     partial(DistDataLoaderAuto, need_data=self.args.need_data, pp_broadcast=not self.args.pp_need_data)
        #     if (self.args.tensor_parallel_degree > 1 or self.args.pipeline_parallel_degree > 1)
        #     else DataLoader
        # )  # fleet初始化之后才能使用`DistDataLoaderAuto`

        train_dataset = self.train_dataset
        if self._is_iterable_dataset(train_dataset):
            return DataLoader(
                train_dataset,
                batch_size=None,  # we do data collation in Stream
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                use_shared_memory=True,
                prefetch_factor=self.args.prefetch_factor,
            )
        if self.args.need_data:
            train_sampler = self._get_train_sampler()
        else:
            train_sampler = None
        return _DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            prefetch_factor=self.args.prefetch_factor,
        )

    def _broadcast_final_loss(self, tr_loss):
        tr_loss = tr_loss._local_value() if tr_loss.is_dist() else tr_loss

        if self.args.pipeline_parallel_degree > 1:
            hcg = fleet.get_hybrid_communicate_group()
            num_stages = hcg.get_pipe_parallel_world_size()

            paddle.distributed.broadcast(
                tr_loss,
                src=hcg.get_rank_from_stage(num_stages - 1),
                sync_op=True,
                group=hcg.get_pipe_parallel_group(),
            )
        return tr_loss

    def _maybe_log_save_evaluate(
        self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs
    ):
        super()._maybe_log_save_evaluate(
            tr_loss, model, epoch, ignore_keys_for_eval, **kwargs
        )
        return
        with _exec_mode_guard("dynamic"):
            tr_loss = tr_loss._local_value() if tr_loss.is_dist() else tr_loss
            flag_log = self.control.should_log
            if self.control.should_log:
                logs = {}
                tr_loss = self._broadcast_final_loss(tr_loss)
                # TODO(Ruibiao): 自动并行下tr_loss已经经过一次allreduce，需要有接口支持获取每路dp下的loss
                tr_loss_single_dp_scalar = tr_loss.item()
                dist.all_reduce(
                    tr_loss, dist.ReduceOp.SUM
                )  # 3级并行时，每个pp下的loss会广播，全局reduce-mean的时候，分子分母都会乘以pp_world_size，结果会被约掉
                tr_loss_scalar = tr_loss.item() / dist.get_world_size()
                tr_loss.zero_()

                # reset tr_loss to zero
                logs["loss"] = tr_loss_scalar / (
                    self.state.global_step - self._globalstep_last_logged
                )
                logs["loss_cur_dp"] = tr_loss_single_dp_scalar / (
                    self.state.global_step - self._globalstep_last_logged
                )
                logs["learning_rate"] = float(self._get_learning_rate())
                logs["global_step"] = int(self.state.global_step)

                divisor = 2**30

                current_device = framework._current_expected_place_()
                device_id = current_device.get_device_id()
                current_memory_allocated = core.device_memory_stat_current_value(
                    "Allocated", device_id
                )
                current_memory_reserved = core.device_memory_stat_current_value(
                    "Reserved", device_id
                )
                max_memory_allocated = core.device_memory_stat_peak_value(
                    "Allocated", device_id
                )
                max_memory_reserved = core.device_memory_stat_peak_value(
                    "Reserved", device_id
                )
                logs["mem_allocated_gb"] = current_memory_allocated / divisor
                logs["max_mem_allocated_gb"] = max_memory_allocated / divisor
                logs["mem_reserved_gb"] = current_memory_reserved / divisor
                logs["max_mem_reserved_gb"] = max_memory_reserved / divisor

                if not self.args.enable_global_training_logs:
                    global_training_logs.global_meters_keys = [
                        "data_not_valid",
                        "experts_per_token",
                    ]

                if get_env_device() == "gpu":
                    info_callback = global_training_logs.dict(use_async=True)

                if hasattr(self, "scaler"):
                    logs["loss_scale"] = float(
                        "{0:.3e}".format(self.scaler._scale.item())
                    )

                total_train_batch_size = (
                    self.args.train_batch_size
                    * self.args.gradient_accumulation_steps
                    * self.args.reeao_dataset_world_size
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

                tokens_per_steps = self.args.max_seq_length * total_train_batch_size
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
                    tokens_per_steps
                    * logs["global_steps_per_second"]
                    / self.args.world_size,
                    1,
                )
                self._tokens_per_sec_per_card_buffer.append(
                    logs["tokens_per_sec_per_card"]
                )
                logs["tokens_per_sec_per_card_average"] = round(
                    np.mean(self._tokens_per_sec_per_card_buffer), 1
                )
                if self.resume_global_step == -1:
                    self.resume_global_step = self.state.global_step - 1
                if (
                    self.state.global_step
                    <= self.resume_global_step + self.first_skip_step
                ):
                    self._tokens_per_sec_per_card_buffer = []
                    self._end_save_time = time.time()

                self._total_loss_scalar += tr_loss_scalar
                self._globalstep_last_logged = self.state.global_step
                self._globalstep_last_start_time = time.time()

                if get_env_device() == "xpu":
                    info, gathered_info = global_training_logs.dict(use_async=False)
                else:
                    info, gathered_info = info_callback()

                global_training_logs.reset()
                logs.update({f"{k}_cur_dp": v for k, v in info.items()})
                logs.update(gathered_info)
                if self.args.enable_global_training_logs:
                    info_list = []
                    dist.all_gather_object(info_list, info)
                    logs.update(
                        {
                            k: np.mean([v[k] for v in info_list if k in v])
                            for k in {key for item in info_list for key in item.keys()}
                        }
                    )

                self.log(logs, **kwargs)

            metrics = None
            if self.control.should_evaluate:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)

            if self.control.should_save:
                self._start_save_time = time.time()
                self._save_checkpoint(model, metrics=metrics)
                paddle.distributed.barrier()
                self.control = self.callback_handler.on_save(
                    self.args, self.state, self.control
                )
                if flag_log:
                    logs = {}
                    tbk = self._start_save_time - self._end_save_time
                    if (
                        self.state.global_step
                        == self.resume_global_step + self.args.save_steps
                    ):
                        actual_tbk = self._start_save_time - self._first_end_save_time
                        actual_avg_speed_step = (
                            self.args.save_steps
                            * tokens_per_steps
                            / actual_tbk
                            / self.args.world_size
                        )
                        tbk = (
                            tbk
                            / (self.args.save_steps - self.first_skip_step)
                            * self.args.save_steps
                        )
                    ts = time.time() - self._start_save_time
                    tokens_per_steps = self.args.max_seq_length * total_train_batch_size
                    avg_speed_step = (
                        self.args.save_steps
                        * tokens_per_steps
                        / tbk
                        / self.args.world_size
                    )
                    logs["global_save_step"] = self.state.global_step
                    logs["train_time_sec_without_save"] = tbk
                    logs["save_ckpt_time_sec"] = ts
                    logs["average_tokens_per_sec_per_card_without_save"] = round(
                        avg_speed_step, 1
                    )
                    logs["average_tokens_per_sec_per_card_with_save"] = round(
                        self.args.save_steps
                        * tokens_per_steps
                        / (tbk + ts)
                        / self.args.world_size,
                        2,
                    )
                    if (
                        self.state.global_step
                        == self.resume_global_step + self.args.save_steps
                    ):
                        logs["actual_average_tokens_per_sec_per_card_without_save"] = (
                            round(actual_avg_speed_step, 1)
                        )
                        logs["actual_average_tokens_per_sec_per_card_with_save"] = (
                            round(
                                self.args.save_steps
                                * tokens_per_steps
                                / (actual_tbk + ts)
                                / self.args.world_size,
                                2,
                            )
                        )
                    logs["one_day_billion_tokens_without_save"] = round(
                        0.0000864 * self.args.save_steps * tokens_per_steps / tbk, 2
                    )
                    logs["one_day_billion_tokens_with_save"] = round(
                        0.0000864
                        * self.args.save_steps
                        * tokens_per_steps
                        / (tbk + ts),
                        2,
                    )
                    self.log(logs, **kwargs)
                    self._globalstep_last_start_time = time.time()
                    self._tokens_per_sec_per_card_buffer = []
                self._end_save_time = time.time()

    def create_scheduler(self, num_training_steps):
        """
        根据给定的参数创建一个学习率调度器。

        Args:
            num_training_steps (int): 训练的总步数。

        Returns:
            torch.optim.lr_scheduler._LRScheduler: 学习率调度器对象。

        """
        if self.args.warmup_steps > 0:
            warmup = self.args.warmup_steps
        else:
            warmup = int(self.args.warmup_ratio * num_training_steps)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.args.learning_rate,
            warmup,
            self.args.max_steps,
            min_lr=self.args.min_lr if self.args.min_lr else 0.0,
        )
        print(f"lr_scheduler : {self.lr_scheduler}")

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
            else self.model.parameters()
        )
        if self.args.train_emb_only:
            logger.info(
                f"using `train-emb-only`, #embedding params={len(optimizer_params)}"
            )
        if self.optimizer is None:

            def need_decay(name):
                if (
                    name == "ernie.norm.weight"
                    and self.args.pipeline_parallel_degree > 1
                ):
                    return True
                return not any(nd in name for nd in ["bias", "norm"])

            decay_parameters = [
                p.name for n, p in self.model.named_parameters() if need_decay(n)
            ]

            def apply_decay_param_fun(x):
                return x in decay_parameters

            optimizer_cls, optimizer_kwargs = AutoTrainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            if self.args.adaptive_norm_clip:
                if "split_param" in self.args.sharding_parallel_config:
                    from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
                        DygraphShardingOptimizerV2,
                    )

                    v2_assign_slice_grad = DygraphShardingOptimizerV2._assign_slice_grad

                    def _assign_slice_grad(self):
                        v2_assign_slice_grad(self)
                        assert isinstance(
                            self._grad_clip, ClipGradByAdaptiveNorm
                        ), "self._grad_clip must be ClipGradByAdaptiveNorm"
                        if not hasattr(self._grad_clip, "pname_to_paramindex"):
                            pname_to_paramindex = {}
                            assert not isinstance(self._parameter_list[0], dict)
                            for idx, param in enumerate(self._parameter_list):
                                param = self._slice_params[param.name]
                                if param._is_initialized():
                                    pname_to_paramindex[param.name] = idx
                            self._grad_clip.pname_to_paramindex = pname_to_paramindex
                            self._grad_clip.num_params = len(self._parameter_list)
                            self._grad_clip.sharding_stage1_v2 = True

                    DygraphShardingOptimizerV2._assign_slice_grad = _assign_slice_grad
                    logger.info(
                        "Hack DygraphShardingOptimizerV2._assign_slice_grad for ClipGradByAdaptiveNorm"
                    )

                grad_clip = ClipGradByAdaptiveNorm(
                    clip_ratio=self.args.adaptive_norm_clip_ratio,
                    start_clip_steps=self.args.adaptive_norm_start_clip_steps,
                    shard_clip=self.args.adaptive_norm_shard_clip,
                    enable_record=self.args.adaptive_norm_enable_record,
                    enable_record_clip_history=self.args.adaptive_norm_enable_record_clip_history,
                    verbose=self.args.adaptive_norm_verbose,
                )
                logger.info("using ClipGradByAdaptiveNorm")
            elif (
                self.args.use_moe
                and not self.args.use_hybrid_parallel
                and not self.args.enable_auto_parallel
            ):
                logger.info("using moe Global clip")

                def expert_fn(p):
                    return getattr(p, "no_sync", False)

                grad_clip = ClipGradForMOEByGlobalNorm(
                    self.args.max_grad_norm,
                    is_expert_param_func=expert_fn,
                    moe_group=_get_global_group(),  # None 为全局通信组,
                    local_clip=False,
                )
            else:
                grad_clip = (
                    nn.ClipGradByGlobalNorm(self.args.max_grad_norm)
                    if self.args.max_grad_norm > 0
                    else None
                )

            self.static_name_to_dyg_name = {
                p.name: n for n, p in self.model.state_dict().items()
            }
            gate_pattern = re.compile(
                r"ernie\.layers\.0\.mlp\.gate\.weight"
            )  # TODO 换成更加通配的方法
            vit_pattern = re.compile(
                r"vision_model\.(cls_token|pos_embed|patch_embed|blocks)"
            )
            vit_blocks_pattern = re.compile(r"vision_model\.blocks\.(\d+)\.")

            def lr_ratio_fn(param):
                if param.name in self.static_name_to_dyg_name.keys():
                    name = self.static_name_to_dyg_name[param.name]
                    # logger.info(f'search {param.name} -> {name}')
                    if self.args.moe_gate_lr_ratio is not None and gate_pattern.match(
                        name
                    ):
                        logger.info(
                            f"apply moe_gate_lr_ratio to {name}, ratio={self.args.moe_gate_lr_ratio}"
                        )
                        return float(self.args.moe_gate_lr_ratio)
                    elif self.args.vit_lr_ratio is not None and vit_pattern.match(name):
                        n_layers = self.model.config.vision_config.layers
                        if vit_blocks_pattern.match(name):
                            layer_id = int(vit_blocks_pattern.match(name).group(1))
                        else:
                            layer_id = 0
                        lr_ratio = self.args.vit_lr_ratio ** (n_layers - 1 - layer_id)
                        logger.info(f"apply vit lr_ratio to {name}, ratio={lr_ratio}")
                        return float(lr_ratio)
                return 1.0

            self.optimizer = optimizer_cls(
                learning_rate=(
                    self.lr_scheduler if lr_scheduler is None else lr_scheduler
                ),
                apply_decay_param_fun=apply_decay_param_fun,
                parameters=optimizer_params,
                weight_decay=self.args.weight_decay,
                grad_clip=grad_clip,
                multi_precision=True,
                lr_ratio=(
                    lr_ratio_fn
                    if (
                        self.args.moe_gate_lr_ratio is not None
                        or self.args.vit_lr_ratio is not None
                    )
                    else None
                ),
                **optimizer_kwargs,
            )

        self.static_name_to_dyg_name = {
            p.name: n for n, p in self.model.named_parameters()
        }

        return self.optimizer

    def save_model(self, output_dir=None):
        """
        保存模型到指定目录，并保存静态名称到动态名称的映射关系。

        Args:
            output_dir (str, optional): 输出目录的路径。默认为 None，即使用父类指定的目录。

        Returns:
            None

        """
        super().save_model(output_dir)
        if self.args.should_save:
            with open(
                os.path.join(output_dir, "static_name_to_dyg_name.json"), "w"
            ) as of:
                of.write(json.dumps(self.static_name_to_dyg_name))

    def _load_rng_state(self, checkpoint):
        # 预训练环节并不需要由框架控制 `rng`
        pass

    def _get_meshes_for_loader(self):
        def _get_mesh(pp_idx=0):
            return self.global_mesh.get_mesh_with_dim("pp")[pp_idx]

        meshes = []
        if self.args.pipeline_parallel_degree > 1:
            # input_ids
            if self.args.multimodal:
                # `input_ids`, `labels`, `data_id`, `src_id`, `data_type`, `images`, `token_type_ids`,
                # `image_type_ids`, `has_images`
                meshes.append(
                    [
                        _get_mesh(0),
                        _get_mesh(-1),
                        _get_mesh(0),
                        _get_mesh(0),
                        _get_mesh(0),
                        _get_mesh(0),
                        _get_mesh(0),
                        _get_mesh(0),
                        _get_mesh(0),
                    ]
                )
            else:
                meshes.append([_get_mesh(0), _get_mesh(-1), _get_mesh(0), _get_mesh(0)])
            # labels
            meshes.append(_get_mesh(self.args.pipeline_parallel_degree - 1))
        else:
            meshes.append(_get_mesh(0))
        return meshes

    def _wrap_for_dist_loader(self, train_dataloader):
        self.dense_tensor_idx = None
        dist_loader = dist.shard_dataloader(
            dataloader=train_dataloader,
            meshes=self._get_meshes_for_loader(),
            shard_dims="dp",
            is_dataset_splitted=True,
        )
        dist_loader._input_keys = ["input_ids", "labels"]
        return dist_loader
