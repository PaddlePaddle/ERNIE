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
import contextlib
from typing import Optional
from collections import OrderedDict
from dataclasses import dataclass, field
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
    from paddleformers.utils.env import PADDLE_OPTIMIZER_NAME
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
from src.datasets.dist_data_loader import (
    DistDataLoaderAuto,
)
from src.utils.misc import global_training_logs
from src.clip import ClipGradByAdaptiveNorm, ClipGradForMOEByGlobalNorm

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
except ImportError:

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


@dataclass
@add_start_docstrings(AutoTrainingArguments.__doc__)
class AutoPreTrainingArguments(AutoTrainingArguments):

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

    input_dir: str = field(default=None, metadata={"help": "data path"})
    split: str = field(
        default="949,50,1", metadata={"help": "Train/valid/test data split ratio"}
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
        default=False,
        metadata={
            "help": "Whether adaptivenormclip enables the recording of statistical history clipping"
        },
    )
    adaptive_norm_verbose: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether adaptivenormclip enables crop log printing"},
    )
    use_async_save: Optional[bool] = field(
        default=False, metadata={"help": "Whether to enable asynchronous save function"}
    )
    pre_alloc_memory: float = field(
        default=0.0,
        metadata={
            "help": "Pre-allocate one specific-capacity empty tensor "
            "and release it for avoiding memory fragmentation"
        },
    )
    enable_global_training_logs: bool = field(
        default=False, metadata={"help": "Whether to enable global_training_logs"}
    )
    use_dummy_dataset: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use dummydataset, only for debug"}
    )
    reshard_save_then_exit: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to exit the program directly after reshard"},
    )
    moe_group: Optional[str] = field(
        default="dp",
        metadata={
            "help": "MOE's communication group currently supports 'dp|sharding|mp|dummy'"
        },
    )
    use_moe: Optional[bool] = field(
        default=False, metadata={"help": "expert parallel 临时替代"}
    )
    moe_use_all2all: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use all2all communication mode"}
    )
    log_global_grad_norm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Print global grade norm, only effective when 'enable_global_training_gogs' is enabled"
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
    def need_data(self):

        if self.pp_need_data_degree:
            assert self.pipeline_parallel_degree > 1
            assert (
                self.pp_need_data_degree >= 2
                and self.pp_need_data_degree <= self.pipeline_parallel_degree
            ), (
                self.pp_need_data_degree,
                self.pipeline_parallel_degree,
            )
            no_need_data_range = list(
                range(self.pp_need_data_degree - 1, self.pipeline_parallel_degree - 1)
            )
            return self.tensor_parallel_rank == 0 and (
                self.pipeline_parallel_rank not in no_need_data_range
            )
        return self.pipeline_parallel_rank == 0 and self.tensor_parallel_rank == 0

    @property
    def combine_batch(self):
        return self.max_seq_length // self.base_seq_length

    @property
    def reeao_dataset_rank(self):
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

        assert not (self.sharding_parallel_degree > 1 and self.data_parallel_rank > 1)
        return (
            max(self.pp_need_data_degree, 1) * self.sharding_parallel_rank
            + reeao_pp_rank
        )

    @property
    def reeao_dataset_world_size(self):
        if not self.pp_need_data:
            return super().dataset_world_size
        return (
            max(self.sharding_parallel_degree, 1)
            * max(self.data_parallel_degree, 1)
            * max(self.pipeline_parallel_degree, 1)
        )

    def __post_init__(self):
        super().__post_init__()
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
            self.max_gradient_accumulation_steps = self.gradient_accumulation_steps
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
                assert hasattr(fleet.fleet, "_user_defined_strategy")
                user_defined_strategy = fleet.fleet._user_defined_strategy
                user_defined_strategy.hybrid_configs[
                    "sharding_configs"
                ].accumulate_steps = self.gradient_accumulation_steps

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


class AutoPretrainingTrainer(AutoTrainer):

    def __init__(self, _shit=None, args=None, model=None, callbacks=[], **kwargs):
        assert _shit is None, "use key-ward argument"
        callbacks = [
            LoggingCallback(),
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
            for k, s in buf[0].items():
                v = state_dict.get(k, paddle.zeros(s, "float32")).cuda()
                v.name = k
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
        return state_dict

    def _save_moe_weights(self, output_dir):
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

        self.model_wrapped.accumulate_steps = self.args.gradient_accumulation_steps
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        start_time = time.time()
        compute_metrics = self.compute_metrics
        eval_loop = self.evaluation_loop

        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if compute_metrics is None else None,
            ignore_keys=ignore_keys,
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

        loss, _, labels = super().prediction_pipeline_step(
            model, inputs, prediction_loss_only, ignore_keys
        )
        num_tokens = (labels != self.tokenizer.ignored_index).sum().item()
        loss_avg = loss * self.model_wrapped.accumulate_steps / num_tokens
        return loss_avg, loss, labels

    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        return PaddleNLPDistributedBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            num_replicas=self.args.dataset_world_size,
            rank=self.args.dataset_rank,
            drop_last=self.args.dataloader_drop_last,
        )

    def get_train_dataloader(self):

        if self.args.need_data and self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        _DataLoader = partial(
            DistDataLoaderAuto,
            need_data=self.args.need_data,
            pp_broadcast=not self.args.pp_need_data,
        )

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
                tr_loss_single_dp_scalar = tr_loss.item()
                dist.all_reduce(tr_loss, dist.ReduceOp.SUM)
                tr_loss_scalar = tr_loss.item() / dist.get_world_size()
                tr_loss.zero_()

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
                    moe_group=_get_global_group(),
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
            gate_pattern = re.compile(r"ernie\.layers\.0\.mlp\.gate\.weight")
            vit_pattern = re.compile(
                r"vision_model\.(cls_token|pos_embed|patch_embed|blocks)"
            )
            vit_blocks_pattern = re.compile(r"vision_model\.blocks\.(\d+)\.")

            def lr_ratio_fn(param):
                if param.name in self.static_name_to_dyg_name.keys():
                    name = self.static_name_to_dyg_name[param.name]
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

        super().save_model(output_dir)
        if self.args.should_save:
            with open(
                os.path.join(output_dir, "static_name_to_dyg_name.json"), "w"
            ) as of:
                of.write(json.dumps(self.static_name_to_dyg_name))

    def _load_rng_state(self, checkpoint):
        pass

    def _get_meshes_for_loader(self):
        def _get_mesh(pp_idx=0):
            return self.global_mesh.get_mesh_with_dim("pp")[pp_idx]

        meshes = []
        if self.args.pipeline_parallel_degree > 1:
            # input_ids
            meshes.append(
                [
                    _get_mesh(0),
                    _get_mesh(-1),
                ]
            )
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
