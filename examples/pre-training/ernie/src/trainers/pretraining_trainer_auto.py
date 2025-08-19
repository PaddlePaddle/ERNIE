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
from dataclasses import dataclass, field
import time
import math
import logging
from functools import partial


import paddle
import paddle.nn as nn
from paddle.io import DataLoader
import paddle.amp.auto_cast as autocast
from paddle.distributed.communication.group import _get_global_group

from paddleformers.trainer import (
    speed_metrics,
)

from paddleformers.trainer.auto_trainer import AutoTrainer


from paddleformers.utils.batch_sampler import (
    DistributedBatchSampler as PaddleNLPDistributedBatchSampler,
)


from paddleformers.trainer.utils import add_start_docstrings
from paddleformers.trainer.trainer_callback import PrinterCallback
from paddle.distributed import fleet
import paddle.distributed as dist


from src.lr_schedulers import get_cosine_schedule_with_warmup
from src.utils_auto.training_utils import (
    reset_per_device_batch_size,
)
from src.callbacks_auto import (
    TensorBoardCallback,
    LoggingCallback,
    StopperCallback,
)
from src.datasets.dist_data_loader import (
    DistDataLoaderAuto,
)
from src.clip import ClipGradForMOEByGlobalNormAuto


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

    prefetch_factor: int = field(
        default=2,
        metadata={"help": "global random seed factor."},
    )
    eval_iters: int = field(
        default=-1,
        metadata={"help": "eval iteration for every evaluation."},
    )

    min_lr: float = field(
        default=0.0,
        metadata={"help": "minus learning rate"},
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

    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )

    sequence_parallel: Optional[int] = field(
        default=0,
        metadata={},
    )

    virtual_pp_degree: Optional[int] = field(
        default=1,
        metadata={
            "help": "vpp",
        },
    )

    use_async_save: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use async_save instead of paddle.save."},
    )
    pre_alloc_memory: float = field(
        default=0.0,
        metadata={
            "help": "Pre-allocate one specific-capacity empty tensor "
            "and release it for avoiding memory fragmentation"
        },
    )

    moe_group: Optional[str] = field(
        default="dp",
        metadata={
            "help": "The communication group of moe currently supports `dp|sharding|mp|dummy`"
        },
    )
    use_moe: Optional[bool] = field(
        default=False, metadata={"help": "Temporary alternative to expert parallelism."}
    )
    moe_use_all2all: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use the all2all communication method."},
    )
    log_global_grad_norm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Print the global gradient norm, which only takes effect when `enable_global_training_logs` is enabled.."
        },
    )

    multi_token_pred_depth: Optional[int] = field(
        default=0,
        metadata={},
    )

    lr_scheduler: str = field(
        default="cosine",
        metadata={
            "help": "The scheduler type to use. suppor linear, cosine, constant, constant_with_warmup"
        },
    )

    moe_gate_lr_ratio: float = field(
        default=None,
        metadata={
            "help": (
                "When enabling MoE, apply special handling to the learning rate (LR) of the gate/router."
            )
        },
    )
    vit_lr_ratio: float = field(
        default=None,
        metadata={
            "help": (
                "When enabling ViT training, apply special handling to the learning rate (LR) of ViT."
            )
        },
    )

    pipeline_schedule_mode: str = field(
        default="1F1B",
        metadata={"help": "The pipeline schedule mode, support 1F1B and VPP"},
    )
    virtual_pipeline_seg_method: str = field(
        default="ErnieDecoderLayerAuto",
        metadata={"help": "The seg method of spliting pp layer for virtual pipeline."},
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
        return self.pipeline_parallel_rank == 0 and self.tensor_parallel_rank == 0

    @property
    def reeao_dataset_world_size(self):
        return super().dataset_world_size

    def __post_init__(self):
        super().__post_init__()

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

        self.max_gradient_accumulation_steps = self.gradient_accumulation_steps

        if self.pipeline_parallel_degree > 1:
            self.per_device_eval_batch_size = (
                self.per_device_train_batch_size * self.gradient_accumulation_steps
            )
            logger.warn(
                f"eval_batch_size set to {self.per_device_eval_batch_size} in Pipeline Parallel!"
            )
            user_defined_strategy = fleet.fleet._user_defined_strategy
            user_defined_strategy.strategy.pipeline_configs.accumulate_steps = (
                self.gradient_accumulation_steps
            )

            self.max_gradient_accumulation_steps = self.gradient_accumulation_steps
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


class AutoPretrainingTrainer(AutoTrainer):

    def __init__(self, args=None, model=None, callbacks=[], **kwargs):
        callbacks = [
            LoggingCallback(),
            StopperCallback(),
            TensorBoardCallback(
                args, model=model, log_tokens_per_step=True, log_flops_per_step=False
            ),
        ] + callbacks

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
            pp_broadcast=True,
        )

        train_dataset = self.train_dataset
        if self._is_iterable_dataset(train_dataset):
            return DataLoader(
                train_dataset,
                batch_size=None,
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

    def _maybe_log_save_evaluate(
        self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs
    ):
        super()._maybe_log_save_evaluate(
            tr_loss, model, epoch, ignore_keys_for_eval, **kwargs
        )
        return

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
        optimizer_params = self.model.parameters()
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

            if (
                self.args.use_moe
                and not self.args.use_hybrid_parallel
                and not self.args.enable_auto_parallel
            ):
                logger.info("using moe Global clip")

                def expert_fn(p):
                    return getattr(p, "no_sync", False)

                grad_clip = ClipGradForMOEByGlobalNormAuto(
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
                    lr_ratio_fn if self.args.moe_gate_lr_ratio is not None else None
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
