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

"""PretrainingTrainer"""

__all__ = [
    "PretrainingTrainer",
]


import sys
import contextlib
from typing import Optional
from dataclasses import dataclass, field
import time
import math
import logging


import paddle
import paddle.nn as nn
import paddle.amp.auto_cast as autocast

from paddleformers.trainer import (
    speed_metrics,
)

from paddleformers.trainer.auto_trainer import AutoTrainer
from paddleformers.trainer import AutoTrainingArguments


from paddleformers.trainer.utils import add_start_docstrings
from paddleformers.trainer.trainer_callback import PrinterCallback
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.pipelining.schedules import get_pipeline_schedule
from typing import Any, Dict, Union
import paddle.distributed as dist
from .callbacks import TensorBoardCallback
from ernie.utils.training_utils import reset_per_device_batch_size
from ernie.callbacks import (
    LoggingCallback,
    StopperCallback,
)
from ernie.lr_schedulers import (
    get_cosine_schedule_with_warmup,
    get_wsd_schedule_with_warmup,
)
from datasets import DistDataLoaderErnie


logger = logging.getLogger(__name__)


@dataclass
@add_start_docstrings(AutoTrainingArguments.__doc__)
class PreTrainingArguments(AutoTrainingArguments):

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

    virtual_pp_degree: Optional[int] = field(
        default=1,
        metadata={
            "help": "vpp",
        },
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
            "help": "The scheduler type to use. support linear, cosine, constant, constant_with_warmup"
        },
    )

    decay_function: str = field(
        default="half_life",
        metadata={
            "help": "The decay function for WSD LR scheduler. support half_life(default), 1-sqrt"
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
        default="ErnieDecoderLayer",
        metadata={"help": "The seg method of spliting pp layer for virtual pipeline."},
    )

    model_type: Optional[str] = field(
        default="ernie",
        metadata={"help": "Only support for ernie pre-training for now."},
    )
    moe_use_aux_free_update_coef: float = field(
        default=1.0e-3,
        metadata={"help": "moe aux free update coef"},
    )
    offload_optimizer: bool = field(
        default=False,
        metadata={
            "help": "Offload optimizer states to CPU, and reload them during optimizer update"
        },
    )

    @property
    def need_data(self):
        return self.pipeline_parallel_rank == 0 and self.tensor_parallel_rank == 0

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


class PretrainingTrainer(AutoTrainer):

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
        if self.args.pipeline_parallel_degree > 1:
            # NOTE: 1F1B version pipeline use intermediate_api instead of pp_schedule, however intermediate_api only support FThenB for now, VPP and 1F1B still need using pp_schedule
            if not (
                self.args.use_intermediate_api
                and self.args.pipeline_schedule_mode == "FThenB"
            ):
                if self.criterion is None:
                    self.criterion = self.model.criterion
                self.pp_schedule = get_pipeline_schedule(
                    model,
                    self.args.gradient_accumulation_steps,
                    self.criterion,
                    self.args.pipeline_schedule_mode,
                    self.args.pipeline_parallel_degree,
                    self.comm_group_in_pp,
                )
                self.args.per_device_train_batch_size = (
                    self.args.per_device_train_batch_size
                    * self.args.gradient_accumulation_steps
                )
                self.args.gradient_accumulation_steps = 1

    def compute_pipeline_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.criterion is not None:
            if "labels" in inputs:
                labels = inputs.pop("labels")
            elif "start_positions" in inputs and "end_positions" in inputs:
                labels = (inputs.pop("start_positions"), inputs.pop("end_positions"))
            elif self.args.label_names is not None:
                labels = []
                for label in self.label_names:
                    labels.append(inputs.pop(label))
                labels = tuple(labels)
            elif "generator_labels" in inputs:
                labels = inputs["generator_labels"]
        else:
            labels = None

        pp_rank = self.comm_group_in_pp.rank
        losses = []
        if pp_rank == 0:
            self.pp_schedule.step(**inputs)
        elif pp_rank == self.args.pipeline_parallel_degree - 1:
            self.pp_schedule.step(target=labels, losses=losses)
        else:
            self.pp_schedule.step()

        final_loss = None
        if len(losses) != 0:
            final_loss = paddle.stack(losses).mean()

        return final_loss

    def dynamic_pipeline_training(
        self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]
    ) -> paddle.Tensor:
        assert (
            self.args.pipeline_parallel_degree > 1
        ), "pipeline_parallel_degree must be greater than 1."
        with self.autocast_smart_context_manager():
            loss = self.compute_pipeline_loss(model, inputs)

        return loss

    def dynamic_training(
        self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]
    ) -> paddle.Tensor:
        if self.args.pipeline_parallel_degree > 1:
            if not (
                self.args.use_intermediate_api
                and self.args.pipeline_schedule_mode == "FThenB"
            ):
                return self.dynamic_pipeline_training(model, inputs)
            else:
                return super().dynamic_training(model, inputs)
        else:
            return super().dynamic_training(model, inputs)

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

    def get_train_dataloader(self):

        if self.args.need_data and self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if self.args.need_data:
            train_sampler = self._get_train_sampler()
        else:
            train_sampler = None
        return DistDataLoaderErnie(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            prefetch_factor=self.args.prefetch_factor,
        )

    def create_scheduler(self, num_training_steps):

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
        optimizer_params = self.model.parameters()
        if self.optimizer is None:

            def need_decay(name):
                return not any(nd in name for nd in ["bias", "norm"])

            decay_parameters = [
                p.name for n, p in self.model.named_parameters() if need_decay(n)
            ]

            def apply_decay_param_fun(x):
                return x in decay_parameters

            optimizer_cls, optimizer_kwargs = AutoTrainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            grad_clip = (
                nn.ClipGradByGlobalNorm(self.args.max_grad_norm)
                if self.args.max_grad_norm > 0
                else None
            )

            def lr_ratio_fn(param):
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

        return self.optimizer

    def _wrap_for_dist_loader(self, train_dataloader, dense_tensor_idx=None):
        self.dense_tensor_idx = dense_tensor_idx
        dist_loader = dist.shard_dataloader(
            dataloader=train_dataloader,
            meshes=self._get_meshes_for_loader(),
            shard_dims="dp",
            dense_tensor_idx=dense_tensor_idx,
            is_dataset_splitted=True,
        )
        return dist_loader
