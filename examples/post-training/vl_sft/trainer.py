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
Trainer for Ernie-MoE VL model with enhanced distributed training support.
"""

import math
import os
import shutil
import sys
import time
from functools import partial
from typing import List, Optional, Union

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet.meta_optimizers.dygraph_optimizer
from tqdm import tqdm
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
    HybridParallelOptimizer,
)
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
)
from paddle.io import DataLoader

try:
    from paddle.distributed.fleet.utils.hybrid_parallel_util import (
        obtain_optimizer_parameters_list,
    )

    _obtain_optimizer_parameters_list = obtain_optimizer_parameters_list
except Exception:
    try:
        from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
            _obtain_optimizer_parameters_list,
        )
    except Exception:
        _obtain_optimizer_parameters_list = None

from distutils.util import strtobool

from paddleformers.peft import LoRAModel, PrefixModelForCausalLM
from paddleformers.trainer import (
    speed_metrics,
)
from paddleformers.trainer.trainer import (
    PADDLE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
)
from paddleformers.trainer.trainer_callback import TrainerState
from paddleformers.trainer.trainer_utils import (
    TrainOutput,
    has_length,
)
from paddleformers.trainer.utils import reshard as reshard_util
from paddleformers.utils.log import logger

# from ernie.unified_checkpoint import load_unified_checkpoint
from paddleformers.trainer.utils.helper import (  # nested_truncate,
    distributed_file,
    distributed_isfile,
)
from paddleformers.transformers.context_parallel_utils import (
    split_inputs_sequence_dim_load_balance,
)
from paddleformers.transformers.model_utils import _add_variant
from paddleformers.transformers.segment_parallel_utils import split_inputs_sequence_dim
from paddleformers.utils.batch_sampler import DistributedBatchSampler
from paddleformers.utils.batch_sampler import (
    DistributedBatchSampler as NlpDistributedBatchSampler,
)

# from paddlenlp.utils.env import PADDLE_WEIGHTS_NAME
from pretraining_trainer import PretrainingTrainer

from ernie.dataset.dist_data_loader import DistDataLoader
from paddle.io import Dataset


class EmptyDataset(Dataset):
    """EmptyDataset"""

    def __init__(self):
        super().__init__()

    def __getitem__(self, idx):
        raise IndexError("Empty dataset")

    def __len__(self):
        return 100


class SFTTrainer(PretrainingTrainer):
    """
    The main trainer class which handles all the logic necessary for fine-tuning models.
    """

    def __init__(
        self,
        _shit=None,
        args=None,
        model=None,
        callbacks=None,
        is_train_text=False,
        is_train_mm=True,
        text_sft_dataset=None,
        modality_ratio=[1, 1],
        **kwargs,
    ):
        super().__init__(
            _shit=_shit, args=args, model=model, callbacks=callbacks, **kwargs
        )
        self.is_train_text = is_train_text
        self.text_sft_dataset = text_sft_dataset
        self.modality_ratio = modality_ratio
        self.is_train_mm = is_train_mm

    def get_train_dataloader(self):
        """get train data loader"""
        if self.args.need_data and self.train_dataset is None:
            self.train_dataset = EmptyDataset()
        # `pp_need_data`，data bradcast in model
        _DataLoader = (
            partial(
                DistDataLoader,
                need_data=self.args.need_data,
                pp_broadcast=not self.args.pp_need_data_degree,
            )
            if (
                self.args.tensor_parallel_degree > 1
                or self.args.pipeline_parallel_degree > 1
            )
            else DataLoader
        )  # use `DistDataLoader` before init fleet
        train_dataset = self.train_dataset

        train_sampler = None
        return _DataLoader(
            train_dataset,
            tokenizer=self.tokenizer,
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            prefetch_factor=self.args.prefetch_factor,
            is_train_text=self.is_train_text,
            text_sft_dataset=self.text_sft_dataset,
            need_data=self.args.need_data,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            modality_ratio=self.modality_ratio,
            is_train_mm=self.is_train_mm,
        )

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
        """
        args = self.args
        self.is_in_train = True

        logger.info(
            f"Starting training from resume_from_checkpoint : {resume_from_checkpoint}"
        )

        # The resume_from_checkpoint could be None in some machine node.
        # Here we reset None to temp directory.
        if args.world_size > 1:
            is_resume_from_checkpoint = paddle.to_tensor(
                [resume_from_checkpoint is not None], dtype="int32"
            )
            paddle.distributed.all_reduce(is_resume_from_checkpoint)
            is_resume_from_checkpoint = is_resume_from_checkpoint.item()
            if (
                is_resume_from_checkpoint > 0
                and is_resume_from_checkpoint < paddle.distributed.get_world_size()
            ):
                if resume_from_checkpoint is None:
                    resume_from_checkpoint = os.path.join(
                        self.args.output_dir, "local_tempdir"
                    )
                    if (
                        os.path.exists(resume_from_checkpoint)
                        and self.args.local_rank == 0
                    ):
                        shutil.rmtree(resume_from_checkpoint)
                    os.makedirs(resume_from_checkpoint, exist_ok=True)
                    logger.info(
                        f"Reset resume_from_checkpoint to temp directory : {resume_from_checkpoint}"
                    )

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        if not self.args.should_load_sharding_stage1_model:
            self._load_from_checkpoint(resume_from_checkpoint)

        train_dataloader = self.get_train_dataloader()

        total_train_batch_size = (
            args.train_batch_size
            * args.gradient_accumulation_steps
            * args.dataset_world_size
        )
        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = (
                len(train_dataloader) // args.gradient_accumulation_steps
            )
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = len(self.train_dataset)

            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = int(len(self.train_dataset) * args.num_train_epochs)

            if args.minimum_eval_times is not None and args.minimum_eval_times > 0:
                if max_steps // args.eval_steps < args.minimum_eval_times:
                    exp_step = max_steps / args.minimum_eval_times
                    exp_step = max(int(exp_step - exp_step % 10), 10)
                    logger.info(
                        "Reset eval step by minimum_eval_times to %d" % exp_step
                    )
                    args.eval_steps = exp_step
        elif (
            args.max_steps > 0
        ):  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                f"""args.max_steps must be set to a positive value
                if dataloader does not have a length, was {args.max_steps}"""
            )

        # delay_optimizer_creation = (
        #     self.sharding is not None
        #     and ShardingOption.SHARD_OP in self.args.sharding
        # )
        delay_optimizer_creation = False

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()

        if self.args.should_load_sharding_stage1_model:
            model = self._wrap_model_and_load_sharded_checkpoint(resume_from_checkpoint)

        elif self.args.should_save_sharding_stage1_model:
            # In the non-sharded mode, should invoke _load_from_checkpoint before _wrap_model.
            # In this mode, the rank0 load all params and the _wrap_model
            # implicitly broadcast params from rank0 to the other ranks.
            model = self._wrap_model(self.model_wrapped)
            if self.sharding_io is not None:
                assert (
                    delay_optimizer_creation is False
                ), "delay_optimizer_creation should be False"
                # the self.optimizer should be wrapped and it is done in _wrap_model
                self.sharding_io.set_optimizer(self.optimizer)
            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model
            if delay_optimizer_creation:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)
            self._load_optimizer_and_scheduler(resume_from_checkpoint)
        else:
            model = self._wrap_model(self.model_wrapped)
            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model
            if delay_optimizer_creation:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)
            self._load_optimizer_and_scheduler(resume_from_checkpoint)

        logger.info(f"{self.runtime_timer.log()}")

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples*args.pp_need_data_degree:,}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size*args.pp_need_data_degree}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps*args.pp_need_data_degree}"
        )
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(
            f"  Total num train samples = {num_train_samples*args.pp_need_data_degree:,}"
        )
        # per_device_trainable_numel = sum(p.numel().item() for p in model.parameters() if not p.stop_gradient)
        # TODO: Temporary fix since Tensor.numel() not supported in distributed mode
        per_device_trainable_numel = sum(
            np.prod(p.shape) for p in model.parameters() if not p.stop_gradient
        )
        logger.debug(
            f"  Number of trainable parameters = {per_device_trainable_numel:,} (per device)"
        )
        if self.args.use_hybrid_parallel:
            # todo fix for pipeline_parallel_degree
            parts_num = max(self.args.tensor_parallel_degree, 1) * max(
                self.args.pipeline_parallel_degree, 1
            )
            if parts_num > 1:
                all_reduce_dtype = "int64"
                if paddle.get_device().split(":")[0] in ["npu", "xpu"]:
                    # TODO(duanyanhui): fix when NPU all_reduce supports int64
                    all_reduce_dtype = "float32"
                trainable_numel_tensor = paddle.to_tensor(
                    per_device_trainable_numel, dtype=all_reduce_dtype
                )
                paddle.distributed.all_reduce(trainable_numel_tensor)
                trainable_numel = (
                    int(trainable_numel_tensor.item()) // self.args.dataset_world_size
                )
                if self.args.sep_parallel_degree > 0:
                    trainable_numel = trainable_numel // self.args.sep_parallel_degree
                if self.args.context_parallel_degree > 0:
                    trainable_numel = (
                        trainable_numel // self.args.context_parallel_degree
                    )
                # the numel is roughly,
                # because the tensor parallel still hold own bias or layer_norm weight without splited
                # so, the trainable numel is a little bigger than real.
                logger.debug(
                    f"  Number of trainable parameters = {trainable_numel:,} (all devices, roughly)"
                )

        return self._inner_training_loop(
            args,
            model,
            train_dataloader,
            len_dataloader,
            max_steps,
            num_train_epochs,
            num_update_steps_per_epoch,
            num_train_samples,
            resume_from_checkpoint,
            ignore_keys_for_eval,
        )

    def _inner_training_loop(
        self,
        args,
        model,
        train_dataloader,
        len_dataloader,
        max_steps,
        num_train_epochs,
        num_update_steps_per_epoch,
        num_train_samples,
        resume_from_checkpoint,
        ignore_keys_for_eval,
    ):
        start_time = time.time()
        self._globalstep_last_start_time = time.time()
        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if (
            resume_from_checkpoint is not None
            and distributed_isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            and not self.args.ignore_load_lr_and_optim
        ):
            self.state = TrainerState.load_from_json(
                distributed_file(
                    os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
                )
            )
            if self.args.world_size > 1:
                global_step_list = []
                paddle.distributed.all_gather(
                    global_step_list,
                    paddle.to_tensor([self.state.global_step], dtype="int64"),
                )
                assert (
                    paddle.sum(paddle.stack(global_step_list) - global_step_list[0])
                    == 0
                ), f"Error, get different globel step, please check! step list: {[x.item() for x in global_step_list]}"

            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(
                        total=steps_trained_in_current_epoch
                    )
                    steps_trained_progress_bar.set_description(
                        "Skipping the first batches"
                    )
            if not args.ignore_data_skip:
                if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                    train_dataloader.batch_sampler, NlpDistributedBatchSampler
                ):
                    consumed_samples = (
                        self.state.global_step
                        * args.train_batch_size
                        * args.gradient_accumulation_steps
                        * args.dataset_world_size
                    )
                    train_dataloader.batch_sampler.set_epoch(
                        consumed_samples=consumed_samples
                    )
                    logger.info(
                        f"Set DistributedBatchSampler consumed_samples to {consumed_samples}"
                    )

        epoch_iterator = train_dataloader
        # steps_in_epoch = len(epoch_iterator)
        steps_in_epoch = (
            len(epoch_iterator)
            if len_dataloader is not None
            else args.max_steps * args.gradient_accumulation_steps
        )
        if len_dataloader is not None:
            if self.args.gradient_accumulation_steps > len(epoch_iterator):
                logger.warning(
                    f"""changing accumulation step from `{self.args.gradient_accumulation_steps}`
                    to `{len(epoch_iterator)}` to avoid, cross epoch accumulate"""
                )
                self.args.gradient_accumulation_steps = len(epoch_iterator)

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader

        self.state.max_steps = int(max_steps)
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        tr_loss = paddle.to_tensor(0.0)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        if self.args.device == "npu" and self.args.flatten_param_grads:
            from .plugins.npu_plugin import npu_accelerate_plugin

            npu_accelerate_plugin(self.optimizer)

        if self.args.ignore_data_skip:
            self.timers and self.timers("read-data").start()

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                train_dataloader.batch_sampler, DistributedBatchSampler
            ):
                train_dataloader.batch_sampler.set_epoch(epoch)

            step_control = 0  # used in loop control, reset to 0 after every step
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            for step, inputs in enumerate(epoch_iterator):
                if self.args.use_hybrid_parallel and self.args.sep_parallel_degree > 1:
                    inputs = split_inputs_sequence_dim(inputs)
                if (
                    self.args.use_hybrid_parallel
                    and self.args.context_parallel_degree > 1
                ):
                    inputs = split_inputs_sequence_dim_load_balance(inputs)
                if self.args.ignore_data_skip:
                    self.timers and self.timers("read-data").stop()

                os.environ["TRAINER_GLOBAL_STEP"] = str(self.state.global_step)
                self.callback_handler.on_load_data_end(
                    args, self.state, self.control, inputs=inputs
                )

                # Skip past any already trained steps if resuming training
                # for paddlenlp.utils.batch_sampler.DistributedBatchSampler
                # We use consumed_samples to reset the status
                if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                    train_dataloader.batch_sampler, NlpDistributedBatchSampler
                ):
                    if step == 0:
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(
                                steps_trained_in_current_epoch
                            )
                            steps_trained_progress_bar.close()
                            steps_trained_progress_bar = None
                        self._load_rng_state(resume_from_checkpoint)
                    step += steps_trained_in_current_epoch
                elif steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step_control % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )
                    self.timers and self.timers("forward-backward").start()

                # stage2 and stage3 should not no_sync, because the is no DDP wrapper and no_sync API
                # hybrid_parallel (tp or pp or sharding stage 1) should not no_sync
                availiable_no_sync = hasattr(model, "no_sync")
                is_no_sync = (
                    (
                        ((step_control + 1) % args.gradient_accumulation_steps != 0)
                        and args._no_sync_in_gradient_accumulation
                    )
                    or args.recompute
                    or args.use_expert_parallel
                ) and availiable_no_sync
                # sharding
                # stage1. the same as ddp
                # stage2. manualy collect gradient on dp group

                dp_master_grad = (
                    self.args.world_size > 1
                    and self.args.amp_master_grad
                    and not self.args.use_hybrid_parallel
                )
                if dp_master_grad:
                    is_no_sync = True

                if is_no_sync:
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)
                tr_loss += tr_loss_step

                def fused_allreduce_gradients_no_sync(paramlist, hcg):
                    paramlist = list(paramlist)
                    nonmoe_list = [
                        p for p in paramlist if not getattr(p, "no_sync", False)
                    ]
                    moelist = [p for p in paramlist if getattr(p, "no_sync", False)]
                    if moelist and not self.args.use_expert_parallel:
                        logger.warning(
                            "found `no sync` param when `use_expert_parallel=False`"
                        )
                    fused_allreduce_gradients(nonmoe_list, hcg)

                if (step_control + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    if (
                        self.args.pipeline_parallel_degree <= 1
                        and self._enable_delay_scale_loss()
                    ):
                        tr_loss /= self.args.gradient_accumulation_steps

                    # assert if loss is invalid
                    self._check_loss_valid(tr_loss)

                    self.timers and self.timers("forward-backward").stop()
                    # Maunally collect gradients
                    # Case 1: Use recompute and dp
                    # Case 2: Hack dp with master_grad
                    # Case 3: Pipeline or sharding overlap
                    # local_rank != -1 don't means dp in networks.
                    self.timers and self.timers("all-reduce").start()

                    # Case 1: Use recompute and dp / sharding stage1,
                    # manualy collect gradient for dp.
                    if (
                        args.recompute or args.use_expert_parallel
                    ) and availiable_no_sync:
                        fused_allreduce_gradients_no_sync(
                            list(model.parameters()), None
                        )

                    # Case 2: hack dp with master_grad
                    elif dp_master_grad:
                        fused_allreduce_gradients_no_sync(
                            list(model.parameters()), None
                        )

                    # Pipeline parallel mode,  handle gradient reduce here to overlap
                    pipeline_parallel_config = (
                        set(args.pipeline_parallel_config.split(" "))
                        if args.pipeline_parallel_degree > 1
                        else set()
                    )
                    sharding_parallel_config = (
                        set(args.sharding_parallel_config.split(" "))
                        if args.sharding_parallel_degree > 1
                        else set()
                    )
                    enable_dp_comm_overlap = (
                        "enable_dp_comm_overlap" in pipeline_parallel_config
                    )
                    enable_release_grads = (
                        "enable_release_grads" in pipeline_parallel_config
                        or "enable_release_grads" in sharding_parallel_config
                    )

                    # Case 3: Pipeline parallel mode, overlap with dp
                    if (
                        isinstance(self.optimizer, HybridParallelOptimizer)
                        and not self.do_grad_scaling
                    ):
                        parameters_list = _obtain_optimizer_parameters_list(
                            self.optimizer._inner_opt
                        )

                        if not enable_dp_comm_overlap:
                            if self.optimizer._sharding_enable:
                                assert reshard_util.is_sharding_opt(self.optimizer)
                                self.optimizer._inner_opt.reduce_gradients(
                                    list(parameters_list), self.optimizer._hcg
                                )

                            if self.optimizer._dp_enable or getattr(
                                self.optimizer, "_sep_enable", False
                            ):
                                fused_allreduce_gradients_no_sync(
                                    list(parameters_list), self.optimizer._hcg
                                )
                    self.timers and self.timers("all-reduce").stop()
                    self.timers and self.timers("optimizer-step").start()

                    if (
                        self.args.gradient_accumulation_steps > 1
                        and self._enable_delay_scale_loss()
                    ):
                        paddle.device.synchronize()
                        for p in model._layers.parameters():
                            with paddle.no_grad():
                                if hasattr(p, "main_grad") and p.main_grad is not None:
                                    assert p.grad is None
                                    p.main_grad.scale_(
                                        1.0 / self.args.gradient_accumulation_steps
                                    )
                                elif p.grad is not None:
                                    p.grad.scale_(
                                        1.0 / self.args.gradient_accumulation_steps
                                    )

                    # Optimizer step
                    self.callback_handler.on_optimizer_begin(
                        args,
                        self.state,
                        self.control,
                        scaler=self.scaler if self.do_grad_scaling else None,
                    )

                    optimizer_was_run = True

                    if self.args.offload_optim:
                        # logger.info("SFT-Trainier starts to reload Optimizer.")
                        self._reload_optimizer()
                        # logger.info("SFT-Trainier reloads Optimizer done.")

                    if self.do_grad_scaling:
                        if args.pipeline_parallel_degree > 1:
                            assert (
                                not self.args.use_expert_parallel
                            ), "pipeline moe not work under fp16"
                        scale_before = paddle.assign(self.scaler._scale)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler._scale
                        # Compatible with paddlepaddle 2.6.0 using typo word.
                        if hasattr(self.scaler, "_cache_founf_inf"):
                            optimizer_was_run = not self.scaler._cache_founf_inf
                        else:
                            optimizer_was_run = not self.scaler._cache_found_inf
                        if not optimizer_was_run:
                            scale_before_value = scale_before.cpu().numpy()
                            scale_after_value = scale_after.cpu().numpy()
                            logger.warning(
                                f"""optimizer not run, scale_before: {scale_before_value[0]},
                                scale_after: {scale_after_value[0]}"""
                            )
                    elif isinstance(self.optimizer, HybridParallelOptimizer):
                        self.optimizer._step(parameters_list)
                    else:
                        self.optimizer.step()

                    if self.args.offload_optim:
                        # logger.info("SFT-Trainier starts to offload Optimizer.")
                        self._offload_optimizer()
                        # logger.info("SFT-Trainier offloads Optimizer done.")

                    self.timers and self.timers("optimizer-step").stop()

                    if optimizer_was_run:
                        self.lr_scheduler.step()

                    if enable_release_grads:
                        self.optimizer.clear_grad(set_to_zero=False)
                        if args.pipeline_parallel_degree > 1:
                            for _, buffers in model._chunk_2_comm_buffers.items():
                                for buffer in buffers:
                                    buffer._clear_grad_storage()
                    else:
                        self.optimizer.clear_grad()

                    self.callback_handler.on_optimizer_end(
                        args,
                        self.state,
                        self.control,
                        scaler=self.scaler if self.do_grad_scaling else None,
                    )

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control
                    )
                    self._maybe_log_save_evaluate(
                        tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs
                    )
                    self._print_timer()
                    step_control = 0
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )
                    step_control += 1

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

                if self.args.ignore_data_skip:
                    self.timers and self.timers("read-data").start()

            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({self.state.max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control
            )
            self._maybe_log_save_evaluate(
                tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs
            )

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\nTraining completed. \n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            if args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, LoRAModel) or isinstance(
                self.model, PrefixModelForCausalLM
            ):
                self._load_best_model_from_peft_checkpoint()
            else:
                weight_name = PADDLE_WEIGHTS_NAME
                best_model_path = os.path.join(
                    self.state.best_model_checkpoint,
                    _add_variant(weight_name, self.args.weight_name_suffix),
                )
                if os.path.exists(best_model_path):
                    # We load the model state dict on the CPU to avoid an OOM error.
                    state_dict = paddle.load(best_model_path, return_numpy=True)
                    # If the model is on the GPU, it still works!
                    self._set_state_dict_in_model(state_dict)
                else:
                    logger.warning(
                        f"Could not locate the best model at {best_model_path}, "
                        "if you are running a distributed training "
                        "on multiple nodes, you should activate `--save_on_each_node`."
                    )

        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
        )

        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        kwargs = {
            "metrics_dumper": self.metrics_dumper,
        }
        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control, **kwargs
        )

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def offload_reload_master_weights(self, action):
        """offload or reload master weights"""
        assert action in ["offload", "reload"], action
        for key, value in self.optimizer._master_weights.items():
            if action == "offload":
                value = value.pin_memory()
            else:
                value = value.cuda()
            self.optimizer._master_weights[key] = value

    def reload(self, exclude_opt=True):
        """reload model parameters and optimizers"""
        if strtobool(os.getenv("FLAGS_use_cuda_managed_memory", "False")):
            logger.warning(
                "FLAGS_use_cuda_managed_memory has been set to True, "
                "offloading strategy is ineffective."
            )
            return

        if not exclude_opt:
            # reload moment1
            for key, value in self.optimizer._accumulators[
                self.optimizer._moment1_acc_str
            ].items():
                self.optimizer._accumulators[self.optimizer._moment1_acc_str][
                    key
                ] = value.cuda()

            # reload moment2
            for key, value in self.optimizer._accumulators[
                self.optimizer._moment2_acc_str
            ].items():
                self.optimizer._accumulators[self.optimizer._moment2_acc_str][
                    key
                ] = value.cuda()

            # reload master_weight
            for key, value in self.optimizer._master_weights.items():
                self.optimizer._master_weights[key] = value.cuda()

            for key, value in self.optimizer._accumulators_holder.items():
                self.optimizer._accumulators_holder[key] = value.cuda()

        self.model.to(paddle.device.get_device())

    def offload(self, exclude_param=False, exclude_opt=True):
        """offload model parameters to CPU"""
        if strtobool(os.getenv("FLAGS_use_cuda_managed_memory", "False")):
            logger.warning(
                "FLAGS_use_cuda_managed_memory has been set to True, "
                "offloading strategy is ineffective."
            )
            return

        if not exclude_opt:

            # offload moment1
            for key, value in self.optimizer._accumulators[
                self.optimizer._moment1_acc_str
            ].items():
                self.optimizer._accumulators[self.optimizer._moment1_acc_str][
                    key
                ] = value.cpu()

            # offload moment2
            for key, value in self.optimizer._accumulators[
                self.optimizer._moment2_acc_str
            ].items():
                self.optimizer._accumulators[self.optimizer._moment2_acc_str][
                    key
                ] = value.cpu()

            # offload master_weight
            for key, value in self.optimizer._master_weights.items():
                self.optimizer._master_weights[key] = value.cpu()

            for key, value in self.optimizer._accumulators_holder.items():
                self.optimizer._accumulators_holder[key] = value.cpu()

        if not exclude_param:
            self.model.to(paddle.CUDAPinnedPlace())
