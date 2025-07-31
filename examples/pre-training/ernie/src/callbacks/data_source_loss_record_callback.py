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

import os
from types import MethodType

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddleformers.trainer.trainer import PREFIX_CHECKPOINT_DIR
from paddleformers.trainer.trainer_utils import IntervalStrategy
from paddleformers.trainer.training_args import TrainingArguments
from paddleformers.trainer.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

try:
    from paddleformers.trainer.trainer import (
        PADDLE_WEIGHT_FILE_NAME as PADDLE_WEIGHTS_NAME,
    )
except ImportError:
    pass


class DataSourseLossRecordCallback(TrainerCallback):
    """逐part打印loss. 用于分析数据loss值

    Args:
        TrainerCallback (_type_): _description_
    """

    def __init__(self, args, model):
        """_summary_

        Args:
            args (_type_): _description_
            model (_type_): _description_

        Returns:
            _type_: _description_
        """
        if args.use_hybrid_parallel:
            hcg = fleet.get_hybrid_communicate_group()
            self.mp_rank = hcg.get_model_parallel_rank()
            if args.pipeline_parallel_degree > 1:
                if args.pp_need_data_degree > 1:
                    assert args.pp_need_data_degree == args.pipeline_parallel_degree
                self.pp_rank = model._stage_id
                self.pp_degree = model._num_stages
                self.pp_group = hcg.get_pipe_parallel_group()
                self.do_record = (
                    self.pp_rank == self.pp_degree - 1 and self.mp_rank == 0
                )
            else:
                self.do_record = self.mp_rank == 0
        else:
            self.do_record = True
        if not self.do_record:
            return

        self.data_source_loss = {"data_id": [], "loss": []}

        if args.pipeline_parallel_degree > 1:
            origin_criterion_fwd = model._loss_fn[0].forward
        else:
            origin_criterion_fwd = model.criterion.forward

        def criterion_forward_wrapper(_self, *args, **kwargs):
            loss = origin_criterion_fwd(*args, **kwargs)
            if isinstance(loss, (tuple, list)):
                self.data_source_loss["loss"].append(loss[0].item())
            else:
                self.data_source_loss["loss"].append(loss.item())
            return loss

        if args.pipeline_parallel_degree > 1:
            model._loss_fn[0].forward = MethodType(
                criterion_forward_wrapper, model._loss_fn[0]
            )
        else:
            model.criterion.forward = MethodType(
                criterion_forward_wrapper, model.criterion
            )

    def on_load_data_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        inputs,
        **kwargs,
    ):
        """Trainer load_data_end 时调用

        Args:
            args (TrainingArguments): _description_
            state (TrainerState): _description_
            control (TrainerControl): _description_
            inputs (_type_): _description_
        """
        src_id_list, data_id_list = [], []
        if args.pp_need_data_degree > 0 and self.mp_rank == 0:
            dist.gather(
                inputs["src_id"].cuda(),
                src_id_list,
                self.pp_group.ranks[-1],
                self.pp_group,
            )
            dist.gather(
                inputs["data_id"].cuda(),
                data_id_list,
                self.pp_group.ranks[-1],
                self.pp_group,
            )
            src_id_list = sum([], src_id_list)
            data_id_list = sum([], data_id_list)
        else:
            src_id_list = inputs["src_id"]
            data_id_list = inputs["data_id"]
        if not self.do_record:
            return

        for part_id, data_id in zip(src_id_list, data_id_list):
            self.data_source_loss["data_id"].append((part_id.item(), data_id.item()))

    def on_save(self, args, state, control, model, **kwargs):
        """_summary_

        Args:
            args (_type_): _description_
            state (_type_): _description_
            control (_type_): _description_
            model (_type_): _description_
        """
        if not self.do_record:
            # mp0 and pp -1
            return

        if not (
            args.save_strategy == IntervalStrategy.STEPS
            and args.save_steps > 0
            and state.global_step % args.save_steps == 0
        ):
            # maybe zcc
            return

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        output_dir = os.path.join(args.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)
        paddle.save(
            self.data_source_loss,
            os.path.join(output_dir, f"data_source_loss.{args.dataset_rank:04d}.bin"),
        )
        self.data_source_loss = {"data_id": [], "loss": []}
