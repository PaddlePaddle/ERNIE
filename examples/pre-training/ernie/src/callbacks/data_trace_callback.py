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

import logging

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddleformers.trainer.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from paddleformers.trainer.training_args import TrainingArguments

logger = logging.getLogger(__name__)


class DataTraceCallback(TrainerCallback):
    """Callback 用于DataStatus记录

    Args:
        TrainerCallback (_type_): _description_
    """

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        训练开始时，初始化或更新 TrainerState 中关于数据加载状态的参数。

        Args:
            args (TrainingArguments): 训练参数。
            state (TrainerState): 训练状态。
            control (TrainerControl): 训练控制。
            **kwargs: 其他可选参数。

        Returns:
            None: 无返回值，直接修改 state 中的参数。

        """
        if args.custom_data_status:
            custom_trainer_state = TrainerState.load_from_json(args.custom_data_status)
            logger.info(f"load custom data status from {args.custom_data_status}")
            state.trial_params = custom_trainer_state.trial_params

        if not args.need_data:
            self.data_status_shape = paddle.zeros([1], dtype="int32")
            if dist.is_initialized():
                logger.info("broadcast data trace callback hook")
                dist.broadcast(self.data_status_shape, 0)  # 呼应 Line:117
            return
        batch_sampler = kwargs["train_dataloader"].batch_sampler

        if state.trial_params is None:
            state.trial_params = {}

        if "saved_data_status" not in state.trial_params:
            state.trial_params["saved_data_status"] = [
                0 for _ in range(batch_sampler.max_part_id + 1)
            ]

        if "last_start_data_status" not in state.trial_params:
            state.trial_params["last_start_data_status"] = [
                0 for _ in state.trial_params["saved_data_status"]
            ]

        # 兼容之前的训练
        if "consumed_samples" not in state.trial_params:
            state.trial_params["consumed_samples"] = sum(
                state.trial_params["saved_data_status"]
            )
        if "global_shuffle_seed" not in state.trial_params:
            state.trial_params["global_shuffle_seed"] = 0

        if not args.same_data:
            # 数据有变化 重置起始data_status, consumed_samplers状态清空，重新开始统计
            state.trial_params["last_start_data_status"] = state.trial_params[
                "saved_data_status"
            ]
            state.trial_params["consumed_samples"] = 0
            state.trial_params["global_shuffle_seed"] = (
                state.trial_params["global_shuffle_seed"] + 1
            )

            logger.debug(
                f"Update global_shuffle_seed to {state.trial_params['global_shuffle_seed']}"
            )
            logger.debug(
                "Due to changes in the underlying data (ratio, number of files, number of dp), \
                    the index needs to be rebuilt by resetting the consumed_samplers to 0."
            )

        if not args.ignore_data_skip:
            # 进行数据skip - sampler load data_status状态与consumed_samples状态
            batch_sampler.load_data_status(
                state.trial_params["last_start_data_status"],
                state.trial_params["global_shuffle_seed"],
            )
            batch_sampler.set_epoch(0, state.trial_params["consumed_samples"])
        else:
            state.trial_params["consumed_samples"] = 0
            state.trial_params["saved_data_status"] = [
                0 for _ in range(batch_sampler.max_part_id + 1)
            ]
            state.trial_params["last_start_data_status"] = [
                0 for _ in range(batch_sampler.max_part_id + 1)
            ]
            batch_sampler.load_data_status(
                state.trial_params["last_start_data_status"],
                state.trial_params["global_shuffle_seed"],
            )
            batch_sampler.set_epoch(0, state.trial_params["consumed_samples"])
            logger.info("Ignore data skipping and data status")

        state.trial_params["data_status"] = [
            0
            for _ in range(
                max(
                    batch_sampler.max_part_id + 1,
                    len(state.trial_params["saved_data_status"]),
                )
            )
        ]
        self.data_status_shape = paddle.to_tensor(
            len(state.trial_params["data_status"]), dtype="int32"
        )
        if dist.is_initialized():
            logger.info("broadcast data trace callback hook")
            dist.broadcast(self.data_status_shape, 0)

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
        if not args.need_data:
            return
        for part_id in inputs["src_id"]:
            state.trial_params["data_status"][part_id] += 1

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """save_checkpoint 在hook on_save前面, 写在on_step_end这里，在后续save_checkpoint时一起落盘

        Args:
            args (TrainingArguments): _description_
            state (TrainerState): _description_
            control (TrainerControl): _description_
        """
        if not args.need_data:
            if (
                args.use_hybrid_parallel
                and control.should_save
                and dist.is_initialized()
                and args.pp_need_data_degree
                and args.pipeline_parallel_degree > 1
            ):
                _hcg = fleet.get_hybrid_communicate_group()
                data_status = paddle.zeros(
                    [self.data_status_shape.item()], dtype="int64"
                )
                dist.all_reduce(data_status, group=_hcg.get_pipe_parallel_group())
                return  # 呼应 Line:178
            return

        if control.should_save:
            data_status = paddle.to_tensor(
                state.trial_params["data_status"], dtype="int64"
            )
            if dist.is_initialized():
                if args.use_hybrid_parallel:
                    _hcg = fleet.get_hybrid_communicate_group()
                    # dp间进行all_reduce
                    if args.data_parallel_degree > 1:
                        dist.all_reduce(
                            data_status, group=_hcg.get_data_parallel_group()
                        )
                    if args.sharding_parallel_degree > 1:
                        dist.all_reduce(
                            data_status, group=_hcg.get_sharding_parallel_group()
                        )
                    if args.pp_need_data_degree and args.pipeline_parallel_degree > 1:
                        dist.all_reduce(
                            data_status, group=_hcg.get_pipe_parallel_group()
                        )
                else:
                    dist.all_reduce(data_status)  # + group
            logger.debug("All reduced `data_status`")

            _saved_data_status = np.array(state.trial_params["saved_data_status"])
            if len(data_status) > len(_saved_data_status):
                # 数据max_part_id变大。
                _saved_data_status = np.append(
                    _saved_data_status,
                    np.zeros(
                        [
                            len(data_status) - len(_saved_data_status),
                        ],
                        dtype="int64",
                    ),
                )

            state.trial_params["saved_data_status"] = (
                data_status.numpy() + _saved_data_status
            ).tolist()
            # consumed_samples 状态变更 (最新数据状态统计)
            state.trial_params["consumed_samples"] += sum(data_status.tolist())

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        保存模型时触发的事件处理函数

        Args:
            args (TrainingArguments): 训练参数对象
            state (TrainerState): 训练状态对象
            control (TrainerControl): 训练控制对象
            **kwargs: 其他关键字参数

        Returns:
            None

        """
        if not args.need_data:
            return
        # 清空数据计数存储状态
        state.trial_params["data_status"] = [
            0 for _ in range(len(state.trial_params["data_status"]))
        ]


class DataTraceCallbackAuto(DataTraceCallback):
    """Callback 用于DataStatus记录

    Args:
        TrainerCallback (_type_): _description_
    """

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
        if not args.need_data:
            return
        for part_id in inputs["input_ids"][3]:  # src_id
            state.trial_params["data_status"][part_id] += 1
