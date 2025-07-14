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
This module provides functions related to training configuration management.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def reset_per_device_batch_size(global_batch_size, per_device_train_batch_size, dataset_world_size):
    """
    根据`global_batch_size` 和 `per_device_train_batch_size`
    调整`gradient_accumulation_steps` 和 `per_device_train_batch_size`。

    `per_device_train_batch_size`制定了每张卡**最大**单次forward的batch_size,
    最后返回的`per_device_train_batch_size`可能比输入值小，返回的组合一定能满足`global_batch_size`
    """
    assert (
        global_batch_size % dataset_world_size == 0
    ), f"global_bsz={global_batch_size} not evenly devided by world_size={dataset_world_size}"
    batch_per_device = global_batch_size // dataset_world_size
    if batch_per_device < per_device_train_batch_size:
        gradient_accumulation_steps = 1
        per_device_train_batch_size = batch_per_device
        logger.info(
            f"reset `per_device_train_batch_size` to {per_device_train_batch_size},"
            f"global_batch_size={global_batch_size}, "
            f"dp_worldsize={ dataset_world_size}, accumulate_steps={gradient_accumulation_steps} "
        )
    else:
        assert (
            batch_per_device % per_device_train_batch_size == 0
        ), f"""global_bsz={global_batch_size} not evenly devided by world_size={dataset_world_size},
        batch_per_device={batch_per_device}"""

        gradient_accumulation_steps = batch_per_device // per_device_train_batch_size
        logger.info(
            f"per_device_train_batch_size={per_device_train_batch_size}, global_batch_size={global_batch_size }, "
            f"dp_worldsize={dataset_world_size}, accumulate_steps={gradient_accumulation_steps} "
        )
    return per_device_train_batch_size, gradient_accumulation_steps


def progressive_accumulate_steps(acc_step_begin, acc_step_end, warmup_global_steps, increment, step):
    """
    计算Progressive Batch size warmup时，在global_step=`step`时的accumulate_step数。
    Args:
        `acc_step_begin`: 初始accumulate step
        `acc_step_end`: 最终accumulate step
        `warmup_global_steps`: progressive batch-size warmup 步数。
        `step`: global_step
    Returns:
        accumulate step at `step`
    """
    assert step >= 0, step
    if step >= warmup_global_steps:
        return acc_step_end
    slope = (acc_step_end - acc_step_begin) / warmup_global_steps
    acc_steps = int(slope * step + acc_step_begin)
    acc_steps = int(np.ceil(acc_steps / increment) * increment)
    return acc_steps


def progressive_consumed_examples_per_device(
    acc_step_begin, acc_step_end, warmup_global_steps, micro_batch_size, increment, step
):
    """
    计算Progressive Batch size warmup时，per_device视角下，在global_step=`step`时消费的样本数。
    Args:
        `acc_step_begin`: 初始accumulate step
        `acc_step_end`: 最终accumulate step
        `warmup_global_steps`: progressive batch-size warmup 步数。
        `micro_batch_size`:
        `dp_world_size`:
        `step`: global_step
    Returns:
        consumed example at `step`
    """
    # 模拟整个train loop.
    if step == 0:
        return 0
    accumulate_steps = 0
    for gstep in range(
        step
    ):  # trainer 落盘时，`step` 已经被+=1了，所以实际模型step的次数刚好是 `state.global_step ` 个。
        accumulate_steps += progressive_accumulate_steps(
            acc_step_begin, acc_step_end, warmup_global_steps, increment, gstep
        )
        # logger.info(f'global-{gstep}: acc={accumulate_steps}')

    return accumulate_steps * micro_batch_size
