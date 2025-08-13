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
from paddleformers.trainer.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)


def progressive_accumulate_steps(
    acc_step_begin, acc_step_end, warmup_global_steps, increment, step
):

    assert step >= 0, step
    if step >= warmup_global_steps:
        return acc_step_end
    slope = (acc_step_end - acc_step_begin) / warmup_global_steps
    acc_steps = int(slope * step + acc_step_begin)
    acc_steps = int(np.ceil(acc_steps / increment) * increment)
    return acc_steps


class ProgreesiveBatchingCallback(TrainerCallback):
    def __init__(self, acc_step_bigin, acc_step_end, warmup_global_steps, increment):
        self.acc_step_bigin = acc_step_bigin
        self.acc_step_end = acc_step_end
        self.warmup_global_steps = warmup_global_steps
        self.increment = increment

    def on_train_begin(self, args, state, control, **kwargs):
        new_acc_step = progressive_accumulate_steps(
            self.acc_step_bigin,
            self.acc_step_end,
            self.warmup_global_steps,
            self.increment,
            state.global_step,
        )
        if new_acc_step != args.gradient_accumulation_steps:
            logger.info(
                f"updating acc_step{args.gradient_accumulation_steps}->{new_acc_step}, global_step={state.global_step}"
            )
            args.gradient_accumulation_steps = new_acc_step

    def on_step_end(self, args, state, control, **kwargs):
        new_acc_step = progressive_accumulate_steps(
            self.acc_step_bigin,
            self.acc_step_end,
            self.warmup_global_steps,
            self.increment,
            state.global_step,
        )
        if new_acc_step != args.gradient_accumulation_steps:
            logger.info(
                f"updating acc_step{args.gradient_accumulation_steps}->{new_acc_step}, global_step={state.global_step}"
            )
            args.gradient_accumulation_steps = new_acc_step
