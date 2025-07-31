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

import paddle
from paddle.distributed import fleet

from paddleformers.trainer.trainer_callback import TrainerCallback
from paddleformers.trainer.trainer import PREFIX_CHECKPOINT_DIR
from paddleformers.trainer.trainer_utils import IntervalStrategy

try:
    from paddleformers.trainer.trainer import (
        PADDLE_WEIGHT_FILE_NAME as PADDLE_WEIGHTS_NAME,
    )
except ImportError:
    pass


class LogProbsSaveCallback(TrainerCallback):
    """_summary_

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
        assert (
            args.pipeline_parallel_degree > 1
        ), "only support pipeline parallel model now"
        hcg = fleet.get_hybrid_communicate_group()
        # self.fc_interval = config["trainer_args"]["flash_ema_interval"]
        self.mp_rank = hcg.get_model_parallel_rank()
        self.pp_rank = model._stage_id
        self.pp_degree = model._num_stages

        self.log_probs = []

        if not (self.pp_rank == self.pp_degree - 1 and self.mp_rank == 0):
            return

        origin_loss_func = model._loss_fn[0].loss_func

        class LossFuncWrap(paddle.nn.Layer):
            """_summary_

            Args:
                paddle (_type_): _description_
            """

            def forward(_self, *args, **kwargs):
                """_summary_

                Args:
                    _self (_type_): _description_

                Returns:
                    _type_: _description_
                """
                masked_lm_loss = origin_loss_func(*args, **kwargs)
                mp_rank = hcg.get_model_parallel_rank()
                if mp_rank == 0:
                    log_p = masked_lm_loss.clone().detach().cpu()
                    self.log_probs.append(log_p)
                return masked_lm_loss

        model._loss_fn[0].loss_func = LossFuncWrap()

    def on_save(self, args, state, control, model, **kwargs):
        """_summary_

        Args:
            args (_type_): _description_
            state (_type_): _description_
            control (_type_): _description_
            model (_type_): _description_
        """
        if not (
            args.save_strategy == IntervalStrategy.STEPS
            and args.save_steps > 0
            and state.global_step % args.save_steps == 0
        ):
            # maybe zcc
            return

        if not self.log_probs:
            # mp0 and pp -1
            return

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        output_dir = os.path.join(args.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)
        paddle.save(
            self.log_probs,
            os.path.join(
                output_dir, f"log_probs.reeao{args.reeao_dataset_rank:04d}.bin"
            ),
        )
        self.log_probs = []
