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


import paddle

from paddleformers.trainer.training_args import TrainingArguments
from paddleformers.trainer.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from src.utils.misc import global_training_logs


class KlDistillCallback(TrainerCallback):
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
        self.log_probs = []
        if args.kl_distill_loss_coef == 0:
            return

        if args.pipeline_parallel_degree > 1:
            origin_loss_func = model._loss_fn[0].loss_func
        else:
            origin_loss_func = model.criterion.loss_func

        class DistillLoss(paddle.nn.Layer):
            """_summary_

            Args:
                paddle (_type_): _description_
            """

            def forward(_self, *_args, **_kwargs):
                """_summary_

                Args:
                    _self (_type_): _description_

                Returns:
                    _type_: _description_
                """
                masked_lm_loss = origin_loss_func(*_args, **_kwargs)
                # TODO recompute loss func
                delta = masked_lm_loss - self.log_probs.pop(0).unsqueeze(-1)
                kl_loss = paddle.exp(delta) - delta - 1  # pad处 kl loss也为0
                global_training_logs.update(kl_loss=kl_loss.detach().mean().item())
                return masked_lm_loss + args.kl_distill_loss_coef * (
                    kl_loss - kl_loss.detach()
                )

        if args.pipeline_parallel_degree > 1:
            model._loss_fn[0].loss_func = DistillLoss()
        else:
            model.criterion.loss_func = DistillLoss()

    def on_load_data_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        inputs,
        **kwargs,
    ):
        """_summary_

        Args:
            args (TrainingArguments): _description_
            state (TrainerState): _description_
            control (TrainerControl): _description_
            inputs (_type_): _description_
        """
        log_prob = inputs.pop("log_prob", None)
        if args.kl_distill_loss_coef == 0:
            return
        if log_prob is not None:
            if args.multi_token_pred_depth:
                log_prob = log_prob[:, : -args.multi_token_pred_depth]
            self.log_probs.append(log_prob)
