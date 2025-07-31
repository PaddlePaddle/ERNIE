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

"""
optimizer callback
"""

import paddle

from paddleformers.utils.log import logger
from paddleformers.trainer.trainer_callback import TrainerCallback
from models.ernie_moe.modeling import ErnieDecoderLayer
from models.utils import global_training_logs_enabled, get_global_training_logs


class OrthogonalCallback(TrainerCallback):
    """
    A [`OptimizerCallback`] that gather and save optimizer momentum statistics.

    Args:
        tb_writer (`SummaryWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, ortho_loss_lambda):
        self.ortho_loss_lambda = ortho_loss_lambda

    def on_optimizer_end(self, args, state, control, **kwargs):
        """
        Callback when optimizer ends.
        """
        # Set `base_lr` back to default value.
        model = kwargs["model"]
        optimizer = kwargs["optimizer"]
        lr = optimizer.get_lr()

        def _varbase_help(param, tmp_tensor):
            tmp_tensor._share_buffer_to(param)
            tmp_tensor._clear()

        def update_(layer):
            if isinstance(layer, ErnieDecoderLayer):

                log = {}
                if not hasattr(layer.mlp, "gate"):
                    logger.info("skip empty layer")
                    return
                gate = layer.mlp.gate
                if (
                    hasattr(gate, "weight") and not gate.weight.stop_gradient
                ):  # 文本gate
                    # 如果开了allgather overlap，参数此时还没有同步，因此必须把gate的参数同步。
                    # gate可能分布在多个param buckets中，这会导致多个param都被同步，从而降低了overlap的收益。
                    # 可以从两个方向解决此问题：
                    # 1. 把gate的weight标记color，这样仅gate的bucket需要在这里同步，但此方法不支持热启，仅适合冷启的情况。
                    # 2. 把sharding_comm_buffer_size_MB调小，即增加bucket的个数。在lite上实测，设置128M可以不降低overlap的效果，其他模型可能需要调试。
                    if getattr(optimizer, "_all_gather_overlap_forward", None):
                        # 目前gate层仅有一个hook，如果后续有新增，这里需要同步修改
                        assert len(gate._forward_pre_hooks) == 1
                        # 调用第一个hook，把参数同步，此hook不需要输入
                        hook_id = list(gate._forward_pre_hooks.keys())[0]
                        gate._forward_pre_hooks[hook_id](gate, inputs=None)
                    assert (
                        gate.weight.dtype == paddle.float32
                    ), f"got unexpected dtype: {gate.weight.dtype}"
                    oloss = gate._cal_orthogonal_loss_opt_each_weight(
                        gate.weight, model.config.moe_group_experts
                    )
                    (oloss_grad,) = paddle.autograd.grad(oloss, gate.weight)
                    with paddle.no_grad():
                        gate.weight.data.add_(-oloss_grad * self.ortho_loss_lambda)
                    gate.weight.stop_gradient = False
                    # 更新日志
                    if global_training_logs_enabled():
                        log[f"orthogonal_loss_layer_{layer.layer_idx}"] = oloss
                        prefix = "lm"
                        global_training_logs = get_global_training_logs()
                        global_training_logs.update(
                            **log,
                            **{
                                k.replace(f"_layer_{layer.layer_idx}", ""): v
                                for k, v in log.items()
                            },
                        )
                        global_training_logs.update(
                            **{
                                prefix
                                + "_"
                                + k.replace(f"_layer_{layer.layer_idx}", ""): v
                                for k, v in log.items()
                            }
                        )
                if (
                    hasattr(gate, "weight_1") and not gate.weight_1.stop_gradient
                ):  # 图gate
                    assert (
                        gate.weight_1.dtype == paddle.float32
                    ), f"got unexpected dtype: {gate.weight_1.dtype}"
                    oloss = gate._cal_orthogonal_loss_opt_each_weight(
                        gate.weight_1, False
                    )
                    (oloss_grad,) = paddle.autograd.grad(oloss, gate.weight_1)
                    with paddle.no_grad():
                        gate.weight_1.data.add_(-oloss_grad * self.ortho_loss_lambda)
                    gate.weight_1.stop_gradient = False
                    # 更新日志
                    if global_training_logs_enabled():
                        log[f"orthogonal_loss_layer_{layer.layer_idx}"] = oloss
                        prefix = "mm"
                        global_training_logs = get_global_training_logs()
                        global_training_logs.update(
                            **log,
                            **{
                                k.replace(f"_layer_{layer.layer_idx}", ""): v
                                for k, v in log.items()
                            },
                        )
                        global_training_logs.update(
                            **{
                                prefix
                                + "_"
                                + k.replace(f"_layer_{layer.layer_idx}", ""): v
                                for k, v in log.items()
                            }
                        )

        model.apply(update_)
