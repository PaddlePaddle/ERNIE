#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Optimization and learning rate scheduling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet

def linear_warmup_decay(learning_rate, warmup_steps, num_train_steps):
    """ Applies linear warmup of learning rate from 0 and decay to 0."""
    with fluid.default_main_program()._lr_schedule_guard():
        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="scheduled_learning_rate")

        global_step = fluid.layers.learning_rate_scheduler._decay_step_counter(
        )

        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(global_step < warmup_steps):
                warmup_lr = learning_rate * (global_step / warmup_steps)
                fluid.layers.tensor.assign(warmup_lr, lr)
            with switch.default():
                decayed_lr = fluid.layers.learning_rate_scheduler.polynomial_decay(
                    learning_rate=learning_rate,
                    decay_steps=num_train_steps,
                    end_learning_rate=0.0,
                    power=1.0,
                    cycle=False)
                fluid.layers.tensor.assign(decayed_lr, lr)

        return lr

def exclude_from_weight_decay(name):
    """exclude_from_weight_decay"""
    if name.find("layer_norm") > -1:
        return True
    bias_suffix = ["_bias", "_b", ".b_0"]
    for suffix in bias_suffix:
        if name.endswith(suffix):
            return True
    return False

def layer_decay(param, param_last, learning_rate, decay_rate, n_layers):
    """layerwise learning rate decay"""
    delta = param - param_last
    if "encoder_layer" in param.name and param.name.index("encoder_layer")==0:
        print(param.name)
        layer = int(param.name.split("_")[2])
        ratio = decay_rate ** (n_layers + 1 - layer)
        ratio = decay_rate ** (n_layers - layer)
        param_update = param + (ratio - 1) * delta
    elif "embedding" in param.name:
        ratio = decay_rate ** (n_layers + 2)
        ratio = decay_rate ** (n_layers + 1)
        param_update = param + (ratio - 1) * delta
    else:
        param_update = None
    return param_update

def optimization(loss,
                 warmup_steps,
                 num_train_steps,
                 learning_rate,
                 train_program,
                 startup_prog,
                 weight_decay,
                 scheduler='linear_warmup_decay',
                 use_amp=False,
                 init_loss_scaling=32768,
                 layer_decay_rate=0,
                 n_layers=12,
                 dist_strategy=None):
    """optimization"""
    grad_clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0)
    if warmup_steps > 0:
        if scheduler == 'noam_decay':
            scheduled_lr = fluid.layers.learning_rate_scheduler\
             .noam_decay(1/(warmup_steps *(learning_rate ** 2)),
                         warmup_steps)
        elif scheduler == 'linear_warmup_decay':
            scheduled_lr = linear_warmup_decay(learning_rate, warmup_steps,
                                               num_train_steps)
        else:
            raise ValueError("Unkown learning rate scheduler, should be "
                             "'noam_decay' or 'linear_warmup_decay'")
        optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr)
    else:
        scheduled_lr = fluid.layers.create_global_var(
            name=fluid.unique_name.generate("learning_rate"),
            shape=[1],
            value=learning_rate,
            dtype='float32',
            persistable=True)
        optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr, epsilon=1e-06, grad_clip=grad_clip)    
        optimizer._learning_rate_map[fluid.default_main_program()] = scheduled_lr
     
    loss_scaling = fluid.layers.create_global_var(
        name=fluid.unique_name.generate("loss_scaling"),
        shape=[1],
        value=init_loss_scaling,
        dtype='float32',
        persistable=True)
    
    param_list = dict()
    for param in train_program.global_block().all_parameters():
        param_list[param.name] = param * 1.0
        param_list[param.name].stop_gradient = True

    if dist_strategy:
        optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)

    loss = fluid.layers.mean(loss)
    _, param_grads = optimizer.minimize(loss)
    
    if use_amp:
        loss_scaling = optimizer._optimizer.get_loss_scaling()
    
    if layer_decay_rate > 0:
        for param, grad in param_grads:
            with param.block.program._optimized_guard(
                [param, grad]), fluid.framework.name_scope("layer_decay"):
                param_decay = layer_decay(param, param_list[param.name], \
                    scheduled_lr, layer_decay_rate, n_layers)
                if param_decay:
                    fluid.layers.assign(output=param, input=param_decay)

    if weight_decay > 0:
        for param, grad in param_grads:
            if exclude_from_weight_decay(param.name):
                continue
            with param.block.program._optimized_guard(
                [param, grad]), fluid.framework.name_scope("weight_decay"):
                updated_param = param - param_list[
                    param.name] * weight_decay * scheduled_lr
                fluid.layers.assign(output=param, input=updated_param)

    return scheduled_lr, loss_scaling
