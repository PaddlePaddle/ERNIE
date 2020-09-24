#    Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" text preprocess """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle.fluid as fluid

def manual_warmup_decay(learning_rate, warmup_steps, num_train_steps, decay_steps=[], lr_decay_ratio=0.1):
    """ 
    Applies linear warmup of learning rate from 0 and keep constant.
    """
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
            for i, step in enumerate(decay_steps):
                with switch.case(global_step < step):
                    decayed_lr = learning_rate * (global_step / global_step) * pow(lr_decay_ratio, i)
                    fluid.layers.tensor.assign(decayed_lr, lr)
            with switch.default():
                constant_lr = learning_rate * (global_step / global_step) * pow(lr_decay_ratio, len(decay_steps))
                fluid.layers.tensor.assign(constant_lr, lr)

        return lr


def linear_warmup_decay(learning_rate, warmup_steps, num_train_steps):
    """ 
    Applies linear warmup of learning rate from 0 and decay to 0.
    """
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

def optimization(loss,
                 warmup_steps,
                 num_train_steps,
                 learning_rate,
                 train_program,
                 startup_prog,
                 weight_decay,
                 scheduler='linear_warmup_decay',
                 decay_steps=[],
                 lr_decay_dict_file="",
                 lr_decay_ratio=0.1):
    """ 
    optimization implementation 
    """
    if warmup_steps > 0:
        if scheduler == 'noam_decay':
            scheduled_lr = fluid.layers.learning_rate_scheduler \
             .noam_decay(1 / (warmup_steps * (learning_rate ** 2)),
                         warmup_steps)
        elif scheduler == 'linear_warmup_decay':
            scheduled_lr = linear_warmup_decay(learning_rate, warmup_steps,
                                               num_train_steps)
        elif scheduler == 'manual_warmup_decay':
            scheduled_lr = manual_warmup_decay(learning_rate, warmup_steps,
                                               num_train_steps, decay_steps, lr_decay_ratio)
        else:
            raise ValueError("Unkown learning rate scheduler, should be "
                             "'noam_decay' or 'linear_warmup_decay' or 'manual_warmup_decay'")
    else:
        scheduled_lr = fluid.layers.create_global_var(
            name=fluid.unique_name.generate("learning_rate"),
            shape=[1],
            value=learning_rate,
            dtype='float32',
            persistable=True)

    lr_decay_dict = {}
    if lr_decay_dict_file != "":
        with open(lr_decay_dict_file) as f:
            for line in f:
                param, decay_rate = line.strip().split('\t')
                lr_decay_dict[param] = float(decay_rate)

    for param in fluid.default_main_program().block(0).all_parameters():
        if param.name in lr_decay_dict:
            print (param.name, lr_decay_dict[param.name])
            param.optimize_attr['learning_rate'] = lr_decay_dict[param.name]

    optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr)
    optimizer._learning_rate_map[fluid.default_main_program(
    )] = scheduled_lr


    fluid.clip.set_gradient_clip(
        clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0))

    def exclude_from_weight_decay(name):
        """ 
        Parameters not use weight decay
        """
        if name.find("layer_norm") > -1:
            return True
        bias_suffix = ["_bias", "_b", ".b_0"]
        for suffix in bias_suffix:
            if name.endswith(suffix):
                return True
        return False

    param_list = dict()

    for param in train_program.global_block().all_parameters():
        param_list[param.name] = param * 1.0
        param_list[param.name].stop_gradient = True

    _, param_grads = optimizer.minimize(loss)

    if weight_decay > 0:
        for param, grad in param_grads:
            if exclude_from_weight_decay(param.name):
                continue
            with param.block.program._optimized_guard(
                [param, grad]), fluid.framework.name_scope("weight_decay"):
                updated_param = param - param_list[
                    param.name] * weight_decay * scheduled_lr * param.optimize_attr['learning_rate']
                fluid.layers.assign(output=param, input=updated_param)

    return scheduled_lr
