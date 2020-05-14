#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import logging
import re

import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D

log = logging.getLogger(__name__)

def linear_warmup_decay(learning_rate, warmup_steps, num_train_steps):
    """ Applies linear warmup of learning rate from 0 and decay to 0."""
    with F.default_main_program()._lr_schedule_guard():
        lr = L.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="scheduled_learning_rate")

        global_step = L.learning_rate_scheduler._decay_step_counter()
        
        warmup_lr = learning_rate * (global_step / warmup_steps)

        poly_decay_lr = L.learning_rate_scheduler.polynomial_decay(
            learning_rate=learning_rate,
            decay_steps=num_train_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False)
#
        decayed_lr = L.elementwise_min(warmup_lr, poly_decay_lr)
        L.assign(decayed_lr, lr)
        return lr


def optimization(loss,
                 warmup_steps,
                 num_train_steps,
                 learning_rate,
                 train_program,
                 startup_prog,
                 weight_decay,
                 scheduler='linear_warmup_decay',
                 use_fp16=False,
                 init_loss_scaling=128,
                 incr_every_n_steps=1000,
                 decr_every_n_nan_or_inf=2,
                 incr_ratio=2.0,
                 decr_ratio=0.8):
    """do backword for static"""

    def exclude_from_weight_decay(param):
        name = param.name.rstrip('.master')
        if name.find("layer_norm") > -1:
            return True
        bias_suffix = ["_bias", "_b", ".b_0"]
        for suffix in bias_suffix:
            if name.endswith(suffix):
                return True
        return False

    if warmup_steps > 0:
        if scheduler == 'noam_decay':
            scheduled_lr = L.learning_rate_scheduler\
             .noam_decay(1/(warmup_steps *(learning_rate ** 2)),
                         warmup_steps)
        elif scheduler == 'linear_warmup_decay':
            scheduled_lr = linear_warmup_decay(learning_rate, warmup_steps,
                                               num_train_steps)
        else:
            raise ValueError("Unkown learning rate scheduler, should be "
                             "'noam_decay' or 'linear_warmup_decay'")
        log.debug('using Adam')
        optimizer = F.optimizer.Adam(learning_rate=scheduled_lr)
    else:
        scheduled_lr = L.create_global_var(
            name=F.unique_name.generate("learning_rate"),
            shape=[1],
            value=learning_rate,
            dtype='float32',
            persistable=True)
        log.debug('using Adam')

        optimizer = F.optimizer.Adam(learning_rate=scheduled_lr)
        optimizer._learning_rate_map[F.default_main_program(
        )] = scheduled_lr

    if use_fp16:
        log.info('AMP activated')
        optimizer = F.contrib.mixed_precision.decorate(optimizer, 
            amp_lists=F.contrib.mixed_precision.AutoMixedPrecisionLists(custom_black_varnames={"loss"}, custom_black_list={'layer_norm', 'arg_max', 'argmax'}),
            init_loss_scaling=init_loss_scaling,
            use_dynamic_loss_scaling=True,
        )
        loss_scaling = optimizer.get_loss_scaling()
    else:
        loss_scaling = None

    F.clip.set_gradient_clip(
        clip=F.clip.GradientClipByGlobalNorm(clip_norm=1.0))

    param_list = {}

    for param in train_program.global_block().all_parameters():
        param_list[param.name] = param * 1.0
        param_list[param.name].stop_gradient = True

    _, param_grads = optimizer.minimize(loss)

    if weight_decay > 0:
        for param, grad in param_grads:
            if exclude_from_weight_decay(param):
                continue
            with param.block.program._optimized_guard(
                [param, grad]), F.framework.name_scope("weight_decay"):
                updated_param = param - param_list[
                    param.name] * weight_decay * scheduled_lr
                L.assign(output=param, input=updated_param)

    return scheduled_lr, loss_scaling


class AdamW(F.optimizer.AdamOptimizer):
    """AdamW object for dygraph"""
    def __init__(self, *args, **kwargs):
        weight_decay = kwargs.pop('weight_decay', None) 
        var_name_to_exclude = kwargs.pop('var_name_to_exclude', '.*layer_norm_scale|.*layer_norm_bias|.*b_0')
        super(AdamW, self).__init__(*args, **kwargs)
        self.wd = weight_decay
        self.pat = re.compile(var_name_to_exclude)

    def apply_optimize(self, loss, startup_program, params_grads):
        super(AdamW, self).apply_optimize(loss, startup_program, params_grads)
        for p, g in params_grads:
            #log.debug(L.reduce_mean(p))
            if not self.pat.match(p.name):
                L.assign(p * (1. - self.wd * self.current_step_lr()), p)
            #log.debug(L.reduce_mean(p))


class LinearDecay(D.learning_rate_scheduler.LearningRateDecay):
    def __init__(self,
                 learning_rate,
                 warmup_steps,
                 decay_steps,
                 end_learning_rate=0,
                 power=1.0,
                 cycle=False,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(LinearDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle

    def step(self):
        if self.step_num < self.warmup_steps:
            decayed_lr = self.learning_rate * (self.step_num /
                                               self.warmup_steps)
            decayed_lr = self.create_lr_var(decayed_lr)
        else:
            tmp_step_num = self.step_num
            tmp_decay_steps = self.decay_steps
            if self.cycle:
                div_res = fluid.layers.ceil(
                    self.create_lr_var(tmp_step_num / float(self.decay_steps)))
                if tmp_step_num == 0:
                    div_res = self.create_lr_var(1.0)
                tmp_decay_steps = self.decay_steps * div_res
            else:
                tmp_step_num = self.create_lr_var(
                    tmp_step_num
                    if tmp_step_num < self.decay_steps else self.decay_steps)
                decayed_lr = (self.learning_rate - self.end_learning_rate) * \
                    ((1 - tmp_step_num / tmp_decay_steps) ** self.power) + self.end_learning_rate

        return decayed_lr

