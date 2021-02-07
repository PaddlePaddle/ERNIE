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
import paddle as P
import paddle.distributed.fleet as fleet
from propeller.paddle.train.hooks import RunHook

log = logging.getLogger(__name__)

from demo.utils import create_if_not_exists, get_warmup_and_linear_decay


def optimization(
        loss,
        warmup_steps,
        num_train_steps,
        learning_rate,
        train_program,
        startup_prog,
        weight_decay,
        scheduler='linear_warmup_decay',
        use_fp16=False, ):
    """do backword for static"""

    def exclude_from_weight_decay(param):
        name = param.rstrip('.master')
        if name.find("layer_norm") > -1:
            return True
        bias_suffix = ["_bias", "_b", ".b_0"]
        for suffix in bias_suffix:
            if name.endswith(suffix):
                return True
        return False

    g_clip = P.nn.ClipGradByGlobalNorm(1.0)
    lr_scheduler = P.optimizer.lr.LambdaDecay(
        learning_rate,
        get_warmup_and_linear_decay(num_train_steps, warmup_steps))

    optimizer = P.optimizer.AdamW(
        learning_rate=lr_scheduler,
        weight_decay=weight_decay,
        grad_clip=g_clip,
        apply_decay_param_fun=exclude_from_weight_decay)

    if use_fp16:
        log.info('AMP activated')
        if weight_decay > 0.:
            raise ValueError(
                'paddle amp will ignore `weight_decay`, see https://github.com/PaddlePaddle/Paddle/issues/29794'
            )
        #amp_list = P.fluid.contrib.mixed_precision.AutoMixedPrecisionLists(
        #    custom_white_list=['softmax', 'layer_norm', 'gelu'])
        optimizer = P.fluid.contrib.mixed_precision.decorate(
            optimizer, init_loss_scaling=3**15, use_dynamic_loss_scaling=True)
        _, param_grads = optimizer.minimize(loss)
        loss_scaling = P.static.default_main_program().global_block().var(
            'loss_scaling_0')
    else:
        _, param_grads = optimizer.minimize(loss)
        loss_scaling = None

    class LRStepHook(RunHook):
        def after_run(self, _, __):
            lr_scheduler.step()
            log.debug('lr step: %.5f' % lr_scheduler.get_lr())

    return LRStepHook(), loss_scaling
