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
from paddle.fluid import framework
from paddle.fluid.framework import Variable, default_main_program
import numpy as np
import paddle as P
import paddle.distributed.fleet as fleet
import sys
sys.path.append("../") 
from propeller.paddle.train.hooks import RunHook
import paddle.fluid as F
log = logging.getLogger(__name__)

from utils import create_if_not_exists, get_warmup_and_linear_decay

class AdamW(P.optimizer.AdamW):
    """AdamW object for dygraph"""
    def __init__(self, *args, **kwargs):
        layerwise_lr_decay = kwargs.pop('layerwise_lr_decay_rate', 0.8) 
        n_layers = kwargs.pop('n_layers', 12) 
        var_name_to_exclude = kwargs.pop('var_name_to_exclude', '.*layer_norm_scale|.*layer_norm_bias|.*b_0')
        super(AdamW, self).__init__(*args, **kwargs)
        self.ld = layerwise_lr_decay
        self.pat = re.compile(var_name_to_exclude)
        self.n_layers = n_layers

    def _get_layerwise_lr_decay_rate(self, param):
        #if self.pat.match(param.name):
        #    return 1.0
        if param.name.startswith("encoder_layer"):
            layer = int(param.name.split("_")[2])
            decay_rate = self.ld ** (self.n_layers - layer)
        elif "embedding" in param.name:
            decay_rate = self.ld ** (self.n_layers + 1)
        else:
            decay_rate = 1.0
        return decay_rate
            

    def _create_param_lr(self, param_and_grad):
        # create learning rate tensor for every parameter
        param = param_and_grad[0]
        param_lr = param.optimize_attr['learning_rate'] * self._get_layerwise_lr_decay_rate(param)
        if type(param_lr) == Variable:
            return param_lr
        else:
            if param_lr == 1.0:
                return self._global_learning_rate()
            else:
                with default_main_program()._lr_schedule_guard(
                        is_with_opt=True), framework.name_scope(
                            'scale_with_param_lr'):
                    return self._global_learning_rate() * param_lr

    def apply_optimize(self, loss, startup_program, params_grads):
        super(AdamW, self).apply_optimize(loss, startup_program, params_grads)
        for p, g in params_grads:
            #log.debug(L.reduce_mean(p))
            if not self.pat.match(p.name):
                L.assign(p * (1. - self.wd * self.current_step_lr()), p)


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

    optimizer = AdamW(
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
            optimizer, init_loss_scaling=2**15, use_dynamic_loss_scaling=True)
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
