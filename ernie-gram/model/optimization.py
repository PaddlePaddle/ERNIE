#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
from paddle.fluid import framework
from paddle.fluid.framework import Variable, default_main_program
from paddle.optimizer.lr import LambdaDecay


def get_warmup_and_linear_decay(max_steps, warmup_steps):
    return lambda step: min(step / warmup_steps, 1. - (step - warmup_steps) / (max_steps - warmup_steps)) if warmup_steps else 1.


class AdamW(paddle.optimizer.AdamW):
    """AdamW object for dygraph"""
    def __init__(self, *args, **kwargs):
        layerwise_lr_decay = kwargs.pop('layerwise_lr_decay_rate', 0.8) 
        n_layers = kwargs.pop('n_layers', 12) 
        super(AdamW, self).__init__(*args, **kwargs)
        self.ld = layerwise_lr_decay
        self.n_layers = n_layers

    def _get_layerwise_lr_decay_rate(self, param):
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
    
    def _append_decoupled_weight_decay(self, block, param_and_grad):
        """
        Add decoupled weight decay op.
            parameter = parameter - parameter * coeff * lr
        Args:
            block: block in which variable is to be created
            param_and_grad: (parameters, gradients) pairs,
                the parameters need to decay.
        Raises:
            Exception: The type of coeff and parameter is not consistent.
        """
        param, grad = param_and_grad

        if self._apply_decay_param_fun is not None \
                and not self._apply_decay_param_fun(param.name):
            return

        learning_rate = self._global_learning_rate()

        with block.program._optimized_guard(
            [param, grad]), framework.name_scope('weight decay'):
            self._params_name.add(param.name)

            # If it has been calculated, the result will be reused.
            # NOTE(wangxi): In dygraph mode, apply_gradient will be executed
            # every step, so need clear _lr_to_coeff every step,
            # we do this in _create_optimization_pass
            decay_coeff = self._lr_to_coeff.get(learning_rate, None)
            if decay_coeff is None:
                decay_coeff = 1.0 - learning_rate * self._coeff
                self._lr_to_coeff[learning_rate] = decay_coeff

            find_master = (self._multi_precision and
                           param.dtype == core.VarDesc.VarType.FP16)
            if find_master:
                master_weight = self._master_weights[param.name]
                scaled_param = master_weight * decay_coeff
                paddle.fluid.layers.assign(
                    input=scaled_param, output=master_weight)
            else:
                scaled_param = param * decay_coeff
                paddle.fluid.layers.assign(input=scaled_param, output=param)


def optimization(loss,
                 warmup_steps,
                 num_train_steps,
                 learning_rate,
                 train_program,
                 startup_prog,
                 weight_decay,
                 scheduler='linear_warmup_decay',
                 dist_strategy=None,
                 use_amp=False,
                 init_loss_scaling=1.0,
                 incr_every_n_steps=1000,
                 decr_every_n_nan_or_inf=2,
                 incr_ratio=2.0,
                 decr_ratio=0.8,
                 layer_decay_rate=0.8,
                 n_layers=12):

    def exclude_from_weight_decay(param):
        name = param.rstrip('.master')
        if name.find("layer_norm") > -1:
            return True
        bias_suffix = ["_bias", "_b", ".b_0"]
        for suffix in bias_suffix:
            if name.endswith(suffix):
                return True
        return False

    grad_clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0)
    scheduled_lr = paddle.optimizer.lr.LambdaDecay(
        learning_rate,
        get_warmup_and_linear_decay(num_train_steps, warmup_steps))

    optimizer = AdamW(
                learning_rate=scheduled_lr,
                beta1=0.9,
                beta2=0.98,
                epsilon=1e-06,
                weight_decay=weight_decay,
                apply_decay_param_fun=exclude_from_weight_decay,
                grad_clip=grad_clip,
                layerwise_lr_decay_rate=layer_decay_rate,
                n_layers=n_layers)

    loss_scaling = fluid.layers.create_global_var(
        name=fluid.unique_name.generate("loss_scaling"),
        shape=[1],
        value=1.0,
        dtype='float32',
        persistable=True)
    
    optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
    
    _, param_grads = optimizer.minimize(loss)
    
    if use_amp:
        loss_scaling = train_program.global_block().vars['loss_scaling_1']

    return scheduled_lr, loss_scaling
        
    
