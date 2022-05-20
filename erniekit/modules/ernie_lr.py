# -*- coding: utf-8 -*
"""
ERNIE 使用的学习率设置
"""
import logging

from paddle.optimizer.lr import LRScheduler
import numpy as np


class LinearWarmupDecay(LRScheduler):
    """LinearWarmupDecay
    """
    def __init__(self, base_lr, end_lr, warmup_steps, decay_steps, num_train_steps, power=1.0, verbose=False,
                 cycle=False):
        """先使用warmup线性衰减，由小变大到base_lr， 再使用多项式衰减由大变小到end_lr
        :param base_lr:
        :param end_lr:
        :param warmup_steps:
        :param decay_steps:
        :param num_train_steps:
        :param power:
        :param verbose:
        :param cycle:
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.num_train_steps = num_train_steps
        self.decay_steps = decay_steps  # 与num_train_steps一致
        self.end_lr = end_lr
        self.power = power
        self.cycle = cycle
        # tips: 基类的__init__方法一定要放在最后，因为基类的__init__中会直接调用一次get_lr()
        LRScheduler.__init__(self, learning_rate=base_lr, last_epoch=-1, verbose=verbose)

    def get_lr(self):
        """即时的学习率计算
        """
        if self.last_epoch < self.warmup_steps:
            return self.base_lr * (self.last_epoch / self.warmup_steps)
        else:
            return self._polynomial_decay(learning_rate=self.base_lr,
                                          decay_steps=self.decay_steps,
                                          end_learning_rate=self.end_lr,
                                          power=self.power,
                                          cycle=self.cycle)

    def _polynomial_decay(self, learning_rate, decay_steps, end_learning_rate=0.0001, power=1.0, cycle=False):
        """
        the same algrithm as
        paddle/fluid/layers/learning_rate_scheduler.py:
        polynomial_decay
        """
        global_step = self.last_epoch
        if cycle:
            div_res = np.ceil(self.last_epoch / decay_steps)
            if self.last_epoch == 0:
                div_res = 1.0
            decay_steps = decay_steps * div_res
        else:
            global_step = min(decay_steps, self.last_epoch)

        decayed_lr = (learning_rate - end_learning_rate) * \
                     ((1.0 - global_step / decay_steps) ** power) + end_learning_rate
        return decayed_lr


def exclude_from_weight_decay(name):
    """exclude_from_weight_decay
    """
    if not isinstance(name, str):
        name = name.name
    if name.find("layer_norm") > -1:
        return True
    bias_suffix = ["_bias", "_b", ".b_0"]
    for suffix in bias_suffix:
        if name.endswith(suffix):
            return True
    return False


def lr_decay_fn(param, decay_rate, n_layers, server_layers):
    """lr_decay_fn
    """
    if "encoder_layer" in param.name and param.name.index("encoder_layer") == 0:
        depth = int(param.name.split("_")[2]) + 1
    elif "server_post_encoder_layer" in param.name or "sharing_to_task_fc.w_0" in param.name:
        depth = server_layers
    elif "nlu_encoder_layer" in param.name and param.name.index("nlu_encoder_layer") == 0:
        depth = int(param.name.split("_")[3]) + 1
    elif "nlg_encoder_layer" in param.name and param.name.index("nlg_encoder_layer") == 0:
        depth = int(param.name.split("_")[3]) + 1
    elif 'nlu_post_encoder_layer' in param.name or "nlg_post_encoder_layer" in param.name:
        depth = n_layers + 1
    elif "embedding" in param.name or "emb_hidden_mapping" in param.name or 'pre_encoder_layer' in param.name:
        depth = 0
    else:
        depth = n_layers + 2
    return decay_rate ** (n_layers + 2 - depth)


def lr_decay_freeze_fn(param, decay_rate, n_layers, server_layers):
    """lr_decay_freeze_fn
    """
    if "encoder_layer" in param.name and param.name.index("encoder_layer") == 0:
        depth = int(param.name.split("_")[2]) + 1
        decay_rate = 0
    elif "server_post_encoder_layer" in param.name or "sharing_to_task_fc.w_0" in param.name:
        depth = server_layers
        decay_rate = 0
    elif "nlu_encoder_layer" in param.name and param.name.index("nlu_encoder_layer") == 0:
        depth = int(param.name.split("_")[3]) + 1
        decay_rate = 1
    elif "nlg_encoder_layer" in param.name and param.name.index("nlg_encoder_layer") == 0:
        depth = int(param.name.split("_")[3]) + 1
        decay_rate = 1
    elif 'nlu_post_encoder_layer' in param.name or "nlg_post_encoder_layer" in param.name:
        depth = n_layers + 1
        decay_rate = 1
    elif "embedding" in param.name or "emb_hidden_mapping" in param.name or 'pre_encoder_layer' in param.name:
        depth = 0
        decay_rate = 0
    else:
        depth = n_layers + 2
        decay_rate = 1
    return decay_rate ** (n_layers + 2 - depth)
