# -*- coding: utf-8 -*
"""
针对于大多数ernie3任务的基类，主要是写优化器
"""
import paddle
from functools import partial
from .model import BaseModel
from ..modules.ernie_lr import lr_decay_fn, exclude_from_weight_decay, LinearWarmupDecay


class BaseErnieModel(BaseModel):
    """BaseErnieModel
    """
    def __init__(self, model_params):
        """
        BaseErnieModel
        """
        BaseModel.__init__(self, model_params)

    def set_optimizer(self):
        """优化器设置
        :return: optimizer
        """
        # 学习率和权重的衰减设置在optimizer中，loss的缩放设置在amp中（各个trainer中进行设置）。
        # TODO:需要考虑学习率衰减、权重衰减设置、 loss的缩放设置
        opt_param = self.model_params.get('optimization', None)
        self.lr = opt_param.get("learning_rate", 2e-5)
        weight_decay = opt_param.get("weight_decay", 0.01)
        self.use_lr_decay = opt_param.get("use_lr_decay", True)
        self.use_default_decay = opt_param.get("use_default_decay", False)

        epsilon = opt_param.get("epsilon", 1e-8)
        beta1 = 0.9
        beta2 = 0.95
        clip_norm_thres = 1.0

        use_layer_decay = opt_param.get("use_layer_decay", False)
        layer_decay_ratio = opt_param.get("layer_decay_ratio", 0.95)
        n_layers = opt_param.get("n_layers", 60)
        server_layers = opt_param.get("sharing_layers", 48)

        g_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm_thres)

        parameters = None
        if self.is_dygraph:
            parameters = self.parameters()

        if use_layer_decay:
            assert layer_decay_ratio > 0
            layer_decay_fn = partial(lr_decay_fn,
                                     decay_rate=layer_decay_ratio,
                                     n_layers=n_layers,
                                     server_layers=server_layers)
        else:
            layer_decay_fn = None

        if self.use_lr_decay:
            max_train_steps = opt_param.get("max_train_steps", 0)
            warmup_steps = opt_param.get("warmup_steps", 0)
            self.lr_scheduler = LinearWarmupDecay(base_lr=self.lr, end_lr=0.0, warmup_steps=warmup_steps,
                                                  decay_steps=max_train_steps, num_train_steps=max_train_steps)

            self.optimizer = paddle.optimizer.AdamW(learning_rate=self.lr_scheduler,
                                                    beta1=beta1,
                                                    beta2=beta2,
                                                    epsilon=epsilon,
                                                    parameters=parameters,
                                                    weight_decay=weight_decay,
                                                    lr_ratio=layer_decay_fn,
                                                    apply_decay_param_fun=lambda n: not exclude_from_weight_decay(n),
                                                    grad_clip=g_clip)
        else:
            self.optimizer = paddle.optimizer.AdamW(self.lr,
                                                    beta1=beta1,
                                                    beta2=beta2,
                                                    epsilon=epsilon,
                                                    parameters=parameters,
                                                    weight_decay=weight_decay,
                                                    lr_ratio=layer_decay_fn,
                                                    apply_decay_param_fun=lambda n: not exclude_from_weight_decay(n),
                                                    grad_clip=g_clip)

        return self.optimizer
