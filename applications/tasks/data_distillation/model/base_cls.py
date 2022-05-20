# -*- coding: utf-8 -*
"""
分类任务的网络基类
"""
import collections
import logging
import numpy as np
import paddle

from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
from erniekit.metrics import metrics
from erniekit.model.model import BaseModel


@RegisterSet.models.register
class BaseClassification(BaseModel):
    """BaseClassification
    """
    def __init__(self, model_params):
        """
        """
        BaseModel.__init__(self, model_params)

    def structure(self):
        """网络结构组织
        :return:
        """
        raise NotImplementedError

    def forward(self, fields_dict, phase):
        """ 前向计算
        :param fields_dict:
        :param phase:
        :return:
        """
        raise NotImplementedError

    def set_optimizer(self):
        """优化器设置
        :return: optimizer
        """
        opt_param = self.model_params.get('optimization', None)
        if opt_param:
            self.lr = opt_param.get('learning_rate', 2e-5)
        else:
            self.lr = 2e-5
        self.optimizer = paddle.optimizer.Adam(learning_rate=self.lr, parameters=self.parameters())
        return self.optimizer

    def get_metrics(self, forward_return_dict, meta_info, phase):
        """
        :param forward_return_dict: 前向计算得出的结果
        :param meta_info: 常用的meta信息，如step, used_time, gpu_id等
        :param phase: 当前调用的阶段，包含训练和评估
        :return:
        """
        predictions = forward_return_dict[InstanceName.PREDICT_RESULT]
        label = forward_return_dict[InstanceName.LABEL]
        # paddle_acc = forward_return_dict["acc"]
        if self.is_dygraph:
            if isinstance(predictions, list):
                predictions = [item.numpy() for item in predictions]
            else:
                predictions = predictions.numpy()

            if isinstance(label, list):
                label = [item.numpy() for item in label]
            else:
                label = label.numpy()

        metrics_acc = metrics.Acc()
        acc = metrics_acc.eval([predictions, label])
        metrics_pres = metrics.Precision()
        precision = metrics_pres.eval([predictions, label])

        if phase == InstanceName.TRAINING:
            step = meta_info[InstanceName.STEP]
            time_cost = meta_info[InstanceName.TIME_COST]
            loss = forward_return_dict[InstanceName.LOSS]
            if isinstance(loss, paddle.Tensor):
                loss_np = loss.numpy()
                mean_loss = np.mean(loss_np)
            else:
                mean_loss = np.mean(loss)

            logging.info("phase = {0} loss = {1} acc = {2} precision = {3} step = {4} time_cost = {5}".format(
                phase, mean_loss, acc, precision, step, round(time_cost, 4)))
        if phase == InstanceName.EVALUATE or phase == InstanceName.TEST:
            time_cost = meta_info[InstanceName.TIME_COST]
            step = meta_info[InstanceName.STEP]
            logging.info("phase = {0} acc = {1} precision = {2} time_cost = {3} step = {4}".format(
                phase, acc, precision, round(time_cost, 4), step))

        metrics_return_dict = collections.OrderedDict()
        metrics_return_dict["acc"] = acc
        metrics_return_dict["precision"] = precision
        return metrics_return_dict

    def fields_process(self, fields_dict, phase):
        """
        对fields_dict中序列化好的id按需做二次处理。
        :return: 处理好的fields
        """
        pass
