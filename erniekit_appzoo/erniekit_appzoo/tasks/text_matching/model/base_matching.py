# -*- coding: utf-8 -*
"""
匹配任务的网络基类
"""
import collections
import logging

import paddle

from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
from erniekit.metrics import metrics
from erniekit.model.model import BaseModel
import numpy as np

@RegisterSet.models.register
class BaseMatching(BaseModel):
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
        :return:
        """
        opt_param = self.model_params.get('optimization', None)
        if opt_param:
            self.lr = opt_param.get('learning_rate', 2e-5)
        else:
            self.lr = 2e-5
        # TODO: parameters参数只有动态图才需要使用，记得验证一下该参数是否影响静态图
        self.optimizer = paddle.optimizer.Adam(learning_rate=self.lr, parameters=self.parameters())
        return self.optimizer

    def get_metrics(self, forward_return_dict, meta_info, phase):
        """
        :param forward_return_dict: 前向计算得出的结果
        :param meta_info: 常用的meta信息，如step, used_time, gpu_id等
        :param phase: 当前调用的阶段，包含训练和评估
        :return:
        """
        metrics_return_dict = collections.OrderedDict()
        if phase == InstanceName.TRAINING:
            step = meta_info[InstanceName.STEP]
            time_cost = meta_info[InstanceName.TIME_COST]
            
            if self.is_dygraph:
                loss = np.mean(forward_return_dict["loss"].numpy())
                pos_score = forward_return_dict["query_pos_title_score"].numpy()
                neg_score = forward_return_dict["query_neg_title_score"].numpy()
            else:
                loss = np.mean(forward_return_dict["loss"])
                pos_score = forward_return_dict["query_pos_title_score"]
                neg_score = forward_return_dict["query_neg_title_score"]
            metrics_pn = metrics.Pn()
            pn = metrics_pn.eval([pos_score, neg_score])
            metrics_return_dict["pn"] = pn
            logging.debug("phase = {0} pn = {1} step = {2} time_cost = {3} loss = {4}".format(phase, round(pn, 3),
                                                                                step, time_cost, round(loss, 3)))
        if phase == InstanceName.EVALUATE or phase == InstanceName.TEST:
            time_cost = meta_info[InstanceName.TIME_COST]
            metrics_auc = metrics.Auc()
            if self.is_dygraph:
                predictions = forward_return_dict[InstanceName.PREDICT_RESULT]
                label = forward_return_dict[InstanceName.LABEL]
                if isinstance(predictions, list):
                    predictions = [item.numpy() for item in predictions]
                else:
                    predictions = predictions.numpy()
                
                if isinstance(label, list):
                    label = [item.numpy() for item in label]
                else:
                    label = label.numpy()
            else:
                predictions = forward_return_dict[InstanceName.PREDICT_RESULT]
                label = forward_return_dict[InstanceName.LABEL]
            auc = metrics_auc.eval([predictions, label])
            metrics_return_dict["auc"] = auc
            logging.debug("phase = {0} auc = {1} time_cost = {2}".format(phase, round(auc, 3), time_cost))

        return metrics_return_dict

    def fields_process(self, fields_dict, phase):
        """
        对fields_dict中序列化好的id按需做二次处理。
        :return: 处理好的fields
        """
        pass
