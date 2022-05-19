# -*- coding: utf-8 -*
"""
文心中的深度学习模型对象，使用飞桨最新的动态图方式建模，同时支持静态图和动态图的运行方式，其核心方法是：
1.组织网络结构:动静结合的组网方式，包含structure（构造方法）和forward（前向计算）两个主要方法
2.设定当前网络的模型评估方式
3.同时支持动态图和静态图
4.预测结果解析？（待定）
"""
import paddle


class BaseModel(paddle.nn.Layer):
    def __init__(self, model_params):
        paddle.nn.Layer.__init__(self)
        self.model_params = model_params
        self.is_dygraph = self.model_params.get("is_dygraph", 0)
        self.lr = None  # 学习率，必须在子类中实现
        self.lr_scheduler = None  # 学习率的衰减设置，必须在子类中实现
        self.optimizer = None  # 优化器设定，必须在子类中实现

    def structure(self):
        """
        网络结构组织
        :return:
        """
        raise NotImplementedError

    def forward(self, fields_dict, phase):
        """
        前向计算
        :param fields_dict:
        :param phase:
        :return:
        """
        raise NotImplementedError

    def get_metrics(self, forward_return_dict, meta_info, phase):
        """
        模型效果评估
        :param forward_return_dict: 前向计算得出的结果
        :param meta_info: 常用的meta信息，如step, used_time, gpu_id等
        :param phase: 当前调用的阶段，包含训练和评估
        :return:
        """
        raise NotImplementedError

    def fields_process(self, fields_dict, phase):
        """
        对fields_dict中序列化好的id按需做二次处理。
        :return: 处理好的fields
        """
        raise NotImplementedError

    def set_optimizer(self):
        """优化器设置
        :return: optimizer
        """
        raise NotImplementedError