# -*- coding: utf-8 -*
"""
:py:class:`ReaderConfig` is an abstract class representing
"""


class ReaderConfig(object):
    """ReaderConfig"""
    def __init__(self):
        self.data_path = None
        self.shuffle = False
        self.batch_size = 8
        self.sampling_rate = 1.0
        self.epoch = 1
        # 数据是否需要按卡数进行分发：多卡模式下，训练集一般需要多卡按卡号分发，测试集和验证集不需要按卡分发，预测集暂时也不进行分发。
        self.need_data_distribute = False
        self.need_generate_examples = False
        self.extra_params = {}

    def build(self, params_dict):
        """
        :param params_dict:
        :return:
        """
        self.has_keyTag = params_dict.get("key_tag", False)
        self.data_path = params_dict["data_path"]
        self.shuffle = params_dict.get("shuffle", False)
        self.batch_size = params_dict.get("batch_size", 8)
        self.sampling_rate = params_dict.get("sampling_rate", 1)
        self.epoch = params_dict.get("epoch", 1)
        self.need_data_distribute = params_dict.get("need_data_distribute", False)
        self.need_generate_examples = params_dict.get("need_generate_examples", False)

        self.extra_params = params_dict.get("extra_params", {})
