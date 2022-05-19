# -*- coding: utf-8 -*
"""
在文心中，reader对象用来对运行过程中数据集进行处理，而多个reader对象的集合被统一放到DataSet对象中进行管理。
一个DataSet对象的核心成员变量有：训练集reader、测试集reader、评估集reader、以及预测集reader
"""


from ..common.register import RegisterSet
from .field import Field
from .reader_config import ReaderConfig
from paddle.io import DataLoader


class DataSet(object):
    """DataSet"""

    def __init__(self, params_dict):
        """
        :param params_dict:
        """
        self.train_reader = None
        self.test_reader = None
        self.dev_reader = None
        self.predict_reader = None

        self.params_dict = params_dict

    def build(self):
        """
        :return:
        """
        reader_list = []
        data_set_reader_dict = {}

        if self.params_dict.__contains__("train_reader"):
            reader_list.append("train_reader")
        if self.params_dict.__contains__("test_reader"):
            reader_list.append("test_reader")
        if self.params_dict.__contains__("dev_reader"):
            reader_list.append("dev_reader")
        if self.params_dict.__contains__("predict_reader"):
            reader_list.append("predict_reader")

        for reader_name in reader_list:
            cfg_list = self.params_dict.get(reader_name).get("fields")
            train_fields = []
            for item in cfg_list:
                item_field = Field()
                item_field.build(item)
                if item_field.reader_info and item_field.reader_info.get("type", None):
                    reader_class = RegisterSet.field_reader.__getitem__(item_field.reader_info["type"])
                    field_reader = reader_class(item_field)
                    item_field.field_reader = field_reader
                    train_fields.append(item_field)

            reader_cfg = ReaderConfig()
            reader_cfg.build(self.params_dict.get(reader_name).get("config"))

            dataset_reader_name = self.params_dict.get(reader_name).get("type")
            dataset_reader_class = RegisterSet.data_set_reader.__getitem__(dataset_reader_name)
            one_reader = dataset_reader_class(name=reader_name, fields=train_fields, config=reader_cfg)
            # TODO:这里需要用dataset来构造出dataloader
            one_loader = DataLoader(one_reader,
                    batch_size=None,
                    shuffle=False,
                    drop_last=False)

            data_set_reader_dict[reader_name] = one_loader

        if data_set_reader_dict.__contains__("train_reader"):
            self.train_reader = data_set_reader_dict["train_reader"]

        if data_set_reader_dict.__contains__("test_reader"):
            self.test_reader = data_set_reader_dict["test_reader"]

        if data_set_reader_dict.__contains__("dev_reader"):
            self.dev_reader = data_set_reader_dict["dev_reader"]

        if data_set_reader_dict.__contains__("predict_reader"):
            self.predict_reader = data_set_reader_dict["predict_reader"]
        elif data_set_reader_dict.__contains__("train_reader"):
            cfg_list = self.params_dict.get("train_reader").get("fields")
            predict_fields = []
            for item in cfg_list:
                item_field = Field()
                item_field.build(item)
                if item_field.reader_info and item_field.reader_info.get("type", None):
                    reader_class = RegisterSet.field_reader.__getitem__(item_field.reader_info["type"])
                    field_reader = reader_class(item_field)
                    item_field.field_reader = field_reader
                    predict_fields.append(item_field)

            reader_cfg = ReaderConfig()
            reader_cfg.build(self.params_dict.get("train_reader").get("config"))

            dataset_reader_name = self.params_dict.get("train_reader").get("type")
            dataset_reader_class = RegisterSet.data_set_reader.__getitem__(dataset_reader_name)
            predict_reader = dataset_reader_class(name="predict_reader", fields=predict_fields,
                                                       config=reader_cfg)

            # TODO:这里需要用dataset来构造出dataloader
            self.predict_reader = DataLoader(predict_reader,
                                    batch_size=None,
                                    shuffle=False,
                                    drop_last=False)
