# -*- coding: utf-8 -*
"""
BaseDataSetReader，继承自飞桨的IterableDataset，主要功能是将数据集按照组网需要的规则进行分词、转id、组batch。
最后使用DataLoader进行加载。
"""
from paddle.io import IterableDataset
from ...common.register import RegisterSet
import time
import collections


@RegisterSet.data_set_reader.register
class BaseDataSetReader(IterableDataset):
    """BaseDataSetReader
    """
    def __init__(self, name, fields, config):
        IterableDataset.__init__(self)
        self.name = name
        self.fields = fields
        self.config = config  # 常用参数，batch_size等，ReaderConfig类型变量
        self.input_data_list = []
        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0
        # 迭代器生成数据的时候是否需要生成明文样本，目前来看，训练的时候不需要，预测的时候需要
        # self.need_generate_examples = config.get("need_data_distribute", False)
        self.dev_count = 1
        self.trainer_id = 0
        self.trainer_nums = self.dev_count

    def create_reader(self):
        """
        静态图模式下用来初始化数据读取的op，调用op为paddle.static.data
        动态图模式下不需要调用
        :return:None
        """
        raise NotImplementedError

    def instance_fields_dict(self):
        """
        必须选项，否则会抛出异常。
        实例化fields_dict, 得到fields_id, 视情况构造embedding，然后结构化成dict类型返回给组网部分。
        :return:dict
                {"field_name":
                    {"RECORD_ID":
                        {"SRC_IDS": [ids],
                         "MASK_IDS": [ids],
                         "SEQ_LENS": [ids]
                        }
                    }
                }
        实例化的dict，保存了各个field的id和embedding(可以没有，是情况而定), 给trainer用.

        """
        raise NotImplementedError

    def convert_fields_to_dict(self, field_list, need_emb=False):
        """instance_fields_dict一般调用本方法实例化fields_dict，保存各个field的id和embedding(可以没有，是情况而定),
        当need_emb=False的时候，可以直接给predictor调用
        :param field_list:
        :param need_emb:
        :return: dict
        """
        raise NotImplementedError

    def __iter__(self):
        """迭代器
        """
        raise NotImplementedError("'{}' not implement in class {}".format('__iter__', self.__class__.__name__))

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def get_num_examples(self):
        """get number of example"""
        return self.num_examples

    def convert_input_list_to_dict(self, input_list):
        """将dataloader读取的样本数据，由list类型转换成dict类型，静态图模式的execute.run调用
        """

        assert len(self.input_data_list) == len(input_list), "len of input_data_list must equal " \
                                                             "input_list in DataSet.convert_input_list_to_dict"

        feed_dict = collections.OrderedDict()
        for index, data in enumerate(self.input_data_list):
            feed_dict[data.name] = input_list[index]

        return feed_dict

    def api_generator(self, query):
        """python api server
        :param query: list
        :return
        """
        pass
