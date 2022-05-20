# -*- coding: utf-8 -*
"""
:py:class:`BaseDataSetReader`
"""

import logging
import time
import numpy as np

from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
from collections import namedtuple

@RegisterSet.data_set_reader.register
class BaseDataSetReaderErnieGen(object):
    """BaseDataSetReader:将样本中数据组装成一个py_reader, 向外提供一个统一的接口。
    核心内容是读取明文文件，转换成id，按py_reader需要的tensor格式灌进去，然后通过调用run方法让整个循环跑起来。
    py_reader拿出的来的是lod-tensor形式的id，这些id可以用来做后面的embedding等计算。
    """
    def __init__(self, name, fields, config):
        self.name = name
        self.fields = fields
        self.config = config  # 常用参数，batch_size等，ReaderConfig类型变量
        self.paddle_data_loader = None
        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0
        # 迭代器生成数据的时候是否需要生成明文样本，目前来看，训练的时候不需要，预测的时候需要
        self.need_generate_examples = False

    def create_reader(self):
        """
        必须选项，否则会抛出异常。
        用于初始化self.paddle_data_loader。
        ``self.paddle_data_loader = fluid.layers.py_reader(
            capacity=capacity,
            shapes=shapes,
            name=self.name,
            dtypes=types,
            lod_levels=levels,
            use_double_buffer=True)
        ``
        :return:None
        """
        raise NotImplementedError

    def prepare_batch_data(self, examples, batch_size):
        """generate batch records"""
        raise NotImplementedError


    def instance_fields_dict(self):
        """
        必须选项，否则会抛出异常。
        实例化fields_dict, 调用pyreader，得到fields_id, 视情况构造embedding，然后结构化成dict类型返回给组网部分。
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

    def add_input(self):
        """warp
        """
        pass

    def data_generator(self):
        """
        必须选项，否则会抛出异常。
        数据生成器：读取明文文件，生成batch化的id数据，绑定到py_reader中
        :return:list
                [[src_ids],
                 [mask_ids],
                 [seq_lens]
                ]
        """
        raise NotImplementedError

    # def set_provider(self):
    #     """
    #     :return:
    #     """
    #     self.paddle_data_loader.decorate_tensor_provider(self.data_generator())
    #     logging.info("set data_generator.......")

    def convert_fields_to_dict(self, field_list, need_emb=False):
        """instance_fields_dict一般调用本方法实例化fields_dict，保存各个field的id和embedding(可以没有，是情况而定),
        当need_emb=False的时候，可以直接给predictor调用
        :param field_list:
        :param need_emb:
        :return: dict
        """
        raise NotImplementedError

    def run(self):
        """
        配置py_reader对应的数据生成器，并启动
        :return:
        """
        if self.paddle_data_loader:
            self.paddle_data_loader.set_batch_generator(self.data_generator())
            self.paddle_data_loader.start()
            logging.info("set data_generator and start.......")
        else:
            raise ValueError("paddle_data_loader is None")

    def stop(self):
        """
        :return:
        """
        if self.paddle_data_loader:
            self.paddle_data_loader.reset()
        else:
            raise ValueError("paddle_data_loader is None")

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def get_num_examples(self):
        """get number of example"""
        return self.num_examples

    def api_generator(self, query_list):
        """python api server"""
        if len(query_list) <= 0:
            raise ValueError("query can't be None")
        field_names = []
        for filed in self.fields:
            field_names.append(filed.name)

        Example = namedtuple('Example', field_names)
        examples = []
        for query in query_list:
            examples.append(Example(**query))
        for batch_data in self.prepare_batch_data(examples, self.config.batch_size):
            yield batch_data

