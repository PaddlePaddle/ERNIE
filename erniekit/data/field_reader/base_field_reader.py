# -*- coding: utf-8 -*
"""
:py:class:`Reader` is an abstract class representing
"""
from ...common.register import RegisterSet
from ...common.rule import InstanceName


@RegisterSet.field_reader.register
class BaseFieldReader(object):
    """BaseFieldReader: 作用于field的reader，主要是定义py_reader的格式，完成id序列化和embedding的操作
    """
    def __init__(self, field_config):
        self.field_config = field_config
        self.tokenizer = None  # 用来分词，需要各个子类实现
        self.token_embedding = None  # 用来生成embedding向量，需要各个子类实现

    def init_reader(self, dataset_type=InstanceName.TYPE_PY_READER):
        """ 初始化reader格式，两种模式，如果是py_reader模式的话，返回reader的shape、type、level；
        如果是data_loader模式，返回fluid.data数组
        :param dataset_type : dataset的类型，目前有两种：py_reader、data_loader， 默认是py_reader
        :return:
        """
        raise NotImplementedError

    def convert_texts_to_ids(self, batch_text):
        """ 明文序列化
        :param:batch_text
        :return: id_list
        """
        raise NotImplementedError

    def get_field_length(self):
        """获取当前这个field在进行了序列化之后，在field_id_list中占多少长度
        :return:
        """
        raise NotImplementedError

    def structure_fields_dict(self, fields_id, start_index, need_emb=True):
        """静态图调用的方法，生成一个dict， dict有两个key:id , emb. id对应的是pyreader读出来的各个field产出的id，emb对应的是各个
        field对应的embedding
        :param fields_id: pyreader输出的完整的id序列
        :param start_index:当前需要处理的field在field_id_list中的起始位置
        :param need_emb:是否需要embedding（预测过程中是不需要embedding的）
        :return:
        """
        raise NotImplementedError
