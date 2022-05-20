# -*- coding: utf-8 -*
"""
:py:class:`Reader` is an abstract class representing
"""
import numpy as np
from paddle import fluid

from erniekit.common.register import RegisterSet
from erniekit.data.field_reader.base_field_reader import BaseFieldReader
from erniekit.common.rule import InstanceName
from erniekit.data.tokenizer.custom_tokenizer import CustomTokenizer


@RegisterSet.field_reader.register
class MultiLabelFieldReader(BaseFieldReader):
    """MultiLabelFieldReader: 作用于field的reader，主要是定义py_reader的格式，完成id序列化和embedding的操作
    """
    def __init__(self, field_config):
        BaseFieldReader.__init__(self, field_config=field_config)
        if field_config.vocab_path and field_config.need_convert:
            self.tokenizer = CustomTokenizer(vocab_file=self.field_config.vocab_path)

    def init_reader(self, dataset_type=InstanceName.TYPE_PY_READER):
        """ 初始化reader格式，两种模式，如果是py_reader模式的话，返回reader的shape、type、level；
        如果是data_loader模式，返回fluid.data数组
        :param dataset_type : dataset的类型，目前有两种：py_reader、data_loader， 默认是py_reader
        :return:
        """

        shape = [[-1, self.field_config.num_labels]]
        types = ["float32"]
        levels = [0]
        data_list = []

        if dataset_type == InstanceName.TYPE_DATA_LOADER:
            data_list.append(fluid.layers.data(name=self.field_config.name + "_" + InstanceName.SRC_IDS, shape=shape[0],
                                               dtype=types[0], lod_level=levels[0]))
            return data_list
        else:
            return shape, types, levels

    def convert_texts_to_ids(self, batch_text):
        """ 明文序列化
        :param:batch_text
        :return: id_list
        """
        batch_src_ids = []
        for text in batch_text:
            src_ids = [0] * self.field_config.num_labels
            indices = text.split(" ")
            if self.tokenizer and self.field_config.need_convert:
                indices = self.tokenizer.covert_tokens_to_ids(indices)
            else:
                indices = [int(index) for index in indices]
            for index in indices:
                src_ids[index] = 1
            batch_src_ids.append(src_ids)

        return_list = []
        # logging.debug("Hello {}".format(np.array(batch_src_ids)))
        return_list.append(np.array(batch_src_ids).astype("float32"))
        return return_list

    def get_field_length(self):
        """获取当前这个field在进行了序列化之后，在field_id_list中占多少长度
        :return:
        """
        return 1

    def structure_fields_dict(self, fields_id, start_index, need_emb=True):
        """静态图调用的方法，生成一个dict， dict有两个key:id , emb. id对应的是pyreader读出来的各个field产出的id，emb对应的是各个
        field对应的embedding
        :param fields_id: pyreader输出的完整的id序列
        :param start_index:当前需要处理的field在field_id_list中的起始位置
        :param need_emb:是否需要embedding（预测过程中是不需要embedding的）
        :return:
        """
        record_id_dict = {}
        record_id_dict[InstanceName.SRC_IDS] = fields_id[start_index]
        record_dict = {}
        record_dict[InstanceName.RECORD_ID] = record_id_dict
        record_dict[InstanceName.RECORD_EMB] = None
        return record_dict
