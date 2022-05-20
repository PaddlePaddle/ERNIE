# -*- coding: utf-8 -*
"""
:py:class:`ScalarFieldReader`

"""
import numpy as np
import paddle
from ...common.register import RegisterSet
from ...common.rule import DataShape, FieldLength, InstanceName
from .base_field_reader import BaseFieldReader
from ..tokenizer.custom_tokenizer import CustomTokenizer
from ...utils.util_helper import convert_to_unicode


@RegisterSet.field_reader.register
class ScalarFieldReader(BaseFieldReader):
    """单个标量的field_reader，直接返回数据本身（数据可以是单个数字，也可以是单个的明文字符，
    明文通过json文件中配置的vocab_path去进行转换），shape= [batch_size,1]
    """
    def __init__(self, field_config):
        """
        :param field_config:
        """
        BaseFieldReader.__init__(self, field_config=field_config)

        if field_config.vocab_path and field_config.need_convert:
            if self.field_config.tokenizer_info:
                tokenizer_class = RegisterSet.tokenizer.__getitem__(self.field_config.tokenizer_info["type"])
                self.tokenizer = tokenizer_class(vocab_file=self.field_config.vocab_path,
                                                 split_char=self.field_config.tokenizer_info["split_char"],
                                                 unk_token=self.field_config.tokenizer_info["unk_token"])
            else:
                self.tokenizer = CustomTokenizer(vocab_file=self.field_config.vocab_path)

    def init_reader(self, dataset_type=InstanceName.TYPE_PY_READER):
        """ 初始化reader格式，两种模式，如果是py_reader模式的话，返回reader的shape、type、level；
        如果是data_loader模式，返回fluid.data数组
        :param dataset_type : dataset的类型，目前有两种：py_reader、data_loader， 默认是py_reader
        :return:
        """

        shape = [[-1, 1]]
        types = []
        levels = [0]
        data_list = []

        if self.field_config.data_type == DataShape.INT:
            types.append('int64')
        elif self.field_config.data_type == DataShape.FLOAT:
            types.append('float32')
        else:
            raise TypeError("ScalarFieldReader's data_type must be int or float")

        if dataset_type == InstanceName.TYPE_DATA_LOADER:
            name = self.field_config.name + "_" + InstanceName.SRC_IDS
            data_list.append(paddle.static.data(name=name, shape=shape[0],
                                               dtype=types[0], lod_level=levels[0]))
            return data_list
        else:
            return shape, types, levels

    def convert_texts_to_ids(self, batch_text):
        """将一个batch的明文text转成id
        :param batch_text:
        :return:
        """
        src_ids = []
        for text in batch_text:
            src_id = text.split(" ")
            # 因为是单个标量数据，所以直接取第0个就行
            if self.tokenizer and self.field_config.need_convert:
                scalar = self.tokenizer.covert_token_to_id(convert_to_unicode(src_id[0]))
            else:
                scalar = src_id[0]
            src_ids.append(scalar)

        return_list = []
        if self.field_config.data_type == DataShape.FLOAT:
            return_list.append(np.array(src_ids).astype("float32").reshape([-1, 1]))

        elif self.field_config.data_type == DataShape.INT:
            return_list.append(np.array(src_ids).astype("int64").reshape([-1, 1]))

        return return_list

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

    def get_field_length(self):
        """获取当前这个field在进行了序列化之后，在field_id_list中占多少长度
        :return:
        """
        return FieldLength.SINGLE_SCALAR_FIELD
