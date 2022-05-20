# -*- coding: utf-8 -*
"""
:py:class:`ErnieSeqlabelLabelFieldReader`

"""
import paddle
from ...common.register import RegisterSet
from ...common.rule import DataShape, FieldLength, InstanceName
from .base_field_reader import BaseFieldReader
from ..util_helper import pad_batch_data
from ...modules.token_embedding.custom_fluid_embedding import CustomFluidTokenEmbedding
from ...utils.util_helper import truncation_words


@RegisterSet.field_reader.register
class ErnieSeqlabelLabelFieldReader(BaseFieldReader):
    """基于ernie的序列标注专用field_reader，处理规则和custom_text_field一样，自动添加padding和mask，并返回length
    不同的地方在于ErnieSeqlabelLabelFieldReader会在序列首尾分别添加UNK_ID来占位，以保证和ernie序列化过程中的[CLS]和[SEP]对应
    """

    def __init__(self, field_config):
        """
        :param field_config:
        """
        BaseFieldReader.__init__(self, field_config=field_config)

        if self.field_config.tokenizer_info:
            tokenizer_class = RegisterSet.tokenizer.__getitem__(self.field_config.tokenizer_info["type"])

            params = None
            if self.field_config.tokenizer_info.__contains__("params"):
                params = self.field_config.tokenizer_info["params"]

            self.tokenizer = tokenizer_class(vocab_file=self.field_config.vocab_path,
                                             split_char=self.field_config.tokenizer_info["split_char"],
                                             unk_token=self.field_config.tokenizer_info["unk_token"],
                                             params=params)

        if self.field_config.embedding_info and self.field_config.embedding_info["use_reader_emb"]:
            self.token_embedding = CustomFluidTokenEmbedding(self.field_config.embedding_info)

    def init_reader(self, dataset_type=InstanceName.TYPE_PY_READER):
        """ 初始化reader格式，两种模式，如果是py_reader模式的话，返回reader的shape、type、level；
        如果是data_loader模式，返回fluid.data数组
        :param dataset_type : dataset的类型，目前有两种：py_reader、data_loader， 默认是py_reader
        :return:
        """
        shape = []
        types = []
        levels = []
        feed_names = []
        data_list = []

        if self.field_config.data_type == DataShape.STRING:
            """src_ids"""
            shape.append([-1, -1])
            levels.append(0)
            types.append('int64')
            feed_names.append(self.field_config.name + "_" + InstanceName.SRC_IDS)
        else:
            raise TypeError("ErnieSeqlabelLabelFieldReader's data_type must be string")

        """mask_ids"""
        shape.append([-1, -1])
        levels.append(0)
        types.append('float32')
        feed_names.append(self.field_config.name + "_" + InstanceName.MASK_IDS)

        """seq_length"""
        if paddle.__version__[:3] <= '1.5':
            shape.append([-1])
        else:
            shape.append([-1])
        levels.append(0)
        types.append('int64')
        feed_names.append(self.field_config.name + "_" + InstanceName.SEQ_LENS)

        if dataset_type == InstanceName.TYPE_DATA_LOADER:
            for i in range(len(feed_names)):
                data_list.append(paddle.static.data(name=feed_names[i], shape=shape[i],
                                                   dtype=types[i], lod_level=levels[i]))
            return data_list
        else:
            return shape, types, levels

    def convert_texts_to_ids(self, batch_text, use_random_pos=False, max_pos_id=2048):
        """将一个batch的明文text转成id
        :param batch_text:
        :return:
        """
        src_ids = []
        for text in batch_text:
            if self.field_config.need_convert:
                tokens = self.tokenizer.tokenize(text)
                src_id = self.tokenizer.convert_tokens_to_ids(tokens)
            else:
                if isinstance(text, str):
                    src_id = text.split(" ")
                src_id = [int(i) for i in text]

            # 加上截断策略
            if len(src_id) > self.field_config.max_seq_len - 2:
                src_id = truncation_words(src_id, self.field_config.max_seq_len - 2, self.field_config.truncation_type)
            unk_id = self.tokenizer.vocabulary.vocab_dict[self.field_config.tokenizer_info["unk_token"]] 
            src_id.insert(0, unk_id)
            src_id.append(unk_id)
            src_ids.append(src_id)
        
        return_list = []
        padded_ids, mask_ids, batch_seq_lens = pad_batch_data(src_ids,
                                                              pad_idx=self.field_config.padding_id,
                                                              return_input_mask=True,
                                                              return_seq_lens=True)
        return_list.append(padded_ids)
        return_list.append(mask_ids)
        return_list.append(batch_seq_lens)

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
        record_id_dict[InstanceName.MASK_IDS] = fields_id[start_index + 1]
        record_id_dict[InstanceName.SEQ_LENS] = fields_id[start_index + 2]

        record_emb_dict = None
        if need_emb and self.token_embedding:
            record_emb_dict = self.token_embedding.get_token_embedding(record_id_dict)

        record_dict = {}
        record_dict[InstanceName.RECORD_ID] = record_id_dict
        record_dict[InstanceName.RECORD_EMB] = record_emb_dict

        return record_dict

    def get_field_length(self):
        """获取当前这个field在进行了序列化之后，在field_id_list中占多少长度
        :return:
        """
        return FieldLength.CUSTOM_TEXT_FIELD
