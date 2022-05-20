# -*- coding: utf-8 -*
"""
:py:class:`CategoricalField`

"""

from erniekit.common.register import RegisterSet
from erniekit.data.field_reader.custom_text_field_reader import CustomTextFieldReader
from erniekit.data.util_helper import pad_batch_data
from erniekit.utils.util_helper import convert_to_unicode


@RegisterSet.field_reader.register
class CategoricalField(CustomTextFieldReader):
    """通用文本（string）类型的field_reader, 不进行分词，直接通过词表将明文转成id
    文本处理规则是，文本类型的数据会自动添加padding和mask，并返回length. 比较特殊的一点是，这个field_reader中的length全是1
    """
    def __init__(self, field_config):
        """
        :param field_config:
        """
        CustomTextFieldReader.__init__(self, field_config=field_config)

    def convert_texts_to_ids(self, batch_text):
        """将一个batch的明文text转成id
        :param batch_text:
        :return:
        """
        src_ids = []
        for text in batch_text:
            src_id = self.tokenizer.convert_tokens_to_ids(convert_to_unicode(text))
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


