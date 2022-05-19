#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#                                                                      #
#       Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved         #
#                                                                      #
########################################################################

"""
File: ernie_text_field_reader_for_multilingual.py
Author: Your name(@baidu.com)
Date: Tue 23 Feb 2021 02:53:36 PM CST
"""
from ...common.register import RegisterSet
from ..util_helper import pad_batch_data
from ...utils.util_helper import truncation_words
from .ernie_text_field_reader import ErnieTextFieldReader


@RegisterSet.field_reader.register
class ErnieTextFieldReaderForMultilingual(ErnieTextFieldReader):
    """使用ernie的文本类型的field_reader，用户不需要自己分词
        处理规则是：自动添加padding,mask,position,task,sentence,并返回length
        """
    def convert_texts_to_ids(self, batch_text):
        """将一个batch的明文text转成id
        :param batch_text:
        :return:
        """
        src_ids = []
        position_ids = []
        task_ids = []
        sentence_ids = []
        for text in batch_text:
            if self.field_config.need_convert:
                tokens_text = self.tokenizer.tokenize(text)
                # 加上截断策略
                if len(tokens_text) > self.field_config.max_seq_len - 2:
                    tokens_text = truncation_words(tokens_text, self.field_config.max_seq_len - 2,
                                                   self.field_config.truncation_type)
                tokens = []
                tokens.append("[CLS]")
                for token in tokens_text:
                    tokens.append(token)
                tokens.append("[SEP]")
                src_id = self.tokenizer.convert_tokens_to_ids(tokens)
            else:
                if isinstance(text, str):
                    src_id = text.split(" ")
                src_id = [int(i) for i in text]
                if len(src_id) > self.field_config.max_seq_len - 2:
                    src_id = truncation_words(src_id, self.field_config.max_seq_len - 2,
                                                   self.field_config.truncation_type)
                src_id.insert(0, self.tokenizer.covert_token_to_id("[CLS]"))
                src_id.append(self.tokenizer.covert_token_to_id("[SEP]"))

            src_ids.append(src_id)
            pos_id = list(range(2, len(src_id) + 2))
            task_id = [0] * len(src_id)
            sentence_id = [0] * len(src_id)
            position_ids.append(pos_id)
            task_ids.append(task_id)
            sentence_ids.append(sentence_id)

        return_list = []
        padded_ids, input_mask, batch_seq_lens = pad_batch_data(src_ids,
                                                                pad_idx=self.field_config.padding_id,
                                                                return_input_mask=True,
                                                                return_seq_lens=True)
        sent_ids_batch = pad_batch_data(sentence_ids, pad_idx=self.field_config.padding_id)
        pos_ids_batch = pad_batch_data(position_ids, pad_idx=self.field_config.padding_id)
        task_ids_batch = pad_batch_data(task_ids, pad_idx=self.field_config.padding_id)

        return_list.append(padded_ids)  # append src_ids
        return_list.append(sent_ids_batch)  # append sent_ids
        return_list.append(pos_ids_batch)  # append pos_ids
        return_list.append(input_mask)  # append mask
        return_list.append(task_ids_batch)  # append task_ids
        return_list.append(batch_seq_lens)  # append seq_lens

        return return_list

