# -*- coding: utf-8 -*
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tokenization classes."""
import unicodedata
import six
import sys
from six.moves import range
import logging

if six.PY3:
    import pickle
else:
    import cPickle as pickle


from ...common.register import RegisterSet
from .tokenizer import Tokenizer
from .tokenization_wp import FullTokenizer




@RegisterSet.tokenizer.register
class MixTokenizer(Tokenizer):
    """MixTokenizer: 同时对文本进行 wordseg 词粒度切分, 以及 FullTokenizer 字粒度切分
    """
    def __init__(self, vocab_file, split_char=" ", unk_token="[UNK]", params=None):
        from ..wordseg_mix import wordseg
        self.wordseg = wordseg
        self.MAX_TERM_COUNT = 1024
        self.dict_handle = wordseg.scw_load_worddict("../../erniekit/data/wordseg_mix/chinese_gbk/")
        self.result_handle = wordseg.scw_create_out(self.MAX_TERM_COUNT * 10)
        token_handle = wordseg.create_tokens(self.MAX_TERM_COUNT)
        self.token_handle = wordseg.init_tokens(token_handle, self.MAX_TERM_COUNT)
        self.full_tokenizer = FullTokenizer(vocab_file, split_char, unk_token, params)

    def wordseg_tokenize(self, line):
        """
        wordseg func
        """
        line = line.lower()
        ret = self.wordseg.scw_segment_words(self.dict_handle, self.result_handle, line, len(line), 1)
        res = []
        res_phrase = []
        c = chr(1)
        if ret < 0:
            logging.error("scw_segment_words() failed!")
            return c.join(res), c.join(res_phrase)

        token_count = self.wordseg.scw_get_token_1(self.result_handle, \
            self.wordseg.SCW_BASIC, self.token_handle, self.MAX_TERM_COUNT)
        l = self.wordseg.tokens_to_list(self.token_handle, token_count)
        for token in l:
            res.append(token[7])

        token_count = self.wordseg.scw_get_token_1(self.result_handle, \
            self.wordseg.SCW_WPCOMP, self.token_handle, self.MAX_TERM_COUNT)
        l = self.wordseg.tokens_to_list(self.token_handle, token_count)
        for token in l:
            res_phrase.append(token[7])
        return res

    def tokenize(self, text):
        """
        :param text:
        :return:
        """
        segment_chars = []
        segment_words = []

        text = text.decode('utf-8').encode('gb18030')
        splited_words = self.wordseg_tokenize(text)
        splited_words = [word.decode('gb18030') for word in splited_words]
        for word in splited_words:
            split_chars = self.full_tokenizer.tokenize(word)
            segment_chars += split_chars
            for i in xrange(0, len(split_chars)):
                if i != len(split_chars) - 1:
                    segment_words += [split_chars[i]]
                else:
                    unk_id = self.full_tokenizer.convert_tokens_to_ids(["[UNK]"])[0]
                    word_id = self.full_tokenizer.convert_tokens_to_ids([word])[0]
                    if word_id != unk_id:
                        segment_words += [word]
                    else:
                        segment_words += [split_chars[i]]

        assert len(segment_chars) == len(segment_words), "length of splited char not equal splited words" 
        return (segment_chars, segment_words)
