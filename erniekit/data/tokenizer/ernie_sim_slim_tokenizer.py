# -*- coding: utf-8 -*
"""
:py:class:`ErnieSimSlimTokenizer`
"""
import logging
from ...common.register import RegisterSet
from .tokenizer import Tokenizer
from ...utils.util_helper import convert_to_unicode


@RegisterSet.tokenizer.register
class ErnieSimSlimTokenizer(Tokenizer):
    """ErnieSimSlimTokenizer"""

    def __init__(self, vocab_file, split_char=" ", unk_token="[UNK]", params=None):
        """
        :param vocab_file: 词表文件路径
        :param split_char: 明文分隔符，默认是空格
        """
        super(ErnieSimSlimTokenizer, self).__init__(vocab_file, split_char, unk_token, params) 
        from ..erniesim_slim_wordseg.wordseg_client import SegWord
        self.wordseg_inst = SegWord()
        seg_stat = self.wordseg_inst.initialize()
        if not seg_stat:
            logging.error("wordseg init error")
            exit(1)
        else:
            logging.debug("wordseg init succeed")

    def bigram_seg(self, text):
        """
        cal freq
        """
        delim = chr(1)
        bigram_list = []
        cols = text.split(" ")
        if len(cols) == 1 or len(cols) > 1024:
            return bigram_list

        for i in xrange(0, len(cols) - 1):
            bigram = cols[i] + delim + cols[i + 1]
            if cols[i].find(delim) != -1 or cols[i + 1].find(delim) != -1:
                continue
            bigram_list.append(bigram)
        return bigram_list

    def tokenize(self, text):
        """
        :param text:
        :return:
        """
        logging.debug("origin text:{}".format(text))
        text = convert_to_unicode(text).encode('gbk')
        seg_basic = self.wordseg_inst.segword(text, 'basic')
        seg_comp = self.wordseg_inst.segword(text, 'comp')
        
        seg_list = []

        tmp_list = []
        for token in seg_basic:
            if token not in tmp_list:
                tmp_list.append(token)
        seg_list += tmp_list

        tmp_list = []
        for token in seg_comp:
            if token not in tmp_list:
                tmp_list.append(token)
        seg_list += tmp_list

        seg_list += self.bigram_seg(" ".join(seg_basic))
        unicode_seg = [token.decode('gbk') for token in seg_list]
        logging.debug("tokenzied text:{}".format(" ".join(unicode_seg)))
        unicode_seg = [token for token in unicode_seg if token in self.vocabulary.vocab_dict]

        return unicode_seg

    def convert_tokens_to_ids(self, tokens):
        """
        :param tokens:
        :return:
        """
        return self.vocabulary.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids):
        """
        :param ids:
        :return:
        """
        return self.vocabulary.convert_ids_to_tokens(ids)
