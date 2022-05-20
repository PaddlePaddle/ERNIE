# -*- coding: utf-8 -*
"""
:py:class:`LACTokenizer`
"""
import six

from ...common.register import RegisterSet
from .tokenizer import Tokenizer
from ...utils.util_helper import convert_to_unicode


@RegisterSet.tokenizer.register
class LACTokenizer(Tokenizer):
    """使用lac进行切词"""

    def __init__(self, vocab_file, split_char=" ", unk_token="[UNK]", params=None):
        """
        :param vocab_file: 词表文件路径
        :param split_char: 明文分隔符，默认是空格
        """
        Tokenizer.__init__(self, vocab_file, split_char, unk_token, params)
        self.split_char = split_char
        from LAC import LAC
        self.lac = LAC(mode="seg")

    def tokenize(self, text):
        """
        :param text:
        :return:
        """
        text = convert_to_unicode(text)
        if len(text) == 0:
            return text

        split_tokens = self.lac.run(text)
        if six.PY2:
            split_tokens = [w[0] for w in split_tokens]

        return split_tokens

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
