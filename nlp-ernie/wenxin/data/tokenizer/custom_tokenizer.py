# -*- coding: utf-8 -*
"""
:py:class:`CustomTokenizer`
"""
from ...common.register import RegisterSet
from .tokenizer import Tokenizer
from ...utils.util_helper import convert_to_unicode


@RegisterSet.tokenizer.register
class CustomTokenizer(Tokenizer):
    """CustomTokenizer:用户自己分好词的明文，该tokenizer只负责按某个分隔符(如" ")分成一个list"""

    def __init__(self, vocab_file, split_char=" ", unk_token="[UNK]", params=None):
        """
        :param vocab_file: 词表文件路径
        :param split_char: 明文分隔符，默认是空格
        """
        Tokenizer.__init__(self, vocab_file, split_char, unk_token, params)
        self.split_char = split_char

    def tokenize(self, text):
        """
        :param text:
        :return:
        """
        text = convert_to_unicode(text)
        split_tokens = text.split(self.split_char)
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
