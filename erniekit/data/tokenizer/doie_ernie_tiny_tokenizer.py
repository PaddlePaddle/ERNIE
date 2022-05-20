# -*- coding: utf-8 -*
"""
Doie Basic Tokenizer
"""

from __future__ import unicode_literals
import os
import sys
import six
if six.PY3:
    import pickle
else:
    import cPickle as pickle
import collections
import unicodedata
import json
import codecs
import sentencepiece as sp
from .doie_basic_tokenizer import DoieBasicTokenizer
from erniekit.common.register import RegisterSet

@RegisterSet.tokenizer.register
class DoieErnieTinyTokenizer(DoieBasicTokenizer):
    """
    Doie ERNIE tiny Tokenizer
    """

    def __init__(self,
                 vocab_file,
                 need_sp=True,
                 wordseg_dict=None,
                 sp_model_dir=None,
                 **kwargs):

        DoieBasicTokenizer.__init__(self, vocab_file, **kwargs)

        self.sp_model = sp.SentencePieceProcessor()
        self.window_size = 5
        self.do_sp = need_sp
        if six.PY3:
            self.dict = pickle.load(open(wordseg_dict, 'rb'), encoding='utf8')
        else:
            self.dict = pickle.load(open(wordseg_dict, 'rb'))
        self.sp_model.Load(sp_model_dir)

    def cut(self, chars):
        """cut"""
        words = []
        idx = 0
        while idx < len(chars):
            matched = False
            if self.is_whitespace(chars[idx]) and self.keep_whitespace:
                i = 1
                matched = True
                words.append(chars[idx])
            else:
                for i in range(self.window_size, 0, - 1):
                    cand = chars[idx: idx + i]
                    if cand in self.dict:
                        words.append(cand)
                        matched = True
                        break
            if not matched:
                i = 1
                words.append(chars[idx])
            idx += i
        return words

    def tokenize(self, text, extra_info=False):
        """
        :param text:
        :return:
        """
        # print("-----")
        # print(text.encode("utf8"))
        tokens, orig_tokens, offsets = [], [], []
        # text = [s for s in self.cut(text) if s != ' ']
        text = self.cut(text)
        if self.do_lower_case:
            text = [s.lower() for s in text]
        for i, t in enumerate(text):
            if t == " ":
                text[i] = u"是"
        if self.do_sp:
            text = ' '.join(text)
            text = self.sp_model.EncodeAsPieces(text)
        for i, t in enumerate(text):
            if t == u"▁是":
                text[i] = self.whitespace_token
        offset = 0
        for index, token in enumerate(text):
            ori_token = token.strip(u"▁")
            tokens.append(token)
            orig_tokens.append(ori_token)
            offsets.append(offset)
            if ori_token == self.whitespace_token:
                offset += 1
            else:
                offset += len(ori_token)

        if extra_info:
            ex_tokens = []
            for token, orig_token, offset in zip(tokens, orig_tokens, offsets):
                ex_token = self.token_cls(token=token, orig_token=orig_token, offset=offset)
                ex_tokens.append(ex_token)
            return ex_tokens

        return zip(tokens, orig_tokens, offsets)

