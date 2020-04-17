#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
# File: tokenizing_ernie.py
# Author: chenxuyi(chenxuyi@baidu.com)
# Date: 2020/03/30 16:44:48
#
########################################################################
"""
    Comment.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import six
import re
import logging
import tempfile
from functools import partial

from tqdm import tqdm
import numpy as np

from ernie.file_utils import _fetch_from_remote
import io

open = partial(io.open, encoding='utf8')

log = logging.getLogger(__name__)

_max_input_chars_per_word = 100

def _wordpiece(token, vocab, unk_token, prefix='##', sentencepiece_prefix=''):
    """ wordpiece: helloworld => [hello, ##world] """
    chars = list(token)
    if len(chars) > _max_input_chars_per_word:
        return [unk_token], [(0, len(chars))]

    is_bad = False
    start = 0
    sub_tokens = []
    sub_pos = []
    while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
            substr = "".join(chars[start:end])
            if start == 0:
                substr = sentencepiece_prefix + substr
            if start > 0:
                substr = prefix + substr
            if substr in vocab:
                cur_substr = substr
                break
            end -= 1
        if cur_substr is None:
            is_bad = True
            break
        sub_tokens.append(cur_substr)
        sub_pos.append((start, end))
        start = end
    if is_bad:
        return [unk_token], [(0, len(chars))]
    else:
        return sub_tokens, sub_pos


class ErnieTokenizer(object):
    bce = 'https://ernie-github.cdn.bcebos.com/'
    resource_map = {
        'ernie-1.0': bce + 'model-ernie1.0.1.tar.gz',
        'ernie-2.0-en': bce + 'model-ernie2.0-en.1.tar.gz',
        'ernie-2.0-large-en':  bce + 'model-ernie2.0-large-en.1.tar.gz',
        'ernie-tiny': bce + 'model-ernie_tiny.1.tar.gz',
    }
    @classmethod
    def from_pretrained(cls, pretrain_dir_or_url):
        if pretrain_dir_or_url in cls.resource_map:
            url = cls.resource_map[pretrain_dir_or_url]
            log.info('get pretrain dir from %s' % url)
            pretrain_dir = _fetch_from_remote(url)
        else:
            log.info('pretrain dir %s not in %s, read from local' % (pretrain_dir_or_url, repr(cls.resource_map)))
            pretrain_dir = pretrain_dir_or_url
        if not os.path.exists(pretrain_dir):
            raise ValueError('pretrain dir not found: %s' % pretrain_dir)
        vocab_path = os.path.join(pretrain_dir, 'vocab.txt')
        if not os.path.exists(vocab_path):
            raise ValueError('no vocab file in pretrain dir: %s' % pretrain_dir)
        vocab_dict = {j.strip().split('\t')[0]: i for i, j in enumerate(open(vocab_path).readlines())}
        t = cls(vocab_dict)
        return t

    def __init__(self, vocab, unk_token='[UNK]', sep_token='[SEP]', cls_token='[CLS]', pad_token='[PAD]', wordpiece_prefix='##', sentencepiece_prefix='', lower=True, encoding='utf8'):
        if not isinstance(vocab, dict):
            raise ValueError('expect `vocab` to be instance of dict, got %s' % type(vocab))
        self.vocab = vocab
        self.pat = re.compile(r'([a-zA-Z0-9]+|\S)')
        self.lower = lower
        self.prefix = wordpiece_prefix
        self.sentencepiece_prefix = sentencepiece_prefix
        self.pad_id = self.vocab[pad_token]
        self.cls_id = self.vocab[cls_token]
        self.sep_id = self.vocab[sep_token]
        self.unk_id = self.vocab[unk_token]
        self.unk_token = unk_token
        self.encoding = encoding

    def tokenize(self, text):
        if len(text) == 0:
            return []
        if six.PY3 and not isinstance(text, six.string_types):
            text = text.decode(self.encoding)
        if six.PY2 and isinstance(text, str):
            text = text.decode(self.encoding)
        if self.lower:
            text = text.lower()

        res = []
        for match in self.pat.finditer(text):
            words, _ = _wordpiece(match.group(0), vocab=self.vocab, unk_token=self.unk_token, prefix=self.prefix, sentencepiece_prefix=self.sentencepiece_prefix)
            res += words
        return res

    def token_to_ids(self, tokens):
        return [self.vocab.get(t, self.unk_id) for t in tokens]

    def truncate(self, id1, id2, seqlen):
        id1_len = seqlen
        id2_len = max(seqlen - len(id1), 0)
        return id1[: id1_len], id2[: id2_len]

    def build_for_ernie(self, text_id, pair_id=None):
        """build sentence type id, add [CLS] [SEP]"""
        text_id_type = np.zeros_like(text_id)
        ret_id = np.concatenate([[self.cls_id], text_id, [self.sep_id]], 0)
        ret_id_type = np.concatenate([[0], text_id_type, [0]], 0)

        if pair_id is not None:
            pair_id_type = np.ones_like(pair_id)
            ret_id = np.concatenate([ret_id, pair_id, [self.sep_id]], 0)
            ret_id_type = np.concatenate([ret_id_type, pair_id_type, [1]], 0)
        return ret_id, ret_id_type

    def encode(self, text, pair=None, truncate_to=None):
        text_id = np.array(self.token_to_ids(self.tokenize(text)), dtype=np.int64)
        text_id_type = np.zeros_like(text_id)
        if pair is not None:
            pair_id = np.array(self.token_to_ids(self.tokenize(pair)), dtype=np.int64)
        else:
            pair_id = None
        if truncate_to is not None:
            text_id, pair_id = self.truncate(text_id, [] if pair_id is None else pair_id, truncate_to)

        ret_id, ret_id_type = self.build_for_ernie(text_id, pair_id)
        return ret_id, ret_id_type



class ErnieTinyTokenizer(ErnieTokenizer):
    bce = 'https://ernie-github.cdn.bcebos.com/'
    resource_map = {'ernie-tiny': bce + 'model-ernie_tiny.1.tar.gz'}
    @classmethod
    def from_pretrained(cls, pretrain_dir_or_url, force_download=False):
        if pretrain_dir_or_url in cls.resource_map:
            url = cls.resource_map[pretrain_dir_or_url]
            log.info('get pretrain dir from %s' % url)
            pretrain_dir = _fetch_from_remote(url, force_download)
        else:
            log.info('pretrain dir %s not in %s, read from local' % (pretrain_dir_or_url, repr(cls.resource_map)))
            pretrain_dir = pretrain_dir_or_url
        if not os.path.exists(pretrain_dir):
            raise ValueError('pretrain dir not found: %s' % pretrain_dir)
        vocab_path = os.path.join(pretrain_dir, 'vocab.txt')
        sp_model_path = os.path.join(pretrain_dir, 'subword/spm_cased_simp_sampled.model')

        if not os.path.exists(vocab_path):
            raise ValueError('no vocab file in pretrain dir: %s' % pretrain_dir)
        vocab_dict = {j.strip().split('\t')[0]: i for i, j in enumerate(open(vocab_path).readlines())}

        t = cls(vocab_dict, sp_model_path)
        return t

    def __init__(self, vocab, sp_model_path, **kwargs):
        super(ErnieTinyTokenizer, self).__init__(vocab, **kwargs)
        import sentencepiece as spm
        self.sp_model = spm.SentencePieceProcessor()
        self.window_size = 5
        self.sp_model.Load(sp_model_path)
        from LAC import LAC
        self.lac = LAC()

    def cut(self, sentence):
        return self.lac.lexer(sentence)

    def tokenize(self, text):
        if len(text) == 0:
            return []
        if not isinstance(text, six.string_types):
            text = text.decode(self.encoding)
        if self.lower:
            text = text.lower()

        res = []
        for match in self.cut(text):
            res += self.sp_model.EncodeAsPieces(match)
        return res

