# coding=utf-8
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sentencepiece as sp

from ...common.register import RegisterSet
from .tokenizer import Tokenizer
from ...utils.util_helper import convert_to_unicode


@RegisterSet.tokenizer.register
class SentencepieceTokenizer(Tokenizer):
    """Runs SentencePiece tokenziation."""

    def __init__(self, vocab_file, split_char=" ", unk_token="[UNK]", params=None):
        Tokenizer.__init__(self, vocab_file, split_char, unk_token=unk_token, params=params)
        self.do_lower_case = True
        if params:
            self.do_lower_case = params["do_lower_case"]
        self.tokenizer = sp.SentencePieceProcessor()
        self.tokenizer.Load(vocab_file + ".model")
        self.sp_unk_token = "<unk>"
        self.unk_token = unk_token

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        Returns:
            A list of wordpiece tokens.
        """
        text = text.lower() if self.do_lower_case else text 
        text = convert_to_unicode(text.replace("\1", " "))
        tokens = self.tokenizer.EncodeAsPieces(text)
        
        output_tokens = []
        for token in tokens:
            if token == self.sp_unk_token:
                token = self.unk_token
            
            if token in self.vocabulary.vocab_dict:
                output_tokens.append(token)
            else:
                output_tokens.append(self.unk_token)
        
        return output_tokens
    
    def convert_tokens_to_ids(self, tokens):
        """convert tokens to ids"""
        return self.vocabulary.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids):
        """convert ids to tokens"""
        return self.vocabulary.convert_ids_to_tokens(ids)


@RegisterSet.tokenizer.register
class WordsegTokenizer(SentencepieceTokenizer):
    """Runs Wordseg tokenziation."""

    def __init__(self, vocab_file, split_char="\1", unk_token="[UNK]", params=None): 
        Tokenizer.__init__(self, vocab_file, split_char, unk_token=unk_token, params=params)
        self.tokenizer = sp.SentencePieceProcessor()
        self.tokenizer.Load(vocab_file + ".model")
        
        self.do_lower_case = True
        if params:
            self.do_lower_case = params["do_lower_case"]
        self.unk_token = unk_token
        self.split_token = split_char

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        Returns:
            A list of wordpiece tokens.
        """
        text = text.lower() if self.do_lower_case else text 
        text = convert_to_unicode(text)
        
        output_tokens = []
        for token in text.split(self.split_token):
            if token in self.vocabulary.vocab_dict:
                output_tokens.append(token)
            else:
                sp_tokens = self.tokenizer.EncodeAsPieces(token)
                for sp_token in sp_tokens:
                    if sp_token in self.vocab:
                        output_tokens.append(sp_token)
        return output_tokens
    
    def convert_tokens_to_ids(self, tokens):
        """convert tokens to ids"""
        return self.vocabulary.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids):
        """convert ids to tokens"""
        return self.vocabulary.convert_ids_to_tokens(ids)


def tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""

    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    output = []
    buff = ""
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp):
            if buff != "":
                output.append(buff)
                buff = ""
            output.append(char)
        else:
            buff += char

    if buff != "":
        output.append(buff)

    return output

