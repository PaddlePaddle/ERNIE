# -*- coding: utf-8 -*
"""
Doie Basic Tokenizer
"""

from __future__ import unicode_literals
import os
import sys
import six
import collections
import unicodedata
import json
import codecs
from erniekit.common.register import RegisterSet

@RegisterSet.tokenizer.register
class DoieBasicTokenizer(object):
    """
    Doie Basic Tokenizer
    """
    CHAR_TOKENIZE = 1
    WORDPIECE_TOKENIZE = 2

    def __init__(self, 
                 vocab_file,
                 unk_token="[UNK]",
                 keep_whitespace=True,
                 merge_whitespace=False,
                 whitespace_token="[PAD]",
                 do_lower_case=True,
                 do_remove_accent=False,
                 eng_word_tokenizer="wordpiece",
                 num_word_tokenizer="wordpiece",
                 max_input_chars_per_word=100,
                 do_char_translate=False,
                 translate_dict=None):

        self.vocab = {}
        self.inv_vocab = {}
        self.load_vocab(vocab_file)

        self.unk_token = unk_token
        if isinstance(unk_token, int):
            self.unk_id = unk_token
        else:
            self.unk_id = self.vocab[unk_token]
        
        self.keep_whitespace = keep_whitespace
        self.merge_whitespace = merge_whitespace
        self.whitespace_token = whitespace_token
        if keep_whitespace:
            if isinstance(whitespace_token, int):
                self.whitespace_id = whitespace_token
            else:
                self.whitespace_id = self.vocab[whitespace_token]

        self.do_lower_case = do_lower_case
        self.do_remove_accent = do_remove_accent
        
        self.eng_word_tokenize_type = 0
        if eng_word_tokenizer == 'char':
            self.eng_word_tokenize_type = self.CHAR_TOKENIZE
        elif eng_word_tokenizer == 'wordpiece':
            self.eng_word_tokenize_type = self.WORDPIECE_TOKENIZE
        self.num_word_tokenize_type = 0
        if num_word_tokenizer == 'char':
            self.num_word_tokenize_type = self.CHAR_TOKENIZE
        elif num_word_tokenizer == 'wordpiece':
            self.num_word_tokenize_type = self.WORDPIECE_TOKENIZE
        
        self.max_input_chars_per_word = max_input_chars_per_word
        self.do_char_translate = do_char_translate
        self.translate_dict = {}
        
        if translate_dict:
            self.translate_dict.update(translate_dict)

        #self.token_cls = collections.namedtuple("Token", ['token', 'orig_token', 'offset'])
        self.token_cls = Token

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        self.vocab = {}
        self.inv_vocab = {}
        if not vocab_file or not os.path.exists(vocab_file):
            raise ValueError("%s: vocab_file not existed: %s" % (self.__class__.__name__, vocab_file))
        with codecs.open(vocab_file, encoding='utf8') as fin:
            for num, line in enumerate(fin):
                items = line.strip().split("\t")
                if len(items) > 2:
                    break
                token = items[0]
                index = items[1] if len(items) == 2 else num
                index = int(index)
                token = token.strip()
                self.vocab[token] = index
                self.inv_vocab[index] = token

    def is_whitespace(self, char):
        """
        is whitespace
        """
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        if len(char) == 1:
            cat = unicodedata.category(six.text_type(char))
            if cat == "Zs":
                return True
        return False

    def is_alpha(self, char):
        """
        is_alpha
        """
        if 'a' <= char <= 'z':
            return True
        if 'A' <= char <= 'Z':
            return True
        return False

    def remove_char_accent(self, char):
        """
        remove_char_accent
        """
        return unicodedata.normalize("NFD", char)[0]
    
    def remove_accents(self, text):
        """remove accents"""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output) 

    def tokenize(self, text, extra_info=False):
        """tokenize"""
        if not isinstance(text, six.text_type):
            raise ValueError("tokenizer input text must be unicode object")
        tokens, orig_tokens, offsets = [], [], []
        lst_char = None
        for offset, orig_char in enumerate(text):
            char = orig_char
            if self.do_char_translate:
                char = self.translate_dict.get(char, orig_char)
            if self.do_remove_accent:
                char = self.remove_char_accent(char)
            if self.do_lower_case:
                char = char.lower()
            if self.is_whitespace(char):
                if not self.keep_whitespace:
                    lst_char = char
                    continue
                elif self.merge_whitespace and self.is_whitespace(lst_char):
                    tokens[-1] += char
                    orig_tokens[-1] += orig_char
                    lst_char = char
                    continue
            elif self.is_alpha(char):
                if self.eng_word_tokenize_type != self.CHAR_TOKENIZE and lst_char and self.is_alpha(lst_char):
                    tokens[-1] += char
                    orig_tokens[-1] += orig_char
                    lst_char = char
                    continue
            elif char.isdigit():
                if self.num_word_tokenize_type != self.CHAR_TOKENIZE and lst_char and lst_char.isdigit():
                    tokens[-1] += char
                    orig_tokens[-1] += orig_char
                    lst_char = char
                    continue
            
            if tokens and len(tokens[-1]) > 1:
                do_wordpiece = False
                lst_token = tokens[-1]
                if self.eng_word_tokenize_type == self.WORDPIECE_TOKENIZE and self.is_alpha(lst_token[0]):
                    do_wordpiece = True
                if self.num_word_tokenize_type == self.WORDPIECE_TOKENIZE and lst_token[0].isdigit():
                    do_wordpiece = True
                if do_wordpiece:
                    sub_tokens = self.wordpiece_for_single_token(lst_token)
                    if len(sub_tokens) > 1:
                        tokens.pop()
                        lst_orig_token = orig_tokens.pop()
                        lst_offset = offsets.pop()
                        sub_offset = 0
                        for i, sub_token in enumerate(sub_tokens):
                            sub_token_len = len(sub_token) if i == 0 else len(sub_token) - 2
                            tokens.append(sub_token)
                            orig_tokens.append(lst_orig_token[sub_offset:sub_offset + sub_token_len])
                            offsets.append(lst_offset + sub_offset)
                            sub_offset += sub_token_len

            tokens.append(char)
            orig_tokens.append(orig_char)
            offsets.append(offset)
            lst_char = char

        if extra_info:
            ex_tokens = []
            for token, orig_token, offset in zip(tokens, orig_tokens, offsets):
                ex_token = self.token_cls(token=token, orig_token=orig_token, offset=offset)
                ex_tokens.append(ex_token)
            return ex_tokens

        return zip(tokens, orig_tokens, offsets)

    def wordpiece_for_single_token(self, token):
        """
        wordpiece_tokenize for single token without UNK replaced
        """
        if len(token) > self.max_input_chars_per_word:
            return [token]
        is_bad = False
        start = 0
        sub_tokens = []
        while start < len(token):
            end = len(token)
            cur_substr = None
            while start < end:
                substr = token[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                is_bad = True
                break
            sub_tokens.append(cur_substr)
            start = end
        if is_bad:
            return [token]
        return sub_tokens

    def convert_tokens_to_ids(self, tokens):
        """
        convert_tokens_to_ids
        """
        output = []
        for token in tokens:
            if isinstance(token, self.token_cls):
                token = token.token
            if self.is_whitespace(token[0]):
                output.append(self.whitespace_id)
            else:
                output.append(self.vocab.get(token, self.unk_id))
        return output

    def convert_ids_to_tokens(self, token_ids):
        """
        convert_ids_to_tokens
        """
        output = []
        for token_id in token_ids:
            output.append(self.inv_vocab[token_id])
        return output

    def insert_token(self, ex_tokens, index, token, orig_token):
        """
        insert token into ex_tokens
        """
        offset = 0
        if index >= len(ex_tokens):
            if len(ex_tokens) != 0:
                offset = ex_tokens[-1].end
            else:
                offset = 0
        else:
            offset = ex_tokens[index].start
        ex_tokens.insert(index, self.token_cls(token, orig_token, offset))
        tk_len = len(orig_token)
        if tk_len > 0:
            for i in range(index + 1, len(ex_tokens)):
                ex_tokens[i].offset += tk_len

    def append_token(self, ex_tokens, token, orig_token):
        """
        append_token
        """
        offset = ex_tokens[-1].end if len(ex_tokens) > 0 else 0
        ex_tokens.append(self.token_cls(token, orig_token, offset))

# register
# Tokenizer = DoieBasicTokenizer
class Token(object):
    """Token class"""
    __slots__ = ['token', 'orig_token', 'offset']
    def __init__(self, token, orig_token, offset):
        self.token = token
        self.orig_token = orig_token
        self.offset = offset

    @property
    def end(self):
        """emd offset of token"""
        return self.offset + len(self.orig_token)
    @property
    def start(self):
        """start offset of token"""
        return self.offset
    def __iter__(self):
        return iter((self.token, self.orig_token, self.offset))
    @property
    def __dict__(self):
        return {'token': self.token,
                'orig_token': self.orig_token,
                'offset': self.offset}


def test():
    """
    test
    """
    vocab_file = 'ernie_model/vocab.txt'
    #text = '格 检 查*T36.0℃  *P108次/分    *R20次/分   *BP120/69mmHg一般情况：发育:正常。'
    #text = 'T 36 8℃腋温 P 129次/分 R 32次/分 BP 0/0mmHg WT6 6Kg一般情况：发育正常，营养良好，正常面容，表情自如，自主体位，神志清楚，查体合作。'
    text = '查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部'
    tokenizer = DoieBasicTokenizer(vocab_file,
        keep_whitespace=True,
        merge_whitespace=True,
        whitespace_token=0, 
        eng_word_tokenizer='wordpiece',    
        num_word_tokenizer='wordpiece',
        do_lower_case=True,
        do_remove_accent=True,
        )
    tokens = tokenizer.tokenize(text, extra_info=True)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    rec_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    for (token, orig_token, offset), token_id, rec_token in zip(tokens, token_ids, rec_tokens):
        print("%s\t%s\t%s\t%s\t%s" % (offset, token, orig_token, token_id, rec_token)).encode('gb18030')


if __name__ == '__main__':
    test()
