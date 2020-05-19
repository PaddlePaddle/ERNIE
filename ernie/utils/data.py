import sys
import numpy as np
import re
from propeller import log
import itertools
from propeller.paddle.data import Dataset
import pickle

import six

if six.PY2:
    import operator
    def accumulate(iterable, func=operator.add, initial=None):
        'Return running totals'
        # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
        # accumulate([1,2,3,4,5], initial=100) --> 100 101 103 106 110 115
        # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
        it = iter(iterable)
        total = initial
        if initial is None:
            try:
                total = next(it)
            except StopIteration:
                return
        yield total
        for element in it:
            total = func(total, element)
            yield total
else:
    from itertools import accumulate


max_input_chars_per_word=100
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def wordpiece(token, vocab, unk_token, sentencepiece_style_vocab=False):
    """call with single word"""
    chars = list(token)
    if len(chars) > max_input_chars_per_word:
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
            if start == 0 and sentencepiece_style_vocab:
                substr = u'\u2581' + substr
            if start > 0 and not sentencepiece_style_vocab:
                substr = "##" + substr
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


class SpaceTokenizer(object):
    def __init__(self, vocab, lower=True):
        """
        char tokenizer (wordpiece english)
        normed txt(space seperated or not) => list of word-piece
        """
        self.vocab = set(vocab)
        self.lower = lower

    def __call__(self, sen):
        if len(sen) == 0:
            return [] #empty line
        sen = sen.decode('utf8')
        if self.lower:
            sen = sen.lower()
        res = []
        for s in sen.split(' '):
            if s == ' ':
                continue
            if s in self.vocab:
                res.append(s)
            else:
                res.append('[UNK]')
        return res


class CharTokenizer(object):
    def __init__(self, vocab, lower=True, sentencepiece_style_vocab=False):
        """
        char tokenizer (wordpiece english)
        normed txt(space seperated or not) => list of word-piece
        """
        self.vocab = set(vocab)
        #self.pat = re.compile(r'([,.!?\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b]|[\u4e00-\u9fa5]|[a-zA-Z0-9]+)')
        self.pat =  re.compile(r'([a-zA-Z0-9]+|\S)')
        self.lower = lower
        self.sentencepiece_style_vocab = sentencepiece_style_vocab

    def __call__(self, sen):
        if len(sen) == 0:
            return [] #empty line
        sen = sen.decode('utf8')
        if self.lower:
            sen = sen.lower()
        res = []
        for match in self.pat.finditer(sen):
            words, _ = wordpiece(match.group(0), vocab=self.vocab, unk_token='[UNK]', sentencepiece_style_vocab=self.sentencepiece_style_vocab)
            res.extend(words)
        return res


class WSSPTokenizer(object):
    def __init__(self, sp_model_dir, word_dict, ws=True, lower=True):
        self.ws = ws
        self.lower = lower
        self.dict = pickle.load(open(word_dict, 'rb'), encoding='utf8')
        import sentencepiece as spm
        self.sp_model = spm.SentencePieceProcessor()
        self.window_size = 5
        self.sp_model.Load(sp_model_dir)

    def cut(self, chars):
        words = []
        idx = 0
        while idx < len(chars):
            matched = False
            for i in range(self.window_size, 0, -1):
                cand = chars[idx: idx+i]
                if cand in self.dict:
                    words.append(cand)
                    matched = True
                    break
            if not matched: 
                i = 1
                words.append(chars[idx])
            idx += i
        return words
 
    def __call__(self, sen):
        sen = sen.decode('utf8')
        if self.ws:
            sen = [s for s in self.cut(sen) if s != ' ']
        else:
            sen = sen.split(' ')
        if self.lower:
            sen = [s.lower() for s in sen]
        sen = ' '.join(sen)
        ret = self.sp_model.EncodeAsPieces(sen)
        return ret


def build_2_pair(seg_a, seg_b, max_seqlen, cls_id, sep_id):
    token_type_a = np.ones_like(seg_a, dtype=np.int64) * 0
    token_type_b = np.ones_like(seg_b, dtype=np.int64) * 1
    sen_emb = np.concatenate([[cls_id], seg_a, [sep_id], seg_b, [sep_id]], 0)
    token_type_emb = np.concatenate([[0], token_type_a, [0], token_type_b, [1]], 0)

    seqlen = sen_emb.shape[0]
    #random truncate
    random_begin = 0 #np.random.randint(0, np.maximum(0, seqlen - max_seqlen) + 1,)
    sen_emb = sen_emb[random_begin: random_begin + max_seqlen]
    token_type_emb = token_type_emb[random_begin: random_begin + max_seqlen]

    return sen_emb, token_type_emb


def build_1_pair(seg_a, max_seqlen, cls_id, sep_id):
    token_type_a = np.ones_like(seg_a, dtype=np.int64) * 0

    sen_emb = np.concatenate([[cls_id], seg_a, [sep_id]], 0)
    token_type_emb = np.concatenate([[0], token_type_a, [0]], 0)

    seqlen = sen_emb.shape[0]
    #random truncate
    random_begin = 0 #np.random.randint(0, np.maximum(0, seqlen - max_seqlen) + 1,)

    sen_emb = sen_emb[random_begin: random_begin + max_seqlen]
    token_type_emb = token_type_emb[random_begin: random_begin + max_seqlen]
    return sen_emb, token_type_emb


def expand_dims(*args):
    func = lambda i: np.expand_dims(i, -1)
    ret = [func(i) for i in args]
    return ret


def interleave(ds1, ds2):
    def gen():
        for i, j in six.moves.zip_longest(iter(ds1), iter(ds2)):
            if i is not None:
                yield i
            if j is not None:
                yield j
    return Dataset.from_generator_func(gen)

