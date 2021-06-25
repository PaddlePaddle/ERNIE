# -*- coding: utf-8 -*-
"""
Byte pair encoding utilities from GPT-2.

Original source: https://github.com/openai/gpt-2/blob/master/src/encoder.py
Original license: MIT
"""

#from functools import lru_cache
try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache
import json
import six
import regex as re


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    if six.PY2:
        bs = list(range(ord("!".decode('utf8')), ord("~".decode('utf8'))+1))+list(range(ord("¡".decode('utf8')),ord("¬".decode('utf8'))+1))+list(range(ord("®".decode('utf8')), ord("ÿ".decode('utf8'))+1))
    else:
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )        
    cs = bs[:]
    
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1

    if six.PY2:
        cs = [unichr(n) for n in cs]
    else:
        cs = [chr(n) for n in cs]

    ddict = dict(zip(bs, cs))
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder(object):

    def __init__(self, encoder, bpe_merges, errors='replace', special_tokens=["[SEP]", "[p]", "[q]", "[/q]"]):
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        # print('111',self.byte_encoder)
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.re = re
        self.special_tokens = special_tokens

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = self.re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    # def tokenize(self, text):
    #     tokens = []
    #     return self.re.findall(self.pat, text)
        
    # def tokenize_bpe(self, token):
    #     token = ''.join(self.byte_encoder[ord(b)] for b in token.encode('utf-8'))
    #     return [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]


    # def encode(self, text):
    #     bpe_tokens = []
    #     for token in self.re.findall(self.pat, text):
    #         #print(token)
    #         #print(self.byte_encoder)
    #         token = ''.join(self.byte_encoder[ord(b)] for b in token.encode('utf-8'))
    #         bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
    #     return bpe_tokens

    # def decode(self, tokens):
    #     text = ''.join([self.decoder[token] for token in tokens])
    #     text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
    #     return text

    def tokenize(self, text):
        tokens = text.split(' ')
        sub_tokens = []
        for token_i, token in enumerate(tokens):
            if self.is_special_token(token):
                if token_i == 0:
                    sub_tokens.extend([token])
                else:
                    sub_tokens.extend([" " + token])
            else:
                if token_i == 0:
                    sub_tokens.extend(self.re.findall(self.pat, token))
                else:
                    sub_tokens.extend(self.re.findall(self.pat, " " + token))
        return sub_tokens

    def tokenize_old(self, text):
        return self.re.findall(self.pat, text)

    def is_special_token(self, tok):
        if isinstance(tok, int):
            return False
        res = False
        for t in self.special_tokens:
            # if tok.find(t) != -1:
            if tok.strip() == t:
                res= True
                break
        return res

    def tokenize_bpe(self, token):
        
        if self.is_special_token(token):
            return [token.strip()] # remove space for convert_to_ids
        else:
            if six.PY2:
                token = ''.join(self.byte_encoder[ord(b)] for b in token.encode('utf-8'))
            else:
                token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            return [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]

    def encode(self, text):
        # bpe_tokens = []
        # for token in self.re.findall(self.pat, text):
        #     #print(token)
        #     #print(self.byte_encoder)
        #     token = ''.join(self.byte_encoder[ord(b)] for b in token.encode('utf-8'))
        #     bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        # return bpe_tokens
        bpe_tokens = []
        for token in self.tokenize(text):
            bpe_tokens.extend(self.tokenize_bpe(token))
        return bpe_tokens

    def decode(self, tokens):
        pre_token_i = 0
        texts = []
        for token_i, token in enumerate(tokens):
            if self.is_special_token(token):
                # proprecess tokens before token_i
                if token_i - pre_token_i > 0:
                    text = ''.join([self.decoder[int(tok)] for tok in tokens[pre_token_i:token_i]])
                    text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
                    texts.append(text)
                # texts.append(token)
                if token_i == 0:
                    texts.append(token) # in the beginning, there is no space before special tokens
                else:
                    texts.extend([" ", token]) # in middle sentence, there must be a space before special tokens
                pre_token_i = token_i + 1
                
        if pre_token_i < len(tokens):
            text = ''.join([self.decoder[int(tok)] for tok in tokens[pre_token_i:]])
            text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
            texts.append(text)

        return ''.join(texts)


def get_encoder(encoder_json_path, vocab_bpe_path):
    with open(encoder_json_path, 'r') as f:
        encoder = json.load(f)
    with open(vocab_bpe_path, 'r') as f:
        bpe_data = f.read()
    if six.PY2:
        bpe_data = bpe_data.decode('utf8')
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]

    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")
