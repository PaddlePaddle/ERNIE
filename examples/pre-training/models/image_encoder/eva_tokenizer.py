# !/usr/bin/env python3
""" CLIP tokenizer

Copied from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import gzip
import html
import os
from functools import lru_cache
from typing import Union, List

import ftfy
import regex as re
import paddle
import numpy as np
from easydict import EasyDict as edict

# https://stackoverflow.com/q/62691279
import logging

logger = logging.getLogger()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@lru_cache()
def default_bpe():
    """dummy"""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz"
    )


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
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
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


def basic_clean(text):
    """dummy"""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    """dummy"""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    """dummy"""

    def __init__(self, bpe_path: str = default_bpe(), special_tokens=None):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        if not special_tokens:
            special_tokens = ["<start_of_text>", "<end_of_text>"]
        else:
            special_tokens = ["<start_of_text>", "<end_of_text>"] + special_tokens
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t: t for t in special_tokens}
        special = "|".join(
            [i.replace("/", "\/").replace("|", "\|") for i in special_tokens]
        )
        logger.info(f"[BPE Tokenizer] special tokens: {special}")
        self.pat = re.compile(
            special
            + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}<>]+""",
            re.IGNORECASE,
        )

        self.pad_token = "!"  # TODO: DEBUG HERE!
        self.pad_token_id = 0
        self.ignored_index = -100
        self.eos_token_id = self.encoder["<end_of_text>"]
        self.cls_token_id = self.encoder["<start_of_text>"]

        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]

    def bpe(self, token):
        """dummy"""
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
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

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
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
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text, add_special_tokens=False):
        """dummy"""
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )

        if add_special_tokens:
            sot_token = self.encoder["<start_of_text>"]
            eot_token = self.encoder["<end_of_text>"]
            all_tokens = [sot_token] + bpe_tokens + [eot_token]

        return bpe_tokens

    def decode(self, tokens):
        """dummy"""
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text

    def batch_decode(self, list_of_tokens):
        """dummy"""
        return [self.decode(tokens) for tokens in list_of_tokens]

    def get_vocab(self):
        """dummy"""
        return self.encoder

    def __call__(
        self,
        texts,
        max_length=None,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    ):
        """dummy"""
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.encoder["<start_of_text>"]
        eot_token = self.encoder["<end_of_text>"]
        all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
        if max_length is None:
            max_length = max([len(text) for text in texts])

        if return_tensors is None:
            result = paddle.zeros((len(all_tokens), max_length), dtype="int64")
        elif return_tensors == "np":
            result = np.zeros((len(all_tokens), max_length), dtype="int64")
        else:
            raise ValueError("return_tensors must be either 'pt' or 'np'.")

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > max_length:
                tokens = tokens[:max_length]  # Truncate
                tokens[-1] = eot_token
            result[i, : len(tokens)] = tokens

        return edict(
            {"input_ids": result}
        )  # TODO: add the rests(position ids, attnmask, etc)


class HFTokenizer:
    "HuggingFace tokenizer wrapper"

    def __init__(self, tokenizer_name: str):
        """dummy"""
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, texts: Union[str, List[str]], context_length: int = 77):
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]
        texts = [whitespace_clean(basic_clean(text)) for text in texts]
        input_ids = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=context_length,
            padding="max_length",
            truncation=True,
        ).input_ids
        return input_ids
