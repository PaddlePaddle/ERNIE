# -*- coding: utf-8 -*
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
import unicodedata
import six
from six.moves import range

if six.PY3:
    import pickle
else:
    import cPickle as pickle


from ...common.register import RegisterSet
from .tokenizer import Tokenizer
from ...utils.util_helper import clean_text, convert_to_unicode, whitespace_tokenize
from ...utils.util_helper import is_control, is_whitespace, is_punctuation

import sentencepiece as sp


@RegisterSet.tokenizer.register
class FullTokenizer(Tokenizer):
    """FullTokenizer:basic, sentence_piece, word_piece的总集，按字粒度进行切分
    """
    def __init__(self, vocab_file, split_char=" ", unk_token="[UNK]", params=None):
        Tokenizer.__init__(self, None, split_char, params=params)
        self.basic_tokenizer = BasicTokenizer(vocab_file=None, unk_token=unk_token, params=params)
        if params and params.get("use_sentence_piece_vocab", None):
            self.wordpiece_tokenizer = SentencePieceTokenizer(vocab_file=vocab_file, params=params)
        else:
            self.wordpiece_tokenizer = WordPieceTokenizer(vocab_file=vocab_file, params=params)
        self.vocabulary = self.wordpiece_tokenizer.vocabulary
        self.vocab, self.inv_vocab = self.vocabulary.vocab_dict, self.vocabulary.id_dict

    def tokenize(self, text):
        """
        :param text:
        :return:
        """
        text = convert_to_unicode(text)
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

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

    def merge_subword(self, tokens):
        """
        :param tokens:
        :return: merged_tokens
        """
        ret = []
        for token in tokens:
            if token.startswith("##"):
                real_token = token[2:]
                if len(ret):
                    ret[-1] += real_token
                else:
                    ret.append(real_token)
            else:
                ret.append(token)
        return ret

@RegisterSet.tokenizer.register
class BasicTokenizer(Tokenizer):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, vocab_file, split_char=" ", unk_token="[UNK]", params=None):
        Tokenizer.__init__(self, vocab_file, split_char, unk_token, params)
        self.do_lower_case = True
        if params:
            self.do_lower_case = params.get("do_lower_case", True)

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def convert_tokens_to_ids(self, tokens):
        """
        :param tokens:
        :return:
        """
        pass

    def convert_ids_to_tokens(self, ids):
        """
        :param ids:
        :return:
        """
        pass

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
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

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or is_control(char):
                continue
            if is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


@RegisterSet.tokenizer.register
class WordPieceTokenizer(Tokenizer):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab_file, split_char=" ", unk_token="[UNK]", params=None):
        Tokenizer.__init__(self, vocab_file, split_char, unk_token=unk_token, params=params)
        self.max_input_chars_per_word = 100
        if params:
            self.max_input_chars_per_word = params.get("max_input_chars_per_word", 100)

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
            input = "unaffable"
            output = ["un", "##aff", "##able"]

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through `BasicTokenizer.

        Returns:
            A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocabulary.vocab_dict:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

    def convert_tokens_to_ids(self, tokens):
        """
        :param tokens:
        :return:
        """
        pass

    def convert_ids_to_tokens(self, ids):
        """
        :param ids:
        :return:
        """
        pass


@RegisterSet.tokenizer.register
class SentencePieceTokenizer(WordPieceTokenizer):
    """tokenize"""
    def tokenize(self, text):
        """
        :param text:
        :return:
        """
        text = convert_to_unicode(text)
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start == 0:
                        substr = u'\u2581' + substr
                    if substr in self.vocabulary.vocab_dict:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


@RegisterSet.tokenizer.register
class CharTokenizer(Tokenizer):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, split_char=" ", unk_token="[UNK]", params=None):
        Tokenizer.__init__(self, vocab_file, split_char, unk_token, params)
        self.wordpiece_tokenizer = WordPieceTokenizer(vocab_file=vocab_file, params=None)

    def tokenize(self, text):
        """
        :param text:
        :return:
        """
        text = convert_to_unicode(text)
        split_tokens = []
        for token in text.lower().split(" "):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

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


@RegisterSet.tokenizer.register
class WordsegTokenizer(Tokenizer):
    """Runs Wordseg tokenziation."""
    def __init__(self, vocab_file, split_char="\1", unk_token="[UNK]", params=None):
        Tokenizer.__init__(self, vocab_file, split_char, unk_token, params)
        model_file = params.get("sentence_piece_model", None)
        self.tokenizer = sp.SentencePieceProcessor()
        self.tokenizer.Load(model_file)
        self.do_lower_case = params.get("do_lower_case", True)

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        Returns:
            A list of wordpiece tokens.
        """
        text = text.lower() if self.do_lower_case else text
        text = convert_to_unicode(text)

        output_tokens = []
        for token in text.split(self.split_token):
            if token in self.vocab:
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


@RegisterSet.tokenizer.register
class SentencepieceTokenizerErnie(Tokenizer):
    """Runs SentencePiece tokenziation."""
    def __init__(self, vocab_file, split_char=" ", unk_token="[UNK]", params=None):
        Tokenizer.__init__(self, vocab_file, split_char, unk_token, params)
        model_file = params.get("sentence_piece_model", None)
        self.tokenizer = sp.SentencePieceProcessor()
        self.tokenizer.Load(model_file)
        self.do_lower_case = params.get("do_lower_case", True)
        self.sp_unk_token = "<unk>"

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

            if token in self.vocab:
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
class WSSPTokenizer(Tokenizer):
    """WSSP:basic, sentence_piece, word_piece的总集，按字粒度进行切分
    """
    def __init__(self, vocab_file, split_char, unk_token, params=None):
        super(WSSPTokenizer, self).__init__(vocab_file, split_char, unk_token, params=None)
        self.sp_model = sp.SentencePieceProcessor()
        self.window_size = 5
        self.do_sp = params.get("need_sp", 1)
        if six.PY3:
            self.dict = pickle.load(open(params["wordseg_dict"], 'rb'), encoding='utf8')
        else:
            self.dict = pickle.load(open(params["wordseg_dict"], 'rb'))
        self.sp_model.Load(params["sp_model_dir"])

    def cut(self, chars):
        """cut"""
        chars = convert_to_unicode(chars)
        chars = clean_text(chars)
        words = []
        idx = 0
        while idx < len(chars):
            matched = False
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

    def tokenize(self, sen):
        """
        :param text:
        :return:
        """
        sen = [s for s in self.cut(sen) if s != ' ']
        sen = [s.lower() for s in sen]
        if self.do_sp:
            sen = ' '.join(sen)
            sen = self.sp_model.EncodeAsPieces(sen)
        return sen

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

if __name__ == "__main__":
    vocab_file = "../../../tasks/model_files/dict/vocab_ernie_2.0_large_ch.txt"

    tokenizer = FullTokenizer(vocab_file=vocab_file, split_char=' ')
    file_path = "./train_data"
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            line = line.rstrip()
            fileds = line.split('\t')
            tokens = tokenizer.tokenize(fileds[0])
            ids = tokenizer.convert_tokens_to_ids(tokens)
            print("index: ", index, "\t", "item: ", line, "\ttokens: ", tokens)
            # print("\t\t\t", "ids: ", ids)
            # break
