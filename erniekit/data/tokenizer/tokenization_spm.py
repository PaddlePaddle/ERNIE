# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
# coding=utf-8
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
from ...common.register import RegisterSet
from .tokenization_utils import PreTrainedFullTokenizer
from .tokenization_utils import convert_by_vocab, printable_text, convert_to_unicode
from .tokenization_utils import _is_whitespace, _is_control, _is_punctuation
import six
from six.moves import range

import sentencepiece as spm

SPIECE_UNDERLINE = u"▁".encode("utf-8")


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
  """Checks whether the casing config is consistent with the checkpoint name."""

  # The casing has to be passed in by the user and there is no explicit check
  # as to whether it matches the checkpoint. The casing information probably
  # should have been stored in the bert_config.json file, but it's not, so
  # we have to heuristically detect it to validate.

  if not init_checkpoint:
    return

  m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt",
               six.ensure_str(init_checkpoint))
  if m is None:
    return

  model_name = m.group(1)

  lower_models = [
      "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
      "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
  ]

  cased_models = [
      "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
      "multi_cased_L-12_H-768_A-12"
  ]

  is_bad_config = False
  if model_name in lower_models and not do_lower_case:
    is_bad_config = True
    actual_flag = "False"
    case_name = "lowercased"
    opposite_flag = "True"

  if model_name in cased_models and do_lower_case:
    is_bad_config = True
    actual_flag = "True"
    case_name = "cased"
    opposite_flag = "False"

  if is_bad_config:
    raise ValueError(
        "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
        "However, `%s` seems to be a %s model, so you "
        "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
        "how the model was pre-training. If this error is wrong, please "
        "just comment out this check." % (actual_flag, init_checkpoint,
                                          model_name, case_name, opposite_flag))


def clean_text(text):
  """Performs invalid character removal and whitespace cleanup on text."""
  text = text.replace(u"“", u'"')\
          .replace(u'”', u'"')\
          .replace(u'‘', "'")\
          .replace(u'’', u"'")\
          .replace(u'—', u'-')

  output = []
  for char in text:
      if _is_control(char):
          continue
      if _is_whitespace(char):
          output.append(" ")
      else:
          output.append(char)
  return "".join(output)


def preprocess_text(inputs, remove_space=True, lower=False):
  """preprocess data by removing extra space and normalize data."""

  outputs = inputs
  if remove_space:
    outputs = " ".join(inputs.strip().split())

  if six.PY2 and isinstance(outputs, str):
    try:
      outputs = six.ensure_text(outputs, "utf-8")
    except UnicodeDecodeError:
      outputs = six.ensure_text(outputs, "latin-1")

  outputs = unicodedata.normalize("NFKD", outputs)
  outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
  if lower:
    outputs = outputs.lower()

  return outputs


def encode_pieces(sp_model, text, return_unicode=True, sample=False):
  """turn sentences into word pieces."""

  # liujiaxiang: add for ernie-albert, mainly consider for “/”/‘/’/— causing too many unk
  text = clean_text(text)

  if six.PY2 and isinstance(text, six.text_type):
    text = six.ensure_binary(text, "utf-8")

  if not sample:
    pieces = sp_model.EncodeAsPieces(text)
  else:
    pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)

  new_pieces = []
  for piece in pieces:
    piece = printable_text(piece)
    if len(piece) > 1 and piece[-1] == "," and piece[-2].isdigit():
      cur_pieces = sp_model.EncodeAsPieces(
          six.ensure_binary(piece[:-1]).replace(SPIECE_UNDERLINE, b""))
      if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
        if len(cur_pieces[0]) == 1:
          cur_pieces = cur_pieces[1:]
        else:
          cur_pieces[0] = cur_pieces[0][1:]
      cur_pieces.append(piece[-1])
      new_pieces.extend(cur_pieces)
    else:
      new_pieces.append(piece)

  # note(zhiliny): convert back to unicode for py2
  if six.PY2 and return_unicode:
    ret_pieces = []
    for piece in new_pieces:
      if isinstance(piece, str):
        piece = six.ensure_text(piece, "utf-8")
      ret_pieces.append(piece)
    new_pieces = ret_pieces

  return new_pieces


def encode_ids(sp_model, text, sample=False):
  """encode ids"""
  pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
  ids = [sp_model.PieceToId(piece) for piece in pieces]
  return ids

@RegisterSet.tokenizer.register
class FullTokenizerSpm(PreTrainedFullTokenizer):
  """Runs end-to-end tokenziation."""
  def __init__(self, vocab_file, split_char=" ", unk_token="[UNK]", params=None):
    do_lower_case = params.get("do_lower_case", False)
    assert params.get("spm_model_file", False), "params must have spm_model_file"
    spm_model_file = params.get("spm_model_file")
    super(FullTokenizerSpm, self).__init__(
      vocab_file=vocab_file,
      do_lower_case=do_lower_case,
    )
    self.sp_model = None
    if spm_model_file:
      self.sp_model = spm.SentencePieceProcessor()
      #tf.logging.info("loading sentence piece model")
      self.sp_model.Load(spm_model_file)
      # Note(mingdachen): For the purpose of consisent API, we are
      # generating a vocabulary for the sentence piece tokenizer.
      self.vocab = {self.sp_model.IdToPiece(i): i for i
                    in range(self.sp_model.GetPieceSize())}
     # import pdb; pdb.set_trace()
    else:
      #self.vocab = load_vocab(vocab_file)
      #self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
      #self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
      # (liujiaxiang) comment useless code for a better diff code
      raise ValueError('albert use spm by default')
    self.inv_vocab = {v: k for k, v in self.vocab.items()}

  def tokenize(self, text):
    """tokenize text"""
    if self.sp_model:
      split_tokens = encode_pieces(self.sp_model, text, return_unicode=False)
    else:
      #split_tokens = []
      #for token in self.basic_tokenizer.tokenize(text):
      #  for sub_token in self.wordpiece_tokenizer.tokenize(token):
      #    split_tokens.append(sub_token)
      # (liujiaxiang) comment useless code for a better diff code
      raise ValueError('albert use spm by default')

    return split_tokens

  def tokenize_for_pretrain(self, tok_list):
    """tokenize text for pretrain"""
    import tok as tok_protocol
    text = " ".join([t.token for t in tok_list])

    #split_tokens = encode_pieces(self.sp_model, text, return_unicode=True)
    split_tokens = encode_pieces(self.sp_model, text, return_unicode=False)
    ids = self.convert_tokens_to_ids(split_tokens)

    # +1 for head _ : 'hello world' -> ['_hello', '_world']

    if not (len(preprocess_text(''.join(split_tokens))) == len(text) + 1):
      return None

    if len(split_tokens) != len(ids):
      return None

    sent_piece_tokens = []
    i = 0
    position_to_nth = self.inverse_index_str("_" + text)
    for t, id in zip(split_tokens, ids):
      t = t.decode('utf8')
      nth = position_to_nth[i]
      token = tok_list[nth]

      tok = tok_protocol.Tok()
      tok.token = t
      tok.id = id
      tok.bio = token.bio
      tok.origin = token.origin
      tok.appear = token.appear
      i += len(t)
      sent_piece_tokens.append(tok)

    return sent_piece_tokens

  def inverse_index_str(self, s):
    """inverse the str of index"""
    nth_tok = 0
    position_to_nth = {}
    for i, c in enumerate(s):
        if c == " ":
            nth_tok += 1
        position_to_nth[i] = nth_tok
    return position_to_nth

  def convert_tokens_to_ids(self, tokens):
    """ get ids of tokens"""
    if self.sp_model:
      #tf.logging.info("using sentence piece tokenzier.")
      return [self.sp_model.PieceToId(
          printable_text(token)) for token in tokens]
    else:
      return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    """get tokens of ids"""
    if self.sp_model:
      #tf.logging.info("using sentence piece tokenzier.")
      return [self.sp_model.IdToPiece(id_) for id_ in ids]
    else:
      return convert_by_vocab(self.inv_vocab, ids)

  def merge_subword(self, tokens):
    """
    :param tokens:
    :return: merged_tokens
    """
    ret = []
    for token in tokens:
      if token.startswith(u"▁"):
        ret.append(token[1:])
      else:
        if len(ret):
          ret[-1] += token
        else:
          ret.append(token)

    ret = [token for token in ret if token]
    return ret
