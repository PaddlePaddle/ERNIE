#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""trigram_blocking for sequence generation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import paddle.fluid as fluid


class TrigramBlocking(object):
    """trigram blocking check data holder
    """

    def __init__(self, init_token, roberta_tokenizer, beam_size, use_fp16=False):
        """use tokenizer to generate the real-tokens from sub-token ids.
        but we can't pass the tokenizer to network, so we need make a trick
        """
        # => [N, T==0, 1]
        self._alive_seq = fluid.layers.fill_constant_batch_size_like(
            input=init_token,
            shape=[-1, 0, 1],
            dtype=init_token.dtype,
            value=0)
        self._cand_seq = fluid.layers.fill_constant_batch_size_like(
            input=init_token,
            shape=[-1, 0, beam_size],
            dtype=init_token.dtype,
            value=0)

        self.beam_size = beam_size

        self._dtype = "float32" if not use_fp16 else "float16"
        _SHAPE_PLACEHOLDER = [10, beam_size]
        self._delta_score_out = fluid.layers.create_parameter(shape=_SHAPE_PLACEHOLDER, dtype=self._dtype,
                                                              name="duplicated_trigram_blocking_delta_score_out")
        self.tokenizer = roberta_tokenizer
        id2is_full_token = self._build_id2is_full_token(self.tokenizer, self._dtype)
        self._id2is_full_token = fluid.layers.create_parameter(
            shape=id2is_full_token.shape,
            dtype=self._dtype,
            name="duplicated_trigram_blocking_id2is_full_token",
            default_initializer=fluid.initializer.NumpyArrayInitializer(id2is_full_token))

    def update_seq(self, new_step_id, gather_idx):
        """update alive sequence. need pre-gather the inner seq then concat the new step id"""
        # new_step_id = fluid.layers.unsqueeze(new_step_id, axes=[1])
        alive_seq = fluid.layers.gather(self._alive_seq, gather_idx)
        # => [N, T==1, 1]
        alive_seq = fluid.layers.concat([alive_seq, new_step_id], axis=1)
        fluid.layers.assign(alive_seq, self._alive_seq)
        return self._alive_seq

    def expand_cand_seq(self, new_topk_indx):
        """expand the alive seq by concatenating the topk candidates"""
        new_topk_indx = fluid.layers.unsqueeze(new_topk_indx, axes=[1])  # (batch_size, 1, beam_size)
        cand_seq = fluid.layers.expand(self._alive_seq, expand_times=[1, 1, self.beam_size])
        # => [N, T+1, beam_size]
        expand_cand_seq = fluid.layers.concat([cand_seq, new_topk_indx], axis=1)
        fluid.layers.assign(expand_cand_seq, self._cand_seq)
        return self._cand_seq

    @property
    def alive_seq(self):
        """alive seq"""
        return self._alive_seq

    @property
    def cand_seq(self):
        """candidate seq"""
        return self._cand_seq

    @property
    def delta_score_out(self):
        """delta score out"""
        return self._delta_score_out

    @property
    def id2is_full_token(self):
        """id->isfulltoken"""
        return self._id2is_full_token

    @staticmethod
    def blocking_forward(cand_seq, id2is_full_token):
        """py_func can't be member function
        run the trigram-blocking logic. return `delta-score` for every sequence.
            for seq which has duplicated trigram, set delta-score = -inf,
            else set delta-score = 0
        in the outer, should do the `seq-score + delta-score` logic

        alive_seq: shape = [N, T, 1]

        Returns
        ---------
        np.array, shape = [N, 1]
        """
        _BLOCKING_DELTA = -65000.0  # -65500.0 is the min value of float16
        _KEEP_DELTA = 0.0
        cand_seq = np.array(cand_seq)  # (batch_size, dec_len, beam_size)
        cand_seq = np.transpose(cand_seq, axes=(0, 2, 1))  # (batch_size, beam_size, dec_len)
        id2is_full_token = np.array(id2is_full_token)

        def _sub_token_id2full_tokens(sub_token_ids):
            full_tokens = []
            for sub_token_id in sub_token_ids:
                is_full_token = bool(id2is_full_token[sub_token_id])

                if is_full_token or not full_tokens:
                    full_tokens.append([sub_token_id])
                else:
                    pre_full_token = full_tokens[-1]
                    pre_full_token.append(sub_token_id)

            full_tokens = ["-".join(map(str, full_token)) for full_token in full_tokens]
            return full_tokens

        _make_trigram_str = lambda trigram_tokens: "_".join(trigram_tokens)
        delta_list = []
        for beam_cand_ids in cand_seq:
            delta_score = []
            for one_seq_ids in beam_cand_ids:
                sub_token_ids = one_seq_ids.reshape(-1)
                tokens = _sub_token_id2full_tokens(sub_token_ids)
                if len(tokens) <= 3:
                    delta_score.append(_KEEP_DELTA)
                    continue
                # don't include the last trigram(checking self)!
                trigrams = [_make_trigram_str(tokens[end - 3: end]) for end in range(3, len(tokens))]
                trigrams_set = set(trigrams)
                last_trigram = _make_trigram_str(tokens[-3:])
                if last_trigram in trigrams_set:
                    # duplicated
                    delta_score.append(_BLOCKING_DELTA)
                else:
                    delta_score.append(_KEEP_DELTA)
            delta_list.append(delta_score)

        return np.array(delta_list, dtype=id2is_full_token.dtype).reshape(cand_seq.shape[0], cand_seq.shape[1])

    @staticmethod
    def blocking_backward(*args):
        """blocking backward"""
        raise ValueError("Impossible call backward.")

    def _build_id2is_full_token(self, tokenizer, dtype):
        vocab_sz = tokenizer.vocab_size()
        is_full_token = [0.0] * vocab_sz
        for token_id in range(vocab_sz):
            token = tokenizer.convert_id_to_token(token_id)
            token_str = tokenizer.gptbpe_tokenizer.decode_token(token)
            if token_str.startswith(' '):
                is_full_token[token_id] = 1.0

        return np.array(is_full_token, dtype=dtype)

