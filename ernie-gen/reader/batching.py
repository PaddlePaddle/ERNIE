#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Mask, padding and batching."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random

from six.moves import xrange


def gen_unidirectional_mask(insts, sent_b_starts=None):
    """
    generate input mask for seq2seq
    """
    max_len = max(len(inst) for inst in insts)
    input_mask_data = np.zeros((len(insts), max_len, max_len))
    for index, mask_data in enumerate(input_mask_data):
        start = sent_b_starts[index]
        end = len(insts[index])
        mask_data[:end, :start] = 1.0
        # Generate the lower triangular matrix using the slice of matrix
        b = np.tril(np.ones([end - start, end - start]), 0)
        mask_data[start:end, start:end] = b
    input_mask_data = np.array(input_mask_data, dtype='float32').reshape([-1, max_len, max_len])
    return input_mask_data


def gen_query_input(token_ids, max_len, sent_b_starts, mask_id):
    """
    generate query input when using two-stream
    """
    bsz = len(sent_b_starts)
    dec_len = map(lambda i:len(token_ids[i]) - sent_b_starts[i], range(bsz))
    max_len_query = max(dec_len)
    mask_datas = np.zeros((bsz, max_len_query, max_len + max_len_query))
    mask_ids = np.ones((bsz, max_len_query, 1)) * mask_id
    tgt_pos = sum(map(lambda i:list(
        range(max_len_query * i + 1, max_len_query * i + dec_len[i])), range(bsz)), [])
    for index, mask_data in enumerate(mask_datas):
        for i in range(dec_len[index]):
            mask_data[i, :sent_b_starts[index] + i] = 1.0
            mask_data[i, max_len + i] = 1.0

    return (mask_datas.astype('float32'),
           mask_ids.astype('int64'),
           np.array(tgt_pos).reshape([-1, 1]).astype('int64'))


def pad_batch_data(insts,
                   pad_idx=0,
                   sent_b_starts=None,
                   is_unidirectional=False,
                   return_pos=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, max_len, 1])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape([-1, 1])]

    return return_list if len(return_list) > 1 else return_list[0]


if __name__ == "__main__":

    pass
