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
"""Mask, padding and batching."""

import paddle
import numpy as np

def get_related_pos(insts, 
                    seq_len,
                    memory_len=128):
    """generate relative postion ids"""
    beg = seq_len + seq_len + memory_len
    r_position = [list(range(beg - 1, seq_len - 1, -1)) + \
                  list(range(0, seq_len)) for i in range(len(insts))]
    return np.array(r_position).astype('int64').reshape([len(insts), beg, 1])

def pad_batch_data(insts,
                   insts_data_type="int64",
                   pad_idx=0,
                   final_cls=False,
                   pad_max_len=None,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    if pad_max_len:
        max_len = pad_max_len
    else:
        max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    # id
    if final_cls:
        inst_data = np.array(
            [inst[:-1] + list([pad_idx] * (max_len - len(inst))) + [inst[-1]] for inst in insts])
    else:
        inst_data = np.array(
            [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype(insts_data_type).reshape([-1, max_len, 1])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        if final_cls:
            input_mask_data = np.array([[1] * len(inst[:-1]) + [0] *
                                        (max_len - len(inst)) + [1] for inst in insts])
        else:
            input_mask_data = np.array([[1] * len(inst) + [0] *
                                        (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        if paddle.__version__[:3] <= '1.5':
            seq_lens_type = [-1, 1]
        else:
            seq_lens_type = [-1]
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape(seq_lens_type)]

    return return_list if len(return_list) > 1 else return_list[0]
