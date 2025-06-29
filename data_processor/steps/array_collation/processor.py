# !/usr/bin/env python3

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""
This file contains the smart_concat and smart_stack functions
"""
import logging

import numpy as np
import paddle

from data_processor.steps.array_collation.fancy_print import fancy_print, print_data_online
from data_processor.utils.constant import IDTYPES_2_ID
from data_processor.utils.processor_base import ProcessorBase

logger = logging.getLogger(__name__)


def smart_concat(tensor, axis=0):
    """
    Intelligent concatenation function that decides whether to use PaddlePaddle's
    `concat` or NumPy's `concatenate` based on the type of the input tensors.

    Args:
        tensor (list of tensors): List of tensors to be concatenated.
        axis (int, optional): The axis along which the tensors will be concatenated. Default is 0.

    Returns:
        paddle.Tensor or np.ndarray: The concatenated tensor.

    Raises:
        TypeError: If the type of the input tensors is neither `paddle.Tensor` nor `np.ndarray`,
        a `TypeError` is raised.
    """
    if isinstance(tensor[0], paddle.Tensor):
        return paddle.concat(tensor, axis=axis)
    else:
        return np.concatenate(tensor, axis=axis)


def smart_stack(tensor, axis=0):
    """
    Stacks multiple tensors along a specified axis into a new tensor.

    Args:
        tensor (list of paddle.Tensor or np.ndarray): List of tensors to be stacked.
        axis (int, optional): The axis along which to stack the tensors. Default is 0.

    Returns:
        paddle.Tensor or np.ndarray: The new tensor formed by stacking.
    """
    if isinstance(tensor[0], paddle.Tensor):
        return paddle.stack(tensor, axis=axis)
    else:
        return np.stack(tensor, axis=axis)


def cumulative_indices_merging(x):
    """
    Cumulative indices merging.
        Args:
        x (list): input list.
        Returns:
        list: output list.
    """
    cur_cumulative_indices = []
    tmp = 0
    for i, indices in enumerate(x):
        if i != 0:
            indices = indices[1:]
        tmp = cur_cumulative_indices[-1] if len(cur_cumulative_indices) != 0 else tmp
        for index in indices:
            cur_cumulative_indices.append(tmp + index)
    return cur_cumulative_indices


def paddle_pad_sequence(sequences, padding_value=0, fix_len=None):
    """
    Fill sequences(paddle.tensor) into a fixed-length matrix.
    Args:
        sequences (list): input sequence list.
    """
    # don't use any paddle.Tensor in collate-fn
    #   which prevent leakage in multi-process
    max_size = sequences[0].shape
    trailing_dims = tuple(max_size[1:])
    # print("trailing_dims: ", trailing_dims)

    max_len = max([s.shape[0] for s in sequences])
    if fix_len is not None:
        if fix_len < max_len:
            logger.warning(f"truncating example from {max_len} to {fix_len}")
        max_len = fix_len
    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = paddle.full(out_dims, padding_value, dtype=sequences[0].dtype)
    for i, tensor in enumerate(sequences):
        tensor = tensor[:max_len]
        length = tensor.shape[0]
        out_tensor[i, :length, ...] = tensor
    return out_tensor


def np_pad_sequence(sequences, padding_value=0, fix_len=None):
    """
    Fill sequences(np.ndarray) into a fixed-length matrix.
    Args:
        sequences (list): input sequence list.
    """

    max_size = sequences[0].shape
    trailing_dims = tuple(max_size[1:])
    # print("trailing_dims: ", trailing_dims)

    max_len = max([s.shape[0] for s in sequences])
    if fix_len is not None:
        if fix_len < max_len:
            logger.warning(f"truncating example from {max_len} to {fix_len}")
        max_len = fix_len
    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = np.full(out_dims, padding_value, dtype=sequences[0].dtype)
    for i, tensor in enumerate(sequences):
        tensor = tensor[:max_len]
        length = tensor.shape[0]
        out_tensor[i, :length, ...] = tensor
    return out_tensor


def pad_sequence(sequences, padding_value=0, fix_len=None):
    """
    Fill sequences(paddle.tensor or np.ndarray) into a fixed-length matrix.
    Args:
        sequences (list): input sequence list.
        padding_value (int, optional): padding value. Defaults to 0.

    """
    if isinstance(sequences[0], paddle.Tensor):
        return paddle_pad_sequence(sequences, padding_value=padding_value, fix_len=fix_len)
    else:
        return np_pad_sequence(sequences, padding_value=padding_value, fix_len=fix_len)


class ArrayCollationProcessor(ProcessorBase):
    """
    Array Collation Processor
    """

    def __init__(self, args, tokenizer):
        """
        init
        """
        super().__init__(args)
        self.DEBUG_PRINT_CNT = 0
        self.tokenizer = tokenizer
        self.pad_to_max_seqlen = args.pad_to_max_seqlen
        self.debug_print = args.debug_print
        self.shift_label = args.shift_label
        self.combine_batch = args.combine_batch
        self.image_dtype = "uint8"

    def collate(
        self,
        batch,
    ):
        """
        collate
        """
        pad_to_max_seqlen = self.pad_to_max_seqlen
        if self.pad_to_max_seqlen and self.shift_label:
            pad_to_max_seqlen += 1

        keys = list(batch[0].keys())

        if self.combine_batch > 1:
            _batch = []
            for group in [batch[i : i + self.combine_batch] for i in range(0, len(batch), self.combine_batch)]:
                if "src_id" in group[0]:
                    src_lst = {b["src_id"] for b in group}
                    assert len(src_lst) == 1, f"src_lst: {src_lst}"

                item = {}
                for k in keys:
                    if isinstance(group[0][k], (int, float)):
                        item[k] = np.stack([i[k] for i in group], 0)
                    else:
                        item[k] = np.concatenate([i[k] for i in group])
                _batch.append(item)
            batch = _batch

        ret = {}
        for k in keys:
            if isinstance(batch[0][k], (int, float)):
                ret[k] = np.stack([b[k] for b in batch], 0)
            elif k in ["src_id", "data_id"]:
                ret[k] = np.concatenate([b[k] for b in batch])
            elif k == "images":
                to_concat = [b[k] for b in batch if b[k] is not None]
                if len(to_concat) != 0:
                    assert self.image_dtype != "bfloat16", f"Currently, not support {self.image_dtype} for numpy"
                    ret[k] = np.concatenate(to_concat, axis=0).astype(self.image_dtype)
                else:
                    ret[k] = None
            elif k == "grid_thw":
                ret[k] = np.concatenate([b[k] for b in batch], axis=0)
            elif k == "audio_ids":
                to_concat = [b[k] for b in batch if b[k] is not None]
                if len(to_concat) != 0:
                    ret[k] = smart_concat(to_concat)
                else:
                    ret[k] = None
            else:
                if k == "input_ids":
                    pad_value = self.tokenizer.pad_token_id
                elif k == "labels" or k == "image_type_ids":
                    pad_value = self.tokenizer.ignored_index
                elif k == "token_type_ids":
                    pad_value = IDTYPES_2_ID["text"]  # pad is also considered as text
                else:
                    pad_value = 0

                if batch[0][k] is not None:
                    ret[k] = pad_sequence(
                        [b[k] for b in batch],
                        padding_value=pad_value,
                        fix_len=pad_to_max_seqlen if k != "token_type_ids" else pad_to_max_seqlen + 1,
                    )

        batch = ret

        if self.DEBUG_PRINT_CNT < self.debug_print:
            self.DEBUG_PRINT_CNT += 1
            for k, v in batch.items():
                if v is not None and v.dtype == np.float32:
                    v = v.shape
                print_data_online(
                    f"Example={self.DEBUG_PRINT_CNT} key={k},  "
                    f"len={len(v[0])if isinstance(v, np.ndarray) and v.ndim > 1 else 0}, "
                    f"value={v if isinstance(v, np.ndarray) else v}"
                )
            print_data_online(f"Example={self.DEBUG_PRINT_CNT} text={fancy_print(batch, self.tokenizer)}")

        if self.shift_label:
            batch["labels"] = batch["labels"][:, 1:]
            batch["input_ids"] = batch["input_ids"][:, :-1]
        return batch
