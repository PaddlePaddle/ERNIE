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
data utils
"""
import logging
import numpy as np
import os
import datetime
import paddle

logger = logging.getLogger(__name__)

DEBUG_PRINT_CNT = 0

log_dir = os.getenv("PADDLE_LOG_DIR", "./log")
local_rank = os.getenv("PADDLE_LOCAL_RANK", "0")
date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
print_data_path = os.path.join(
    log_dir, "data_rank_{}_{}.txt".format(local_rank, date_str)
)


def print_data_online(msg):
    """
    print data online
    """
    with open(print_data_path, "a+") as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "\n")
        f.write(msg + "\n")


def pad_sequence(sequences, padding_value=0, fix_len=None):
    """Fill sequences(np.ndarray) into a fixed-length matrix."""
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
    out_tensor = np.full(out_dims, padding_value, dtype=sequences[0].dtype)
    for i, tensor in enumerate(sequences):
        tensor = tensor[:max_len]
        length = tensor.shape[0]
        out_tensor[i, :length, ...] = tensor
    return out_tensor


DEBUG_PRINT_CNT = 0


def smart_concat(tensor, axis=0):
    """_summary_

    Args:
        tensor (_type_): _description_
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    if isinstance(tensor[0], paddle.Tensor):
        return paddle.concat(tensor, axis=axis)
    else:
        return np.concatenate(tensor, axis=axis)


def merge_fn_group_batch(
    tokenizer,
    batch,
    pad_to_max_seqlen=None,
    debug_print=1,
    shift_label=False,
    combine_batch: int = 1,
    image_dtype="bfloat16",
    doc_pack_attn=False,
):
    """
    batch 内 n合一
    """
    bsz = len(batch)
    global DEBUG_PRINT_CNT
    if pad_to_max_seqlen and shift_label:
        pad_to_max_seqlen += 1

    keys = list(batch[0].keys())

    if combine_batch > 1:
        _batch = []
        for group in [
            batch[i : i + combine_batch] for i in range(0, len(batch), combine_batch)
        ]:

            if "src_id" in group[0]:
                src_lst = list(set([b["src_id"] for b in group]))
                assert len(src_lst) == 1, f"src_lst: {src_lst}"

            item = {}
            for k in keys:
                if group[0][k] is None:
                    item[k] = None
                    continue
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
        elif k in ["src_id", "data_id", "data_type"]:
            ret[k] = np.concatenate([b[k] for b in batch])
        elif k == "images":
            to_concat = [b[k] for b in batch if b[k] is not None]
            if len(to_concat) != 0:
                assert (
                    image_dtype != "bfloat16"
                ), f"Currently, not support {image_dtype} for numpy"
                ret[k] = np.concatenate(to_concat, axis=0).astype(image_dtype)
            else:
                ret[k] = None
        elif k == "grid_thw" and batch[0][k] is not None:
            ret[k] = np.concatenate([b[k] for b in batch], axis=0).astype("int64")
            if pad_to_max_seqlen:
                tmp = max(0, pad_to_max_seqlen * bsz - ret[k].shape[0])
                if tmp > 0:
                    ret[k] = np.concatenate(
                        [ret[k], np.zeros([tmp, 3])], axis=0
                    ).astype("int64")
        elif k in ["audio_input_ids", "audio_labels"]:
            to_concat = [b[k] for b in batch if b[k] is not None]
            if len(to_concat) != 0:
                concat_audio_ids = smart_concat(to_concat)
                assert (
                    len(concat_audio_ids.shape) == 2
                ), "拼接完的audio_ids必须是2维tensor，且shape=[sum(frames), depth]"
                ret[k] = pad_sequence(
                    [concat_audio_ids],
                    padding_value=tokenizer.ignored_index,
                    fix_len=pad_to_max_seqlen * bsz,
                )[0]
                assert (
                    len(ret[k].shape) == 2
                ), "padding完的audio_ids 必须是2维tensor，且shape=[bsz*pad_to_max_seqlen, depth]"
            else:
                ret[k] = None
        else:
            if k == "input_ids":
                pad_value = tokenizer.pad_token_id
            elif k == "labels" or k == "image_type_ids":
                pad_value = tokenizer.ignored_index
            elif k == "token_type_ids":
                pad_value = 0  # pad is also considered as text
            else:
                pad_value = 0

            if batch[0][k] is not None:
                ret[k] = pad_sequence(
                    [b[k] for b in batch],
                    padding_value=pad_value,
                    fix_len=(
                        pad_to_max_seqlen
                        if k != "token_type_ids"
                        else pad_to_max_seqlen + 1
                    ),
                )

    batch = ret

    if DEBUG_PRINT_CNT < debug_print:
        DEBUG_PRINT_CNT += 1
        for k, v in batch.items():
            if v is not None and v.dtype == np.float32:  # do not show image
                v = v.shape
            print_data_online(
                f"Example={DEBUG_PRINT_CNT} key={k},  "
                f"len={len(v[0])if isinstance(v, np.ndarray) and v.ndim > 1 else 0}, "
                f"value={v if isinstance(v, np.ndarray) else v}"
            )

    if shift_label:
        batch["labels"] = batch["labels"][:, 1:]
        batch["input_ids"] = batch["input_ids"][:, :-1]

    if doc_pack_attn:
        doc_marks = (batch["input_ids"] == 2).astype(np.int64)
        doc_marks[:, -1] = 1
        _offset = np.where(doc_marks.reshape([-1]))[0]
        _offset = (_offset + 1).tolist()
        offset = np.expand_dims(np.array([0] + _offset, dtype=np.int64), axis=0)
        offset = pad_sequence(
            offset, padding_value=-1, fix_len=batch["input_ids"].shape[1]
        )
        batch["inbatch_pack_offset"] = offset

    return batch
