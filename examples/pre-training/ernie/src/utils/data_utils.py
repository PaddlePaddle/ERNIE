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
import re
import numpy as np
import os
import datetime
import paddle
from models.ernie_mm_moe.modeling import IDTYPES_2_ID
from .mm_data_utils import MMSpecialTokensConfig

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


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def fancy_print(data, tokenizer):
    """_summary_

    Args:
        data (_type_): _description_
        tokenizer (_type_): _description_

    Returns:
        _type_: _description_
    """
    marker1 = "[unused99]"
    marker2 = "[unused98]"
    image_token = tokenizer.encode(
        MMSpecialTokensConfig.image_placeholder, add_special_tokens=False
    )["input_ids"][0]
    logger.info(f"IMAGE_TOKEN_ID: {image_token}")
    for ids, labels in zip(data["input_ids"].tolist(), data["labels"].tolist()):
        # log.info(labels)
        ids2 = []
        assert len(ids) == len(labels)
        last_j = 0
        for i, j in zip(ids, labels):
            j = int(j != tokenizer.ignored_index)
            if i == image_token:
                ids2 += tokenizer.encode("<|image|>", return_attention_mask=False)[
                    "input_ids"
                ]
            else:
                ids2.append(i)
            if j != last_j:
                ids2 += tokenizer.encode(
                    marker1 if (j > last_j) else marker2,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
            last_j = j
        if j == 1:
            ids2 += tokenizer.encode(
                marker2, add_special_tokens=False, return_attention_mask=False
            )["input_ids"]
        ret = (
            tokenizer.decode(ids2)
            .replace("[unused99]", bcolors.FAIL)
            .replace("[unused98]", bcolors.ENDC)
        )

        # 花里胡哨的代码
        image_tag = "<|image|>"
        pat = re.compile(f"({re.escape(image_tag)})+")
        build = []
        for i in pat.finditer(ret):
            cnt = i.group(0).count(image_tag)
            build.append((i.span(), f"<|image@{cnt}|>"))

        pad_tag = ["<pad>", "<mask:0>", "<unk>"]
        pat = re.compile(f"({'|'.join(pad_tag)})+")
        for i in pat.finditer(ret):
            cnt = sum(i.group(0).count(t) for t in pad_tag)
            build.append((i.span(), f"<pad@{cnt}>"))

        for s, t in build[::-1]:
            l, r = s
            ret = ret[:l] + t + ret[r:]
        return ret


def merge_fn(
    tokenizer, batch, pad_to_max_seqlen=None, debug_print=1, shift_label=False
):
    """
    collate_fn
    **seqlen超长时会从右边开始截断，seqlen不够时会从右边开始pad**
    """
    global DEBUG_PRINT_CNT
    if pad_to_max_seqlen and shift_label:
        pad_to_max_seqlen += 1

    keys = list(batch[0].keys())

    ret = {}
    for k in keys:
        if isinstance(batch[0][k], (int, float)):
            ret[k] = np.stack([b[k] for b in batch], 0)
        elif k == "images":
            ret[k] = np.concatenate([b[k] for b in batch])
        else:
            if k == "input_ids":
                pad_value = tokenizer.pad_token_id
            elif k == "labels":
                pad_value = tokenizer.ignored_index
            else:
                pad_value = 0

            if batch[0][k] is not None:
                ret[k] = pad_sequence(
                    [b[k] for b in batch],
                    padding_value=pad_value,
                    fix_len=pad_to_max_seqlen,
                )

    batch = ret
    # batch.update(ignored_index=tokenizer.ignored_index)

    if DEBUG_PRINT_CNT < debug_print:
        DEBUG_PRINT_CNT += 1
        for k, v in batch.items():
            print_data_online(
                f"Example={DEBUG_PRINT_CNT} key={k}, len={len(v[0])if isinstance(v, np.ndarray) and v.ndim > 1 else 0}, value={v[0] if isinstance(v, np.ndarray) else v}"
            )
        print_data_online(
            f"Example={DEBUG_PRINT_CNT} text={fancy_print(batch, tokenizer)}"
        )

    if shift_label:
        batch["labels"] = batch["labels"][:, 1:]
        batch["input_ids"] = batch["input_ids"][:, :-1]
    return batch


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


def smart_stack(tensor, axis=0):
    """_summary_

    Args:
        tensor (_type_): _description_
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    if isinstance(tensor[0], paddle.Tensor):
        return paddle.stack(tensor, axis=axis)
    else:
        return np.stack(tensor, axis=axis)


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
                pad_value = IDTYPES_2_ID["text"]  # pad is also considered as text
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
        print_data_online(
            f"Example={DEBUG_PRINT_CNT} text={fancy_print(batch, tokenizer)}"
        )

    if shift_label:
        batch["labels"] = batch["labels"][:, 1:]
        batch["input_ids"] = batch["input_ids"][:, :-1]

    if doc_pack_attn:
        # 计算inbatch_pack_offset 来实现DovAtten
        doc_marks = (batch["input_ids"] == 2).astype(np.int64)
        doc_marks[:, -1] = 1  # 每条样本最后一个位置的doc_marks为1
        _offset = np.where(doc_marks.reshape([-1]))[0]
        _offset = (_offset + 1).tolist()
        # 开头补 一个 0
        offset = np.expand_dims(np.array([0] + _offset, dtype=np.int64), axis=0)
        # 使用 -1 pad到固定长度
        offset = pad_sequence(
            offset, padding_value=-1, fix_len=batch["input_ids"].shape[1]
        )
        batch["inbatch_pack_offset"] = offset

    return batch


def get_in_batch_mask(batch):
    length_list = [len(b) for b in batch]
    scale = np.concatenate(
        [np.full(length_list[i], i, dtype=batch[0].dtype) for i in range(len(batch))], 0
    )
    mask = np.expand_dims(
        np.tril(np.expand_dims(scale, axis=0) == np.expand_dims(scale, axis=-1)), axis=0
    ).astype(batch[0].dtype)
    return mask


def merge_fn_in_batch(
    tokenizer,
    batch,
    pad_to_max_seqlen=None,
    debug_print=1,
    use_attn_mask=False,
    shift_label=False,
):
    """
    in-batch merge策略，将数据横向连接。目前没有处理attention-mask.
    `pad_to_max_seqlen`指定后， **seqlen超长时会从右边开始阶段，seqlen不够时会从右边开始pad**
        不指定的话，没有padding。
    """
    global DEBUG_PRINT_CNT
    if pad_to_max_seqlen and shift_label:
        pad_to_max_seqlen += 1
    keys = list(batch[0].keys())
    if "data_id" in keys:
        data_id = np.stack([b.pop("data_id") for b in batch], 0)

    ret = {}
    for k in keys:
        if k == "data_id":
            continue
        if k == "input_ids":
            pad_value = tokenizer.pad_token_id
        elif k == "labels":
            pad_value = tokenizer.ignored_index
        else:
            pad_value = 0

        if batch[0][k] is not None:
            array_list = [b[k] for b in batch]
            for a in array_list:
                if k != "images":
                    assert len(a.shape) == 1, (a.shape, k)
                else:
                    assert len(a.shape) == 4, (a.shape, k)
            concated = np.concatenate(array_list, 0)
            if pad_to_max_seqlen and k != "images":
                concated = concated[:pad_to_max_seqlen]
                pad_len = pad_to_max_seqlen - concated.shape[0]
                if pad_len > 0:
                    concated = np.pad(concated, (0, pad_len), constant_values=pad_value)

            if k != "images":
                concated = np.expand_dims(concated, 0)
            ret[k] = concated

            if use_attn_mask and k == "input_ids":
                attn_mask = get_in_batch_mask(array_list)
                if pad_to_max_seqlen:
                    attn_mask = attn_mask[:pad_to_max_seqlen]
                    pad_len = pad_to_max_seqlen - attn_mask.shape[1]
                    if pad_len > 0:
                        attn_mask = np.pad(
                            attn_mask,
                            ((0, 0), (0, pad_len), (0, pad_len)),
                            constant_values=0,
                        )
                ret["attention_mask"] = attn_mask

    batch = ret
    batch.update(ignored_index=tokenizer.ignored_index)
    if "data_id" in keys:
        batch.update(data_id=data_id)

    if DEBUG_PRINT_CNT < debug_print:
        DEBUG_PRINT_CNT += 1
        for k, v in batch.items():
            print_data_online(
                f"Example={DEBUG_PRINT_CNT} key={k}, len={len(v[0])if isinstance(v, np.ndarray) else 0}, value={v[0] if isinstance(v, np.ndarray) else v}"
            )
        print_data_online(
            f"Example={DEBUG_PRINT_CNT} text={fancy_print(batch, tokenizer)}"
        )

    if shift_label:
        batch["labels"] = batch["labels"][:, 1:]
        batch["input_ids"] = batch["input_ids"][:, :-1]
    return batch
