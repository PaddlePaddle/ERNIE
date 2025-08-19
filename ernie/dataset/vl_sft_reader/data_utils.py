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
import copy
import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

DEBUG_PRINT_CNT = 0


def pad_sequence(sequences, padding_value=0, fix_len=None):
    """Fill sequences(np.ndarray) into a fixed-length matrix."""
    # don't use any paddle.Tensor in collate-fn
    #   which prevent leakage in multi-process
    max_size = sequences[0].shape
    trailing_dims = tuple(max_size[1:])

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


def cumulative_indices_merging(x):
    """
    Merge and generate a cumulative index list.

    Args:
        x (list of lists of int): The input list of index lists,
        where each sublist represents a group of indices.

    Returns:
        list of int: The merged cumulative index list.

    Example:
        >>> cumulative_indices_merging([[1, 2], [3, 4], [5]])
        [1, 3, 6, 10]

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


class Bcolors:
    """
    Define a set of color constants to facilitate printing colored text in the terminal
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def fancy_print(data, tokenizer, im_prefix_length):
    """
    Format and output the given data.

    Args:
        data (dict): A dictionary containing input and label data.
        tokenizer (Tokenizer): The tokenizer used for text encoding.
        im_prefix_length (int): The length of the image prefix, which is not used in this function.

    Returns:
        str: The formatted output string.

    """
    marker1 = "[unused99]"
    marker2 = "[unused98]"
    image_token = len(tokenizer.get_vocab())

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
            )
        ret = (
            tokenizer.decode(ids2)
            .replace("[unused99]", Bcolors.FAIL)
            .replace("[unused98]", Bcolors.ENDC)
        )

        image_tag = "<|image|>"
        pat = re.compile(f"({re.escape(image_tag)})+")
        build = []
        for i in pat.finditer(ret):
            cnt = i.group(0).count(image_tag)
            build.append((i.span(), f"<|image@{cnt}|>"))

        pad_tag = ["<pad>", "<mask:0>"]
        pat = re.compile(f"({'|'.join(pad_tag)})+")
        for i in pat.finditer(ret):
            cnt = sum(i.group(0).count(t) for t in pad_tag)
            build.append((i.span(), f"<pad@{cnt}>"))

        for s, t in build[::-1]:
            left, right = s
            ret = ret[:left] + t + ret[right:]
        return ret


def merge_rope_3d_position(list_position_ids):
    """merge two 3d position"""

    def merge(position_ids, new_position_ids):
        position_ids = np.array(position_ids)
        new_position_ids = np.array(new_position_ids)
        return np.concatenate(
            (position_ids, np.max(position_ids) + new_position_ids), axis=0
        )

    returned = list_position_ids[0]
    for position_ids in list_position_ids[1:]:
        returned = merge(returned, position_ids)
    return returned


def unmerge_rope_3d_position(position_ids, input_ids_batch):
    """unmerge 3d position according to input_ids_batch"""
    accu_id_length = 0
    position_ids = copy.deepcopy(position_ids)
    for idx, item in enumerate(input_ids_batch):
        if idx == 0:
            assert (
                position_ids[accu_id_length][0] == 0
            ), "the fisrt postition-ids should be 0"
        position_ids[accu_id_length : accu_id_length + len(item)] -= position_ids[
            accu_id_length
        ][0]

        accu_id_length += len(item)
    return position_ids


def merge_fn_group_batch(
    batch,
    tokenizer=None,
    pad_to_max_seqlen=None,
    debug_print=1,
    im_prefix_length=32,
    shift_label=False,
    rng=None,
    combine_batch: int = 1,
    image_dtype="uint8",
    multimodal_multiround_ratio=0.3,
):
    """
    N-into-1 within a batch
    """
    need_multiround = batch[0].get(
        "need_multiround", rng.random() < multimodal_multiround_ratio
    )
    if "need_multiround" in batch[0]:
        del batch[0]["need_multiround"]

    global DEBUG_PRINT_CNT
    if pad_to_max_seqlen and shift_label:
        pad_to_max_seqlen += 1

    assert len(batch) == 1, f"zzz len(batch) != 1, {len(batch)}"
    keys = list(batch[0].keys())
    if combine_batch > 1:
        _batch = []
        for group in [
            batch[i : i + combine_batch] for i in range(0, len(batch), combine_batch)
        ]:
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
        if k == "input_ids_batch":
            continue

        if isinstance(batch[0][k], (int, float, np.int64, np.float64)):
            ret[k] = np.stack([b[k] for b in batch], 0)
        elif k == "grid_thw":
            to_concat = [b[k] for b in batch if b[k] is not None]
            ret[k] = np.concatenate(to_concat, axis=0)
            if pad_to_max_seqlen:
                tmp = max(0, pad_to_max_seqlen * len(batch) - ret[k].shape[0])
                if tmp > 0:
                    ret[k] = np.concatenate(
                        [ret[k], np.zeros([tmp, 3])], axis=0
                    ).astype("int64")
        elif k in ["data_id", "part_id", "src_id"]:
            ret[k] = np.concatenate([b[k] for b in batch])
        elif k == "data_type":
            ret[k] = np.array(batch[0][k])
        elif k == "images":
            to_concat = [b[k] for b in batch if b[k] is not None]
            if len(to_concat) != 0:
                assert (
                    image_dtype != "bfloat16"
                ), f"Currently, not support {image_dtype} for numpy"
                # ret[k] = np.concatenate([b[k] for b in batch if b[k] is not None])
                ret[k] = np.concatenate(to_concat, axis=0).astype(image_dtype)
            else:
                ret[k] = None
        elif k == "cumulative_indices":
            if batch[0][k] is not None:
                cur_cumulative_indices = cumulative_indices_merging(
                    [b[k] for b in batch]
                )
                cur_cumulative_indices = np.array(cur_cumulative_indices)
                ret[k] = cur_cumulative_indices
            else:
                ret[k] = None
        else:
            if k == "input_ids":
                pad_value = tokenizer.pad_token_id
            elif k in ["labels", "image_type_ids"]:
                pad_value = tokenizer.ignored_index
            elif k == "image_position_ids":
                pad_value = -1
            elif k in ["position_ids"]:
                pad_value = [0, 0, 0]
            elif k in ["token_type_ids"]:
                pad_value = 0
            else:
                pad_value = 0

            if batch[0][k] is not None:
                if k in ["tmp_images", "image_position_ids"]:
                    tmp = [i for b in batch for i in b[k]]
                else:
                    tmp = [b[k] for b in batch]
                try:
                    if k == "token_type_ids":
                        ret[k] = pad_sequence(
                            tmp, padding_value=pad_value, fix_len=pad_to_max_seqlen + 1
                        )
                    else:
                        ret[k] = pad_sequence(
                            tmp, padding_value=pad_value, fix_len=pad_to_max_seqlen
                        )
                except Exception as e:
                    logger.info(
                        f"k: {k}, tmp: {tmp}, original: {[b[k] for b in batch]}"
                    )
                    logger.info(f"e: {e}")
                    # exit()

                if k == "image_position_ids":
                    ret["image_attention_mask"] = ret[k] != pad_value
                    ret[k][ret[k] == pad_value] = 0

    inbatch_pack_offset = [0]
    if not need_multiround:
        for item in batch[0]["input_ids_batch"]:
            inbatch_pack_offset.append(inbatch_pack_offset[-1] + len(item))
        # fix position-ids
        if "position_ids" in ret:
            ret["position_ids"] = [
                unmerge_rope_3d_position(
                    ret["position_ids"][0], batch[0]["input_ids_batch"]
                )
            ]

            # logger.debug(f"unmerge position_ids: ret['position_ids']")
        inbatch_pack_offset[-1] = (
            pad_to_max_seqlen  # include padding in the last interval
        )
    else:
        inbatch_pack_offset.append(pad_to_max_seqlen)
    padded_inbatch_pack_offset = np.reshape(
        np.array(
            inbatch_pack_offset
            + [-1] * (pad_to_max_seqlen + 1 - len(inbatch_pack_offset)),
            dtype=np.int64,
        ),
        [1, -1],
    )
    ret["inbatch_pack_offset"] = padded_inbatch_pack_offset
    batch = ret

    if DEBUG_PRINT_CNT < debug_print:
        DEBUG_PRINT_CNT += 1
        for k, v in batch.items():
            logger.debug(
                f"""Example={DEBUG_PRINT_CNT} key={k},
                len={len(v[0])if isinstance(v, np.ndarray) and v.ndim > 1 else 0},
                value={v[0] if isinstance(v, np.ndarray) else v}"""
            )
        logger.debug(
            f"Example={DEBUG_PRINT_CNT} text={fancy_print(batch, tokenizer, im_prefix_length)}"
        )

    if shift_label:
        batch["labels"] = batch["labels"][:, 1:]
        batch["input_ids"] = batch["input_ids"][:, :-1]
    return batch
