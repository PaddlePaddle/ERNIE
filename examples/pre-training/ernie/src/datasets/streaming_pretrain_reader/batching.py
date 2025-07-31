#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import, division, print_function

import copy
import math
import random

import numpy as np
import paddle
from six.moves import xrange
from src.datasets.streaming_pretrain_reader.glm import (
    mask_chunk,
    mask_ditto_lm,
    mask_entity,
    mask_fewshot_unidirectional,
    mask_partial_lm,
    mask_sentence,
)
from src.datasets.streaming_pretrain_reader.span_selection import (
    create_recurring_span_selection_predictions,
)

if paddle.is_compiled_with_cuda():
    int_type = "int64"
else:
    raise Exception("paddle must compiled with cuda or npu")


def shuffle_entity(
    batch_tokens, seg_labels, total_token_num, max_seq_len=None, random_seed=0
):

    # np.random.seed(random_seed)
    max_len = (
        max_seq_len
        if max_seq_len is not None
        else max(len(inst) for inst in batch_tokens)
    )
    prob_mask = np.random.rand(total_token_num)
    # Note: the first token is [CLS], so [low=1]
    pre_sent_len = 0
    prob_index = 0
    for sent_index, sent in enumerate(batch_tokens):
        prob_index += pre_sent_len
        beg = 0
        for token_index, token in enumerate(sent):
            seg_label = seg_labels[sent_index][token_index]
            if seg_label == 1:
                continue
            if beg == 0:
                if seg_label != -1:
                    beg = token_index
                continue

            prob = prob_mask[prob_index + beg]
            if prob > 0.10:
                pass
            else:
                tmp = sent[beg:token_index]
                np.random.shuffle(tmp)
                sent[beg:token_index] = tmp

            if seg_label == -1:
                beg = 0
            else:
                beg = token_index
        pre_sent_len = len(sent)

    return batch_tokens


def mask(
    batch_tokens,
    seg_labels,
    mask_word_tags,
    total_token_num,
    vocab_size,
    max_seq_len=None,
    CLS=1,
    SEP=2,
    MASK=3,
    random_seed=0,
):
    """
    Add mask for batch_tokens, return out, mask_label, mask_pos;
    Note: mask_pos responding the batch_tokens after padded;
    """
    # np.random.seed(random_seed)
    max_len = (
        max_seq_len
        if max_seq_len is not None
        else max(len(inst) for inst in batch_tokens)
    )
    mask_label = []
    mask_pos = []
    prob_mask = np.random.rand(total_token_num)
    # Note: the first token is [CLS], so [low=1]
    replace_ids = np.random.randint(1, high=vocab_size, size=total_token_num)
    pre_sent_len = 0
    prob_index = 0
    for sent_index, sent in enumerate(batch_tokens):
        mask_flag = False
        mask_word = mask_word_tags[sent_index]
        prob_index += pre_sent_len
        if mask_word:
            beg = 0
            for token_index, token in enumerate(sent):
                seg_label = seg_labels[sent_index][token_index]
                if seg_label == 1:
                    continue
                if beg == 0:
                    if seg_label != -1:
                        beg = token_index
                    continue

                prob = prob_mask[prob_index + beg]
                if prob > 0.15:
                    pass
                else:
                    for index in xrange(beg, token_index):
                        prob = prob_mask[prob_index + index]
                        base_prob = 1.0
                        if index == beg:
                            base_prob = 0.15
                        if base_prob * 0.2 < prob <= base_prob:
                            mask_label.append(sent[index])
                            sent[index] = MASK
                            mask_flag = True
                            mask_pos.append(sent_index * max_len + index)
                        elif base_prob * 0.1 < prob <= base_prob * 0.2:
                            mask_label.append(sent[index])
                            sent[index] = replace_ids[prob_index + index]
                            mask_flag = True
                            mask_pos.append(sent_index * max_len + index)
                        else:
                            mask_label.append(sent[index])
                            mask_pos.append(sent_index * max_len + index)

                if seg_label == -1:
                    beg = 0
                else:
                    beg = token_index
        else:
            for token_index, token in enumerate(sent):
                prob = prob_mask[prob_index + token_index]
                if prob > 0.15:
                    continue
                elif 0.03 < prob <= 0.15:
                    # mask
                    if token != SEP and token != CLS:
                        mask_label.append(sent[token_index])
                        sent[token_index] = MASK
                        mask_flag = True
                        mask_pos.append(sent_index * max_len + token_index)
                elif 0.015 < prob <= 0.03:
                    # random replace
                    if token != SEP and token != CLS:
                        mask_label.append(sent[token_index])
                        sent[token_index] = replace_ids[prob_index + token_index]
                        mask_flag = True
                        mask_pos.append(sent_index * max_len + token_index)
                else:
                    # keep the original token
                    if token != SEP and token != CLS:
                        mask_label.append(sent[token_index])
                        mask_pos.append(sent_index * max_len + token_index)

        pre_sent_len = len(sent)

    if len(mask_label) == 0:
        print("mask_label == 0")
        mask_label_all_bei = list(range(20))
        mask_pos_all_bei = list(range(20))
        mask_label = np.array(mask_label_all_bei).astype(int_type).reshape([-1, 1])
        mask_pos = np.array(mask_pos_all_bei).astype(int_type).reshape([-1, 1])
        need_cal_loss = np.array([0]).astype("float32")
    else:
        mask_label = np.array(mask_label).astype(int_type).reshape([-1, 1])
        mask_pos = np.array(mask_pos).astype(int_type).reshape([-1, 1])
        need_cal_loss = np.array([1]).astype("float32")

    return batch_tokens, mask_label, mask_pos, need_cal_loss


def mask_recurring_span_selection(
    batch_src_ids,
    batch_seg_labels,
    max_seq_len=None,
    vocab_size=None,
    CLS=None,
    MASK=None,
    lm_ratio=0.15,
):
    max_seq_len = (
        max_seq_len
        if max_seq_len is not None
        else max(len(inst) for inst in batch_src_ids)
    )

    span_label_beginnings_all = []
    span_label_endings_all = []
    gather_idx = (
        []
    )  # for duplicating the output mask and the encoder output with the same bsz size of question tensor
    gather_nd_idx = (
        []
    )  # for gathering question tensor from the encoder output with size of (bsz, L, D)

    batch_src_ids_new = []
    batch_seg_labels_new = []
    # filter_idx = []
    for batch_idx, src_ids in enumerate(batch_src_ids):
        (
            src_ids,
            seg_labels,
            masked_span_positions,
            span_label_beginnings,
            span_label_endings,
        ) = create_recurring_span_selection_predictions(
            src_ids, batch_seg_labels[batch_idx]
        )
        assert (
            len(masked_span_positions)
            == len(span_label_beginnings)
            == len(span_label_endings)
        )

        batch_src_ids_new.append(src_ids)
        batch_seg_labels_new.append(seg_labels)
        # filter_idx.append(batch_idx)
        if (
            all(
                map(
                    len,
                    [masked_span_positions, span_label_beginnings, span_label_endings],
                )
            )
            == False
        ):
            # skip when span list is empty
            continue

        # prepare gathering idx
        gather_nd_idx.extend(
            [[batch_idx, pos] for pos in masked_span_positions]
        )  # (N, 2)
        gather_idx.extend([batch_idx] * len(span_label_beginnings))  # (N)
        span_label_beginnings_all.extend(span_label_beginnings)
        span_label_endings_all.extend(span_label_endings)

    assert len(batch_src_ids_new) != 0, "{} is empty".format(len(batch_src_ids_new))

    # check validation
    for item_i, item in enumerate(gather_nd_idx):
        try:
            batch_src_ids_new[item[0]][item[1]]
            batch_src_ids_new[gather_idx[item_i]]
            batch_src_ids_new[gather_idx[item_i]][span_label_beginnings_all[item_i]]
            batch_src_ids_new[gather_idx[item_i]][span_label_endings_all[item_i]]
        except Exception as e:
            print("batch_src_ids_new", batch_src_ids_new)
            print("gather_nd_idx", gather_nd_idx)
            print("gather_idx", gather_idx)
            print("span_label_beginnings_all", span_label_beginnings_all)
            print("span_label_endings_all", span_label_endings_all)
            raise e

    need_cal_loss = np.array([1]).astype("float32")
    if len(gather_idx) == 0:
        # no update
        need_cal_loss = np.array([0]).astype("float32")
        gather_nd_idx = [
            [batch_i, len(batch_item) // 2]
            for batch_i, batch_item in enumerate(batch_src_ids_new)
        ]
        gather_idx = list(range(0, len(gather_nd_idx)))
        span_label_beginnings_all = [
            len(batch_item) // 2 for batch_i, batch_item in enumerate(batch_src_ids_new)
        ]
        span_label_endings_all = [
            len(batch_item) // 2 for batch_i, batch_item in enumerate(batch_src_ids_new)
        ]

    span_label_beginnings_all = np.array(
        span_label_beginnings_all, dtype=int_type
    ).reshape(
        [-1, 1]
    )  # (bsz*pos_num, 1)
    span_label_endings_all = np.array(span_label_endings_all, dtype=int_type).reshape(
        [-1, 1]
    )  # (bsz*pos_num, 1)
    gather_idx = np.array(gather_idx, dtype=int_type)  # (bsz*pos_num)
    gather_nd_idx = np.array(gather_nd_idx, dtype=int_type)  # (bsz*pos_num, 2)

    return [
        batch_src_ids_new,
        batch_seg_labels_new,
        gather_nd_idx,
        gather_idx,
        span_label_beginnings_all,
        span_label_endings_all,
        need_cal_loss,
    ]


def mask_lm(
    batch_tokens,
    prefix_lengths=None,
    lm_ratio=0.15,
    cal_loss=None,
    max_seq_len=None,
    mem_len=128,
    is_ends=None,
    is_starts=None,
    CLS=1,
    SEP=2,
    END=3,
    is_nlm=True,
):
    def get_n_gram(mask_len, sent_len):
        token_index, cnt_time = 0, 0
        n_gram_indic = {}
        while token_index < sent_len - 1:
            n_gram = 1  # random.randint(1, 3)
            while token_index + n_gram > sent_len - 1:
                n_gram = 1  # random.randint(1, 3)
                cnt_time += 1
                if cnt_time > 5:
                    n_gram = 1
            n_gram_indic[token_index] = n_gram
            for i in range(1, n_gram + 1):
                n_gram_indic[token_index + i] = 1
            token_index += n_gram + 1
        n_gram_indic[sent_len - 1] = 1
        return n_gram_indic

    max_len = (
        max_seq_len
        if max_seq_len is not None
        else max([len(sent) for sent in batch_tokens])
    )
    mask_label = []
    mask_pos = []
    input_mask_all = []
    for sent_index, sent in enumerate(batch_tokens):
        if cal_loss[sent_index] == 1:  ## cal lm loss
            mask_len = int(math.ceil(len(sent) * lm_ratio))
            if lm_ratio == 1.0:
                mask_len = len(sent) - 1
            pad_len = max_len - len(sent)
            input_mask_each_sent = []
            if is_nlm:
                n_gram_indic = get_n_gram(mask_len, len(sent))

            for token_index, token in enumerate(sent):
                # if token_index < len(sent) - mask_len:
                #    input_mask = [1]*(len(sent)-mask_len) + [0]*(mask_len+pad_len)
                # else:
                # if is_nlm:
                #    if n_gram_indic[token_index] == 0:
                #        input_mask = [1]*token_index + [0]*(max_len-token_index)
                #    else:
                #        input_mask = input_mask_each_sent[-1]
                # else:
                if token_index >= max_seq_len:
                    break

                input_mask = [1] * (token_index + 1) + [0] * (max_len - token_index - 1)
                input_mask_each_sent.append(input_mask)
                if token_index < len(sent) - mask_len:
                    continue

                if token_index < prefix_lengths[sent_index]:  # mask out prefix tokens
                    continue

                if is_nlm:
                    mask_label.extend(
                        sent[token_index : token_index + n_gram_indic[token_index - 1]]
                    )
                    mask_pos.extend(
                        [sent_index * max_len + token_index - 1]
                        * n_gram_indic[token_index - 1]
                    )
                else:
                    mask_label.append(token)
                    mask_pos.append(sent_index * max_len + token_index - 1)

            ## add [PAD] stop symbol
            if is_ends[sent_index] == 1:  ## the end of passage
                end_pos = (
                    token_index - 1 if token_index >= max_seq_len else len(sent) - 1
                )
                mask_label.append(END)
                mask_pos.append(sent_index * max_len + end_pos)
            else:
                mask_label.append(sent[token_index])
                mask_pos.append(sent_index * max_len + token_index - 1)

            pad_input_mask = [[0] * max_len] * pad_len
            input_mask_each_sent.extend(pad_input_mask)
            input_mask_all.append(input_mask_each_sent)
        elif cal_loss[sent_index] == 0:  ## not cal lm loss
            input_mask_each_sent = [[0] * max_len] * max_len
            input_mask_all.append(input_mask_each_sent)

        batch_tokens[sent_index] = sent[:max_seq_len]

    if len(mask_label) == 0:
        print("mask_label == 0")
        mask_label_all_bei = list(range(20))
        mask_pos_all_bei = list(range(20))
        mask_label = np.array(mask_label_all_bei).astype(int_type).reshape([-1, 1])
        mask_pos = np.array(mask_pos_all_bei).astype(int_type).reshape([-1, 1])
        need_cal_loss = np.array([0]).astype("float32")
    else:
        mask_label = np.array(mask_label).astype(int_type).reshape([-1, 1])
        mask_pos = np.array(mask_pos).astype(int_type).reshape([-1, 1])
        need_cal_loss = np.array([1]).astype("float32")

    input_mask_all = (
        np.array(input_mask_all).astype("float32").reshape([-1, max_len, max_len])
    )
    if mem_len:
        memory_mask_list = []
        for is_start in is_starts:
            each_memory_mask = (
                np.zeros([max_len, mem_len])
                if is_start
                else np.ones([max_len, mem_len])
            )
            memory_mask_list.append(each_memory_mask)
        memory_mask = np.array(memory_mask_list).astype("float32")
        input_mask_all = np.concatenate((memory_mask, input_mask_all), axis=-1)

    return batch_tokens, mask_label, mask_pos, input_mask_all, need_cal_loss


def get_range(seg_labels, start):
    try:
        start = seg_labels[start:].index(0) + start
    except ValueError:
        start = -1
    try:
        end = seg_labels[start + 1 :].index(0) + start + 1
    except ValueError:
        end = len(seg_labels) - 1
    return start, end


def mask_kg(
    batch_tokens,
    seg_labels,
    total_token_num,
    vocab_size,
    CLS=1,
    SEP=2,
    MASK=3,
    max_seq_len=None,
    random_seed=0,
):
    # np.random.seed(random_seed)
    mask_label = []
    mask_pos = []
    max_len = (
        max_seq_len
        if max_seq_len is not None
        else max([len(sent) for sent in batch_tokens])
    )
    prob_mask = np.random.rand(total_token_num)
    replace_ids = np.random.randint(1, high=vocab_size, size=total_token_num)
    pre_sent_len = 0
    prob_index = 0
    for sent_index, sent in enumerate(batch_tokens):
        start, end = get_range(seg_labels[sent_index], 2)
        mask_label.extend(sent[start:end])
        mask_pos.extend(
            [sent_index * max_len + token_index for token_index in range(start, end)]
        )
        for token_index in range(start, end):
            sent[token_index] = MASK
        start, end = get_range(seg_labels[sent_index], end + 1)  # start is txt beg
        mask_flag = False
        prob_index += pre_sent_len
        beg = 0
        if start == -1:
            pre_sent_len = len(sent)
            continue
        for token_index, token in enumerate(sent[start:]):
            token_index += start
            seg_label = seg_labels[sent_index][token_index]
            if seg_label == 1:
                continue
            if beg == 0:
                if seg_label != -1:
                    beg = token_index
                continue

            prob = prob_mask[prob_index + beg]
            if prob > 0.10:
                pass
            else:
                for index in xrange(beg, token_index):
                    prob = prob_mask[prob_index + index]
                    base_prob = 1.0
                    if index == beg:
                        base_prob = 0.1
                    if base_prob * 0.2 < prob <= base_prob:
                        mask_label.append(sent[index])
                        sent[index] = MASK
                        mask_flag = True
                        mask_pos.append(sent_index * max_len + index)
                    elif base_prob * 0.1 < prob <= base_prob * 0.2:
                        mask_label.append(sent[index])
                        sent[index] = replace_ids[prob_index + index]
                        mask_flag = True
                        mask_pos.append(sent_index * max_len + index)
                    else:
                        mask_label.append(sent[index])
                        mask_pos.append(sent_index * max_len + index)
            if seg_label == -1:
                beg = 0
            else:
                beg = token_index
        pre_sent_len = len(sent)

    if len(mask_label) == 0:
        print("mask_label == 0")
        mask_label_all_bei = list(range(20))
        mask_pos_all_bei = list(range(20))
        mask_label = np.array(mask_label_all_bei).astype(int_type).reshape([-1, 1])
        mask_pos = np.array(mask_pos_all_bei).astype(int_type).reshape([-1, 1])
        need_cal_loss = np.array([0]).astype("float32")
    else:
        mask_label = np.array(mask_label).astype(int_type).reshape([-1, 1])
        mask_pos = np.array(mask_pos).astype(int_type).reshape([-1, 1])
        need_cal_loss = np.array([1]).astype("float32")

    return batch_tokens, mask_label, mask_pos, need_cal_loss


def get_random_pos_id(
    batch_pos_ids, is_ends=None, random_pos_start_set=None, random_seed=0
):
    # random.seed(random_seed)
    max_pos_len = 3072
    batch_size = len(batch_pos_ids)
    random_batch_pos_ids = []
    for index, pos_ids in enumerate(batch_pos_ids):
        len_pos = len(pos_ids)
        if random_pos_start_set[index] is None:
            random_pos_start = random.randint(1, max_pos_len - 1)
        else:
            random_pos_start = random_pos_start_set[index]
        random_pos_ids = [0, random_pos_start]
        last_pos_id = random_pos_start
        for _ in range(len_pos - 2):
            random_gap = 1  # random.sample(range(1,4), 1)[0]
            pos_id = (last_pos_id + random_gap) % max_pos_len
            if pos_id == 0:
                pos_id += 1
            random_pos_ids.append(pos_id)
            last_pos_id = pos_id
        assert len(random_pos_ids) == len_pos
        random_batch_pos_ids.append(random_pos_ids)

        if is_ends[index] == 1:
            random_pos_start_set[index] = random.randint(1, max_pos_len - 1)
        else:
            random_pos_start_set[index] = last_pos_id

    return random_batch_pos_ids


def prepare_batch_data(
    insts,
    total_token_num,
    task_index,
    lm_weight,
    task_num,
    task_name,
    branch="nlu",
    random_seed=0,
    lm_ratio=0.15,
    voc_size=0,
    mem_len=128,
    vocab=None,
    prompt_sampler=None,
    max_seq_len=None,
    pad_id=None,
    cls_id=None,
    sep_id=None,
    mask_id=None,
    question_id=None,
    g_mask_id_list=None,
    sent_mask_id=None,
    g_end_id=None,
    end_id=None,
    start_id=None,
    s_id=None,
    return_input_mask=True,
    return_max_len=True,
    return_num_token=False,
    random_pos_start_set=None,
    batch_size_fact=None,
    is_contrast=False,
    add_ditto=False,
    QAReader=None,
    training_mode="AR",
    task_need_convert=[],
):
    max_vocab_len = 15
    if len(g_mask_id_list) == 1:
        g_mask_id = g_mask_id_list[0]
    else:
        g_mask_id, g_mask_id_s, g_mask_id_c, g_mask_id_p = g_mask_id_list
    assert batch_size_fact is not None
    batch_src_ids = [inst[0] for inst in insts]
    batch_sent_ids = [inst[1] for inst in insts]
    batch_pos_ids = [inst[2] for inst in insts]
    # batch_task_ids = [inst[3] for inst in insts]
    if task_index in task_need_convert:
        batch_pos_ids_extra = [inst[3] for inst in insts]
        batch_last_token_label = [inst[4] for inst in insts]
        labels = [inst[6] for inst in insts]
        labels = np.array(labels).astype("int64").reshape([-1, 1])
        seg_labels = [inst[7] for inst in insts]
        batch_is_dulplicated = [inst[8] for inst in insts]
        prefix_lengths = [inst[9] for inst in insts]
        cal_loss = [inst[10] for inst in insts]
        is_starts = [inst[11] for inst in insts]
        is_ends = [inst[12] for inst in insts]
        mask_word_tags = [inst[-1] for inst in insts]
    else:
        labels = [inst[4] for inst in insts]
        labels = np.array(labels).astype("int64").reshape([-1, 1])
        seg_labels = [inst[5] for inst in insts]
        batch_is_dulplicated = [inst[6] for inst in insts]
        prefix_lengths = [inst[7] for inst in insts]
        cal_loss = [inst[8] for inst in insts]
        is_starts = [inst[9] for inst in insts]
        is_ends = [inst[10] for inst in insts]
        mask_word_tags = [inst[-1] for inst in insts]

    if branch == "nlg":
        # batch_pos_ids = get_random_pos_id(batch_pos_ids, is_ends, random_pos_start_set, random_seed=random_seed)
        pass  # 目前不用随机位置编码
    bsz = len(batch_src_ids)
    # First step: do mask without padding
    assert mask_id >= 0, "[FATAL] mask_id must >= 0"
    if task_index not in task_need_convert:
        batch_pos_ids_extra = batch_pos_ids
    use_2d_pos = np.array([0]).astype("float32")
    is_span = np.array(
        [-1]
    )  # np.array([glm_task_name_to_index.get(task_name, -1)]).astype(int_type)

    if add_ditto:
        need_ditto_loss = np.array([0]).astype("float32")
        ditto_pos_1 = np.array(list(range(20))).astype("int64").reshape([-1, 1])
        ditto_pos_2 = np.array(list(range(20))).astype("int64").reshape([-1, 1])
        ditto_label = np.array(list(range(20))).astype("int64").reshape([-1, 1])

    not_mask = False
    if lm_weight < 0.01:
        not_mask = True

    uns_prompt = None
    gather_flag = 0

    if not_mask:
        copy_batch_src_ids = copy.deepcopy(batch_src_ids)
        out = shuffle_entity(
            batch_src_ids,
            seg_labels,
            total_token_num,
            max_seq_len=max_seq_len,
            random_seed=random_seed,
        )
        _, mask_label, mask_pos, need_cal_loss = mask(
            copy_batch_src_ids,
            seg_labels,
            mask_word_tags,
            total_token_num,
            max_seq_len=max_seq_len,
            vocab_size=voc_size,
            CLS=cls_id,
            SEP=sep_id,
            MASK=mask_id,
            random_seed=random_seed,
        )
    else:
        if branch == "nlu":
            if task_name != "relation_pred":
                out, mask_label, mask_pos, need_cal_loss = mask(
                    batch_src_ids,
                    seg_labels,
                    mask_word_tags,
                    total_token_num,
                    max_seq_len=max_seq_len,
                    vocab_size=voc_size,
                    CLS=cls_id,
                    SEP=sep_id,
                    MASK=mask_id,
                    random_seed=random_seed,
                )
            else:
                out, mask_label, mask_pos, need_cal_loss = mask_kg(
                    batch_src_ids,
                    seg_labels,
                    total_token_num,
                    max_seq_len=max_seq_len,
                    vocab_size=voc_size,
                    CLS=cls_id,
                    SEP=sep_id,
                    MASK=mask_id,
                    random_seed=random_seed,
                )
        elif branch == "nlg":
            if task_name == "autogressive_lm":
                out, mask_label, mask_pos, self_input_mask, need_cal_loss = mask_lm(
                    batch_src_ids,
                    prefix_lengths=prefix_lengths,
                    lm_ratio=lm_ratio,
                    mem_len=mem_len,
                    max_seq_len=max_seq_len,
                    cal_loss=cal_loss,
                    is_ends=is_ends,
                    is_starts=is_starts,
                    CLS=cls_id,
                    SEP=sep_id,
                    END=end_id,
                )
            elif "glm_span" in task_name or "multi_prompt_glm_loss" in task_name:
                if is_contrast:
                    src_ids_copy = copy.deepcopy(batch_src_ids)
                    uns_prompt = (
                        prompt_sampler if "multi_prompt" not in task_name else None
                    )
                    gather_flag = 1 if "multi_prompt" not in task_name else 0

                (
                    out,
                    batch_pos_ids,
                    batch_pos_ids_extra,
                    mask_label,
                    mask_pos,
                    mask_sent_pos,
                    self_input_mask,
                    need_cal_loss,
                ) = mask_entity(
                    batch_src_ids,
                    seg_labels,
                    total_token_num,
                    prefix_lengths=prefix_lengths,
                    is_ends=is_ends,
                    max_seq_len=max_seq_len,
                    mem_len=mem_len,
                    START=start_id,
                    END=g_end_id,
                    MASK=g_mask_id,
                    SENT_MASK=sent_mask_id,
                    batch_size_fact=batch_size_fact,
                    uns_prompt=uns_prompt,
                )
                if is_contrast:
                    (
                        out_aug,
                        batch_pos_ids_aug,
                        batch_pos_ids_extra_aug,
                        mask_label_aug,
                        mask_pos_aug,
                        mask_sent_pos_aug,
                        self_input_mask_aug,
                        need_cal_loss_aug,
                    ) = mask_entity(
                        src_ids_copy,
                        seg_labels,
                        total_token_num,
                        prefix_lengths=prefix_lengths,
                        is_ends=is_ends,
                        max_seq_len=max_seq_len,
                        mem_len=mem_len,
                        START=start_id,
                        END=g_end_id,
                        MASK=g_mask_id,
                        SENT_MASK=sent_mask_id,
                        batch_size_fact=batch_size_fact,
                        uns_prompt=uns_prompt,
                    )
                    assert (
                        mask_sent_pos.shape == mask_sent_pos_aug.shape
                    ), "the number of sentence should be equal"
                    out += out_aug
                    batch_pos_ids += batch_pos_ids_aug
                    batch_pos_ids_extra += batch_pos_ids_extra_aug
                    self_input_mask = np.concatenate(
                        (self_input_mask, self_input_mask_aug), axis=0
                    )
                use_2d_pos = np.array([1]).astype("float32")
            elif "glm_sentence" in task_name:
                if is_contrast:
                    src_ids_copy = copy.deepcopy(batch_src_ids)

                (
                    out,
                    batch_pos_ids,
                    batch_pos_ids_extra,
                    mask_label,
                    mask_pos,
                    self_input_mask,
                    need_cal_loss,
                ) = mask_sentence(
                    batch_src_ids,
                    batch_sent_ids,
                    total_token_num,
                    prefix_lengths=prefix_lengths,
                    is_ends=is_ends,
                    doc_end_id=end_id,
                    max_seq_len=max_seq_len,
                    mem_len=mem_len,
                    START=start_id,
                    END=g_end_id,
                    MASK=g_mask_id_s,
                    SID=s_id,
                    batch_size_fact=batch_size_fact,
                )
                if is_contrast:
                    (
                        out_aug,
                        batch_pos_ids_aug,
                        batch_pos_ids_extra_aug,
                        mask_label_aug,
                        mask_pos_aug,
                        self_input_mask_aug,
                        need_cal_loss_aug,
                    ) = mask_sentence(
                        src_ids_copy,
                        batch_sent_ids,
                        total_token_num,
                        prefix_lengths=prefix_lengths,
                        is_ends=is_ends,
                        doc_end_id=end_id,
                        max_seq_len=max_seq_len,
                        mem_len=mem_len,
                        START=start_id,
                        END=g_end_id,
                        MASK=g_mask_id_s,
                        SID=s_id,
                        batch_size_fact=batch_size_fact,
                    )
                    out += out_aug
                    batch_pos_ids += batch_pos_ids_aug
                    batch_pos_ids_extra += batch_pos_ids_extra_aug
                    self_input_mask = np.concatenate(
                        (self_input_mask, self_input_mask_aug), axis=0
                    )
                    mask_sent_pos = (
                        np.array([20]).astype("int64").reshape([-1, 1])
                    )  # 不对比，fake pos
                    mask_sent_pos_aug = mask_sent_pos
                use_2d_pos = np.array([1], dtype="float32")
            elif "glm_chunk" in task_name:
                if is_contrast:
                    src_ids_copy = copy.deepcopy(batch_src_ids)

                (
                    out,
                    batch_pos_ids,
                    batch_pos_ids_extra,
                    mask_label,
                    mask_pos,
                    self_input_mask,
                    need_cal_loss,
                ) = mask_chunk(
                    batch_src_ids,
                    batch_sent_ids,
                    total_token_num,
                    prefix_lengths=prefix_lengths,
                    is_ends=is_ends,
                    doc_end_id=end_id,
                    max_seq_len=max_seq_len,
                    mem_len=mem_len,
                    START=start_id,
                    END=g_end_id,
                    MASK=g_mask_id_c,
                    SID=s_id,
                    batch_size_fact=batch_size_fact,
                )
                if is_contrast:
                    (
                        out_aug,
                        batch_pos_ids_aug,
                        batch_pos_ids_extra_aug,
                        mask_label_aug,
                        mask_pos_aug,
                        self_input_mask_aug,
                        need_cal_loss_aug,
                    ) = mask_chunk(
                        src_ids_copy,
                        batch_sent_ids,
                        total_token_num,
                        prefix_lengths=prefix_lengths,
                        is_ends=is_ends,
                        doc_end_id=end_id,
                        max_seq_len=max_seq_len,
                        mem_len=mem_len,
                        START=start_id,
                        END=g_end_id,
                        MASK=g_mask_id_c,
                        SID=s_id,
                        batch_size_fact=batch_size_fact,
                    )
                    out += out_aug
                    batch_pos_ids += batch_pos_ids_aug
                    batch_pos_ids_extra += batch_pos_ids_extra_aug
                    self_input_mask = np.concatenate(
                        (self_input_mask, self_input_mask_aug), axis=0
                    )
                    mask_sent_pos = (
                        np.array([20]).astype("int64").reshape([-1, 1])
                    )  # 不对比，fake pos
                    mask_sent_pos_aug = mask_sent_pos
                use_2d_pos = np.array([1], dtype="float32")
            elif "glm_partial_lm" in task_name:
                if is_contrast:
                    src_ids_copy = copy.deepcopy(batch_src_ids)

                (
                    out,
                    batch_pos_ids,
                    batch_pos_ids_extra,
                    mask_label,
                    mask_pos,
                    self_input_mask,
                    need_cal_loss,
                ) = mask_partial_lm(
                    batch_src_ids,
                    total_token_num,
                    prefix_lengths=prefix_lengths,
                    is_ends=is_ends,
                    doc_end_id=end_id,
                    max_seq_len=max_seq_len,
                    mem_len=mem_len,
                    START=start_id,
                    END=g_end_id,
                    MASK=g_mask_id_p,
                    SID=s_id,
                    batch_size_fact=batch_size_fact,
                )
                if is_contrast:
                    (
                        out_aug,
                        batch_pos_ids_aug,
                        batch_pos_ids_extra_aug,
                        mask_label_aug,
                        mask_pos_aug,
                        self_input_mask_aug,
                        need_cal_loss_aug,
                    ) = mask_partial_lm(
                        src_ids_copy,
                        total_token_num,
                        prefix_lengths=prefix_lengths,
                        is_ends=is_ends,
                        doc_end_id=end_id,
                        max_seq_len=max_seq_len,
                        mem_len=mem_len,
                        START=start_id,
                        END=g_end_id,
                        MASK=g_mask_id_p,
                        SID=s_id,
                        batch_size_fact=batch_size_fact,
                    )
                    out += out_aug
                    batch_pos_ids += batch_pos_ids_aug
                    batch_pos_ids_extra += batch_pos_ids_extra_aug
                    self_input_mask = np.concatenate(
                        (self_input_mask, self_input_mask_aug), axis=0
                    )
                    mask_sent_pos = (
                        np.array([20]).astype("int64").reshape([-1, 1])
                    )  # 不对比，fake pos
                    mask_sent_pos_aug = mask_sent_pos
                use_2d_pos = np.array([1]).astype("float32")
            elif "qa_partial_lm" in task_name or "sft_partial_lm" in task_name:
                (
                    out,
                    batch_pos_ids,
                    batch_pos_ids_extra,
                    mask_label,
                    mask_pos,
                    self_input_mask,
                    need_cal_loss,
                ) = QAReader.mask_qa_partial_lm_flatten(
                    batch_src_ids,
                    batch_last_token_label,
                    batch_pos_ids,
                    batch_pos_ids_extra,
                    batch_size_fact=batch_size_fact,
                )
                use_2d_pos = np.array([1]).astype("float32")
            elif "fewshot_unidirectional" in task_name:
                if is_contrast:
                    src_ids_copy = copy.deepcopy(batch_src_ids)

                (
                    out,
                    batch_pos_ids,
                    batch_pos_ids_extra,
                    mask_label,
                    mask_pos,
                    self_input_mask,
                    need_cal_loss,
                ) = mask_fewshot_unidirectional(
                    batch_src_ids,
                    seg_labels,
                    total_token_num,
                    prefix_lengths=prefix_lengths,
                    is_ends=is_ends,
                    doc_end_id=end_id,
                    max_seq_len=max_seq_len,
                    mem_len=mem_len,
                    START=start_id,
                    END=g_end_id,
                    MASK=g_mask_id_p,
                    SID=s_id,
                    batch_size_fact=batch_size_fact,
                )
                if is_contrast:
                    (
                        out_aug,
                        batch_pos_ids_aug,
                        batch_pos_ids_extra_aug,
                        mask_label_aug,
                        mask_pos_aug,
                        self_input_mask_aug,
                        need_cal_loss_aug,
                    ) = mask_fewshot_unidirectional(
                        src_ids_copy,
                        seg_labels,
                        total_token_num,
                        prefix_lengths=prefix_lengths,
                        is_ends=is_ends,
                        doc_end_id=end_id,
                        max_seq_len=max_seq_len,
                        mem_len=mem_len,
                        START=start_id,
                        END=g_end_id,
                        MASK=g_mask_id_p,
                        SID=s_id,
                        batch_size_fact=batch_size_fact,
                    )
                    out += out_aug
                    batch_pos_ids += batch_pos_ids_aug
                    batch_pos_ids_extra += batch_pos_ids_extra_aug
                    self_input_mask = np.concatenate(
                        (self_input_mask, self_input_mask_aug), axis=0
                    )
                    mask_sent_pos = (
                        np.array([20]).astype("int64").reshape([-1, 1])
                    )  # 不对比，fake pos
                    mask_sent_pos_aug = mask_sent_pos
                use_2d_pos = np.array([1]).astype("float32")
            elif "ditto" in task_name:
                (
                    out,
                    batch_pos_ids,
                    batch_pos_ids_extra,
                    mask_label,
                    mask_pos,
                    ditto_pos_1,
                    ditto_pos_2,
                    ditto_label,
                    self_input_mask,
                    need_ditto_loss,
                ) = mask_ditto_lm(
                    batch_src_ids,
                    batch_is_dulplicated,
                    total_token_num,
                    prefix_lengths=prefix_lengths,
                    is_ends=is_ends,
                    doc_end_id=end_id,
                    max_seq_len=max_seq_len,
                    mem_len=mem_len,
                    START=start_id,
                    END=g_end_id,
                    MASK=g_mask_id_p,
                    SID=s_id,
                    batch_size_fact=batch_size_fact,
                )
                use_2d_pos = np.array([1]).astype("float32")
                need_cal_loss = np.array([0]).astype("float32")

    # Second step: padding
    src_id, input_mask = pad_batch_data(
        out, pad_idx=pad_id, max_seq_len=batch_size_fact, return_input_mask=True
    )
    if training_mode == "AR":
        pos_id = np.array([list(range(len(inst))) for inst in src_id]).astype("int64")
        pos_id_extra = np.array([list(range(len(inst))) for inst in src_id]).astype(
            "int64"
        )
    else:
        pos_id = pad_batch_data(batch_pos_ids, pad_idx=0, max_seq_len=batch_size_fact)
        pos_id_extra = pad_batch_data(
            batch_pos_ids_extra, pad_idx=0, max_seq_len=batch_size_fact
        )
    # sent_id = pad_batch_data(batch_sent_ids, pad_idx=0, max_seq_len=max_seq_len)
    # task_id = pad_batch_data(batch_task_ids, pad_idx=0, max_seq_len=max_seq_len)
    lm_w = np.array([lm_weight]).astype("float32")

    eff_mask_pos = np.ones_like(mask_pos).astype("float32").reshape([-1, 1])
    eff_ratio = 1.0 * batch_size_fact / mask_pos.shape[0]
    eff_ratio = np.array([eff_ratio]).astype("float32")
    if branch == "nlu":
        is_nlu = np.array([1]).astype("float32")
        is_lm = np.array([0]).astype("float32")
        if mem_len:
            mem_mask = np.zeros([bsz, mem_len, 1], dtype="float32")
            data_mask = np.concatenate((mem_mask, input_mask), axis=1)
            self_input_mask = np.matmul(input_mask, data_mask.transpose([0, 2, 1]))
        else:
            self_input_mask = np.matmul(input_mask, input_mask.transpose([0, 2, 1]))
        # need_cal_loss = np.array([1]).astype("int64")
    elif branch == "nlg":
        is_nlu = np.array([0]).astype("float32")
        is_lm = np.array([1]).astype("float32")
        labels *= 0

    return_list = [src_id, pos_id, pos_id_extra, self_input_mask, mask_label, mask_pos]
    if is_contrast:
        assert len(src_id) == 2
        num_sent = mask_sent_pos.shape[0]
        gather_flag_index = np.array([gather_flag] * num_sent).astype("int64")
        gather_flag_index = gather_flag_index.reshape([-1, 1])
        indicate = np.array([int(need_cal_loss == need_cal_loss_aug)]).astype("float32")
        return_list += [
            mask_sent_pos,
            mask_label_aug,
            mask_pos_aug,
            mask_sent_pos_aug,
            need_cal_loss_aug,
            gather_flag_index,
            indicate,
        ]

    return_list += [
        eff_mask_pos,
        eff_ratio,
        lm_w,
        need_cal_loss,
        is_nlu,
        is_lm,
        use_2d_pos,
        is_span,
    ]
    if add_ditto:
        return_list += [ditto_pos_1, ditto_pos_2, ditto_label, need_ditto_loss]

    task_num = 0  # TODO: hard-code, we don't need task head in pure LM training
    for i in xrange(task_num):
        if i == task_index and branch == "nlu" and task_name not in ["relation_pred"]:
            return_list.append(labels)
            return_list.append(np.array([1.0]).astype("float32"))
        else:
            return_list.append(np.zeros_like(labels))
            return_list.append(np.array([0.0]).astype("float32"))

    return return_list


def get_deep_id(batch_seg_ids, name, max_seq_len=None):
    batch_size = len(batch_seg_ids)
    if max_seq_len is not None:
        max_len = max_seq_len
    else:
        max_len = max(len(seg_ids) for seg_ids in batch_seg_ids)
    padded_deep_ids = np.zeros((batch_size, max_len))
    if name.endswith("graph_list"):
        for i, seg_ids in enumerate(batch_seg_ids):
            value = 0
            last = -1
            for j, seg_id in enumerate(seg_ids):
                if last == 1 and seg_id == 0:
                    value += 1
                padded_deep_ids[i][j] = value
                last = seg_id
    else:
        pass
    return padded_deep_ids.astype(int_type).reshape([batch_size, max_len])


def pad_batch_data(
    insts,
    pad_idx=0,
    return_pos=False,
    max_seq_len=None,
    return_input_mask=False,
    return_max_len=False,
    return_num_token=False,
    return_seq_lens=False,
):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = (
        max_seq_len if max_seq_len is not None else max(len(inst) for inst in insts)
    )
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts]
    )
    return_list += [inst_data.astype(int_type).reshape([-1, max_len])]

    # position data
    if return_pos:
        inst_pos = np.array(
            [
                list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
                for inst in insts
            ]
        )

        return_list += [inst_pos.astype(int_type).reshape([-1, max_len])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array(
            [[1] * len(inst) + [0] * (max_len - len(inst)) for inst in insts]
        )
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
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype(int_type).reshape([-1, 1])]

    return return_list if len(return_list) > 1 else return_list[0]


if __name__ == "__main__":
    cal_loss = [1]
    batch_tokens, mask_label, mask_pos, input_mask_all, need_cal_loss = mask_lm(
        [[i for i in range(16)]],
        max_seq_len=15,
        cal_loss=cal_loss,
        lm_ratio=1.0,
        mem_len=2,
        is_ends=[1],
    )
    print(batch_tokens)
    print(mask_label)
    print(mask_pos)
    print(input_mask_all)
    print(need_cal_loss)
    pass
