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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random


def _gen_self_attn_mask_for_glm(batch_token_ids, max_seq_len=None, start_id=None):
    max_len = max(map(len, batch_token_ids)) if max_seq_len is None else max_seq_len
    input_mask_data = np.zeros(
        (len(batch_token_ids), max_len, max_len), dtype="float32"
    )
    for index, mask_data in enumerate(input_mask_data):
        end = len(batch_token_ids[index])
        # Generate the lower triangular matrix using the slice of matrix
        b = np.tril(np.ones([end, end]), 0)
        mask_data[:end, :end] = b
        # find first glm start token index
        if start_id not in batch_token_ids[index]:
            first_start_index = 0
        else:
            first_start_index = batch_token_ids[index].index(start_id)
        mask_data[:first_start_index, :first_start_index] = (
            1  # mask tokens before first
        )

    return input_mask_data


def _gen_self_attn_mask_for_glm_flatten(
    batch_token_ids, batch_size_fact=None, start_id=None, unbid_idx_1=[], unbid_idx_2=[]
):
    assert (
        len(sum(batch_token_ids, [])) <= batch_size_fact
    ), f"{len(sum(batch_token_ids, []))} > {batch_size_fact} is not allowed"

    input_mask_data = np.ones((1, batch_size_fact, batch_size_fact), dtype="float32")
    input_mask_data[0] = np.tril(input_mask_data, 0)

    return input_mask_data


def mask_partial_lm(
    batch_tokens,
    total_token_num,
    prefix_lengths=None,
    is_ends=None,
    doc_end_id=None,
    max_seq_len=None,
    mem_len=128,
    START=1,
    END=2,
    MASK=3,
    SID=4,
    batch_size_fact=None,
):
    # mask partial language modeling from left to right
    # max_len = max_seq_len if max_seq_len is not None else max(len(inst) for inst in batch_tokens)
    mask_label = []
    mask_pos = []
    # Note: the first token is [CLS], so [low=1]

    new_batch_tokens = []
    batch_pos_ids_1 = []
    batch_pos_ids_2 = []
    batch_last_token = [0] * len(batch_tokens)

    for sent_index, sent in enumerate(batch_tokens):
        new_tokens = []
        masked_spans = []

        if is_ends[sent_index] == 1:
            batch_last_token[sent_index] = START
        else:
            batch_last_token[sent_index] = sent[-1]
            sent = sent[:-1]

        # mask partial
        if random.random() > 0.0:
            plm_ratio = 1.0
        else:
            if random.random() > 0.1:
                plm_ratio = (random.random() + 1) / 2  # range-> 0.5 ~ 1
            else:
                # 20%的概率mask 0~0.5 %
                plm_ratio = random.random() / 2  # range-> 0.0 ~ 0.5

        # DEBUG #
        # plm_ratio = 0.5
        ####
        mask_len = int(np.ceil((len(sent) - prefix_lengths[sent_index]) * plm_ratio))
        # print('plm_ratio', plm_ratio, 'mask_len', mask_len)

        new_tokens.extend(sent[: (len(sent) - mask_len)])
        # new_tokens.append(MASK)  # do not add MASK for AR

        # masked_spans.append(START)  # do not add START for AR
        masked_spans.extend(sent[-mask_len:])

        mask_pos_id = len(new_tokens) - 1  # [MASK] position id

        new_batch_tokens.append(new_tokens + masked_spans)
        pos_id_1 = list(range(len(new_tokens)))
        pos_id_1 += [mask_pos_id] * len(masked_spans)

        pos_id_2 = [0] * len(new_tokens)
        pos_id_2 += list(range(1, len(masked_spans) + 1))

        assert len(new_tokens + masked_spans) == len(pos_id_1) == len(pos_id_2)

        batch_pos_ids_1.append(pos_id_1)
        batch_pos_ids_2.append(pos_id_2)

    # 获取input_mask_all: shape=(1, batch_size_fact, batch_size_fact)
    input_mask_all = _gen_self_attn_mask_for_glm_flatten(
        new_batch_tokens, batch_size_fact=batch_size_fact, start_id=START
    )

    mask_label = []
    mask_pos = []
    pos_offset = 0
    for bsz, tokens in enumerate(new_batch_tokens):
        mask_pos.append([1] * len(tokens))
        is_append_flag = False  # 当遇到第一个start后，开始append pos和label
        for token_i, token in enumerate(tokens):
            if token == START or is_append_flag:
                is_append_flag = True
                # mask_pos.append(pos_offset + token_i)
                if token_i + 1 == len(tokens) or tokens[token_i + 1] == START:
                    # 当前为最后一个token或者下一个token是开始符号，则预测label改为last_token (可能是文章结束，也可能是被截断预测截断的第一个字符)
                    mask_label.append(batch_last_token[bsz])
                else:
                    mask_label.append(tokens[token_i + 1])
            else:
                if token_i + 1 == len(tokens) or tokens[token_i + 1] == START:
                    mask_label.append(batch_last_token[bsz])
                else:
                    mask_label.append(tokens[token_i + 1])

        pos_offset += len(tokens)

    mask_label += [doc_end_id] * (batch_size_fact - len(mask_label))
    mask_pos = sum(mask_pos, [])
    mask_pos += [0] * (batch_size_fact - len(mask_pos))

    if len(mask_label) == 0:
        print("mask_label == 0")
        # fake sample
        new_batch_tokens = [
            [
                1,
                1750,
                1764,
                1799,
                3560,
                1557,
                1601,
                1681,
                1558,
                2608,
                29981,
                1681,
                29980,
            ]
        ]
        tokens_length = len(new_batch_tokens[0])
        batch_pos_ids_1 = [list(range(0, tokens_length))]
        batch_pos_ids_2 = [[0] * tokens_length]
        input_mask_all = _gen_self_attn_mask_for_glm_flatten(
            new_batch_tokens, batch_size_fact=batch_size_fact, start_id=START
        )

        mask_label_all_bei = list(range(10))
        mask_pos_all_bei = list(range(10))
        mask_label = np.array(mask_label_all_bei).astype("int64").reshape([1, -1])
        mask_pos = np.array(mask_pos_all_bei).astype("int64").reshape([1, -1])
        need_cal_loss = np.array([0]).astype("float32")
    else:
        mask_label = np.array(mask_label, dtype="int64").reshape([1, -1])
        mask_pos = np.array(mask_pos, dtype="float32").reshape([1, -1])
        need_cal_loss = np.array([1], dtype="float32")

    # if mem_len:
    #     memory_mask_list = []
    #     for i in range(len(new_batch_tokens)):
    #         each_memory_mask = np.zeros([max_len, mem_len])
    #         memory_mask_list.append(each_memory_mask)
    #     memory_mask = np.array(memory_mask_list).astype("float32")
    #     input_mask_all = np.concatenate((memory_mask, input_mask_all), axis=-1)

    # flatten src_ids and pos ids
    new_batch_tokens = [sum(new_batch_tokens, [])]
    batch_pos_ids_1 = [sum(batch_pos_ids_1, [])]
    batch_pos_ids_2 = [sum(batch_pos_ids_2, [])]

    return (
        new_batch_tokens,
        batch_pos_ids_1,
        batch_pos_ids_2,
        mask_label,
        mask_pos,
        input_mask_all,
        need_cal_loss,
    )


def mask_sentence(
    batch_tokens,
    batch_sent_ids,
    total_token_num,
    prefix_lengths=None,
    is_ends=None,
    doc_end_id=None,
    max_seq_len=None,
    mem_len=128,
    START=1,
    END=2,
    MASK=3,
    SID=4,
    batch_size_fact=None,
):
    """
    Add mask for batch_tokens, return out, mask_label, mask_pos;
    Note: mask_pos responding the batch_tokens after padded;
    """
    # max_len = max_seq_len if max_seq_len is not None else max(len(inst) for inst in batch_tokens)
    # mask_label = []
    # mask_pos = []
    # Note: the first token is [CLS], so [low=1]

    new_batch_tokens = []
    batch_pos_ids_1 = []
    batch_pos_ids_2 = []
    batch_masked_sent_last_token = []
    batch_last_token = [0] * len(batch_tokens)

    for sent_index, sent in enumerate(batch_tokens):
        new_tokens = []
        masked_spans = []

        sent_seg = []
        begin_index, end_index = (
            prefix_lengths[sent_index],
            1 + prefix_lengths[sent_index],
        )
        sent_last_token = []  # 记录被选句子应该预测END还是下一段第一个token

        if prefix_lengths[sent_index] > 0:
            sent_seg.append(sent[: prefix_lengths[sent_index]])  # add prefix ids
        if is_ends[sent_index] == 1:
            batch_last_token[sent_index] = doc_end_id
        else:
            # 被截断过
            batch_last_token[sent_index] = sent[-1]  # 预测截断后的第一个token
            sent = sent[:-1]

        # for <S></S>
        # while end_index < len(sent):
        #    if sent[end_index - 1] == SID:
        #        sent_seg.append(sent[begin_index:end_index])
        #        begin_index = end_index
        #    end_index += 1

        # for /n
        while end_index < len(sent):
            if (
                batch_sent_ids[sent_index][end_index - 1]
                != batch_sent_ids[sent_index][end_index]
            ):
                sent_seg.append(sent[begin_index:end_index])
                begin_index = end_index
            end_index += 1

        if end_index - 1 >= begin_index:
            sent_seg.append(sent[begin_index:end_index])

        sent_count = len(sent_seg)
        # assert sent_count > 1

        # 如果有前缀，则从第二个句子开始预测
        permuated_sent_indices = np.random.permutation(
            np.arange(int(prefix_lengths[sent_index] > 0), sent_count)
        )
        chosen_span_indices = []
        cur_total_tokens = 0
        max_masked_tokens = int(
            sum([len(seg) for seg in sent_seg[int(prefix_lengths[sent_index] > 0) :]])
            * 0.15
        )
        for idx, chosen_span_idx in enumerate(permuated_sent_indices):
            if idx == 0:
                # 总是加入第一个sent，即使超过15%的tokens数
                chosen_span_indices.append(chosen_span_idx)
                cur_total_tokens += len(sent_seg[chosen_span_idx])
                continue

            if cur_total_tokens > max_masked_tokens:
                break

            chosen_span_indices.append(chosen_span_idx)
            cur_total_tokens += len(sent_seg[chosen_span_idx])

        for i in range(sent_count):
            if i in chosen_span_indices:
                if i - 1 in chosen_span_indices:
                    # 合并连续的mask。
                    masked_spans[-1].extend(sent_seg[i])
                    if i == sent_count - 1:
                        sent_last_token[-1] = batch_last_token[sent_index]
                else:
                    new_tokens.append(MASK)
                    masked_spans.append([START] + sent_seg[i])
                    sent_last_token.append(doc_end_id)
                    if i == sent_count - 1:
                        sent_last_token[-1] = batch_last_token[sent_index]
            else:
                new_tokens.extend(sent_seg[i])

        pos_ids_1, pos_ids_2, masked_spans, masked_sent_last_token = gen_glm_pos_ids(
            new_tokens, masked_spans, MASK, sent_last_token
        )
        batch_pos_ids_1.append(pos_ids_1)
        batch_pos_ids_2.append(pos_ids_2)
        batch_masked_sent_last_token.append(masked_sent_last_token)

        masked_spans = [item for sublist in masked_spans for item in sublist]
        new_batch_tokens.append(new_tokens + masked_spans)

        assert len(new_tokens + masked_spans) == len(pos_ids_1) == len(pos_ids_2)

    # 获取input_mask_all: shape=(1, batch_size_fact, batch_size_fact)
    assert (
        len(sum(new_batch_tokens, [])) <= batch_size_fact
    ), f"{len(sum(new_batch_tokens, []))} > {batch_size_fact} is not allowed, {batch_tokens}"
    input_mask_all = _gen_self_attn_mask_for_glm_flatten(
        new_batch_tokens, batch_size_fact=batch_size_fact, start_id=START
    )

    mask_label = []
    mask_pos = []
    pos_offset = 0
    for bsz, tokens in enumerate(new_batch_tokens):
        is_append_flag = False  # 当遇到第一个start后，开始append pos和label
        mask_counter = 0
        for token_i, token in enumerate(tokens):
            if token == START or is_append_flag:
                is_append_flag = True
                mask_pos.append(pos_offset + token_i)
                if token_i + 1 == len(tokens) or tokens[token_i + 1] == START:
                    # 当前为最后一个token或者下一个token是开始符号，则预测label改为END(可能是段落的结尾，或者是句子的结尾)或者
                    # mask_label.append(batch_last_token[bsz])
                    mask_label.append(batch_masked_sent_last_token[bsz][mask_counter])
                    mask_counter += 1
                else:
                    mask_label.append(tokens[token_i + 1])

        pos_offset += len(tokens)

    if len(mask_label) == 0:
        print("mask_label == 0")

        # fake sample
        new_batch_tokens = [
            [
                1,
                1750,
                1764,
                1799,
                3560,
                1557,
                1601,
                1681,
                1558,
                2608,
                29981,
                1681,
                29980,
            ]
        ]
        tokens_length = len(new_batch_tokens[0])
        batch_pos_ids_1 = [list(range(0, tokens_length))]
        batch_pos_ids_2 = [[0] * tokens_length]
        input_mask_all = _gen_self_attn_mask_for_glm_flatten(
            new_batch_tokens, batch_size_fact=batch_size_fact, start_id=START
        )

        mask_label_all_bei = list(range(10))
        mask_pos_all_bei = list(range(10))
        mask_label = np.array(mask_label_all_bei).astype("int64").reshape([-1, 1])
        mask_pos = np.array(mask_pos_all_bei).astype("int64").reshape([-1, 1])
        need_cal_loss = np.array([0]).astype("float32")
    else:
        mask_label = np.array(mask_label, dtype="int64").reshape([-1, 1])
        mask_pos = np.array(mask_pos, dtype="int64").reshape([-1, 1])
        need_cal_loss = np.array([1], dtype="float32")

    # if mem_len:
    #     memory_mask_list = []
    #     for i in range(len(new_batch_tokens)):
    #         each_memory_mask = np.zeros([max_len, mem_len])
    #         memory_mask_list.append(each_memory_mask)
    #     memory_mask = np.array(memory_mask_list).astype("float32")
    #     input_mask_all = np.concatenate((memory_mask, input_mask_all), axis=-1)

    # flatten src_ids and pos ids
    new_batch_tokens = [sum(new_batch_tokens, [])]
    batch_pos_ids_1 = [sum(batch_pos_ids_1, [])]
    batch_pos_ids_2 = [sum(batch_pos_ids_2, [])]

    return (
        new_batch_tokens,
        batch_pos_ids_1,
        batch_pos_ids_2,
        mask_label,
        mask_pos,
        input_mask_all,
        need_cal_loss,
    )


def mask_chunk(
    batch_tokens,
    batch_sent_ids,
    total_token_num,
    prefix_lengths=None,
    is_ends=None,
    doc_end_id=None,
    max_seq_len=None,
    mem_len=128,
    START=1,
    END=2,
    MASK=3,
    SID=4,
    batch_size_fact=None,
):
    """
    Add mask for batch_tokens, return out, mask_label, mask_pos;
    Note: mask_pos responding the batch_tokens after padded;
    """
    # max_len = max_seq_len if max_seq_len is not None else max(len(inst) for inst in batch_tokens)
    # mask_label = []
    # mask_pos = []
    # Note: the first token is [CLS], so [low=1]

    new_batch_tokens = []
    batch_pos_ids_1 = []
    batch_pos_ids_2 = []
    batch_last_token = [0] * len(batch_tokens)

    for sent_index, sent in enumerate(batch_tokens):
        new_tokens = []
        masked_spans = []

        sent_seg = []
        begin_index, end_index = (
            prefix_lengths[sent_index],
            1 + prefix_lengths[sent_index],
        )

        if prefix_lengths[sent_index] > 0:
            sent_seg.append(sent[: prefix_lengths[sent_index]])  # add prefix ids
        if is_ends[sent_index] == 1:
            batch_last_token[sent_index] = doc_end_id
        else:
            # 被截断过
            batch_last_token[sent_index] = sent[-1]  # 预测截断后的第一个token
            sent = sent[:-1]

        # while end_index < len(sent):
        #    if sent[end_index - 1] == SID:
        #        sent_seg.append(sent[begin_index:end_index])
        #        begin_index = end_index
        #    end_index += 1

        while end_index < len(sent):
            if (
                batch_sent_ids[sent_index][end_index - 1]
                != batch_sent_ids[sent_index][end_index]
            ):
                sent_seg.append(sent[begin_index:end_index])
                begin_index = end_index
            end_index += 1

        if end_index - 1 >= begin_index:
            sent_seg.append(sent[begin_index:end_index])

        sent_count = len(sent_seg)
        # assert sent_count > 1

        # 如果有前缀，则从第二个句子开始预测
        start_position = random.randint(
            int(prefix_lengths[sent_index] > 0), sent_count - 1
        )
        span_length = random.randint(1, sent_count - start_position)

        if span_length + start_position != sent_count:
            # 没有mask到最后一个句子，改成预测句子的END
            batch_last_token[sent_index] = END

        mask_pos_id = 0

        for i in range(sent_count):
            if i == start_position:
                new_tokens.append(MASK)
                masked_spans.append(START)
                mask_pos_id = len(new_tokens) - 1

            if i >= start_position and i < start_position + span_length:
                masked_spans.extend(sent_seg[i])
            else:
                new_tokens.extend(sent_seg[i])

        new_batch_tokens.append(new_tokens + masked_spans)
        pos_id_1 = list(range(len(new_tokens)))
        pos_id_1 += [mask_pos_id] * len(masked_spans)

        pos_id_2 = [0] * len(new_tokens)
        pos_id_2 += list(range(1, len(masked_spans) + 1))

        assert len(new_tokens + masked_spans) == len(pos_id_1) == len(pos_id_2)

        batch_pos_ids_1.append(pos_id_1)
        batch_pos_ids_2.append(pos_id_2)

    # 获取input_mask_all: shape=(1, batch_size_fact, batch_size_fact)
    input_mask_all = _gen_self_attn_mask_for_glm_flatten(
        new_batch_tokens, batch_size_fact=batch_size_fact, start_id=START
    )

    mask_label = []
    mask_pos = []
    pos_offset = 0
    for bsz, tokens in enumerate(new_batch_tokens):
        is_append_flag = False  # 当遇到第一个start后，开始append pos和label
        for token_i, token in enumerate(tokens):
            if token == START or is_append_flag:
                is_append_flag = True
                mask_pos.append(pos_offset + token_i)
                if token_i + 1 == len(tokens) or tokens[token_i + 1] == START:
                    # 当前为最后一个token或者下一个token是开始符号，则预测label改为END(可能是段落的结尾，或者是句子的结尾)或者
                    mask_label.append(batch_last_token[bsz])
                else:
                    mask_label.append(tokens[token_i + 1])

        pos_offset += len(tokens)

    if len(mask_label) == 0:
        print("mask_label == 0")

        # fake sample
        new_batch_tokens = [
            [
                1,
                1750,
                1764,
                1799,
                3560,
                1557,
                1601,
                1681,
                1558,
                2608,
                29981,
                1681,
                29980,
            ]
        ]
        tokens_length = len(new_batch_tokens[0])
        batch_pos_ids_1 = [list(range(0, tokens_length))]
        batch_pos_ids_2 = [[0] * tokens_length]
        input_mask_all = _gen_self_attn_mask_for_glm_flatten(
            new_batch_tokens, batch_size_fact=batch_size_fact, start_id=START
        )

        mask_label_all_bei = list(range(10))
        mask_pos_all_bei = list(range(10))
        mask_label = np.array(mask_label_all_bei).astype("int64").reshape([-1, 1])
        mask_pos = np.array(mask_pos_all_bei).astype("int64").reshape([-1, 1])
        need_cal_loss = np.array([0]).astype("float32")
    else:
        mask_label = np.array(mask_label, dtype="int64").reshape([-1, 1])
        mask_pos = np.array(mask_pos, dtype="int64").reshape([-1, 1])
        need_cal_loss = np.array([1], dtype="float32")

    # if mem_len:
    #     memory_mask_list = []
    #     for i in range(len(new_batch_tokens)):
    #         each_memory_mask = np.zeros([max_len, mem_len])
    #         memory_mask_list.append(each_memory_mask)
    #     memory_mask = np.array(memory_mask_list).astype("float32")
    #     input_mask_all = np.concatenate((memory_mask, input_mask_all), axis=-1)

    # flatten src_ids and pos ids
    new_batch_tokens = [sum(new_batch_tokens, [])]
    batch_pos_ids_1 = [sum(batch_pos_ids_1, [])]
    batch_pos_ids_2 = [sum(batch_pos_ids_2, [])]

    return (
        new_batch_tokens,
        batch_pos_ids_1,
        batch_pos_ids_2,
        mask_label,
        mask_pos,
        input_mask_all,
        need_cal_loss,
    )


def mask_entity(
    batch_tokens,
    seg_labels,
    total_token_num,
    prefix_lengths=None,
    is_ends=None,
    max_seq_len=None,
    mem_len=128,
    START=1,
    END=2,
    MASK=3,
    SENT_MASK=4,
    prob_ratio=0.15,
    batch_size_fact=None,
    uns_prompt=None,
):
    """
    Add mask for batch_tokens, return out, mask_label, mask_pos;
    Note: mask_pos responding the batch_tokens after padded;
    mask prob_ratio number of spans
    """
    new_batch_tokens = []
    batch_pos_ids_1 = []
    batch_pos_ids_2 = []

    uns_prompt_index_1 = []
    uns_prompt_index_2 = []
    for sent_index, sent in enumerate(batch_tokens):
        prefix_length = prefix_lengths[sent_index]
        if is_ends[sent_index] != 1:
            sent = sent[:-1]

        new_tokens = []
        masked_spans = []

        # 计算该句子一共有多少个span
        total_span = 0
        for seg in seg_labels[sent_index]:
            total_span += seg == 0
        max_masked_span_num = int(np.ceil(prob_ratio * total_span))

        chosen_span_indices = np.random.choice(
            list(range(total_span)), max_masked_span_num, replace=False
        ).tolist()

        ## 合并[MASK]：如果选择了连续的[MASK]，则合并成一个。 ##
        span_counter = -1
        new_span_counter = -1
        new_chosen_span_indices = []  # 新的span-index,因为有些span被合并了
        for seg_i, seg in enumerate(seg_labels[sent_index]):
            if seg == 0:
                span_counter += 1
                new_span_counter += 1
                if span_counter in chosen_span_indices:
                    # 当前是选择的span
                    if (
                        span_counter - 1 in chosen_span_indices
                        and seg_labels[sent_index][seg_i - 1] != -1
                    ):
                        # 前一个也是选择的span, 修改当前seg=0为1，合并mask
                        seg_labels[sent_index][seg_i] = 1
                        new_span_counter -= 1
                        new_chosen_span_indices.append(new_span_counter)
                    else:
                        # 前一个不是选择的span，则直接加入
                        new_chosen_span_indices.append(new_span_counter)
        new_chosen_span_indices = list(set(new_chosen_span_indices))
        assert len(new_chosen_span_indices) <= len(chosen_span_indices)

        chosen_span_indices = new_chosen_span_indices

        cur_span_num = 0
        cur_span_index = -1
        cur_span = []
        for token_index, token in enumerate(sent):
            seg_label = seg_labels[sent_index][token_index]
            if seg_label == 1:
                cur_span.append(token)  # 1表示是中间的subtoken，直接加入cur_span
            else:
                # 判断是否要把之前的span加入预测
                # if len(cur_span) != 0 and np.random.rand(1)[0] < add_masked_span_prob and cur_span_num < max_masked_span_num:
                # 非空，小于add_masked_span_prob，并且不超过masked span总数，则加入
                if len(cur_span) != 0 and cur_span_index in chosen_span_indices:
                    # 非空，且在提前选中的span中
                    new_tokens.append(MASK)
                    masked_spans.append([START] + cur_span)
                    cur_span_num += 1

                    if cur_span_num > len(chosen_span_indices):
                        # error
                        print(
                            "ValueError",
                            "seg_label",
                            seg_label,
                            "cur_span_index",
                            cur_span_index,
                            "cur_span",
                            cur_span,
                        )
                        raise ValueError(f"{cur_span_num} > {len(chosen_span_indices)}")
                else:
                    # 不加入预测
                    new_tokens.extend(cur_span)

                # 对cur_span重新赋值
                if seg_label == 0:
                    # 0 表示新span的开头
                    cur_span = [token]
                    cur_span_index += 1
                else:
                    # -1 表示 [CLS], [<S>] 等非masked的token
                    cur_span = []
                    new_tokens.append(token)

        # 最后的处理
        if len(cur_span) != 0 and cur_span_index in chosen_span_indices:
            # 非空，且在提前选中的span中
            new_tokens.append(MASK)
            masked_spans.append([START] + cur_span)
            cur_span_num += 1
        else:
            new_tokens.extend(cur_span)
        assert cur_span_num == len(
            chosen_span_indices
        ), f"{cur_span_num} != {len(chosen_span_indices)}"

        # 如果需要加入无监督的prompt
        sampled_prompt = None
        if uns_prompt:
            insert_pos = prefix_length  # 以prefix_length为起点
            curr_token = new_tokens[insert_pos]
            sampled_prompt = uns_prompt.sample()

            # while curr_token == CLS:
            #    insert_pos += 1
            #    curr_token = new_tokens[insert_pos]

            new_tokens = (
                new_tokens[:insert_pos] + sampled_prompt[0] + new_tokens[insert_pos:]
            )
            st = len(new_tokens)
            new_tokens += sampled_prompt[1]
            new_tokens.append(SENT_MASK)

            uns_prompt_index_1.append((insert_pos, insert_pos + len(sampled_prompt[0])))
            uns_prompt_index_2.append((st, len(new_tokens)))

        pos_ids_1, pos_ids_2, masked_spans = gen_glm_pos_ids(
            new_tokens, masked_spans, MASK
        )
        batch_pos_ids_1.append(pos_ids_1)
        batch_pos_ids_2.append(pos_ids_2)

        masked_spans = [item for sublist in masked_spans for item in sublist]

        new_batch_tokens.append(new_tokens + masked_spans)

        assert len(new_tokens + masked_spans) == len(pos_ids_1) == len(pos_ids_2)

    if uns_prompt:
        assert len(uns_prompt_index_1) == len(uns_prompt_index_2) == len(batch_tokens)
    # 获取input_mask_all: shape=(1, batch_size_fact, batch_size_fact)
    input_mask_all = _gen_self_attn_mask_for_glm_flatten(
        new_batch_tokens,
        batch_size_fact=batch_size_fact,
        start_id=START,
        unbid_idx_1=uns_prompt_index_1,
        unbid_idx_2=uns_prompt_index_2,
    )

    mask_label = []
    mask_pos = []
    mask_sent_pos = []
    pos_offset = 0
    for bsz, tokens in enumerate(new_batch_tokens):
        is_append_flag = False  # 当遇到第一个start后，开始append pos和label
        sent_rep_pos = 0
        for token_i, token in enumerate(tokens):
            if token == START or is_append_flag:
                is_append_flag = True
                mask_pos.append(pos_offset + token_i)
                if token_i + 1 == len(tokens) or tokens[token_i + 1] == START:
                    # 当前为最后一个token或者下一个token是开始符号，则预测label改为END
                    mask_label.append(END)
                else:
                    mask_label.append(tokens[token_i + 1])

            if token == SENT_MASK:
                sent_rep_pos = token_i
                mask_sent_pos.append(pos_offset + sent_rep_pos)

        pos_offset += len(tokens)

    if len(mask_label) == 0:
        print("mask_label == 0")
        # print('batch_tokens', batch_tokens, 'new_batch_tokens', new_batch_tokens, 'seg_labels', seg_labels)

        # fake sample
        new_batch_tokens = [
            [
                1,
                1750,
                1764,
                1799,
                3560,
                1557,
                1601,
                1681,
                1558,
                2608,
                29981,
                1681,
                29980,
            ]
        ]
        tokens_length = len(new_batch_tokens[0])
        batch_pos_ids_1 = [list(range(0, tokens_length))]
        batch_pos_ids_2 = [[0] * tokens_length]
        input_mask_all = _gen_self_attn_mask_for_glm_flatten(
            new_batch_tokens, batch_size_fact=batch_size_fact, start_id=START
        )

        mask_label_all_bei = list(range(len(new_batch_tokens[0])))
        mask_pos_all_bei = list(range(len(new_batch_tokens[0])))
        mask_label = np.array(mask_label_all_bei).astype("int64").reshape([-1, 1])
        mask_pos = np.array(mask_pos_all_bei).astype("int64").reshape([-1, 1])
        need_cal_loss = np.array([0]).astype("float32")
    else:
        mask_label = np.array(mask_label, dtype="int64").reshape([-1, 1])
        mask_pos = np.array(mask_pos, dtype="int64").reshape([-1, 1])
        need_cal_loss = np.array([1], dtype="float32")
    mask_sent_pos = np.array(mask_sent_pos).astype("int64").reshape([-1, 1])

    # if mem_len:
    #     memory_mask_list = []
    #     for i in range(len(new_batch_tokens)):
    #         each_memory_mask = np.zeros([max_len, mem_len])
    #         memory_mask_list.append(each_memory_mask)
    #     memory_mask = np.array(memory_mask_list).astype("float32")
    #     input_mask_all = np.concatenate((memory_mask, input_mask_all), axis=-1)

    # flatten src_ids and pos ids
    new_batch_tokens = [sum(new_batch_tokens, [])]
    batch_pos_ids_1 = [sum(batch_pos_ids_1, [])]
    batch_pos_ids_2 = [sum(batch_pos_ids_2, [])]

    return (
        new_batch_tokens,
        batch_pos_ids_1,
        batch_pos_ids_2,
        mask_label,
        mask_pos,
        mask_sent_pos,
        input_mask_all,
        need_cal_loss,
    )


def mask_fewshot_unidirectional(
    batch_tokens,
    seg_labels,
    total_token_num,
    prefix_lengths=None,
    is_ends=None,
    doc_end_id=None,
    max_seq_len=None,
    mem_len=128,
    START=1,
    END=2,
    MASK=3,
    SID=4,
    batch_size_fact=None,
    uns_prompt=None,
):
    # mask partial language modeling from left to right
    # max_len = max_seq_len if max_seq_len is not None else max(len(inst) for inst in batch_tokens)
    mask_label = []
    mask_pos = []
    # Note: the first token is [CLS], so [low=1]

    new_batch_tokens = []
    batch_pos_ids_1 = []
    batch_pos_ids_2 = []
    batch_last_token = [0] * len(batch_tokens)

    for sent_index, sent in enumerate(batch_tokens):
        new_tokens = []
        masked_spans = []

        if is_ends[sent_index] == 1:
            batch_last_token[sent_index] = doc_end_id
        else:
            batch_last_token[sent_index] = sent[-1]
            sent = sent[:-1]

        new_tokens.append(MASK)  # 全单向

        masked_spans.append(START)
        masked_spans.extend(sent)

        mask_pos_id = len(new_tokens) - 1  # [MASK] position id

        new_batch_tokens.append(new_tokens + masked_spans)
        pos_id_1 = list(range(len(new_tokens)))
        pos_id_1 += [mask_pos_id] * len(masked_spans)

        pos_id_2 = [0] * len(new_tokens)
        pos_id_2 += list(range(1, len(masked_spans) + 1))

        assert len(new_tokens + masked_spans) == len(pos_id_1) == len(pos_id_2)

        batch_pos_ids_1.append(pos_id_1)
        batch_pos_ids_2.append(pos_id_2)

    # 获取input_mask_all: shape=(1, batch_size_fact, batch_size_fact)
    input_mask_all = _gen_self_attn_mask_for_glm_flatten(
        new_batch_tokens, batch_size_fact=batch_size_fact, start_id=START
    )

    mask_label = []
    mask_pos = []
    pos_offset = 0
    max_len = input_mask_all.shape[-1]
    for bsz, tokens in enumerate(new_batch_tokens):
        is_append_flag = False  # 当遇到第一个start后，开始append pos和label
        for token_i, token in enumerate(tokens):
            seg_label_i = token_i - 2  # -2 因为最开头加了[MASK]和[START]
            if seg_label_i >= 0:
                if seg_labels[bsz][seg_label_i] != -1:
                    # 为标签，加入预测
                    mask_pos.append(pos_offset + token_i - 1)
                    if (
                        seg_label_i + 1 < len(seg_labels[bsz])
                        and seg_labels[bsz][seg_label_i + 1] == -1
                    ) or (seg_label_i + 1 >= len(seg_labels[bsz])):
                        # 下一个token非标签，或者当前为最后一个token
                        mask_label.append(END)
                    else:
                        # 下一个token为标签，且当前非最后一个token
                        mask_label.append(tokens[token_i])

        pos_offset += len(tokens)

    if len(mask_label) == 0:
        print("mask_label == 0")
        # fake sample
        new_batch_tokens = [
            [
                1,
                1750,
                1764,
                1799,
                3560,
                1557,
                1601,
                1681,
                1558,
                2608,
                29981,
                1681,
                29980,
            ]
        ]
        tokens_length = len(new_batch_tokens[0])
        batch_pos_ids_1 = [list(range(0, tokens_length))]
        batch_pos_ids_2 = [[0] * tokens_length]
        input_mask_all = _gen_self_attn_mask_for_glm_flatten(
            new_batch_tokens, batch_size_fact=batch_size_fact, start_id=START
        )
        mask_label_all_bei = list(range(10))
        mask_pos_all_bei = list(range(10))
        mask_label = np.array(mask_label_all_bei).astype("int64").reshape([-1, 1])
        mask_pos = np.array(mask_pos_all_bei).astype("int64").reshape([-1, 1])
        need_cal_loss = np.array([0]).astype("float32")
    else:
        mask_label = np.array(mask_label).astype("int64").reshape([-1, 1])
        mask_pos = np.array(mask_pos).astype("int64").reshape([-1, 1])
        need_cal_loss = np.array([1]).astype("float32")

    # if mem_len:
    #    memory_mask_list = []
    #    for i in range(len(new_batch_tokens)):
    #        each_memory_mask = np.zeros([max_len, mem_len])
    #        memory_mask_list.append(each_memory_mask)
    #    memory_mask = np.array(memory_mask_list).astype("float32")
    #    input_mask_all = np.concatenate((memory_mask, input_mask_all), axis=-1)

    # flatten src_ids and pos ids
    new_batch_tokens = [sum(new_batch_tokens, [])]
    batch_pos_ids_1 = [sum(batch_pos_ids_1, [])]
    batch_pos_ids_2 = [sum(batch_pos_ids_2, [])]

    return (
        new_batch_tokens,
        batch_pos_ids_1,
        batch_pos_ids_2,
        mask_label,
        mask_pos,
        input_mask_all,
        need_cal_loss,
    )


def mask_ditto_lm(
    batch_tokens,
    batch_is_dulplicated,
    total_token_num,
    prefix_lengths=None,
    is_ends=None,
    doc_end_id=None,
    max_seq_len=None,
    mem_len=128,
    START=1,
    END=2,
    MASK=3,
    SID=4,
    batch_size_fact=None,
    uns_prompt=None,
):
    # mask partial language modeling from left to right
    # max_len = max_seq_len if max_seq_len is not None else max(len(inst) for inst in batch_tokens)
    mask_label = []
    mask_pos = []
    # Note: the first token is [CLS], so [low=1]

    new_batch_tokens = []
    new_batch_dulplicated_ids = []
    batch_pos_ids_1 = []
    batch_pos_ids_2 = []
    batch_last_token = [0] * len(batch_tokens)

    for sent_index, sent in enumerate(batch_tokens):
        new_tokens = []
        new_dulplicated_ids = []
        masked_spans = []

        is_dulplicated = batch_is_dulplicated[sent_index]

        if is_ends[sent_index] == 1:
            batch_last_token[sent_index] = doc_end_id
        else:
            batch_last_token[sent_index] = sent[-1]
            sent = sent[:-1]

        # mask partial
        if random.random() > 0.0:
            plm_ratio = 1.0
        else:
            if random.random() > 0.2:
                plm_ratio = (random.random() + 1) / 2  # range-> 0.5 ~ 1
            else:
                # 20%的概率mask 0~0.5 %
                plm_ratio = random.random() / 2  # range-> 0.0 ~ 0.5

        dulplicate_index = 0
        while (
            dulplicate_index < len(is_dulplicated)
            and is_dulplicated[dulplicate_index] == -1
        ):
            dulplicate_index += 1

        mask_len = int(
            np.ceil((dulplicate_index - prefix_lengths[sent_index]) * plm_ratio)
        ) + len(is_dulplicated[dulplicate_index:])

        # mask_len = int(np.ceil((len(sent) - 1 - prefix_lengths[sent_index]) * plm_ratio)) # -1 for skipping [CLS]

        new_tokens.extend(sent[: (len(sent) - mask_len)])
        new_dulplicated_ids.extend(is_dulplicated[: (len(sent) - mask_len)])

        new_tokens.append(MASK)
        new_dulplicated_ids.append(-1)

        masked_spans.append(START)
        new_dulplicated_ids.append(-1)

        masked_spans.extend(sent[-mask_len:])
        new_dulplicated_ids.extend(is_dulplicated[-mask_len:])

        mask_pos_id = len(new_tokens) - 1

        new_batch_tokens.append(new_tokens + masked_spans)
        new_batch_dulplicated_ids.append(new_dulplicated_ids)

        pos_id_1 = list(range(len(new_tokens)))
        pos_id_1 += [mask_pos_id] * len(masked_spans)

        pos_id_2 = [0] * len(new_tokens)
        pos_id_2 += list(range(1, len(masked_spans) + 1))

        assert (
            len(new_tokens + masked_spans)
            == len(pos_id_1)
            == len(pos_id_2)
            == len(new_dulplicated_ids)
        )

        batch_pos_ids_1.append(pos_id_1)
        batch_pos_ids_2.append(pos_id_2)

    # 获取input_mask_all: shape=(1, batch_size_fact, batch_size_fact)
    input_mask_all = _gen_self_attn_mask_for_glm_flatten(
        new_batch_tokens, batch_size_fact=batch_size_fact, start_id=START
    )

    mask_label = []
    ditto_label = []
    mask_pos = []

    ditto_pos_1 = []
    ditto_pos_2 = []

    pos_offset = 0
    max_len = input_mask_all.shape[-1]
    for bsz, tokens in enumerate(new_batch_tokens):
        is_append_flag = False  # 当遇到第一个start后，开始append pos和label
        start_pos = None
        dulplicated_ids = new_batch_dulplicated_ids[bsz]

        assert len(dulplicated_ids) == len(tokens)

        for token_i, token in enumerate(tokens):
            if token == START:
                start_pos = token_i

            if token == START or is_append_flag:
                is_append_flag = True

                if token_i + 1 < len(tokens) and dulplicated_ids[token_i + 1] > 0:

                    bias = dulplicated_ids[token_i + 1]
                    if token_i - bias < start_pos:
                        continue

                    if tokens[token_i + 1 - bias] != START:
                        assert tokens[token_i + 1 - bias] == tokens[token_i + 1]

                    ditto_pos_1.append(pos_offset + token_i - bias)
                    ditto_pos_2.append(pos_offset + token_i)

                    if token_i + 1 == len(tokens) or tokens[token_i + 1] == START:
                        ditto_label.append(batch_last_token[bsz])
                    else:
                        ditto_label.append(tokens[token_i + 1])

        pos_offset += len(tokens)

    if len(mask_label) == 0:
        mask_label_all_bei = list(range(10))
        mask_pos_all_bei = list(range(10))
        mask_label = np.array(mask_label_all_bei).astype("int64").reshape([-1, 1])
        mask_pos = np.array(mask_pos_all_bei).astype("int64").reshape([-1, 1])
        need_cal_loss = np.array([0]).astype("float32")
    else:
        mask_label = np.array(mask_label, dtype="int64").reshape([-1, 1])
        mask_pos = np.array(mask_pos, dtype="int64").reshape([-1, 1])
        need_cal_loss = np.array([1], dtype="float32")

    if len(ditto_label) == 0:
        print("ditto pos == 0")
        mask_pos_all_bei = list(range(10))
        ditto_pos_1 = np.array(mask_pos_all_bei).astype("int64").reshape([-1, 1])
        ditto_pos_2 = np.array(mask_pos_all_bei).astype("int64").reshape([-1, 1])
        ditto_label = np.array(mask_pos_all_bei).astype("int64").reshape([-1, 1])
        need_ditto_loss = np.array([0]).astype("float32")
    else:
        ditto_pos_1 = np.array(ditto_pos_1).astype("int64").reshape([-1, 1])
        ditto_pos_2 = np.array(ditto_pos_2).astype("int64").reshape([-1, 1])
        ditto_label = np.array(ditto_label).astype("int64").reshape([-1, 1])
        pre_index = np.arange(0, ditto_label.shape[0]).astype("int64").reshape([-1, 1])
        ditto_label = np.concatenate([pre_index, ditto_label], axis=1)
        need_ditto_loss = np.array([1]).astype("float32")

    # if mem_len:
    #    memory_mask_list = []
    #    for i in range(len(new_batch_tokens)):
    #        each_memory_mask = np.zeros([max_len, mem_len])
    #        memory_mask_list.append(each_memory_mask)
    #    memory_mask = np.array(memory_mask_list).astype("float32")
    #    input_mask_all = np.concatenate((memory_mask, input_mask_all), axis=-1)

    # flatten src_ids and pos ids
    new_batch_tokens = [sum(new_batch_tokens, [])]
    batch_pos_ids_1 = [sum(batch_pos_ids_1, [])]
    batch_pos_ids_2 = [sum(batch_pos_ids_2, [])]

    return (
        new_batch_tokens,
        batch_pos_ids_1,
        batch_pos_ids_2,
        mask_label,
        mask_pos,
        ditto_pos_1,
        ditto_pos_2,
        ditto_label,
        input_mask_all,
        need_ditto_loss,
    )


def gen_glm_pos_ids(new_tokens, masked_spans, MASK, sent_last_token=None):
    pos_ids_1 = list(range(len(new_tokens)))
    pos_ids_2 = [0] * len(new_tokens)

    if len(masked_spans) == 0:
        return pos_ids_1, pos_ids_2, masked_spans

    masked_span_pos = [i for i in range(len(pos_ids_1)) if new_tokens[i] == MASK]

    if sent_last_token is None:
        assert len(masked_span_pos) == len(masked_spans)

        zipped_span_and_pos = list(zip(masked_spans, masked_span_pos))
        random.shuffle(zipped_span_and_pos)

        masked_spans, masked_span_pos = list(zip(*zipped_span_and_pos))
    else:
        assert len(masked_span_pos) == len(masked_spans) == len(sent_last_token)

        zipped_span_and_pos = list(zip(masked_spans, masked_span_pos, sent_last_token))
        random.shuffle(zipped_span_and_pos)

        masked_spans, masked_span_pos, masked_sent_last_token = list(
            zip(*zipped_span_and_pos)
        )

    masked_spans = list(masked_spans)
    masked_span_pos_1 = [
        [masked_span_pos[i]] * len(masked_spans[i]) for i in range(len(masked_spans))
    ]
    masked_span_pos_1 = [item for sublist in masked_span_pos_1 for item in sublist]

    masked_span_pos_2 = [list(range(1, len(span) + 1)) for span in masked_spans]
    masked_span_pos_2 = [item for sublist in masked_span_pos_2 for item in sublist]

    pos_ids_1 += masked_span_pos_1
    pos_ids_2 += masked_span_pos_2

    if sent_last_token is None:
        return pos_ids_1, pos_ids_2, masked_spans
    else:
        return pos_ids_1, pos_ids_2, masked_spans, masked_sent_last_token


if __name__ == "__main__":
    # cal_loss = [1]
    # batch_tokens, batch_pos_ids_1, batch_pos_ids_2, mask_label, mask_pos, input_mask_all, need_cal_loss = \
    # mask_sentence(
    #     batch_tokens=[[0, 1, 2, 104, 3, 4]],
    #     #seg_labels=[[-1,0,1,0,0,1,0,1,-1,0,0,0,1,0,0,1,1,0]],
    #     is_ends=[1],
    #     doc_end_id=100,
    #     total_token_num=5,
    #     max_seq_len=5 + 10,
    #     mem_len=0,
    #     START=101,
    #     END=102,
    #     MASK=103,
    #     SID=104
    # )
    # print(batch_tokens)
    # print(batch_pos_ids_1)
    # print(batch_pos_ids_2)
    # print(mask_label)
    # print(mask_pos)
    # print(input_mask_all)
    # print(need_cal_loss)

    # cal_loss = [1]
    # batch_tokens, batch_pos_ids_1, batch_pos_ids_2, mask_label, mask_pos, input_mask_all, need_cal_loss = \
    # mask_partial_lm(
    #     batch_tokens=[[0, 1, 2, 104, 3, 4]],
    #     #seg_labels=[[-1,0,1,0,0,1,0,1,-1,0,0,0,1,0,0,1,1,0]],
    #     is_ends=[1],
    #     doc_end_id=100,
    #     total_token_num=5,
    #     max_seq_len=5 + 10,
    #     mem_len=0,
    #     START=101,
    #     END=102,
    #     MASK=103,
    #     SID=104
    # )
    # print(batch_tokens)
    # print(batch_pos_ids_1)
    # print(batch_pos_ids_2)
    # print(mask_label)
    # print(mask_pos)
    # print(input_mask_all)
    # print(need_cal_loss)

    (
        new_batch_tokens,
        batch_pos_ids_1,
        batch_pos_ids_2,
        mask_label,
        mask_pos,
        input_mask_all,
        need_cal_loss,
    ) = mask_entity(
        batch_tokens=[
            [
                30128,
                30129,
                30130,
                30131,
                30132,
                30133,
                30134,
                30135,
                30136,
                30137,
                30138,
                30139,
                30140,
                30141,
                30142,
                30143,
                30144,
                30145,
                30146,
                30147,
                30148,
                30149,
                30150,
                30151,
                30152,
                30153,
                30154,
                30155,
                30156,
                30157,
                30158,
                30159,
                30160,
                30161,
                30162,
                30163,
                30164,
                30165,
                30166,
                30167,
                30168,
                30169,
                30170,
                30171,
                30172,
                30173,
                30174,
                30175,
                30176,
                30177,
                30178,
                30179,
                30180,
                30181,
                30182,
                30183,
                30184,
                30185,
                30186,
                30187,
                30188,
                30189,
                30190,
                30191,
                1,
                1750,
                1764,
                1799,
                3560,
                1557,
                1601,
                1681,
                1558,
                2608,
                2442,
                1681,
                29980,
            ]
        ],
        seg_labels=[
            [
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                0,
                1,
                1,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                -1,
            ]
        ],
        total_token_num=1024,
        max_seq_len=1024,
        mem_len=0,
        START=29981,
        END=29983,
        MASK=29982,
        prob_ratio=0.15,
        is_ends=[1],
    )

    print(new_batch_tokens)
    print(batch_pos_ids_1)
    print(batch_pos_ids_2)
    print(mask_label)
    print(mask_pos)
    print(input_mask_all)
