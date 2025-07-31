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

import copy
import random
import numpy as np
from collections import namedtuple

# import paddle
# import tokenization_wp


class MultiPromptTuningReader(object):
    def __init__(
        self,
        training_mode,
        max_seq_len,
        mask_token="<mask:5>",
        random_seed=None,
        is_debug=False,
        tokenizer=None,
    ):
        self.use_2d_pos = True
        self.training_mode = training_mode
        assert self.training_mode in [
            "Partial_AR",
            "AR",
            "Both",
        ], f"not support the mode: {self.training_mode}, please check it !"
        self.is_debug = is_debug
        if self.is_debug:
            print(
                "[warning] debug switcher is on, will use only 20 samples for testing"
            )
        print("-" * 20 + "\n" + "use_glm:", self.use_2d_pos, "\n" + "-" * 20 + "\n")

        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.pad_id = self.vocab["<pad>"]
        self.cls_id = self.vocab["<cls>"]
        self.sep_id = self.vocab["<sep>"]
        self.gend_id = self.vocab["<mask:7>"]
        self.start_id = self.vocab["<s>"]
        self.gmask_id = self.vocab[mask_token]
        self.mask_token = mask_token

        self.DEBUG_PRINT = 5

        if random_seed is None:
            assert False, "random_seed can not be None"
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)
        np.random.seed(random_seed)

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _truncate_seqs(self, tokens_of_sub_sentence, max_num_tokens, just_begin=False):
        """truncate_seqs"""
        while True:
            ls = [len(ts) for ts in tokens_of_sub_sentence]
            total_length = sum(ls)
            if total_length <= max_num_tokens:
                break
            max_l = max(ls)
            ind = ls.index(max_l)
            trunc_tokens = tokens_of_sub_sentence[ind]

            assert len(trunc_tokens) >= 1

            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if just_begin:
                del trunc_tokens[0]
            else:
                # if self.rng.random() < 0.5:
                # del trunc_tokens[0]
                # else:
                trunc_tokens.pop()

    def _convert_example_to_record(self, example):
        tokenizer = self.tokenizer
        tokens_prefix = []

        text_src = example.src
        if "问题：" and "回答：" in text_src:  # del hard prompt
            text_src = text_src.replace("问题：", "").replace("回答：", "")
        text_src = text_src.replace(self.mask_token, "")

        if text_src == "":
            return None

        if text_src[-1] != "\n":  # add \n at the end of question
            text_src += "\n"

        is_partial_ar = False
        is_ar = False
        if self.training_mode == "Partial_AR":
            if self.mask_token not in text_src:
                text_src += self.mask_token
            else:
                text_src = text_src
            is_partial_ar = True
        elif self.training_mode == "AR":
            if self.mask_token not in text_src:
                text_src = self.mask_token + text_src
            else:
                text_src = text_src
            is_ar = True
        elif self.training_mode == "Both":
            if self.rng.random() < 0.5:  # set to Partial_AR
                if self.mask_token not in text_src:
                    text_src += self.mask_token
                else:
                    text_src = text_src
                is_partial_ar = True
            else:  # set to AR
                if self.mask_token not in text_src:
                    text_src = self.mask_token + text_src
                else:
                    text_src = text_src
                is_ar = True
        else:
            assert False, f"not support the mode: {training_mode}, please check it !"
        assert (
            is_partial_ar or is_ar
        ), "should set a training mode in ['Partial_AR', 'AR', 'Both'] !"
        assert (is_partial_ar and is_ar) == False, "only choose one training mode !"

        ################
        prompt_keep_in_src_tokens = []

        mask_splitter = self.mask_token
        text_l, text_r = text_src.split(mask_splitter)

        text_target = example.tgt
        tokens_target = tokenizer.tokenize(text_target)

        tokens_l = tokenizer.tokenize(text_l)
        tokens_r = tokenizer.tokenize(text_r)

        token_number_need_to_be_truncated = (
            len(tokens_l)
            + len(tokens_r)
            + len(tokens_target)
            + 2
            + len(prompt_keep_in_src_tokens)
            + len(tokens_prefix)
        ) - self.max_seq_len

        last_token_label = "<s>"
        if token_number_need_to_be_truncated > 0:
            tokens_target_tmp = copy.deepcopy(tokens_target)
            self._truncate_seqs(
                [tokens_l, tokens_r, tokens_target_tmp],
                max_num_tokens=self.max_seq_len
                - 2
                - len(prompt_keep_in_src_tokens)
                - len(tokens_prefix),
            )
            if len(tokens_target_tmp) < len(tokens_target):
                # 被截断过，则修改last_token_label为下一个token
                last_token_label = tokens_target[len(tokens_target_tmp)]
            tokens_target = tokens_target_tmp

        if self.use_2d_pos:
            if is_partial_ar:
                tokens_target = tokens_target
            elif is_ar:
                tokens_target = tokens_r + tokens_target
                tokens_r = []
            else:
                assert False, "not support !"
            tokens_target = ["<s>"] + tokens_target + ["</s>"]
            tokens_src = tokens_prefix + tokens_l + prompt_keep_in_src_tokens + tokens_r
        else:
            tokens_src = tokens_prefix + tokens_l
            assert len(tokens_r) == 0, "单向生成，tokens_r不会被考虑"

        tokens = tokens_src + tokens_target
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        if self.use_2d_pos:
            pos_ids = [0] * len(tokens)
            pos_ids_extra = list(range(len(tokens)))
        else:
            # if self.args.only_opt_prompt_emb:
            # 老百亿模型如果优化soft promps，则从29979向前推self.prompt_num个tokens
            # token_ids = list(range(29979 - self.prompt_num, 29979)) + token_ids[self.prompt_num:]
            pos_ids = list(range(0, len(tokens)))
            pos_ids_extra = [0] * len(tokens_src) + [1] * len(
                tokens_target
            )  # 单向时，该字段用来区分src和tgt

        Record = namedtuple(
            "Record",
            [
                "token_ids",
                "position_ids",
                "position_ids_extra",
                "label",
                "last_token_label",
            ],
        )
        label = 0  # fake label
        if self.DEBUG_PRINT > 0:
            print(
                "[DEBUG]",
                "tokens:",
                tokens,
                "token_ids:",
                token_ids,
                "pos_ids:",
                pos_ids,
                "pos_ids_extra:",
                pos_ids_extra,
                "label",
                label,
                "last_token_label",
                last_token_label,
            )

            if token_number_need_to_be_truncated > 0:
                print(
                    "warning: 数据被截断了，可能会导致训练效果异常，设置更大的MAX_LEN可以避免这个问题！"
                )

        assert len(token_ids) != 0, f"{self.qid_num}, "

        record = Record(
            token_ids=token_ids,
            position_ids=pos_ids,
            position_ids_extra=pos_ids_extra,
            label=label,
            last_token_label=tokenizer.convert_tokens_to_ids([last_token_label])[0],
        )

        self.DEBUG_PRINT -= 1
        return record

    def generator(self, data, task_index=None):
        data = {
            key: data[key]
            for key in list(data.keys())
            if key in ["src", "tgt", "content"]
        }
        Example = namedtuple("Example", list(data.keys()))
        try:
            example = Example(**data)
        except Exception as e:
            print(line)
            raise e
        record = self._convert_example_to_record(example)
        if record is None:
            return None
        token_ids = record.token_ids
        pos_ids = record.position_ids
        pos_ids_extra = record.position_ids_extra
        last_token_label = record.last_token_label

        # generate fake output
        sent_ids = [0] * len(token_ids)
        if isinstance(task_index, int):
            task_ids = [task_index] * len(token_ids)
        else:
            task_ids = [0] * len(token_ids)
        label = 0
        seg_labels = [0] * len(token_ids)
        is_dulplicated = [-1] * len(token_ids)
        prefix_length = 0  # self.prompt_num

        return (
            token_ids,
            sent_ids,
            pos_ids,
            pos_ids_extra,
            last_token_label,
            task_ids,
            label,
            seg_labels,
            is_dulplicated,
            prefix_length,
            1,
            1,
            1,
        )  ## [....., cal_lm_loss, is_starts, is_ends]

    def _gen_self_attn_mask_for_glm(self, batch_token_ids, max_seq_len=None):
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
            if self.start_id not in batch_token_ids[index]:
                first_start_index = 0
            else:
                first_start_index = batch_token_ids[index].index(self.start_id)
            mask_data[:first_start_index, :first_start_index] = (
                1  # mask tokens before first
            )

        return input_mask_data.astype("float32")

    def _gen_self_attn_mask_for_glm_flatten(
        self, batch_token_ids, batch_size_fact=None, unbid_idx_1=[], unbid_idx_2=[]
    ):
        assert (
            len(sum(batch_token_ids, [])) <= batch_size_fact
        ), f"{len(sum(batch_token_ids, []))} > {batch_size_fact} is not allowed"

        input_mask_data = np.zeros(
            (1, batch_size_fact, batch_size_fact), dtype="float32"
        )
        offset = 0
        for index, token_ids in enumerate(batch_token_ids):
            cur_len = len(token_ids)
            b = np.tril(np.ones([cur_len, cur_len]), 0)
            if self.start_id not in batch_token_ids[index]:
                first_start_index = 0
            else:
                first_start_index = batch_token_ids[index].index(self.start_id)
            b[:first_start_index, :first_start_index] = (
                1  # bi-directional attention before the first [START]
            )
            if (
                unbid_idx_1 != [] and unbid_idx_2 != []
            ):  # mask the prompt for sentence embedding
                uns1_s, uns1_e = unbid_idx_1[index]
                uns2_s, uns2_e = unbid_idx_2[index]
                b[:, uns1_s:uns1_e] = 0
                b[:, uns2_s:uns2_e] = 0
                b[uns1_s:uns1_e, uns1_s:uns1_e] = 1
                b[uns1_s:uns1_e, uns2_s:uns2_e] = 1
                b[uns2_s:uns2_e, uns1_s:uns1_e] = 1
                b[uns2_s:uns2_e, uns2_s:uns2_e] = 1
            input_mask_data[0, offset : offset + cur_len, offset : offset + cur_len] = b
            offset += cur_len

        return input_mask_data

    def mask_qa_partial_lm_with_batch(self, batch_token_ids, batch_last_token_label):
        # get input mask
        input_mask = self._gen_self_attn_mask_for_glm(batch_token_ids)

        # obtain mask_pos and mask_label
        mask_label = []
        mask_pos = []
        pos_offset = 0
        max_len = input_mask.shape[-1]
        for bsz, token_ids in enumerate(batch_token_ids):
            is_append_flag = False  # 当遇到第一个start后，开始append pos和label
            for token_i, token_id in enumerate(token_ids):
                if token_id == self.start_id or is_append_flag:
                    is_append_flag = True
                    mask_pos.append(
                        pos_offset + token_i
                    )  # 当时用prefix-tuning后，最前面的N个token会被裁剪掉，所以-N偏移
                    if (
                        token_i + 1 == len(token_ids)
                        or token_ids[token_i + 1] == self.start_id
                    ):
                        # 当前为最后一个token或者下一个token是开始符号，则预测label改为last_token (可能是文章结束，也可能是被截断预测截断的第一个字符)
                        mask_label.append(batch_last_token_label[bsz])
                    else:
                        mask_label.append(token_ids[token_i + 1])
            pos_offset += max_len

        if len(mask_label) == 0:
            print("mask_label == 0")
            mask_label_all_bei = list(range(10))
            mask_pos_all_bei = list(range(10))
            mask_label = np.array(mask_label_all_bei).astype("int64").reshape([-1, 1])
            mask_pos = np.array(mask_pos_all_bei).astype("int64").reshape([-1, 1])
            need_cal_loss = np.array([0]).astype("float32")
        else:
            mask_label = np.array(mask_label).astype("int64").reshape([-1, 1])
            mask_pos = np.array(mask_pos).astype("int64").reshape([-1, 1])
            need_cal_loss = np.array([1]).astype("float32")

        return_list = [batch_token_ids, mask_label, mask_pos, input_mask, need_cal_loss]
        return return_list

    def mask_qa_partial_lm_flatten(
        self,
        batch_token_ids,
        batch_last_token_label,
        batch_pos_ids,
        batch_pos_ids_extra,
        batch_size_fact=None,
    ):
        # 获取input_mask_all: shape=(1, batch_size_fact, batch_size_fact)
        input_mask_all = self._gen_self_attn_mask_for_glm_flatten(
            batch_token_ids, batch_size_fact=batch_size_fact
        )

        mask_label = []
        mask_pos = []
        pos_offset = 0
        for bsz, tokens in enumerate(batch_token_ids):
            mask_pos.append([1] * len(tokens))
            is_append_flag = False  # 当遇到第一个start后，开始append pos和label
            for token_i, token in enumerate(tokens):
                if token == self.start_id or is_append_flag:
                    is_append_flag = True
                    # mask_pos.append(pos_offset + token_i)
                    if (
                        token_i + 1 == len(tokens)
                        or tokens[token_i + 1] == self.start_id
                    ):
                        # 当前为最后一个token或者下一个token是开始符号，则预测label改为last_token (可能是文章结束，也可能是被截断预测截断的第一个字符)
                        mask_label.append(batch_last_token_label[bsz])
                    else:
                        mask_label.append(tokens[token_i + 1])
                else:
                    mask_label.append(tokens[token_i + 1])

            pos_offset += len(tokens)

        mask_label += [self.gend_id] * (batch_size_fact - len(mask_label))
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
                new_batch_tokens, batch_size_fact=batch_size_fact
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

        # flatten src_ids and pos ids
        new_batch_tokens = [sum(batch_token_ids, [])]
        batch_pos_ids = [sum(batch_pos_ids, [])]
        batch_pos_ids_extra = [sum(batch_pos_ids_extra, [])]

        return (
            new_batch_tokens,
            batch_pos_ids,
            batch_pos_ids_extra,
            mask_label,
            mask_pos,
            input_mask_all,
            need_cal_loss,
        )


if __name__ == "__main__":
    pass
