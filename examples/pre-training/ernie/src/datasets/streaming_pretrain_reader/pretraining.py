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

from __future__ import print_function
from __future__ import division

import os
import numpy as np
import gzip
import re
import six
import collections
import copy
import json
import pickle


# import paddle.fluid as fluid
# from utils.glm_uns_prompt import GLMUnsupervisedPrompt # eb35
# os.environ.pop('CREDENTIAL_PROFILES_FILE', None)

from tokenizers.tokenization_eb import ErnieBotTokenizer

# from src.datasets.streaming_pretrain_reader.tokenization import ErnieTokenizer
from src.datasets.streaming_pretrain_reader.qa_reader import MultiPromptTuningReader
from src.datasets.streaming_pretrain_reader.general_reader import GeneralReader
from src.datasets.streaming_pretrain_reader.batching import prepare_batch_data
from src.utils import logger


def cut_sent(para):
    para = re.sub(
        "([。！？\?\!；;])([^。！？\?\!；;”’])", r"\1\n\2", para
    )  # 单字符断句符
    para = re.sub(
        "([。！？\?\!；;][。！？\?\!；;”’])([^。！？\?\!；;”’])", r"\1\n\2", para
    )  # 双字符断句，双引号前有终止符，那么双引号才是句子的终点，
    para = re.sub("(？！”)([^”’])", r"\1\n\2", para)  # 三字符
    para = re.sub("([！。]’”)([^”’])", r"\1\n\2", para)  # 三字符
    para = re.sub("(\…{2}”)([^”’])", r"\1\n\2", para)  # 三字符
    para = re.sub("(\.{6})([^”’])", r"\1\n\2", para)  # 英文省略号
    para = re.sub("(\…{2})([^”’])", r"\1\n\2", para)  # 中文省略号
    # 把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，
    # 需要的再做些简单调整即可。
    sents = para.split("\n")
    sents = [sent for sent in sents if len(sent) != 0]
    return sents


def get_sent_num(sent_ids, prefix_lengths):
    sent_seg = []
    begin_index, end_index = prefix_lengths, 1 + prefix_lengths

    if prefix_lengths > 0:
        sent_seg.append(sent_ids[:prefix_lengths])  # add prefix ids

    # for /n
    while end_index < len(sent_ids):
        if sent_ids[end_index - 1] != sent_ids[end_index]:
            sent_seg.append(sent_ids[begin_index:end_index])
            begin_index = end_index
        end_index += 1
    if end_index - 1 >= begin_index:
        sent_seg.append(sent_ids[begin_index:end_index])
    sent_count = len(sent_seg)

    return sent_count


def get_expected_tokens_number(inst, task_name, s_id):
    src_ids, sent_ids, seg_labels, prefix_lengths = inst[0], inst[1], inst[5], inst[7]
    expect_tokens_num = len(src_ids)
    if (
        "glm_partial_lm" in task_name
        or "glm_chunk" in task_name
        or "fewshot_unidirectional" in task_name
        or "ditto" in task_name
    ):
        # expect_tokens_num = 2 + len(src_ids) + 8 # 2 for [gMASK] and [START], 7 for sentence prompt
        expect_tokens_num = len(src_ids)  # add BOS & EOS in reader

    if "glm_sentence" in task_name:
        sents_num = get_sent_num(sent_ids, prefix_lengths)
        max_masked_span_num = int(np.ceil(sents_num))
        expect_tokens_num = (
            len(src_ids) + 2 * max_masked_span_num + 8
        )  # 每个句子都增加2个token

    if "glm_loss" in task_name or "glm_span" in task_name:
        total_span = 0
        for seg in seg_labels:
            total_span += seg == 0
        max_masked_span_num = int(np.ceil(0.2 * total_span))  # org: 0.15
        # each span will add two extra tokens, i.e., [gMASK] and [START]
        expect_tokens_num = 2 * max_masked_span_num + len(src_ids) + 8

    if "qa_partial_lm" in task_name:
        # expect_tokens_num = len(src_ids) + 2 # 2 for 冗余量
        expect_tokens_num = len(src_ids)  # add BOS & EOS in reader

    return expect_tokens_num


def split_into_sentences(para):
    """
    ditto shit, unused
    """
    para_split = para.split("\n")
    sentences = []
    for idx, sents in enumerate(para_split):
        tmp_sent_split = cut_sent(sents)
        if len(tmp_sent_split) == 0:
            if len(sentences) == 0:
                sentences.extend(["\n"])
            else:
                sentences[-1] = sentences[-1] + "\n"
        else:
            if idx + 1 < len(para_split):
                tmp_sent_split[-1] = tmp_sent_split[-1] + "\n"
            sentences.extend(tmp_sent_split)
    return sentences


def split_and_dulplicated(tokenizer, text):
    """
    ditto shit, unused
    """
    MAX_LEN = 4096
    sentences = split_into_sentences(text)
    token_list = []
    all_len = 0
    for index, sent in enumerate(sentences):
        curr_tokens = tokenizer.tokenize(sent)
        all_len += len(curr_tokens)
        if all_len >= MAX_LEN // 2:
            break

        token_list.append(curr_tokens)

    duplicated_tokens = []
    if len(token_list) < 1:
        return []
    end_ids = self.rng.choice(list(range(0, len(token_list))))
    for index in range(end_ids):
        duplicated_tokens.extend(token_list[index])

    is_dulplicated = [-1] * len(duplicated_tokens)
    duplicated_len = sum([len(token_list[i]) for i in range(end_ids, len(token_list))])
    curr_len = len(duplicated_tokens)

    dup_times = 0
    while curr_len + duplicated_len < MAX_LEN:
        for i in range(end_ids, len(token_list)):
            duplicated_tokens.extend(token_list[i])
            if dup_times == 0:
                is_dulplicated.extend([0] * len(token_list[i]))
            else:
                is_dulplicated.extend([duplicated_len] * len(token_list[i]))

        curr_len += duplicated_len
        dup_times += 1

    assert len(duplicated_tokens) <= MAX_LEN
    assert dup_times > 1
    duplicated_token_ids = tokenizer.convert_tokens_to_ids(duplicated_tokens)
    return duplicated_token_ids, is_dulplicated


class ErnieDataReader:
    def __init__(
        self,
        task_group,
        is_valid,
        vocab_path,
        lm_ratio=0.15,
        batch_size=4096,
        max_seq_len=512,
        shuffle_files=True,
        epoch=100,
        voc_size=0,
        mem_len=128,
        is_test=False,
        generate_neg_sample=False,
        hack_old_trainset=False,
        random_seed=None,
        dp_worldsize=None,
        worker_index=None,
        is_contrast=False,
        add_ditto=False,
        training_mode="AR",
        task_need_convert="",
        tokenizer="FullTokenizer",
        micro_bsz=1,
        acc_steps=None,
        output_dir="./output/ernie35",
    ):
        self.is_debug = False
        # self.tokenizer = ErnieTokenizer(model_file=vocab_path) if vocab_path.endswith('.model') else \
        self.tokenizer = ErnieBotTokenizer.from_pretrained(vocab_path)
        self.vocab = self.tokenizer.get_vocab()
        self.voc_size = voc_size
        print(
            f"vocab size in args = {voc_size}, vocab size in vocab: {self.tokenizer.vocab_size}"
        )
        assert voc_size == self.tokenizer.vocab_size
        logger.warning(f"task_group:\t{task_group}")
        logger.warning(f"random_seed:\t{random_seed}")
        logger.warning(f"dp_worldsize:\t{dp_worldsize}")
        logger.warning(f"worker_index:\t{worker_index}")
        self.glm_prompt_sampler = None  # GLMUnsupervisedPrompt(self.tokenizer)
        # =============================================
        # !!! (tobefixed). Inconsistent vocab in "dict/prompt_vocab.txt"
        self.prompt_vocab = {}  # self.load_vocab("dict/prompt_vocab.txt")
        # =============================================
        self.task_group = task_group
        self.is_valid = is_valid
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.epoch = epoch
        self.current_epoch = 0
        self.current_file_index = 0
        self.total_file = 0
        self.current_file = None
        self.max_seq_len = max_seq_len
        self.is_contrast = is_contrast
        self.add_ditto = add_ditto
        self.training_mode = training_mode
        self.acc_steps = acc_steps
        self.micro_bsz = micro_bsz
        self.start_id = self.vocab["<s>"]
        self.pad_id = self.vocab["<pad>"]
        self.cls_id = self.vocab["<cls>"]
        self.sep_id = self.vocab["<sep>"]
        self.mask_id = self.vocab["<mask:0>"]  # [MASK]
        self.s_id = self.vocab["<mask:1>"]  # [<S>]

        self.glm_mask_id = self.vocab["<mask:2>"]  # [gMASK]
        self.glm_mask_id_s = self.vocab["<mask:3>"]  # [sMASK]
        self.glm_mask_id_c = self.vocab["<mask:4>"]  # [cMASK]
        self.glm_mask_id_p = self.vocab["<mask:5>"]  # [pMASK]
        self.sent_mask_id = self.vocab["<mask:6>"]  # [CLSMASK]

        # 统一end
        self.glm_end_id = self.vocab["<mask:7>"]  # [gEND]
        self.end_id = self.vocab["</s>"]  # [END]
        self.return_id = self.vocab["<mask:8>"]  # [<N>]
        self.tab_id = self.vocab["<mask:9>"]  # [<T>]
        self.halftab_id = self.vocab["<mask:10>"]  # [<t>]

        self.question_id = self.vocab["<mask:11>"]  # [QUESTION]

        self.comma_id = self.vocab.get("，", self.vocab["<mask:12>"])

        self.input_slots = 5
        self.is_test = is_test
        self.lm_ratio = lm_ratio
        self.mem_len = mem_len
        self.generate_neg_sample = generate_neg_sample
        self.random_pos_start_set = None
        self.bsz_per_core = self.batch_size // self.max_seq_len
        self.random_seed = random_seed
        self.dp_worldsize = dp_worldsize
        self.worker_index = worker_index
        self.IS_PRINT_CNT_FOR_DEBUG = (
            1  # [FOR DEBUG] print samples for IS_PRINT_CNT_FOR_DEBUG times.
        )
        self.output_dir = output_dir
        self.rng = np.random.RandomState()
        self.rng.seed(random_seed)

        os.makedirs(output_dir, exist_ok=True)

        # np.random.seed(random_seed)
        # random.seed(random_seed)

        assert (
            self.batch_size > 100
        ), "Current batch size means total token's number, \
                                       it should not be set to too small number."

        if hack_old_trainset:
            self.input_slots = 4

        if self.is_test:
            self.epoch = 1
            self.shuffle_files = False

        self.QAReader = MultiPromptTuningReader(
            training_mode,
            max_seq_len,
            random_seed=random_seed,
            tokenizer=self.tokenizer,
        )
        assert (
            task_need_convert != ""
        ), f"the task need to convert should not be None !, but now is : {task_need_convert}"
        self.task_need_convert = [
            int(t) for t in task_need_convert.split(",")
        ]  # task_need_convert
        self.split_task_idx = self.task_need_convert[-1]

        self.GeneralReader = GeneralReader(
            random_seed=random_seed, tokenizer=self.tokenizer
        )

    def get_progress(self):
        """return current progress of traning data"""
        return (
            self.current_epoch,
            self.current_file_index,
            self.total_file,
            self.current_file,
            self.mask_type,
        )

    def gen_ngram_seg(self, length):
        seg_labels = []
        while len(seg_labels) < length:
            rand_n_gram = self.rng.randint(1, min(length - len(seg_labels), 4))
            seg_labels.extend([0] + [1] * (rand_n_gram - 1))

        assert len(seg_labels) == length
        return seg_labels

    def parse_line(self, line, max_seq_len=512, task_index=None):
        """parse one line to token_ids, sentence_ids, pos_ids, label"""
        task_type = None
        control_type = None
        dataset_name = None
        is_dulplicated = None

        if isinstance(line, dict):
            task_name = self.task_group[task_index].get("task_name", None)
            if task_index in self.task_need_convert:  # for qa datasets, directly return
                outs = self.QAReader.generator(line, task_index)
                if outs is None:
                    return None
                (
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
                    cal_lm_loss,
                    is_starts,
                    is_ends,
                ) = outs

                if label != -1:
                    if len(token_ids) > max_seq_len:
                        return None
                assert (
                    len(token_ids)
                    == len(sent_ids)
                    == len(pos_ids)
                    == len(seg_labels)
                    == len(task_ids)
                ), "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids) == len(seg_labels) == len(task_ids)"

                return [
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
                    cal_lm_loss,
                    is_starts,
                    is_ends,
                ]
            elif "ditto" in task_name:
                if "content" in line and isinstance(line["content"], str):
                    token_ids, is_dulplicated = split_and_dulplicated(
                        self.tokenizer, line["content"]
                    )
                    for i in range(len(token_ids)):
                        bias = int(is_dulplicated[i])
                        if bias > 0:
                            assert token_ids[i] == token_ids[i - bias]
                    sent_ids = [0] * len(token_ids)
                    natual_sent_ids = [0] * len(token_ids)
                    task_ids = [0] * len(token_ids)
                    pos_ids = [0] * len(token_ids)
                    pos_ids_extra = [0] * len(token_ids)
                    seg_labels = [0] * len(token_ids)
                    last_token_label = "<s>"
                    prefix_length = 0
                    label = 0
                else:
                    return None
            else:  # for general datasets, prepare for data_util func
                outs = self.GeneralReader.generator(line, task_index, task_name)
                if outs is None:
                    return None
                (
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
                    cal_lm_loss,
                    is_starts,
                    is_ends,
                ) = outs
                natual_sent_ids = sent_ids
                assert (
                    len(token_ids)
                    == len(sent_ids)
                    == len(pos_ids)
                    == len(seg_labels)
                    == len(task_ids)
                ), "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids) == len(seg_labels) == len(task_ids)"
        else:
            return None

        task = self.task_group[task_index]
        data_func = task.get("data_func", None)
        prefix_length = 0  # prefix_length for masking out the prefix tokens

        # print('line length:', len(line), 'data_func:', data_func, 'task:', task)

        if data_func and data_func != "":
            if "glm" in task.get("data_func", None):
                if control_type is None:
                    # token_ids, sent_ids, pos_ids, label, seg_labels, natual_sent_ids, cls_id, sep_id, s_id, task_type=None, dataset_name=None, vocab=None
                    all_vocab = {**self.vocab, **self.prompt_vocab}
                    (
                        token_ids,
                        sent_ids,
                        pos_ids,
                        label,
                        seg_labels,
                        prefix_length,
                    ) = eval("data_utils." + data_func)(
                        token_ids,
                        sent_ids,
                        pos_ids,
                        label,
                        seg_labels,
                        natual_sent_ids,
                        self.cls_id,
                        self.sep_id,
                        self.s_id,
                        task_type=task_type,
                        dataset_name=dataset_name,
                        vocab=all_vocab,
                        task_name=task.get("task_name", None),
                        max_seq_len=self.max_seq_len,
                    )
                else:
                    ret = eval("data_utils." + data_func)(
                        token_ids,
                        sent_ids,
                        pos_ids,
                        label,
                        seg_labels,
                        natual_sent_ids,
                        self.cls_id,
                        self.sep_id,
                        self.s_id,
                        comma_id=self.comma_id,
                        control_type=control_type,
                        dataset_name=dataset_name,
                        vocab=self.vocab,
                        task_name=task.get("task_name", None),
                        max_seq_len=self.max_seq_len,
                        topic_ls=topic,
                        keyphrase_ls=keyphrase,
                        sentiment=sentiment,
                        words=words,
                        prompt_vocab=self.prompt_vocab,
                    )
                    if ret is None:
                        return None
                    else:
                        (
                            token_ids,
                            sent_ids,
                            pos_ids,
                            label,
                            seg_labels,
                            prefix_length,
                        ) = ret
            elif "ditto" in task.get("data_func", None):
                (
                    token_ids,
                    sent_ids,
                    pos_ids,
                    is_dulplicated,
                    seg_labels,
                    prefix_length,
                ) = eval("data_utils." + data_func)(
                    token_ids,
                    sent_ids,
                    pos_ids,
                    is_dulplicated,
                    seg_labels,
                    natual_sent_ids,
                    self.cls_id,
                    self.sep_id,
                    self.s_id,
                    task_type=task_type,
                    dataset_name=dataset_name,
                    vocab=self.vocab,
                    task_name=task.get("task_name", None),
                    max_seq_len=self.max_seq_len,
                )
            else:
                if task_type is None:
                    token_ids, sent_ids, pos_ids, label, seg_labels = eval(
                        "data_utils." + data_func
                    )(
                        token_ids,
                        sent_ids,
                        pos_ids,
                        label,
                        seg_labels,
                        natual_sent_ids,
                        self.cls_id,
                        self.sep_id,
                        self.s_id,
                    )
                else:
                    ret = eval("data_utils." + data_func)(
                        token_ids,
                        sent_ids,
                        pos_ids,
                        label,
                        seg_labels,
                        natual_sent_ids,
                        self.cls_id,
                        self.sep_id,
                        self.s_id,
                        self.comma_id,
                        task_type,
                        topic,
                        keyphrase,
                        sentiment,
                        words,
                        self.vocab,
                    )
                    if ret is None:
                        return None
                    else:
                        (
                            token_ids,
                            sent_ids,
                            pos_ids,
                            label,
                            seg_labels,
                            prefix_length,
                        ) = ret

        if isinstance(task_index, int):
            task_ids = [task_index] * len(token_ids)
        else:
            task_ids = [0] * len(token_ids)
        assert (
            len(token_ids)
            == len(sent_ids)
            == len(pos_ids)
            == len(seg_labels)
            == len(task_ids)
        ), "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids) == len(seg_labels) == len(task_ids)"

        if label != -1:
            if len(token_ids) > max_seq_len:
                return None
        return [
            token_ids,
            sent_ids,
            pos_ids,
            task_ids,
            label,
            seg_labels,
            is_dulplicated,
            prefix_length,
            1,
            1,
            1,
        ]  ## [....., cal_lm_loss, is_starts, is_ends]

    def cnt_list(self, inp):
        cnt = 0
        for lit in inp:
            if lit:
                cnt += 1
        return cnt

    def get_sample(self, pre_list):
        pre_batch_list = pre_list
        samples = []
        len_doc = [len(doc) for doc in pre_batch_list]
        min_len = min(len_doc)
        for cnt in range(min_len):
            for idx in range(self.bsz_per_core):
                try:
                    sample = pre_batch_list[idx][cnt]
                    samples.append(sample)
                except IndexError:
                    print("IndexError", idx, cnt, len(pre_batch_list))
        for idx in range(self.bsz_per_core):
            pre_batch_list[idx] = pre_batch_list[idx][min_len:]

        return samples

    def get_last_sample(self, data):
        pre_batch_list = data
        # print(pre_seg_labels)
        len_doc = [len(doc) for doc in pre_batch_list]
        max_len = max(len_doc)
        max_len_idx = len_doc.index(max_len)
        copy_sample = copy.deepcopy(pre_batch_list[max_len_idx][-1])
        copy_sample[-3] = 0
        copy_sample[-2] = 0
        copy_sample[-1] = 1
        for i in range(len(pre_batch_list)):
            if len_doc[i] < max_len:
                diff = max_len - len_doc[i]
                pre_batch_list[i].extend([copy_sample] * diff)

        # print(pre_batch_list, pre_sent_ids, pre_time_inds)
        samples = self.get_sample(pre_batch_list)
        return samples

    def preprocess(self, data_line, pre_batch_list, insert_idx):
        (
            data,
            sent_ids,
            pos_ids,
            task_ids,
            label,
            seg_labels,
            is_dulplicated,
            prefix_length,
            cal_lm_loss,
            is_start,
            is_end,
        ) = data_line

        samples = []
        start_index = 0 if prefix_length == 0 else self.max_seq_len - 1
        if prefix_length != 0:
            pos_ids = np.array(pos_ids)
            pos_ids = pos_ids.tolist()

            inp = data[: self.max_seq_len]
            sent_id = sent_ids[: self.max_seq_len - 1]
            pos_id = pos_ids[: self.max_seq_len - 1]
            task_id = task_ids[: self.max_seq_len - 1]
            seg_label = seg_labels[: self.max_seq_len - 1]
            is_dulplicated = is_dulplicated[: self.max_seq_len - 1]
            is_end = 0
            is_start = 0
            samples.append(
                [
                    inp,
                    sent_id,
                    pos_id,
                    task_id,
                    label,
                    seg_label,
                    is_dulplicated,
                    prefix_length,
                    1,
                    is_start,
                    is_end,
                ]
            )  ## [token,*****, prefix_length, cal_lm_loss, is_start, is_end]

        for cnt in range(start_index, len(data), self.max_seq_len - 1):
            inp = data[cnt : cnt + self.max_seq_len]
            sent_id = sent_ids[cnt : cnt + self.max_seq_len - 1]
            pos_id = pos_ids[cnt : cnt + self.max_seq_len - 1]
            task_id = task_ids[cnt : cnt + self.max_seq_len - 1]
            seg_label = seg_labels[cnt : cnt + self.max_seq_len - 1]
            if cnt == 0:
                is_dulplicated = is_dulplicated[: self.max_seq_len - 1]
            else:
                is_dulplicated = [0] * len(seg_label)

            is_end = 0
            is_start = 0
            samples.append(
                [
                    inp,
                    sent_id,
                    pos_id,
                    task_id,
                    label,
                    seg_label,
                    is_dulplicated,
                    0,
                    1,
                    is_start,
                    is_end,
                ]
            )  ## [token,*****, prefix_length, cal_lm_loss, is_start, is_end]

        assert len(pos_id) <= self.max_seq_len
        ## set the is_end == 1 for the last sample of a parsed_line
        samples[-1][-1] = 1
        ## set the is_start == 1 for the first sample of a parsed_line
        samples[0][-2] = 1

        if self.cnt_list(pre_batch_list) < self.bsz_per_core:
            if insert_idx:
                pre_batch_list[insert_idx[0]] = samples
                insert_idx.pop(0)
            else:
                pre_batch_list.append(samples)

        if self.cnt_list(pre_batch_list) == self.bsz_per_core:
            assert self.cnt_list(pre_batch_list) == len(
                pre_batch_list
            ), "the two value must be equal"
            assert not insert_idx, "the insert_idx must be null"
            samples_batch = self.get_sample(pre_batch_list)

            for idx, lit in enumerate(pre_batch_list):
                if not lit:
                    insert_idx.append(idx)

            for sample in samples_batch:
                # print(sample[-2:])
                if sample is None:
                    continue
                yield sample

        # if self.is_valid:
        #    if self.cnt_list(pre_batch_list):
        #        samples_batch = get_last_sample(pre_batch_list)
        #        for sample in samples_batch:
        #            if sample is None:
        #                continue
        #            yield sample

    def read_file(
        self, file, task_index
    ):  # assert file.endswith('.gz'), "[ERROR] %s is not a gzip file" % file
        if file.endswith(".gz"):
            f = gzip.open(file, "rb")
        else:
            f = open(file, "r")
        lines = f.readlines()

        if not self.is_valid:
            # np.random.shuffle(lines)
            self.rng.shuffle(lines)

        if self.IS_PRINT_CNT_FOR_DEBUG > 0:
            # FOR DEBUG to check the loaded data form is correct, set IS_PRINT_CNT_FOR_DEBUG as 0 to turn off the DEBUG mode
            print(lines[0])
            self.IS_PRINT_CNT_FOR_DEBUG -= 1

        pre_batch_list, insert_idx = [], []
        for line in lines:
            try:
                line = json.loads(line.strip())
            except:
                # line = line
                continue
            parsed_line = self.parse_line(
                line, max_seq_len=self.max_seq_len, task_index=task_index
            )
            if parsed_line is None:
                continue
            else:
                if parsed_line[-7] == -1:  # label == -1
                    parsed_line = self.preprocess(
                        parsed_line, pre_batch_list, insert_idx
                    )
                    for inp in parsed_line:
                        yield inp
                else:
                    yield parsed_line

        if self.is_valid:
            if self.cnt_list(pre_batch_list):
                samples_batch = self.get_last_sample(pre_batch_list)
                for sample in samples_batch:
                    if sample is None:
                        continue
                    yield sample
        f.close()

    def convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        fin = open(vocab_file)
        for num, line in enumerate(fin):
            items = self.convert_to_unicode(line.strip()).split("\t")
            if len(items) > 2:
                break
            token = items[0]
            index = items[1] if len(items) == 2 else num
            token = token.strip()
            vocab[token] = int(index)
        return vocab

    def random_pair_neg_samples(self, pos_samples):
        """randomly generate negtive samples using pos_samples

        Args:
            pos_samples: list of positive samples

        Returns:
            neg_samples: list of negtive samples
        """
        # np.random.shuffle(pos_samples)
        self.rng.shuffle(pos_samples)
        num_sample = len(pos_samples)
        neg_samples = []
        miss_num = 0

        def split_sent(sample, max_len, sep_id):
            token_seq, type_seq, pos_seq, task_seq, label, seg_labels = sample
            sep_index = token_seq.index(sep_id)
            left_len = sep_index - 1
            if left_len <= max_len:
                return (token_seq[1:sep_index], seg_labels[1:sep_index])
            else:
                return [token_seq[sep_index + 1 : -1], seg_labels[sep_index + 1 : -1]]

        for i in range(num_sample):
            pair_index = (i + 1) % num_sample
            left_tokens, left_seg_labels = split_sent(
                pos_samples[i], (self.max_seq_len - 3) // 2, self.sep_id
            )
            right_tokens, right_seg_labels = split_sent(
                pos_samples[pair_index],
                self.max_seq_len - 3 - len(left_tokens),
                self.sep_id,
            )

            token_seq = (
                [self.cls_id]
                + left_tokens
                + [self.sep_id]
                + right_tokens
                + [self.sep_id]
            )
            if len(token_seq) > self.max_seq_len:
                miss_num += 1
                continue
            type_seq = [0] * (len(left_tokens) + 2) + [1] * (len(right_tokens) + 1)
            pos_seq = range(len(token_seq))
            task_seq = [task_seq[0]] * len(token_seq)
            seg_label_seq = [-1] + left_seg_labels + [-1] + right_seg_labels + [-1]

            assert (
                len(token_seq)
                == len(type_seq)
                == len(pos_seq)
                == len(seg_label_seq)
                == len(task_seq)
            ), "[ERROR]len(src_id) == lne(sent_id) == len(pos_id) must be True"
            neg_samples.append(
                [token_seq, type_seq, pos_seq, task_seq, 0, seg_label_seq]
            )

        return neg_samples, miss_num

    def mixin_negtive_samples(self, pos_sample_generator, buffer=1000):
        """1. generate negtive samples by randomly group sentence_1 and sentence_2 of positive samples
        2. combine negtive samples and positive samples

        Args:
            pos_sample_generator: a generator producing a parsed positive sample, which is a list: [token_ids, sent_ids, pos_ids, 1]

        Returns:
            sample: one sample from shuffled positive samples and negtive samples
        """
        pos_samples = []
        num_total_miss = 0
        pos_sample_num = 0
        try:
            while True:
                while len(pos_samples) < buffer:
                    pos_sample = next(pos_sample_generator)
                    label = pos_sample[3]
                    assert label == 1, "positive sample's label must be 1"
                    pos_samples.append(pos_sample)
                    pos_sample_num += 1

                neg_samples, miss_num = self.random_pair_neg_samples(pos_samples)
                num_total_miss += miss_num
                samples = pos_samples + neg_samples
                pos_samples = []
                # np.random.shuffle(samples)
                self.rng.shuffle(samples)
                for sample in samples:
                    yield sample
        except StopIteration:
            print("stopiteration: reach end of file")
            if len(pos_samples) == 1:
                yield pos_samples[0]
            elif len(pos_samples) == 0:
                yield None
            else:
                neg_samples, miss_num = self.random_pair_neg_samples(pos_samples)
                num_total_miss += miss_num
                samples = pos_samples + neg_samples
                pos_samples = []
                self.rng.shuffle(samples)
                for sample in samples:
                    yield sample
            print(
                "miss_num:%d\tideal_total_sample_num:%d\tmiss_rate:%f"
                % (
                    num_total_miss,
                    pos_sample_num * 2,
                    num_total_miss / (pos_sample_num * 2),
                )
            )

    def data_generator(self):
        """
        data_generator
        """

        filelist_key = "train_filelist"
        if self.is_valid:
            filelist_key = "valid_filelist"

        all_files = []
        task_probs = []
        sum = 0.0
        for task in self.task_group:
            all_files.append(open(task[filelist_key]).readlines())
            task_probs.append(task["prob"])
            sum += task["prob"]

        for i in range(len(task_probs)):
            task_probs[i] = task_probs[i] / sum

        task_probs = np.array(task_probs).ravel()
        # logger.info(f"number of task_probs: {task_probs.shape}")

        def wrapper():
            def reader(task_index):
                files = all_files[task_index]
                for epoch in range(self.epoch):
                    if self.shuffle_files:
                        # np.random.shuffle(files)
                        self.rng.shuffle(files)
                    for index, file in enumerate(files):
                        file, mask_word_prob = file.strip().split("\t")
                        # mask_word = (np.random.random() < float(mask_word_prob))
                        mask_word = self.rng.random() < float(mask_word_prob)

                        if mask_word:
                            self.mask_type = "mask_word"
                        else:
                            self.mask_type = "mask_char"

                        sample_generator = self.read_file(file, task_index)
                        if not self.is_test and self.generate_neg_sample:
                            sample_generator = self.mixin_negtive_samples(
                                sample_generator
                            )

                        for sample in sample_generator:
                            self.current_epoch = epoch + 1
                            self.current_file_index = index + 1
                            self.current_file = file
                            self.total_file = len(files)
                            self.current_epoch = epoch + 1

                            if sample is None:
                                continue
                            sample.append(mask_word)
                            yield sample

            def batch_reader(reader, batch_size):
                batch, total_token_num, total_expected_token_num = [], 0, 0
                dev_count = 1
                buff = []
                readers = []
                for i in range(len(task_probs)):
                    buff.append(None)
                    readers.append(reader(i))

                # logger.info(f"number of readers: {len(readers)}")

                task_indices = range(len(task_probs))
                # logger.info(f"task_indices: {task_indices}")
                # logger.info(f"task_indices: {task_probs}")
                end_times = 0
                while end_times < 50:
                    # logger.warning(f"random_seed in generator:\t{self.random_seed}")
                    task_index = self.rng.choice(task_indices, p=task_probs)
                    # task_index = np.random.choice(task_indices, p=task_probs)
                    # logger.info(f"choiced task_index: {task_index}")
                    dev_num = 0
                    cur_reader = readers[task_index]

                    while dev_num < dev_count:
                        if buff[task_index] is not None:
                            cur_expected_token_num = get_expected_tokens_number(
                                buff[task_index],
                                task_name=self.task_group[task_index]["task_name"],
                                s_id=self.s_id,
                            )
                            while cur_expected_token_num > batch_size:
                                buff[task_index] = list(
                                    map(
                                        lambda x: x[:-1] if isinstance(x, list) else x,
                                        buff[task_index],
                                    )
                                )
                                cur_expected_token_num = get_expected_tokens_number(
                                    buff[task_index],
                                    task_name=self.task_group[task_index]["task_name"],
                                    s_id=self.s_id,
                                )
                            cur_len = len(buff[task_index][0])
                            batch.append(buff[task_index])
                            total_token_num += cur_len
                            total_expected_token_num += cur_expected_token_num
                            buff[task_index] = None
                            if self.is_debug:
                                print(
                                    "[debug]: buff[task_index] is not None",
                                    "cur_expected_token_num",
                                    cur_expected_token_num,
                                    "total_expected_token_num",
                                    total_expected_token_num,
                                    "cur_len",
                                    cur_len,
                                    "total_token_num",
                                    total_token_num,
                                    "task_name",
                                    self.task_group[task_index]["task_name"],
                                )

                        parsed_line = next(cur_reader, None)
                        if parsed_line is None:
                            end_times += 1
                            dev_num += 1
                            if len(batch) > 0:
                                if self.is_debug:
                                    print(
                                        "[debug]: parsed_line is None",
                                        "total_expected_token_num",
                                        total_expected_token_num,
                                        "total_token_num",
                                        total_token_num,
                                        "task_name",
                                        self.task_group[task_index]["task_name"],
                                    )
                                yield batch, total_token_num, task_index, self.task_group[
                                    task_index
                                ][
                                    "lm_weight"
                                ], self.task_group[
                                    task_index
                                ][
                                    "branch"
                                ]
                                batch, total_token_num, total_expected_token_num = (
                                    [],
                                    0,
                                    0,
                                )
                            continue

                        end_times = 0
                        cur_len = len(parsed_line[0])
                        cur_expected_token_num = get_expected_tokens_number(
                            parsed_line,
                            task_name=self.task_group[task_index]["task_name"],
                            s_id=self.s_id,
                        )
                        if (
                            total_expected_token_num + cur_expected_token_num
                            > batch_size
                        ):
                            if self.is_debug:
                                print(
                                    "debug: before yield",
                                    "cur_expected_token_num",
                                    cur_expected_token_num,
                                    "total_expected_token_num",
                                    total_expected_token_num,
                                    "cur_len",
                                    cur_len,
                                    "total_token_num",
                                    total_token_num,
                                    "task_name",
                                    self.task_group[task_index]["task_name"],
                                )
                            yield batch, total_token_num, task_index, self.task_group[
                                task_index
                            ]["lm_weight"], self.task_group[task_index]["branch"]
                            batch, total_token_num, total_expected_token_num = [], 0, 0
                            dev_num += 1
                            buff[task_index] = parsed_line
                        else:
                            batch.append(parsed_line)
                            total_token_num += cur_len
                            total_expected_token_num += cur_expected_token_num
                            if self.is_debug:
                                print(
                                    "debug: batch.append(parsed_line)",
                                    "cur_expected_token_num",
                                    cur_expected_token_num,
                                    "total_expected_token_num",
                                    total_expected_token_num,
                                    "cur_len",
                                    cur_len,
                                    "total_token_num",
                                    total_token_num,
                                    "task_name",
                                    self.task_group[task_index]["task_name"],
                                )
                    assert len(batch) == 0

            return_list = []
            micro_tmp = []
            for (
                batch_data,
                total_token_num,
                task_index,
                lm_weight,
                branch,
            ) in batch_reader(reader, self.batch_size):
                if self.random_pos_start_set is None:
                    self.random_pos_start_set = [None] * len(batch_data)
                out = prepare_batch_data(
                    batch_data,
                    total_token_num,
                    task_index,
                    lm_weight,
                    len(self.task_group),
                    self.task_group[task_index]["task_name"],
                    random_seed=self.random_seed,
                    branch=branch,
                    max_seq_len=self.max_seq_len,
                    lm_ratio=self.lm_ratio,
                    voc_size=self.voc_size,
                    mem_len=self.mem_len,
                    vocab=self.vocab,
                    prompt_sampler=self.glm_prompt_sampler,
                    pad_id=self.pad_id,
                    cls_id=self.cls_id,
                    sep_id=self.sep_id,
                    mask_id=self.mask_id,
                    g_mask_id_list=[
                        self.glm_mask_id,
                        self.glm_mask_id_s,
                        self.glm_mask_id_c,
                        self.glm_mask_id_p,
                    ],
                    sent_mask_id=self.sent_mask_id,
                    g_end_id=self.glm_end_id,
                    end_id=self.end_id,
                    start_id=self.start_id,
                    s_id=self.s_id,
                    question_id=self.question_id,
                    return_input_mask=True,
                    return_max_len=False,
                    return_num_token=False,
                    random_pos_start_set=self.random_pos_start_set,
                    batch_size_fact=self.batch_size,
                    is_contrast=self.is_contrast,
                    add_ditto=self.add_ditto,
                    QAReader=self.QAReader,
                    training_mode=self.training_mode,
                    task_need_convert=self.task_need_convert,
                )
                num_out_tensor = len(out)
                micro_tmp.append(out)

                # for local_batch
                # if len(micro_tmp) == self.micro_bsz:
                #    tmp = []
                #    for n in range(num_out_tensor):
                #        tmp.append([t[n] for t in micro_tmp])
                #    tmp_concat = [np.concatenate(i, axis=0) for i in tmp]
                #    micro_tmp = []

                #    return_list.append(tmp_concat)
                #    if len(return_list) == self.acc_steps:
                #        return_list_new = []
                #        for n in range(num_out_tensor):
                #            return_list_new.append([t[n] for t in return_list])
                #        return_list = []
                #        yield return_list_new

                # for micro_batch
                if len(micro_tmp) == self.micro_bsz:
                    tmp = []
                    for n in range(num_out_tensor):
                        tmp.append([t[n] for t in micro_tmp])
                    # tmp_concat = [paddle.to_tensor(np.concatenate(i, axis=0)) for i in tmp]
                    tmp_concat = [np.concatenate(i, axis=0) for i in tmp]
                    with open(
                        os.path.join(
                            self.output_dir,
                            f"data_seq.epoch_{self.epoch}_dp_rank_{self.worker_index}_of_{self.dp_worldsize}.pth",
                        ),
                        "ab",
                    ) as wf:
                        pickle.dump(tmp_concat[0], wf)
                    micro_tmp = []
                    yield tmp_concat

        return wrapper


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="main")
    parser.add_argument(
        "-g",
        "--train_task_config",
        type=str,
        default="./conf/streaming_reader_config/train_filelist_0424.json",
    )
    parser.add_argument(
        "-v",
        "--vocab_path",
        type=str,
        default="./conf/streaming_reader_config/ernie_token_100k.model",
    )
    # parser.add_argument("-v", "--tokenizer_name", type=str, default="/root/afs_ro/szth.afs.baidu.com/user/nlp/nlp-sz/ol/chenxuyi/hub/ernie4-v100k-word")
    parser.add_argument("-m", "--max_seq_len", type=str, default=4096)
    parser.add_argument(
        "-d", "--add_ditto", type=str, default=None
    )  # "./conf/ratio_file/ratio-ch40-red40-co19-tr1-exp3-0510"
    parser.add_argument("-t", "--training_mode", type=str, default="AR")
    parser.add_argument(
        "-n",
        "--task_need_convert",
        type=str,
        default="2,3,5,30,31,32,33,34,35,36,37,39,53,62,63,78,79",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=4096)
    parser.add_argument("-r", "--random_seed", type=int, default=43)
    parser.add_argument("-w", "--dp_worldsize", type=int, default=1)
    parser.add_argument("-s", "--voc_size", type=int, default=100224)
    parser.add_argument(
        "-i", "--is_valid", type=lambda x: x.lower() == "true", default=False
    )
    parser.add_argument("--worker_index", type=int, default=0)
    parser.add_argument("--acc_steps", type=int, default=1)

    args = parser.parse_args()

    with open(args.train_task_config) as f:
        train_task_group = json.load(f)

    args.task_group = train_task_group

    dict_args = vars(args)

    del dict_args["train_task_config"]

    for k in dict_args:
        print(k, dict_args[k])

    edr = ErnieDataReader(**dict_args)
    tokenizer = ErnieBotTokenizer.from_pretrained(args.tokenizer_name)
    # import inspect
    # this_args = inspect.getfullargspec(tokenizer.encode).args

    all_pads_len = 0
    for bdx, batch in enumerate(edr.data_generator()()):
        # print(batch[0].shape)
        batch = [np.squeeze(b, 0) for b in [batch[0], batch[4]]]
        pads = np.where(batch[0] == 100000)
        all_pads_len += pads[0].shape[0]
        print(f"Batch id: {bdx}, all_pads_len: {all_pads_len}")
        if bdx > 9999:
            break
        # import pdb; pdb.set_trace()
