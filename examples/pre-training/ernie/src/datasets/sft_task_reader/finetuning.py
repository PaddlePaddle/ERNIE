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

import csv
import random
import re
from collections import defaultdict, namedtuple

import numpy as np
import paddle
import ujson as json
from .data_utils import (
    RandomNoReplacementSampler,
    pad_batch_data,
    sampling_pseudo_examples,
)
from .pvp import EBMarkUpRouter


class BaseReader(object):
    def __init__(
        self,
        task_group,
        is_valid,
        batch_size=1,
        max_seq_len=4096,
        epoch=10,
        random_seed=None,
        tokenizer="ErnieTokenizer",
        is_debug=False,
        dp_worldrank=0,
        dp_worldsize=1,
        number_of_samples_each_epoch=50000,
        pseudo_strategy=0,
        example_from_same_task_prob=0.1,
        pseudo_sampling_prob=0.5,
        trigger_data_prob=0.5,
        add_break_token_multi_turn_for_nontrigger_data=True,
        use_role_embedding=False,
        use_anti_k_sampling=False,
        device="gpu",
        **kwargs,
    ):
        self.task_group = task_group
        self.is_valid = is_valid
        self.batch_size = batch_size
        self.in_tokens = True
        self.max_seq_len = max_seq_len
        self.epoch = epoch
        self.random_seed = random_seed if random_seed is not None else 465
        self.tokenizer = tokenizer
        self.is_debug = is_debug
        # TODO: remove these configs
        self.dp_worldrank = dp_worldrank  # should be dp_index
        self.dp_worldsize = dp_worldsize  # should be dp_num
        self.number_of_samples_each_epoch = number_of_samples_each_epoch
        self.pseudo_strategy = pseudo_strategy
        self.example_from_same_task_prob = example_from_same_task_prob
        self.pseudo_sampling_prob = pseudo_sampling_prob
        self.trigger_data_prob = trigger_data_prob
        self.add_break_token_multi_turn_for_nontrigger_data = (
            add_break_token_multi_turn_for_nontrigger_data
        )
        self.use_role_embedding = use_role_embedding
        self.use_anti_k_sampling = use_anti_k_sampling
        self.place = paddle.set_device(device)

        # setup special tokens
        vocab = self.tokenizer.get_vocab()
        self.start_token = "<s>"
        self.end_token = "</s>"
        self.pad_token = "<pad>"
        if "<cls>" in vocab:
            self.cls_token = "<cls>"
            self.sep_token = "<sep>"
            self.space_token = "<mask:1>"
            self.mask_token = "<mask:0>"
            self.gend_token = "<mask:7>"
        else:
            self.cls_token = "[<unused0>]"
            self.sep_token = "[<unused1>]"
            self.space_token = "[<unused2>]"
            self.mask_token = "[<unused3>]"
            self.gend_token = "[<unused4>]"
        self.break_token = self.sep_token
        self.break_turn_token = self.cls_token
        self.start_id = vocab[self.start_token]
        self.end_id = vocab[self.end_token]  # origin: [END]
        self.pad_id = vocab[self.pad_token]
        self.cls_id = vocab[self.cls_token]
        self.sep_id = vocab[self.sep_token]
        self.space_id = vocab[self.space_token]  # origin: [<S>]
        self.mask_id = vocab[self.mask_token]  # origin: [MASK]
        self.gend_id = vocab[self.gend_token]  # origin: [gEND]

        # setup random seed
        self.rng = random.Random(self.random_seed)
        np.random.seed(self.random_seed)

        # setup markups
        self.eb_markup_rounter = EBMarkUpRouter(
            self.tokenizer, self.break_token, self.break_turn_token
        )
        markups = [
            "search",
            "compute",
            "kg",
            "imagegen",
            "kg-yes",
            "kg-no",
            "image",
            "kg-cs-yes",
            "kg-cs-no",
            "prompt",
        ]
        self.tokenizer.markup_tokens = []
        for markup_token in markups:
            self.tokenizer.markup_tokens.extend(
                [
                    f"[<{markup_token}>]",
                    f"[</{markup_token}>]",
                    f"[<{markup_token}-res>]",
                    f"[</{markup_token}-res>]",
                ]
            )
        self.tokenizer.markup_tokens.extend(
            ["[<citation>]", "[</citation>]", "[<citation-ref>]", "[</citation-ref>]"]
        )

        # setup debug
        self.current_example = 0
        self.source_to_num_opt = defaultdict(int)
        self.current_epoch = 0
        self.num_examples = 0
        self.qid_num = 0
        self.qid_int_to_str = {}
        self.DEBUG_PRINT = 5

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            headers = next(reader)
            Example = namedtuple("Example", headers)

            examples = []
            cnt = 0
            for line in reader:
                try:
                    example = Example(*line)
                except Exception as e:
                    print(line)
                    raise e
                examples.append(example)
                cnt += 1

                if self.is_debug and cnt > 10:
                    break

            return examples

    def _read_jsonl(self, input_file):
        """Reads jsonl file."""
        with open(input_file, "r") as f:
            examples = []
            cnt = 0
            Example = None
            for line_i, line in enumerate(f):
                try:
                    data = json.loads(line)
                except Exception:
                    print(
                        "data error in{}: skipping".format(input_file),
                        "line_i:",
                        line_i,
                        "line:",
                        line,
                    )
                    # raise e
                    continue
                if Example is None:
                    names = ["src", "tgt", "label", "is_memory", "source"]
                    Example = namedtuple("Example", names)
                if isinstance(data["src"], str):
                    data["src"] = [data["src"]]
                if isinstance(data["tgt"], str):
                    data["tgt"] = [data["tgt"]]

                flag = False
                for item in data["tgt"]:
                    if len(item.strip()) == 0:
                        flag = True
                        break
                if flag:
                    print(
                        "data error(empty tgt) in {}: skipping".format(input_file),
                        "line_i:",
                        line_i,
                        "line:",
                        line,
                    )
                    continue
                if len(data["src"]) != len(data["tgt"]):
                    print(
                        "data error(src != tgt) in {}: skipping".format(input_file),
                        "line_i:",
                        line_i,
                        "line:",
                        line,
                    )
                    continue
                if "label" not in data:
                    # label 为 1，表示某一轮都算loss
                    data["label"] = [1] * len(data["src"])
                if "is_memory" not in data:
                    # is_memory 为 1，表示带有memory，不能在该样本前拼接样本
                    data["is_memory"] = 0

                try:
                    example = Example(
                        **{
                            "src": data["src"],
                            "tgt": data["tgt"],
                            "label": data["label"],
                            "is_memory": data["is_memory"],
                            "source": input_file,
                        }
                    )
                except Exception as e:
                    print(line)
                    raise e
                examples.append(example)
                cnt += 1

            return examples

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        markups_poped_a = []
        markups_poped_b = []

        def pop(tokens_list, poped_list):
            while len(tokens_list) > 0:
                poped_token = tokens_list.pop()
                if poped_token in self.tokenizer.markup_tokens:
                    poped_list.append(poped_token)
                else:
                    # 非markup，则退出
                    break

        is_parts_a_truncated, is_parts_b_truncated = False, False
        while True:
            total_length = (
                len(tokens_a)
                + len(tokens_b)
                + len(markups_poped_a)
                + len(markups_poped_b)
            )
            if total_length <= max_length or (len(tokens_a) + len(tokens_b) == 0):
                break
            if len(tokens_a) > len(tokens_b):
                is_parts_a_truncated = True
                pop(tokens_a, markups_poped_a)
            else:
                is_parts_b_truncated = True
                pop(tokens_b, markups_poped_b)

        # 拼接回markup
        if (len(tokens_a) + len(tokens_b)) != 0:
            tokens_a.extend(markups_poped_a)
            tokens_b.extend(markups_poped_b)

        return is_parts_a_truncated, is_parts_b_truncated

    def _convert_example_to_record(self, example, max_seq_length, tokenizer, index):
        """Converts a single `Example` into a single `Record`."""
        raise NotImplementedError

    def _prepare_batch_data(
        self, examples_all, examples_per_task, batch_size, phase=None
    ):
        """generate batch records"""
        batch_records, max_len = [], 0
        cur_len_so_far = 0
        rng = np.random.RandomState(self.random_seed)
        for index, (example, source_to_num_opt) in enumerate(
            sampling_pseudo_examples(
                examples_all,
                examples_per_task,
                self.tokenizer,
                self.eb_markup_rounter,
                rng,
                self.max_seq_len,
                self.pseudo_strategy,
                self.example_from_same_task_prob,
                self.pseudo_sampling_prob,
                self.trigger_data_prob,
                self.use_anti_k_sampling,
            )
        ):
            if phase == "train":
                self.current_example += sum(source_to_num_opt.values())
            for k, v in source_to_num_opt.items():
                self.source_to_num_opt[k] += v
            records = self._convert_example_to_record(
                example, self.max_seq_len, self.tokenizer, index
            )
            for record in records:
                max_len = max(max_len, len(record.token_ids))
                if self.in_tokens:
                    assert (
                        batch_size == 1
                    ), "batch_size is always set to 1 for batch-based iterator"
                    to_append = (
                        cur_len_so_far + len(record.token_ids)
                    ) <= self.max_seq_len
                else:
                    to_append = len(batch_records) < batch_size
                if to_append:
                    batch_records.append(record)
                    cur_len_so_far += len(record.token_ids)
                else:
                    yield self._pad_batch_records(batch_records)
                    batch_records, max_len = [record], len(record.token_ids)
                    cur_len_so_far = len(record.token_ids)
            # print("index", index, '_prepare_batch_data', len(batch_records))
        # print("left batch_records, phase", phase, 'len(batch_records)', len(batch_records))
        if phase != "train" and len(batch_records) > 0:
            while len(batch_records) < batch_size:
                batch_records.append(batch_records[-1])
                print("in while", "len(batch_records)", len(batch_records))
            yield self._pad_batch_records(batch_records)

    def data_generator(self):
        phase = "train" if not self.is_valid else "valid"
        shuffle = True if not self.is_valid else False
        if phase == "train":
            tasks = self.task_group
            total_probs = sum(float(task["prob"]) for task in tasks)
            for task in tasks:
                task["prob"] = float(task["prob"]) / total_probs
                examples = self._read_jsonl(task["filepath"])
                task["examples"] = examples
                task["target_num_each_epoch"] = int(
                    float(task["prob"]) * self.number_of_samples_each_epoch
                )
                task["sampler"] = RandomNoReplacementSampler(
                    task["examples"], self.rng
                ).getter()
                print(
                    task["filepath"],
                    " task probs: ",
                    task["prob"],
                    " ori number of examples:",
                    len(task["examples"]),
                    " target_num_each_epoch:",
                    task["target_num_each_epoch"],
                    " target_num_total_epoch: ",
                    task["target_num_each_epoch"] * self.epoch,
                )

            print("total_probs should be 1, current is ", total_probs)
        else:
            if self.task_group.endswith("jsonl"):
                examples = self._read_jsonl(self.task_group)
            else:
                examples = self._read_tsv(self.task_group)
        print("examples", examples[0])

        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0
        self.qid_num = 0
        self.qid_int_to_str = {}

        def wrapper():
            nonlocal examples
            all_dev_batches = []
            for epoch_index in range(100000 if phase == "train" else self.epoch):
                self.qid_num = 0
                self.current_epoch = epoch_index
                self.random_seed = epoch_index

                self.global_rng = np.random.RandomState(self.random_seed)
                # examples = []

                if phase == "train":
                    examples_per_task = []  # 每个任务分开
                    examples_all = []  # 所有任务混合在一起
                    for task in tasks:
                        examples = []
                        examples.extend(
                            [
                                next(task["sampler"])
                                for _ in range(task["target_num_each_epoch"])
                            ]
                        )

                        if shuffle:
                            self.global_rng.shuffle(examples)
                        examples_per_task.append(examples)
                        examples_all.extend(examples)

                    if shuffle:
                        self.global_rng.shuffle(examples_all)
                        self.global_rng.shuffle(examples_per_task)

                for batch_data in self._prepare_batch_data(
                    examples_all, examples_per_task, self.batch_size, phase=phase
                ):
                    # print('self.qid_num', self.qid_num, 'all_dev_batches', len(all_dev_batches))
                    if len(all_dev_batches) < self.dp_worldsize:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == self.dp_worldsize:
                        yield all_dev_batches[self.dp_worldrank]
                        all_dev_batches = []

            if phase != "train" and len(all_dev_batches) > 0:
                while len(all_dev_batches) < self.dp_worldsize:
                    all_dev_batches.append(all_dev_batches[-1])
                yield all_dev_batches[self.dp_worldrank]

        return wrapper

    def _gen_self_attn_mask_for_glm_flatten(
        self, batch_token_ids, batch_size_fact=None, unbid_idx_1=[], unbid_idx_2=[]
    ):
        assert (
            len(sum(batch_token_ids, [])) <= batch_size_fact
        ), f"{len(sum(batch_token_ids, []))} > {batch_size_fact} is not allowed"

        # Note(gongenlei): unsqueeze attention mask to 4 dims
        input_mask_data = np.zeros(
            (1, 1, batch_size_fact, batch_size_fact), dtype="float32"
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
            input_mask_data[
                0, 0, offset : offset + cur_len, offset : offset + cur_len
            ] = b
            offset += cur_len

        return input_mask_data


class KnowledgeBasedSFTReader(BaseReader):
    def _convert_example_to_record(
        self, example, max_seq_length, tokenizer, index, label_map=None
    ):
        tokens = []
        labels = []
        loss_mask = []
        previous_cur_len = 2  # start_token, break_turn_token
        resever_multi_turn_break_length = 4 if self.break_turn_token == "[<N>]" else 2

        def extract_knowledge(result):
            add_token = ""
            if not self.add_break_token_multi_turn_for_nontrigger_data:
                add_token = self.break_turn_token
            if "kg-" in result:
                for markup in [
                    "[<kg-res>]",
                    "[</kg-res>]",
                    "[<kg-yes>]",
                    "[</kg-yes>]",
                    "[</kg-cs-yes>]",
                    "[</kg-cs-yes>]",
                    "[</kg-cs-no>]",
                    "[</kg-cs-no>]",
                    "[<image>]",
                    "[</image>]",
                ]:
                    result = result.replace(markup, "")
                result = (
                    f"知识库：{result.strip()}\n根据所提供的知识库信息，回答问题并补全对话："
                    + add_token
                )
            elif "search-res" in result:
                result = re.findall(
                    r"\[<search-res>\](.*?)\[<\/search-res>\]", result, re.S | re.M
                )[0]
                result = (
                    f"{result.strip()}\n根据以上参考文章回答问题，补全对话" + add_token
                )
            elif "prompt-res" in result:
                result = re.findall(
                    r"\[<prompt-res>\](.*?)\[<\/prompt-res>\]", result, re.S | re.M
                )[0]
                result = result.strip() + add_token
            elif "compute-res" in result:
                result = re.findall(
                    r"\[<compute-res>\](.*?)\[<\/compute-res>\]", result, re.S | re.M
                )[0]
                result = (
                    f"参考文章1：{result.strip()}\n根据以上参考文章回答问题，补全对话"
                    + add_token
                )
            elif "citation-ref" in result:
                result = re.findall(
                    r"\[<citation-ref>\](.*?)\[<\/citation-ref>\]", result, re.S | re.M
                )[0]
                result = f"请参考搜索结果回答下面问题并使用引用标记来标注回答内容参考的搜索结果序号，例如^[2]^ (引用单个搜索结果）,^[1][2]^（引用多个搜索结果），其中方括号中的数字是搜索结果序号。引用标记只能出现在句尾标点符号前。\n以下是搜索结果（每行开头[1]、[2]、...是搜索结果序号）：\n{result.strip()}\n根据以上搜索结果回答问题并标注引用，补全对话"
            else:
                assert False, result

            return result

        if self.is_debug:
            CONTAINS_SAME_SRC_TGT = False
            for i, (_src, _tgt) in enumerate(zip(example.src, example.tgt)):
                for j in range(len(example.src)):
                    if j == i:
                        continue
                    if _src == example.src[j] and _tgt == example.tgt[j]:
                        CONTAINS_SAME_SRC_TGT = True
                        print("_src", [_src], "_tgt", [_tgt])
                        print("src:", example.src, "tgt:", example.tgt)
                        break
            self.debug_total_pesudo_example += 1
            self.debug_contain_same_tgt += int(CONTAINS_SAME_SRC_TGT)

            print(
                "self.debug_total_pesudo_example:",
                self.debug_total_pesudo_example,
                " self.debug_contain_same_tgt:",
                self.debug_contain_same_tgt,
            )

        turn_index = len(example.src) - 1
        knowledge_tokens = []
        role_embedding_ids = []
        if self.use_role_embedding:
            role_embedding_start_type = (
                4  # MASK START:4 (K:2, prompt:3), [CLS]:1 Q:1, [SEP]:0, A:0,
            )
            role_embedding_knowledge_type = 2
            role_embedding_src_type = 1
            role_embedding_tgt_type = 0
            role_embedding_break_token_type = 0
            role_embedding_break_token_multi_turn_type = 1
        else:
            role_embedding_start_type = 0
            role_embedding_knowledge_type = 0
            role_embedding_src_type = 0
            role_embedding_tgt_type = 0
            role_embedding_break_token_type = 0
            role_embedding_break_token_multi_turn_type = 0

        last_token_label = self.end_token
        while turn_index >= 0:
            src, tgt = example.src[turn_index].strip(), example.tgt[turn_index].strip()
            CONTAINS_MARKUP = False
            for markup in self.tokenizer.markup_tokens:
                if markup in src:
                    # 只用判断src里是不是有markup
                    CONTAINS_MARKUP = True
                    break

            if CONTAINS_MARKUP:
                # 存在markup，则抽取出knowledge, H, B
                if "prompt" in src and self.use_role_embedding:
                    role_embedding_knowledge_type = 3

                knowledge_tokens = tokenizer.tokenize(extract_knowledge(src))
                src = example.src[turn_index - 1]
                previous_cur_len += len(knowledge_tokens)

            tokens_src, tokens_target = tokenizer.tokenize(src), tokenizer.tokenize(tgt)
            is_parts_a_truncated, is_parts_b_truncated = self._truncate_seq_pair(
                tokens_src,
                tokens_target,
                self.max_seq_len - previous_cur_len - resever_multi_turn_break_length,
            )
            tokens_src = tokens_src + [self.break_token]

            if is_parts_b_truncated or is_parts_a_truncated:
                # 被截断，则退出
                break
            role_src = [role_embedding_src_type] * len(tokens_src[:-1]) + [
                role_embedding_break_token_type
            ]
            role_tgt = [role_embedding_tgt_type] * len(tokens_target)

            break_token_multi_turn = []
            if self.break_turn_token == "[<N>]":
                break_token_multi_turn = [self.break_turn_token] * 3
            else:
                break_token_multi_turn = [self.break_turn_token]

            cur_tokens = tokens_src + tokens_target
            cur_role_embedding_ids = role_src + role_tgt

            if is_parts_b_truncated:
                last_token_label = self.pad_token

            tokens = cur_tokens + break_token_multi_turn + tokens
            labels = (
                cur_tokens[1:] + break_token_multi_turn + [last_token_label] + labels
            )
            last_token_label = cur_tokens[0]
            role_embedding_ids = (
                cur_role_embedding_ids
                + [role_embedding_break_token_multi_turn_type]
                * len(break_token_multi_turn)
                + role_embedding_ids
            )

            # 只优化 tgt
            # 根据label判断当前轮需不需要优化loss，数据默认是优化每轮tgt的loss
            loss_mask = (
                [0] * (len(tokens_src) - 1)
                + [example.label[turn_index]] * (len(tokens_target) + 1)
                + [0] * len(break_token_multi_turn)
                + loss_mask
            )

            # if last_token_label != self.end_token:
            #     loss_mask[len(cur_tokens) - 1] = 0
            previous_cur_len += len(cur_tokens) + len(break_token_multi_turn)

            turn_index -= 1
            if CONTAINS_MARKUP:
                turn_index -= 1

        if len(tokens) <= 4:
            return []

        if self.add_break_token_multi_turn_for_nontrigger_data:
            labels = knowledge_tokens + [self.break_turn_token, tokens[0]] + labels
            tokens = (
                [self.start_token] + knowledge_tokens + [self.break_turn_token] + tokens
            )
            loss_mask = [0] * (
                2 + len(knowledge_tokens)
            ) + loss_mask  # 2 for start_token & break_turn_token
            role_embedding_ids = (
                [role_embedding_start_type] * 2
                + len(knowledge_tokens) * [role_embedding_knowledge_type]
                + [role_embedding_break_token_multi_turn_type]
                + role_embedding_ids
            )
        else:
            labels = knowledge_tokens + [tokens[0]] + labels
            tokens = [self.start_token] + knowledge_tokens + tokens
            loss_mask = [0] * (
                1 + len(knowledge_tokens)
            ) + loss_mask  # 1 for start_token
            role_embedding_ids = (
                [role_embedding_start_type] * 2
                + len(knowledge_tokens) * [role_embedding_knowledge_type]
                + role_embedding_ids
            )

        assert len(tokens) <= self.max_seq_len, "{}-{}".format(
            len(tokens), self.max_seq_len
        )

        self.qid_num += 1
        self.current_example += 1
        if "qid" in example._fields:
            self.qid_int_to_str[self.qid_num] = example.qid
        else:
            self.qid_int_to_str[self.qid_num] = str(self.qid_num)

        # ! force setup labels
        labels = tokens[1:] + [self.end_token]

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = tokenizer.convert_tokens_to_ids(labels)
        # mask_token_index = tokens.index(self.mask_token)
        # pos_ids = role_embedding_ids
        # pos_ids_extra = [0] * (mask_token_index + 1) +  list(range(1, len(tokens) - mask_token_index))
        pos_ids = list(range(len(tokens)))
        pos_ids_extra = pos_ids
        assert len(pos_ids) == len(pos_ids_extra)

        Record = namedtuple(
            "Record",
            [
                "token_ids",
                "position_ids",
                "position_ids_extra",
                "qid",
                "label",
                "loss_mask",
            ],
        )

        if sum(loss_mask) == 0:
            print("[BAD CASE] loss_mask all 0", example.src, example.tgt)
            return []

        assert len(token_ids) != 0, f"{self.qid_num}, "

        records = []
        record = Record(
            token_ids=token_ids,
            position_ids=pos_ids,
            position_ids_extra=pos_ids_extra,
            qid=self.qid_num,
            label=label_ids,
            loss_mask=loss_mask,
        )
        records.append(record)

        self.DEBUG_PRINT -= 1
        return records

    def _pad_batch_records(self, batch_records):
        batch_record_token_ids = [
            record.token_ids for record in batch_records
        ]  # leave one token for tgt_ids
        batch_token_ids = [sum(batch_record_token_ids, [])]

        batch_position_ids = [record.position_ids for record in batch_records]
        batch_position_ids = [sum(batch_position_ids, [])]

        batch_position_ids_extra = [
            record.position_ids_extra for record in batch_records
        ]
        batch_position_ids_extra = [sum(batch_position_ids_extra, [])]

        batch_qids = [record.qid for record in batch_records]
        batch_qids = np.array(batch_qids).astype("int64").reshape([-1, 1])

        batch_loss_mask = [record.loss_mask for record in batch_records]
        batch_loss_mask = [sum(batch_loss_mask, [])]

        batch_labels = [record.label for record in batch_records]
        batch_labels = [sum(batch_labels, [])]

        # padding
        padded_token_ids = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            return_input_mask=False,
            max_seq_len=self.max_seq_len,
        )
        # padded_position_ids = pad_batch_data(batch_position_ids, pad_idx=0, max_seq_len=self.max_seq_len)
        padded_position_ids_extra = pad_batch_data(
            batch_position_ids_extra, pad_idx=0, max_seq_len=self.max_seq_len
        )
        padded_batch_loss_mask = pad_batch_data(
            batch_loss_mask, pad_idx=0, max_seq_len=self.max_seq_len
        )
        padded_batch_labels = pad_batch_data(
            batch_labels, pad_idx=self.pad_id, max_seq_len=self.max_seq_len
        )
        # add in-batch mask
        input_mask = self._gen_self_attn_mask_for_glm_flatten(
            batch_record_token_ids, self.max_seq_len
        )
        # input_mask = self._gen_self_attn_mask_for_glm(batch_token_ids, max_seq_len=self.max_seq_len)

        # return_list = [
        #     padded_token_ids, padded_position_ids, padded_position_ids_extra, input_mask, padded_batch_labels, padded_batch_loss_mask
        # ]
        # Note(gongenlei): rm padded_position_ids. padded_position_ids is same as padded_position_ids_extra
        return_list = [
            padded_token_ids,
            padded_position_ids_extra,
            input_mask,
            padded_batch_labels,
            padded_batch_loss_mask,
        ]
        return return_list
