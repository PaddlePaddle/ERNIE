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
Base reader class that provides basic functions such as reading data from files,
"""

import copy
import logging
import os
import random
import re
from collections import defaultdict, namedtuple

import numpy as np
import paddle
import ujson as json
from paddleformers.trainer import TrainerState
from paddleformers.trainer.trainer import TRAINER_STATE_NAME

from .data_utils import RandomNoReplacementSampler, sampling_pseudo_examples
from ernie.dataset.data_utils import pad_batch_data

logger = logging.getLogger(__name__)

Record = namedtuple(
    "Record", ["token_ids", "position_ids", "position_ids_extra", "label", "loss_mask"]
)


class BaseReader:
    """
    Base Reader Class
    """

    def __init__(
        self,
        task_group,
        is_valid,
        batch_size=1,
        in_tokens=False,
        max_seq_len=4096,
        epoch=10,
        random_seed=None,
        tokenizer=None,
        dp_worldrank=0,
        dp_worldsize=1,
        number_of_samples_each_epoch=50000,
        pseudo_strategy=0,
        example_from_same_task_prob=0.1,
        pseudo_sampling_prob=0.5,
        trigger_data_prob=0.5,
        add_break_token_multi_turn_for_nontrigger_data=True,
        use_anti_k_sampling=False,
        device="gpu",
        ignore_load_lr_and_optim=False,
        resume_from_checkpoint="",
        sampling_wo_replacement_data_resuming=False,
        drop_history_with_k=False,
        add_sys_token=False,
        min_shot=2,
        max_shot=8,
        simplify=False,
        use_train_part_sharding=False,
        rope_3d=False,
        **kwargs,
    ):
        self.task_group = copy.deepcopy(task_group)
        self.is_valid = is_valid
        self.batch_size = batch_size
        self.in_tokens = in_tokens
        self.max_seq_len = max_seq_len
        self.epoch = epoch
        self.random_seed = random_seed if random_seed is not None else 465
        self.tokenizer = tokenizer
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
        self.use_anti_k_sampling = use_anti_k_sampling
        self.drop_history_with_k = drop_history_with_k
        self.add_sys_token = add_sys_token
        self.min_shot = min_shot
        self.max_shot = max_shot
        self.simplify = simplify
        self.use_train_part_sharding = use_train_part_sharding
        self.rope_3d = rope_3d
        self.place = paddle.set_device(device)

        # setup special tokens
        vocab = self.tokenizer.get_vocab()
        self.start_token = self.tokenizer.special_tokens_map.get("bos_token", "<s>")
        self.end_token = self.tokenizer.special_tokens_map.get("eos_token", "</s>")
        self.pad_token = self.tokenizer.special_tokens_map.get("pad_token", "<unk>")
        self.cls_token = self.tokenizer.special_tokens_map.get("cls_token", "<mask:0>")
        self.sep_token = self.tokenizer.special_tokens_map.get(
            "sep_token", "<|endofprompt|>"
        )
        self.sys_start_token = self.tokenizer.special_tokens_map.get(
            "sys_start_token", "<mask:4>"
        )
        self.sys_end_token = self.tokenizer.special_tokens_map.get(
            "sys_end_token", "<mask:5>"
        )
        self.header_start_token = self.tokenizer.special_tokens_map.get(
            "header_start_token", "<mask:6>"
        )
        self.header_end_token = self.tokenizer.special_tokens_map.get(
            "header_end_token", "<mask:7>"
        )

        self.break_token = self.sep_token
        self.break_turn_token = self.cls_token
        self.start_id = vocab[self.start_token]
        self.end_id = vocab[self.end_token]  # origin: [END]
        self.pad_id = vocab[self.pad_token]
        self.cls_id = vocab[self.cls_token]
        self.sep_id = vocab[self.sep_token]
        self.sys_start_id = vocab[self.sys_start_token]
        self.sys_end_id = vocab[self.sys_end_token]
        self.header_start_id = vocab[self.header_start_token]
        self.header_end_id = vocab[self.header_end_token]

        self.begin_of_query = self.tokenizer.tokenize("User: ")
        self.begin_of_response = self.tokenizer.tokenize("\nAssistant: ")
        self.end_of_response = "<|end_of_sentence|>"  # "<|endofprompt|>"
        self.begin_token = "<|begin_of_sentence|>"  # "<mask:0>"  ##self.sys_start_token
        self.newline_token = self.tokenizer.tokenize("\n")  ##self.sys_end_token

        # setup random seed
        self.global_rng = random.Random(self.random_seed)
        np.random.seed(self.random_seed)

        # setup markups
        markups = [
            "search",
            "kg",
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
            [
                "[<citation>]",
                "[</citation>]",
                "[<citation-ref>]",
                "[</citation-ref>]",
                "[<retrieve>]",
                "[</retrieve>]",
                "[<retrieve-ref>]",
                "[</retrieve-ref>]",
            ]
        )

        self.current_example = 0
        self.source_to_num_opt = defaultdict(int)
        self.current_epoch = 0

        self.batch_task_id_counter = defaultdict(int)
        self.batch_exact_total_task_id_counter = defaultdict(int)
        state = TrainerState()
        if (
            not ignore_load_lr_and_optim
            and resume_from_checkpoint is not None
            and os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            and os.path.exists(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            and sampling_wo_replacement_data_resuming
        ):
            state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        self.state = {} if state.trial_params is None else state.trial_params

    def _read_jsonl(self, input_file):
        """Reads jsonl file."""
        with open(input_file, "r") as f:
            examples = []
            cnt = 0
            Example = None
            all_lines = []

            if self.use_train_part_sharding:
                for line_i, line in enumerate(f):
                    if line_i % self.dp_worldsize == self.dp_worldrank:
                        all_lines.append(line)
            else:
                for line in f:
                    all_lines.append(line)

            # for line_i, line in enumerate(f):
            for line in all_lines:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if Example is None:
                    names = [
                        "src",
                        "tgt",
                        "label",
                        "disable_pseudo_multi_turn",
                        "is_memory",
                        "is_system",
                        "source",
                        "is_q2code",
                        "math_is_end",
                        "ctxt_src",
                        "ctxt_tgt",
                        "system",
                        "prefix",
                    ]
                    Example = namedtuple("Example", names)

                if isinstance(data["src"], str):
                    data["src"] = [data["src"]]
                if isinstance(data["tgt"], str):
                    data["tgt"] = [data["tgt"]]

                flag = False
                for item in data["tgt"]:
                    if not item or len(item.strip()) == 0:
                        flag = True
                        break
                if flag:
                    continue

                if len(data["src"]) != len(data["tgt"]):
                    continue
                if ("label" in data and not data["label"]) or (
                    "label" in data and len(data["src"]) != len(data["label"])
                ):
                    # print("label error in {}: skipping".format(input_file), "line_i:", line_i, "line:", line)
                    continue
                if "ctxt_src" in data and len(data["src"]) != len(data["ctxt_src"]):
                    print(
                        f"data error(ctxt_src != src) in {input_file}: skipping",
                        "line_i:",
                        line_i,
                        "line:",
                        line,
                    )
                    continue
                if len(data["src"]) == 0:
                    # Some PreSFT data have empty `src` fields, making it unsuitable to
                    # concatenate additional samples before such instances.
                    data["is_system"] = 1
                if "label" not in data:
                    data["label"] = [1] * len(data["src"])

                if "disable_pseudo_multi_turn" not in data:
                    data["disable_pseudo_multi_turn"] = 0

                if "is_memory" not in data:
                    data["is_memory"] = 0

                if "is_system" not in data:
                    data["is_system"] = 0

                if "is_q2code" not in data:
                    data["is_q2code"] = 0

                if "math_is_end" not in data:
                    data["math_is_end"] = 2

                if "ctxt_src" not in data:
                    data["ctxt_src"] = []

                if "ctxt_tgt" not in data:
                    data["ctxt_tgt"] = []

                if "system" not in data:
                    data["system"] = ""
                else:
                    if self.add_sys_token:
                        if data["is_system"] == 1:
                            data["system"] = data["src"][0]
                            data["src"] = data["src"][1:]
                            data["tgt"] = data["tgt"][1:]
                            data["label"] = data["label"][1:]
                            if data["ctxt_src"] != 0:
                                data["ctxt_src"] = data["ctxt_src"][1:]
                            if data["ctxt_tgt"] != 0:
                                data["ctxt_tgt"] = data["ctxt_tgt"][1:]

                    if self.add_sys_token:
                        if data["system"] != "":
                            data["disable_pseudo_multi_turn"] = 1

                    else:
                        if (
                            data["system"] != ""
                            and data["tgt"][0] != "好的，我将遵守您上面的系统设定。"
                        ):
                            data["src"] = [data["system"]] + data["src"]
                            data["tgt"] = ["好的，我将遵守您上面的系统设定。"] + data[
                                "tgt"
                            ]
                            data["label"] = [0] + data["label"]

                try:
                    if len(data["tgt"]) > 0:
                        last_tgt = data["tgt"][-1]

                        if "<think>" in last_tgt and "</think>" in last_tgt:
                            data["prefix"] = ""
                        else:
                            data["prefix"] = "<think>\n\n</think>\n\n"
                            data["label"] = [0] * len(data["tgt"])
                            data["label"][-1] = 1
                    else:
                        data["prefix"] = ""

                    example = Example(
                        **{
                            "src": data["src"],
                            "tgt": data["tgt"],
                            "label": data["label"],
                            "disable_pseudo_multi_turn": data[
                                "disable_pseudo_multi_turn"
                            ],
                            "is_memory": data["is_memory"],
                            "is_system": data["is_system"],
                            "source": input_file,
                            "is_q2code": data["is_q2code"],
                            "math_is_end": data["math_is_end"],
                            "ctxt_src": data["ctxt_src"],
                            "ctxt_tgt": data["ctxt_tgt"],
                            "system": data["system"],
                            "prefix": data["prefix"],
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

        if (len(tokens_a) + len(tokens_b)) != 0:
            tokens_a.extend(markups_poped_a)
            tokens_b.extend(markups_poped_b)

        return is_parts_a_truncated, is_parts_b_truncated

    def _convert_example_to_record(self, example, max_seq_length, tokenizer, index):
        """Converts a single `Example` into a single `Record`."""
        raise NotImplementedError

    def _prepare_batch_data(
        self,
        tasks,
        weighted_task_indices,
        sample_from_same_source_flags,
        batch_size,
        phase=None,
    ):
        """generate batch records"""
        batch_records, max_len = [], 0
        cur_len_so_far = 0
        for index, (
            example,
            source_to_num_opt,
            task_id_counter,
            exact_total_task_id_counter,
        ) in enumerate(
            sampling_pseudo_examples(
                tasks,
                weighted_task_indices,
                sample_from_same_source_flags,
                self.tokenizer,
                self.global_rng,
                self.max_seq_len,
                self.pseudo_strategy,
                self.pseudo_sampling_prob,
                self.trigger_data_prob,
                self.use_anti_k_sampling,
                self.drop_history_with_k,
                self.use_train_part_sharding,
                self.dp_worldsize,
                self.dp_worldrank,
            )
        ):
            if phase == "train":
                self.current_example += sum(source_to_num_opt.values())
            for k, v in source_to_num_opt.items():
                self.source_to_num_opt[k] += v

            records = self._convert_example_to_record(
                example, self.max_seq_len, self.tokenizer, index
            )
            if len(records) == 0:

                for k, v in task_id_counter.items():
                    self.batch_task_id_counter[k] += v
                for k, v in exact_total_task_id_counter.items():
                    self.batch_exact_total_task_id_counter[k] += v

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
                    yield self._pad_batch_records(batch_records, self.simplify)
                    self.batch_task_id_counter = defaultdict(int)
                    self.batch_exact_total_task_id_counter = defaultdict(int)
                    batch_records, max_len = [record], len(record.token_ids)
                    cur_len_so_far = len(record.token_ids)

                for k, v in task_id_counter.items():
                    self.batch_task_id_counter[k] += v
                for k, v in exact_total_task_id_counter.items():
                    self.batch_exact_total_task_id_counter[k] += v
                task_id_counter = defaultdict(int)
                exact_total_task_id_counter = defaultdict(int)

        if phase != "train" and len(batch_records) > 0:
            while len(batch_records) < batch_size:
                batch_records.append(batch_records[-1])
                print("in while", "len(batch_records)", len(batch_records))
            yield self._pad_batch_records(batch_records, self.simplify)

    def data_generator(self):
        """
        Method to generate data.

        Args:
            None

        Returns:
            A generator that returns a batch of data each time it is called.
        """
        phase = "train" if not self.is_valid else "valid"
        shuffle = True if not self.is_valid else False
        total_data_num_each_epoch = 0
        if phase == "train":
            tasks = self.task_group
            tasks = [task for task in tasks if task["prob"] > 0]
            self.num_tasks = (
                len(tasks) * self.dp_worldsize
                if self.use_train_part_sharding
                else len(tasks)
            )
            total_probs = sum(float(task["prob"]) for task in tasks)

            # reset the data status when the number of tasks is different
            if len(self.state.get("saved_task_ids", [])) != self.num_tasks:
                self.state = {}

            for task_id, task in enumerate(tasks):
                task_id = (
                    task_id * self.dp_worldsize + self.dp_worldrank
                    if self.use_train_part_sharding
                    else task_id
                )
                task["task_id"] = task_id
                task["prob"] = float(task["prob"]) / total_probs
                examples = self._read_jsonl(task["filepath"])
                task["target_num_each_epoch"] = int(
                    float(task["prob"]) * self.number_of_samples_each_epoch
                )
                total_data_num_each_epoch += task["target_num_each_epoch"]

                task["total_num_examples"] = len(examples)

                consumed_data = self.state.get("saved_task_ids", [])
                consumed_data = (
                    consumed_data[task["task_id"]] if len(consumed_data) > 0 else 0
                )
                print(f"task_id: {task['task_id']}: {consumed_data}")

                task_sampler = RandomNoReplacementSampler(
                    examples, task["task_id"], self.random_seed
                )
                task_sampler.set_data_status(consumed_data)
                task["task_sampler"] = task_sampler
                task["sampler"] = task_sampler.getter()

                print(
                    task["filepath"],
                    " task probs: ",
                    task["prob"],
                    " ori number of examples:",
                    task["total_num_examples"],
                    " target_num_each_epoch:",
                    task["target_num_each_epoch"],
                    " target_num_total_epoch: ",
                    task["target_num_each_epoch"] * self.epoch,
                    f"sampler start from epoch:{task_sampler.epoch} offset:{task_sampler.offset}",
                )

            print("total_probs should be 1, current is ", total_probs)
        else:
            examples = self._read_jsonl(self.task_group)
        print("examples", examples[0])

        self.current_example = 0
        self.current_epoch = 0

        def wrapper():
            all_dev_batches = []
            consumed_data_num = sum(self.state.get("saved_task_ids", []))
            init_epoch = consumed_data_num // total_data_num_each_epoch
            offset = consumed_data_num % total_data_num_each_epoch
            print(f"data generator resuming from epoch:{init_epoch}, offset{offset}")
            for epoch_index in range(
                init_epoch, 100000 if phase == "train" else self.epoch
            ):
                self.current_epoch = epoch_index
                if phase == "train":
                    weighted_task_indices = []  # weighted task_ids
                    sample_from_same_source_flags = []

                    if shuffle:
                        rng = np.random.RandomState(self.random_seed + epoch_index)

                    for task in tasks:
                        same_source_num = int(
                            task["target_num_each_epoch"]
                            * self.example_from_same_task_prob
                        )
                        non_same_source_num = (
                            task["target_num_each_epoch"] - same_source_num
                        )

                        task_indices = [[task["task_id"]]] * non_same_source_num

                        idx = 0
                        while idx < same_source_num:
                            n_shot = rng.randint(self.min_shot, self.max_shot + 1)
                            if idx + n_shot > same_source_num:
                                n_shot = same_source_num - idx

                            task_indices.append([task["task_id"]] * n_shot)
                            idx += n_shot

                        weighted_task_indices.extend(task_indices)

                    if shuffle:
                        rng.shuffle(weighted_task_indices)
                    # print(f"weighted_task_indices: {weighted_task_indices[:30]}")

                    sample_from_same_source_flags = []
                    flatten_weighted_task_indices = []
                    for item in weighted_task_indices:
                        sample_from_same_source_flags.extend(
                            [int(len(item) > 1)] * len(item)
                        )
                        flatten_weighted_task_indices.extend(item)

                    weighted_task_indices = flatten_weighted_task_indices
                    assert len(weighted_task_indices) == len(
                        sample_from_same_source_flags
                    ), "采样列表应该具有相同源的条目数量"

                    if epoch_index == init_epoch:
                        weighted_task_indices = weighted_task_indices[offset:]
                        sample_from_same_source_flags = sample_from_same_source_flags[
                            offset:
                        ]

                num_batch_to_yield = self.dp_worldsize
                rank_to_yield = self.dp_worldrank
                if self.use_train_part_sharding:
                    num_batch_to_yield = 1
                    rank_to_yield = 0
                for batch_data in self._prepare_batch_data(
                    tasks,
                    weighted_task_indices,
                    sample_from_same_source_flags,
                    self.batch_size,
                    phase=phase,
                ):
                    if len(all_dev_batches) < num_batch_to_yield:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == num_batch_to_yield:
                        yield all_dev_batches[rank_to_yield]
                        all_dev_batches = []

        return wrapper

    def _gen_self_attn_mask_for_glm_flatten(
        self, batch_token_ids, batch_size_fact=None, unbid_idx_1=None, unbid_idx_2=None
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
                unbid_idx_1 is not None and unbid_idx_2 is not None
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
    """
    Knowledge Based SFT Reader
    """

    def _convert_example_to_record(self, example, max_seq_length, tokenizer, index):
        tokens = []
        labels = []
        loss_mask = []
        previous_cur_len = 2  # start_token, break_turn_token
        resever_multi_turn_break_length = 8

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
                    r"\[<search-res>\](.*?)\[<\/search-res>\]",
                    result,
                    re.DOTALL | re.MULTILINE,
                )[0]
                result = (
                    f"{result.strip()}\n根据以上参考文章回答问题，补全对话" + add_token
                )
            elif "prompt-res" in result:
                result = re.findall(
                    r"\[<prompt-res>\](.*?)\[<\/prompt-res>\]",
                    result,
                    re.DOTALL | re.MULTILINE,
                )[0]
                result = result.strip() + add_token
            elif "compute-res" in result:
                result = re.findall(
                    r"\[<compute-res>\](.*?)\[<\/compute-res>\]",
                    result,
                    re.DOTALL | re.MULTILINE,
                )[0]
                result = (
                    f"参考文章1：{result.strip()}\n根据以上参考文章回答问题，补全对话"
                    + add_token
                )
            elif "citation-ref" in result:
                result = re.findall(
                    r"\[<citation-ref>\](.*?)\[<\/citation-ref>\]",
                    result,
                    re.DOTALL | re.MULTILINE,
                )[0]
                result = (
                    f"""请参考搜索结果回答下面问题并使用引用标记来标注回答内容参考的搜索结果序号，
                    例如^[1]^ (引用单个搜索结果）,^[1][2]^（引用多个搜索结果），其中方括号中的数字是搜索结果序号。
                    引用标记只能出现在句尾标点符号前。
                    \n以下是搜索结果（每行开头[1]、[2]、...是搜索结果序号），
                    可以对答案中的核心部分进行markdown加粗（**加粗内容**）：
                    \n{result.strip()}\n根据以上搜索结果回答问题并标注引用，补全对话"""
                    + add_token
                )
            elif "retrieve-ref" in result:
                result = re.findall(
                    r"\[<retrieve-ref>\](.*?)\[<\/retrieve-ref>\]",
                    result,
                    re.DOTALL | re.MULTILINE,
                )[0]
                result = (
                    f"""请你扮演一个专家，参考搜索结果中正确、可信、高质量的信息回答问题，
                    并注明答案中引用的搜索结果，格式为^[2]^表示引用了第2条搜索结果，
                    ^[1][3]^表示引用第1和第3条搜索结果。每条搜索结果包含若干相关内容片段。同时你需要遵循以下原则回答问题：
                    \n1. 严格遵循搜索结果作答，可以承认不知道答案，并尝试给出一些搜索结果中的相关背景信息。
                    \n2. 如果搜索结果存在多种可能的答案，要罗列出每种情况。
                    \n3. 如果问题涉及金融、医疗、法律等存在风险的领域，请在结尾提醒用户注意并进行免责说明。
                    \n搜索结果：\n{result.strip()}\n\n现在，请根据上面的搜索结果回答问题并标注引用，补全对话"""
                    + add_token
                )
            else:
                assert False, result

            result += "\n"
            return result

        if self.add_sys_token:
            system_info = example.system
            system_tokens = (
                [self.begin_token]
                + tokenizer.tokenize(system_info)
                + self.newline_token
            )
            previous_cur_len += len(system_tokens)

        turn_index = len(example.src) - 1
        knowledge_tokens = []
        max_turn_index = turn_index
        while turn_index >= 0:
            src, tgt = (
                example.src[turn_index].strip(),
                example.tgt[turn_index].strip(),
            )
            tokens_src, tokens_target = tokenizer.tokenize(src), tokenizer.tokenize(tgt)

            if turn_index == len(example.src) - 1:
                tokens_src = knowledge_tokens + tokens_src
            tokens_src = self.begin_of_query + tokens_src

            is_parts_a_truncated, is_parts_b_truncated = self._truncate_seq_pair(
                tokens_src,
                tokens_target,
                self.max_seq_len
                + 1
                - previous_cur_len
                - resever_multi_turn_break_length,
            )
            if is_parts_b_truncated or is_parts_a_truncated:

                break

            tokens_src = tokens_src + self.begin_of_response
            break_token_multi_turn = [self.end_of_response]

            if turn_index == max_turn_index and example.prefix:
                prefix_token = tokenizer.tokenize(example.prefix)
                cur_tokens = tokens_src + prefix_token + tokens_target
                extra_loss_mask = [0] * len(prefix_token)
            else:
                cur_tokens = tokens_src + tokens_target
                extra_loss_mask = []
            tokens = cur_tokens + break_token_multi_turn + tokens

            loss_mask = (
                [0] * (len(tokens_src) - 1)
                + extra_loss_mask
                + [example.label[turn_index]] * (len(tokens_target) + 1)
                + [0] * len(break_token_multi_turn)
                + loss_mask
            )
            assert len(tokens) == len(loss_mask), f"{len(tokens)}-{len(loss_mask)}"

            previous_cur_len += len(cur_tokens) + len(break_token_multi_turn)

            turn_index -= 1

        if len(tokens) <= 4:
            return []

        if self.add_sys_token:
            if system_info and turn_index == -1:
                tokens = system_tokens + tokens
                loss_mask = [0] * (len(system_tokens)) + loss_mask
                assert len(tokens) == len(loss_mask), f"{len(tokens)}-{len(loss_mask)}"

        if tokens[0] != self.begin_token:
            tokens = [self.begin_token] + tokens
            loss_mask = [0] + loss_mask

        assert len(tokens) <= self.max_seq_len, f"{len(tokens)}-{self.max_seq_len}"
        assert (
            len(loss_mask) <= self.max_seq_len
        ), f"{len(loss_mask)}-{self.max_seq_len}"

        self.current_example += 1

        # ! force setup labels
        del tokens[-1]  # del last cls_token, there is no </s> in the last position
        del loss_mask[-1]
        labels = tokens[1:] + [self.end_token]

        # let the last token of result to predict </s>
        labels = [
            label if label != self.end_of_response else self.end_token
            for label in labels
        ]

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = tokenizer.convert_tokens_to_ids(labels)

        if self.rope_3d:
            pos_ids = np.array([[i] * 3 for i in range(len(tokens))])

        pos_ids_extra = pos_ids
        assert len(pos_ids) == len(pos_ids_extra)

        if sum(loss_mask) == 0:
            print("[BAD CASE] loss_mask all 0", example.src, example.tgt)
            return []

        records = []
        record = Record(
            token_ids=token_ids,
            position_ids=pos_ids,
            position_ids_extra=pos_ids_extra,
            label=label_ids,
            loss_mask=loss_mask,
        )
        records.append(record)

        return records

    def _pad_batch_records(self, batch_records, simplify=False):
        """
        simplify
        """
        batch_record_token_ids = [
            record.token_ids for record in batch_records
        ]  # leave one token for tgt_ids
        batch_token_ids = [sum(batch_record_token_ids, [])]

        if not self.rope_3d:
            batch_position_ids = [record.position_ids for record in batch_records]
            batch_position_ids = [sum(batch_position_ids, [])]
            batch_position_ids_extra = [
                record.position_ids_extra for record in batch_records
            ]
            batch_position_ids_extra = [sum(batch_position_ids_extra, [])]
        else:
            batch_position_ids = [
                np.array(record.position_ids) for record in batch_records
            ]
            batch_position_ids = np.concatenate(batch_position_ids)
            batch_position_ids_extra = [
                np.array(record.position_ids_extra) for record in batch_records
            ]
            batch_position_ids_extra = np.concatenate(batch_position_ids_extra)

        batch_loss_mask = [record.loss_mask for record in batch_records]
        batch_loss_mask = [sum(batch_loss_mask, [])]

        batch_labels = [record.label for record in batch_records]
        batch_labels = [sum(batch_labels, [])]

        batch_task_id_counter = self.batch_task_id_counter
        batch_exact_total_task_id_counter = self.batch_exact_total_task_id_counter

        max_task_id = self.num_tasks - 1
        max_exact_total_task_id = self.num_tasks - 1

        task_ids = [0] * (max_task_id + 1)
        exact_total_task_ids = [0] * (max_exact_total_task_id + 1)

        for task_id, consumed_cnt in batch_task_id_counter.items():
            task_ids[task_id] = consumed_cnt
        for task_id, consumed_cnt in batch_exact_total_task_id_counter.items():
            exact_total_task_ids[task_id] = consumed_cnt

        batch_task_ids = [task_ids]
        batch_exact_total_task_ids = [exact_total_task_ids]

        ##############################
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

        # padding
        if self.rope_3d:
            padded_position_ids_extra = pad_sequence(
                np.array([batch_position_ids_extra]),
                padding_value=[0, 0, 0],
                fix_len=self.max_seq_len,
            )
        else:
            padded_position_ids_extra = pad_batch_data(
                batch_position_ids_extra, pad_idx=0, max_seq_len=self.max_seq_len
            )

        padded_token_ids = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            return_input_mask=False,
            max_seq_len=self.max_seq_len,
        )
        # padded_position_ids = pad_batch_data(batch_position_ids, pad_idx=0, max_seq_len=self.max_seq_len)

        padded_batch_loss_mask = pad_batch_data(
            batch_loss_mask, pad_idx=0, max_seq_len=self.max_seq_len
        )
        padded_batch_labels = pad_batch_data(
            batch_labels, pad_idx=self.pad_id, max_seq_len=self.max_seq_len
        )
        # add in-batch mask
        if not simplify:
            input_mask = self._gen_self_attn_mask_for_glm_flatten(
                batch_record_token_ids, self.max_seq_len
            )

        padded_batch_task_ids = pad_batch_data(
            batch_task_ids, pad_idx=0, max_seq_len=self.num_tasks
        )
        padded_batch_exact_total_task_ids = pad_batch_data(
            batch_exact_total_task_ids, pad_idx=0, max_seq_len=self.num_tasks
        )

        inbatch_pack_offset = [0]
        for item in batch_record_token_ids:
            inbatch_pack_offset.append(inbatch_pack_offset[-1] + len(item))
        inbatch_pack_offset[-1] = (
            self.max_seq_len
        )  # include padding in the last interval
        padded_inbatch_pack_offset = np.reshape(
            np.array(
                inbatch_pack_offset
                + [-1] * (self.max_seq_len + 1 - len(inbatch_pack_offset)),
                dtype=np.int64,
            ),
            [1, -1],
        )
        # Note(gongenlei): rm padded_position_ids. padded_position_ids is same as padded_position_ids_extra
        if not simplify:
            return_list = [
                padded_token_ids,
                padded_position_ids_extra,
                input_mask,
                padded_inbatch_pack_offset,
                padded_batch_labels,
                padded_batch_loss_mask,
                padded_batch_task_ids,
                padded_batch_exact_total_task_ids,
            ]
        else:
            return_list = [
                padded_token_ids.astype("int64"),
                padded_position_ids_extra.astype("int64"),
                padded_inbatch_pack_offset.astype("int64"),
                padded_batch_labels.astype("int64"),
                padded_batch_loss_mask.astype("bool"),
                padded_batch_exact_total_task_ids.astype("int64"),
            ]
        return return_list
