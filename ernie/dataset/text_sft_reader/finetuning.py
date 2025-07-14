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
from ernie.dataset.data_utils import contains_markup, pad_batch_data
from ernie.dataset.pvp import EBMarkUpRouter

logger = logging.getLogger(__name__)

Record = namedtuple("Record", ["token_ids", "position_ids", "position_ids_extra", "label", "loss_mask"])


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
        ignore_load_lr_and_optim=False,
        resume_from_checkpoint="",
        sampling_wo_replacement_data_resuming=False,
        drop_history_with_k=False,
        add_sys_token=False,
        add_header_eos_token=False,
        min_shot=2,
        max_shot=8,
        simplify=False,
        h5_output_dir=None,
        add_start_token=False,
        knowledge_after_history=False,
        neihua=False,
        all_use_skip_align=False,
        skip_align_s_policy="default",
        skip_align_target_length=100000,
        skip_align_sub_sample_prob=0.5,
        skip_align_method="skip_outter",
        follow_deepseek=False,
        use_agent_sys2=False,
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
        self.is_debug = is_debug
        self.dp_worldrank = dp_worldrank  # should be dp_index
        self.dp_worldsize = dp_worldsize  # should be dp_num
        self.number_of_samples_each_epoch = number_of_samples_each_epoch
        self.pseudo_strategy = pseudo_strategy
        self.example_from_same_task_prob = example_from_same_task_prob
        self.pseudo_sampling_prob = pseudo_sampling_prob
        self.trigger_data_prob = trigger_data_prob
        self.add_break_token_multi_turn_for_nontrigger_data = add_break_token_multi_turn_for_nontrigger_data
        self.use_role_embedding = use_role_embedding
        self.use_anti_k_sampling = use_anti_k_sampling
        self.drop_history_with_k = drop_history_with_k
        self.add_sys_token = add_sys_token
        self.add_header_eos_token = add_header_eos_token
        self.min_shot = min_shot  
        self.max_shot = max_shot  
        self.simplify = simplify  
        self.h5_output_dir = h5_output_dir  
        self.add_start_token = add_start_token 
        self.knowledge_after_history = knowledge_after_history
        self.neihua = neihua
        self.all_use_skip_align = all_use_skip_align
        self.skip_align_s_policy = skip_align_s_policy
        self.skip_align_target_length = skip_align_target_length
        self.skip_align_sub_sample_prob = skip_align_sub_sample_prob
        self.skip_align_method = skip_align_method
        self.use_train_part_sharding = use_train_part_sharding
        self.rope_3d = rope_3d
        self.place = paddle.set_device(device)

        # setup special tokens
        vocab = self.tokenizer.get_vocab()
        self.start_token = self.tokenizer.special_tokens_map.get("bos_token", "<s>")
        self.end_token = self.tokenizer.special_tokens_map.get("eos_token", "</s>")
        self.pad_token = self.tokenizer.special_tokens_map.get("pad_token", "<unk>")
        self.cls_token = self.tokenizer.special_tokens_map.get("cls_token", "<mask:0>")
        self.sep_token = self.tokenizer.special_tokens_map.get("sep_token", "<|endofprompt|>")
        self.sys_start_token = self.tokenizer.special_tokens_map.get("sys_start_token", "<mask:4>")
        self.sys_end_token = self.tokenizer.special_tokens_map.get("sys_end_token", "<mask:5>")
        self.header_start_token = self.tokenizer.special_tokens_map.get("header_start_token", "<mask:6>")
        self.header_end_token = self.tokenizer.special_tokens_map.get("header_end_token", "<mask:7>")

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
        self.end_of_response = "<|endofprompt|>"
        self.begin_token = "<mask:0>"  ##self.sys_start_token
        self.newline_token = self.tokenizer.tokenize("\n")  ##self.sys_end_token
        self.follow_deepseek = follow_deepseek

        self.use_agent_sys2 = use_agent_sys2
        if self.use_agent_sys2:  
            self.use_agent_sys2_print = 60
            self.follow_deepseek = False
            self.add_sys_token = False
            self.add_header_eos_token = False
            self.knowledge_after_history = False
            self.neihua = False
            print(f">>> use_agent_sys2:{self.use_agent_sys2};")
            print(f">>> follow_deepseek:{self.follow_deepseek};")
            print(f">>> add_sys_token:{self.add_sys_token};")
            print(f">>> add_header_eos_token:{self.add_header_eos_token};")
            print(f">>> knowledge_after_history:{self.knowledge_after_history};")
            print(f">>> neihua:{self.neihua}")
        ###debug###
        # self.add_sys_token = True
        # self.all_use_skip_align = True
        # self.skip_align_method = "skip_inner"
        # self.follow_deepseek = True

        # setup random seed
        self.global_rng = random.Random(self.random_seed)
        np.random.seed(self.random_seed)

        # setup markups
        self.eb_markup_rounter = EBMarkUpRouter(self.tokenizer, self.break_token, self.break_turn_token)
        markups = [
            "search",
            "kg",
            "prompt",
        ]
        self.tokenizer.markup_tokens = []
        for markup_token in markups:
            self.tokenizer.markup_tokens.extend(
                [f"[<{markup_token}>]", f"[</{markup_token}>]", f"[<{markup_token}-res>]", f"[</{markup_token}-res>]"]
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

        # setup debug
        self.current_example = 0
        self.source_to_num_opt = defaultdict(int)
        self.current_epoch = 0
        self.DEBUG_PRINT = 5

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
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
        self.state = {} if state.trial_params is None else state.trial_params
        self.debug_count = 0

    def _read_jsonl(self, input_file):
        """Reads jsonl file."""
        with open(input_file, "r") as f:
            examples = []
            cnt = 0
            Example = None
            all_lines = []
            for line_i, line in enumerate(f):
                all_lines.append(line)

            if self.use_train_part_sharding:
                all_lines = all_lines[self.dp_worldrank :: self.dp_worldsize]

            # for line_i, line in enumerate(f):
            for line in all_lines:
                try:
                    data = json.loads(line)
                except:
                    # print("data error in{}: skipping".format(input_file), "line_i:", line_i, "line:", line)
                    # raise e
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
                if contains_markup(data['src'], self.tokenizer.markup_tokens) and len(data['src']) < 2:
                    continue
                if 'ctxt_src' in data and len(data['src']) != len(data['ctxt_src']):
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

           
                if self.use_agent_sys2:  
                    data["disable_pseudo_multi_turn"] = 1  

                else:
                    if self.add_sys_token or self.add_header_eos_token:
                        if data["is_system"] == 1:
                            data["system"] = data["src"][0]
                            data["src"] = data["src"][1:]
                            data["tgt"] = data["tgt"][1:]
                            data["label"] = data["label"][1:]
                            if data["ctxt_src"] != 0:
                                data["ctxt_src"] = data["ctxt_src"][1:]
                            if data["ctxt_tgt"] != 0:
                                data["ctxt_tgt"] = data["ctxt_tgt"][1:]

                    if self.add_sys_token or self.add_header_eos_token:
                        if data["system"] != "":
                            data["disable_pseudo_multi_turn"] = 1  
                            
                    else:
                        if (
                            data["system"] != "" and data["tgt"][0] != "好的，我将遵守您上面的系统设定。"
                        ):  
                            data["src"] = [data["system"]] + data["src"]
                            data["tgt"] = ["好的，我将遵守您上面的系统设定。"] + data["tgt"]
                            data["label"] = [0] + data["label"]

                   
                    if self.neihua:
                        if (
                            "<citation-ref>" in data['src'][-1]
                            or "<retrieve-ref>" in data['src'][-1]
                            or "<kg-" in data['src'][-1]
                        ):
                            continue
                        if "<prompt-res>" in data['src'][-1]:
                            data["src"] = data["src"][:-1]
                            data["tgt"] = data["tgt"][:-2] + [data["tgt"][-1]]

                try:
                    example = Example(
                        **{
                            "src": data["src"],
                            "tgt": data["tgt"],
                            "label": data["label"],
                            "disable_pseudo_multi_turn": data["disable_pseudo_multi_turn"],
                            "is_memory": data["is_memory"],
                            "is_system": data["is_system"],
                            "source": input_file,
                            "is_q2code": data["is_q2code"],
                            "math_is_end": data["math_is_end"],
                            "ctxt_src": data["ctxt_src"],
                            "ctxt_tgt": data["ctxt_tgt"],
                            "system": data["system"],
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
            total_length = len(tokens_a) + len(tokens_b) + len(markups_poped_a) + len(markups_poped_b)
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

    def _prepare_batch_data(self, tasks, weighted_task_indices, sample_from_same_source_flags, batch_size, phase=None):
        """generate batch records"""
        batch_records, max_len = [], 0
        cur_len_so_far = 0
        for index, (example, source_to_num_opt, task_id_counter, exact_total_task_id_counter) in enumerate(
            sampling_pseudo_examples(
                tasks,
                weighted_task_indices,
                sample_from_same_source_flags,
                self.tokenizer,
                self.eb_markup_rounter,
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

            records = self._convert_example_to_record(example, self.max_seq_len, self.tokenizer, index)
            if len(records) == 0:
                
                for k, v in task_id_counter.items():
                    self.batch_task_id_counter[k] += v
                for k, v in exact_total_task_id_counter.items():
                    self.batch_exact_total_task_id_counter[k] += v

            for record in records:
                max_len = max(max_len, len(record.token_ids))
                if self.in_tokens:
                    assert batch_size == 1, "batch_size is always set to 1 for batch-based iterator"
                    to_append = (cur_len_so_far + len(record.token_ids)) <= self.max_seq_len
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
            self.num_tasks = len(tasks) * self.dp_worldsize if self.use_train_part_sharding else len(tasks)
            total_probs = sum(float(task["prob"]) for task in tasks)

            # reset the data status when the number of tasks is different
            if len(self.state.get("saved_task_ids", [])) != self.num_tasks:
                self.state = {}

            for task_id, task in enumerate(tasks):
                task_id = task_id * self.dp_worldsize + self.dp_worldrank if self.use_train_part_sharding else task_id
                task["task_id"] = task_id
                task["prob"] = float(task["prob"]) / total_probs
                examples = self._read_jsonl(task["filepath"])
                task["target_num_each_epoch"] = int(float(task["prob"]) * self.number_of_samples_each_epoch)
                total_data_num_each_epoch += task["target_num_each_epoch"]

                task["total_num_examples"] = len(examples)

                consumed_data = self.state.get("saved_task_ids", [])
                consumed_data = consumed_data[task["task_id"]] if len(consumed_data) > 0 else 0
                print(f"task_id: {task['task_id']}: {consumed_data}")

                task_sampler = RandomNoReplacementSampler(examples, task["task_id"], self.random_seed)
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
            for epoch_index in range(init_epoch, 100000 if phase == "train" else self.epoch):
                self.current_epoch = epoch_index
                if phase == "train":
                    weighted_task_indices = []  # weighted task_ids
                    sample_from_same_source_flags = [] 

                    if shuffle:
                        rng = np.random.RandomState(self.random_seed + epoch_index)

                    for task in tasks:
                        same_source_num = int(task["target_num_each_epoch"] * self.example_from_same_task_prob)
                        non_same_source_num = task["target_num_each_epoch"] - same_source_num

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
                        sample_from_same_source_flags.extend([int(len(item) > 1)] * len(item))
                        flatten_weighted_task_indices.extend(item)

                    weighted_task_indices = flatten_weighted_task_indices
                    assert len(weighted_task_indices) == len(
                        sample_from_same_source_flags
                    ), "采样列表应该具有相同源的条目数量"

                    if epoch_index == init_epoch:
                        weighted_task_indices = weighted_task_indices[offset:]  
                        sample_from_same_source_flags = sample_from_same_source_flags[offset:]  

                num_batch_to_yield = self.dp_worldsize
                rank_to_yield = self.dp_worldrank
                if self.use_train_part_sharding:
                    num_batch_to_yield = 1
                    rank_to_yield = 0
                for batch_data in self._prepare_batch_data(
                    tasks, weighted_task_indices, sample_from_same_source_flags, self.batch_size, phase=phase
                ):
                    if len(all_dev_batches) < num_batch_to_yield:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == num_batch_to_yield:
                        yield all_dev_batches[rank_to_yield]
                        all_dev_batches = []

            # if phase != "train" and len(all_dev_batches) > 0:
            #     while len(all_dev_batches) < self.dp_worldsize:
            #         all_dev_batches.append(all_dev_batches[-1])
            #     yield all_dev_batches[self.dp_worldrank]

        return wrapper

    def _gen_self_attn_mask_for_glm_flatten(
        self, batch_token_ids, batch_size_fact=None, unbid_idx_1=None, unbid_idx_2=None
    ):
        assert (
            len(sum(batch_token_ids, [])) <= batch_size_fact
        ), f"{len(sum(batch_token_ids, []))} > {batch_size_fact} is not allowed"

        # Note(gongenlei): unsqueeze attention mask to 4 dims
        input_mask_data = np.zeros((1, 1, batch_size_fact, batch_size_fact), dtype="float32")
        offset = 0
        for index, token_ids in enumerate(batch_token_ids):
            cur_len = len(token_ids)
            b = np.tril(np.ones([cur_len, cur_len]), 0)
            if self.start_id not in batch_token_ids[index]:
                first_start_index = 0
            else:
                first_start_index = batch_token_ids[index].index(self.start_id)
            b[:first_start_index, :first_start_index] = 1  # bi-directional attention before the first [START]
            if unbid_idx_1 is not None and unbid_idx_2 is not None:  # mask the prompt for sentence embedding
                uns1_s, uns1_e = unbid_idx_1[index]
                uns2_s, uns2_e = unbid_idx_2[index]
                b[:, uns1_s:uns1_e] = 0
                b[:, uns2_s:uns2_e] = 0
                b[uns1_s:uns1_e, uns1_s:uns1_e] = 1
                b[uns1_s:uns1_e, uns2_s:uns2_e] = 1
                b[uns2_s:uns2_e, uns1_s:uns1_e] = 1
                b[uns2_s:uns2_e, uns2_s:uns2_e] = 1
            input_mask_data[0, 0, offset : offset + cur_len, offset : offset + cur_len] = b
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
        if self.follow_deepseek:
            resever_multi_turn_break_length = 8
        else:
            resever_multi_turn_break_length = 2

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
                result = f"知识库：{result.strip()}\n根据所提供的知识库信息，回答问题并补全对话：" + add_token
            elif "search-res" in result:
                result = re.findall(r"\[<search-res>\](.*?)\[<\/search-res>\]", result, re.DOTALL | re.MULTILINE)[0]
                result = f"{result.strip()}\n根据以上参考文章回答问题，补全对话" + add_token
            elif "prompt-res" in result:
                result = re.findall(r"\[<prompt-res>\](.*?)\[<\/prompt-res>\]", result, re.DOTALL | re.MULTILINE)[0]
                result = result.strip() + add_token
            elif "compute-res" in result:
                result = re.findall(r"\[<compute-res>\](.*?)\[<\/compute-res>\]", result, re.DOTALL | re.MULTILINE)[0]
                result = f"参考文章1：{result.strip()}\n根据以上参考文章回答问题，补全对话" + add_token
            elif "citation-ref" in result:
                result = re.findall(r"\[<citation-ref>\](.*?)\[<\/citation-ref>\]", result, re.DOTALL | re.MULTILINE)[
                    0
                ]
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
                result = re.findall(r"\[<retrieve-ref>\](.*?)\[<\/retrieve-ref>\]", result, re.DOTALL | re.MULTILINE)[
                    0
                ]
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

            if self.knowledge_after_history or self.follow_deepseek:
                result += "\n"
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

        if self.add_header_eos_token:
            system_header_tokens = (
                [self.header_start_token] + tokenizer.tokenize('system') + [self.header_end_token]
            )  # ['<mask:6>', 'system', '<mask:7>']
            user_header_tokens = (
                [self.header_start_token] + tokenizer.tokenize('user') + [self.header_end_token]
            )  # ['<mask:6>', 'user', '<mask:7>']
            assisant_header_tokens = (
                [self.header_start_token] + tokenizer.tokenize('assisant') + [self.header_end_token]
            )  # ['<mask:6>', 'assis', 'ant', '<mask:7>']
            knowledge_header_tokens = (
                [self.header_start_token] + tokenizer.tokenize('knowledge') + [self.header_end_token]
            )

        if self.add_sys_token or self.add_header_eos_token:
            system_info = example.system
            if self.add_sys_token:
                if self.follow_deepseek:
                    system_tokens = [self.begin_token] + tokenizer.tokenize(system_info) + self.newline_token
                else:
                    system_tokens = [self.sys_start_token] + tokenizer.tokenize(system_info) + [self.sys_end_token]
            elif self.add_header_eos_token:
                system_tokens = system_header_tokens + tokenizer.tokenize(system_info) + [self.end_token]
            previous_cur_len += len(system_tokens)  

        turn_index = len(example.src) - 1
        if self.use_agent_sys2:  # use_agent_sys2
            while turn_index >= 0:
                src, tgt = example.src[turn_index], example.tgt[turn_index]
                tokens_src = tokenizer.tokenize(src)
                tokens_target = tokenizer.tokenize(tgt)
                is_parts_a_truncated, is_parts_b_truncated = self._truncate_seq_pair(
                    tokens_src,
                    tokens_target,
                    self.max_seq_len + 1 - previous_cur_len - resever_multi_turn_break_length,
                )
                if is_parts_b_truncated or is_parts_a_truncated:
          
                    break

                cur_tokens = tokens_src + tokens_target
                tokens = cur_tokens + tokens

        
                tmp_loss_mask = [0] * (len(tokens_src) - 1) + [example.label[turn_index]] * len(tokens_target) + [0]
                loss_mask = tmp_loss_mask + loss_mask
                assert len(tokens) == len(loss_mask), f"{len(tokens)}-{len(loss_mask)}"
                previous_cur_len += len(cur_tokens)

                turn_index -= 1

            if len(tokens) <= 4:
                return []
        else:
            knowledge_tokens = []
            while turn_index >= 0:
                src, tgt = example.src[turn_index].strip(), example.tgt[turn_index].strip()
                CONTAINS_MARKUP = False
                for markup in self.tokenizer.markup_tokens:
                    if markup in src:
                    
                        CONTAINS_MARKUP = True
                        break

                if CONTAINS_MARKUP:
                   
                    knowledge_tokens = tokenizer.tokenize(extract_knowledge(src))
                    if self.add_header_eos_token:
                        knowledge_tokens = knowledge_header_tokens + knowledge_tokens + [self.end_token]
                    src = example.src[turn_index - 1]
                    if not self.knowledge_after_history:
                        previous_cur_len += len(knowledge_tokens)

                tokens_src, tokens_target = tokenizer.tokenize(src), tokenizer.tokenize(tgt)
                if self.add_header_eos_token:
                    tokens_src = user_header_tokens + tokens_src + [self.end_token]
                    tokens_target = assisant_header_tokens + tokens_target + [self.end_token]
                if turn_index == len(example.src) - 1 and self.knowledge_after_history:
                    tokens_src = knowledge_tokens + tokens_src
                if self.follow_deepseek:
                    if turn_index == len(example.src) - 1:
                        tokens_src = knowledge_tokens + tokens_src
                    tokens_src = self.begin_of_query + tokens_src

                is_parts_a_truncated, is_parts_b_truncated = self._truncate_seq_pair(
                    tokens_src,
                    tokens_target,
                    self.max_seq_len + 1 - previous_cur_len - resever_multi_turn_break_length,
                )
                if is_parts_b_truncated or is_parts_a_truncated:
                
                    break

                if self.add_header_eos_token:
                    # tokens_src = user_header_tokens + tokens_src + [self.end_token]
                    # tokens_target = assisant_header_tokens + tokens_target + [self.end_token]
                    assert len(tokens) == len(loss_mask), f"{len(tokens)}-{len(loss_mask)}"
                    break_token_multi_turn = []

                    cur_tokens = tokens_src + tokens_target
                    tokens = cur_tokens + tokens

                    
                    loss_mask = (
                        [0] * (len(tokens_src) + len(assisant_header_tokens) - 1)
                        + [example.label[turn_index]]
                        * (len(tokens_target) - len(assisant_header_tokens) - len([self.end_token]) + 1)
                        + [0] * len([self.end_token])
                        + loss_mask
                    )
                    assert len(tokens) == len(loss_mask), f"{len(tokens)}-{len(loss_mask)}"

                elif self.follow_deepseek:
                    tokens_src = tokens_src + self.begin_of_response
                    break_token_multi_turn = [self.end_of_response]

                    cur_tokens = tokens_src + tokens_target
                    tokens = cur_tokens + break_token_multi_turn + tokens

                 
                    loss_mask = (
                        [0] * (len(tokens_src) - 1)
                        + [example.label[turn_index]] * (len(tokens_target) + 1)
                        + [0] * len(break_token_multi_turn)
                        + loss_mask
                    )
                    assert len(tokens) == len(loss_mask), f"{len(tokens)}-{len(loss_mask)}"

                else:
                    tokens_src = tokens_src + [self.break_token]
                    break_token_multi_turn = [self.break_turn_token]

                    cur_tokens = tokens_src + tokens_target
                    tokens = cur_tokens + break_token_multi_turn + tokens

                   
                    loss_mask = (
                        [0] * (len(tokens_src) - 1)
                        + [example.label[turn_index]] * (len(tokens_target) + 1)
                        + [0] * len(break_token_multi_turn)
                        + loss_mask
                    )
                    assert len(tokens) == len(loss_mask), f"{len(tokens)}-{len(loss_mask)}"

                previous_cur_len += len(cur_tokens) + len(break_token_multi_turn)

                turn_index -= 1
                if CONTAINS_MARKUP:
                    turn_index -= 1

            if len(tokens) <= 4:
                return []

            if not self.knowledge_after_history and not self.follow_deepseek:
                if self.add_break_token_multi_turn_for_nontrigger_data:
                    if self.add_start_token:  # for eb35
                        tokens = [self.start_token] + knowledge_tokens + [self.break_turn_token] + tokens
                        # 2 for start_token & break_turn_token
                        loss_mask = [0] * (2 + len(knowledge_tokens)) + loss_mask
                    else:
                        if self.add_header_eos_token:
                            tokens = knowledge_tokens + tokens
                            loss_mask = [0] * (len(knowledge_tokens)) + loss_mask  # 1 for  break_turn_token
                        else:
                            tokens = knowledge_tokens + [self.break_turn_token] + tokens
                            loss_mask = [0] * (1 + len(knowledge_tokens)) + loss_mask  # 1 for  break_turn_token
                        assert len(tokens) == len(loss_mask), f"{len(tokens)}-{len(loss_mask)}"
                else:
                    if self.add_start_token:  # for eb35
                        tokens = [self.start_token] + knowledge_tokens + tokens
                        loss_mask = [0] * (1 + len(knowledge_tokens)) + loss_mask  # 1 for start_token
                    else:
                        tokens = knowledge_tokens + tokens
                        loss_mask = [0] * (len(knowledge_tokens)) + loss_mask

            if self.add_sys_token or self.add_header_eos_token:
                if system_info and turn_index == -1:
                    tokens = system_tokens + tokens
                    loss_mask = [0] * (len(system_tokens)) + loss_mask
                    assert len(tokens) == len(loss_mask), f"{len(tokens)}-{len(loss_mask)}"

            if tokens[0] != self.begin_token and self.follow_deepseek:
                tokens = [self.begin_token] + tokens
                loss_mask = [0] + loss_mask

            # print(self.add_sys_token)
            # print(self.add_header_eos_token)
            # print(''.join(tokens))

        if self.debug_count <= 5:
            print(">>> hello_debug [loss_mask]:", loss_mask)
            print(">>> hello_debug [tokens]:", tokens)
            self.debug_count += 1

        assert len(tokens) <= self.max_seq_len, f"{len(tokens)}-{self.max_seq_len}"
        assert len(loss_mask) <= self.max_seq_len, f"{len(loss_mask)}-{self.max_seq_len}"

        self.current_example += 1
        skip_align_flag = self.all_use_skip_align
        if skip_align_flag and len(tokens) > self.skip_align_target_length:
            skip_align_flag = False
        if skip_align_flag:

            def compute_s_by_policy(skip_distance, i, turns, skip_align_s_policy="default"):
                if skip_align_s_policy == "linear":
                  
                    skip_distance = int(skip_distance * (i + 1) / turns) 
                    skip_distance = skip_distance if skip_distance != 0 else 1
                    s = self.global_rng.randint(1, skip_distance)
                    # Generated Variable Skips (turn: 15, skip_distance: 1024):
                    # [1, 78, 43, 68, 32, 28, 334, 475, 204, 135, 276, 808, 174, 562, 940]
                elif skip_align_s_policy == "beta":
                   
                    a_start, b_start = 0.5, 2.0  
                    a_end, b_end = 2.0, 0.5  
                    progress = i / max(turns - 1, 1)  
                    a = a_start + (a_end - a_start) * progress
                    b = b_start + (b_end - b_start) * progress
                    s = int(self.global_rng.betavariate(a, b) * (skip_distance - 1)) + 1
                    # Generated Skip Distances (turn: 15, skip_distance: 1024):
                    # [17, 88, 193, 100, 325, 410, 487, 818, 333, 616, 982, 558, 726, 1009, 261]
                else:
                   
                    s = self.global_rng.randint(
                        1, skip_distance
                    )  
                return s

            def find_sublist_indices(lst, sublist):
                indices = []
                sublist_length = len(sublist)

                # Iterate over the list
                for i in range(len(lst) - sublist_length + 1):
                    # Check if the sublist matches the current slice of the list
                    if lst[i : i + sublist_length] == sublist:
                        indices.append(i)

                return indices

            # print(f"------\nUsing Skip align, method: {self.skip_align_method}\n")
            # print(f"Target length: {self.skip_align_target_length}\n")
            # print(f"Subsample prob: {self.skip_align_sub_sample_prob}\n")
            # pos_ids = list(range(len(tokens)))
            pos_ids = np.arange(len(tokens))

  
            assert not self.add_header_eos_token  
            assert not self.knowledge_after_history  

            # K break_turn_token X1 break_token Y1 break_turn_token X2 break_token Y2 break_turn_token
            if self.follow_deepseek:
                break_token_ids = find_sublist_indices(tokens, self.begin_of_response)
                break_turn_token_ids = [index for index, token in enumerate(tokens) if token == self.end_of_response]
            else:
                break_token_ids = [index for index, token in enumerate(tokens) if token == self.break_token]
                break_turn_token_ids = [index for index, token in enumerate(tokens) if token == self.break_turn_token]

 
            if len(break_turn_token_ids) > len(break_token_ids):
                del break_turn_token_ids[0]
            assert len(break_token_ids) == len(break_turn_token_ids)

            u = 0
            turns = len(break_token_ids)
            for i in range(turns):
           
                assert break_token_ids[i] + 1 < break_turn_token_ids[i]
                max_skip_distance = self.skip_align_target_length - len(tokens) - u
                max_skip_distance = max_skip_distance if max_skip_distance != 0 else 1
                s = compute_s_by_policy(max_skip_distance, i, turns, self.skip_align_s_policy)
                if self.global_rng.random() < self.skip_align_sub_sample_prob and max_skip_distance != 1:
                    u += s
                # X    YX    Y
                if self.skip_align_method == "skip_inner":
                    pos_ids[break_token_ids[i] : break_turn_token_ids[i] + 1] += u  # sepY(i)cls
                    if i != turns - 1:
                        pos_ids[break_turn_token_ids[i] + 1 : break_token_ids[i + 1]] += u  # X(i+1)
                # XY    XY
                elif self.skip_align_method == "skip_outter":
                    if i != turns - 1:
                        pos_ids[
                            break_turn_token_ids[i] + 1 : break_turn_token_ids[i + 1] + 1
                        ] += u  # X(i+1)sepY(i+1)cls
                # X    Y    X    Y
                elif self.skip_align_method == "skip_all":
                    pos_ids[break_token_ids[i] : break_turn_token_ids[i] + 1] += u  # sepY(i)cls
                    max_skip_distance = self.skip_align_target_length - len(tokens) - u
                    max_skip_distance = max_skip_distance if max_skip_distance != 0 else 1
                    s = compute_s_by_policy(max_skip_distance, i, turns, self.skip_align_s_policy)
                    if self.global_rng.random() < self.skip_align_sub_sample_prob and max_skip_distance != 1:
                        u += s
                    if i != turns - 1:
                        pos_ids[break_turn_token_ids[i] + 1 : break_token_ids[i + 1]] += u  # X(i+1)
                else:
                    raise Exception

            pos_ids = pos_ids.tolist()
            if self.is_debug:
                for t, p in zip(tokens, pos_ids):
                    print(f'{t}{p} ', end='')

            del pos_ids[-1]

        if self.use_agent_sys2:
            zhli_token_map = {
                "<mask:2>": "<|prefixoftext|>",
                "<mask:3>": "<|middleoftext|>",
            }
            for i in range(len(tokens)):
                if tokens[i] in zhli_token_map:
                    tokens[i] = zhli_token_map[tokens[i]]

            labels = tokens[1:] + [self.end_token]
            for i in range(len(labels)):
                if labels[i] in ["<|prefixoftext|>", "<|middleoftext|>"]:
                    loss_mask[i] = 1
                elif labels[i] == "<mask:1>" and loss_mask[i] > 0:
                    labels[i] = self.end_token
            if self.use_agent_sys2_print > 0:
                print(f">>> use_agent_sys2 [loss_mask]:{loss_mask}")
                print(f">>> use_agent_sys2 [tokens]:{tokens}")
                print(f">>> use_agent_sys2 [labels]:{labels}")
                self.use_agent_sys2_print -= 1
        else:
            # ! force setup labels
            del tokens[-1]  # del last cls_token, there is no </s> in the last position
            del loss_mask[-1]
            labels = tokens[1:] + [self.end_token]

            # let the last token of result to predict </s>
            if self.follow_deepseek:
                labels = [l if l != self.end_of_response else self.end_token for l in labels]
            else:
                labels = [l if l != self.break_turn_token else self.end_token for l in labels]

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = tokenizer.convert_tokens_to_ids(labels)

        if self.rope_3d:
            pos_ids = np.array([[i] * 3 for i in range(len(tokens))])
        elif not skip_align_flag:
            pos_ids = list(range(len(tokens)))
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

        self.DEBUG_PRINT -= 1
        return records

    def _pad_batch_records(self, batch_records, simplify=False):
        """
        simplify
        """
        batch_record_token_ids = [record.token_ids for record in batch_records]  # leave one token for tgt_ids
        batch_token_ids = [sum(batch_record_token_ids, [])]

        if not self.rope_3d:
            batch_position_ids = [record.position_ids for record in batch_records]
            batch_position_ids = [sum(batch_position_ids, [])]
            batch_position_ids_extra = [record.position_ids_extra for record in batch_records]
            batch_position_ids_extra = [sum(batch_position_ids_extra, [])]
        else:
            batch_position_ids = [np.array(record.position_ids) for record in batch_records]
            batch_position_ids = np.concatenate(batch_position_ids)
            batch_position_ids_extra = [np.array(record.position_ids_extra) for record in batch_records]
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
                np.array([batch_position_ids_extra]), padding_value=[0, 0, 0], fix_len=self.max_seq_len
            )
        else:
            padded_position_ids_extra = pad_batch_data(
                batch_position_ids_extra, pad_idx=0, max_seq_len=self.max_seq_len
            )
        padded_token_ids = pad_batch_data(
            batch_token_ids, pad_idx=self.pad_id, return_input_mask=False, max_seq_len=self.max_seq_len
        )
        # padded_position_ids = pad_batch_data(batch_position_ids, pad_idx=0, max_seq_len=self.max_seq_len)

        padded_batch_loss_mask = pad_batch_data(batch_loss_mask, pad_idx=0, max_seq_len=self.max_seq_len)
        padded_batch_labels = pad_batch_data(batch_labels, pad_idx=self.pad_id, max_seq_len=self.max_seq_len)
        # add in-batch mask
        if not simplify:
            input_mask = self._gen_self_attn_mask_for_glm_flatten(batch_record_token_ids, self.max_seq_len)

        padded_batch_task_ids = pad_batch_data(batch_task_ids, pad_idx=0, max_seq_len=self.num_tasks)
        padded_batch_exact_total_task_ids = pad_batch_data(
            batch_exact_total_task_ids, pad_idx=0, max_seq_len=self.num_tasks
        )

        inbatch_pack_offset = [0]
        for item in batch_record_token_ids:
            inbatch_pack_offset.append(inbatch_pack_offset[-1] + len(item))
        inbatch_pack_offset[-1] = self.max_seq_len  # include padding in the last interval
        padded_inbatch_pack_offset = np.reshape(
            np.array(inbatch_pack_offset + [-1] * (self.max_seq_len + 1 - len(inbatch_pack_offset)), dtype=np.int64),
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
