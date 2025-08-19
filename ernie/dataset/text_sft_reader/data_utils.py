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
data utils for text sft
"""

import logging
import random
from collections import defaultdict, namedtuple

import numpy as np

logger = logging.getLogger(__name__)

SFTExample = namedtuple(
    "SFTExample",
    [
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
    ],
)


class RandomNoReplacementSampler:
    """
    RandomNoReplacementSampler
    """

    def __init__(self, examples, task_id, random_seed) -> None:
        self.examples = examples
        self.task_id = task_id
        self.random_seed = random_seed
        self.epoch = 0
        self.offset = 0

    def set_data_status(self, data_status):
        """
        Set the data status.

        Args:
            data_status (int): The data status value.

        Returns:
            None

        Set the epoch and offset of the current object.
        `epoch` represents the number of epochs corresponding to the current data status,
        and `offset` represents the offset of the current data status within the corresponding epoch.
        The formulas for calculating epoch and offset are:
        - epoch = data_status // len(self)
        - offset = data_status % len(self)
        """
        self.epoch = data_status // len(self)
        self.offset = data_status % len(self)

    def getter(self):
        """
        Get the data generator.

        Args:
            None

        Returns:
            A generator that yields one sample at a time.
        """
        while True:
            indices = list(range(len(self.examples)))
            rng = random.Random(self.random_seed + self.epoch + self.task_id)
            rng.shuffle(indices)
            for index in indices[self.offset :]:
                yield self.examples[index]

            self.epoch += 1
            self.offset = 0

    def __len__(self):
        return len(self.examples)


def convert_pseudo_example_list_to_example_only_opt_kb(
    previous_pseudo_example_list,
    current_pseudo_example_list,
    tokenizer,
    stop_by_k=False,
    no_opt_markups=None,
    rng=None,
    use_anti_k_sampling=False,
    drop_history_with_k=False,
):
    """
    Convert a list of pseudo examples into a list containing only examples optimized for KB.

    Args:
        previous_pseudo_example_list (list): List of pseudo examples from the previous round.
        current_pseudo_example_list (list): List of pseudo examples from the current round.
        tokenizer (Tokenizer): The tokenizer used for tokenization.
        stop_by_k (bool, optional): Whether to stop optimization based on K.
        Defaults to False.
        no_opt_markups (list, optional): List of markup tags that do not
        require optimization. Defaults to empty list.
        rng (Random, optional): Random number generator. Defaults to None.
        use_anti_k_sampling (bool, optional): Whether to use anti-K sampling strategy.
        Defaults to False.
        drop_history_with_k (bool, optional): Whether to drop history containing K.
        Defaults to False.

    Returns:
        tuple: A tuple containing the converted new examples and a mapping
        from source to optimization counts.
    """
    multi_turn_src, multi_turn_tgt, multi_turn_label = [], [], []
    source_to_num_opt = defaultdict(int)
    if no_opt_markups is None:
        no_opt_markups = []
    for i, example in enumerate(previous_pseudo_example_list):

        if example.math_is_end == 1:
            assert len(example.src) == len(
                example.ctxt_src
            ), "len(example.src) == len(example.ctxt_src)"
            assert len(example.tgt) == len(
                example.ctxt_tgt
            ), "len(example.tgt) == len(example.ctxt_tgt)"
            example = example._replace(src=example.ctxt_src, tgt=example.ctxt_tgt)

        src = example.src
        tgt = example.tgt

        label = [x and int(not stop_by_k) for x in example.label]

        if 1 in label:
            source_to_num_opt[example.source] += 1

        if not drop_history_with_k:
            multi_turn_src.extend(src)
            multi_turn_tgt.extend(tgt)
            multi_turn_label.extend(label)
        else:
            multi_turn_src.append(src)
            multi_turn_tgt.append(tgt)
            multi_turn_label.append(label)

    random_choice_index_to_opt_anti_k = -1
    if (
        len(current_pseudo_example_list) > 1
        and use_anti_k_sampling
        and rng.random() < 0.5
    ):

        random_choice_index_to_opt_anti_k = rng.choice(
            range(len(current_pseudo_example_list) - 1)
        )

    for i, example in enumerate(current_pseudo_example_list):

        if example.math_is_end == 1 and i != len(current_pseudo_example_list) - 1:
            assert len(example.src) == len(
                example.ctxt_src
            ), f"'src': {example.src}, 'ctxt_src': {example.ctxt_src}, 'source': {example.source}"
            assert len(example.tgt) == len(
                example.ctxt_tgt
            ), f"'src': {example.tgt}, 'ctxt_src': {example.ctxt_tgt}, 'source': {example.source}"
            example = example._replace(src=example.ctxt_src, tgt=example.ctxt_tgt)

        src = example.src
        tgt = example.tgt
        if i != len(current_pseudo_example_list) - 1:
            label = [x and int(not stop_by_k) for x in example.label]
            if i == random_choice_index_to_opt_anti_k:
                inner_random_opt_index = rng.choice(range(len(example.label)))
                label[inner_random_opt_index] = example.label[inner_random_opt_index]

            if 1 in label:
                source_to_num_opt[example.source] += 1
        else:

            label = example.label

            if 1 in label:
                source_to_num_opt[example.source] += 1

        if not drop_history_with_k:
            multi_turn_src.extend(src)
            multi_turn_tgt.extend(tgt)
            multi_turn_label.extend(label)
        else:
            multi_turn_src.append(src)
            multi_turn_tgt.append(tgt)
            multi_turn_label.append(label)

    # K = 30
    # cc = [x / 1000 + 1.0 / K for x in range(0, K)]
    # [i /sum(cc) for i in cc]
    if drop_history_with_k:
        K = len(multi_turn_src)
        cc = [x / 100 + 1.0 / K for x in range(0, K)]
        p = [i / sum(cc) for i in cc]
        pos = int(np.random.choice(list(range(K)), 1, p=p))  # 0, k-1
        multi_turn_src = multi_turn_src[pos:]
        multi_turn_tgt = multi_turn_tgt[pos:]
        multi_turn_label = multi_turn_label[pos:]
        multi_turn_src = sum(multi_turn_src, [])
        multi_turn_tgt = sum(multi_turn_tgt, [])
        multi_turn_label = sum(multi_turn_label, [])

    if len(previous_pseudo_example_list) != 0:
        first_example = previous_pseudo_example_list[0]
    else:
        first_example = current_pseudo_example_list[0]

    new_example = SFTExample(
        **{
            "src": multi_turn_src,
            "tgt": multi_turn_tgt,
            "label": multi_turn_label,
            "disable_pseudo_multi_turn": 0,
            "is_memory": 0,
            "is_system": 0,
            "source": "",
            "is_q2code": 0,
            "math_is_end": 2,
            "ctxt_src": [],
            "ctxt_tgt": [],
            "system": first_example.system,
            "prefix": example.prefix,
        }
    )
    return new_example, source_to_num_opt


def get_length(example, tokenizer, add_number=4):
    """
    Calculate the total sample length.
        add_number: Reserved length for [START], [MASK], [SEP], [CLS]

    Returns:
        cur_len_w_k (int): Total length including K.
        cur_len_wo_k (int): Total length excluding K.
    """
    cur_len_w_k = 0
    cur_len_wo_k = 0
    for src, tgt in zip(example.src, example.tgt):
        cur_len_wo_k += (
            len(tokenizer.tokenize(src)) + len(tokenizer.tokenize(tgt)) + add_number
        )
    cur_len_w_k = cur_len_wo_k

    return cur_len_w_k, cur_len_wo_k


def sampling_pseudo_examples(
    tasks,
    weighted_task_indices,
    sample_from_same_source_flags,
    tokenizer,
    rng,
    max_seq_len,
    pseudo_strategy,
    pseudo_sampling_prob,
    trigger_data_prob,
    use_anti_k_sampling,
    drop_history_with_k,
    use_train_part_sharding,
    dp_worldsize,
    dp_worldrank,
):
    """
    Sample pseudo examples from tasks.

    Args:
        tasks (List[Dict[str, Any]]): List of tasks, each task
        is a dictionary containing a sampler.
        weighted_task_indices (List[int]): List of weighted task indices.
        sample_from_same_source_flags (List[bool]): List of flags indicating
        whether to sample from the same source.
        tokenizer (Any): Tokenizer object.
        rng (Any): Random number generator object.
        max_seq_len (int): Maximum sequence length.
        pseudo_strategy (int): Pseudo strategy, value in [0, 3].
        pseudo_sampling_prob (float): Pseudo sampling probability.
        trigger_data_prob (float): Trigger data probability.
        use_anti_k_sampling (bool): Whether to use anti-K sampling.
        drop_history_with_k (bool): Whether to drop history containing K.
        use_train_part_sharding (bool): Whether to use training part sharding.
        dp_worldsize (int): Data parallel world size.
        dp_worldrank (int): Data parallel world rank.

    Yields:
        Tuple[Any, Dict[str, int], Dict[int, int], Dict[int, int]]:
            - example (Any): Sampled example.
            - source_to_num_opt (Dict[str, int]): Mapping from
            source to number of optimizations.
            - task_id_counter (Dict[int, int]): Task ID counter.
            - exact_total_task_id_counter (Dict[int, int]): Exact total task ID counter.
    """
    if pseudo_strategy == 0:
        no_opt_markups = []
    elif pseudo_strategy == 1:
        no_opt_markups = ["[<kg>]", "[<kg-res>]"]
    elif pseudo_strategy == 2:
        no_opt_markups = ["[<kg>]", "[<kg-res>]", "[<search>]", "[<search-res>]"]
    elif pseudo_strategy == 3:
        no_opt_markups = [
            "[<kg>]",
            "[<kg-res>]",
            "[<search>]",
            "[<search-res>]",
            "[<prompt>]",
            "[<prompt-res>]",
        ]
    else:
        no_opt_markups = []

    no_opt_markups = no_opt_markups + [
        "[<citation>]",
        "[<citation-ref>]",
        "[<kg>]",
        "[<kg-res>]",
        "[<retrieve>]",
        "[<retrieve-ref>]",
    ]

    previous_pseudo_example_list = []
    current_pseudo_example_list = []
    total_len_wo_k = 0
    total_example_num = 0

    task_id_counter = defaultdict(int)
    exact_total_task_id_counter = defaultdict(int)

    def gen_example(task_id, task_id_counter, exact_total_task_id_counter):
        task_id_local = (
            (task_id - dp_worldrank) // dp_worldsize
            if use_train_part_sharding
            else task_id
        )
        example = next(tasks[task_id_local]["sampler"])

        task_id_counter[task_id] += 1
        exact_total_task_id_counter[task_id] += 1

        return example

    for task_id, same_source_flag in zip(
        weighted_task_indices, sample_from_same_source_flags
    ):
        example = gen_example(task_id, task_id_counter, exact_total_task_id_counter)
        if (
            not same_source_flag and rng.random() > pseudo_sampling_prob
        ) or example.disable_pseudo_multi_turn:

            yield example, {
                example.source: 1
            }, task_id_counter, exact_total_task_id_counter
            task_id_counter = defaultdict(int)
            exact_total_task_id_counter = defaultdict(int)
            continue

        CONTAINS_SAME_TGT = False
        for previous_example in (
            previous_pseudo_example_list + current_pseudo_example_list
        ):
            for current_tgt_str in example.tgt:
                for previous_tgt_str in previous_example.tgt:
                    if current_tgt_str.strip() == previous_tgt_str.strip():
                        CONTAINS_SAME_TGT = True
                        break
                if CONTAINS_SAME_TGT:
                    break
            if CONTAINS_SAME_TGT:
                break
        if CONTAINS_SAME_TGT:

            yield example, {
                example.source: 1
            }, task_id_counter, exact_total_task_id_counter
            task_id_counter = defaultdict(int)
            exact_total_task_id_counter = defaultdict(int)
            continue

        len_w_k, len_wo_k = get_length(example, tokenizer, 3)
        if (
            total_len_wo_k + len_w_k > max_seq_len
            or example.is_memory
            or example.is_system
            or example.is_q2code == 1
            or example.math_is_end == 0
        ):
            ### Termination Conditions 1 & 3 & 4: ###
            # 1. Exceeds maximum length limit, clear the historical pseudo-multi-turn samples #
            # 3. Encounters a sample containing 'memory', clear the historical pseudo-multi-turn
            # samples because 'memory' must be the first sample in the sequence #
            # 4. Encounters data that only triggers computation (e.g., compute),
            # or q2code data, clear the historical pseudo-multi-turn samples #

            if example.is_q2code == 1 or example.math_is_end == 0:
                current_pseudo_example_list.append(example)
                total_example_num += 1

            if len(current_pseudo_example_list) != 0:

                new_example, source_to_num_opt = (
                    convert_pseudo_example_list_to_example_only_opt_kb(
                        previous_pseudo_example_list,
                        current_pseudo_example_list,
                        tokenizer,
                        False,
                        no_opt_markups,
                        rng,
                        use_anti_k_sampling=use_anti_k_sampling,
                        drop_history_with_k=False,
                    )
                )
                # When yielding, also output the `task_id` for
                # both non-source-equivalent and source-equivalent consumptions,
                # to record the number of consumption times.
                yield new_example, source_to_num_opt, task_id_counter, exact_total_task_id_counter
                task_id_counter = defaultdict(int)
                exact_total_task_id_counter = defaultdict(int)

            previous_pseudo_example_list, current_pseudo_example_list = [], []

            total_len_wo_k = 0
            total_example_num = 0

        if not (example.is_q2code == 1 or example.math_is_end == 0):

            current_pseudo_example_list.append(example)
            total_example_num += 1

        if not (example.is_q2code == 1 or example.math_is_end == 0):
            total_len_wo_k += len_wo_k
