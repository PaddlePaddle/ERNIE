# !/usr/bin/env python3
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
import numpy as np
import paddle

from collections import defaultdict, namedtuple

SFTExample = namedtuple("SFTExample", ["src", "tgt", "label", "is_memory", "source"])


class RandomNoReplacementSampler(object):
    def __init__(self, examples, rng=None) -> None:
        self.examples = examples
        self.rng = rng
        self.indices = list(range(len(self.examples)))

    def set_rng(self, rng):
        self.rng = rng

    def getter(
        self,
    ):
        while True:
            self.rng.shuffle(self.indices)
            for i in self.indices:
                yield self.examples[i]

    def __len__(self):
        return len(self.examples)


def contains_markup(text, special_markups):
    FOUND_MARKUP = False
    for sp_token in special_markups:
        for x in text:
            if sp_token in x:
                FOUND_MARKUP = True
                break
    return FOUND_MARKUP


def convert_pseudo_example_list_to_example_only_opt_kb(
    previous_pseudo_example_list,
    current_pseudo_example_list,
    tokenizer,
    stop_by_k=False,
    no_opt_markups=[],
    rng=None,
    use_anti_k_sampling=False,
):
    multi_turn_src, multi_turn_tgt, multi_turn_label = [], [], []
    source_to_num_opt = defaultdict(int)
    for i, example in enumerate(previous_pseudo_example_list):
        # 历史多轮处理
        if contains_markup(example.src, tokenizer.markup_tokens):
            # 根据no_opt_markups判断是否需要优化原始带K，但此时去掉K的B
            is_contain_no_opt_markups = contains_markup(example.src, no_opt_markups)
            # 包含K，则去掉K
            src = example.src[:-2] + [example.src[-2]]
            tgt = example.tgt[:-2] + [example.tgt[-1]]

            label = [
                x and int(not stop_by_k) for x in example.label[:-2]
            ]  # 因为k停止，则不优化
            label.append(
                int(not stop_by_k and not is_contain_no_opt_markups)
            )  # 非k停止，并且不包含no_opt_markups，才为1
            if 1 in label:
                source_to_num_opt[example.source] += 1
        else:
            src = example.src
            tgt = example.tgt

            label = [
                x and int(not stop_by_k) for x in example.label
            ]  # 因为k停止，则不优化

            if 1 in label:
                source_to_num_opt[example.source] += 1

        # 涉黄问题前序轮去掉prompt
        new_src = []
        for x in src:
            x = x.replace("\n这是一个涉习问题", "")
            x = x.replace("\n这是一个涉政问题", "")
            x = x.replace("\n这是一个涉黄问题", "")
            x = x.replace("\n这是一个违法犯罪问题", "")
            new_src.append(x)

        multi_turn_src.extend(new_src)
        multi_turn_tgt.extend(tgt)
        multi_turn_label.extend(label)

    # 随机选择一个优化
    random_choice_index_to_opt_anti_k = -1
    if (
        len(current_pseudo_example_list) > 1
        and use_anti_k_sampling
        and rng.random() < 0.5
    ):
        # 开启anti k策略后，50%概率选择一个不带k的b进行优化
        random_choice_index_to_opt_anti_k = rng.choice(
            range(len(current_pseudo_example_list) - 1)
        )

    for i, example in enumerate(current_pseudo_example_list):
        # 当前轮处理，只有最后一轮可能有K
        src = example.src
        tgt = example.tgt
        if i != len(current_pseudo_example_list) - 1:
            label = [
                x and int(not stop_by_k) for x in example.label
            ]  # 因为k停止，则不优化
            if i == random_choice_index_to_opt_anti_k:
                inner_random_opt_index = rng.choice(range(len(example.label)))
                label[inner_random_opt_index] = example.label[inner_random_opt_index]

            if 1 in label:
                source_to_num_opt[example.source] += 1

            # 涉黄问题前序轮去掉prompt
            new_src = []
            for x in src:
                x = x.replace("\n这是一个涉习问题", "")
                x = x.replace("\n这是一个涉政问题", "")
                x = x.replace("\n这是一个涉黄问题", "")
                x = x.replace("\n这是一个违法犯罪问题", "")
                new_src.append(x)
            src = new_src
        else:
            # 最后一个
            if contains_markup(src, tokenizer.markup_tokens):
                # 包含K，只优化KB，前序的不优化
                label = [0] * len(src[:-2])
                label.extend([1] * (len(src) - len(src[:-2])))
            else:
                label = example.label

            if 1 in label:
                source_to_num_opt[example.source] += 1

        multi_turn_src.extend(src)
        multi_turn_tgt.extend(tgt)
        multi_turn_label.extend(label)

    new_example = SFTExample(
        **{
            "src": multi_turn_src,
            "tgt": multi_turn_tgt,
            "label": multi_turn_label,
            "is_memory": 0,
            "source": "",
        }
    )
    return new_example, source_to_num_opt


def get_length(example, tokenizer, eb_markup_rounter, add_number=4):
    """
    计算样本总长度
        add_number: [START] [MASK] [SEP] [CLS] 预留长度

    返回：
        cur_len_w_k 包含k的总长度
        cur_len_wo_k 不包含k的总长度
    """
    cur_len_w_k = 0
    cur_len_wo_k = 0
    if contains_markup(example.src, tokenizer.markup_tokens) and contains_markup(
        example.tgt, tokenizer.markup_tokens
    ):
        try:
            tokens_src, tokens_target, is_parts_a_truncated, is_parts_b_truncated = (
                eb_markup_rounter.encode(example.src[-2:], example.tgt[-2:], 10000)
            )
        except Exception:
            print(example, example.src)
            assert False
        cur_len_w_k += len(tokens_src) + len(tokens_target)

        for src, tgt in zip(example.src[:-2], example.tgt[:-2]):
            src_len = len(tokenizer.tokenize(src))
            tgt_len = len(tokenizer.tokenize(tgt))
            cur_len_w_k += src_len + tgt_len + add_number
            cur_len_wo_k += src_len + tgt_len + add_number

        cur_len_wo_k += (
            len(tokenizer.tokenize(example.src[-2]))
            + len(tokenizer.tokenize(example.tgt[-1]))
            + add_number
        )
    else:
        for src, tgt in zip(example.src, example.tgt):
            cur_len_wo_k += (
                len(tokenizer.tokenize(src)) + len(tokenizer.tokenize(tgt)) + add_number
            )
        cur_len_w_k = cur_len_wo_k

    return cur_len_w_k, cur_len_wo_k


def sampling_pseudo_examples(
    examples_all,
    examples_per_task,
    tokenizer,
    eb_markup_rounter,
    rng,
    max_seq_len,
    pseudo_strategy,
    example_from_same_task_prob,
    pseudo_sampling_prob,
    trigger_data_prob,
    use_anti_k_sampling,
):
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

    # 构造伪多轮
    previous_pseudo_example_list = []
    current_pseudo_example_list = []
    total_len_wo_k = 0
    total_example_num = 0

    total_task_num = len(examples_per_task)
    FORCE_EXAMPLE_FROM_SAME_TASK = False
    task_index = 0
    while len(examples_all) > 0:
        if rng.random() > pseudo_sampling_prob:
            # 不走伪多轮
            example = examples_all.pop()
            # yield example, 1
            yield example, {example.source: 1}
            continue

        total_task_num = len(examples_per_task)
        # 如果examples_all没有值，则退出
        if (
            not FORCE_EXAMPLE_FROM_SAME_TASK
            and rng.random() < example_from_same_task_prob
            and total_task_num > 0
        ):
            # example_from_same_task_prob的概率开启连续从相同类型样本中的采样策略
            FORCE_EXAMPLE_FROM_SAME_TASK = True
            # 确定4096长度的伪多轮均来自于task_index
            task_index = rng.randint(0, total_task_num)

        if FORCE_EXAMPLE_FROM_SAME_TASK:
            # 从同类样本从采样
            example = examples_per_task[task_index].pop()
            if len(examples_per_task[task_index]) == 0:
                examples_per_task = (
                    examples_per_task[:task_index] + examples_per_task[task_index + 1 :]
                )  # 删除空列表
                FORCE_EXAMPLE_FROM_SAME_TASK = (
                    False  # 该任务已经采样完，则提前终止从相同类型样本中的采样策略
                )
        else:
            # 从正常的合并后的样本中采样
            example = examples_all.pop()

        # 如果当前example存在相同的tgt在之前轮，则不加入
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
            continue

        if (
            contains_markup(example.src, tokenizer.markup_tokens)
            and rng.random() > trigger_data_prob
        ):
            # 如果是触发数据，并且采样为非触发，则跳过删除knowledge，当做普通QA训练
            example = SFTExample(
                **{
                    "src": example.src[:-1],
                    "tgt": example.tgt[:-2] + example.tgt[-1:],
                    "label": example.label[:-2] + example.label[-1:],
                    "is_memory": example.is_memory,
                    "source": example.source,
                }
            )

        len_w_k, len_wo_k = get_length(example, tokenizer, eb_markup_rounter, 3)
        if (
            total_len_wo_k + len_w_k > max_seq_len
            or example.is_memory
            or contains_markup(example.tgt, ["[<compute>]"])
        ):
            # 终止条件1: #
            # 1. 超过最大长度限制，需清空历史伪多轮 #
            # 3. 遇到包含memory的样本，需清空历史伪多轮，因为memory必须为开头第一个样本 #
            # 4. 遇到包含只触发的数据，需清空历史伪多轮，如compute #

            if contains_markup(example.tgt, ["[<compute>]"]):
                current_pseudo_example_list.append(example)
                total_example_num += 1

            if len(current_pseudo_example_list) != 0:
                # 当前有新增需要优化的，则输出
                new_example, source_to_num_opt = (
                    convert_pseudo_example_list_to_example_only_opt_kb(
                        previous_pseudo_example_list,
                        current_pseudo_example_list,
                        tokenizer,
                        False,
                        no_opt_markups,
                        rng,
                        use_anti_k_sampling=use_anti_k_sampling,
                    )
                )
                FORCE_EXAMPLE_FROM_SAME_TASK = False  # 重置从相同来源采样逻辑
                # yield new_example, total_example_num
                yield new_example, source_to_num_opt
            # 清空结果
            previous_pseudo_example_list, current_pseudo_example_list = [], []
            # 加入当前example
            total_len_wo_k = 0
            total_example_num = 0

        if not contains_markup(example.tgt, ["[<compute>]"]):
            # 非单独触发数据，才加入当前集合
            current_pseudo_example_list.append(example)
            total_example_num += 1

        if contains_markup(example.src, tokenizer.markup_tokens):
            # 终止条件2: #
            # 2. 包含Markup #

            new_example, source_to_num_opt = (
                convert_pseudo_example_list_to_example_only_opt_kb(
                    previous_pseudo_example_list,
                    current_pseudo_example_list,
                    tokenizer,
                    True,
                    no_opt_markups,
                    rng,
                    use_anti_k_sampling=use_anti_k_sampling,
                )
            )
            # yield new_example, total_example_num
            yield new_example, source_to_num_opt
            total_example_num = 0

            # 新增加入历史伪多轮中
            previous_pseudo_example_list.extend(current_pseudo_example_list)
            current_pseudo_example_list = []

        total_len_wo_k += len_wo_k  # 更新当前样本长度


if paddle.is_compiled_with_cuda():
    int_type = "int64"
else:
    raise Exception("paddle must compiled with cuda or npu")


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
