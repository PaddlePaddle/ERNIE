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
from src.datasets.streaming_pretrain_reader.span_selection import (
    create_recurring_span_selection_predictions,
)


def fix_seq_natual_id(token_ids, natual_sent_ids, sep_id):
    assert len(token_ids) == len(natual_sent_ids)
    len_seq = len(token_ids)
    for idx in range(len_seq):
        token_id, sent_id = token_ids[idx], natual_sent_ids[idx]
        if idx == len(token_ids) - 1:
            break
        if token_id == sep_id and (
            natual_sent_ids[idx - 1 : idx + 2] == [0, 1, 0]
            or natual_sent_ids[idx - 1 : idx + 2] == [1, 0, 1]
        ):
            natual_sent_ids[idx + 1 :] = list(
                map(lambda x: int(not x), natual_sent_ids[idx + 1 :])
            )

    return natual_sent_ids


def extract_single(token_ids, seg_labels, cls_id, sep_id, must=None):
    sep_index = token_ids.index(sep_id)

    if must == "first":
        return (token_ids[1 : sep_index + 1], seg_labels[1 : sep_index + 1])
    elif must == "second":
        return (token_ids[sep_index + 1 :], seg_labels[sep_index + 1 :])

    if np.random.random() < 0.5:
        return (token_ids[sep_index + 1 :], seg_labels[sep_index + 1 :])
    else:
        return (token_ids[1 : sep_index + 1], seg_labels[1 : sep_index + 1])


def merge_pair(sent1, sent2, cls_id, sep_id):
    token_ids = [cls_id] + sent1[0] + sent2[0]
    seg_labels = [-1] + sent1[1] + sent2[1]
    sent_ids = [0] * (len(sent1[0]) + 1) + [1] * len(sent2[0])
    pos_ids = range(len(token_ids))
    return token_ids, sent_ids, pos_ids, seg_labels


def sent_rel_16cls(token_ids, sent_ids, pos_ids, label, seg_labels, cls_id, sep_id):
    if not hasattr(sent_rel_16cls, "last_data"):
        sent_rel_16cls.last_data = None

    # np.random.seed(dp_rank)
    prob = np.random.random()
    if label != 0 and prob < 0.4:
        sep_index = token_ids.index(sep_id)
        token_ids = [cls_id] + token_ids[sep_index + 1 :] + token_ids[1 : sep_index + 1]
        sent_ids = [0] * (len(token_ids) - sep_index) + [1] * sep_index
        seg_labels = [-1] + seg_labels[sep_index + 1 :] + seg_labels[1 : sep_index + 1]
        if label % 2 == 0:
            label -= 1
        else:
            label += 1

    elif sent_rel_16cls.last_data and prob > 0.8:
        last_token_ids, last_sent_ids, last_pos_ids, last_label, last_seg_labels = (
            sent_rel_16cls.last_data
        )
        token_ids, sent_ids, pos_ids, seg_labels = merge_pair(
            extract_single(last_token_ids, last_seg_labels, cls_id, sep_id),
            extract_single(token_ids, seg_labels, cls_id, sep_id),
            cls_id,
            sep_id,
        )
        label = 15

    sent_rel_16cls.last_data = (token_ids, sent_ids, pos_ids, label, seg_labels)
    return (token_ids, sent_ids, pos_ids, label, seg_labels)


def qa_3cls(token_ids, sent_ids, pos_ids, label, seg_labels, cls_id, sep_id):
    if not hasattr(qa_3cls, "last_data"):
        qa_3cls.last_data = None

    prob = np.random.random()
    if qa_3cls.last_data and prob < 0.5:
        last_token_ids, last_sent_ids, last_pos_ids, last_label, last_seg_labels = (
            qa_3cls.last_data
        )
        token_ids, sent_ids, pos_ids, seg_labels = merge_pair(
            extract_single(last_token_ids, last_seg_labels, cls_id, sep_id),
            extract_single(token_ids, seg_labels, cls_id, sep_id),
            cls_id,
            sep_id,
        )
        label = 0

    qa_3cls.last_data = (token_ids, sent_ids, pos_ids, label, seg_labels)
    return (token_ids, sent_ids, pos_ids, label, seg_labels)


def qt_3cls(token_ids, sent_ids, pos_ids, label, seg_labels, cls_id, sep_id):
    if not hasattr(qt_3cls, "last_data"):
        qt_3cls.last_data = None

    prob = np.random.random()
    if qt_3cls.last_data and prob < 0.5:
        last_token_ids, last_sent_ids, last_pos_ids, last_label, last_seg_labels = (
            qt_3cls.last_data
        )
        if np.random.random() < 0.5:
            token_ids, sent_ids, pos_ids, seg_labels = merge_pair(
                extract_single(
                    last_token_ids, last_seg_labels, cls_id, sep_id, "first"
                ),
                extract_single(token_ids, seg_labels, cls_id, sep_id, "second"),
                cls_id,
                sep_id,
            )
        else:
            token_ids, sent_ids, pos_ids, seg_labels = merge_pair(
                extract_single(token_ids, seg_labels, cls_id, sep_id, "first"),
                extract_single(
                    last_token_ids, last_seg_labels, cls_id, sep_id, "second"
                ),
                cls_id,
                sep_id,
            )
        label = 0

    qt_3cls.last_data = (token_ids, sent_ids, pos_ids, label, seg_labels)
    return (token_ids, sent_ids, pos_ids, label, seg_labels)


def next_sent_3cls(token_ids, sent_ids, pos_ids, label, seg_labels, cls_id, sep_id):
    # np.random.seed(dp_rank)
    if label == 1 and np.random.random() < 0.5:
        sep_index = token_ids.index(sep_id)
        token_ids = [cls_id] + token_ids[sep_index + 1 :] + token_ids[1 : sep_index + 1]
        sent_ids = [0] * (len(token_ids) - sep_index) + [1] * sep_index
        label = 2
        seg_labels = [-1] + seg_labels[sep_index + 1 :] + seg_labels[1 : sep_index + 1]

    return (token_ids, sent_ids, pos_ids, label, seg_labels)


def sent_distance(
    token_ids,
    sent_ids,
    pos_ids,
    label,
    seg_labels,
    natual_sent_ids,
    cls_id,
    sep_id,
    s_id,
):
    if natual_sent_ids == None:
        return (token_ids, sent_ids, pos_ids, label, seg_labels)
    else:
        natual_sent_ids = fix_seq_natual_id(token_ids, natual_sent_ids, sep_id)
    len_seq = len(token_ids)
    begin_index = 0
    end_index, count = 1, 0
    token_ids_new, sent_ids_new, seg_labels_new = [], [], []
    while end_index < len_seq:
        if natual_sent_ids[end_index] != natual_sent_ids[end_index - 1]:
            count += 1
            token_ids_new.extend(token_ids[begin_index:end_index] + [s_id])
            sent_ids_new.extend(
                sent_ids[begin_index:end_index] + [sent_ids[end_index - 1]]
            )
            seg_labels_new.extend(seg_labels[begin_index:end_index] + [-1])
            begin_index = end_index
        end_index += 1

    token_ids_new.extend([sep_id])
    sent_ids_new.extend([sent_ids[-1]])
    seg_labels_new.extend([-1])
    pos_ids_new = list(range(len(token_ids_new)))
    assert (
        len(token_ids_new)
        == len(sent_ids_new)
        == len(pos_ids_new)
        == len(seg_labels_new)
    )
    return (token_ids_new, sent_ids_new, pos_ids_new, label, seg_labels_new)


def few_shot_pre(
    token_ids,
    sent_ids,
    pos_ids,
    label,
    seg_labels,
    natual_sent_ids,
    cls_id,
    sep_id,
    s_id,
    task_type=None,
    dataset_name=None,
    vocab=None,
    task_name=None,
    max_seq_len=None,
):
    def extra_tokens_length(sent_ids):
        add_s_cnt = 0
        for i, sent_id in enumerate(sent_ids):
            if i + 1 < len(sent_ids) and sent_ids[i + 1] != sent_id:
                add_s_cnt += 1
        return add_s_cnt

    def find_next_shot_start(seg_labels):
        """
        返回下一个ex的起始位置, 最后一个ex返回None
        """
        start = None
        try:
            start = seg_labels.index(-1, seg_labels.index(0))
        except:
            pass
        return start

    # 削减few-shot的个数，如果len(tokens) > max_seq_len.
    # 多个few-shot可以按seg_labels分割
    # examples: -1 -1 0 1 -1 -1 0 -1 -1 0 1 1
    #   ex1: -1 -1 0 1
    #   ex2: -1 -1 0
    #   ex3: -1 -1 0 1 1

    # 1. 分离few-shot和当前要预测的sample
    fewshot_token_ids_ls, fewshot_sent_ids_ls, fewshot_seg_labels_ls = [], [], []
    next_shot_start = find_next_shot_start(seg_labels)
    while next_shot_start is not None:
        fewshot_token_ids_ls.append(token_ids[:next_shot_start])
        fewshot_sent_ids_ls.append(sent_ids[:next_shot_start])
        fewshot_seg_labels_ls.append(seg_labels[:next_shot_start])

        token_ids = token_ids[next_shot_start:]
        sent_ids = sent_ids[next_shot_start:]
        seg_labels = seg_labels[next_shot_start:]

        next_shot_start = find_next_shot_start(seg_labels)

    fewshot_total_cnt = len(fewshot_token_ids_ls)
    assert fewshot_total_cnt >= 1, "预训练数据确保至少1-shot"

    # 2. 逐步合并few-shot
    fewshot_index = fewshot_total_cnt - 1
    cur_len = (
        len(token_ids)
        + len(fewshot_token_ids_ls[fewshot_index])
        + extra_tokens_length(fewshot_sent_ids_ls[fewshot_index] + sent_ids)
    )
    while cur_len < max_seq_len:
        token_ids = fewshot_token_ids_ls[fewshot_index] + token_ids
        sent_ids = fewshot_sent_ids_ls[fewshot_index] + sent_ids

        # 屏蔽掉fewshot的labels
        if "bidirectional" in task_name:
            # 双向few-shot，只保留最后一个标签用于预测，即除了最后一个label外的seg_labels全部置为-1
            seg_labels = [-1] * len(fewshot_token_ids_ls[fewshot_index]) + seg_labels
        elif "unidirectional" in task_name:
            # 单向few-shot，保持原始seg_labels不变
            seg_labels = fewshot_seg_labels_ls[fewshot_index] + seg_labels
        else:
            raise NotImplementedError()

        fewshot_index -= 1
        cur_len = (
            len(token_ids)
            + len(fewshot_token_ids_ls[fewshot_index])
            + extra_tokens_length(fewshot_sent_ids_ls[fewshot_index] + sent_ids)
        )
        if fewshot_index < 0:
            break

    natual_sent_ids = sent_ids
    return (
        token_ids,
        sent_ids,
        pos_ids,
        label,
        seg_labels,
        natual_sent_ids,
        cls_id,
        sep_id,
        s_id,
        task_type,
        dataset_name,
        vocab,
        task_name,
        max_seq_len,
    )


def glm(
    token_ids,
    sent_ids,
    pos_ids,
    label,
    seg_labels,
    natual_sent_ids,
    cls_id,
    sep_id,
    s_id,
    task_type=None,
    dataset_name=None,
    vocab=None,
    task_name=None,
    max_seq_len=None,
):
    if "fewshot" in task_name:
        (
            token_ids,
            sent_ids,
            pos_ids,
            label,
            seg_labels,
            natual_sent_ids,
            cls_id,
            sep_id,
            s_id,
            task_type,
            dataset_name,
            vocab,
            task_name,
            max_seq_len,
        ) = few_shot_pre(
            token_ids,
            sent_ids,
            pos_ids,
            label,
            seg_labels,
            natual_sent_ids,
            cls_id,
            sep_id,
            s_id,
            task_type,
            dataset_name,
            vocab,
            task_name,
            max_seq_len,
        )

    # 已经把中文间的空格，回车符号做入数据中，无需下述切句添加s_id的操作
    # len_seq = len(token_ids)
    # begin_index = 0
    # end_index, count = 1, 0
    # token_ids_new, sent_ids_new, pos_ids_new, seg_labels_new = [], [], [], []
    # while end_index < len_seq:
    #    if natual_sent_ids[end_index] != natual_sent_ids[end_index - 1]:
    #        count += 1
    #        token_ids_new.extend(token_ids[begin_index: end_index] + [s_id])
    #        sent_ids_new.extend(sent_ids[begin_index: end_index] + [sent_ids[end_index-1]])
    #        seg_label_new = [1] if seg_labels[end_index-1] != -1 else [-1]
    #        seg_labels_new.extend(seg_labels[begin_index: end_index] + seg_label_new)
    #        begin_index = end_index
    #    end_index += 1
    # if end_index - 1 >= begin_index:
    #    token_ids_new.extend(token_ids[begin_index: end_index] + [s_id])
    #    sent_ids_new.extend(sent_ids[begin_index: end_index] + [sent_ids[end_index-1]])
    #    seg_label_new = [1] if seg_labels[end_index-1] != -1 else [-1]
    #    seg_labels_new.extend(seg_labels[begin_index: end_index] + seg_label_new)
    token_ids_new, sent_ids_new, seg_labels_new = token_ids, sent_ids, seg_labels

    if task_name in [
        "glm_partial_lm",
        "glm_partial_lm_en",
        "glm_partial_lm_code_python",
        "glm_partial_lm_code_c",
        "glm_partial_lm_code_cpp",
        "glm_partial_lm_code_java",
        "glm_partial_lm_code_other",
        "glm_partial_lm_code_markdown",
        "glm_partial_lm_code_tex",
        "glm_partial_lm_code_shell",
        "glm_partial_lm_code_html",
        "glm_partial_lm_code_go",
        "glm_partial_lm_code_php",
        "glm_partial_lm_code_sql",
        "glm_partial_lm_code_javascript",
        "glm_partial_lm_mt",
        "glm_partial_lm_baby_article",
        "glm_partial_lm_edu_composition",
        "glm_partial_lm_scholar",
        "glm_partial_lm_youjia",
        "glm_partial_lm_wenan",
        "glm_partial_lm_bk_time",
        "glm_partial_lm_tushu",
        "glm_partial_lm_ly",
        "glm_partial_lm_fin",
    ]:
        task_type = "gPARAGRAPH"
    elif task_name in ["glm_chunk", "glm_chunk_en", "glm_chunk_code"]:
        task_type = "gCHUNK"
    elif task_name in ["glm_sentence", "glm_sentence_en", "glm_sentence_code"]:
        task_type = "gSENT"
    elif task_name in ["glm_span", "glm_span_en", "glm_span_code"]:
        task_type = "gENTITY"

    # 对于有监督数据的LM和sent任务，必须加上任务和数据的soft prompt，避免通用模型莫名其妙生成翻译数据的情况。
    must_add_prefix = task_name in [
        "multi_prompt_glm_partial_lm",
        "multi_prompt_glm_sentence",
        "multi_prompt_glm_chunk",
    ]
    add_prefix_prob = 0.0  # if must_add_prefix else 0.5
    if "fewshot" in task_name:
        # fewshot任务，原始文本前不加soft prompt
        add_prefix_prob = 0.0
    prefix_token_ids = []
    if np.random.random() < add_prefix_prob:
        add_task_type_prefix_prob = 0.5
        add_dataset_prefix_prob = 0.5

        added_task_type_prefix = False
        if np.random.random() < add_task_type_prefix_prob:
            prefix_token_ids.extend([vocab[f"[{task_type}{i}]"] for i in range(64)])
            added_task_type_prefix = True

        if not added_task_type_prefix and must_add_prefix:
            # 对有监督数据的LM和sent任务，如果没有添加task-type，则一定得添加dataset
            prefix_token_ids.extend([vocab[f"[{dataset_name}{i}]"] for i in range(64)])
        else:
            if (
                np.random.random() < add_dataset_prefix_prob
                and dataset_name is not None
            ):
                prefix_token_ids.extend(
                    [vocab[f"[{dataset_name}{i}]"] for i in range(64)]
                )

    token_ids_new = prefix_token_ids + token_ids_new
    sent_ids_new = [0] * len(prefix_token_ids) + sent_ids_new
    seg_labels_new = [-1] * len(prefix_token_ids) + seg_labels_new

    pos_ids_new = list(range(len(token_ids_new)))
    assert (
        len(token_ids_new)
        == len(sent_ids_new)
        == len(pos_ids_new)
        == len(seg_labels_new)
    )
    prefix_length = len(prefix_token_ids)

    return (
        token_ids_new,
        sent_ids_new,
        pos_ids_new,
        label,
        seg_labels_new,
        prefix_length,
    )


def ditto(
    token_ids,
    sent_ids,
    pos_ids,
    dulpliccate_ids,
    seg_labels,
    natual_sent_ids,
    cls_id,
    sep_id,
    s_id,
    task_type=None,
    dataset_name=None,
    vocab=None,
    task_name=None,
    max_seq_len=None,
):

    token_ids_new, sent_ids_new, pos_ids_new, seg_labels_new = (
        token_ids,
        sent_ids,
        pos_ids,
        seg_labels,
    )

    task_type = "gPARAGRAPH"

    # 对于有监督数据的LM和sent任务，必须加上任务和数据的soft prompt，避免通用模型莫名其妙生成翻译数据的情况。
    prefix_token_ids = []

    add_task_type_prefix_prob = 0.0
    if np.random.random() < add_task_type_prefix_prob:
        prefix_token_ids.extend([vocab[f"[{task_type}{i}]"] for i in range(64)])

    token_ids_new = prefix_token_ids + token_ids_new
    sent_ids_new = [0] * len(prefix_token_ids) + sent_ids_new
    seg_labels_new = [-1] * len(prefix_token_ids) + seg_labels_new
    dulpliccate_ids = [-1] * len(prefix_token_ids) + dulpliccate_ids

    pos_ids_new = list(range(len(token_ids_new)))
    assert (
        len(token_ids_new)
        == len(sent_ids_new)
        == len(pos_ids_new)
        == len(seg_labels_new)
        == len(dulpliccate_ids)
    )
    prefix_length = len(prefix_token_ids)
    return (
        token_ids_new,
        sent_ids_new,
        pos_ids_new,
        dulpliccate_ids,
        seg_labels_new,
        prefix_length,
    )


def glm_autogressive_lm(
    token_ids,
    sent_ids,
    pos_ids,
    label,
    seg_labels,
    natual_sent_ids,
    cls_id,
    sep_id,
    s_id,
    comma_id=None,
    task_type=None,
    control_type=None,
    dataset_name=None,
    vocab=None,
    task_name=None,
    max_seq_len=None,
    topic_ls=None,
    keyphrase_ls=None,
    sentiment=None,
    words=None,
    prompt_vocab=None,
):

    # 已经把中文间的空格，回车符号做入数据中，无需下述切句添加s_id的操作
    # len_seq = len(token_ids)
    # begin_index = 0
    # end_index, count = 1, 0
    # token_ids_new, sent_ids_new, pos_ids_new, seg_labels_new = [], [], [], []
    # while end_index < len_seq:
    #    if natual_sent_ids[end_index] != natual_sent_ids[end_index - 1]:
    #        count += 1
    #        token_ids_new.extend(token_ids[begin_index: end_index] + [s_id])
    #        sent_ids_new.extend(sent_ids[begin_index: end_index] + [sent_ids[end_index-1]])
    #        seg_label_new = [1] if seg_labels[end_index-1] != -1 else [-1]
    #        seg_labels_new.extend(seg_labels[begin_index: end_index] + seg_label_new)
    #        begin_index = end_index
    #    end_index += 1
    # if end_index - 1 >= begin_index:
    #    token_ids_new.extend(token_ids[begin_index: end_index] + [s_id])
    #    sent_ids_new.extend(sent_ids[begin_index: end_index] + [sent_ids[end_index-1]])
    #    seg_label_new = [1] if seg_labels[end_index-1] != -1 else [-1]
    #    seg_labels_new.extend(seg_labels[begin_index: end_index] + seg_label_new)
    token_ids_new, sent_ids_new, seg_labels_new = token_ids, sent_ids, seg_labels

    if "glm_partial_lm" == task_name or "control_glm_partial_lm" == task_name:
        task_type = "gPARAGRAPH"
    elif "glm_chunk" == task_name:
        task_type = "gCHUNK"
    elif "glm_sentence" == task_name:
        task_type = "gSENT"
    elif "glm_span" == task_name:
        task_type = "gENTITY"

    # 对于有监督数据的LM和sent任务，必须加上任务和数据的soft prompt，避免通用模型莫名其妙生成翻译数据的情况。
    must_add_prefix = task_name in [
        "multi_prompt_glm_partial_lm",
        "multi_prompt_glm_sentence",
        "multi_prompt_glm_chunk",
    ]
    add_prefix_prob = 0.0  # if must_add_prefix else 0.5
    prefix_token_ids = []
    if np.random.random() < add_prefix_prob:
        add_task_type_prefix_prob = 0.5
        add_dataset_prefix_prob = 0.5

        added_task_type_prefix = False
        if np.random.random() < add_task_type_prefix_prob:
            prefix_token_ids.extend([vocab[f"[{task_type}{i}]"] for i in range(64)])
            added_task_type_prefix = True

        if not added_task_type_prefix and must_add_prefix:
            # 对有监督数据的LM和sent任务，如果没有添加task-type，则一定得添加dataset
            prefix_token_ids.extend([vocab[f"[{dataset_name}{i}]"] for i in range(64)])
        else:
            if (
                np.random.random() < add_dataset_prefix_prob
                and dataset_name is not None
            ):
                prefix_token_ids.extend(
                    [vocab[f"[{dataset_name}{i}]"] for i in range(64)]
                )

    add_control_prefix_prob = 0.5 if comma_id is not None else 0
    if np.random.random() < add_control_prefix_prob:
        # transform original task_type to general one
        transform_to_general_prob = 0.25
        control_type = (
            "general"
            if control_type != "mt" and np.random.random() < transform_to_general_prob
            else control_type
        )
        if control_type == "poet":
            # hard code
            control_type = "poetry"

        add_prob = 0.5

        all_vocab = {**vocab, **prompt_vocab}
        # 1. add task soft prompts
        if np.random.random() < add_prob:
            total_soft_prompt_num = 64  # hard code
            task_soft_prompt_num = np.random.randint(
                1, total_soft_prompt_num + 1, size=1
            )[0]
            prefix_token_ids.extend(
                [all_vocab[f"[{control_type}{i}]"] for i in range(task_soft_prompt_num)]
            )

        # 2. add topic control codes
        if np.random.random() < add_prob:
            topic_control_codes = []
            topic_item = []
            for token in topic_ls:
                if token == -1:
                    topic_control_codes.append(topic_item)
                    topic_item = []
                    continue
                topic_item.append(token)
            if len(topic_item):
                topic_control_codes.append(topic_item)
            total_topic_num = len(topic_control_codes)
            if total_topic_num > 0:
                topic_num = np.random.randint(1, total_topic_num + 1, size=1)[0]

                # random choice method
                selected_indices = np.random.choice(
                    range(total_topic_num), size=topic_num, replace=False
                )
                topic_control_codes = [
                    topic_control_codes[idx] for idx in selected_indices
                ]

                # truncate method
                # topic_control_codes = topic_control_codes[:topic_num]

                prefix_token_ids.append(prompt_vocab["[t]"])
                for topic_i, topic in enumerate(topic_control_codes):
                    prefix_token_ids.extend(topic)
                    if topic_i != len(topic_control_codes) - 1:
                        prefix_token_ids.append(comma_id)
                prefix_token_ids.append(prompt_vocab["[/t]"])

        # 3. add keyphrase control codes
        if np.random.random() < add_prob:
            keyphrase_control_codes = []
            keyphrase_item = []
            for token in keyphrase_ls:
                if token == -1:
                    keyphrase_control_codes.append(keyphrase_item)
                    keyphrase_item = []
                    continue
                keyphrase_item.append(token)
            if len(keyphrase_item):
                keyphrase_control_codes.append(keyphrase_item)
            total_keyphrase_num = len(keyphrase_control_codes)
            if total_keyphrase_num > 0:
                keyphrase_num = np.random.randint(1, total_keyphrase_num + 1, size=1)[0]

                # random choice method
                selected_indices = np.random.choice(
                    range(total_keyphrase_num), size=keyphrase_num, replace=False
                )
                keyphrase_control_codes = [
                    keyphrase_control_codes[idx] for idx in selected_indices
                ]

                # truncate method
                # keyphrase_control_codes = keyphrase_control_codes[:keyphrase_num]

                prefix_token_ids.append(prompt_vocab["[k]"])
                for keyphrase_i, keyphrase in enumerate(keyphrase_control_codes):
                    prefix_token_ids.extend(keyphrase)
                    if keyphrase_i != len(keyphrase_control_codes) - 1:
                        prefix_token_ids.append(comma_id)
                prefix_token_ids.append(prompt_vocab["[/k]"])

        # 4. add sentiment control codes
        if np.random.random() < add_prob:
            prefix_token_ids.append(prompt_vocab["[senti]"])
            prefix_token_ids.extend(sentiment)
            prefix_token_ids.append(prompt_vocab["[/senti]"])

        # 5. add lengths
        if np.random.random() < add_prob:
            prefix_token_ids.append(prompt_vocab["[w]"])
            prefix_token_ids.extend(words)
            prefix_token_ids.append(prompt_vocab["[/w]"])

    token_ids_new = prefix_token_ids + token_ids_new
    sent_ids_new = [0] * len(prefix_token_ids) + sent_ids_new
    seg_labels_new = [-1] * len(prefix_token_ids) + seg_labels_new

    pos_ids_new = list(range(len(token_ids_new)))
    assert (
        len(token_ids_new)
        == len(sent_ids_new)
        == len(pos_ids_new)
        == len(seg_labels_new)
    )
    prefix_length = len(prefix_token_ids)

    return (
        token_ids_new,
        sent_ids_new,
        pos_ids_new,
        label,
        seg_labels_new,
        prefix_length,
    )


def autogressive_lm(
    token_ids,
    sent_ids,
    pos_ids,
    label,
    seg_labels,
    natual_sent_ids,
    cls_id,
    sep_id,
    s_id,
    comma_id=None,
    task_type=None,
    topic_ls=None,
    keyphrase_ls=None,
    sentiment=None,
    words=None,
    prompt_vocab=None,
):
    if natual_sent_ids == None:
        return (token_ids, sent_ids, pos_ids, label, seg_labels)

    # 已经把中文间的空格，回车符号做入数据中，无需下述切句添加s_id的操作
    # len_seq = len(token_ids)
    # begin_index = 0
    # end_index, count = 1, 0
    # token_ids_new, sent_ids_new, pos_ids_new, seg_labels_new = [], [], [], []
    # while end_index < len_seq:
    #    if natual_sent_ids[end_index] != natual_sent_ids[end_index - 1]:
    #        count += 1
    #        token_ids_new.extend(token_ids[begin_index: end_index] + [s_id])
    #        sent_ids_new.extend(sent_ids[begin_index: end_index] + [sent_ids[end_index-1]])
    #        seg_labels_new.extend(seg_labels[begin_index: end_index] + [-1])
    #        begin_index = end_index
    #    end_index += 1
    # if end_index - 1 >= begin_index:
    #    token_ids_new.extend(token_ids[begin_index: end_index] + [s_id])
    #    sent_ids_new.extend(sent_ids[begin_index: end_index] + [sent_ids[end_index-1]])
    #    seg_labels_new.extend(seg_labels[begin_index: end_index] + [-1])
    token_ids_new, sent_ids_new, seg_labels_new = token_ids, sent_ids, seg_labels

    prefix_token_ids = []
    add_prefix_prob = 0.0  # 0.5 if comma_id is not None else 0
    if np.random.random() < add_prefix_prob:
        # transform original task_type to general one
        transform_to_general_prob = 0.25
        task_type = (
            "general"
            if task_type != "mt" and np.random.random() < transform_to_general_prob
            else task_type
        )
        if task_type == "poet":
            # hard code
            task_type = "poetry"

        add_prob = 0.5

        all_vocab = {**vocab, **prompt_vocab}
        # 1. add task soft prompts
        if np.random.random() < add_prob:
            total_soft_prompt_num = 64  # hard code
            task_soft_prompt_num = np.random.randint(
                1, total_soft_prompt_num + 1, size=1
            )[0]
            prefix_token_ids.extend(
                [all_vocab[f"[{task_type}{i}]"] for i in range(task_soft_prompt_num)]
            )

        # 2. add topic control codes
        if np.random.random() < add_prob:
            topic_control_codes = []
            topic_item = []
            for token in topic_ls:
                if token == -1:
                    topic_control_codes.append(topic_item)
                    topic_item = []
                    continue
                topic_item.append(token)
            if len(topic_item):
                topic_control_codes.append(topic_item)
            total_topic_num = len(topic_control_codes)
            if total_topic_num > 0:
                topic_num = np.random.randint(1, total_topic_num + 1, size=1)[0]

                # random choice method
                selected_indices = np.random.choice(
                    range(total_topic_num), size=topic_num, replace=False
                )
                topic_control_codes = [
                    topic_control_codes[idx] for idx in selected_indices
                ]

                # truncate method
                # topic_control_codes = topic_control_codes[:topic_num]

                prefix_token_ids.append(prompt_vocab["[t]"])
                for topic_i, topic in enumerate(topic_control_codes):
                    prefix_token_ids.extend(topic)
                    if topic_i != len(topic_control_codes) - 1:
                        prefix_token_ids.append(comma_id)
                prefix_token_ids.append(prompt_vocab["[/t]"])

        # 3. add keyphrase control codes
        if np.random.random() < add_prob:
            keyphrase_control_codes = []
            keyphrase_item = []
            for token in keyphrase_ls:
                if token == -1:
                    keyphrase_control_codes.append(keyphrase_item)
                    keyphrase_item = []
                    continue
                keyphrase_item.append(token)
            if len(keyphrase_item):
                keyphrase_control_codes.append(keyphrase_item)
            total_keyphrase_num = len(keyphrase_control_codes)
            if total_keyphrase_num > 0:
                keyphrase_num = np.random.randint(1, total_keyphrase_num + 1, size=1)[0]

                # random choice method
                selected_indices = np.random.choice(
                    range(total_keyphrase_num), size=keyphrase_num, replace=False
                )
                keyphrase_control_codes = [
                    keyphrase_control_codes[idx] for idx in selected_indices
                ]

                # truncate method
                # keyphrase_control_codes = keyphrase_control_codes[:keyphrase_num]

                prefix_token_ids.append(prompt_vocab["[k]"])
                for keyphrase_i, keyphrase in enumerate(keyphrase_control_codes):
                    prefix_token_ids.extend(keyphrase)
                    if keyphrase_i != len(keyphrase_control_codes) - 1:
                        prefix_token_ids.append(comma_id)
                prefix_token_ids.append(prompt_vocab["[/k]"])

        # 4. add sentiment control codes
        if np.random.random() < add_prob:
            prefix_token_ids.append(prompt_vocab["[senti]"])
            prefix_token_ids.extend(sentiment)
            prefix_token_ids.append(prompt_vocab["[/senti]"])

        # 5. add lengths
        if np.random.random() < add_prob:
            prefix_token_ids.append(prompt_vocab["[w]"])
            prefix_token_ids.extend(words)
            prefix_token_ids.append(prompt_vocab["[/w]"])

    token_ids_new = prefix_token_ids + token_ids_new
    sent_ids_new = [0] * len(prefix_token_ids) + sent_ids_new
    seg_labels_new = [0] * len(prefix_token_ids) + seg_labels_new

    pos_ids_new = list(range(len(token_ids_new)))
    assert (
        len(token_ids_new)
        == len(sent_ids_new)
        == len(pos_ids_new)
        == len(seg_labels_new)
    )
    prefix_length = len(prefix_token_ids)

    if add_prefix_prob == 0:
        return (token_ids_new, sent_ids_new, pos_ids_new, label, seg_labels_new)
    return (
        token_ids_new,
        sent_ids_new,
        pos_ids_new,
        label,
        seg_labels_new,
        prefix_length,
    )


def relation_pred(
    token_ids,
    sent_ids,
    pos_ids,
    label,
    seg_labels,
    natual_sent_ids,
    cls_id,
    sep_id,
    s_id,
):
    if natual_sent_ids == None:
        return (token_ids, sent_ids, pos_ids, label, seg_labels)
    len_seq = len(token_ids)
    begin_index = 0
    end_index, count = 1, 0
    token_ids_new, sent_ids_new, seg_labels_new = [], [], []
    while end_index < len_seq:
        if natual_sent_ids[end_index] != natual_sent_ids[end_index - 1]:
            count += 1
            if token_ids[end_index - 1] == sep_id:
                token_ids_new.extend(
                    token_ids[begin_index : end_index - 1]
                    + [s_id]
                    + [token_ids[end_index - 1]]
                )
            else:
                token_ids_new.extend(token_ids[begin_index:end_index] + [s_id])
            sent_ids_new.extend(
                sent_ids[begin_index:end_index] + [sent_ids[end_index - 1]]
            )
            seg_labels_new.extend(seg_labels[begin_index:end_index] + [-1])
            begin_index = end_index
        end_index += 1
    if end_index - 1 >= begin_index:
        if token_ids[end_index - 1] == sep_id:
            token_ids_new.extend(
                token_ids[begin_index : end_index - 1]
                + [s_id]
                + [token_ids[end_index - 1]]
            )
        else:
            token_ids_new.extend(token_ids[begin_index:end_index] + [s_id])
        sent_ids_new.extend(sent_ids[begin_index:end_index] + [sent_ids[end_index - 1]])
        seg_labels_new.extend(seg_labels[begin_index:end_index] + [-1])
    pos_ids_new = list(range(len(token_ids_new)))
    assert (
        len(token_ids_new)
        == len(sent_ids_new)
        == len(pos_ids_new)
        == len(seg_labels_new)
    )
    return (token_ids_new, sent_ids_new, pos_ids_new, label, seg_labels_new)


def multi_sent_sorted(
    token_ids,
    sent_ids,
    pos_ids,
    label,
    seg_labels,
    natual_sent_ids,
    cls_id,
    sep_id,
    s_id,
):
    # np.random.seed(dp_rank)
    if natual_sent_ids != None:
        natual_sent_ids = fix_seq_natual_id(token_ids, natual_sent_ids, sep_id)
        len_seq = len(token_ids)
        begin_index = 0
        end_index, count = 1, 1
        token_ids_new, sent_ids_new, seg_labels_new = [], [], []
        while end_index < len_seq:
            if natual_sent_ids[end_index] != natual_sent_ids[end_index - 1]:
                count += 1
                token_ids_new.extend(token_ids[begin_index:end_index] + [s_id])
                sent_ids_new.extend(
                    sent_ids[begin_index:end_index] + [sent_ids[end_index - 1]]
                )
                seg_labels_new.extend(seg_labels[begin_index:end_index] + [-1])
                begin_index = end_index
            end_index += 1
        token_ids_new.extend([sep_id])
        sent_ids_new.extend([sent_ids[-1]])
        seg_labels_new.extend([-1])
        pos_ids_new = list(range(len(token_ids_new)))
        assert (
            len(token_ids_new)
            == len(sent_ids_new)
            == len(pos_ids_new)
            == len(seg_labels_new)
        )
        token_ids, sent_ids, pos_ids, seg_labels = (
            token_ids_new,
            sent_ids_new,
            pos_ids_new,
            seg_labels_new,
        )

    start = 0
    token_ids_list = []
    seg_labels_list = []
    while True:
        try:
            sep_index = token_ids[start + 1 :].index(sep_id)
            end = start + 1 + sep_index
            token_ids_list.append(token_ids[start + 1 : end])
            seg_labels_list.append(seg_labels[start + 1 : end])
            start = end
        except Exception:
            break
    premutation_2_sent = [[0, 1], [1, 0]]
    premutation_3_sent = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]
    premutation_4_sent = [
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [0, 2, 1, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        [0, 3, 2, 1],
        [1, 0, 2, 3],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 2, 3, 0],
        [1, 3, 0, 2],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 0, 3, 1],
        [2, 1, 0, 3],
        [2, 1, 3, 0],
        [2, 3, 0, 1],
        [2, 3, 1, 0],
        [3, 0, 1, 2],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
        [3, 1, 2, 0],
        [3, 2, 0, 1],
        [3, 2, 1, 0],
    ]

    shuffle_token_ids = [cls_id]
    shuffle_sent_ids = [0]
    shuffle_seg_labels = [-1]
    if label == 0 and len(token_ids_list) == 2:
        choice_index = np.random.choice(2)
        for index, order in enumerate(premutation_2_sent[choice_index]):
            shuffle_token_ids += token_ids_list[order] + [sep_id]
            shuffle_sent_ids += [index] * len(token_ids_list[order]) + [index]
            shuffle_seg_labels += seg_labels_list[order] + [-1]
        shuffle_label = label + choice_index
    elif label == 2 and len(token_ids_list) == 3:
        choice_index = np.random.choice(6)
        for index, order in enumerate(premutation_3_sent[choice_index]):
            shuffle_token_ids += token_ids_list[order] + [sep_id]
            shuffle_sent_ids += [index] * len(token_ids_list[order]) + [index]
            shuffle_seg_labels += seg_labels_list[order] + [-1]
        shuffle_label = label + choice_index
    elif label == 8 and len(token_ids_list) == 4:
        choice_index = np.random.choice(24)
        for index, order in enumerate(premutation_4_sent[choice_index]):
            shuffle_token_ids += token_ids_list[order] + [sep_id]
            shuffle_sent_ids += [index] * len(token_ids_list[order]) + [index]
            shuffle_seg_labels += seg_labels_list[order] + [-1]
        shuffle_label = label + choice_index
    else:
        print("format error")
    return (
        shuffle_token_ids,
        shuffle_sent_ids,
        pos_ids,
        shuffle_label,
        shuffle_seg_labels,
    )


def truncate_context(token_ids, sent_ids, seg_labels, max_seq_len=512):
    """
    args:
        token_ids
            type: LIST[LIST], shape: [sent_num, token_num]
    """
    token_ids_new = []
    sent_ids_new = []
    seg_labels_new = []
    for i, seg in enumerate(token_ids):
        if len(token_ids_new) + len(seg) < max_seq_len:
            token_ids_new.extend(seg)
            sent_ids_new.extend(sent_ids[i])
            seg_labels_new.extend(seg_labels[i])
        else:
            break
    if len(token_ids_new) == 0:
        token_ids_new = token_ids[0][: max_seq_len - 1] + [token_ids[0][-1]]
        sent_ids_new = sent_ids[0][: max_seq_len - 1] + [sent_ids[0][-1]]
        seg_labels_new = seg_labels[0][: max_seq_len - 1] + [seg_labels[0][-1]]

    return token_ids_new, sent_ids_new, seg_labels_new


def span_selection(
    token_ids,
    sent_ids,
    pos_ids,
    label,
    seg_labels,
    natual_sent_ids,
    cls_id,
    sep_id,
    s_id,
    comma_id=None,
    task_type=None,
    topic_ls=None,
    keyphrase_ls=None,
    sentiment=None,
    words=None,
    prompt_vocab=None,
    max_seq_len=512,
):
    label = 1  # skip preprocess for original nlg task
    (
        src_ids,
        _,
        masked_span_positions,
        span_label_beginnings,
        span_label_endings,
    ) = create_recurring_span_selection_predictions(token_ids, seg_labels)
    if len(span_label_beginnings) == 0:
        # return None is there is no recurring span.
        return None

    if natual_sent_ids == None:
        return (token_ids, sent_ids, pos_ids, label, seg_labels)
    # add s_id based on natual_sent_ids
    len_seq = len(token_ids)
    prefix_length = 0
    begin_index = 0
    end_index, count = 1, 0
    token_ids_new, sent_ids_new, pos_ids_new, seg_labels_new = [], [], [], []
    while end_index < len_seq:
        if natual_sent_ids[end_index] != natual_sent_ids[end_index - 1]:
            count += 1
            token_ids_new.append(token_ids[begin_index:end_index] + [s_id])
            sent_ids_new.append(
                sent_ids[begin_index:end_index] + [sent_ids[end_index - 1]]
            )
            seg_labels_new.append(seg_labels[begin_index:end_index] + [-1])
            begin_index = end_index
        end_index += 1
    if end_index - 1 >= begin_index:
        token_ids_new.append(token_ids[begin_index:end_index] + [s_id])
        sent_ids_new.append(sent_ids[begin_index:end_index] + [sent_ids[end_index - 1]])
        seg_labels_new.append(seg_labels[begin_index:end_index] + [-1])

    token_ids_new, sent_ids_new, seg_labels_new = truncate_context(
        token_ids_new, sent_ids_new, seg_labels_new, max_seq_len - 1
    )  # -1 for CLS
    token_ids_new = [cls_id] + token_ids_new
    sent_ids_new = [sent_ids_new[0]] + sent_ids_new
    seg_labels_new = [0] + seg_labels_new

    pos_ids_new = list(range(len(token_ids_new)))
    assert (
        len(token_ids_new)
        == len(sent_ids_new)
        == len(pos_ids_new)
        == len(seg_labels_new)
    )

    return (
        token_ids_new,
        sent_ids_new,
        pos_ids_new,
        label,
        seg_labels_new,
        prefix_length,
    )


if __name__ == "__main__":
    cls_id = 1
    sep_id = 2
    token_ids = [1, 3, 4, 5, 6, 2, 7, 8, 9, 10, 11, 2, 12, 13, 2, 14, 2]
    sent_ids = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 2]
    pos_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15]
    label = 8
    seg_labels = [-1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1]
    for tmp in multi_sent_sorted(
        token_ids, sent_ids, pos_ids, label, seg_labels, cls_id, sep_id
    ):
        print(tmp)
