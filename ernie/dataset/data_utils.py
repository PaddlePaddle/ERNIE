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
"""Useful data utility."""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from paddleformers.utils.log import logger

INF = 1000000
OPT_MULTI_OF = 256


@dataclass
class Example:
    """Data format for raw SFT (Supervised Fine-Tuning) examples."""

    request: Dict
    system: str
    label: List[int]
    is_system: int
    source: str


# 不能去掉 K 单独训练tgt 的 markup 类型
NO_OPT_MARKUPS = [
    "[<citation>]",
    "[<citation-ref>]",
    "[<kg>]",
    "[<kg-res>]",
    "[<retrieve>]",
    "[<retrieve-ref>]",
]

KG_RES_MARKUPS = [
    "[<kg-res>]",
    "[</kg-res>]",
    "[<kg-yes>]",
    "[</kg-yes>]",
    "[<kg-cs-yes>]",
    "[</kg-cs-yes>]",
    "[<kg-cs-no>]",
    "[</kg-cs-no>]",
]


def extract_knowledge(text):
    """Extracts structured knowledge from text markup.
    Args:
        text (str): Input text containing markup.
    Returns:
        str: Processed knowledge string.
    Raises:
        ValueError: If no valid knowledge pattern found.
    """

    if any(markup in text for markup in KG_RES_MARKUPS):
        for markup in KG_RES_MARKUPS + ["[<image>]", "[</image>]"]:
            text = text.replace(markup, "")
        text = f"知识库：{text.strip()}\n根据所提供的知识库信息，回答问题并补全对话："
        return text

    res = re.findall(
        r"\[<search-res>\](.*?)\[<\/search-res>\]",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if len(res) > 0:
        text = res[0]
        text = f"{text.strip()}\n根据以上参考文章回答问题，补全对话"
        return text

    res = re.findall(
        r"\[<prompt-res>\](.*?)\[<\/prompt-res>\]",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if len(res) > 0:
        text = res[0]
        text = text.strip()
        return text

    res = re.findall(
        r"\[<compute-res>\](.*?)\[<\/compute-res>\]",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if len(res) > 0:
        text = res[0]
        text = f"参考文章1：{text.strip()}\n根据以上参考文章回答问题，补全对话"
        return text

    res = re.findall(
        r"\[<citation-ref>\](.*?)\[<\/citation-ref>\]",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if len(res) > 0:
        text = res[0]
        text = (
            "请参考搜索结果回答下面问题并使用引用标记来标注回答内容参考的搜索结果序号，"
            "例如^[1]^ (引用单个搜索结果）,^[1][2]^（引用多个搜索结果），"
            "其中方括号中的数字是搜索结果序号。引用标记只能出现在句尾标点符号前。\n"
            "以下是搜索结果（每行开头[1]、[2]、...是搜索结果序号），"
            f"可以对答案中的核心部分进行markdown加粗（**加粗内容**）：\n{text.strip()}\n"
            "根据以上搜索结果回答问题并标注引用，补全对话"
        )
        return text

    res = re.findall(
        r"\[<retrieve-ref>\](.*?)\[<\/retrieve-ref>\]",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if len(res) > 0:
        text = res[0]
        text = (
            "请你扮演一个专家，参考搜索结果中正确、可信、高质量的信息回答问题，并注明答案中引用的搜索结果，"
            "格式为^[2]^表示引用了第2条搜索结果，^[1][3]^表示引用第1和第3条搜索结果。"
            "每条搜索结果包含若干相关内容片段。同时你需要遵循以下原则回答问题：\n"
            "1. 严格遵循搜索结果作答，可以承认不知道答案，并尝试给出一些搜索结果中的相关背景信息。\n"
            "2. 如果搜索结果存在多种可能的答案，要罗列出每种情况。\n"
            "3. 如果问题涉及金融、医疗、法律等存在风险的领域，请在结尾提醒用户注意并进行免责说明。\n"
            f"搜索结果：\n{text.strip()}\n\n现在，请根据上面的搜索结果回答问题并标注引用，补全对话"
        )
        return text

    raise ValueError(f"Cannot extract knowledge from `{text}`")


def get_markup_tokens():
    """Collects all special markup tokens including K-related components.

    Returns:
        List[str]: A comprehensive list of special markup tokens.
    """

    markups = ["kg", "prompt", "search"]
    markup_tokens = []
    for markup_token in markups:
        markup_tokens.extend(
            [
                f"[<{markup_token}>]",
                f"[</{markup_token}>]",
                f"[<{markup_token}-res>]",
                f"[</{markup_token}-res>]",
            ]
        )
    markup_tokens.extend(
        [
            "[<citation>]",
            "[</citation>]",
            "[<citation-ref>]",
            "[</citation-ref>]",
            "[<retrieve>]",
            "[<retrieve-ref>]",
        ]
    )
    return markup_tokens


def contains_markup(text, special_markups):
    """Checks if any markup tokens exist in the text.

    Args:
        text (List[str]): Input text sequences to check.
        special_markups (List[str]): Markup tokens to search for.

    Returns:
        bool: True if any markup is found, False otherwise.
    """

    for sp_token in special_markups:
        for x in text:
            if sp_token in x:
                return True
    return False


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
    max_len = max_seq_len if max_seq_len is not None else max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array([inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, max_len])]

    # position data
    if return_pos:
        inst_pos = np.array([list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst)) for inst in insts])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] * (max_len - len(inst)) for inst in insts])
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
        return_list += [seq_lens.astype("int64").reshape([-1, 1])]

    return return_list if len(return_list) > 1 else return_list[0]


def convert_to_tokens_for_pt(
    dial: List[dict],
    tokenizer,
    max_src_len,
):
    """Convert a dial to tokens for PT model."""
    # content_1+"\n"+content_2+"\n"+content_3
    sentence = "\n".join([x["content"] for x in dial])
    tokens = tokenizer.tokenize(sentence)
    if len(tokens) > max_src_len:
        logger.warning(
            f"The length of text ({len(tokens)}) cannot "
            f"be greater than max input length \
            ({max_src_len}). \
            We will truncate it."
        )
        # NOTE: LLM lost in middle
        tokens = tokens[: max_src_len // 2] + tokens[-max_src_len:]

    return tokens


def convert_to_tokens_for_sft(
    dial: List[dict],
    tokenizer,
    max_src_len,
):
    """
    Convert dialogue format into token sequences for supervised fine-tuning (SFT).

    Args:
        dial: Dialogue history as list of message dictionaries with:
              - role: "system", "knowledge", "user" or "assistant"
              - content: Text content
        tokenizer: Tokenizer instance for text processing
        max_src_len: Maximum allowed length for source tokens

    Returns:
        List of processed tokens ready for model input
    """
    encoded_messages = tokenizer.encode_chat_inputs({"messages": dial})

    num_reserved_tokens_for_each_dialog = 1  # only break_turn_token or end_token
    num_reserved_tokens_for_each_turn = 8

    cur_len = num_reserved_tokens_for_each_dialog

    turn_index = len(encoded_messages) - 1

    tokens = []
    tokens = encoded_messages[turn_index][0]
    turn_index -= 1

    while turn_index >= 0:
        tokens_src, tokens_target = encoded_messages[turn_index]
        if len(tokens_src) + len(tokens_target) > (max_src_len + 1 - cur_len - num_reserved_tokens_for_each_turn):
            break

        tokens = tokens_src + tokens_target + tokens
        cur_len = len(tokens)
        turn_index -= 1

    return tokens


def convert_to_input_ids(
    dials: List[List[dict]],
    tokenizer,
    data_format,
    max_src_len,
) -> Tuple[List[List[int]], int]:
    """Convert batch dialogue into input_ids.

    The API support multiple data format: `pt`, `sft.

    Args:
        dials (List[List[dict]]): A batch of dialogue.
        tokenizer (Ernie4_5_Tokenizer): The used tokenizer.
        data_format (str): The data format for converting dialogue to input_ids,
            support `base`, `chat`.
        max_src_len (int): The maximum length of input_ids.

    Returns:
        input_ids (List[List[int]]): The raw input_ids with truncation, but without padding.
        num_input_tokens (int): The total input tokens in a batch.

    Raises:
        ValueError: Invalid data format.
    """
    input_ids = []
    num_input_tokens = 0
    for dial in dials:
        if data_format == "base":
            tokens = convert_to_tokens_for_pt(dial, tokenizer, max_src_len)
            input_ids.append(tokenizer.convert_tokens_to_ids(tokens))
        elif data_format == "chat":
            input_ids.append(convert_to_tokens_for_sft(dial, tokenizer, max_src_len))
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
        num_input_tokens += len(input_ids[-1])
    return input_ids, num_input_tokens
