# !/usr/bin/env python3

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
logger utils
"""
import logging
import re
import sys
from functools import partial

bce_bns_proxy_log = logging.getLogger("bce_bns_proxy.wrapper")
bce_bns_proxy_log.disabled = True
filelock_log = logging.getLogger("filelock")
filelock_log.disabled = True

logger = logging.getLogger("data_processor.global.logging")
logger.propagate = False
logger.handlers.clear()
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter(
    fmt="[%(asctime)s] %(levelname)-8s | %(name)s:%(filename)s:%(lineno)d | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def fancy_print(input_ids, tokenizer, image_token_len=None, image_token_id=None):
    """
    input_ids: input_ids
    tokenizer: the tokenizer of models
    image_token_len: fake
    """
    i = 0
    res = ""
    text_ids = []
    real_image_token_len = 0
    while i < len(input_ids):
        if input_ids[i] == image_token_id:
            if len(text_ids) > 0:
                res += tokenizer.decode(text_ids)
                text_ids = []

            real_image_token_len += 1
        else:
            if real_image_token_len != 0:
                res += f"<|IMAGE@{real_image_token_len}|>"
                if image_token_len and image_token_len != real_image_token_len:
                    logger.warning(
                        "real image token length (%s) != input image token length (%s)",
                        real_image_token_len,
                        image_token_len,
                    )
                real_image_token_len = 0

            text_ids.append(input_ids[i])

        i += 1
    if len(text_ids) > 0:

        res += tokenizer.decode(text_ids)
        text_ids = []
    return res


def process_image_sections(image_pattern, image_token_len, match):
    """
    merge <|IMAGE@|> between <|IMAGE_START|> and <|IMAGE_END|>
    """
    content = match.group(1)
    image_count = len(re.findall(image_pattern, content))
    return f"<|IMAGE_START|><|IMAGE@{image_token_len}|>*{image_count}<|IMAGE_END|>"


def split_string_by_keywords(input_string, keywords, image_token_len):
    """
    split string by keywords
    """
    pattern = "|".join(map(re.escape, keywords))

    image_pattern = re.compile(rf"<\|IMAGE@{image_token_len}\|>")
    input_string = re.sub(
        r"<\|IMAGE_START\|>(.*?)<\|IMAGE_END\|>",
        partial(process_image_sections, image_pattern, image_token_len),
        input_string,
    )

    items = re.split(f"({pattern})", input_string)

    items = [item for item in items if item.strip()]

    merged_items = []
    i = 0
    while i < len(items):
        if items[i] in ["Picture", "Video"] and i + 1 < len(items):
            merged_items.append(f"{items[i]}{items[i + 1]}")
            i += 2
        else:
            merged_items.append(items[i])
            i += 1

    return merged_items


def fancy_print_sft(input_ids, tokenizer, image_token_len, image_token_id):
    """
    fancy_print sft的input_ids
    """
    text = fancy_print(input_ids, tokenizer, image_token_len, image_token_id)
    keywords = ["Picture", "Video", "<mask:0>", "<|endofprompt|>"]
    result = split_string_by_keywords(text, keywords, image_token_len)
    return "\n".join(result)
