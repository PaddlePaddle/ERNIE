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
This file contains the utils for the model.
"""

import datetime
import logging
import os
import re

logger = logging.getLogger(__name__)

log_dir = os.getenv("PADDLE_LOG_DIR", "./log")
local_rank = os.getenv("PADDLE_LOCAL_RANK", "0")
date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
print_data_path = os.path.join(log_dir, f"data_rank_{local_rank}_{date_str}.txt")


def print_data_online(msg):
    """
    print data online
    """
    with open(print_data_path, "a+") as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "\n")
        f.write(msg + "\n")


class MMSpecialTokensConfig:
    """_summary_"""

    use_ocr_specialtoken = True
    use_crop_specialtoken = True
    coor_num = 1001
    image_placeholder = "<|IMAGE_PLACEHOLDER|>"
    crop = ["<|CROP_COL_SEP|>", "<|CROP_ROW_SEP|>", "<|IMAGE_SEP|>"]
    ocr_coor = [f"<|LOC_{i}|>" for i in range(coor_num)]
    ocr_begin_end = ["<|LOC_BEGIN|>", "<|LOC_END|>", "<|LOC_SEP|>"]
    mm_begin_end = ["<|BOI|>", "<|EOI|>", "<|BOA|>", "<|EOA|>", "<|BOV|>", "<|EOV|>"]

    @classmethod
    def get_special_tokens_info(cls):
        """_summary_

        Returns:
            _type_: _description_
        """
        return {k: getattr(cls, k) for k in ["image_placeholder", "crop", "ocr_coor", "ocr_begin_end", "mm_begin_end"]}


class Bcolors:
    """colors for fancy print"""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def fancy_print(data, tokenizer):
    """_summary_

    Args:
        data (_type_): _description_
        tokenizer (_type_): _description_

    Returns:
        _type_: _description_
    """
    marker1 = "[unused99]"
    marker2 = "[unused98]"
    image_token = tokenizer.encode(MMSpecialTokensConfig.image_placeholder, add_special_tokens=False)["input_ids"][0]
    logger.info(f"IMAGE_TOKEN_ID: {image_token}")
    for ids, labels in zip(data["input_ids"].tolist(), data["labels"].tolist()):
        # log.info(labels)
        ids2 = []
        assert len(ids) == len(labels)
        last_j = 0
        for i, j in zip(ids, labels):
            j = int(j != tokenizer.ignored_index)
            if i == image_token:
                ids2 += tokenizer.encode("<|image|>", return_attention_mask=False)["input_ids"]
            else:
                ids2.append(i)
            if j != last_j:
                ids2 += tokenizer.encode(
                    marker1 if (j > last_j) else marker2, add_special_tokens=False, return_attention_mask=False
                )["input_ids"]
            last_j = j
        if j == 1:
            ids2 += tokenizer.encode(marker2, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        ret = tokenizer.decode(ids2).replace("[unused99]", Bcolors.FAIL).replace("[unused98]", Bcolors.ENDC)

        image_tag = "<|image|>"
        pat = re.compile(f"({re.escape(image_tag)})+")
        build = []
        for i in pat.finditer(ret):
            cnt = i.group(0).count(image_tag)
            build.append((i.span(), f"<|image@{cnt}|>"))

        pad_tag = ["<pad>", "<mask:0>", "<unk>"]
        pat = re.compile(f"({'|'.join(pad_tag)})+")
        for i in pat.finditer(ret):
            cnt = sum(i.group(0).count(t) for t in pad_tag)
            build.append((i.span(), f"<pad@{cnt}>"))

        for s, t in build[::-1]:
            l, r = s
            ret = ret[:l] + t + ret[r:]
        return ret
