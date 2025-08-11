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

import time
import json
from ernie.tokenizer import Ernie4_5_Tokenizer

data_file = "./examples/data/sft-train.jsonl"
tokenizer_0_3b_path = "./baidu/ERNIE-4.5-0.3B-Paddle"
tokenizer_21b_path = "./baidu/ERNIE-4.5-21B-A3B-Paddle"
tokenizer_28b_path = "./baidu/ERNIE-4.5-VL-28B-A3B-Base-Paddle"
tokenizer_300b_path = "./baidu/ERNIE-4.5-300B-A47B-Paddle"


with open(data_file, "r", encoding="utf-8") as f:
    data_list = [json.loads(line)["src"][0] for line in f if line.strip()]


def get_tokenizer_time(tokenizer_path):
    cus_time = 0
    Tokenizer = Ernie4_5_Tokenizer.from_pretrained(tokenizer_path)
    for text in data_list:
        start_time = time.time() * 10000
        tokens = Tokenizer.tokenize(text)
        tokens_ids = Tokenizer.convert_tokens_to_ids(tokens)
        Tokenizer.decode(tokens_ids)
        end_time = time.time() * 10000
        cus_time += end_time - start_time
    return cus_time / len(data_list)


def assert_tokenizer_correctness(test_io, base_io):
    """assert_tokenizer_correctness"""
    assert abs(test_io - base_io) < 0.1, "Tokenizer I/O exist diff ! "


def test_0_3b_tokenizer_io():
    assert_tokenizer_correctness(
        get_tokenizer_time(tokenizer_0_3b_path), 1.134697265625
    )


def test_21b_tokenizer_io():
    assert_tokenizer_correctness(get_tokenizer_time(tokenizer_21b_path), 1.159931640625)


def test_28b_tokenizer_io():
    assert_tokenizer_correctness(get_tokenizer_time(tokenizer_28b_path), 1.92681640625)


def test_300b_tokenizer_io():
    assert_tokenizer_correctness(get_tokenizer_time(tokenizer_300b_path), 1.0728515625)
