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

import random
import numpy as np
from collections import namedtuple

# import paddle
# import tokenization_wp


class GeneralReader(object):
    def __init__(self, random_seed=None, is_debug=False, tokenizer=None):
        self.use_2d_pos = True
        self.is_debug = is_debug
        if self.is_debug:
            print(
                "[warning] debug switcher is on, will use only 20 samples for testing"
            )

        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.start_token = "<s>"
        self.end_token = "</s>"
        self.start_id = self.vocab[self.start_token]
        self.end_id = self.vocab[self.end_token]
        self.DEBUG_PRINT = 5

        if random_seed is None:
            assert False, "random_seed can not be None"
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)
        np.random.seed(random_seed)

    def _flatten_text(self, text):
        """convert list or nested list of text to plain text"""
        if isinstance(text, str):  # leaf
            return text
        elif isinstance(text, list):
            new_text = [self._flatten_text(t) for t in text]
            return "\n".join(new_text)
        else:
            return text

    def _convert_example_to_record(self, example, task_name=None):
        tokenizer = self.tokenizer
        tokens_prefix = []

        if "code" in task_name:
            text_src = example.code
        elif "contents" in example._fields:  # for novel:
            text_src = "\n".join([c["content"] for c in example.contents[0]])
        else:
            text_src = None
            possible_list = ["content", "text", "src"]
            for p in possible_list:
                if p in example._fields:
                    text_src = eval(f"example.{p}")
            if text_src is None:
                return None

        if text_src == "":
            return None

        # flatten text
        if isinstance(text_src, list):
            text_src = self._flatten_text(text_src)

        if "title" in example._fields:
            title = self._flatten_text(example.title)
            text_src = "\n".join([title, text_src])
        tokens = tokenizer.tokenize(text_src)
        tokens = [self.start_token] + tokens + [self.end_token]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        pos_ids_extra = list(range(0, len(tokens)))
        pos_ids = [0] * len(tokens)  # 单向时，该字段用来区分src和tgt

        Record = namedtuple(
            "Record",
            [
                "token_ids",
                "position_ids",
                "position_ids_extra",
                "label",
                "last_token_label",
            ],
        )
        label = 0  # fake label
        last_token_label = self.start_token
        if self.DEBUG_PRINT > 0:
            print(
                "[DEBUG]",
                "tokens:",
                tokens,
                "token_ids:",
                token_ids,
                "pos_ids:",
                pos_ids,
                "pos_ids_extra:",
                pos_ids_extra,
                "label",
                label,
                "last_token_label",
                last_token_label,
            )
        assert len(token_ids) != 0

        record = Record(
            token_ids=token_ids,
            position_ids=pos_ids,
            position_ids_extra=pos_ids_extra,
            label=label,
            last_token_label=tokenizer.convert_tokens_to_ids([last_token_label])[0],
        )

        self.DEBUG_PRINT -= 1
        return record

    def generator(self, data, task_index=None, task_name=None):
        data = {
            key: data[key]
            for key in list(data.keys())
            if key in ["src", "tgt", "content"]
        }
        Example = namedtuple("Example", list(data.keys()))
        try:
            example = Example(**data)
        except Exception as e:
            print(line)
            raise e
        try:
            record = self._convert_example_to_record(example, task_name)
        except Exception as e:
            print(e)
            print("error!", data)
            return None
        if record is None:
            return None
        token_ids = record.token_ids
        pos_ids = record.position_ids
        pos_ids_extra = record.position_ids_extra
        last_token_label = record.last_token_label

        # generate fake output
        sent_ids = [0] * len(token_ids)
        if isinstance(task_index, int):
            task_ids = [task_index] * len(token_ids)
        else:
            task_ids = [0] * len(token_ids)
        label = -1
        seg_labels = [0] * len(token_ids)
        is_dulplicated = [-1] * len(token_ids)
        prefix_length = 0

        return (
            token_ids,
            sent_ids,
            pos_ids,
            pos_ids_extra,
            last_token_label,
            task_ids,
            label,
            seg_labels,
            is_dulplicated,
            prefix_length,
            1,
            1,
            1,
        )  ## [....., cal_lm_loss, is_starts, is_ends]


if __name__ == "__main__":
    pass
