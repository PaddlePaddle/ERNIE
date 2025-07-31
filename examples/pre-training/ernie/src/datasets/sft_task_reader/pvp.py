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

import re
from abc import ABC
from typing import List, Tuple


class BasePrompt(ABC):
    def __init__(self, tokenizer, break_token, break_turn_token):
        self.tokenizer = tokenizer
        self.break_token = break_token
        self.break_turn_token = break_turn_token

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    @staticmethod
    def flatten(s):
        """Return an instance of this string that is marked as shortenable"""
        tokens = []
        for x in s:
            if isinstance(x, tuple):
                tokens.extend(x[0])
            else:
                tokens.extend(x)

        return tokens

    @staticmethod
    def _seq_length(parts: List[Tuple[List, bool]], only_shortenable: bool = False):
        return (
            sum(
                [
                    len(x)
                    for x, shortenable in parts
                    if not only_shortenable or shortenable
                ]
            )
            if parts
            else 0
        )

    @staticmethod
    def _remove_last(
        parts: List[Tuple[List, bool]],
        just_begin: bool = False,
        truncate_first: bool = True,
    ):
        # print(parts)
        first_idx = min(
            idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq
        )
        last_idx = max(
            idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq
        )

        idx = first_idx if truncate_first else last_idx
        if just_begin:
            parts[idx] = (parts[idx][0][1:], parts[idx][1])
        else:
            parts[idx] = (parts[idx][0][:-1], parts[idx][1])

    def truncate(
        self,
        parts_a: List[Tuple[List, bool]],
        parts_b: List[Tuple[List, bool]],
        max_length: int,
    ):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)

        num_tokens_to_remove = total_len - max_length
        if num_tokens_to_remove <= 0:
            return False, False

        is_parts_a_truncated = False
        is_parts_b_truncated = False
        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(
                parts_b, only_shortenable=True
            ):
                self._remove_last(parts_a)
                is_parts_a_truncated = True
            else:
                self._remove_last(parts_b)
                is_parts_b_truncated = True
        return is_parts_a_truncated, is_parts_b_truncated

    def encode(self, src, tgt, max_seq_len):
        tokenizer = self.tokenizer

        if isinstance(src, list):
            for i, (x, y) in enumerate(zip(src, tgt)):
                src[i] = x.strip()
                tgt[i] = y.strip()
        else:
            src, tgt = src.strip(), tgt.strip()

        raw_parts_a, raw_parts_b = self.prompt(src, tgt)

        raw_parts_a = [x if isinstance(x, tuple) else (x, False) for x in raw_parts_a]
        raw_parts_b = [x if isinstance(x, tuple) else (x, False) for x in raw_parts_b]

        def encode_input(raw_parts):
            parts = []
            for x, s in raw_parts:
                if isinstance(x, str):
                    x = tokenizer.tokenize(x)
                else:
                    pass
                parts.append((x, s))
            return parts

        parts_a, parts_b = encode_input(raw_parts_a), encode_input(raw_parts_b)

        is_parts_a_truncated, is_parts_b_truncated = self.truncate(
            parts_a, parts_b, max_seq_len
        )

        flatten_parts_a = self.flatten(parts_a)
        flatten_parts_b = self.flatten(parts_b)
        return (
            flatten_parts_a,
            flatten_parts_b,
            is_parts_a_truncated,
            is_parts_b_truncated,
        )

    def prompt(self, src, tgt):
        return [self.shortenable(src), self.break_token], [self.shortenable(tgt)]


class SearchPrompt(BasePrompt):
    def prompt(self, src, tgt):
        assert len(src) == len(tgt) == 2

        question = src[0]
        result = re.findall(
            r"\[<search-res>\](.*?)\[<\/search-res>\]", src[1], re.S | re.M
        )[0]
        # question_modified = re.findall(r"\[<search>\](.*?)\[<\/search>\]", tgt[0], re.S | re.M)[0]
        answer = tgt[1]

        parts_a = [
            self.shortenable(result),
            "\n根据以上参考文章回答问题，补全对话",
            self.break_turn_token,
            self.shortenable(question),
            self.break_token,
        ]
        parts_b = [self.shortenable(answer)]
        return parts_a, parts_b


class KGPrompt(BasePrompt):
    def prompt(self, src, tgt):
        assert len(src) == len(tgt) == 2
        question = src[0]
        result = src[1]
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

        # question_modified = re.findall(r"\[<kg>\](.*?)\[<\/kg>\]", tgt[0], re.S | re.M)[0]
        answer = tgt[1]

        parts_a = [
            "知识库：",
            self.shortenable(result),
            "\n根据所提供的知识库信息，回答问题并补全对话：",
            self.break_turn_token,
            self.shortenable(question),
            self.break_token,
        ]
        parts_b = [self.shortenable(answer)]
        return parts_a, parts_b


class ComputePrompt(BasePrompt):
    def prompt(self, src, tgt):
        assert len(src) == len(tgt) == 2

        question = src[0]
        result = re.findall(
            r"\[<compute-res>\](.*?)\[<\/compute-res>\]", src[1], re.S | re.M
        )[0]
        # question_modified = re.findall(r"\[<compute>\](.*?)\[<\/compute>\]", tgt[0], re.S | re.M)[0]
        answer = tgt[1]

        parts_a = [
            "参考文章1：",
            self.shortenable(result),
            "\n根据以上参考文章回答问题，补全对话",
            self.break_turn_token,
            self.shortenable(question),
            self.break_token,
        ]
        parts_b = [self.shortenable(answer)]
        return parts_a, parts_b


class PromptEngine(BasePrompt):
    def prompt(self, src, tgt):
        assert len(src) == len(tgt) == 2

        question = src[0]
        result = re.findall(
            r"\[<prompt-res>\](.*?)\[<\/prompt-res>\]", src[1], re.S | re.M
        )[0]
        # question_modified = re.findall(r"\[<prompt>\](.*?)\[<\/prompt>\]", tgt[0], re.S | re.M)[0]
        answer = tgt[1]

        parts_a = [
            self.shortenable(result),
            self.break_turn_token,
            self.shortenable(question),
            self.break_token,
        ]
        parts_b = [self.shortenable(answer)]
        return parts_a, parts_b


class CitationEngine(BasePrompt):
    def prompt(self, src, tgt):
        assert len(src) == len(tgt) == 2

        question = src[0]
        result = re.findall(
            r"\[<citation-ref>\](.*?)\[<\/citation-ref>\]", src[1], re.S | re.M
        )[0]
        # question_modified = re.findall(r"\[<citation>\](.*?)\[<\/citation>\]", tgt[0], re.S | re.M)[0]
        answer = tgt[1]

        parts_a = [
            "请参考搜索结果回答下面问题并使用引用标记来标注回答内容参考的搜索结果序号，例如^[2]^ (引用单个搜索结果）,^[1][2]^（引用多个搜索结果），其中方括号中的数字是搜索结果序号。引用标记只能出现在句尾标点符号前。\n以下是搜索结果（每行开头[1]、[2]、...是搜索结果序号）：\n",
            self.shortenable(result),
            "\n根据以上搜索结果回答问题并标注引用，补全对话",
            self.break_turn_token,
            self.shortenable(question),
            self.break_token,
        ]
        parts_b = [self.shortenable(answer)]
        return parts_a, parts_b


class EBMarkUpRouter(object):
    def __init__(self, tokenizer, break_token, break_turn_token) -> None:
        self.search_prompt = SearchPrompt(tokenizer, break_token, break_turn_token)
        self.kg_prompt = KGPrompt(tokenizer, break_token, break_turn_token)
        self.compute_prompt = ComputePrompt(tokenizer, break_token, break_turn_token)
        self.prompt_engine = PromptEngine(tokenizer, break_token, break_turn_token)
        self.citation_engine = CitationEngine(tokenizer, break_token, break_turn_token)

    def encode(self, src, tgt, max_seq_len):
        assert len(src) == len(tgt) == 2, "src:{}, tgt:{}".format(src, tgt)
        if "[<search" in tgt[0]:
            return self.search_prompt.encode(src, tgt, max_seq_len)
        elif "[<kg" in tgt[0]:
            return self.kg_prompt.encode(src, tgt, max_seq_len)
        elif "[<compute" in tgt[0]:
            return self.compute_prompt.encode(src, tgt, max_seq_len)
        elif "[<prompt" in tgt[0]:
            return self.prompt_engine.encode(src, tgt, max_seq_len)
        elif "[<citation" in tgt[0]:
            return self.citation_engine.encode(src, tgt, max_seq_len)
        else:
            assert False, "src:{}, tgt:{}".format(src, tgt)
