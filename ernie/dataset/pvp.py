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

import re
from abc import ABC
from typing import List, Tuple


class BasePrompt(ABC):
    """Abstract base class for different prompt types."""

    def __init__(self, tokenizer, break_token, break_turn_token):
        """
        Initialize base prompt configuration.

        Args:
            tokenizer (Ernie4_5_Tokenizer): Tokenizer used for text processing.
            break_token (str): Token indicating dialogue turn break.
            break_turn_token (str): Token indicating major context break.
        """
        self.tokenizer = tokenizer
        self.break_token = break_token
        self.break_turn_token = break_turn_token

    @staticmethod
    def shortenable(s):
        """Mark a string as shortenable during truncation.

        Returns:
            Tuple[str, bool]: The original string with True flag.
        """
        return s, True

    @staticmethod
    def flatten(s):
        """Flatten nested token sequences into a single list.

        Returns:
            List[str]: Flattened list of tokens from nested sequences.
        """
        tokens = []
        for x in s:
            if isinstance(x, tuple):
                tokens.extend(x[0])
            else:
                tokens.extend(x)

        return tokens

    @staticmethod
    def _seq_length(parts: List[Tuple[List, bool]], only_shortenable: bool = False):
        """Calculate total length of sequences, optionally counting only shortenable parts.

        Args:
            parts (List[Tuple[List, bool]]): List of tuples containing token sequences and shortenable flags.
            only_shortenable (bool): Whether to count only shortenable parts.

        Returns:
            int: Total length of specified parts.
        """
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(
        parts: List[Tuple[List, bool]],
        just_begin: bool = False,
        truncate_first: bool = True,
    ):
        """Remove last token from specified part for truncation.

        Args:
            parts (List[Tuple[List, bool]]): List of tuples containing token sequences and shortenable flags.
            just_begin (bool): Whether to remove token from the beginning of the sequence.
            truncate_first (bool): Whether to truncate from the first or last shortenable part.

        """

        first_idx = min(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)

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
        """Truncate two sequences to meet total maximum length requirement.
        Args:
            parts_a (List[Tuple[List, bool]]): List of tuples containing token sequences
                                               and shortenable flags for part A.
            parts_b (List[Tuple[List, bool]]): List of tuples containing token sequences
                                               and shortenable flags for part B.
            max_length (int): Maximum allowed sequence length.

        Returns:
            Tuple[bool, bool]: Flags indicating whether parts_a/parts_b were truncated.
        """
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)

        num_tokens_to_remove = total_len - max_length
        if num_tokens_to_remove <= 0:
            return False, False

        is_parts_a_truncated = False
        is_parts_b_truncated = False
        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
                is_parts_a_truncated = True
            else:
                self._remove_last(parts_b)
                is_parts_b_truncated = True
        return is_parts_a_truncated, is_parts_b_truncated

    def encode(self, src, tgt, max_seq_len):
        """Encode source and target into token sequences with truncation.

        Args:
            src: Source text or list of source texts.
            tgt: Target text or list of target texts.
            max_seq_len: Maximum sequence length.

        Returns:
            Tuple[List, List, bool, bool]: Encoded parts_a, parts_b, and truncation flags.
        """
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

        is_parts_a_truncated, is_parts_b_truncated = self.truncate(parts_a, parts_b, max_seq_len)

        flatten_parts_a = self.flatten(parts_a)
        flatten_parts_b = self.flatten(parts_b)
        return (
            flatten_parts_a,
            flatten_parts_b,
            is_parts_a_truncated,
            is_parts_b_truncated,
        )

    def prompt(self, src, tgt):
        """Generate prompt structure for given source and target.
        Args:
            src: Source text or list of source texts.
            tgt: Target text or list of target texts.

        Returns:
            Tuple[List, List]: Formatted prompt parts for source and target.
        """
        return [self.shortenable(src), self.break_token], [self.shortenable(tgt)]


class SearchPrompt(BasePrompt):
    """
    A prompt class for handling search-related tasks, extracting search results and formatting the response.
    """

    def prompt(self, src, tgt):
        """
        Args:
            src (list): Source data containing the question and search result markup.
            tgt (list): Target data containing the answer markup.

        Returns:
            tuple: Two lists (parts_a, parts_b) containing formatted prompts and answers.
        """
        assert len(src) == len(tgt) == 2

        question = src[0]
        result = re.findall(
            r"\[<search-res>\](.*?)\[<\/search-res>\]",
            src[1],
            re.DOTALL | re.MULTILINE,
        )[0]
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
    """
    A prompt class for knowledge graph tasks, processing KG markup and generating responses.
    """

    def prompt(self, src, tgt):
        """
        Args:
            src (list): Source data containing the question and KG result markup.
            tgt (list): Target data containing the answer markup.

        Returns:
            tuple: Two lists (parts_a, parts_b) for formatted prompts and answers.
        """
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
    """
    A prompt class for computation tasks, extracting compute results and formatting responses.
    """

    def prompt(self, src, tgt):
        """
        Args:
            src (list): Source data containing the question and compute result markup.
            tgt (list): Target data containing the answer markup.

        Returns:
            tuple: Two lists (parts_a, parts_b) for formatted prompts and answers.
        """
        assert len(src) == len(tgt) == 2

        question = src[0]
        result = re.findall(
            r"\[<compute-res>\](.*?)\[<\/compute-res>\]",
            src[1],
            re.DOTALL | re.MULTILINE,
        )[0]
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
    """
    Engine for handling prompt-related tasks, extracting prompt results and generating responses.
    """

    def prompt(self, src, tgt):
        """
        Args:
            src (list): Source data containing the question and prompt result markup.
            tgt (list): Target data containing the answer markup.

        Returns:
            tuple: Two lists (parts_a, parts_b) for formatted prompts and answers.
        """
        assert len(src) == len(tgt) == 2

        question = src[0]
        result = re.findall(
            r"\[<prompt-res>\](.*?)\[<\/prompt-res>\]",
            src[1],
            re.DOTALL | re.MULTILINE,
        )[0]
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
    """
    Engine for citation tasks, formatting responses with citation references.
    """

    def prompt(self, src, tgt):
        """
        Args:
            src (list): Source data containing the question and citation reference markup.
            tgt (list): Target data containing the answer markup.

        Returns:
            tuple: Two lists (parts_a, parts_b) for formatted prompts and answers with citations.
        """
        assert len(src) == len(tgt) == 2

        question = src[0]
        result = re.findall(
            r"\[<citation-ref>\](.*?)\[<\/citation-ref>\]",
            src[1],
            re.DOTALL | re.MULTILINE,
        )[0]
        answer = tgt[1]

        parts_a = [
            "请参考搜索结果回答下面问题并使用引用标记来标注回答内容参考的搜索结果序号，"
            "例如^[1]^ (引用单个搜索结果）,^[1][2]^（引用多个搜索结果），"
            "其中方括号中的数字是搜索结果序号。引用标记只能出现在句尾标点符号前。\n"
            "以下是搜索结果（每行开头[1]、[2]、...是搜索结果序号），"
            "可以对答案中的核心部分进行markdown加粗（**加粗内容**）：\n",
            self.shortenable(result),
            "\n根据以上搜索结果回答问题并标注引用，补全对话",
            self.break_turn_token,
            self.shortenable(question),
            self.break_token,
        ]
        parts_b = [self.shortenable(answer)]
        return parts_a, parts_b


class RetrieveEngine(BasePrompt):
    """
    Engine for retrieve-ref tasks, handling retrieval references and generating responses with citations.
    """

    def prompt(self, src, tgt):
        """
        Args:
            src (list): Source data containing the question and retrieve-ref markup.
            tgt (list): Target data containing the modified question and answer markup.

        Returns:
            tuple: Two lists (parts_a, parts_b) for formatted prompts and answers with citations.
        """
        assert len(src) == len(tgt) == 2

        question = src[0]
        result = re.findall(
            r"\[<retrieve-ref>\](.*?)\[<\/retrieve-ref>\]",
            src[1],
            re.DOTALL | re.MULTILINE,
        )[0]
        question_modified = re.findall(
            r"\[<retrieve>\](.*?)\[<\/retrieve>\]",
            tgt[0],
            re.DOTALL | re.MULTILINE,
        )[0]
        answer = tgt[1]

        parts_a = [
            "请你扮演一个专家，参考搜索结果中正确、可信、高质量的信息回答问题，"
            "并注明答案中引用的搜索结果，格式为^[2]^表示引用了第2条搜索结果，"
            "^[1][3]^表示引用第1和第3条搜索结果。每条搜索结果包含若干相关内容片段。"
            "同时你需要遵循以下原则回答问题：\n"
            "1. 严格遵循搜索结果作答，可以承认不知道答案，并尝试给出一些搜索结果中的相关背景信息。\n"
            "2. 如果搜索结果存在多种可能的答案，要罗列出每种情况。\n"
            "3. 如果问题涉及金融、医疗、法律等存在风险的领域，请在结尾提醒用户注意并进行免责说明。\n"
            "搜索结果：\n",
            self.shortenable(result),
            "\n\n现在，请根据上面的搜索结果回答问题并标注引用，补全对话",
            self.break_turn_token,
            self.shortenable(question),
            self.break_token,
        ]
        parts_b = [self.shortenable(answer)]
        return parts_a, parts_b


class EBMarkUpRouter:
    """
    Router class to encode inputs using appropriate prompt engines based on markup tags.
    """

    def __init__(self, tokenizer, break_token, break_turn_token) -> None:
        """
        Args:
            tokenizer: Tokenizer for processing text.
            break_token (str): Token used to separate prompt sections.
            break_turn_token (str): Token used to separate dialogue turns.
        """
        self.search_prompt = SearchPrompt(tokenizer, break_token, break_turn_token)
        self.kg_prompt = KGPrompt(tokenizer, break_token, break_turn_token)
        self.compute_prompt = ComputePrompt(tokenizer, break_token, break_turn_token)
        self.prompt_engine = PromptEngine(tokenizer, break_token, break_turn_token)
        self.citation_engine = CitationEngine(tokenizer, break_token, break_turn_token)
        self.retrieve_engine = RetrieveEngine(tokenizer, break_token, break_turn_token)

    def encode(self, src, tgt, max_seq_len):
        """
        Args:
            src (list): Source data for encoding.
            tgt (list): Target data for encoding.
            max_seq_len (int): Maximum sequence length for encoding.

        Returns:
            Encoded output from the appropriate prompt engine based on markup tags.
        """
        assert len(src) == len(tgt) == 2, f"src:{src}, tgt:{tgt}"
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
        elif "[<retrieve" in tgt[0]:
            return self.retrieve_engine.encode(src, tgt, max_seq_len)
        else:
            raise AssertionError(f"src:{src}, tgt:{tgt}")
