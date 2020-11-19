#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sys
import argparse
import logging

import json
from collections import namedtuple

log = logging.getLogger(__name__)

Example = namedtuple('Example', [
    'qas_id', 'question_text', 'doc_tokens', 'orig_answer_text',
    'start_position', 'end_position'
])

Feature = namedtuple("Feature", [
    "unique_id", "example_index", "doc_span_index", "tokens",
    "token_to_orig_map", "token_is_max_context", "token_ids", "position_ids",
    "text_type_ids", "start_position", "end_position"
])


def _tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""

    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    output = []
    buff = ""
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp):
            if buff != "":
                output.append(buff)
                buff = ""
            output.append(char)
        else:
            buff += char

    if buff != "":
        output.append(buff)

    return output


def _check_is_max_context(doc_spans, cur_span_index, position):
    """chech is max context"""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context,
                    num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """improve answer span"""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def read_files(input_file, is_training):
    """read file"""
    examples = []
    with open(input_file, "r") as f:
        input_data = json.load(f)["data"]
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_pos = None
                    end_pos = None
                    orig_answer_text = None

                    if is_training:
                        if len(qa["answers"]) != 1:
                            raise ValueError(
                                "For training, each question should have exactly 1 answer."
                            )

                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        doc_tokens = [
                            paragraph_text[:answer_offset], paragraph_text[
                                answer_offset:answer_offset + answer_length],
                            paragraph_text[answer_offset + answer_length:]
                        ]

                        start_pos = 1
                        end_pos = 1

                        actual_text = " ".join(doc_tokens[start_pos:(end_pos +
                                                                     1)])
                        if actual_text.find(orig_answer_text) == -1:
                            log.info("Could not find answer: '%s' vs. '%s'",
                                     actual_text, orig_answer_text)
                            continue
                    else:
                        doc_tokens = _tokenize_chinese_chars(paragraph_text)

                    example = Example(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_pos,
                        end_position=end_pos)
                    examples.append(example)

    return examples


def convert_example_to_features(examples,
                                max_seq_length,
                                tokenizer,
                                is_training,
                                doc_stride=128,
                                max_query_length=64):
    """convert example to feature"""
    features = []
    unique_id = 1000000000

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        #log.info(orig_to_tok_index, example.start_position)

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position +
                                                     1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position,
                tokenizer, example.orig_answer_text)

        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        _DocSpan = namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            text_type_ids = []
            tokens.append("[CLS]")
            text_type_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                text_type_ids.append(0)
            tokens.append("[SEP]")
            text_type_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[
                    split_token_index]

                is_max_context = _check_is_max_context(
                    doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                text_type_ids.append(1)
            tokens.append("[SEP]")
            text_type_ids.append(1)

            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            position_ids = list(range(len(token_ids)))
            start_position = None
            end_position = None
            if is_training:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            feature = Feature(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                token_ids=token_ids,
                position_ids=position_ids,
                text_type_ids=text_type_ids,
                start_position=start_position,
                end_position=end_position)
            features.append(feature)

            unique_id += 1

    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()

    from ernie.tokenizing_ernie import ErnieTokenizer
    tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
    examples = read_files(args.input, True)
    features = convert_example_to_features(examples, 512, tokenizer, True)
    log.debug(len(examples))
    log.debug(len(features))
