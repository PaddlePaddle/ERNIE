#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import sys
import six
import json
import logging

import numpy as np

from io import open
from collections import namedtuple

from reader import tokenization
from reader.batching import pad_batch_data

log = logging.getLogger(__name__)

if six.PY3:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def csv_reader(fd, delimiter='\t'):
    def gen():
        for i in fd:
            yield i.rstrip('\n').split(delimiter)
    return gen()

class BaseReader(object):
    def __init__(self,
                 vocab_path,
                 piece_model_path,
                 max_seq_len=512,
                 in_tokens=False,
                 tokenizer="FullTokenizer",
                 label_map_config=None,
                 is_inference=False,
                 random_seed=1234):
        self.tokenizer = tokenization.FullTokenizer(
                vocab_file=vocab_path,
                model_file=piece_model_path)
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]

        self.max_seq_len = max_seq_len
        self.in_tokens = in_tokens

        if label_map_config:
            with open(label_map_config) as fp:
                self.label_map = json.load(fp)
        else:
            self.label_map = None

        self.is_inference = is_inference
        np.random.seed(random_seed)
        
        self.trainer_id = int(os.getenv(
                "PADDLE_TRAINER_ID", "0"))

        self.current_example = 0
        self.current_epoch = 0
    
    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf8") as f:
            reader = csv_reader(f)
            headers = next(reader)
            Example = namedtuple("Example", headers)

            examples = []
            for line in reader:
                example = Example(*line)
                examples.append(example)
            return examples

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        """Converts a single `Example` into a single `Record`."""

        text_a = tokenization.convert_to_unicode(example.text_a)
        tokens_a = tokenizer.tokenize(text_a)
        
        tokens_b = None
        if "text_b" in example._fields:
            text_b = tokenization.convert_to_unicode(example.text_b)
            tokens_b = tokenizer.tokenize(text_b)

        if tokens_b:
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 4)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        tokens.append("[CLS]")
        for token in tokens_a:
            tokens.append(token)
        tokens.append("[SEP]")

        if tokens_b:
            tokens.append("[SEP]")
            for token in tokens_b:
                tokens.append(token)
            tokens.append("[SEP]")
        
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(2, len(token_ids) + 2))
        
        if self.is_inference:
            Record = namedtuple("Record",
                    ["token_ids", "position_ids"])
            record = Record(
                    token_ids=token_ids,
                    position_ids=position_ids)
        else:
            if self.label_map:
                label_id = self.label_map[example.label]
            else:
                label_id = example.label

            Record = namedtuple("Record", [
                "token_ids", "position_ids", "label_id", "qid"
            ])

            qid = None
            if "qid" in example._fields:
                qid = example.qid

            record = Record(
               token_ids=token_ids,
               position_ids=position_ids,
               label_id=label_id,
               qid=qid)
        return record

    def _prepare_batch_data(self, examples, batch_size, phase=None):
        """generate batch records"""
        batch_records, max_len = [], 0
        for index, example in enumerate(examples):
            if phase == "train":
                self.current_example = index
            record = self._convert_example_to_record(example, self.max_seq_len, 
                    self.tokenizer)
            max_len = max(max_len, len(record.token_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                yield self._pad_batch_records(batch_records)
                batch_records, max_len = [record], len(record.token_ids)

        if batch_records:
            yield self._pad_batch_records(batch_records)

    def get_num_examples(self, input_file):
        examples = self._read_tsv(input_file)
        return len(examples)

    def data_generator(self,
                       input_file,
                       batch_size,
                       epoch,
                       dev_count=1,
                       shuffle=True,
                       phase=None):
        examples = self._read_tsv(input_file)

        def wrapper():
            all_dev_batches = []
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index

                if shuffle:
                    np.random.shuffle(examples)

                for batch_data in self._prepare_batch_data(
                        examples, batch_size, phase=phase):
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == dev_count:
                        yield all_dev_batches[self.trainer_id]
                        all_dev_batches = []
                if phase != "train":
                    if self.trainer_id < len(all_dev_batches):
                        yield all_dev_batches[self.trainer_id]
        return wrapper


class ClassifyReader(BaseReader):
    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]

        if not self.is_inference:
            batch_labels = [record.label_id for record in batch_records]
            batch_labels = np.array(batch_labels).astype("int64").reshape(
                    [-1, 1])

            if batch_records[0].qid is not None:
                batch_qids = [record.qid for record in batch_records]
                batch_qids = np.array(batch_qids).astype("int64").reshape(
                        [-1, 1])
            else:
                batch_qids = np.array([]).astype("int64").reshape([-1, 1])

        padded_token_ids, input_mask = pad_batch_data(
            batch_token_ids, pad_idx=self.pad_id, return_input_mask=True)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)
        
        return_list = [
            padded_token_ids, padded_position_ids, input_mask, 
        ]
        if not self.is_inference:
            return_list += [batch_labels, batch_qids]
        
        return return_list


class SequenceLabelReader(BaseReader):
    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_label_ids = [record.label_ids for record in batch_records]

        padded_token_ids, input_mask, batch_seq_lens = pad_batch_data(
            batch_token_ids, 
            pad_idx=self.pad_id, 
            return_input_mask=True, 
            return_seq_lens=True)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)
        padded_label_ids = pad_batch_data(
            batch_label_ids, pad_idx=len(self.label_map) - 1)

        return_list = [
            padded_token_ids, padded_position_ids, input_mask, 
            padded_label_ids, batch_seq_lens
        ]
        return return_list

    def _reseg_token_label(self, tokens, labels, tokenizer):
        assert len(tokens) == len(labels)
        ret_tokens = []
        ret_labels = []
        for token, label in zip(tokens, labels):
            sub_token = tokenizer.tokenize(token)
            if len(sub_token) == 0:
                continue
            ret_tokens.extend(sub_token)
            if len(sub_token) == 1:
                ret_labels.append(label)
                continue

            if label == "O" or label.startswith("I-"):
                ret_labels.extend([label] * len(sub_token))
            elif label.startswith("B-"):
                i_label = "I-" + label[2:]
                ret_labels.extend([label] + [i_label] * (len(sub_token) - 1))
            elif label.startswith("S-"):
                b_laebl = "B-" + label[2:]
                e_label = "E-" + label[2:]
                i_label = "I-" + label[2:]
                ret_labels.extend([b_laebl] + [i_label] * (len(sub_token) - 2) + [e_label])
            elif label.startswith("E-"):
                i_label = "I-" + label[2:]
                ret_labels.extend([i_label] * (len(sub_token) - 1) + [label])

        assert len(ret_tokens) == len(ret_labels)
        return ret_tokens, ret_labels

    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        tokens = tokenization.convert_to_unicode(example.text_a).split("\2")
        labels = tokenization.convert_to_unicode(example.label).split("\2")
        tokens, labels = self._reseg_token_label(tokens, labels, tokenizer)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(2, len(token_ids) + 2))
        no_entity_id = len(self.label_map) - 1
        label_ids = [no_entity_id] + [
            self.label_map[label] for label in labels
        ] + [no_entity_id]

        Record = namedtuple(
            "Record",
            ["token_ids", "position_ids", "label_ids"])
        record = Record(
            token_ids=token_ids,
            position_ids=position_ids,
            label_ids=label_ids)
        return record


class MRCReader(BaseReader):
    def __init__(self,
                 vocab_path,
                 piece_model_path,
                 max_seq_len=512,
                 in_tokens=False,
                 tokenizer="FullTokenizer",
                 label_map_config=None,
                 doc_stride=128,
                 max_query_length=64,
                 is_inference=False,
                 random_seed=1234):
        self.tokenizer = tokenization.FullTokenizer(
                vocab_file=vocab_path,
                model_file=piece_model_path)
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"] 
        
        self.max_seq_len = max_seq_len
        self.in_tokens = in_tokens
 
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length

        self.examples = {}
        self.features = {}

        np.random.seed(random_seed)

        self.trainer_id = int(os.getenv(
                "PADDLE_TRAINER_ID", "0"))

        self.current_example = 0
        self.current_epoch = 0

    def _read_json(self, input_file, is_training):
        examples = []
        with open(input_file, "r") as f:
            input_data = json.load(f)["data"]
            for entry in input_data:
                for paragraph in entry["paragraphs"]:
                    paragraph_text = paragraph["context"]
                    doc_tokens = []
                    char_to_word_offset = []
                    prev_is_whitespace = True 
                    for char in paragraph_text:
                        if tokenization._is_whitespace(char):
                            prev_is_whitespace = True
                        else:
                            if prev_is_whitespace:
                                doc_tokens.append(char)
                            else:
                                doc_tokens[-1] += char
                            prev_is_whitespace = False
                        char_to_word_offset.append(len(doc_tokens) - 1)
                    
                    for qa in paragraph["qas"]:
                        qas_id = qa["id"]
                        question_text = qa["question"]
                        start_position = None
                        end_position = None
                        orig_answer_text = None
                        version_2_with_negative = False
                        is_impossible = False
                        if is_training:
                            if version_2_with_negative:
                                is_impossible = qa["is_impossible"]
                            if (len(qa["answers"]) != 1) and (not is_impossible):
                                raise ValueError(
                                    "For training, each question should have exactly 1 answer."
                                )
                            if not is_impossible:
                                answer = qa["answers"][0]
                                orig_answer_text = answer["text"]
                                answer_offset = answer["answer_start"]
                                answer_length = len(orig_answer_text) 
                                start_position = char_to_word_offset[answer_offset]
                                end_position = char_to_word_offset[answer_offset + 
                                                   answer_length - 1]
                                actual_text = " ".join(doc_tokens[start_position:(
                                                   end_position + 1)])
                                cleaned_answer_text = " ".join(
                                    tokenization.whitespace_tokenize(orig_answer_text))

                                if actual_text.find(cleaned_answer_text) == -1:
                                    log.info("Could not find answer: '%s' vs. '%s'",
                                             actual_text, cleaned_answer_text)
                                    continue
                            else:
                                start_position = 0
                                end_position = 0
                                orig_answer_text = ""
                        else:
                            start_position = 0
                            end_position = 0
                            orig_answer_text = ""

                        Example = namedtuple("Example",
                                  ["qas_id", "question_text", "doc_tokens", "orig_answer_text",
                                   "start_position", "end_position", "is_impossible"])
                        example = Example(
                            qas_id=qas_id,
                            question_text=question_text,
                            doc_tokens=doc_tokens,
                            orig_answer_text=orig_answer_text,
                            start_position=start_position,
                            end_position=end_position,
                            is_impossible=is_impossible)
                        examples.append(example)

        return examples
        
    def _improve_answer_span(self, doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
        tok_answer_text = " ".join([bytes.decode(token) if isinstance(token, bytes) else token for token in tokenizer.tokenize(orig_answer_text)])
        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)
  
        return (input_start, input_end)

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
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

    def _convert_example_to_feature(self, examples, max_seq_length, tokenizer, is_training):
        Feature = namedtuple("Feature", ["unique_id", "example_index", "doc_span_index",
                    "tokens", "token_to_orig_map", "token_is_max_context",
                    "token_ids", "position_ids",
                    "start_position", "end_position", "is_impossible", "label"])
        features = []
        unique_id = 1000000000

        for (example_index, example) in enumerate(examples):
            query_tokens = tokenizer.tokenize(example.question_text)

            if len(query_tokens) > self.max_query_length:
                query_tokens = query_tokens[0:self.max_query_length]
            
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
                 orig_to_tok_index.append(len(all_doc_tokens))
                 sub_tokens = tokenizer.tokenize(token)
                 sub_tokens = [bytes.decode(sub_token) if isinstance(sub_token, bytes) else sub_token for sub_token in sub_tokens]
                 for sub_token in sub_tokens:
                     tok_to_orig_index.append(i)
                     all_doc_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None
            if is_training and example.is_impossible:
                tok_start_position = -1
                tok_end_position = -1

            if is_training and not example.is_impossible:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = self._improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position,
                    tokenizer, example.orig_answer_text)

            max_tokens_for_doc = max_seq_length - len(query_tokens) - 4
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
                start_offset += min(length, self.doc_stride)
                
            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                tokens.append("[CLS]")
                for token in query_tokens:
                    tokens.append(token)
                tokens.append("[SEP]")

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[
                        split_token_index]
                
                    is_max_context = self._check_is_max_context(
                        doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                tokens.append("[SEP]")

                token_ids = tokenizer.convert_tokens_to_ids(tokens) 
                position_ids = list(range(2, len(token_ids) + 2))

                start_position = None
                end_position = None
                if is_training and not example.is_impossible:
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
                label = 0
                if is_training and example.is_impossible:
                    start_position = 0
                    end_position = 0
                    label = 1

                feature = Feature(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    token_ids=token_ids,
                    position_ids=position_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible,
                    label=label)
                features.append(feature)

                unique_id += 1

        return features

    def _prepare_batch_data(self, records, batch_size, phase=None):
        """generate batch records"""
        batch_records, max_len = [], 0

        for index, record in enumerate(records):
            if phase == "train":
                self.current_example = index
            max_len = max(max_len, len(record.token_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                yield self._pad_batch_records(batch_records, phase=="train")
                batch_records, max_len = [record], len(record.token_ids)
        
        if batch_records:
            yield self._pad_batch_records(batch_records, phase=="train")

    def _pad_batch_records(self, batch_records, is_training):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_labels = [record.label for record in batch_records] 
        if is_training:
            batch_start_position = [record.start_position for record in batch_records]
            batch_end_position = [record.end_position for record in batch_records]
            batch_start_position = np.array(batch_start_position).astype("int64").reshape([-1, 1])
            batch_end_position = np.array(batch_end_position).astype("int64").reshape([-1, 1])

        else:
            batch_size = len(batch_token_ids)
            batch_start_position = np.zeros(shape=[batch_size, 1], dtype="int64")
            batch_end_position = np.zeros(shape=[batch_size, 1], dtype="int64")

        batch_unique_ids = [record.unique_id for record in batch_records]
        batch_unique_ids = np.array(batch_unique_ids).astype("int64").reshape([-1, 1])
        batch_labels = np.array(batch_labels).astype("int64").reshape([-1, 1])

        padded_token_ids, input_mask = pad_batch_data(
            batch_token_ids, pad_idx=self.pad_id, return_input_mask=True)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id) 
        
        return_list = [
            padded_token_ids, padded_position_ids, input_mask,
            batch_start_position, batch_end_position, batch_unique_ids, batch_labels
        ]

        return return_list

    def get_num_examples(self, phase):
        return len(self.features[phase])

    def get_features(self, phase):
        return self.features[phase]

    def get_examples(self, phase):
        return self.examples[phase]

    def data_generator(self,
                       input_file,
                       batch_size,
                       epoch,
                       dev_count=1,
                       shuffle=True,
                       phase=None):

        examples = self.examples.get(phase, None)         
        features = self.features.get(phase, None)

        examples = self._read_json(input_file, phase=="train")
        features = self._convert_example_to_feature(examples, self.max_seq_len,
                self.tokenizer, phase=="train")
        self.examples[phase] = examples
        self.features[phase] = features

        def wrapper():
            all_dev_batches = []
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                
                if shuffle:
                    if shuffle:
                        np.random.seed(epoch_index)
                        np.random.shuffle(examples)

                for batch_data in self._prepare_batch_data(
                     features, batch_size, phase=phase):
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == dev_count:
                        yield all_dev_batches[self.trainer_id]
                        all_dev_batches = []
                if phase != "train":
                    if self.trainer_id < len(all_dev_batches):
                        yield all_dev_batches[self.trainer_id]
        return wrapper                                        


class ExtractEmbeddingReader(BaseReader):
    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]

        padded_token_ids, input_mask, seq_lens = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            return_input_mask=True,
            return_seq_lens=True)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)
        return_list = [
                padded_token_ids, padded_position_ids, 
                input_mask, seq_lens]
        
        return return_list


if __name__ == '__main__':
    pass
