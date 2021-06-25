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

import os
import six
import csv
import json
import numpy as np
from collections import namedtuple

import reader.tokenization as tokenization
from reader.batching import pad_batch_data, _get_rel_pos_scaler

class BaseReader(object):
    def __init__(self,
                 vocab_path,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 in_tokens=False,
                 tokenizer="FullTokenizer", 
                 is_classify=True, 
                 is_regression=False,
                 eval_span=False):
        self.max_seq_len = max_seq_len
        self.tokenizer = getattr(tokenization, tokenizer)(
                vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.rel_pos = _get_rel_pos_scaler(512)
        
        self.in_tokens = in_tokens
        self.is_classify = is_classify
        self.is_regression = is_regression
        self.eval_span = eval_span

        
        self.random_seed = int(os.getenv("RANDSEED"))
        print("reader", self.random_seed)
        self.global_rng = np.random.RandomState(self.random_seed)
        
        self.trainer_id = 0
        self.trainer_nums = 1
        if os.getenv("PADDLE_TRAINER_ID"):
            self.trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        if os.getenv("PADDLE_NODES_NUM"):
            self.trainer_nums = int(os.getenv("PADDLE_TRAINERS_NUM"))

        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

        if label_map_config:
            with open(label_map_config) as f:
                self.label_map = json.load(f)
        else:
            self.label_map = None
    
    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            headers = next(reader)
            Example = namedtuple('Example', headers)

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

        #text_a = tokenization.convert_to_unicode(example.text_a)
        text_a = example.text_a.decode('utf8') if six.PY2 else example.text_a
        tokens_a = tokenizer.tokenize(text_a)
        if len(tokens_a) > 510:
            tokens_b = tokens_a[-381:]
            tokens_a = tokens_a[:128]

        tokens_b = None
        has_text_b = False
        if isinstance(example, dict):
            has_text_b = "text_b" in example.keys()
        else:
            has_text_b = "text_b" in example._fields

        if has_text_b:
            #text_b = tokenization.convert_to_unicode(example.text_b)
            text_b = example.text_b.decode('utf8') if six.PY2 else example.text_b
            tokens_b = tokenizer.tokenize(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        text_type_ids = []
        tokens.append("[CLS]")
        text_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            text_type_ids.append(0)
        tokens.append("[SEP]")
        text_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                text_type_ids.append(1)
            tokens.append("[SEP]")
            text_type_ids.append(1)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(len(token_ids)))
        task_ids = [0] * len(token_ids)

        if self.label_map:
            label_id = self.label_map[example.label]
        else:
            try:
                label_id = example.labels
            except:
                label_id = example.label

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'task_ids', 'label_id', 'qid'])

        qid = None
        if "qid" in example._fields:
            qid = example.qid

        record = Record(
            token_ids=token_ids,
            text_type_ids=text_type_ids,
            position_ids=position_ids,
            task_ids=task_ids,
            label_id=label_id,
            qid=qid)
        return record

    def _stride(self, text, max_len=510):
        spans = []
        index = 0

        if len(text) > max_len:
            spans.append(text[:128] + text[-382:])
            spans.append(text[:max_len])
            spans.append(text[-max_len:])
        else:
            spans.append(text)

        return spans
    
    def _convert_example_to_record_spans(self, example, max_seq_length, tokenizer, qid, max_len=512):
        """Converts a single `Example` into a single `Record`."""
        if self.label_map:
            label_id = self.label_map[example.label]
        else:
            label_id = example.label
        records = [] 
        text_a = example.text_a.decode('utf8') if six.PY2 else example.text_a
        tokens_a = tokenizer.tokenize(text_a)
        spans = self._stride(tokens_a, max_len-2)
        for span in spans:
            tokens = []
            text_type_ids = []
            tokens.append("[CLS]")
            text_type_ids.append(0)
            for token in span:
                tokens.append(token)
                text_type_ids.append(0)
            tokens.append("[SEP]")
            text_type_ids.append(0)
            
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            position_ids = list(range(len(token_ids)))
            task_ids = [0] * len(token_ids)
            
            Record = namedtuple(
                'Record',
                ['token_ids', 'text_type_ids', 'position_ids', 'task_ids', 'label_id', 'qid'])

            records.append(Record(
                token_ids=token_ids,
                text_type_ids=text_type_ids,
                position_ids=position_ids,
                task_ids=task_ids,
                label_id=label_id,
                qid=qid))
        return records

    def _prepare_batch_data(self, examples, batch_size, phase=None):
        """generate batch records"""
        batch_records, max_len = [], 0
        for index, example in enumerate(examples):
            if phase == "train":
                self.current_example = index
            if not self.eval_span or phase == "train":
                records = [self._convert_example_to_record(example, self.max_seq_len,
                                                        self.tokenizer)]
            else:
                records = self._convert_example_to_record_spans(example, self.max_seq_len,
                                                         self.tokenizer, index)
            for record in records:
                if isinstance(record.token_ids[0], list):
                    max_len = max(max_len, max(map(lambda x:len(x), record.token_ids)))
                else:
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
            trainer_id = 0
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                    self.random_seed = epoch_index
                    self.global_rng = np.random.RandomState(self.random_seed)
                    trainer_id = self.trainer_id
                else:
                    #dev/test
                    #assert dev_count == 1, "only supports 1 GPU prediction"
                    trainer_id = self.trainer_id

                current_examples = [ins for ins in examples]
                if shuffle:
                    self.global_rng.shuffle(current_examples)
                #if phase == "train" and self.trainer_nums > 1:
                #    current_examples = [ins for index, ins in enumerate(current_examples) 
                #                if index % self.trainer_nums == self.trainer_id]
                for batch_data in self._prepare_batch_data(
                        current_examples, batch_size, phase=phase):
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == dev_count:
                        #trick: handle batch inconsistency caused by data sharding for each trainer
                        yield all_dev_batches[trainer_id]
                        all_dev_batches = []
                if phase != "train":
                    if trainer_id < len(all_dev_batches):
                        yield all_dev_batches[trainer_id]

        return wrapper


class ClassifyReader(BaseReader):
    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            headers = next(reader)
            text_indices = [
                index for index, h in enumerate(headers) if h != "label"
            ]
            label_indices = [
                index for index, h in enumerate(headers) if h == "label"
            ]
            Example = namedtuple('Example', headers)

            examples = []
            for line in reader:
                for index, text in enumerate(line):
                    if index in text_indices:
                        line[index] = text #.replace(' ', '')

                example = Example(*line)
                examples.append(example)
            return examples

    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_task_ids = [record.task_ids for record in batch_records]
        batch_labels = [record.label_id for record in batch_records]
        if self.is_classify:
            batch_labels = np.array(batch_labels).astype("int64").reshape([-1, 1])
        elif self.is_regression:
            batch_labels = np.array(batch_labels).astype("float32").reshape([-1, 1])
        if batch_records[0].qid:
            batch_qids = [record.qid for record in batch_records]
            batch_qids = np.array(batch_qids).astype("int64").reshape([-1, 1])
        else:
            batch_qids = np.array([]).astype("int64").reshape([-1, 1])

        # padding
        padded_token_ids, input_mask = pad_batch_data(
            batch_token_ids, pad_idx=self.pad_id, return_input_mask=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)
        padded_task_ids = pad_batch_data(
            batch_task_ids, self.pad_id)#pad_idx=self.pad_id)
        if padded_token_ids.shape[1] > 512:
            rel_pos_scaler = _get_rel_pos_scaler(padded_token_ids.shape[1])
        else:
            rel_pos_scaler = self.rel_pos[:padded_token_ids.shape[1], :padded_token_ids.shape[1], :]
        rel_pos_scaler = np.array([rel_pos_scaler for i in range(padded_token_ids.shape[0])]).astype("int64")
        
        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            padded_task_ids, input_mask, batch_labels, batch_qids, rel_pos_scaler

        ]

        return return_list


class ClassifyReaderRace(ClassifyReader):
    def _convert_example_to_record_race(self, example, max_seq_length, tokenizer):
        """Converts a single `Example` into a single `Record`."""

        #text_a = tokenization.convert_to_unicode(example.text_a)
        total_len = 0
        if six.PY3:
            text_p = example.text_a
            text_q = example.text_b
            text_a = example.text_c.split("")
        else:
            text_p = example.text_a.decode('utf8')
            text_q = example.text_b.decode('utf8')
            text_a = example.text_c.decode('utf8').split("")
        tokens_p = tokenizer.tokenize(text_p)
        assert len(text_a) == 4
        tokens_all = []
        position_all = []
        seg_all = []
        task_all = []
        for i in range(4):
            if "_" in text_q:
                text_qa = text_q.replace("_", text_a[i])
            else:
                text_qa = " ".join([text_q, text_a[i]])
            tokens_qa = tokenizer.tokenize(text_qa)
            tokens_p = tokens_p[:max_seq_length - len(tokens_qa) - 3]
            tokens = []
            text_type_ids = []
            tokens.append("[CLS]")
            text_type_ids.append(0)
            for token in tokens_qa:
                tokens.append(token)
                text_type_ids.append(0)
            tokens.append("[SEP]")
            text_type_ids.append(0)

            for token in tokens_p:
                tokens.append(token)
                text_type_ids.append(1)
            tokens.append("[SEP]")
            text_type_ids.append(1)
            tokens_id = tokenizer.convert_tokens_to_ids(tokens)
            tokens_all.append(tokens_id)
            position_all.append(list(range(len(tokens_id))))
            task_all.append([0] * len(tokens_id))
            seg_all.append(text_type_ids)

        if self.label_map:
            label_id = self.label_map[example.label]
        else:
            label_id = example.labels

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'task_ids', 'label_id', 'qid'])

        qid = None
        if "qid" in example._fields:
            qid = example.qid

        record = Record(
            token_ids=tokens_all,
            text_type_ids=seg_all,
            position_ids=position_all,
            task_ids=task_all,
            label_id=label_id,
            qid=qid)
        return record
    
    def _prepare_batch_data(self, examples, batch_size, phase=None):
        """generate batch records"""
        batch_records, max_len = [], 0
        for index, example in enumerate(examples):
            if phase == "train":
                self.current_example = index
            record = self._convert_example_to_record_race(example, self.max_seq_len,
                                                     self.tokenizer)
            if isinstance(record.token_ids[0], list):
                max_len = max(max_len, max(map(lambda x:len(x), record.token_ids)))
            else:
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
    
    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_task_ids = [record.task_ids for record in batch_records]
        batch_labels = [record.label_id for record in batch_records]
        label_all = []
        for l in batch_labels:
            tmp = [0, 0, 0, 0]
            tmp[int(l)] = 1
            label_all.extend(tmp)
        batch_labels = np.array(batch_labels).astype("int64").reshape([-1, 1])
        batch_labels_all = np.array(label_all).astype("float32").reshape([-1, 1])

        if batch_records[0].qid:
            batch_qids = [record.qid for record in batch_records]
            batch_qids = np.array(batch_qids).astype("int64").reshape([-1, 1])
        else:
            batch_qids = np.array([]).astype("int64").reshape([-1, 1])

        # padding
        batch_token_ids = sum(batch_token_ids, [])
        batch_text_type_ids = sum(batch_text_type_ids, [])
        batch_position_ids = sum(batch_position_ids, [])
        batch_task_ids = sum(batch_task_ids, [])
        padded_token_ids, input_mask = pad_batch_data(
            batch_token_ids, pad_idx=self.pad_id, return_input_mask=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)
        padded_task_ids = pad_batch_data(
            batch_task_ids, self.pad_id)#pad_idx=self.pad_id)
        if padded_token_ids.shape[1] > 512:
            rel_pos_scaler = _get_rel_pos_scaler(padded_token_ids.shape[1])
        else:
            rel_pos_scaler = self.rel_pos[:padded_token_ids.shape[1], :padded_token_ids.shape[1], :]
        rel_pos_scaler = np.array([rel_pos_scaler for i in range(padded_token_ids.shape[0])]).astype("int64")
        
        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            padded_task_ids, input_mask, batch_labels, batch_qids, rel_pos_scaler, batch_labels_all

        ]

        return return_list


class SequenceLabelReader(BaseReader):
    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_task_ids = [record.task_ids for record in batch_records]
        batch_label_ids = [record.label_ids for record in batch_records]

        # padding
        padded_token_ids, input_mask, batch_seq_lens = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            return_input_mask=True,
            return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)
        padded_task_ids = pad_batch_data(
            batch_task_ids, 0)#pad_idx=self.pad_id)
        padded_label_ids = pad_batch_data(
            batch_label_ids, pad_idx=len(self.label_map) - 1)

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            padded_task_ids, input_mask, padded_label_ids, batch_seq_lens
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
            ret_labels.append(label)
            if len(sub_token) < 2:
                continue
            sub_label = label
            if label.startswith("B-"):
                sub_label = "I-" + label[2:]
            ret_labels.extend([sub_label] * (len(sub_token) - 1))

        assert len(ret_tokens) == len(ret_labels)
        return ret_tokens, ret_labels

    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        tokens = tokenization.convert_to_unicode(example.text_a).split(u"")
        labels = tokenization.convert_to_unicode(example.label).split(u"")
        tokens, labels = self._reseg_token_label(tokens, labels, tokenizer)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(len(token_ids)))
        text_type_ids = [0] * len(token_ids)
        task_ids = [0] * len(token_ids)
        no_entity_id = len(self.label_map) - 1
        label_ids = [no_entity_id] + [
            self.label_map[label] for label in labels
        ] + [no_entity_id]

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'task_ids', 'label_ids'])
        record = Record(
            token_ids=token_ids,
            text_type_ids=text_type_ids,
            position_ids=position_ids,
            task_ids=task_ids,
            label_ids=label_ids)
        return record


class ExtractEmbeddingReader(BaseReader):
    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_task_ids = [record.task_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]

        # padding
        padded_token_ids, input_mask, seq_lens = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            return_input_mask=True,
            return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)
        padded_task_ids = pad_batch_data(
            batch_task_ids, pad_idx=0)#self.pad_id)

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            padded_task_ids, input_mask, seq_lens
        ]

        return return_list


class MRCReader(BaseReader):
    def __init__(self,
                 vocab_path,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 in_tokens=False,
                 random_seed=None,
                 tokenizer="FullTokenizer",
                 is_classify=True,
                 is_regression=False,
                 for_cn=True,
                 task_id=0,
                 doc_stride=128,
                 max_query_length=64,
                 version_2_with_negative=False):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.in_tokens = in_tokens
        self.for_cn = for_cn
        self.task_id = task_id
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.examples = {}
        self.features = {}
        self.rel_pos = _get_rel_pos_scaler(512)
        self.version_2_with_negative = version_2_with_negative

        #if random_seed is not None:
        #    np.random.seed(random_seed)
        
        self.trainer_id = 0
        self.trainer_nums = 1
        if os.getenv("PADDLE_TRAINER_ID"):
            self.trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        if os.getenv("PADDLE_NODES_NUM"):
            self.trainer_nums = int(os.getenv("PADDLE_TRAINERS_NUM"))

        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0
        
    def is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

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
                    for c in paragraph_text:
                        if self.is_whitespace(c):
                            prev_is_whitespace = True
                        else:
                            if prev_is_whitespace:
                                doc_tokens.append(c)
                            else:
                                doc_tokens[-1] += c
                            prev_is_whitespace = False
                        char_to_word_offset.append(len(doc_tokens) - 1)

                    for qa in paragraph["qas"]:
                        qas_id = qa["id"]
                        question_text = qa["question"]
                        start_position = None
                        end_position = None
                        orig_answer_text = None
                        is_impossible = False
                        if is_training:
                            if self.version_2_with_negative:
                                is_impossible = qa["is_impossible"]
                            if len(qa["answers"]) != 1 and (not is_impossible):
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
                                    print("Could not find answer: '%s' vs. '%s'",
                                          actual_text, cleaned_answer_text)
                                    continue
                            else:
                                start_position = 0
                                end_position = 0
                                orig_answer_text = ""
                        Example = namedtuple('Example',
                                ['qas_id', 'question_text', 'doc_tokens', 'orig_answer_text',
                                 'start_position', 'end_position', 'is_impossible'])
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
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

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
                    "token_ids", "position_ids", "text_type_ids",
                    "start_position", "end_position", "is_impossible"])
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
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)
            #print(orig_to_tok_index, example.start_position)

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
                start_offset += min(length, self.doc_stride)

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

                    is_max_context = self._check_is_max_context(
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
                if is_training and example.is_impossible:
                    start_position = 0
                    end_position = 0

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
                    end_position=end_position,
                    is_impossible=example.is_impossible)
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
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
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

        # padding
        padded_token_ids, input_mask = pad_batch_data(
            batch_token_ids, pad_idx=self.pad_id, return_input_mask=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)
        padded_task_ids = np.ones_like(padded_token_ids, dtype="int64") * self.task_id
        if padded_token_ids.shape[1] > 512:
            rel_pos_scaler = _get_rel_pos_scaler(padded_token_ids.shape[1])
        else:
            rel_pos_scaler = self.rel_pos[:padded_token_ids.shape[1], :padded_token_ids.shape[1], :]
        rel_pos_scaler = np.array([rel_pos_scaler for i in range(padded_token_ids.shape[0])]).astype("int64")

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids, padded_task_ids,
            input_mask, batch_start_position, batch_end_position, batch_unique_ids, rel_pos_scaler
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
        if not examples:
            examples = self._read_json(input_file, phase=="train")
            features = self._convert_example_to_feature(examples, self.max_seq_len,
                    self.tokenizer, phase=="train")
            self.examples[phase] = examples
            self.features[phase] = features

        def wrapper():
            #features = self.features.get(phase, None)
            all_dev_batches = []
            trainer_id = 0
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                if phase == "train" and shuffle:
                    self.random_seed = epoch_index
                    self.global_rng = np.random.RandomState(self.random_seed)
                    trainer_id = self.trainer_id
                    self.global_rng.shuffle(features)
                if phase != "train":
                    trainer_id = self.trainer_id

                for batch_data in self._prepare_batch_data(
                        features, batch_size, phase=phase):
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == dev_count:
                        #for batch in all_dev_batches:
                            #yield batch
                        yield all_dev_batches[trainer_id]
                        all_dev_batches = []
                if phase != "train":
                    if trainer_id < len(all_dev_batches):
                        yield all_dev_batches[trainer_id]

        return wrapper


if __name__ == '__main__':
    data_reader = ClassifyReader(
            vocab_path="./package/vocab.txt",
            label_map_config="./package/task_data/xnli/label_map.json",
            max_seq_len=512,
            do_lower_case=True,
            in_tokens=True)
    train_data_generator = data_reader.data_generator(
            input_file="./package/task_data/xnli/train.tsv",
            batch_size=8192,
            epoch=3,
            shuffle=True,
            phase="train")
    for batch_data in train_data_generator():
        tokens, text_types, postions, tasks, masks, labels, qids = batch_data
        print(tokens.tolist())
