#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""data reader for text classification tasks"""

import os
import csv
import numpy as np
import copy
from collections import namedtuple
from model import tokenization
from reader.batching import pad_batch_data


class RegressionReader(object):
    """RegressionReader"""
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id

        self.max_seq_len = args.max_seq_len
        self.in_tokens = args.in_tokens

        self.random_seed = 0
        self.global_rng = np.random.RandomState(self.random_seed)

        self.trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        self.trainer_nums = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))

        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            headers = next(reader)
            text_indices = [
                index for index, h in enumerate(headers) if h != "label"
            ]
            Example = namedtuple('Example', headers)

            examples = []
            for line in reader:
                example = Example(*line)
                examples.append(example)
            return examples

    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_labels = [record.label_id for record in batch_records]
        batch_labels = np.array(batch_labels).astype('float32').reshape([-1, 1])

        if batch_records[0].qid:
            batch_qids = [record.qid for record in batch_records]
            batch_qids = np.array(batch_qids).astype('int64').reshape([-1, 1])
        else:
            batch_qids = np.array([]).astype('int64').reshape([-1, 1])

        # padding
        padded_token_ids, input_mask = pad_batch_data(
            batch_token_ids, pretraining_task='nlu', pad_idx=self.pad_id, return_input_mask=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pretraining_task='nlu', pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pretraining_task='nlu', pad_idx=self.pad_id)
        input_mask = np.matmul(input_mask, np.transpose(input_mask, (0, 2, 1)))

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            input_mask, batch_labels, batch_qids
        ]

        return return_list

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
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
        position_ids = list(range(2, len(token_ids) + 2))
        label_id = example.label

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'label_id', 'qid'])

        qid = None
        if "qid" in example._fields:
            qid = example.qid

        record = Record(
            token_ids=token_ids,
            text_type_ids=text_type_ids,
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
        """get_num_examples"""
        examples = self._read_tsv(input_file)
        return len(examples)

    def data_generator(self,
                       input_file,
                       batch_size,
                       epoch,
                       dev_count=1,
                       shuffle=True,
                       phase=None):
        """data_generator"""
        examples = self._read_tsv(input_file)

        def wrapper():
            """wrapper"""
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
                    trainer_id = 0
                    assert dev_count == 1, "only supports 1 GPU while prediction"
                current_examples = copy.deepcopy(examples)
                if shuffle:
                    self.global_rng.shuffle(current_examples)
                for batch_data in self._prepare_batch_data(
                        current_examples, batch_size, phase=phase):
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == dev_count:
                        yield all_dev_batches[trainer_id]
                        all_dev_batches = []
                if phase != "train" and self.trainer_id < len(all_dev_batches):
                    yield all_dev_batches[self.trainer_id]
        return wrapper


if __name__ == '__main__':
    pass
