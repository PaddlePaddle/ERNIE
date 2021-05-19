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

import os
import csv
import json
import random
import numpy as np
from collections import namedtuple

from reader import tokenization
from reader.batching import pad_batch_data, get_related_pos

class BaseReader(object):
    def __init__(self,
                 trainer_id,
                 trainer_num,
                 vocab_path,
                 memory_len=128,
                 repeat_input=False,
                 train_all=False,
                 eval_all=False,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 in_tokens=False,
                 random_seed=None,
                 tokenizer="FullTokenizer",
                 is_zh=True):
        self.is_zh = is_zh
        self.trainer_id = trainer_id
        self.trainer_num = trainer_num
        self.max_seq_len = max_seq_len
        self.memory_len = memory_len
        if tokenizer == 'BPETokenizer':
            self.tokenizer = getattr(tokenization, tokenizer)(vocab_file=vocab_path)
            self.vocab = self.tokenizer.vocabulary.vocab_dict
        else:            
            self.tokenizer = getattr(tokenization, tokenizer)(
                    vocab_file=vocab_path, do_lower_case=do_lower_case)
            self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.in_tokens = in_tokens
        self.repeat_input = repeat_input
        self.train_all = train_all
        self.eval_all = eval_all

        if random_seed is None:
            random_seed = 12345
        self.rng = random.Random(random_seed)

        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

        if label_map_config:
            with open(label_map_config) as f:
                self.label_map = json.load(f)
        else:
            self.label_map = None
    
    def cnt_list(self, inp):
        """cnt_list"""
        cnt = 0
        for lit in inp:
            if lit:
                cnt += 1
        return cnt

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch
    
    def get_num_examples(self, input_file):
        """get number of example"""
        # return 30000
        examples = self._read_tsv(os.path.join(input_file))
        for example in examples:
            self.num_examples += len(self._convert_to_instance(example, "train"))
        return self.num_examples
    
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

    def _prepare_batch_data(self, examples, batch_size, phase=None):
        """generate batch records"""
        batch_records, max_len, gather_idx = [], 0, []
        for index, example in enumerate(examples):
            if phase == "train":
                self.current_example += 1
            if example.cal_loss == 1:
                gather_idx.append(index % batch_size)

            max_len = max(max_len, len(example.src_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(example)
            else:
                yield self._pad_batch_records(batch_records, gather_idx, phase)
                batch_records, max_len = [example], len(example.src_ids)
                gather_idx = [index % batch_size] if example.cal_loss == 1 else []

        yield self._pad_batch_records(batch_records, gather_idx, phase)
        # if self.trainer_num == 1 or (phase != "train" and batch_records):
        #     while len(batch_records) < batch_size:
        #         batch_records.append(batch_records[-1]._replace(cal_loss=0))
        #     yield self._pad_batch_records(batch_records, gather_idx, phase)
    
    def _get_samples(self, pre_list, batch_size, is_last=False):
        """get samples"""
        if is_last:
            len_doc = [len(doc) for doc in pre_list]
            max_len_idx = len_doc.index(max(len_doc))
            dirty_sample = pre_list[max_len_idx][-1]._replace(cal_loss=0)
            for sample_list in pre_list:
                sample_list.extend([dirty_sample] * (max(len_doc) - len(sample_list)))
        
        samples = []
        min_len = min([len(doc) for doc in pre_list])
        for cnt in range(min_len):
            for idx in range(batch_size * self.trainer_num):
                sample = pre_list[idx][cnt]
                samples.append(sample)
        for idx in range(len(pre_list)):
            pre_list[idx] = pre_list[idx][min_len:]
        return samples

    def _convert_to_instance(self, example, phase, qid=None):
        "convert example to instance"   
        doc_spans = []
        _DocSpan = namedtuple("DocSpan", ["start", "length"])
        start_offset = 0
        max_tokens_for_doc = self.max_seq_len - 2 
        tokens_a = self.tokenizer.tokenize(example.text_a)
        while start_offset < len(tokens_a):
            length  = len(tokens_a) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(tokens_a):
                break
            start_offset += min(length, self.memory_len)
        
        features = []
        Feature = namedtuple("Feature", ["src_ids", "label_id", "qid", "cal_loss"])
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = tokens_a[doc_span.start: doc_span.start + doc_span.length] + ["[SEP]"] + ["[CLS]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            if "qid" in example._fields:
                qid = example.qid
            if phase != "infer":
                if self.label_map:
                    label_id = self.label_map[example.label]
                else:
                    label_id = example.label
            else:
                label_id = None
            features.append(Feature(
                src_ids=token_ids,
                label_id=label_id,
                qid=qid,
                cal_loss=1))
            
        # repeat
        if self.repeat_input:
            features_repeat = features
            if not self.train_all:
                if phase == "train":
                    features = list(map(lambda x: x._replace(cal_loss=0), features)) 
            if not self.eval_all:
                if phase == "eval" or phase == "test":
                    features = list(map(lambda x: x._replace(cal_loss=0), features))
            features = features + features_repeat
        return features

    def _create_instances(self, examples, batch_size, phase):
        """generate batch records"""
        pre_batch_list = []
        insert_idx = []
        for index, example in enumerate(examples):
            features = self._convert_to_instance(example, phase, index)
            if self.cnt_list(pre_batch_list) < batch_size * self.trainer_num:
                if insert_idx:
                    pre_batch_list[insert_idx[0]] = features
                    insert_idx.pop(0)
                else:
                    pre_batch_list.append(features)

            if self.cnt_list(pre_batch_list) == batch_size * self.trainer_num:
                assert self.cnt_list(pre_batch_list) == len(pre_batch_list), "the two value must be equal"
                assert not insert_idx, "the insert_idx must be null"
                sample_batch = self._get_samples(pre_batch_list, batch_size)

                for idx, lit in enumerate(pre_batch_list):
                    if not lit:
                        insert_idx.append(idx)

                for batch_records in self._prepare_batch_data(sample_batch, batch_size, phase):
                    yield batch_records

        if phase != "train":
            if self.cnt_list(pre_batch_list):
                pre_batch_list += [[] for _ in range(batch_size * self.trainer_num - self.cnt_list(pre_batch_list))]
                sample_batch = self._get_samples(pre_batch_list, batch_size, is_last=True)
                for batch_records in self._prepare_batch_data(sample_batch, batch_size, phase):
                    yield batch_records
    
    
    def data_generator(self,
                       input_file,
                       batch_size,
                       epoch,
                       shuffle=True,
                       phase=None):
        """date generator"""
        assert phase in ["train", "eval", "test", "infer"], "phase should be one of the four choices"
        examples = self._read_tsv(input_file)

        def wrapper():
            all_dev_batches = []
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                    self.random_seed = epoch_index
                if shuffle:
                    self.global_rng = np.random.RandomState(self.random_seed)
                    self.global_rng.shuffle(examples)
                
                for batch_data in self._create_instances(
                        examples, batch_size, phase=phase):
                    if len(all_dev_batches) < self.trainer_num:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == self.trainer_num:
                        yield all_dev_batches[self.trainer_id]
                        all_dev_batches = []
                
                if phase != "train":
                    #while len(all_dev_batches) < self.trainer_num:
                    #    all_dev_batches.append(all_dev_batches[-1])
                    if self.trainer_id < len(all_dev_batches):
                        yield all_dev_batches[self.trainer_id]

        return wrapper


class ClassifyReader(BaseReader):
    """Reader for classifier task"""
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
                if len(line) > len(headers):
                    print('[warning]: text_a may contains tab which will be used as split_char')
                if self.is_zh:
                    for index, text in enumerate(line):
                        if index in text_indices:
                            line[index] = text.replace(' ', '')
                example = Example(*line)
                examples.append(example)
            return examples

    def _pad_batch_records(self, batch_records, gather_idx=None, phase=None):
        """padding batch records"""
        batch_token_ids = [record.src_ids for record in batch_records]
        if batch_records[0].label_id:
            batch_labels = [record.label_id for record in batch_records]
            batch_labels = np.array(batch_labels).astype("int64").reshape([-1, 1])
        else:
            batch_labels = np.array([]).astype("int64").reshape([-1, 1])

        if batch_records[-1].qid:
            batch_qids = [record.qid for record in batch_records]
            batch_qids = np.array(batch_qids).astype("int64").reshape([-1, 1])
        else:
            batch_qids = np.array([]).astype("int64").reshape([-1, 1])
        
        if gather_idx:
            batch_gather_idx = np.array(gather_idx).astype("int64").reshape([-1, 1])
            need_cal_loss = np.array([1]).astype("int64")
        else:
            batch_gather_idx = np.array(list(range(len(batch_records)))).astype("int64").reshape([-1, 1])
            need_cal_loss = np.array([0]).astype("int64")

        # padding
        padded_token_ids, input_mask = pad_batch_data(
            batch_token_ids, pad_idx=self.pad_id, pad_max_len=self.max_seq_len, \
            final_cls=True, return_input_mask=True)
        padded_task_ids = np.zeros_like(padded_token_ids, dtype="int64")
        padded_position_ids = get_related_pos(padded_token_ids, \
            self.max_seq_len, self.memory_len)
        
        return_list = [
            padded_token_ids, padded_position_ids, padded_task_ids,
            input_mask, batch_labels, batch_qids, batch_gather_idx, need_cal_loss
        ]

        return return_list

class MRCReader(BaseReader):
    """Reader for MRC tasks"""
    def __init__(self, 
                 trainer_id,
                 trainer_num,
                 vocab_path,
                 memory_len=128,
                 repeat_input=False,
                 train_all=False,
                 eval_all=False,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 in_tokens=False,
                 random_seed=None,
                 tokenizer="FullTokenizer",
                 is_zh=True,
                 for_cn=True,
                 doc_stride=128,
                 max_query_length=64):
        super(MRCReader, self).__init__(trainer_id,
                                        trainer_num,
                                        vocab_path,
                                        memory_len=128,
                                        repeat_input=False,
                                        train_all=False,
                                        eval_all=False,
                                        label_map_config=None,
                                        max_seq_len=512,
                                        do_lower_case=True,
                                        in_tokens=False,
                                        random_seed=None,
                                        tokenizer="FullTokenizer")
        self.for_cn = for_cn
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.examples = {}
        self.features = {}
        
    def get_num_examples(self, phase, data_path):
        """get number of example"""
        examples, features = self._pre_process_data(phase, data_path)
        return len(sum(self.features_all, []))

    def get_features(self, phase):
        """get features"""
        return self.features[phase]

    def get_examples(self, phase):
        """get examples"""
        return self.examples[phase]

    def _read_tsv(self, input_file, is_training):
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
                            doc_tokens = [paragraph_text[:answer_offset],
                                paragraph_text[answer_offset: answer_offset + answer_length],
                                paragraph_text[answer_offset + answer_length:]]

                            start_pos = 1
                            end_pos = 1

                            actual_text = " ".join(doc_tokens[start_pos:(end_pos + 1)])
                            if actual_text.find(orig_answer_text) == -1:
                                logging.info("Could not find answer: '%s' vs. '%s'",
                                        actual_text, orig_answer_text)
                                continue
                        else:
                            doc_tokens = tokenization.tokenize_chinese_chars(paragraph_text)

                        Example = namedtuple('Example',
                                ['qas_id', 'question_text', 'doc_tokens', 'orig_answer_text',
                                 'start_position', 'end_position'])

                        example = Example(
                            qas_id=qas_id,
                            question_text=question_text,
                            doc_tokens=doc_tokens,
                            orig_answer_text=orig_answer_text,
                            start_position=start_pos,
                            end_position=end_pos)
                        examples.append(example)

        return examples

    def _improve_answer_span(self, doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
        """improve answer span"""
        tok_answer_text = " ".join(self.tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
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

    def _convert_example_to_feature(self, examples, max_seq_length, tokenizer, phase):
        """convert example to feature"""
        Feature = namedtuple("Feature", ["qid", "example_index", "doc_span_index",
                    "tokens", "token_to_orig_map", "token_is_max_context",
                    "src_ids", "start_position", "end_position", "cal_loss"])
        features = []
        self.features_all = []
        unique_id = 1000
        is_training = phase == "train"
        for (example_index, example) in enumerate(examples):
            query_tokens = self.tokenizer.tokenize(example.question_text)
            if len(query_tokens) > self.max_query_length:
                query_tokens = query_tokens[0:self.max_query_length]
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = self.tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None
            if is_training:
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

            features_each = []
            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                tokens.append("[CLS]")

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[
                        split_token_index]

                    is_max_context = self._check_is_max_context(
                        doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                tokens.append("[SEP]")

                for token in query_tokens:
                    tokens.append(token)
                tokens.append("[SEP]")

                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
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
                        doc_offset = 1 #len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

                feature = Feature(
                    qid=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    src_ids=token_ids,
                    start_position=start_position,
                    end_position=end_position,
                    cal_loss=1)
                features.append(feature)
                features_each.append(feature)

                unique_id += 1

            #repeat
            if self.repeat_input:
                features_each_repeat = features_each
                if not self.train_all:
                    if phase == "train": 
                        features_each = list(map(lambda x: x._replace(cla_loss=0), features_each))
                if not self.eval_all:
                    if phase == "eval" or phase == "test":
                        features_each = list(map(lambda x: x._replace(cla_loss=0), features_each))
                features_each += features_each_repeat
            
            self.features_all.append(features_each)

        return features

    def _create_instances(self, records, batch_size, phase=None):
        """generate batch records"""
        pre_batch_list = []
        insert_idx = []
        for idx, record in enumerate(records):
            if self.cnt_list(pre_batch_list) < batch_size * self.trainer_num:
                if insert_idx:
                    pre_batch_list[insert_idx[0]] = record
                    insert_idx.pop(0)
                else:
                    pre_batch_list.append(record)

            if self.cnt_list(pre_batch_list) == batch_size * self.trainer_num:
                assert self.cnt_list(pre_batch_list) == len(pre_batch_list), "the two value must be equal"
                assert not insert_idx, "the insert_idx must be null"
                samples_batch = self._get_samples(pre_batch_list, batch_size)

                for idx, lit in enumerate(pre_batch_list):
                    if not lit:
                        insert_idx.append(idx)
                
                for batch_records in  self._prepare_batch_data(samples_batch, batch_size, phase):
                    yield batch_records

        if phase != "train":
            if self.cnt_list(pre_batch_list):
                pre_batch_list += [[] for _ in range(batch_size * self.trainer_num - self.cnt_list(pre_batch_list))]
                samples_batch = self._get_samples(pre_batch_list, batch_size, is_last=True)
                for batch_records in self._prepare_batch_data(samples_batch, batch_size, phase):
                    yield batch_records


    def _pad_batch_records(self, batch_records, gather_idx=None, phase=None):
        """pad batch data"""
        batch_token_ids = [record.src_ids for record in batch_records]
        
        if phase == "train":
            batch_start_position = [record.start_position for record in batch_records]
            batch_end_position = [record.end_position for record in batch_records]
            batch_start_position = np.array(batch_start_position).astype("int64").reshape([-1, 1])
            batch_end_position = np.array(batch_end_position).astype("int64").reshape([-1, 1])
        else:
            batch_size = len(batch_token_ids)
            batch_start_position = np.zeros(shape=[batch_size, 1], dtype="int64")
            batch_end_position = np.zeros(shape=[batch_size, 1], dtype="int64")
        
        batch_qids = [record.qid for record in batch_records]
        batch_qids = np.array(batch_qids).astype("int64").reshape([-1, 1])
        
        if gather_idx:
            batch_gather_idx = np.array(gather_idx).astype("int64").reshape([-1, 1])
            need_cal_loss = np.array([1]).astype("int64")
        else:
            batch_gather_idx = np.array(list(range(len(batch_records)))).astype("int64").reshape([-1, 1])
            need_cal_loss = np.array([0]).astype("int64")

        # padding
        padded_token_ids, input_mask = pad_batch_data(
            batch_token_ids, pad_idx=self.pad_id, pad_max_len=self.max_seq_len, return_input_mask=True)
        padded_task_ids = np.zeros_like(padded_token_ids, dtype="int64")
        padded_position_ids = get_related_pos(padded_task_ids, self.max_seq_len, self.memory_len)

        return_list = [
            padded_token_ids, padded_position_ids, padded_task_ids, input_mask,
            batch_start_position, batch_end_position, batch_qids, batch_gather_idx, need_cal_loss
        ]

        return return_list
 
    def _pre_process_data(self, phase, data_path):
        """preprocess data"""
        assert os.path.exists(data_path), "%s is not exist !" % self.config.data_path
        examples = self._read_tsv(data_path, phase == "train")
        features = self._convert_example_to_feature(examples, self.max_seq_len,
                self.tokenizer, phase)
        self.examples[phase] = examples
        self.features[phase] = features
        return examples, features

    def data_generator(self, 
                       input_file, 
                       batch_size, 
                       epoch,
                       shuffle=True,
                       phase=None):
        """data generate"""
        assert phase in ["train", "eval", "test", "infer"], "phase should be one of the four choices"
        examples = self.examples.get(phase, None)
        features = self.features.get(phase, None)

        if not examples:
            examples, features = self._pre_process_data(phase, input_file)

        def wrapper():
            """wrapper"""
            all_dev_batches = []
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                    self.random_seed = epoch_index
                if shuffle:
                    self.global_rng = np.random.RandomState(self.random_seed)
                    self.global_rng.shuffle(self.features_all)

                for batch_data in self._create_instances(
                        self.features_all, batch_size, phase=phase):
                    if len(all_dev_batches) < self.trainer_num:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == self.trainer_num:
                        yield all_dev_batches[self.trainer_id]
                        all_dev_batches = []

                if phase != "train":
                    if self.trainer_id < len(all_dev_batches):
                        yield all_dev_batches[self.trainer_id]

        return wrapper



if __name__ == '__main__':
    pass
