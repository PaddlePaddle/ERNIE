#    Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" VCR Data Reader implementation """

from __future__ import print_function
from __future__ import division

import os
import base64
import numpy as np
import re
import random
import json
import json_lines
import csv
import sys
import itertools

from reader._image_features_reader import ImageFeaturesH5Reader
from preprocess import preprocessor
from batching.finetune_batching import prepare_batch_data

import paddle.fluid as fluid

def _converId(img_id):
    """ conversion for image ID """
    img_id = img_id.split('-')
    if 'train' in img_id[0]:
        new_id = int(img_id[1])
    elif 'val' in img_id[0]:
        new_id = int(img_id[1]) + 1000000
    elif 'test' in img_id[0]:
        new_id = int(img_id[1]) + 2000000
    else:
        print("no split known")
    return new_id


def _load_annotationsQ_A(annotations_jsonpath, split):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""
    entries = []
    with open(annotations_jsonpath) as f:
        for annotation in json_lines.reader(f):
            det_names = ""
            question = annotation["question"]
            if split == 'test':
                ans_label = 0
            else:
                ans_label = annotation["answer_label"]
            img_id = _converId(annotation["img_id"])
            anno_id = int(annotation["annot_id"].split('-')[1])
            entries.append(
                     {"question": question,
                      "answers": annotation["answer_choices"],
                      "metadata_fn": annotation["metadata_fn"],
                      "target": ans_label,
                      "img_id": img_id,
                      "anno_id": anno_id,
                      "det_names": annotation['objects']
                    })
    return entries


def _load_annotationsQA_R(annotations_jsonpath, split):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""
    entries = []
    with open(annotations_jsonpath, 'rb') as f: 
        for annotation in json_lines.reader(f):
            if split == 'test':
                for answer in annotation["answer_choices"]:
                    question = annotation["question"] + ["[MARK]"] + answer
                    img_id = _converId(annotation["img_id"])
                    ans_label = 0
                    anno_id = int(annotation["annot_id"].split('-')[1])
                    entries.append(
                             {"question": question,
                              "answers": annotation["rationale_choices"],
                              "metadata_fn": annotation["metadata_fn"],
                              "target": ans_label,
                              "img_id": img_id,
                              "anno_id": anno_id,
                              "det_names": annotation['objects']
                            })
            else:
                det_names = ""
                question = annotation["question"] + ["[MARK]"]  + \
                               annotation["answer_choices"][annotation['answer_label']]
                ans_label = annotation["rationale_label"]
                img_id = _converId(annotation["img_id"])
                anno_id = int(annotation["annot_id"].split('-')[1])
                entries.append(
                         {"question": question,
                          "answers": annotation["rationale_choices"],
                          "metadata_fn": annotation["metadata_fn"],
                          "target": ans_label,
                          "img_id": img_id,
                          "anno_id": anno_id, 
                          "det_names": annotation['objects']})
    return entries


class VCRDataReader(object):
    """ 
        data reader task for vcr
    """
    def __init__(self,
                 task_conf,
                 split,
                 vocab_path=None,
                 batch_size=4096,
                 shuffle=True,
                 epoch=100,
                 is_test=False,
                 feature_reader_dict={},
                 random_seed=None,
                 task_index=0,
                 task_num=1):

        self.task_conf = task_conf
        self.processor = getattr(preprocessor,
                                 task_conf["Proprocessor"])(tokenizer_name=self.task_conf["tokenizer_name"],
                                 vocab_path=vocab_path)
        self.vocab = self.processor.vocab
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = epoch
        self.current_epoch = 0
        self.current_file_index = 0
        self.total_file = 0
        self.current_file = None
        self.random_seed = random_seed
        self.max_seq_len = self.task_conf['max_seq_len']
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.mask_id = self.vocab["[MASK]"]
        self.is_test = is_test
        self.task_index = task_index
        self.task_num = task_num

        if self.is_test:
            self.epoch = 1
            self.shuffle_files = False
        if self.shuffle:
            shufflekeep_across_task = self.task_conf.get('shufflekeep_across_task', True)
            if shufflekeep_across_task:
                self.global_rng = np.random.RandomState(random_seed)
            else:
                self.global_rng = np.random.RandomState()
            self.shuffle_every_epoch = self.task_conf.get('shuffle_every_epoch', False)
        task=self.task_conf['task']
        annotations_jsonpath=self.task_conf['annotations_jsonpath_' + split]
        self.num_choice = int(self.task_conf['num_choice'])
        if task == 'VCR_Q-A':
            self._entries = _load_annotationsQ_A(annotations_jsonpath, split)
        elif task == "VCR_QA-R":
            self._entries = _load_annotationsQA_R(annotations_jsonpath, split)
        else:
            assert False
        self._split = split
        self._names = []
        with open(self.task_conf['unisex_names_table']) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[1] != 'name':
                    self._names.append(row[1])
        self._feature_reader = feature_reader_dict[self.task_conf['feature_lmdb_path']]
        self.use_gt_fea = task_conf.get('use_gt_fea', False)
        if self.use_gt_fea:
            self._gt_feature_reader = feature_reader_dict[self.task_conf['gt_feature_lmdb_path']]
            self._max_region_num = self.task_conf.get('max_region_num', 100)
            print("use gt featurre")
        else:
            self._max_region_num = self.task_conf.get('max_region_num', 37)
            print("only butd feature")
        self.tokenize()

    def generate_random_name(self, det_names):
        """ 
            replace "person" with a random name
        """
        random_name = []
        for name in det_names:
            if name == 'person':
                word = random.choice(self._names)
            else:
                word = name
            random_name.append(word)

        return random_name

    def replace_det_with_name(self, inputs, random_names):
        """
            replace det with name
        """
        tokens = []
        mask = []
        for w in inputs:
            if isinstance(w, list):
                for idx in w:
                    word = random_names[idx]
                    tokens.append(word)
            else:
                word = w.encode('utf-8')
                tokens.append(word)

        return tokens, mask

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """
            Truncates a sequence pair in place to the maximum length.
        """
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def get_progress(self):
        """
            return current progress of traning data
        """
        progress_dict = {"current_epoch": self.current_epoch,
                         "current_file_index": self.current_file_index,
                         "total_file": self.total_file,
                         "current_file": self.current_file
                         }
        return progress_dict

    def tokenize(self):
        """
            Tokenizes the captions.
        """
        # This will add caption_tokens in each entry of the dataset.
        # -1 represents nil, and should be treated as padding_idx in embedding.
        count = 0
        for entry in self._entries:
            det_names = entry["det_names"]
            random_names = self.generate_random_name(det_names)
            # replace with name
            tokens_a, mask_a = self.replace_det_with_name(entry["question"], random_names)
            q_str = " ".join(tokens_a)
            ids_a = []
            for i, q in enumerate(q_str.split(" [MARK] ")):
                if i == 1:
                    ids_a.append(self.vocab["[SEP]"])
                ids_a = ids_a + self.processor.convert_sentence_to_ids_without_cls(q)

            input_ids_all = []
            segment_ids_all = []
            input_poss_all = []
            input_len_all = []

            for answer in entry["answers"]:
                tokens_b, mask_b = self.replace_det_with_name(answer, random_names)
                ids_b = self.processor.convert_sentence_to_ids_without_cls(" ".join(tokens_b))

                self._truncate_seq_pair(ids_a, ids_b, self.max_seq_len - 3)

                input_ids = []
                segment_ids = []
                input_ids.append(self.vocab["[CLS]"])
                segment_ids.append(0)

                for id in ids_a:
                    input_ids.append(id)
                    segment_ids.append(0)

                input_ids.append(self.vocab["[SEP]"])
                segment_ids.append(0)

                assert len(ids_b) > 0
                for id in ids_b:
                    input_ids.append(id)
                    segment_ids.append(1)
                input_ids.append(self.vocab["[SEP]"])
                segment_ids.append(1)

                input_ids_all.append(input_ids)
                segment_ids_all.append(segment_ids)
                input_poss = [str(pos) for pos in range(len(input_ids))]
                input_poss_all.append(input_poss)
                input_len_all.append(len(input_ids))

            entry["input_ids"] = input_ids_all
            entry["input_poss"] = input_poss_all
            entry["segment_ids"] = segment_ids_all
            entry["input_lens"] = input_len_all

            sys.stdout.write('%d/%d\r' % (count, len(self._entries)))
            sys.stdout.flush()
            count += 1

    def parse_line(self, s_index):
        """
           form the slot info from line
        """
        entry = self._entries[s_index]
        image_id = entry["img_id"]
        image_fea_json = self._feature_reader[image_id]
        features = image_fea_json["features"]
        num_boxes = image_fea_json["num_boxes"]
        boxes = image_fea_json["image_location"]
        if not self.use_gt_fea:
            num_boxes = min(num_boxes, self._max_region_num)
            boxes = boxes[:num_boxes]
            features = features[:num_boxes]
        else:
            boxes = boxes[:num_boxes]
            features = features[:num_boxes]
            image_fea_json = self._gt_feature_reader[image_id]
            gt_features = image_fea_json["features"]
            gt_num_boxes = image_fea_json["num_boxes"]
            gt_boxes = image_fea_json["image_location"]
            features[0] = (features[0] * num_boxes + gt_features[0] * gt_num_boxes) / (num_boxes + gt_num_boxes)

            gt_boxes = gt_boxes[1: gt_num_boxes]
            gt_features = gt_features[1: gt_num_boxes]
            gt_num_boxes = gt_num_boxes - 1

            gt_box_preserve = min(self._max_region_num - 1, gt_num_boxes)
            gt_boxes = gt_boxes[:gt_box_preserve]
            gt_features = gt_features[:gt_box_preserve]
            gt_num_boxes = gt_box_preserve

            num_box_preserve = min(self._max_region_num - int(gt_num_boxes), int(num_boxes))
            boxes = boxes[:num_box_preserve]
            features = features[:num_box_preserve]

            # concatenate the boxes
            mix_boxes = np.concatenate((boxes, gt_boxes), axis=0)
            mix_features = np.concatenate((features, gt_features), axis=0)
            mix_num_boxes = num_box_preserve + int(gt_num_boxes)

            num_boxes = min(mix_num_boxes, self._max_region_num)
            boxes = mix_boxes[:num_boxes]
            features = mix_features[:num_boxes]
            record = {
                "input_ids": entry["input_ids"],
                "input_pos": entry["input_poss"],
                "segment_ids": entry["segment_ids"],
                "input_lens": entry["input_lens"],
                "target": int(entry["target"]),
                "features": features,
                "boxes": boxes,
                "anno_id": entry["anno_id"]
                }
        return record

    def data_generator(self):
        """ data_generator """
        sample_indice = range(len(self._entries))
        def wrapper():
            """
            wrapper
            """
            for epoch_index in range(self.epoch):
                if self._split == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                if self.shuffle:
                    if epoch_index == 0:
                        self.global_rng.shuffle(sample_indice)
                        print("shuffle epoch %d" % epoch_index)
                    elif self.shuffle_every_epoch:
                        self.global_rng.shuffle(sample_indice)
                        print("shuffle epoch %d" % epoch_index)
                batch_records = []
                for index in sample_indice:
                    batch_records.append(self.parse_line(index))
                    if len(batch_records) == self.batch_size:
                        yield prepare_batch_data(
                            batch_records, self.num_choice, self.pad_id, \
                            self.task_index, self.task_num), self.task_conf['task']
                        batch_records = []
                if len(batch_records) > 0:
                    yield prepare_batch_data(
                        batch_records, self.num_choice, self.pad_id, \
                        self.task_index, self.task_num), self.task_conf['task']
        return wrapper


class VCRDataJointReader(object):
    """ Joint data reader for Q2A task and QA2R task"""
    def __init__(self,
                 task_conf_group,
                 split,
                 batch_size=4096,
                 shuffle=True,
                 epoch=100,
                 vocab_path=None,
                 is_test=False):

        self.task_readers = []
        feature_reader_dict = {}
        self.task_dup_cnt = []
        for task_conf in task_conf_group:
            if 'feature_lmdb_path' in task_conf:
                if task_conf['feature_lmdb_path'] not in feature_reader_dict:
                    feature_reader_dict[task_conf['feature_lmdb_path']] =    \
                        ImageFeaturesH5Reader(task_conf['feature_lmdb_path'])
            if 'gt_feature_lmdb_path' in task_conf and task_conf.get('use_gt_fea', False):
                if task_conf['gt_feature_lmdb_path'] not in feature_reader_dict:
                    feature_reader_dict[task_conf['gt_feature_lmdb_path']] =    \
                        ImageFeaturesH5Reader(task_conf['gt_feature_lmdb_path'])
            task_batch_size = task_conf.get('batch_size', 64)
            self.task_dup_cnt.append(max(int(task_batch_size / batch_size), 1))
        random_seed=np.random.randint(1000)
        for task_index, task_conf in enumerate(task_conf_group):
            self.task_readers.append(VCRDataReader(task_conf, split, vocab_path, batch_size, shuffle,
                epoch, is_test, feature_reader_dict, random_seed, task_index, len(task_conf_group)))
        self.task_generators = [reader.data_generator() for reader in self.task_readers]

    def get_progress(self):
        """return current progress of traning data
        """
        current_epoch = max([reader.current_epoch for reader in self.task_readers])
        current_file_index = max([reader.current_file_index for reader in self.task_readers])
        total_file = max([reader.total_file for reader in self.task_readers])
        current_file = ""
        self.progress_dict = {"current_epoch": current_epoch,
                         "current_file_index": current_file_index,
                         "total_file": total_file,
                         "current_file": current_file
                         }
        return self.progress_dict

    def data_generator(self):
        """ data_generator """
        def wrapper():
            """
            warpper
            """
            task_buffer = [[] for i in range(len(self.task_dup_cnt))]
            for data in itertools.izip(*[generator() for generator in self.task_generators]):
                for i, d in enumerate(data):
                    task_buffer[i].append(d)
                    if len(task_buffer[i]) >= self.task_dup_cnt[i]:
                        for t in task_buffer[i]:
                            yield t[0]
                        task_buffer[i] = []

        return wrapper


if __name__ == "__main__":
    pass
