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

""" RefcocoPlus DataReader implementation """

from __future__ import print_function
from __future__ import division

import os
import base64
import numpy as np
import types
import gzip
import logging
import re
import six
import collections
import random

import paddle
import paddle.fluid as fluid

from batching.finetune_batching import prepare_refcoco_plus_batch_data
from preprocess import preprocessor

class RefcocoPlusDataReader(object):
    """ 
        data reader task for refcoco plus
    """
    def __init__(self,
                 task_group,
                 split,
                 vocab_path,
                 batch_size=4096,
                 max_seq_len=512,
                 shuffle_files=True,
                 epoch=100,
                 voc_size=0,
                 is_test=False):

        self.vocab = self.load_vocab(vocab_path)
        self.task_group = task_group
        self.processor = getattr(preprocessor, task_group[0]["Proprocessor"])(
            tokenizer_name =self.task_group[0]["tokenizer_name"],
            vocab_path = vocab_path)
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.epoch = epoch
        self.split = split
        self.current_epoch = 0
        self.current_file_index = 0
        self.total_file = 0
        self.current_file = None
        self.voc_size = voc_size
        self.max_seq_len = max_seq_len
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.mask_id = self.vocab["[MASK]"]
        self.input_slots = 9
        self.is_test = is_test

        if is_test:
            self.epoch = 1
            self.shuffle_files = False

    def get_progress(self):
        """
        return current progress of traning data
        """
        self.progress_dict = {"current_epoch": self.current_epoch,
                         "current_file_index": self.current_file_index,
                         "total_file": self.total_file,
                         "current_file": self.current_file
                         }
        return self.progress_dict

    def process_vl(self, line, max_seq_len):
        """
        process single v+l data
        """
        def decode_feature(base64_str, size):
            """ 
            decode feature from base64 
            """
            fea_base64 = base64.b64decode(base64_str)
            fea_decode = np.frombuffer(fea_base64, dtype=np.float32)
            shape = size, int(fea_decode.shape[0] / size)
            features = np.resize(fea_decode, shape)
            return features

        text, image_w, image_h, number_boxes, number_boxes_gl, image_loc, \
             image_embeddings, box_label, label = line
        
        token_ids = []
        raw_ids = self.processor.convert_sentence_to_ids_without_cls(text)
        token_ids.append(self.vocab["[CLS]"])
        token_ids.extend(raw_ids)
        token_ids.append(self.vocab["[SEP]"])
        sent_ids = [0] * len(token_ids)
        pos_ids = range(0, len(token_ids))

        #print("sent_ids:", sent_ids)
        token_ids = [int(token) for token in token_ids]
        sent_ids = [int(token) for token in sent_ids]
        pos_ids = [int(token) for token in pos_ids]
        assert len(token_ids) == len(sent_ids) == len(pos_ids), \
                "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids)"
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[0: self.max_seq_len - 1] + [token_ids[-1]]
            sent_ids = sent_ids[0: self.max_seq_len - 1] + [sent_ids[-1]]
            pos_ids = pos_ids[0: self.max_seq_len]

        all_number_box = int(number_boxes) + int(number_boxes_gl)
        image_loc = decode_feature(image_loc, all_number_box)
        shape_np = np.repeat(np.array(\
            [[float(image_w), float(image_h), float(image_w), float(image_h)]]), all_number_box, axis=0)
        boxes_np = image_loc / shape_np
        
        area = (boxes_np[:, 3] - boxes_np[:, 1]) * (boxes_np[:, 2] - boxes_np[:, 0])
        image_loc = np.concatenate((boxes_np, np.expand_dims(area, 1)), axis = 1)
        loc_cls = np.array([[0.0, 0.0, 1.0, 1.0, 1.0]], dtype = "float32")
        image_loc = np.concatenate([loc_cls, image_loc], 0)

        image_embeddings = decode_feature(image_embeddings, all_number_box)
        image_embeddings_cls = np.mean(image_embeddings, axis = 0, keepdims = True)
        image_embeddings =  np.concatenate([image_embeddings_cls, image_embeddings], 0)
        x1, y1, x2, y2 = [float(item) for item in box_label.split(" ")]
        cls_label = (x2 - x1 + 1) * (y2 - y1 + 1) /(float(image_w) * float(image_h))
        score_th = 0.5
        if cls_label < score_th:
            cls_label = 0.0

        label_tmp = label.split(" ")
        if not self.is_test:
            for i  in range(len(label_tmp)):
                if float(label_tmp[i]) < score_th:
                    label_tmp[i] = 0.0

        label = [[cls_label]] + [[float(token)] for token in label_tmp]
        label = np.array(label, dtype="float32")
        add_item = [all_number_box + 1, image_w, image_h] + [float(item) for item in box_label.split(" ")]
        sample_json = {
            "token_ids": token_ids,
            "sent_ids": sent_ids,
            "pos_ids": pos_ids,
            "label": label,
            "image_loc": image_loc,
            "image_embeddings": image_embeddings,
            "all_number_box": all_number_box,
            "add_item": add_item
        }
        return sample_json

    def parse_line(self, line, max_seq_len=512, task_index=None):
        """ parse one line to token_ids, sentence_ids, pos_ids, label """
        line = line.strip().split("\t")
        assert len(line) == self.input_slots, "One sample must have %d fields!" % self.input_slots
        sample_json = self.process_vl(line, max_seq_len)
        token_ids = sample_json["token_ids"]
        return sample_json

    def read_file(self, file, task_index):
        """ read line data from a file """
        try:
            assert file.endswith('.gz'), "[ERROR] %s is not a gzip file" % file
            with gzip.open(file, "rb") as f:
                lines = f.readlines()
        except:
            with open(file, "rb") as f:
                lines = f.readlines()
        if not self.is_test:
            np.random.shuffle(lines)
        for line in lines:
            parsed_line = self.parse_line(
                line, max_seq_len=self.max_seq_len, task_index=task_index)
            if parsed_line is None:
                continue
            yield parsed_line

    def convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        fin = open(vocab_file)
        for num, line in enumerate(fin):
            items = self.convert_to_unicode(line.strip()).split("\t")
            if len(items) > 2:
                break
            token = items[0]
            index = items[1] if len(items) == 2 else num
            token = token.strip()
            vocab[token] = int(index)
        return vocab

    def data_generator(self):
        """ data_generator """
        if self.split == "train":
            filelist_key = "train_filelist"
        elif self.split == "val":
            filelist_key = "val_filelist"
        elif self.split == "testA": 
            filelist_key = "testA_filelist"
        else: filelist_key = "testB_filelist"

        all_files = []
        task_probs = []
        sum = 0.0
        for task in self.task_group:
            all_files.append(open(task[filelist_key]).readlines())
            task_probs.append(task["prob"])
            sum += task["prob"]
        for i in xrange(len(task_probs)):
            task_probs[i] = task_probs[i] / sum
        task_probs = np.array(task_probs).ravel()

        def wrapper():
            """ 
            wrapper 
            """
            def reader(task_index):
                """
                reader
                """
                files = all_files[task_index]
                for epoch in range(self.epoch):
                    if self.shuffle_files:
                        if epoch < 0:
                            files = files + open(task["gt_train_filelist"]).readlines()
                        np.random.shuffle(files)
                    for index, file in enumerate(files):
                        file = file.strip()
                        sample_generator = self.read_file(file, task_index)
                        for sample in sample_generator:
                            self.current_epoch = epoch + 1
                            self.current_file_index = index + 1
                            self.current_file = file
                            self.total_file = len(files)
                            if sample is None:
                                continue
                            yield sample

            def batch_reader(reader, batch_size):
                """
                batch reader
                """
                batch, total_token_num, max_len = [], 0, 0
                cur_size = 0
                dev_count = 1
                buff = []
                readers = []
                for i in xrange(len(task_probs)):
                    buff.append(None)
                    readers.append(reader(i))
                task_indices = range(len(task_probs))
                end_times = 0
                while end_times < 50:
                    task_index = np.random.choice(task_indices, p=task_probs)
                    dev_num = 0
                    cur_reader = readers[task_index]
                    while dev_num < dev_count:
                        if buff[task_index] is not None:
                            cur_len = len(buff[task_index]["token_ids"])
                            max_len = max(max_len, cur_len)
                            batch.append(buff[task_index])
                            total_token_num += cur_len
                            buff[task_index] = None
                            cur_size += 1

                        parsed_line = next(cur_reader, None)

                        if parsed_line is None:
                            end_times += 1
                            dev_num += 1
                            if len(batch) > 0:
                                yield batch, total_token_num, task_index
                                batch, total_token_num, max_len = [], 0, 0
                            continue

                        end_times = 0
                        cur_len = len(parsed_line["token_ids"])
                        max_len = max(max_len, cur_len)
                        if cur_size >= batch_size:
                            yield batch, total_token_num, task_index
                            batch, total_token_num, max_len = [], 0, 0
                            cur_size = 0
                            dev_num += 1
                            buff[task_index] = parsed_line
                        else:
                            batch.append(parsed_line)
                            cur_size += 1
                            total_token_num += cur_len

            for batch_data, total_token_num, task_index in batch_reader(reader, self.batch_size):
                yield prepare_refcoco_plus_batch_data(
                    batch_data,
                    total_token_num,
                    task_index,
                    len(self.task_group),
                    voc_size=self.voc_size,
                    pad_id=self.pad_id,
                    return_input_mask=True,
                    return_max_len=False,
                    return_num_token=False)

        return wrapper


if __name__ == "__main__":
    pass
