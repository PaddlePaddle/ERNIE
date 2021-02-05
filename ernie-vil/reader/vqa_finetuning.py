#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

""" VQA Data Reader implementation """

from __future__ import print_function
from __future__ import division

import os
import base64
import functools
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

from batching.finetune_batching import prepare_vqa_batch_data
from preprocess import preprocessor

class VQADataReader(object):
    """ 
        data reader task for vqa
    """
    def __init__(self,
                 task_group,
                 split,
                 vocab_path,
                 batch_size=4096,
                 num_class=3129,
                 max_seq_len=512,
                 shuffle_files=True,
                 epoch=100,
                 voc_size=0,
                 cls_size=0,
                 is_test=False):

        self.vocab = self.load_vocab(vocab_path)
        self.task_group = task_group
        self.processor = getattr(preprocessor, task_group[0]["Proprocessor"])(
            tokenizer_name =self.task_group[0]["tokenizer_name"],
            vocab_path = vocab_path)
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.epoch = epoch
        self.current_epoch = 0
        self.current_file_index = 0
        self.total_file = 0
        self.num_class=num_class
        self.current_file = None
        self.voc_size = voc_size
        self.cls_size = cls_size
        self.max_seq_len = max_seq_len
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.mask_id = self.vocab["[MASK]"]
        self.is_test = is_test
        self.split = split
        if self.is_test:
            self.epoch = 1
            self.shuffle_files = False
            self.vg_init_epochs = 0
        else:
            self.vg_init_epochs = int(self.task_group[0]["vg_init_epochs"])

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
        trans the orgin tokens to the wanted tokens
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
        question_id, text, match_label, score, image_w, image_h, number_box, \
            image_loc, image_embeddings = line
        
        token_ids = []
        raw_ids = self.processor.convert_sentence_to_ids_without_cls(text)
        token_ids.append(self.vocab["[CLS]"])
        token_ids.extend(raw_ids)
        token_ids.append(self.vocab["[SEP]"])
        sent_ids = [0] * len(token_ids)
        pos_ids = range(0, len(token_ids))
        token_ids = [int(token) for token in token_ids]
        sent_ids = [int(token) for token in sent_ids]
        pos_ids = [int(token) for token in pos_ids]

        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[0: self.max_seq_len - 1] + [token_ids[-1]]
            sent_ids = sent_ids[0: self.max_seq_len - 1] + [sent_ids[-1]]
            pos_ids = pos_ids[0: self.max_seq_len] 
       
        labels = [int(label_tok) for label_tok in match_label.split("|")] 
        scores = [float(score_tok) for score_tok in score.split("|")]
        number_box = int(number_box)
        
        question_id = int(question_id)
        image_loc = decode_feature(image_loc, number_box)
        shape_np = np.repeat(np.array(\
            [[float(image_w), float(image_h), float(image_w), float(image_h)]]), number_box, axis=0)
        boxes_np = image_loc / shape_np
        area = (boxes_np[:, 3] - boxes_np[:, 1]) * (boxes_np[:, 2] - boxes_np[:, 0])
        image_loc = np.concatenate((boxes_np, np.expand_dims(area, 1)), axis = 1)
        loc_cls = np.array([[0.0, 0.0, 1.0, 1.0, 1.0]], dtype = "float32")
        image_loc = np.concatenate([loc_cls, image_loc], 0)
        try:
            image_embeddings = decode_feature(image_embeddings, number_box)
            image_embeddings_cls = np.mean(image_embeddings, axis = 0, keepdims = True)
            image_embeddings =  np.concatenate([image_embeddings_cls, image_embeddings], 0)
            self.default_image_emb = image_embeddings
        except:
            print("error data occur, a random default image emb will be assin to this one")
            print("the wrong line occur")
            image_embeddings = self.default_image_emb
        weight_labels = self.get_weight_label(self.num_class, labels, scores)

        sample_json = {
            "question_id": question_id,
            "token_ids": token_ids,
            "sent_ids": sent_ids,
            "pos_ids": pos_ids,
            "weight_labels": weight_labels,
            "image_loc": image_loc,
            "image_embeddings": image_embeddings,
        }
        return sample_json

    def get_weight_label(self, num_class, labels, scores):
        """assign the corresponding score for the labels
        Input: labels  (Indefinite length list, like [1, 2, 3])
               scores  (Indefinite length list, like [0.1, 0.2, 0.3])
        Output: weight_score  (list, length equals num_class)
        """
        assert len(labels) == len(scores), \
            "unequals length with labels has %d number(s) while scores has %d number(s)!" % (len(labels), len(scores))
        weight_score = [0] * num_class
        for i in range(len(labels)):
            weight_score[labels[i]] =  scores[i]
        return weight_score

    def parse_line(self, line, max_seq_len=512, task_index=None):
        """ parse one line to token_ids, sentence_ids, pos_ids, label """

        line = line.strip().split("\t")
        sample_json = self.process_vl(line, max_seq_len)
        return sample_json

    def read_file(self, file, task_index):
        """ read line data from a file """
        with open(file, "rb") as f:
            lines = f.readlines()
            if not self.is_test:
                np.random.shuffle(lines)
            for line in lines:
                yield line

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
        filelist_key = "train_filelist"
        if self.is_test:
            if self.split == "val":
                filelist_key = "val_filelist"
            elif self.split == "test_dev":
                filelist_key = "test_dev_filelist"
            elif self.split == "test_std":
                filelist_key = "test_std_filelist"
            else:
                print("*************no split named as :", self.split, "********************")
                return None
                

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
            warpper
            """
            def reader(task_index):
                """
                reader
                """
                files = all_files[task_index]
                global_rng = np.random.RandomState(0)
                for epoch in range(self.epoch):
                    if epoch < self.vg_init_epochs:
                        files =  open(task["vg_train_filelist"]).readlines() + all_files[task_index]
                    if self.shuffle_files:
                        global_rng.shuffle(files)
                    for index, file in enumerate(files):
                        file = file.strip()
                        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
                        try:
                            trainers_num = int(os.getenv("PADDLE_TRAINERS_NUM"))
                        except:
                            print("can not get env PADDLE_TRAINERS_NUM, set trainer_nums to 1")
                            trainers_num = 1

                        if index % trainers_num != trainer_id:
                            continue
                        sample_generator = paddle.reader.xmap_readers(self.parse_line, \
                            functools.partial(self.read_file, file=file, task_index=task_index), 4, 200)
                        for sample in sample_generator():
                            self.current_epoch = epoch + 1
                            self.current_file_index = index + 1
                            self.current_file = file
                            self.total_file = len(files)
                            if sample is None:
                                continue
                            yield sample

            def batch_reader(reader, batch_size):
                """
                Batch data reader
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
                yield prepare_vqa_batch_data(
                    batch_data,
                    total_token_num,
                    task_index,
                    len(self.task_group),
                    voc_size=self.voc_size,
                    pad_id=self.pad_id,
                    cls_id=self.cls_id,
                    sep_id=self.sep_id,
                    mask_id=self.mask_id,
                    return_input_mask=True,
                    return_max_len=False,
                    return_num_token=False)

        return wrapper


if __name__ == "__main__":
    pass
