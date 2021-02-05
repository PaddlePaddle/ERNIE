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
import copy
import random
import pickle

import paddle
import paddle.fluid as fluid

from batching.finetune_batching import prepare_flickr_data
from preprocess import preprocessor


class FlickrDataReader(object):
    """ 
        data reader task for flickr
    """
    def __init__(self,
                 task_group,
                 vocab_path,
                 split,
                 batch_size=4096,
                 max_seq_len=512,
                 shuffle_files=True,
                 epoch=100,
                 voc_size=0,
                 cls_size=0,
                 is_test=False):

        self.vocab = self.load_vocab(vocab_path)
        self.task_group = task_group
        self.max_seq_len = max_seq_len
        self.processor = getattr(preprocessor, task_group[0]["Proprocessor"])(
            tokenizer_name =self.task_group[0]["tokenizer_name"],
            vocab_path = vocab_path)

        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.epoch = epoch
        self.current_epoch = 0
        self.current_file_index = 0
        self.total_file = 0
        self.current_file = None
        self.voc_size = voc_size
        self.cls_size = cls_size
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.mask_id = self.vocab["[MASK]"]
        self.is_test = is_test

        if self.is_test:
            self.epoch = 1
            self.shuffle_files = False
            self._test_image_list = []
            if split == "dev":
                image_path = self.task_group[0]["dev_image_path"]
            else:
                image_path = self.task_group[0]["test_image_path"]
        else:
             caption_path = self.task_group[0]["train_caption_path"]
             self._load_caption_dict(caption_path)
             image_path = self.task_group[0]["train_image_path"]
             self._get_hardest_setting(self.task_group[0]["hardest_setting_path"])
             self._negative_schema=self.task_group[0]["negative_schema"] 
        self._load_image_dict(image_path)

    def decode_all(self, image_id, width, height, number_box, boxes, image_embeddings):
        """ decode all data """
        def decode_feature(base64_str, size):
            """ decode feature from base64 """
            size = int(size)
            fea_base64 = base64.b64decode(base64_str)
            fea_decode = np.frombuffer(fea_base64, dtype=np.float32)
            shape = size, int(fea_decode.shape[0] / size)
            features = np.resize(fea_decode, shape)
            return features

        image_embeddings = decode_feature(image_embeddings, number_box)
        image_embeddings_cls = np.mean(image_embeddings, axis = 0, keepdims = True)
        image_embeddings =  np.concatenate([image_embeddings_cls, image_embeddings], 0)

        boxes = decode_feature(boxes, number_box)
        shape = np.repeat(np.array([[float(width), float(height), float(width), float(height)]]), \
                number_box, axis=0)
        boxes = boxes / shape
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        image_loc = np.concatenate((boxes, np.expand_dims(area, 1)), axis = 1)
        loc_cls = np.array([[0.0, 0.0, 1.0, 1.0, 1.0]], dtype = "float32")
        image_loc = np.concatenate([loc_cls, image_loc], 0)
        return int(number_box) + 1, image_loc, image_embeddings

    def _load_image_dict(self, image_path):
        self._image_feature_dict = {}
        image_items = image_path.split(',')
        cnt = 0
        for image_item in image_items:
            with open(image_item) as f:
                for line in f:
                    cnt += 1
                    if cnt % 1000 == 0:
                        print('precessing image feature:', cnt)
                    image_id, width, height, number_box, image_loc, image_embeddings \
                            = line.strip().split('\t')
                    number_box, image_loc, image_embeddings = self.decode_all( \
                            image_id, width, height, number_box, image_loc, image_embeddings)
                    self._image_feature_dict[int(image_id)] = (width, height, image_embeddings, number_box, image_loc)
                    if self.is_test:
                        self._test_image_list.append(int(image_id))

    def _load_caption_dict(self, image_caption):
        """
        Load caption dict for flickr 
        """
        self._caption_ids_dict = {}
        self._image_sent_map = {}
        with open(image_caption) as f:
            cnt = 0
            for line in f:
                cnt += 1
                line = line.strip().split("\t")
                image_id, sent_id, text = line
                token_ids = []
                raw_ids = self.processor.convert_sentence_to_ids_without_cls(text)
                token_ids.append(self.vocab["[CLS]"])
                token_ids.extend(raw_ids)
                token_ids.append(self.vocab["[SEP]"])
                sent_ids = [0] * len(token_ids)
                pos_ids = range(0, len(token_ids))
                
                if cnt % 5000 == 0:
                    print(cnt)

                if len(token_ids) > self.max_seq_len:
                    token_ids = token_ids[0: self.max_seq_len - 1] + [token_ids[-1]]
                    sent_ids = sent_ids[0: self.max_seq_len - 1] + [sent_ids[-1]]
                    pos_ids = pos_ids[0: self.max_seq_len]

                assert len(token_ids) == len(sent_ids) == len(pos_ids), \
                        "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids)"

                self._caption_ids_dict[int(sent_id)] = \
                        [token_ids, sent_ids, pos_ids, int(image_id)]
                self._image_sent_map.setdefault(int(image_id), [])
                self._image_sent_map[int(image_id)].append(int(sent_id))
        self._train_caption_ids = self._caption_ids_dict.keys()

    def _get_hardest_setting(self, hardest_setting_path):
        """
        Get the training metrix 
        """
        with open(hardest_setting_path, 'rb') as f:
            data = pickle.load(f)
            self._train_hard_pool = data['train_hard_pool']
            self._train_image_list = data['train_image_list']
            self._train_imgId2pool = {imageId:i for i, imageId in enumerate(self._train_image_list)}

    def get_progress(self):
        """
        Return current progress of traning data
        """
        progress_dict = {"current_epoch": self.current_epoch,
                         "current_file_index": self.current_file_index,
                         "total_file": self.total_file,
                         "current_file": self.current_file
                         }
        return progress_dict

    def process_vl(self, line, max_seq_len):
        """
        Process single v+l data
        """
        if self.is_test:
            line = line.strip().split("\t")
            image_id, sent_id, text = line
            token_ids = []
            raw_ids = self.processor.convert_sentence_to_ids_without_cls(text)
            token_ids.append(self.vocab["[CLS]"])
            token_ids.extend(raw_ids)
            token_ids.append(self.vocab["[SEP]"])
            sent_ids = [0] * len(token_ids)
            pos_ids = range(0, len(token_ids))
            if len(token_ids) > self.max_seq_len:
                token_ids = token_ids[0: self.max_seq_len - 1] + [token_ids[-1]]
                sent_ids = sent_ids[0: self.max_seq_len - 1] + [sent_ids[-1]]
                pos_ids = pos_ids[0: self.max_seq_len]
            width, height, image_embeddings, number_box, image_loc =  self._image_feature_dict[int(image_id)]
        else:
            sent_id = line
            captions_pos = self._caption_ids_dict[sent_id]
            image_id = captions_pos[-1]
            captions = [captions_pos]

            _, _, features, number_box, box = self._image_feature_dict[image_id]
        
            images = [[features, number_box, box]]
            for item in self._negative_schema:
                if item[0] == "h":
                    rand_img_id_pool = self._train_hard_pool[self._train_imgId2pool[image_id]]
                    rand_idx = rand_img_id_pool[random.randint(1, len(rand_img_id_pool) - 1)]
                    image_id_neg = self._train_image_list[int(rand_idx)]
                elif item[0] == "e":
                    while True:
                        image_id_neg = random.choice(self._train_image_list)
                        if image_id_neg != image_id:
                            break
                else:
                    print("error negative schema")
                    exit()
                if item[1] == "i":
                    _, _, features_neg, number_box_neg, box_neg = self._image_feature_dict[image_id_neg]
                    captions.append(self._caption_ids_dict[sent_id])
                    images.append([features_neg, number_box_neg, box_neg])
                elif item[1] == "c":
                    sent_id_neg = random.choice(self._image_sent_map[image_id_neg])
                    captions.append(self._caption_ids_dict[sent_id_neg])
                    images.append([features, number_box, box])
                else:
                    print("error negative schema")
                    exit()

            token_ids, sent_ids, pos_ids, _ = zip(*captions)
            image_embeddings, number_box, image_loc = zip(*images)

        sample_json = {
            "token_ids": token_ids,
            "sent_ids": sent_ids,
            "pos_ids": pos_ids,
            "image_loc": image_loc,
            "image_embeddings": image_embeddings,
            "image_id": int(image_id),
            "sent_id": int(sent_id),
            "ids": [image_id, sent_id]
        }
        return sample_json

    def parse_line(self, line, max_seq_len=512, task_index=None):
        """ parse one line to token_ids, sentence_ids, pos_ids, label """

        sample_json = self.process_vl(line, max_seq_len)
        token_ids = sample_json["token_ids"]
        return sample_json

    def read_file(self, file, task_index):
        """
        read line data from file
        """
        if self.is_test:
            with open(file) as f:
                lines = f.readlines()
                for line in lines:
                    yield line
        else:
            random.shuffle(self._train_caption_ids)
            for item in self._train_caption_ids:
                yield item

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
            filelist_key = "dev_filelist"

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
            """ wrapper """
            def reader(task_index):
                """ reader """
                files = all_files[task_index]
                for epoch in range(self.epoch):
                    if self.shuffle_files:
                        np.random.shuffle(files)
                    for index, file in enumerate(files):
                        file = file.strip()
                        sample_generator = paddle.reader.xmap_readers(self.parse_line, \
                            functools.partial(self.read_file, file=file, task_index=task_index), 8, 2000)
                        for sample in sample_generator():
                            if not self.is_test:
                                self.current_epoch = epoch + 1
                                self.current_file_index = index + 1
                                self.current_file = file
                                self.total_file = len(files)
                                yield sample
                            else:
                                cap_id = sample["ids"][1]
                                for image_id in self._test_image_list:
                                    line_json = copy.deepcopy(sample)
                                    _, _, image_embeddings, number_box, image_loc = self._image_feature_dict[image_id]
                                    line_json["image_embeddings"] = image_embeddings
                                    line_json["image_loc"] = image_loc
                                    line_json["ids"][0] = image_id
                                    yield line_json

            def batch_reader(reader, batch_size):
                """ batch reader """
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
                if self.is_test:
                    outs = 1
                else:
                    outs = len(self._negative_schema)+1
                yield prepare_flickr_data(
                    batch_data,
                    total_token_num,
                    task_index,
                    len(self.task_group),
                    voc_size=self.voc_size,
                    pad_id=self.pad_id,
                    cls_id=self.cls_id,
                    sep_id=self.sep_id,
                    mask_id=self.mask_id,
                    outs=outs,
                    return_input_mask=True,
                    return_max_len=False,
                    return_num_token=False)

        return wrapper


if __name__ == "__main__":
    pass
