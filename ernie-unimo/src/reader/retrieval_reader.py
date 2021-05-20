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
"""data reader for image-text retrieval tasks"""

import os
import pickle
import base64
import codecs
import numpy as np
from collections import namedtuple
from reader.batching import pad_feature_data, pad_batch_data


class RetrievalTrainReader(object):
    """RetrievalTrainReader"""
    def __init__(self, tokenizer, args, image_feature_dir, image_caption):
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id
        self.max_seq_len = args.max_seq_len

        self.trainer_id = int(os.getenv("PADDLE_TRAINER_ID", 0))
        self.trainers_num = int(os.getenv("PADDLE_TRAINERS_NUM", 1))

        self.current_example = 0
        self.current_epoch = 0

        self._load_image_feature(image_feature_dir)
        self._load_caption_dict(image_caption)
        self._load_img_id(args.img_id_path)

        if args.samples_num == 20:
            self._negative_schema = ['ei'] * 10 + ['ec'] * 10
            self.outs = len(self._negative_schema) + 1
        else:
            raise ValueError('dont support')

    def _load_caption_dict(self, image_caption):
        '''parse dataset_flickr30k.json which is made by karpathy'''
        self._caption_ids_dict = {}
        self._image_sent_map = {}

        with codecs.open(image_caption, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(";")
                token_ids, sent_ids, pos_ids, image_name, sent_id = line
                token_ids = [int(token) for token in token_ids.split(" ")]
                sent_ids = [int(token) for token in sent_ids.split(" ")]
                pos_ids = [int(token) for token in pos_ids.split(" ")]
                if len(token_ids) > self.max_seq_len:
                    token_ids = [token_ids[0]] + token_ids[1:self.max_seq_len - 1] + [token_ids[-1]]
                    sent_ids = sent_ids[:self.max_seq_len]
                    pos_ids = pos_ids[:self.max_seq_len]
                assert len(token_ids) <= self.max_seq_len, \
                        "token length must be less than max_seq_len"
                assert len(token_ids) == len(sent_ids) == len(pos_ids), \
                        "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids)"

                self._caption_ids_dict[int(sent_id)] = \
                        [token_ids, sent_ids, pos_ids, int(image_name)]
                self._image_sent_map.setdefault(int(image_name), [])
                self._image_sent_map[int(image_name)].append(int(sent_id))
                
        self._train_caption_ids = list(self._caption_ids_dict.keys())
        self._train_image_list = list(self._image_sent_map.keys())

    def _parse_image_line(self, line):
        def decode_feature(base64_str, size):
            """decode_feature"""
            fea_base64 = base64.b64decode(base64_str)
            fea_decode = np.frombuffer(fea_base64, dtype=np.float32)
            shape = size, int(fea_decode.shape[0] / size)
            features = np.resize(fea_decode, shape)
            return features

        items = line.strip('\r\n').split('\t')
        assert len(items) == 7
        img_filename, image_w, image_h, number_box, boxes, image_embeddings, probs = items

        number_box = int(number_box)
        boxes = decode_feature(boxes, number_box)
        probs = decode_feature(probs, number_box)
        image_embeddings = decode_feature(image_embeddings, number_box)
        image_embeddings_cls = np.mean(image_embeddings, axis=0, keepdims=True)
        image_embeddings = np.concatenate([image_embeddings_cls, image_embeddings], 0)
        image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        image_location[:, :4] = boxes
        image_location[:, 4] = (image_location[:, 3] - image_location[:, 1]) * (
                image_location[:, 2] - image_location[:, 0]) / (float(image_w) * float(image_h))
        image_location[:, 0] = image_location[:, 0] / float(image_w)
        image_location[:, 1] = image_location[:, 1] / float(image_h)
        image_location[:, 2] = image_location[:, 2] / float(image_w)
        image_location[:, 3] = image_location[:, 3] / float(image_h)
        g_location = np.array([0, 0, 1, 1, 1])
        image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)
        image_loc = image_location
        cls_prob = np.mean(probs, axis=0, keepdims=True)
        probs = np.concatenate([cls_prob, probs], 0)

        output = namedtuple('output', ["img_filename", "number_box", "image_loc", "probs", "image_embeddings"])
        return output(img_filename=img_filename, 
                number_box=number_box + 1, 
                image_loc=image_loc, 
                probs=probs, 
                image_embeddings=image_embeddings)

    def _load_image_feature(self, data_dir):
        self._image_feature_dict = {}
        for file in os.listdir(data_dir):
            file = os.path.join(data_dir, file)
            with codecs.open(file, 'r', encoding='utf-8') as fr:
                for line in fr.readlines():
                    items = self._parse_image_line(line)
                    self._image_feature_dict[int(items[0])] = items[1:]

    def _load_img_id(self, img_id_path):
        self.imgname2id = {}
        self.id2imgname = {}
        with codecs.open(img_id_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                items = line.strip('\r\n').split('\t')
                self.imgname2id[int(items[0])] = int(items[1])
                self.id2imgname[int(items[1])] = int(items[0])
    
    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def _prepare_batch_data(self, insts):
        """generate batch and pad"""
        batch_src_ids = [inst["token_ids"][out] for inst in insts for out in range(self.outs)]
        batch_sent_ids = [inst["sent_ids"][out] for inst in insts for out in range(self.outs)]
        batch_pos_ids = [inst["pos_ids"][out] for inst in insts for out in range(self.outs)]
        batch_image_loc = [inst["image_loc"][out] for inst in insts for out in range(self.outs)]
        batch_image_embedding = [inst["image_embeddings"][out] for inst in insts for out in range(self.outs)]
        batch_image_size = [inst["number_box"][out] for inst in insts for out in range(self.outs)]

        batch_size = int(len(batch_src_ids) / self.outs)
        label = np.array([[0]] * batch_size, dtype="int64")
        ids = np.array([[0, 0]] * batch_size, dtype="int64")

        padded_token_ids, token_mask = pad_batch_data(
            batch_src_ids, pretraining_task='nlu', pad_idx=self.pad_id, return_input_mask=True)
        padded_sent_ids = pad_batch_data(
            batch_sent_ids, pretraining_task='nlu', pad_idx=self.pad_id)
        padded_pos_ids = pad_batch_data(
            batch_pos_ids, pretraining_task='nlu', pad_idx=self.pad_id)

        padded_image_embedding, image_mask = pad_feature_data(batch_image_embedding,
                                                       return_mask=True,
                                                       batch_image_size=batch_image_size)
        padded_image_loc = pad_feature_data(batch_image_loc)

        input_mask = np.concatenate((image_mask, token_mask), axis=1)
        input_mask = np.matmul(input_mask, np.transpose(input_mask, (0, 2, 1)))
        return_list = [
            padded_token_ids, padded_pos_ids, padded_sent_ids, input_mask,
            padded_image_embedding, padded_image_loc, label, ids
        ]
        return return_list

    def get_num_examples(self):
        """get_num_examples"""
        cap_len = len(self._train_caption_ids)
        img_len = len(self._train_image_list)
        total_samples = cap_len
        return total_samples, cap_len, img_len

    def process_vl(self, sent_id):
        """trans the orgin tokens to the wanted tokens"""
        captions_pos = self._caption_ids_dict[sent_id]
        image_name = captions_pos[-1]
        image_id = self.imgname2id[image_name]
        number_box, image_loc, _, image_embeddings = self._image_feature_dict[image_name]

        images = [[image_embeddings, number_box, image_loc]]
        captions = [captions_pos]

        for item in self._negative_schema:
            if item[0] == "e":
                while True:
                    image_name_neg = self.neg_rng.choice(self._train_image_list)
                    if image_name_neg != image_name:
                        break
            else:
                print("error negative schema")
                exit()

            if item[1] == "i":
                number_box_neg, image_loc_neg, _, image_embeddings_neg = self._image_feature_dict[image_name_neg]
                captions.append(self._caption_ids_dict[sent_id])
                images.append([image_embeddings_neg, number_box_neg, image_loc_neg])
            elif item[1] == "c":
                sent_id_neg = self.neg_rng.choice(self._image_sent_map[image_name_neg])
                captions.append(self._caption_ids_dict[sent_id_neg])
                images.append([image_embeddings, number_box, image_loc])
            else:
                print("error negative schema")
                exit()

        token_ids_list, sent_ids_list, pos_ids_list, _ = zip(*captions)
        image_embeddings_list, number_box_list, image_loc_list = zip(*images)

        sample_json = {
            "token_ids": token_ids_list,
            "sent_ids": sent_ids_list,
            "pos_ids": pos_ids_list,
            "image_loc": image_loc_list,
            "image_embeddings": image_embeddings_list,
            "number_box": number_box_list,
        }
        return sample_json

    def read_caption_id(self):
        """read_caption_id"""
        self.global_rng.shuffle(self._train_caption_ids)
        for index, item in enumerate(self._train_caption_ids):
            if index % self.trainers_num != self.trainer_id:
                continue
            yield self.process_vl(item)

    def shuffle_samples(self, sample_generator, buffer=128):
        """shuffle_samples"""
        samples = []
        try:
            while True:
                while len(samples) < buffer:
                    sample = next(sample_generator)
                    samples.append(sample)
                for sample in samples:
                    yield sample
                samples = []
        except StopIteration:
            if len(samples) == 0:
                yield None
            else:
                for sample in samples:
                    yield sample

    def data_generator(self):
        """data_generator"""
        def wrapper():
            """wrapper"""
            for epoch_index in range(self.epoch):
                self.global_rng = np.random.RandomState(epoch_index)
                self.neg_rng = np.random.RandomState(epoch_index)
                self.current_epoch = epoch_index
                batch_records = []
                self.current_example = 0
                for sample in self.shuffle_samples(self.read_caption_id()):
                    self.current_example = self.current_example + 1
                    if len(batch_records) < self.batch_size:
                        batch_records.append(sample)
                    if len(batch_records) == self.batch_size:
                        yield self._prepare_batch_data(batch_records)
                        batch_records = []
                if batch_records:
                    yield self._prepare_batch_data(batch_records)
        return wrapper


class RetrievalTestReader(object):
    """RetrievalTrainReader"""
    def __init__(self, tokenizer, args, image_feature_dir, image_caption):
        self.batch_size = args.test_batch_size
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id
        self.max_seq_len = args.max_seq_len
        self.trainer_id = int(os.getenv("PADDLE_TRAINER_ID", 0))
        self.trainers_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        self.current_example = 0

        self._load_image_feature(image_feature_dir)
        self._load_caption_dict(image_caption)

    def _load_caption_dict(self, image_caption):
        '''parse dataset_flickr30k.json which is made by karpathy'''
        self._caption_ids_dict = {}
        self._image_sent_map = {}

        with codecs.open(image_caption, 'r', encoding='utf-8') as f:
            cnt = 0
            for line in f:
                line = line.strip().split(";")
                token_ids, sent_ids, pos_ids, image_name, sent_id = line
                token_ids = [int(token) for token in token_ids.split(" ")]
                sent_ids = [int(token) for token in sent_ids.split(" ")]
                pos_ids = [int(token) for token in pos_ids.split(" ")]
                if len(token_ids) > self.max_seq_len:
                    token_ids = [token_ids[0]] + token_ids[1:self.max_seq_len - 1] + [token_ids[-1]]
                    sent_ids = sent_ids[:self.max_seq_len]
                    pos_ids = pos_ids[:self.max_seq_len]
                assert len(token_ids) <= self.max_seq_len, \
                        "token length must be less than max_seq_len"
                assert len(token_ids) == len(sent_ids) == len(pos_ids), \
                        "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids)"

                self._caption_ids_dict[int(sent_id)] = \
                        [token_ids, sent_ids, pos_ids, int(image_name)]
                self._image_sent_map.setdefault(int(image_name), [])
                self._image_sent_map[int(image_name)].append(int(sent_id))

        self._train_caption_ids = list(self._caption_ids_dict.keys())
        self._train_image_list = list(self._image_sent_map.keys())

    def _parse_image_line(self, line):
        def decode_feature(base64_str, size):
            """decode_feature"""
            fea_base64 = base64.b64decode(base64_str)
            fea_decode = np.frombuffer(fea_base64, dtype=np.float32)
            shape = size, int(fea_decode.shape[0] / size)
            features = np.resize(fea_decode, shape)
            return features

        items = line.strip('\r\n').split('\t')
        assert len(items) == 7
        img_filename, image_h, image_w, number_box, boxes, image_embeddings, probs = items

        number_box = int(number_box)
        boxes = decode_feature(boxes, number_box)
        probs = decode_feature(probs, number_box)
        image_embeddings = decode_feature(image_embeddings, number_box)
        image_embeddings_cls = np.mean(image_embeddings, axis=0, keepdims=True)
        image_embeddings = np.concatenate([image_embeddings_cls, image_embeddings], 0)
        image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        image_location[:, :4] = boxes
        image_location[:, 4] = (image_location[:, 3] - image_location[:, 1]) * (
                image_location[:, 2] - image_location[:, 0]) / (float(image_w) * float(image_h))
        image_location[:, 0] = image_location[:, 0] / float(image_w)
        image_location[:, 1] = image_location[:, 1] / float(image_h)
        image_location[:, 2] = image_location[:, 2] / float(image_w)
        image_location[:, 3] = image_location[:, 3] / float(image_h)
        g_location = np.array([0, 0, 1, 1, 1])
        image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)
        image_loc = image_location
        cls_prob = np.mean(probs, axis=0, keepdims=True)
        probs = np.concatenate([cls_prob, probs], 0)

        output = namedtuple('output', ["img_filename", "number_box", "image_loc", "probs", "image_embeddings"])
        return output(img_filename=img_filename, 
                number_box=number_box + 1, 
                image_loc=image_loc, 
                probs=probs, 
                image_embeddings=image_embeddings)

    def _load_image_feature(self, data_dir):
        self._image_feature_dict = {}
        for file in os.listdir(data_dir):
            file = os.path.join(data_dir, file)
            with codecs.open(file, 'r', encoding='utf-8') as fr:
                for line in fr.readlines():
                    items = self._parse_image_line(line)
                    self._image_feature_dict[int(items[0])] = items[1:]

    def _prepare_batch_data(self, insts):
        """generate batch and pad"""
        batch_src_ids = [inst["token_ids"] for inst in insts]
        batch_sent_ids = [inst["sent_ids"] for inst in insts]
        batch_pos_ids = [inst["pos_ids"] for inst in insts]
        batch_image_loc = [inst["image_loc"] for inst in insts]
        batch_image_embedding = [inst["image_embeddings"] for inst in insts]
        batch_image_size = [inst["number_box"] for inst in insts]
        batch_ids = [inst["cur_ids"] for inst in insts]
        batch_labels = [[0]] * len(insts)

        padded_token_ids, token_mask = pad_batch_data(
            batch_src_ids, pretraining_task='nlu', pad_idx=self.pad_id, return_input_mask=True)
        padded_sent_ids = pad_batch_data(
            batch_sent_ids, pretraining_task='nlu', pad_idx=self.pad_id)
        padded_pos_ids = pad_batch_data(
            batch_pos_ids, pretraining_task='nlu', pad_idx=self.pad_id)

        padded_image_embedding, image_mask = pad_feature_data(batch_image_embedding,
                                                       return_mask=True,
                                                       batch_image_size=batch_image_size)
        padded_image_loc = pad_feature_data(batch_image_loc)
        ids = np.array(batch_ids, dtype="int64")
        label = np.array(batch_labels, dtype="int64")
        input_mask = np.concatenate((image_mask, token_mask), axis=1)
        input_mask = np.matmul(input_mask, np.transpose(input_mask, (0, 2, 1)))

        return_list = [
            padded_token_ids, padded_pos_ids, padded_sent_ids, input_mask,
            padded_image_embedding, padded_image_loc, label, ids
        ]
        return return_list

    def get_num_examples(self):
        """get_num_examples"""
        cap_len = len(self._train_caption_ids)
        img_len = len(self._train_image_list)
        total_samples = cap_len
        return total_samples, cap_len, img_len

    def process_vl(self, sent_id):
        """trans the orgin tokens to the wanted tokens"""
        token_ids, sent_ids, pos_ids, image_name = self._caption_ids_dict[sent_id]

        for cur_img_name in self._train_image_list:
            number_box, image_loc, _, image_embeddings = self._image_feature_dict[cur_img_name]
            sample_json = {
                "token_ids": token_ids,
                "sent_ids": sent_ids,
                "pos_ids": pos_ids,
                "image_loc": image_loc,
                "image_embeddings": image_embeddings,
                "number_box": number_box,
                "cur_ids": [cur_img_name, sent_id],
            }
            yield sample_json

    def read_caption_id(self):
        """read_caption_id"""
        for item in self._train_caption_ids:
            sent_id = item
            token_ids, sent_ids, pos_ids, image_name = self._caption_ids_dict[sent_id]

            for cur_img_name in self._train_image_list:
                number_box, image_loc, _, image_embeddings = self._image_feature_dict[cur_img_name]
                sample_json = {
                    "token_ids": token_ids,
                    "sent_ids": sent_ids,
                    "pos_ids": pos_ids,
                    "image_loc": image_loc,
                    "image_embeddings": image_embeddings,
                    "number_box": number_box,
                    "cur_ids": [cur_img_name, sent_id],
                }
                yield sample_json

    def shuffle_samples(self, sample_generator, buffer=128):
        """shuffle_samples"""
        samples = []
        try:
            while True:
                while len(samples) < buffer:
                    sample = next(sample_generator)
                    samples.append(sample)
                for sample in samples:
                    yield sample
                samples = []
        except StopIteration:
            if len(samples) == 0:
                yield None
            else:
                for sample in samples:
                    yield sample

    def data_generator(self):
        """data_generator"""
        def wrapper():
            """"wrapper"""
            def batch_reader():
                """batch_reader"""
                batch_records = []
                self.current_example = 0
                for sample in self.shuffle_samples(self.read_caption_id()):
                    self.current_example = self.current_example + 1
                    if len(batch_records) < self.batch_size:
                        batch_records.append(sample)
                    if len(batch_records) == self.batch_size:
                        yield self._prepare_batch_data(batch_records)
                        batch_records = []
                if batch_records:
                    yield self._prepare_batch_data(batch_records)

            all_dev_batches = []
            for batch_data in batch_reader():
                if len(all_dev_batches) < self.trainers_num:
                    all_dev_batches.append(batch_data)
                if len(all_dev_batches) == self.trainers_num:
                    yield all_dev_batches[self.trainer_id]
                    all_dev_batches = []
            if self.trainer_id < len(all_dev_batches):
                yield all_dev_batches[self.trainer_id]
        return wrapper


if __name__ == '__main__':
    pass
