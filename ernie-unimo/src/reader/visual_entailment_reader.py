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
"""data reader for multimodal pretraining"""

from __future__ import print_function
from __future__ import division

import json
import base64
import os
import numpy as np
import gzip
import six
import functools
import paddle.fluid as fluid
from reader.batching import pad_feature_data, pad_batch_data


class ClassifyReader(object):
    """ClassifyReader"""
    def __init__(self,
                 filelist,
                 max_seq_len,
                 tokenizer):

        self.files = open(filelist).readlines()
        self.current_file_index = 0
        self.total_file = len(self.files)
        self.current_file = None
        self.tot_examples_nums = 0

        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.pad_token_id
        self.sep_id = tokenizer.sep_token_id

        self.trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        self.trainer_nums = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))

    def get_num_examples(self):
        """get_num_examples"""
        for index, file_ in enumerate(self.files):
            self.tot_examples_nums += int(os.popen('wc -l '+file_.strip()).read().split()[0])
        return self.tot_examples_nums

    def get_progress(self):
        """return current progress of traning data
        """
        return self.current_epoch, self.current_example, self.current_file_index, self.total_file, self.current_file

    def parse_line(self, line, max_seq_len=512):
        """ parse one line to token_ids, sentence_ids, pos_ids, label
        """
        line = line.strip('\r\n').split(";")

        if len(line) == 14:
            (image_id, data_id, label, token_ids, sent_ids, pos_ids, _, image_w, image_h, \
             number_box, boxes, image_embeddings, _, _) = line
        else:
            raise ValueError("One sample have %d fields!" % len(line))

        def decode_feature(base64_str, size):
            fea_base64 = base64.b64decode(base64_str)
            fea_decode = np.frombuffer(fea_base64, dtype=np.float32)
            shape = size, int(fea_decode.shape[0] / size)
            features = np.resize(fea_decode, shape)
            return features

        token_ids = [int(token) for token in token_ids.split(" ")]
        sent_ids = [int(token) for token in sent_ids.split(" ")]
        pos_ids = [int(token) for token in pos_ids.split(" ")]
        assert len(token_ids) == len(sent_ids) == len(pos_ids), \
            "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids)"

        number_box = int(number_box)
        boxes = decode_feature(boxes, number_box)

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

        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len - 1] + [self.sep_id]
            sent_ids = sent_ids[:max_seq_len]
            pos_ids = pos_ids[:max_seq_len]

        return [token_ids, sent_ids, pos_ids, label, image_loc, image_embeddings, number_box + 1]

    def _prepare_batch_data(self, insts, pad_id=None):
        batch_src_ids = [inst[0] for inst in insts]
        batch_sent_ids = [inst[1] for inst in insts]
        batch_pos_ids = [inst[2] for inst in insts]
        batch_labels = [inst[3] for inst in insts]
        batch_image_loc = [inst[4] for inst in insts]
        batch_image_embedding = [inst[5] for inst in insts]
        batch_image_size = [inst[6] for inst in insts]

        batch_labels = np.array(batch_labels).astype("int64").reshape([-1, 1])

        src_ids, token_mask = pad_batch_data(
            batch_src_ids, pretraining_task='nlu', pad_idx=pad_id, return_input_mask=True)
        sent_ids = pad_batch_data(
            batch_sent_ids, pretraining_task='nlu', pad_idx=pad_id)
        pos_ids = pad_batch_data(
            batch_pos_ids, pretraining_task='nlu', pad_idx=pad_id)


        image_loc = pad_feature_data(batch_image_loc)
        image_embedding, image_mask = pad_feature_data(batch_image_embedding,
                                                       return_mask=True,
                                                       batch_image_size=batch_image_size)

        input_mask = np.concatenate((image_mask, token_mask), axis=1)
        input_mask = np.matmul(input_mask, np.transpose(input_mask, (0, 2, 1)))

        return_list = [
            src_ids, pos_ids, sent_ids, input_mask, image_mask, token_mask,
            image_embedding, image_loc, batch_labels
        ]

        return return_list

    def read_file(self, file):
        """read_file"""
        if file.endswith('.gz'):
            with gzip.open(file, "rt") as f:
                for line in f:
                    parsed_line = self.parse_line(
                        line, max_seq_len=self.max_seq_len)
                    if parsed_line is None:
                        continue
                    yield parsed_line
        else:
            with open(file, "r") as f:
                for line in f:
                    parsed_line = self.parse_line(
                        line, max_seq_len=self.max_seq_len)
                    if parsed_line is None:
                        continue
                    yield parsed_line

    def shuffle_samples(self, sample_generator, buffer=1000):
        """shuffle_samples"""
        samples = []
        try:
            while True:
                while len(samples) < buffer:
                    sample = next(sample_generator)
                    samples.append(sample)
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
                samples = []
        except StopIteration:
            if len(samples) == 0:
                yield None
            else:
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample

    def data_generator(self,
                       batch_size,
                       epoch,
                       phase):
        """
        data_generator
        """
        if phase != "train":
            epoch = 1
        def wrapper():
            """wrapper"""
            def batch_reader():
                """batch_reader"""
                for epoch_index in range(epoch):
                    self.global_rng = np.random.RandomState(epoch_index)
                    self.current_epoch = epoch_index
                    self.current_example = 0

                    if phase == "train":
                        self.global_rng.shuffle(self.files)

                    for index, file_ in enumerate(self.files):
                        self.current_file_index = index + 1
                        self.current_file = file_

                        batch_records = []
                        for sample in self.shuffle_samples(self.read_file(file=file_.strip())):
                            self.current_example = self.current_example + 1
                            if sample is None:
                                continue
                            if len(batch_records) < batch_size:
                                batch_records.append(sample)
                            else:
                                yield self._prepare_batch_data(batch_records, self.pad_id)
                                batch_records = [sample]
                        if batch_records:
                            yield self._prepare_batch_data(batch_records, self.pad_id)

            all_dev_batches = []
            for batch_data in batch_reader():
                if len(all_dev_batches) < self.trainer_nums:
                    all_dev_batches.append(batch_data)
                if len(all_dev_batches) == self.trainer_nums:
                    yield all_dev_batches[self.trainer_id]
                    all_dev_batches = []

            if phase == "train":
                all_dev_batches = all_dev_batches * self.trainer_nums
                np.random.shuffle(all_dev_batches)

            if self.trainer_id < len(all_dev_batches):
                yield all_dev_batches[self.trainer_id]

        return wrapper


if __name__ == "__main__":
    pass
