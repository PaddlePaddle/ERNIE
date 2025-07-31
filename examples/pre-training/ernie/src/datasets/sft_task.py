# !/usr/bin/env python3
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
# copy from `http://gitlab.baidu.com/gongenlei/PaddleNLP/blob/eb_data_stream/model_zoo/ernie-bot/data.py`

import json

import numpy as np
import paddle
from paddle.io import IterableDataset
from src.datasets.sft_task_reader.finetuning import KnowledgeBasedSFTReader


def erniebot_reader(data_path):
    with open(data_path, "r", encoding="utf-8") as fp:
        for line in fp:
            yield json.loads(line.strip())


def create_pyreader(config_dataset):
    if config_dataset["dataset_name"] == "KnowledgeBasedSFTReader":
        data_reader = KnowledgeBasedSFTReader(**config_dataset)
    else:
        raise ValueError(f"Unknown dataset: {config_dataset['dataset_name']}")
    return data_reader


class KnoverDataset(IterableDataset):
    """The dataset wrapper of a generator. 在原来KnoverDataset上修改而来"""

    def __init__(self, generator, batch_size, ignored_index, input_keys=None):
        self._generator = generator
        self.batch_size = batch_size
        self.ignored_index = ignored_index
        if input_keys is None:
            self.input_keys = [
                "input_ids",
                "position_ids",
                "attention_mask",
                "labels",
                "loss_mask",
            ]
        else:
            self.input_keys = input_keys

    def __iter__(self):
        buf = []
        for batch in self._generator():
            # batch = list(batch)
            # 内层reader走 in-tokens策略，返回的batchsize只有1
            batch = [np.squeeze(b, 0) for b in batch]
            batch = dict(zip(self.input_keys, batch))
            mask = batch.pop("loss_mask").astype("bool")
            batch.pop("position_ids")
            batch["labels"][~mask] = self.ignored_index  # we use ignored-index
            buf.append(batch)
            if len(buf) == self.batch_size:
                yield buf  # list of dict
                buf = []
        # Drop last
        # if buf:
        #     yield buf
        #     buf = []


def collate_fn(batch_list, input_keys=None):
    if input_keys is None:
        input_keys = [
            "input_ids",
            "position_ids",
            "attention_mask",
            "labels",
            "loss_mask",
        ]
    return_list = []
    for n in range(len(batch_list[0])):
        return_list.append(paddle.to_tensor(np.concatenate([t[n] for t in batch_list])))
    input_dict = dict(zip(input_keys, return_list))
    return input_dict
