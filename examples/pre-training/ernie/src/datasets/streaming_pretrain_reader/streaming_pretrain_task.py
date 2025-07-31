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

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np

try:
    from collections import Sequence
except ImportError:
    pass


from paddle.io import IterableDataset

from src.datasets.streaming_pretrain_reader.pretraining import ErnieDataReader


logger = logging.getLogger(__name__)


class KnoverDataset(IterableDataset):
    """The dataset wrapper of a generator. 在原来KnoverDataset上修改而来"""

    def __init__(self, generator, batch_size, ignored_index, pad_id, input_keys=None):
        self._generator = generator
        self.batch_size = batch_size
        self.ignored_index = ignored_index
        self.pad_id = pad_id
        if input_keys is None:
            self.input_keys = ["input_ids", "labels"]
        else:
            self.input_keys = input_keys

    def __iter__(self):
        buf = []
        for batch in self._generator():
            src_id, _, _, _, mask_label, _, _, _, _, _, _, _, _, _ = (
                batch  # eb35 reader return 14 segements
            )
            if self.pad_id >= 0:
                pads = np.where(src_id == self.pad_id)
                if pads:
                    mask_label[pads] = self.ignored_index
            # 内层reader走 in-tokens策略，返回的batchsize只有1
            batch = [np.squeeze(b, 0) for b in [src_id, mask_label]]
            batch = dict(zip(self.input_keys, batch))
            buf.append(batch)
            if len(buf) == self.batch_size:
                yield buf  # list of dict
                buf = []


def create_pyreader(config_dataset, **config_kwargs):
    """create_pyreader"""
    config_dataset.update(config_kwargs)
    if "num_samples" in config_dataset:
        config_dataset.pop("num_samples")
    data_reader = ErnieDataReader(**config_dataset)
    return data_reader


if __name__ == "__main__":
    pass
