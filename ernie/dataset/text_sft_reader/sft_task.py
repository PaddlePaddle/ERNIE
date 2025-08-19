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

"""This module provides readers that read data from files."""

import numpy as np
from paddle.io import IterableDataset

from .finetuning import KnowledgeBasedSFTReader


DATATYPE_2_ID = {"mm": 0, "lm": 1, "audio": 2}


def create_pyreader(config_dataset):
    """
    Create the corresponding data reader based on the configuration.

    Args:
        config_dataset (dict): Dataset configuration dictionary,
        containing the dataset name and other related configurations.

    Returns:
        data_reader (BaseReader): The data reader instance created based on the configuration.

    Raises:
        ValueError: If the dataset name in the configuration is unknown.
    """
    if config_dataset["dataset_name"] == "KnowledgeBasedSFTReader":
        data_reader = KnowledgeBasedSFTReader(**config_dataset)
    else:
        raise ValueError(f"Unknown dataset: {config_dataset['dataset_name']}")
    return data_reader


class KnoverDataset(IterableDataset):
    """The dataset wrapper of a generator"""

    def __init__(
        self,
        generator,
        batch_size,
        ignored_index,
        task_group=None,
        input_keys=None,
        use_mem_eff_attn=False,
    ):
        self._generator = generator
        self.batch_size = batch_size
        self.ignored_index = ignored_index
        self.task_group = task_group
        if input_keys is None:
            self.input_keys = [
                "input_ids",
                "position_ids",
                "attention_mask",
                "inbatch_pack_offset",
                "labels",
                "loss_mask",
                "task_ids",
                "exact_total_task_ids",
            ]
        else:
            self.input_keys = input_keys
        self.use_mem_eff_attn = use_mem_eff_attn

    def __iter__(self):
        buf = []
        for batch in self._generator():
            batch = [np.squeeze(b, 0) for b in batch]
            batch = dict(zip(self.input_keys, batch))
            mask = batch.pop("loss_mask").astype("bool")

            if self.use_mem_eff_attn:
                batch.pop("attention_mask")

            batch["labels"][~mask] = self.ignored_index  # we use ignored-index
            batch["token_type_ids"] = np.zeros(batch["input_ids"].shape[0] + 1).astype(
                "int64"
            )
            batch["data_type"] = np.array(DATATYPE_2_ID["lm"]).astype("int64")
            buf.append(batch)
            if len(buf) == self.batch_size:
                yield buf  # list of dict
                buf = []
