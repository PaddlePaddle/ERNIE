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

"""SFT utils"""
from dataclasses import dataclass, field
from typing import Optional

from train import DataArgument, SFTTrainingArguments


class DataGenerator:
    """Generates an infinite stream of examples"""

    def __init__(self, data_source):
        """
            Initializes the iterator for a given data source.

        Args:
            data_source : IterableDataset

        Returns:
            None. - Initialization only. No return value.
        """
        self.data_source_iter = iter(data_source)
        self.data_source = data_source

    def __iter__(self):
        """
        Returns:
            Iterator: The iterator object itself.
        """
        return self

    def __next__(self):
        """
        Get the next item from the iterator. If there are no more items left, reset the iterator.

        Returns:
            Any: The next item from the iterator.
        """
        try:
            return next(self.data_source_iter)
        except StopIteration:
            self.data_source_iter = iter(self.data_source)
            return next(self.data_source_iter)


@dataclass
class BuildSFTTrainingArguments(SFTTrainingArguments):
    """TrainingArguments for building SFT MapDataset"""

    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    num_of_gpus: int = field(
        default=1,
        metadata={"help": "The number of GPUs."},
    )
    estimation_output_file: Optional[str] = field(
        default=None, metadata={"help": "The file to save estimation results."}
    )
    pp_degree: int = field(default=1, metadata={"help": "Pipeline parallel degree."})
    sdp_degree: int = field(default=1, metadata={"help": "Sharding parallel degree."})
    tp_degree: int = field(default=1, metadata={"help": "Tensor parallel degree."})


@dataclass
class BuildDataArgument(DataArgument):
    """DataArgument for building SFT MapDataset"""

    dataset_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The output directory where the SFT MapDataset will be written."},
    )
