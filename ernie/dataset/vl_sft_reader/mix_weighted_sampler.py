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

"""Mix Example Set"""

import logging

from paddle.io import IterableDataset

logger = logging.getLogger(__name__)

DATATYPE_2_ID = {"mm": 0, "lm": 1, "audio": 2}


class MixExampleSetJsonMap(IterableDataset):
    """
    Mix Example Set Json
    """

    def __init__(
        self,
        args,
        mm_weights: float,
        lm_weights: float,
        lm_example_set=None,
        mm_example_set=None,
    ) -> None:
        self.lm_example_set = lm_example_set
        self.mm_example_set = mm_example_set
        self.lm_weights = lm_weights / (lm_weights + mm_weights)
        self.mm_weights = mm_weights / (lm_weights + mm_weights)
        self.num_samples = 0
        self.args = args

    def _transform_lm_example(self, instance):
        """A reformatting is required to
        bridge the discrepancy in example formats between text and image data

        Args:
            ex (_type_): _description_
        """
        instance["part_id"] = instance["src_id"]
        instance["image"] = 0
        instance["invalid"] = 0
        instance["data_type"] = DATATYPE_2_ID["h5lm"]
        return instance

    def __len__(self):
        return (
            len(self.lm_example_set)
            if self.lm_example_set is not None
            else 0 + len(self.mm_example_set) if self.mm_example_set is not None else 0
        )

    def __getitem__(self, idx):
        """
        Basic function of `MapDataset` to get sample from dataset with a given
        index.
        """
        return self.mm_example_set[idx]


class MixExampleSetJson(IterableDataset):
    """
    Mix Example Set Json
    """

    def __init__(
        self,
        args,
        mm_weights: float,
        lm_weights: float,
        lm_example_set=None,
        mm_example_set=None,
    ) -> None:
        self.lm_example_set = lm_example_set
        self.mm_example_set = mm_example_set
        self.lm_weights = lm_weights / (lm_weights + mm_weights)
        self.mm_weights = mm_weights / (lm_weights + mm_weights)
        self.num_samples = 0
        self.args = args

    def _transform_lm_example(self, instance):
        """A reformatting is required to
        bridge the discrepancy in example formats between text and image data

        Args:
            ex (_type_): _description_
        """
        instance["part_id"] = instance["src_id"]
        instance["image"] = 0
        instance["invalid"] = 0
        instance["data_type"] = DATATYPE_2_ID["h5lm"]
        return instance

    def __len__(self):
        return (
            len(self.lm_example_set)
            if self.lm_example_set is not None
            else 0 + len(self.mm_example_set) if self.mm_example_set is not None else 0
        )

    def __iter__(self):
        if self.lm_example_set is not None and self.num_samples < len(
            self.lm_example_set
        ):
            for example in self.lm_example_set:
                yield self._transform_lm_example(example)
                self.num_samples += 1
        else:
            # cur = 0
            for example in self.mm_example_set:
                yield example
                # 这里这个if判断有问题，每次iter来自的数据源不同
                # if cur % self.args.pp_need_data_degree == self.args.pipeline_parallel_rank:
                # import os, threading
                # with open('process_info.txt', 'a+') as f: f.write(f"cur: {cur}, PID: {os.getpid()}, TID: {threading.current_thread().ident}\n")
                # 检查这里的cur，每个need data rank看看
                # print("accept: ", example, "cur: ", cur, "pp: ", self.args.pp_need_data_degree, "rank: ", self.args.pipeline_parallel_rank, "PID:", os.getpid())
                # cur += 1
            self.num_samples = 0
