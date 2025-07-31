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


__all__ = []


import gzip
import os
import json

import numpy as np
from paddle.io import IterableDataset
import paddle.distributed as dist

from src.utils import logger


class LineIterableDataset(IterableDataset):
    def __init__(self, path):
        super().__init__()
        self.path = os.path.expanduser(path)

    def __iter__(self):
        if self.path.endswith(".gz"):
            with gzip.open(self.path, "r") as f:
                for line in f:
                    line = line.decode("utf8")
                    yield line
        else:
            with open(self.path) as f:
                for line in f:
                    yield line


class MultiPartIterableDataset(IterableDataset):
    def __init__(self, inner_dataset_builder, path, max_num_dataiter, rng=None):
        super().__init__()
        self.path_list = [
            line.strip() for line in open(os.path.expanduser(path)).readlines()
        ]
        if len(self.path_list) == 0:
            raise RuntimeError("len(self.path_list) = 0")

        self.inner_dataset_list = []
        for i, path in enumerate(self.path_list):
            logger.info(
                f"MultiPartDataset load inner dataset [{i} / {len(self.path_list)}]"
            )
            inner_dataset = inner_dataset_builder(path)
            self.inner_dataset_list.append(inner_dataset)

        self.max_num_dataiter = max_num_dataiter

        self.rng = rng or np.random.RandomState()

    def __iter__(self):
        inner_dataset_list = list(self.inner_dataset_list)
        self.rng.shuffle(inner_dataset_list)
        dataiter_list = []
        while True:
            while (
                len(dataiter_list) < self.max_num_dataiter
                and len(inner_dataset_list) > 0
            ):
                inner_dataset = inner_dataset_list.pop(0)
                dataiter_list.append(iter(inner_dataset))
            index = self.rng.choice(range(len(dataiter_list)))
            dataiter = dataiter_list[index]
            try:
                yield next(dataiter)
            except StopIteration:
                del dataiter
                del dataiter_list[index]


class DistributedShardingIterableDataset(IterableDataset):
    def __init__(self, inner_dataset):
        super().__init__()
        self.inner_dataset = inner_dataset

    def __iter__(self):
        world_size = dist.get_world_size()  # 本次训练的全部卡数
        rank = dist.get_rank()  # 当前所在卡的序号
        for i, sample in enumerate(self.inner_dataset):
            if i % world_size == rank:
                yield sample


class InfiniteIterableDataset(IterableDataset):
    def __init__(self, inner_dataset):
        super().__init__()
        self.inner_dataset = inner_dataset

    def __iter__(self):
        while True:
            for sample in iter(self.inner_dataset):
                yield sample


class MultiSourceIterableDataset(IterableDataset):
    def __init__(self, source_list, inner_dataset_builder, rng=None):
        super().__init__()
        if isinstance(source_list, str):
            with open(os.path.expanduser(str(source_list))) as f:
                source_list = [line.strip() for line in f]
        elif isinstance(source_list, (list, tuple)):
            source_list = list(source_list)
        else:
            raise RuntimeError(f"invalid type for source_list: {type(source_list)}")

        self.path_list = []
        self.weight_list = []  # 暂时没用上
        for source in source_list:
            if isinstance(source, str):
                tokens = source.split(" ", maxsplit=1)
            elif isinstance(source, (list, tuple)):
                tokens = source
            else:
                raise RuntimeError(f"invalid type for source: {type(source)}")
            if len(tokens) == 1:
                self.path_list.append(source)
            elif len(tokens) == 2:
                self.path_list.append(tokens[0])
                self.weights_list.append(float(tokens[0]))
            else:
                raise RuntimeError(f"invalid token number for source: {source}")
        if len(self.weight_list) == 0:
            self.weight_list = None
            self.normalized_weight_list = None
        else:
            weight_sum = sum(self.weight_list)
            self.normalized_weight_list = [
                weight / weight_sum for weight in self.weight_list
            ]

        if self.weight_list is not None and len(self.path_list) != len(
            self.weight_list
        ):
            raise RuntimeError(
                f"weight_list is not empty (len = {len(self.weight_list)}), "
                f"but mismatch with path_list (len={len(self.path_list)})"
            )

        self.inner_dataset_list = []
        for i, path in enumerate(self.path_list):
            logger.info(
                f"MultiSourceDataset load inner dataset [{i} / {len(self.path_list)}"
            )
            inner_dataset = inner_dataset_builder(path)
            self.inner_dataset_list.append(inner_dataset)

        self.rng = rng or np.random.RandomState()

    def __iter__(self):
        iter_inner_dataset_list = [
            iter(inner_dataset) for inner_dataset in self.inner_dataset_list
        ]
        while True:
            iter_inner_dataset = self.rng.choice(
                iter_inner_dataset_list, self.weight_list
            )
            yield next(iter_inner_dataset)


class ShuffleBufferIterableDataset(IterableDataset):
    def __init__(self, inner_dataset, shuffle_buffer_size, rng=None):
        super().__init__()
        self.inner_dataset = inner_dataset
        self.shuffle_buffer_size = shuffle_buffer_size
        self.rng = rng or np.random.RandomState()

    def __iter__(self):
        buffer = []
        for sample in iter(self.inner_dataset):
            buffer.append(sample)
            self.rng.shuffle(buffer)
            if len(buffer) >= self.shuffle_buffer_size:
                sample = buffer.pop(0)
                yield sample
        for sample in buffer:
            yield sample


class DocumentIterableDataset(IterableDataset):
    def __init__(self, inner_dataset, seqlen, overlap, tokenizer):
        super().__init__()
        self.inner_dataset = inner_dataset
        self.seqlen = seqlen
        self.overlap = overlap
        self.stride = self.seqlen - self.overlap
        self.tokenizer = tokenizer

    def __iter__(self):
        token_ids = []
        for line in iter(self.inner_dataset):
            info = json.loads(line)
            content = info["content"]
            if content == "":
                continue
            token_ids.extend(self.tokenizer(content + "</s>").input_ids)
            while len(token_ids) > self.seqlen:
                sample = dict(
                    ids=token_ids[: self.seqlen],
                    sids=None,
                    task="lm",
                    labels=None,
                )
                yield self.transform_sample(sample)
                token_ids = token_ids[self.stride :]

    def transform_sample(self, sample):
        assert sample["labels"] is None
        _tokens = list(sample["ids"])
        tokens, lm_labels = _tokens[:-1], _tokens[1:]
        lm_labels = [self.tokenizer.ignored_index] * len(
            lm_labels[: self.overlap]
        ) + lm_labels[self.overlap :]
        assert len(lm_labels) == len(tokens)
        token_ids = np.array(tokens, dtype="int64")
        lm_labels = np.array(lm_labels, dtype="int64")
        sample = dict(input_ids=token_ids, labels=lm_labels)
        return sample


def create_pretrain_iterable_dataset(
    source_list,
    seqlen,
    overlap,
    tokenizer,
    max_num_dataiter=10,
    shuffle_buffer_size=100,
    rng=None,
):
    line_dataset_builder = lambda path: LineIterableDataset(path)
    multi_part_dataset_builder = lambda path: MultiPartIterableDataset(
        line_dataset_builder, path, max_num_dataiter, rng
    )
    dist_dataset_builder = lambda path: DistributedShardingIterableDataset(
        multi_part_dataset_builder(path)
    )
    infinite_dataset_builder = lambda path: InfiniteIterableDataset(
        dist_dataset_builder(path)
    )
    dataset = MultiSourceIterableDataset(source_list, infinite_dataset_builder, rng)
    dataset = ShuffleBufferIterableDataset(dataset, shuffle_buffer_size)
    dataset = DocumentIterableDataset(
        dataset, seqlen=seqlen, overlap=overlap, tokenizer=tokenizer
    )
    return dataset
