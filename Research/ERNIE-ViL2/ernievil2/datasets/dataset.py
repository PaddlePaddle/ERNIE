""" Datasets """
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import io
import math
import os
import sys
from multiprocess import Pool, RLock
import time
import random
import numpy as np
import paddle.distributed as dist
from paddle.io import Dataset, IterableDataset
from ernievil2.utils.env import DATA_HOME
import importlib

__all__ = ['MapDataset', 'load_dataset']
DATASETS_MODULE_PATH = "ernievil2.datasets."

def import_main_class(module_path):
    """
    Import a module at module_path and return its DatasetBuilder class.

    """
    module_path = DATASETS_MODULE_PATH + module_path
    module = importlib.import_module(module_path)
    main_cls_type = Dataset

    # Find the main class in our imported module
    module_main_cls = None
    for name, obj in module.__dict__.items():

        if isinstance(obj, type) and issubclass(obj, main_cls_type):
            if name == 'Dataset':
                continue

            module_main_cls = obj
            break

    return module_main_cls


def load_dataset(filetype, fileslist):
    """ load_dataset """
    reader_cls = import_main_class(filetype)
    reader_instance = reader_cls()
    dataset = []
    examples = reader_instance._read(fileslist)

    # Then some validation.
    if not isinstance(examples, list):
        examples = list(examples)

    if not examples:
        raise ValueError(
            "No instances were read from the given filepath {}. "
            "Is the path correct?".format(fileslist))
    
    dataset.extend(examples)

    return dataset


class MapDataset(Dataset):
    """
    Wraps a map-style dataset-like object as an instance of `MapDataset`, and equips it 
    with `map` and other utility methods. All non-magic methods of the raw object
    are also accessible.

    Args:
        data (list|Dataset): An object with `__getitem__` and `__len__` methods. It could 
            be a list or a subclass of `paddle.io.Dataset`.
        kwargs (dict, optional): Other information to be passed to the dataset. 

    """

    def __init__(self, data, **kwargs):
        self.data = data
        self._transform_pipline = []
        self.new_data = self.data

        self.label_list = kwargs.pop('label_list', None)
        self.vocab_info = kwargs.pop('vocab_info', None)

    def _transform(self, data):
        for fn in self._transform_pipline:
            data = fn(data)
        return data

    def __getitem__(self, idx):
        """
        Basic function of `MapDataset` to get sample from dataset with a given 
        index.
        """
        return self._transform(self.new_data[
            idx]) if self._transform_pipline else self.new_data[idx]

    def __len__(self):
        """
        Returns the number of samples in dataset.
        """
        return len(self.new_data)

    def filter(self, fn, num_workers=0):
        """
        Filters samples by the filter function and uses the filtered data to
        update this dataset.

        Args:
            fn (callable): A filter function that takes a sample as input and
                returns a boolean. Samples that return False would be discarded.
            num_workers(int, optional): Number of processes for multiprocessing. If 
                set to 0, it doesn't use multiprocessing. Defaults to `0`.
        """
        assert num_workers >= 0, "num_workers should be a non-negative value"
        if num_workers > 0:
            with Pool(num_workers, initargs=(RLock(), )) as pool:

                def filter_shard(num_workers, index, fn):
                    """ filter_shard """
                    self.shard(
                        num_shards=num_workers, index=index, contiguous=True)
                    self._filter(fn=fn)
                    return self

                kwds_per_shard = [
                    dict(
                        num_workers=num_workers, index=rank, fn=fn)
                    for rank in range(num_workers)
                ]
                results = [
                    pool.apply_async(
                        filter_shard, kwds=kwds) for kwds in kwds_per_shard
                ]
                transformed_shards = [r.get() for r in results]

                self.new_data = []
                for i in range(num_workers):
                    self.new_data += transformed_shards[i].new_data
            return self
        else:
            return self._filter(fn)

    def _filter(self, fn):
        self.new_data = [
            self.new_data[idx] for idx in range(len(self.new_data))
            if fn(self.new_data[idx])
        ]
        return self

    def shard(self, num_shards=None, index=None, contiguous=False):
        """
        Split the dataset into `num_shards` pieces. Note that the size of each
        shard might be different because the original dataset may not be evenly
        divisible.

        Args:
            num_shards (int, optional): An integer representing the number of
                data shards. If None, `num_shards` would be number of trainers.
                Defaults to `None`.
            index (int, optional): An integer representing the index of the
                current shard. If None, `index` would be the current trainer rank
                id. Defaults to `None`.
            contiguous: (bool, optional): If true, contiguous chunks of data 
                will be select for sharding. And total number of examples will 
                be the same. Otherwise each shard will contain all examples of 
                dataset whose index mod `num_shards` = `index`. Defaults to `False`.
        """
        if num_shards is None:
            num_shards = dist.get_world_size()
        if index is None:
            index = dist.get_rank()

        if contiguous:
            div = len(self) // num_shards
            mod = len(self) % num_shards
            start = div * index + min(index, mod)
            end = start + div + (1 if index < mod else 0)
            self.new_data = self.new_data[start:end]
        else:
            num_samples = int(math.ceil(len(self.new_data) * 1.0 / num_shards))
            self.new_data = [
                self.new_data[idx] for idx in range(len(self.new_data))
                if idx % num_shards == index
            ]

        return self

    def map(self, fn, lazy=True, batched=False, num_workers=0):
        """
        Performs specific function on the dataset to transform and update every sample.

        Args:
            fn (callable): Transformations to be performed. It receives single
                sample as argument if batched is False. Else it receives all examples.
            lazy (bool, optional): If True, transformations would be delayed and
                performed on demand. Otherwise, transforms all samples at once. Note that 
                if `fn` is stochastic, `lazy` should be True or you will get the same
                result on all epochs. Defaults to False.
            batched(bool, optional): If True, transformations would take all examples as 
                input and return a collection of transformed examples. Note that if set 
                True, `lazy` option would be ignored. Defaults to False.
            num_workers(int, optional): Number of processes for multiprocessing. If 
                set to 0, it doesn't use multiprocessing. Note that if set to positive
                value, `lazy` option would be ignored. Defaults to 0.
        """

        assert num_workers >= 0, "num_workers should be a non-negative value"
        if num_workers > 0:
            with Pool(num_workers, initargs=(RLock(), )) as pool:

                def map_shard(num_workers, index, fn, batched):
                    """ map_shard """
                    self.shard(
                        num_shards=num_workers, index=index, contiguous=True)
                    self._map(fn=fn, lazy=False, batched=batched)
                    return self

                kwds_per_shard = [
                    dict(
                        num_workers=num_workers,
                        index=rank,
                        fn=fn,
                        batched=batched) for rank in range(num_workers)
                ]
                results = [
                    pool.apply_async(
                        map_shard, kwds=kwds) for kwds in kwds_per_shard
                ]
                transformed_shards = [r.get() for r in results]

                self.new_data = []
                for i in range(num_workers):
                    self.new_data += transformed_shards[i].new_data

            return self
        else:
            return self._map(fn, lazy=lazy, batched=batched)

    def _map(self, fn, lazy=True, batched=False):
        if batched:
            self.new_data = fn(self.new_data)
        elif lazy:
            self._transform_pipline.append(fn)
        else:
            self.new_data = [
                fn(self.new_data[idx]) for idx in range(len(self.new_data))
            ]
        return self