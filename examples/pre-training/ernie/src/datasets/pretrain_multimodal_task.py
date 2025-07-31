# # -*- coding: utf-8 -*-

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

# # !/usr/bin/env python3
# """
# @author: kebo
# @contact: kebo01@baidu.com

# @version: 1.0
# @file: pretrain_multimodal_task.py
# @time: 2024/08/19 20:16:39
# @Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved

# 这一行开始写关于本文件的说明与解释


# """
# from __future__ import absolute_import, division, print_function, unicode_literals

# import atexit
# import io
# import os
# import re
# import math
# import itertools
# import json
# import logging
# import random
# import traceback
# from typing import List
# from collections import OrderedDict, defaultdict, namedtuple
# from copy import deepcopy
# from time import time

# try:
#     import baidubce.exception
# except (ImportError, ModuleNotFoundError):
#     pass

# import h5py
# import numpy as np
# import paddle
# from src.datasets.pretrain_task import ExampleSetSingleDataSource as _LmExampleSetSingleDataSource
# from src.datasets.pretrain_task import (
#     DatasetHolder,
#     DatasetHolderIniter,
#     IPCServer,
#     IPCH5Resource,
#     IPCH5MetaResource,
# )
# from src.datasets.pretrain_task import create_ipc_h5_resources as create_ipc_lm_h5_resources
# from src.utils.mm_data_utils import bos_path_to_pil_image, MMSpecialTokensConfig
# from src.utils.image_enhance import apply_effect

# from data_processor.steps.image_modification.processor import (
#     ImageModificationProcessor,
#     Example,
#     VisionExample,
#     AudioExample,
#     DATATYPE_2_ID,
#     IDTYPES_2_ID,
# )


# logger = logging.getLogger(__name__)
# logging.getLogger("PIL").setLevel(logging.WARNING)


# class MMDatasetHolder(DatasetHolder):
#     """_summary_

#     Args:
#         DatasetHolder (_type_): _description_
#     """

#     def __init__(self, paths, server_idx, server_num):
#         """_summary_

#         Args:
#             paths (_type_): _description_
#             server_idx (_type_): _description_
#             server_num (_type_): _description_

#         Raises:
#             OSError: _description_
#         """
#         self.fps = {}
#         path_num = len(paths)
#         start_t = time()
#         for idx, path in enumerate(paths):
#             assert path not in self.fps, path

#             ds = h5py.File(path, mode="r")

#             shape = ds["meta"].shape
#             offset_shape = ds["offset"].shape
#             if shape[0] <= 0 or shape[0] >= 1000000000000:
#                 raise OSError
#             self.fps[path] = {
#                 "meta": ds["meta"],
#                 "shape": shape,
#                 "example_offset": jiefix(ds["example_offset"]),
#                 "meta_offset": jiefix(ds["offset"]),
#                 "example_offset_shape": offset_shape,
#                 "ids": ds["ds16"],
#                 "ids_offset": jiefix(ds["ds16_offset"]),
#                 "lossmask": ds["ds16_lossmask"],
#                 "token_type_ids": ds["ds16_tokenwise_type_id"],
#                 "image_type_ids": ds["ds16_imagewise_type_id"],
#                 "image_offsets": jiefix(ds["ds16_imagewise_type_id_offset"]),
#             }
#             end_t = time()
#             logger.info(
#                 f"Done loading {path}, shape: {shape}, in server-{server_idx}/{server_num}, "
#                 f"accumulated time = {end_t - start_t}, progress: {idx}/{path_num}"
#             )
#         end_t = time()
#         logger.info(
#             f"Server-{server_idx}/{server_num} load ends with path number {path_num}, "
#             f"accumulated time = {end_t - start_t}"
#         )


# class MMDatasetHolderIniter(DatasetHolderIniter):
#     """_summary_

#     Args:
#         DatasetHolderIniter (_type_): _description_
#     """

#     def __call__(self, server_idx, server_num):
#         """_summary_

#         Args:
#             server_idx (_type_): _description_
#             server_num (_type_): _description_

#         Returns:
#             _type_: _description_
#         """
#         return MMDatasetHolder(self.paths, server_idx, server_num)


# def create_ipc_mm_h5_resources(paths, num_server):
#     """
#     构造主进程里感知的IPCH5Resource, 并构造对应的IPCServer
#     """
#     n = len(paths)
#     if n <= 0:
#         return []

#     num_server = min(n, num_server)

#     router_keys = [[] for _ in range(num_server)]
#     for i, p in enumerate(paths):
#         router_keys[i % num_server].append(p)

#     init_funcs = [MMDatasetHolderIniter(rk) for rk in router_keys]
#     server = IPCServer(router_keys, init_funcs)
#     atexit.register(lambda: server.close())
#     fps = []
#     for p in paths:
#         tmp = {
#             "shape": IPCH5MetaResource(p, server),
#             "example_offset_shape": IPCH5MetaResource(p, server),
#             "ids": IPCH5Resource(p, "ids", server),
#             "meta": IPCH5Resource(p, "meta", server),
#             "lossmask": IPCH5Resource(p, "lossmask", server),
#             "example_offset": IPCH5Resource(p, "example_offset", server),
#             "meta_offset": IPCH5Resource(p, "meta_offset", server),
#             "ids_offset": IPCH5Resource(p, "ids_offset", server),
#             "token_type_ids": IPCH5Resource(p, "token_type_ids", server),
#             "image_type_ids": IPCH5Resource(p, "image_type_ids", server),
#             "image_offsets": IPCH5Resource(p, "image_offsets", server),
#         }
#         fps.append(tmp)
#     return fps


# class AudioDatasetHolder(DatasetHolder):
#     """_summary_

#     Args:
#         DatasetHolder (_type_): _description_
#     """

#     def __init__(self, paths, server_idx, server_num):
#         """_summary_

#         Args:
#             paths (_type_): _description_
#             server_idx (_type_): _description_
#             server_num (_type_): _description_

#         Raises:
#             OSError: _description_
#         """
#         self.fps = {}
#         path_num = len(paths)
#         start_t = time()
#         for idx, path in enumerate(paths):
#             assert path not in self.fps, path

#             ds = h5py.File(path, mode="r")
#             shape = ds["ds16"].shape
#             offset_shape = ds["offset"].shape
#             if shape[0] <= 0 or shape[0] >= 1000000000000:  # 1000000000000 for max tokens of h5
#                 raise OSError

#             self.fps[path] = {
#                 "shape": shape,
#                 "ids": ds["ds16"],
#                 "ids_offset": jiefix(ds["ds16_offset"]),
#                 "lossmask": ds["ds16_lossmask"],
#                 "token_type_ids": ds["ds16_tokenwise_type_id"],
#                 "audio_ids": ds["ds16_audio_ids"],
#                 "audio_ids_offset": jiefix(ds["ds16_audio_ids_offset"]),
#                 "audio_ids_lossmask": ds["ds16_audio_ids_lossmask"],
#                 "meta": ds["meta"],
#                 "example_offset": jiefix(ds["example_offset"]),
#                 "meta_offset": jiefix(ds["offset"]),
#             }

#             end_t = time()
#             logger.info(
#                 f"Done loading {path}, shape: {shape}, in server-{server_idx}/{server_num}, "
#                 f"accumulated time = {end_t - start_t}, progress: {idx}/{path_num}"
#             )
#         end_t = time()
#         logger.info(
#             f"Server-{server_idx}/{server_num} load ends with path number {path_num}, "
#             f"accumulated time = {end_t - start_t}"
#         )


# class AudioDatasetHolderIniter(DatasetHolderIniter):
#     """_summary_

#     Args:
#         DatasetHolderIniter (_type_): _description_
#     """

#     def __call__(self, server_idx, server_num):
#         """_summary_

#         Args:
#             server_idx (_type_): _description_
#             server_num (_type_): _description_

#         Returns:
#             _type_: _description_
#         """
#         return AudioDatasetHolder(self.paths, server_idx, server_num)


# def create_ipc_audio_h5_resources(paths, num_server):
#     """
#     构造主进程里感知的IPCH5Resource, 并构造对应的IPCServer
#     """
#     n = len(paths)
#     if n <= 0:
#         return []

#     num_server = min(n, num_server)

#     router_keys = [[] for _ in range(num_server)]
#     for i, p in enumerate(paths):
#         router_keys[i % num_server].append(p)

#     init_funcs = [AudioDatasetHolderIniter(rk) for rk in router_keys]
#     server = IPCServer(router_keys, init_funcs)
#     atexit.register(lambda: server.close())
#     fps = []
#     for p in paths:
#         tmp = {
#             "shape": IPCH5MetaResource(p, server),
#             "ids": IPCH5Resource(p, "ids", server),
#             "ids_offset": IPCH5Resource(p, "ids_offset", server),
#             "lossmask": IPCH5Resource(p, "lossmask", server),
#             "token_type_ids": IPCH5Resource(p, "token_type_ids", server),
#             "audio_ids": IPCH5Resource(p, "audio_ids", server),
#             "audio_ids_offset": IPCH5Resource(p, "audio_ids_offset", server),
#             "audio_ids_lossmask": IPCH5Resource(p, "audio_ids_lossmask", server),
#             "meta": IPCH5Resource(p, "meta", server),
#             "example_offset": IPCH5Resource(p, "example_offset", server),
#             "meta_offset": IPCH5Resource(p, "meta_offset", server),
#         }
#         fps.append(tmp)
#     return fps


# def equal_shard(datasets, rank, world_size):
#     """
#     如果有权重，根据权重概率累计概率相等的原则切分 train part.
#     没有权重直接均分parts.
#     args:
#         datasets: List[ExampleSetSingleDataSource]
#         rank: int
#         world_size: int
#     """
#     assert len(datasets) >= world_size, f"#filelist={len(datasets)} < world_size{world_size}"
#     if world_size == 1:
#         return datasets
#     if datasets[0].weights is None:
#         ran = np.array_split(np.arange(len(datasets)), world_size)[rank]
#         s, e = ran[0], ran[-1]
#         shard = datasets[s : e + 1]
#         return shard
#     buckets = [[] for _ in range(world_size)]

#     bucketsize = np.zeros(len(buckets), dtype="float64")
#     total_w = sum([d.weights for d in datasets])
#     for d in datasets:
#         d.weights = d.weights / total_w

#     datasets = sorted(datasets, key=lambda d: d.weights, reverse=True)  # 先分大part，或许有利于均匀分发？
#     for d in datasets:
#         this_bucket = np.argmin(bucketsize)
#         buckets[this_bucket].append(d)
#         bucketsize[this_bucket] += d.weights

#     logger.info(f"sharding dataset according to prob, group vs probs={[sum([rr.weights for rr in r])for r in buckets]}")
#     bucketsize = bucketsize[rank]
#     diff = bucketsize - (1 / world_size)
#     logger.info(f"unable to perfect shard. prob sum of this bucket:{bucketsize}, diff to perfect portion:{diff}")
#     assert len(buckets) == world_size, f"#ret={len(buckets)} prob not normalized:{[d.weights for d in datasets]}"
#     return buckets[rank]


# def jiefix(offset):
#     """temp fix for int32 overflow"""
#     new_offset = np.zeros_like(offset, dtype="uint64")
#     mask = offset[1:] < offset[:-1]
#     new_offset[1:][mask] = 2**32
#     new_offset = np.cumsum(new_offset) + offset.astype("uint64")
#     return new_offset


# class LmExampleSetSingleDataSource(_LmExampleSetSingleDataSource):
#     """_summary_

#     Args:
#         _LmExampleSetSingleDataSource (_type_): _description_
#     """

#     data_type = DATATYPE_2_ID["lm"]


# class MmExampleSetSingleDataSource:
#     """Use to pick data from h5"""

#     data_type = DATATYPE_2_ID["mm"]

#     def __init__(
#         self,
#         path,
#         seqlen,
#         weights=None,
#         seed: int = 42,
#         shuffle: bool = False,
#         num_consecutive: int = 1,
#         combine_batch: int = 1,
#         name=None,
#     ):
#         if weights is not None:
#             assert isinstance(weights, tuple) and len(weights) == 3, weights
#             self.weights, self.src, self.part = weights
#         else:
#             self.weights, self.src, self.part = None, 0, 0

#         self.path = [os.path.expanduser(p) for p in path]
#         self.name = name
#         self.seqlen = seqlen
#         self.seed = seed
#         self.shuffle = shuffle

#         self.epoch = 0
#         self.fps = []
#         self._load = False
#         self._data_status = 0
#         self.num_consecutive = num_consecutive
#         self.combine_batch = combine_batch

#     @property
#     def data_status(self):
#         """_summary_

#         Returns:
#             _type_: _description_
#         """
#         return self._data_status

#     @data_status.setter
#     def data_status(self, value):
#         """_summary_

#         Args:
#             value (_type_): _description_
#         """
#         logger.info(f"part-{self.part}-load_data_status: {value}")
#         self._data_status = value

#     def set_loaded(self, fps):
#         """
#         Set loaded fps
#         """
#         self._load = True
#         self.int16_ds = True
#         self.fps = fps

#     def load(self):
#         """_summary_

#         Raises:
#             OSError: _description_
#         """
#         self._load = True
#         self.int16_ds = True
#         logger.info("using int16 ds")

#         for path in self.path:
#             logger.info(f"loading {path}, weight={self.weights}")
#             ds = h5py.File(path, mode="r")
#             shape = ds["meta"].shape
#             offset_shape = ds["offset"].shape
#             # hot fix for offset overflow int32

#             if shape[0] <= 0 or shape[0] >= 1000000000000:  # 1000000000000 for max tokens of h5
#                 raise OSError

#             self.fps.append(
#                 {
#                     "meta": ds["meta"],
#                     "shape": shape,
#                     "example_offset": jiefix(ds["example_offset"]),
#                     "meta_offset": jiefix(ds["offset"]),
#                     "example_offset_shape": offset_shape,
#                     "ids": ds["ds16"],
#                     "ids_offset": jiefix(ds["ds16_offset"]),
#                     "lossmask": ds["ds16_lossmask"],
#                     "token_type_ids": ds["ds16_tokenwise_type_id"],
#                     "image_type_ids": ds["ds16_imagewise_type_id"],
#                     "image_offsets": jiefix(ds["ds16_imagewise_type_id_offset"]),
#                     "path": path
#                 }
#             )

#     def __getitem__(self, idx):
#         """_summary_

#         Args:
#             idx (_type_): _description_

#         Returns:
#             _type_: _description_
#         """
#         assert len(idx) == 2, f"idx format must be (`epoch, span_id`), but got {idx} instead"
#         epoch, eidx = idx
#         assert eidx != -1, "eidx is -1"
#         assert self._load
#         fp = self.fps[epoch % len(self.fps)]
#         # span id 解开
#         offset = fp["ids_offset"]
#         if eidx == 0:
#             start = 0
#             e_start = 0
#             im_start = 0
#         else:
#             start = offset[eidx - 1]
#             e_start = fp["example_offset"][eidx - 1]
#             im_start = fp["image_offsets"][eidx - 1]
#         end = offset[eidx]
#         e_end = fp["example_offset"][eidx]
#         im_end = fp["image_offsets"][eidx]

#         meta_lst = []
#         for i in range(e_start, e_end):
#             if i == 0:
#                 meta_s = 0
#             else:
#                 meta_s = fp["meta_offset"][i - 1]
#             meta_e = fp["meta_offset"][i]
#             meta_lst.append(fp["meta"][meta_s:meta_e])

#         ret = VisionExample(
#             meta=meta_lst,
#             ids=fp["ids"][start:end],
#             sids=None,
#             task="mm",
#             src=self.part,
#             part=self.part,
#             lossmask=fp["lossmask"][start:end],
#             info=idx,
#             name=self.name,
#             data_type=self.data_type,
#             token_type_ids=fp["token_type_ids"][start:end],
#             image_type_ids=fp["image_type_ids"][im_start:im_end],
#         )
#         return ret

#     def __len__(self):
#         """_summary_

#         Returns:
#             _type_: _description_
#         """
#         assert self._load
#         fp = self.fps[self.epoch % len(self.fps)]
#         return len(fp["ids_offset"])

#     def __iter__(self):
#         """_summary_

#         Yields:
#             _type_: _description_
#         """
#         for i in range(len(self)):
#             yield self[(0, i)]

#     @property
#     def example_id(self):
#         """_summary_

#         Returns:
#             _type_: _description_
#         """
#         example_id = range(0, len(self), self.num_consecutive)
#         tmp = []
#         for start in example_id:
#             if start + self.num_consecutive > len(self):
#                 if tmp:
#                     # 如果tmp不是空的，则随机选择一个补到最后
#                     rng = random.Random(self.epoch + self.seed + self.part)
#                     cur = list(range(start, len(self))) + rng.choice(tmp)[: start + self.num_consecutive - len(self)]
#                 else:
#                     # 如果tmp是空的，则说明这个part的样本数少于num consecutive，直接重复到满为止
#                     assert start == 0
#                     cur = list(range(start, len(self)))
#                     while len(cur) < self.num_consecutive:
#                         cur += cur[: start + self.num_consecutive - len(self)]
#                 tmp.append(cur)
#             else:
#                 end = start + self.num_consecutive
#                 tmp.append(list(range(start, end)))
#         example_id = tmp
#         if self.shuffle:
#             rng = random.Random(self.epoch + self.seed + self.part)
#             rng.shuffle(example_id)
#         return np.array(example_id)

#     @property
#     def num_examples(self):
#         """_summary_

#         Returns:
#             _type_: _description_
#         """
#         assert self.epoch == 0
#         return len(list(range(0, len(self), self.num_consecutive)))

#     def sampler(self):
#         """_summary_

#         Yields:
#             _type_: _description_
#         """
#         # 在DistributedBatchSampler初始化 构造indices 时被调用。（主进程中）
#         assert paddle.io.get_worker_info() is None
#         self.epoch = 0
#         while 1:
#             if self._data_status >= len(self):
#                 self._data_status -= len(self)
#             else:
#                 logger.debug(f"...gen_index_from-[{self.part}]-[{self.epoch}]-offset-[{self.data_status}/{len(self)}]")
#                 for eid_list in self.example_id:
#                     _length = len(eid_list)
#                     if self._data_status > 0:
#                         if self._data_status >= _length:
#                             self._data_status -= _length
#                             continue
#                         else:
#                             eid_list = eid_list[self._data_status :]
#                             self._data_status = 0
#                     yield self.epoch, list(eid_list)
#             self.epoch += 1


# class AudioExampleSetSingleDataSource:
#     """Use to pick data from h5"""

#     data_type = DATATYPE_2_ID["audio"]

#     def __init__(
#         self,
#         path,
#         seqlen,
#         weights=None,
#         seed: int = 42,
#         shuffle: bool = False,
#         num_consecutive: int = 1,
#         combine_batch: int = 1,
#         name=None,
#     ):
#         if weights is not None:
#             assert isinstance(weights, tuple) and len(weights) == 3, weights
#             self.weights, self.src, self.part = weights
#         else:
#             self.weights, self.src, self.part = None, 0, 0

#         self.path = [os.path.expanduser(p) for p in path]
#         self.name = name
#         self.seqlen = seqlen
#         self.seed = seed
#         self.shuffle = shuffle

#         self.epoch = 0
#         self.fps = []
#         self._load = False
#         self._data_status = 0
#         self.num_consecutive = num_consecutive
#         self.combine_batch = combine_batch

#     @property
#     def data_status(self):
#         """_summary_

#         Returns:
#             _type_: _description_
#         """
#         return self._data_status

#     @data_status.setter
#     def data_status(self, value):
#         """_summary_

#         Args:
#             value (_type_): _description_
#         """
#         logger.info(f"part-{self.part}-load_data_status: {value}")
#         self._data_status = value

#     def set_loaded(self, fps):
#         """
#         Set loaded fps
#         """
#         self._load = True
#         self.int16_ds = True
#         self.fps = fps

#     def load(self):
#         """_summary_

#         Raises:
#             OSError: _description_
#         """
#         self._load = True
#         self.int16_ds = True
#         logger.info("using int16 ds")

#         for path in self.path:
#             logger.info(f"loading {path}, weight={self.weights}")
#             ds = h5py.File(path, mode="r")
#             shape = ds["ds16"].shape
#             offset_shape = ds["offset"].shape
#             if shape[0] <= 0 or shape[0] >= 1000000000000:  # 1000000000000 for max tokens of h5
#                 raise OSError

#             self.fps.append(
#                 {
#                     "shape": shape,
#                     "ids": ds["ds16"],
#                     "ids_offset": jiefix(ds["ds16_offset"]),
#                     "lossmask": ds["ds16_lossmask"],
#                     "token_type_ids": ds["ds16_tokenwise_type_id"],
#                     "audio_ids": ds["ds16_audio_ids"],
#                     "audio_ids_offset": jiefix(ds["ds16_audio_ids_offset"]),
#                     "audio_ids_lossmask": ds["ds16_audio_ids_lossmask"],
#                     "meta": ds["meta"],
#                     "example_offset": jiefix(ds["example_offset"]),
#                     "meta_offset": jiefix(ds["offset"]),
#                 }
#             )

#     def __getitem__(self, idx):
#         """_summary_

#         Args:
#             idx (_type_): _description_

#         Returns:
#             _type_: _description_
#         """
#         assert len(idx) == 2, f"idx format must be (`epoch, span_id`), but got {idx} instead"
#         epoch, eidx = idx
#         if eidx == -1:
#             return AudioExample(
#                 ids=[],
#                 sids=None,
#                 task="audio",
#                 src=self.part,
#                 part=self.part,
#                 name=self.name,
#                 lossmask=None,
#                 info=None,
#                 token_type_ids=None,
#                 data_type=self.data_type,
#                 audio_ids=None,
#                 audio_ids_lossmask=None,
#                 metas=None,
#             )
#         assert self._load
#         fp = self.fps[epoch % len(self.fps)]

#         ids_offset = fp["ids_offset"]
#         audio_ids_offset = fp["audio_ids_offset"]
#         example_offset = fp["example_offset"]
#         meta_offset = fp["meta_offset"]
#         if eidx == 0:
#             start = 0
#             audio_start = 0
#             example_offset_start = 0
#         else:
#             start = ids_offset[eidx - 1]
#             audio_start = audio_ids_offset[eidx - 1]
#             example_offset_start = example_offset[eidx - 1]
#         end = ids_offset[eidx]
#         audio_end = audio_ids_offset[eidx]
#         example_offset_end = example_offset[eidx]

#         metas = []
#         for i in range(example_offset_start, example_offset_end):
#             if i == 0:
#                 meta_s = 0
#             else:
#                 meta_s = meta_offset[i - 1]
#             meta_e = meta_offset[i]

#             meta = fp["meta"][meta_s:meta_e]
#             meta = json.loads(meta.tobytes().decode())
#             metas += meta

#         ret = AudioExample(
#             ids=fp["ids"][start:end],
#             sids=None,
#             task="audio",
#             src=self.part,
#             part=self.part,
#             lossmask=fp["lossmask"][start:end],
#             info=idx,
#             name=self.name,
#             data_type=self.data_type,
#             token_type_ids=fp["token_type_ids"][start:end],
#             audio_ids=fp["audio_ids"][audio_start:audio_end],
#             audio_ids_lossmask=fp["audio_ids_lossmask"][audio_start:audio_end],
#             metas=metas,
#         )
#         return ret

#     def __len__(self):
#         """_summary_

#         Returns:
#             _type_: _description_
#         """
#         assert self._load
#         fp = self.fps[self.epoch % len(self.fps)]
#         return len(fp["ids_offset"])

#     def __iter__(self):
#         """_summary_

#         Yields:
#             _type_: _description_
#         """
#         for i in range(len(self)):
#             yield self[(0, i)]

#     @property
#     def example_id(self):
#         """_summary_

#         Returns:
#             _type_: _description_
#         """
#         example_id = range(0, len(self), self.num_consecutive)
#         tmp = []
#         for start in example_id:
#             if start + self.num_consecutive > len(self):
#                 if tmp:
#                     # 如果tmp不是空的，则随机选择一个补到最后
#                     rng = random.Random(self.epoch + self.seed + self.part)
#                     cur = list(range(start, len(self))) + rng.choice(tmp)[: start + self.num_consecutive - len(self)]
#                 else:
#                     # 如果tmp是空的，则说明这个part的样本数少于num consecutive，直接重复到满为止
#                     assert start == 0
#                     cur = list(range(start, len(self)))
#                     while len(cur) < self.num_consecutive:
#                         cur += cur[: start + self.num_consecutive - len(self)]
#                 tmp.append(cur)
#             else:
#                 end = start + self.num_consecutive
#                 tmp.append(list(range(start, end)))
#         example_id = tmp
#         if self.shuffle:
#             rng = random.Random(self.epoch + self.seed + self.part)
#             rng.shuffle(example_id)
#         return np.array(example_id)

#     @property
#     def num_examples(self):
#         """_summary_

#         Returns:
#             _type_: _description_
#         """
#         assert self.epoch == 0
#         return len(list(range(0, len(self), self.num_consecutive)))

#     def sampler(self):
#         """_summary_

#         Yields:
#             _type_: _description_
#         """
#         # 在DistributedBatchSampler初始化 构造indices 时被调用。（主进程中）
#         assert paddle.io.get_worker_info() is None
#         self.epoch = 0
#         while 1:
#             if self._data_status >= len(self):
#                 self._data_status -= len(self)
#             else:
#                 logger.debug(f"...gen_index_from-[{self.part}]-[{self.epoch}]-offset-[{self.data_status}/{len(self)}]")
#                 for eid_list in self.example_id:
#                     _length = len(eid_list)
#                     if self._data_status > 0:
#                         if self._data_status >= _length:
#                             self._data_status -= _length
#                             continue
#                         else:
#                             eid_list = eid_list[self._data_status :]
#                             self._data_status = 0
#                     yield self.epoch, list(eid_list)
#             self.epoch += 1


# class ExampleSet:
#     """use to manage all h5 data"""

#     def __init__(self, exs, fn, load_process_num=0):
#         """_summary_

#         Args:
#             exs (_type_): _description_
#             fn (function): _description_
#         """
#         self.exs = exs
#         self.fn = fn
#         self._load = False
#         # 分shard之前 统计最大的part
#         self.global_max_part_id = max([ex.part for ex in exs])
#         self.partid2ex = {ex.part: ex for ex in exs}
#         self.load_process_num = load_process_num

#     def append(self, new_exs):
#         """_summary_

#         Args:
#             new_exs (_type_): _description_
#         """
#         logger.info(f"updating exs, #new example: {len(new_exs)}")
#         self.exs.append(new_exs)
#         lens = [len(e) for e in self.exs]
#         len_sum = sum(lens)
#         logger.info(f"multi task data portion")
#         logger.info("\n".join([f"{e.part}={l/len_sum}" for l, e in zip(lens, self.exs)]))

#     def load(self, use_shard, dp_rank, dp_size):
#         """_summary_

#         Args:
#             use_shard (_type_): _description_
#             dp_rank (_type_): _description_
#             dp_size (_type_): _description_
#             data_type (_type_, optional): _description_. Defaults to None.

#         Returns:
#             _type_: _description_
#         """
#         self._load = True
#         logger.info(f"loading h5... use_shard={use_shard}, {self._load} {id(self)}")
#         logger.info(f"loading h5 in dp_env:{dp_rank}/{dp_size}")
#         if use_shard:
#             logger.info(f"#shard train file, before load")

#             def keyfn(e):
#                 l = e.file_list.strip("/").split("/")
#                 return l[0]  # l[-3] if len(l) >= 3 else 1

#             lm_exs = [ex for ex in self.exs if ex.data_type == DATATYPE_2_ID["lm"]]
#             lm_path_per_dp = equal_shard(lm_exs, dp_rank, dp_size) if len(lm_exs) > 0 else []
#             logger.debug(f"using lm data shard, # files before shard={len(lm_exs)}, after shard={len(lm_path_per_dp)}")
#             mm_exs = [ex for ex in self.exs if ex.data_type == DATATYPE_2_ID["mm"]]
#             mm_path_per_dp = equal_shard(mm_exs, dp_rank, dp_size) if len(mm_exs) > 0 else []
#             logger.debug(f"using mm data shard, # files before shard={len(mm_exs)}, after shard={len(mm_path_per_dp)}")
#             audio_exs = [ex for ex in self.exs if ex.data_type == DATATYPE_2_ID["audio"]]
#             audio_path_per_dp = equal_shard(audio_exs, dp_rank, dp_size) if len(audio_exs) > 0 else []
#             logger.debug(
#                 f"using audio data shard, # files before shard={len(audio_exs)}, after shard={len(audio_path_per_dp)}"
#             )
#             path_per_dp = lm_path_per_dp + mm_path_per_dp + audio_path_per_dp
#             logger.debug(f"using source shard, # files before shard={len(self.exs)}, after shard={len(path_per_dp)}")
#             self.exs = path_per_dp
#         if self.load_process_num > 0:
#             for idx, exs in enumerate([lm_path_per_dp, mm_path_per_dp, audio_path_per_dp]):
#                 paths = []
#                 ranges = []
#                 start_idx = 0
#                 for i, ex in enumerate(exs):
#                     # assert isinstance(ex, ExampleSetSingleDataSource), type(ex)
#                     cur_len = len(ex.path)
#                     paths.extend(ex.path)
#                     ranges.append((ex, start_idx, start_idx + cur_len))
#                     start_idx += cur_len
#                 if idx == 0:
#                     fps = create_ipc_lm_h5_resources(paths, self.load_process_num)
#                 elif idx == 1:
#                     fps = create_ipc_mm_h5_resources(paths, self.load_process_num)
#                 else:
#                     fps = create_ipc_audio_h5_resources(paths, self.load_process_num)
#                 for ex, start, end in ranges:
#                     ex.set_loaded(fps[start:end])
#         else:
#             loaded_exs, err_cnt = [], 0
#             for ex in self.exs:
#                 try:
#                     ex.load()
#                 except OSError as e:
#                     logger.warning(f"loading {ex.path} error:{e}, skip...")
#                     err_cnt += 1
#                     continue
#                 loaded_exs.append(ex)
#             assert loaded_exs, f"data_dir {[e.path for e in self.exs]} empty, #err:{err_cnt}"
#             if err_cnt > 0:
#                 raise ValueError(f"some data load failed, #parts={len(self.exs)}, #err={err_cnt}")
#             self.exs = loaded_exs
#             logger.info(f"done loading h5 #parts={len(self.exs)}, #err={err_cnt}")

#     def __getitem__(self, idx):
#         # index 为三维坐标 (partid, part_epoch, part_data_idx)
#         if isinstance(idx, int):
#             # dev data
#             s = 0
#             for ex in self.exs:
#                 if s + len(ex) < idx:
#                     s += len(ex)
#                 else:
#                     ret = ex[(0, idx - s)]
#                     break
#             eidx = idx
#         else:
#             assert len(idx) == 3, f"idx format must be (`part`, `part_epoch`, `data_id`), but got {idx} instead"
#             part, epoch, eidx = idx
#             ret = self.partid2ex[part][(epoch, eidx)]
#         ret = self.fn(ret)
#         if ret["data_not_valid"]:
#             logger.warning(f"error data: {idx}")
#         ret.update(data_id=eidx)
#         return ret

#     def __len__(self):
#         """_summary_

#         Returns:
#             _type_: _description_
#         """
#         assert self._load
#         return sum(map(len, self.exs))

#     def __iter__(self):
#         # print(f"real len: {len(self)}")
#         for i in range(len(self)):
#             yield self[i]


# # class PretrainMultimodalTask:
# #     """_summary_"""

# #     def __init__(
# #         self,
# #         args,
# #         data_dir,
# #         tokenizer,
# #         image_preprocess,
# #         im_patch_id=None,
# #         image_token_len=64,
# #         bos_retry_max_time=0,
# #         bos_retry_interval=1,
# #         image_dtype="uint8",
# #         **kwargs,
# #     ):
# #         self.img_data_processor = ImageModificationProcessor(args)
# #         self.data_dir = data_dir
# #         self.tokenizer = tokenizer
# #         self.vocab = self.tokenizer.get_vocab()
# #         self.im_patch_id = im_patch_id or len(self.vocab)  # TODO 外部传入更合适
# #         self.image_token_len = image_token_len
# #         self.bos_retry_max_time = bos_retry_max_time
# #         self.bos_retry_interval = bos_retry_interval
# #         self.image_preprocess = image_preprocess
# #         self.tokenizer.ignored_index = -100
# #         self.image_dtype = image_dtype
# #         self.audio_placeholder_id = self.tokenizer.get_vocab()[
# #             MMSpecialTokensConfig.get_special_tokens_info()["audio_placeholder"]
# #         ]

# #     def train_data(
# #         self,
# #         max_seq_len=512,
# #         stride=None,
# #         overlap_len=0,
# #         rng=None,
# #         weights=None,
# #         evaluate=False,
# #         seed=0,
# #         num_consecutive=1,
# #         shuffle=True,
# #         combine_batch=1,
# #         load_process_num=0,
# #     ):
# #         """_summary_

# #         Args:
# #             shuffle (bool, optional): _description_. Defaults to True.
# #             num_consecutive (int, optional): _description_. Defaults to 1.

# #         Returns:
# #             _type_: _description_
# #         """
# #         examples = []
# #         path = [i for i in self.data_dir if not i[0][0].endswith("meta")]
# #         if not weights:
# #             weights = [(None, None, i) for i in range(len(path))]
# #         for (p, data_type), w in zip(path, weights):
# #             if data_type == "mm":
# #                 example = MmExampleSetSingleDataSource(
# #                     p,
# #                     max_seq_len,
# #                     weights=w,
# #                     seed=seed,
# #                     shuffle=shuffle,
# #                     num_consecutive=num_consecutive,
# #                     combine_batch=combine_batch,
# #                 )
# #             elif data_type == "audio":
# #                 example = AudioExampleSetSingleDataSource(
# #                     p,
# #                     max_seq_len,
# #                     weights=w,
# #                     seed=seed,
# #                     shuffle=shuffle,
# #                     num_consecutive=num_consecutive,
# #                     combine_batch=combine_batch,
# #                 )
# #             else:
# #                 example = LmExampleSetSingleDataSource(
# #                     p,
# #                     max_seq_len,
# #                     stride=stride,
# #                     weights=w,
# #                     seed=seed,
# #                     num_consecutive=num_consecutive,
# #                     shuffle=shuffle,
# #                     combine_batch=combine_batch,
# #                 )
# #             examples.append(example)
# #         example_set = ExampleSet(examples, self.example_to_feature, load_process_num)
# #         return example_set

# #     def example_to_feature(self, example):
# #         """Convert an Example proto into a TensorFlow `Feature` protocol buffer."""
# #         # assert example.labels is None
# #         return self.img_data_processor.process(example, download_fn=bos_path_to_pil_image)
