# !/usr/bin/env python3
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import atexit
import os
import sys
import math
import re
from glob import glob
from pathlib import Path
import random
import argparse
import logging
import json
from copy import deepcopy
import itertools
from functools import partial, reduce
import numpy as np
from bisect import bisect
from collections import OrderedDict, defaultdict, namedtuple, Counter
from scipy.special import softmax
from typing import List
import datetime
import copy
from itertools import groupby

import paddle
import paddle.nn.functional as F
from paddle.io import IterableDataset
import h5py
from time import time

import enum
import traceback
import threading
from dataclasses import dataclass
from multiprocessing import Process, Queue, Lock


def parse_filelist(filelist):
    """parse filelist

    Args:
        filelist (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if isinstance(filelist, str):
        filelist = [filelist]
    part_id_offset = 0
    h5, partids = [], []
    for f in filelist:
        lines = [i.strip().split("\t") for i in open(f).readlines()]
        if len(lines[0]) == 1:
            h5.extend([i[0] for i in lines])
            partids.extend([i + part_id_offset for i in range(len(lines))])
        elif len(lines[0]) == 2:
            _ids, _flst = zip(*lines)
            h5.extend(_flst)
            partids.extend([int(i) + part_id_offset for i in _ids])
        else:
            raise ValueError("part format error")
        part_id_offset = max(partids) + 1
    assert len(h5) == len(set(h5)), "duplicated filelist"
    return partids, h5


def parse_weights(weights):
    """parse weights

    Args:
        weights (_type_): _description_

    Returns:
        _type_: _description_
    """
    patterns = []
    if isinstance(weights, str):
        weights = [weights]
    for w in weights:
        for i in open(w):
            cols = i.strip().split()
            assert len(cols) >= 3, f"配比文件至少要4列，格式为：pattern weight num_parts - {cols}"
            pattern, w, num_parts = cols[:3]
            if len(cols) >= 4 and cols[3] in ["lm", "mm", "audio"]:
                # 允许第4列手动指定 data_type ，后续配比文件建议follow这个格式。
                data_type = cols[3]
            else:
                # 向前兼容
                data_type = "mm" if "multimodal" in i else "lm"

            num_parts = int(num_parts)
            pattern = re.compile(pattern)
            patterns.append((pattern, float(w) / num_parts, data_type))
    return patterns


def parse_data_weight(weights, filelist):
    """
    解析data_weights文件:
    Inputs:
     `weights`  : file
     `filelist` : List[str]
    Returns:
        Dict[str, Tuple[float, int]]
    """
    partids, filelist = parse_filelist(filelist)
    patterns = parse_weights(weights)
    partid2files, weight_filelist = {}, {}
    for part_id, f in zip(partids, filelist):
        if part_id not in partid2files:
            partid2files[part_id] = [f]
        else:
            partid2files[part_id].append(f)

        for ipattern, (pattern, w, data_type) in enumerate(patterns):
            if pattern.search(f):
                # weight_filelist[f] = (float(w), ipattern, part_id)
                weight_filelist[part_id] = (float(w), ipattern, data_type)
                break
        else:
            log.warning(f"{f} does not match any pattern")
    
    train_filelist, weights = [], []
    for part_id, (v, source_id, data_type) in weight_filelist.items():
        train_filelist.append((partid2files[part_id], data_type))
        weights.append((v, source_id, part_id))
    return train_filelist, weights


log = logging.getLogger(__name__)

class ServerStatus(enum.Enum):
    """
    ServerStatus
    """

    WAIT_RUNNING = 0
    RUNNING = 1
    EXIT_WITH_FAILURE = 2
    EXIT_WITH_CLOSE = 3


class ResponseTag(enum.Enum):
    """
    ResponseTag
    """

    SUCCESS = 0
    FAILURE = 1


class ExitFlag:
    """
    ExitFlag
    """

    pass


@dataclass
class MethodRequest:
    """
    MethodRequest
    """

    router_key: object
    name: str
    args: list
    kwargs: dict


@dataclass
class AttrRequest:
    """
    AttrRequest
    """

    router_key: object
    name: str


@dataclass
class Response:
    """
    Response
    """

    tag: ResponseTag
    value: object
    exception: Exception


def server_loop(init_func, server_idx, server_num, init_queue, send_queue, recv_queue):
    """
    server_loop
    """
    try:
        init_obj = init_func(server_idx, server_num)
        init_queue.put(Response(tag=ResponseTag.SUCCESS, exception=None, value=ServerStatus.RUNNING))
    except Exception as e:
        logger.exception(e)
        init_queue.put(Response(tag=ResponseTag.FAILURE, exception=e, value=ServerStatus.EXIT_WITH_FAILURE))
        return

    while True:
        request = send_queue.get()
        if isinstance(request, ExitFlag):
            break

        try:
            value = getattr(init_obj, request.name)
            if isinstance(request, MethodRequest):
                args = request.args or tuple()
                kwargs = request.kwargs or dict()
                value = value(*args, **kwargs)
            response = Response(tag=ResponseTag.SUCCESS, exception=None, value=value)
        except Exception as e:
            response = Response(tag=ResponseTag.FAILURE, exception=e, value=None)
            print("Exception inside process", e)

        recv_queue.put(response)


class SubIPCServer:
    """
    SubIPCServer
    """

    def __init__(self, server_idx, server_num, init_func):
        """
        __init__
        """
        self.send_queue = Queue()
        self.recv_queue = Queue()
        self.init_queue = Queue()
        self.server_status = ServerStatus.WAIT_RUNNING
        self.server_idx = server_idx
        self.server_num = server_num
        self.process = Process(
            target=server_loop,
            args=(init_func, server_idx, server_num, self.init_queue, self.send_queue, self.recv_queue),
        )
        self.process.daemon = True
        self.process.start()
        self.lock = Lock()

    def wait_started(self):
        """
        wait_started
        """
        if self.server_status == ServerStatus.RUNNING:
            return
        elif self.server_status == ServerStatus.WAIT_RUNNING:
            init_response = self.init_queue.get()
            assert init_response.value in [ServerStatus.RUNNING, ServerStatus.EXIT_WITH_FAILURE], init_response.value
            self.server_status = init_response.value
            if init_response.value == ServerStatus.EXIT_WITH_FAILURE:
                self.server_status = ServerStatus.EXIT_WITH_FAILURE
                raise init_response.exception
        elif self.server_status == ServerStatus.EXIT_WITH_FAILURE:
            raise RuntimeError("IPCServer does not start successfully")
        elif self.server_status == ServerStatus.EXIT_WITH_CLOSE:
            raise RuntimeError("IPCServer has been closed")
        else:
            raise RuntimeError(f"Unknown server status {self.server_status}")

    def response(self, request):
        """
        response
        """
        with self.lock:
            self.wait_started()
            self.send_queue.put(request)
            ret = self.recv_queue.get()
        return ret

    def close(self):
        """
        close
        """
        with self.lock:
            if self.process is not None:
                self.wait_started()
                self.send_queue.put(ExitFlag())
                self.process.join()
                self.process = None
                self.server_status = ServerStatus.EXIT_WITH_CLOSE


class IPCServer:
    """
    IPCServer
    """

    def __init__(self, router_groups, init_funcs):
        """
        __init__
        """
        server_num = len(init_funcs)
        group_num = len(router_groups)
        assert server_num == group_num, f"{server_num} vs {group_num}"
        assert server_num > 0, f"server_num should be larger than 0, but got {server_num}"
        self.router_map = {}
        self.sub_servers = [None] * server_num
        for i, (group, init_func) in enumerate(zip(router_groups, init_funcs)):
            sub_server = SubIPCServer(i, server_num, init_func)
            for router_key in group:
                if router_key in self.router_map:
                    prev_idx = self.router_map[router_key].server_idx
                    assert prev_idx == i, f"{router_key}: {prev_idx} vs {i}"
                else:
                    self.router_map[router_key] = sub_server

    def _response(self, request):
        """
        _response
        """
        server = self.router_map[request.router_key]
        response = server.response(request)
        if response.exception is not None:
            raise response.exception
        else:
            return response.value

    def call(self, router_key, name, args=tuple(), kwargs=dict()):
        """
        IPC call method
        """
        request = MethodRequest(router_key=router_key, name=name, args=args, kwargs=kwargs)
        return self._response(request)

    def attr(self, router_key, name):
        """
        IPC get attribute
        """
        request = AttrRequest(router_key=router_key, name=name)
        return self._response(request)

    def close(self):
        """
        IPC close server
        """
        for server in self.sub_servers:
            if server is not None:
                server.close()


class IPCH5Resource:
    """
    主进程感知的HDF5 handle, 通过server.call与子进程交互
    """

    def __init__(self, path, name, server):
        """
        构造函数, path代表HDF5文件路径, name代表"fp", "lossmask", "off", "meta"这些字段
        """
        self.path = path
        self.name = name
        self.server = server
        self._length = None
        self._to_bool = None

    def __getitem__(self, key):
        """
        __getitem__
        """
        return self.server.call(self.path, "get", (self.path, self.name, key))

    def __len__(self):
        """
        获取长度(带cache)
        """
        if self._length is None:
            self._length = self.server.call(self.path, "len", (self.path, self.name))
        return self._length

    def __bool__(self):
        """
        转换为bool(带cache)
        """
        if self._to_bool is None:
            self._to_bool = self.server.call(self.path, "to_bool", (self.path, self.name))
        return self._to_bool


class IPCH5MetaResource:
    """
    主进程感知的HDF5 Meta handle
    """

    def __init__(self, path, server):
        """
        __init__
        """
        self.path = path
        self.server = server
        self._meta = None

    def _get_meta(self):
        """
        get_meta once
        """
        if self._meta is None:
            self._meta = self.server.call(self.path, "get_meta", (self.path,))

    def __getitem__(self, key):
        """
        __getitem__
        """
        self._get_meta()
        return self._meta[key]

    def __len__(self):
        """
        __len__
        """
        self._get_meta()
        return len(self._meta)


class DatasetHolder:
    """
    子进程感知的HDF5 handle
    """

    def __init__(self, paths, server_idx, server_num):
        """
        构造函数
        """
        self.fps = {}
        path_num = len(paths)
        start_t = time()
        for idx, path in enumerate(paths):
            assert path not in self.fps, path

            ds = h5py.File(path, mode="r")
            fp = ds["ds16"]
            assert "ds16_tokenwise_type_id" not in ds, f"this file maybe a multimodal H5, path={path}"
            if "ds16_lossmask" in ds:
                fp_lossmask = ds["ds16_lossmask"]
                assert len(ds["ds16_lossmask"]) == len(ds["ds16"]), (len(ds["ds16_lossmask"]), len(ds["ds16"]))
            else:
                fp_lossmask = None

            if "ds16_off" in ds:
                off = ds["ds16_off"]
            else:
                off = None

            if "log_prob" in ds:
                log_prob = ds["log_prob"]
            else:
                log_prob = None

            shape = fp.shape
            meta = {"shape": shape}
            if shape[0] <= 0 or shape[0] >= 1000000000000:
                raise OSError
            self.fps[path] = {"fp": fp, "lossmask": fp_lossmask, "meta": meta, "off": off, "log_prob": log_prob}
            end_t = time()
            log.info(
                f"Done loading {path}, shape: {shape}, in server-{server_idx}/{server_num}, "
                f"accumulated time = {end_t - start_t}, progress: {idx}/{path_num}"
            )
        end_t = time()
        log.info(
            f"Server-{server_idx}/{server_num} load ends with path number {path_num}, "
            f"accumulated time = {end_t - start_t}"
        )

    def get(self, path, name, key):
        """
        get
        """
        return self.fps[path][name][key]

    def len(self, path, name):
        """
        len
        """
        return len(self.fps[path][name])

    def to_bool(self, path, name):
        """
        to_bool
        """
        return True if self.fps[path][name] else False

    def get_meta(self, path):
        """
        get_meta
        """
        return self.fps[path]["meta"]


class DatasetHolderIniter:
    """
    用于在子进程里构造DatasetHolder
    """

    def __init__(self, paths):
        """
        __init__
        """
        self.paths = paths

    def __call__(self, server_idx, server_num):
        """
        在子进程里调用,构造DatasetHolder
        """
        return DatasetHolder(self.paths, server_idx, server_num)


def create_ipc_h5_resources(paths, num_server):
    """
    构造主进程里感知的IPCH5Resource, 并构造对应的IPCServer
    """
    n = len(paths)
    if n <= 0:
        return []

    num_server = min(n, num_server)

    router_keys = [[] for _ in range(num_server)]
    for i, p in enumerate(paths):
        router_keys[i % num_server].append(p)

    init_funcs = [DatasetHolderIniter(rk) for rk in router_keys]
    server = IPCServer(router_keys, init_funcs)
    atexit.register(lambda: server.close())
    fps = []
    for p in paths:
        tmp = {
            "fp": IPCH5Resource(p, "fp", server),
            "lossmask": IPCH5Resource(p, "lossmask", server),
            "meta": IPCH5MetaResource(p, server),
            "off": IPCH5Resource(p, "off", server),
            "log_prob": IPCH5Resource(p, "log_prob", server),
        }
        fps.append(tmp)
    return fps


def parse_filelist(filelist):
    """parse filelist

    Args:
        filelist (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if isinstance(filelist, str):
        filelist = [filelist]
    part_id_offset = 0
    h5, partids = [], []
    for f in filelist:
        lines = [i.strip().split("\t") for i in open(f).readlines()]
        if len(lines[0]) == 1:
            h5.extend([i[0] for i in lines])
            partids.extend([i + part_id_offset for i in range(len(lines))])
        elif len(lines[0]) == 2:
            _ids, _flst = zip(*lines)
            h5.extend(_flst)
            partids.extend([int(i) + part_id_offset for i in _ids])
        else:
            raise ValueError("part format error")
        part_id_offset = max(partids) + 1
    assert len(h5) == len(set(h5)), "duplicated filelist"
    return partids, h5


def parse_weights(weights):
    """parse weights

    Args:
        weights (_type_): _description_

    Returns:
        _type_: _description_
    """
    patterns = []
    if isinstance(weights, str):
        weights = [weights]
    for w in weights:
        for i in open(w):
            cols = i.strip().split()
            assert len(cols) >= 3, f"配比文件至少要4列，格式为：pattern weight num_parts - {cols}"
            pattern, w, num_parts = cols[:3]
            if len(cols) >= 4 and cols[3] in ["lm", "mm", "audio"]:
                # 允许第4列手动指定 data_type ，后续配比文件建议follow这个格式。
                data_type = cols[3]
            else:
                # 向前兼容
                data_type = "mm" if "multimodal" in i else "lm"

            num_parts = int(num_parts)
            pattern = re.compile(pattern)
            patterns.append((pattern, float(w) / num_parts, data_type))
    return patterns


def parse_data_weight(weights, filelist):
    """
    解析data_weights文件:
    Inputs:
     `weights`  : file
     `filelist` : List[str]
    Returns:
        Dict[str, Tuple[float, int]]
    """
    partids, filelist = parse_filelist(filelist)
    patterns = parse_weights(weights)
    partid2files, weight_filelist = {}, {}
    for part_id, f in zip(partids, filelist):
        if part_id not in partid2files:
            partid2files[part_id] = [f]
        else:
            partid2files[part_id].append(f)

        for ipattern, (pattern, w, data_type) in enumerate(patterns):
            if pattern.search(f):
                # weight_filelist[f] = (float(w), ipattern, part_id)
                weight_filelist[part_id] = (float(w), ipattern, data_type)
                break
        else:
            log.warning(f"{f} does not match any pattern")

    train_filelist, weights = [], []
    for part_id, (v, source_id, data_type) in weight_filelist.items():
        train_filelist.append((partid2files[part_id], data_type))
        weights.append((v, source_id, part_id))
    return train_filelist, weights


def equal_shard(datasets, rank, world_size):
    """
    如果有权重，根据权重概率累计概率相等的原则切分 train part.
    没有权重直接均分parts.
    args:
        datasets: List[ExampleSetSingleDataSource]
        rank: int
        world_size: int
    """
    assert len(datasets) >= world_size, f"#filelist={len(datasets)} < world_size{world_size}"
    if world_size == 1:
        return datasets
    if datasets[0].weights is None:
        ran = np.array_split(np.arange(len(datasets)), world_size)[rank]
        s, e = ran[0], ran[-1]
        shard = datasets[s : e + 1]
        return shard
    buckets = [[] for _ in range(world_size)]

    bucketsize = np.zeros(len(buckets), dtype="float64")
    total_w = sum([d.weights for d in datasets])
    for d in datasets:
        d.weights = d.weights / total_w
    datasets = sorted(datasets, key=lambda d: d.weights, reverse=True)  # 先分大part，或许有利于均匀分发？
    for d in datasets:
        this_bucket = np.argmin(bucketsize)
        buckets[this_bucket].append(d)
        bucketsize[this_bucket] += d.weights

    log.info(f"sharding dataset according to prob, group vs probs={[sum([rr.weights for rr in r])for r in buckets]}")
    bucketsize = bucketsize[rank]
    diff = bucketsize - (1 / world_size)
    log.info(f"unable to perfect shard. prob sum of this bucket:{bucketsize}, diff to perfect portion:{diff}")
    assert len(buckets) == world_size, f"#ret={len(buckets)} prob not normalized:{[d.weights for d in datasets]}"
    return buckets[rank]


Example = namedtuple("Example", ["ids", "sids", "task", "lossmask", "src", "log_prob"])


class ExampleSetSingleDataSource:
    """Use to pick data from h5"""

    def __init__(
        self,
        path: List[str],
        seqlen,
        stride=None,
        weights=None,
        shuffle: bool = False,
        num_consecutive: int = 1,
        seed: int = 42,
        combine_batch: int = 1,
    ):
        """
        Args:
            path: str.
            seqle: int, 序列长度.
            stride: int, 划窗offset.
            weights: Tuple[float, int], 二元组(`数据权重`,`数据源id`, `part_id`)
        """
        self.seqlen = seqlen
        if weights is not None:
            assert isinstance(weights, tuple) and len(weights) == 3, weights
            self.weights, self.src, self.part = weights
        else:
            self.weights, self.src, self.part = None, 0, 0
        if not stride:
            self.stride = seqlen
        else:
            self.stride = stride
        self.path = [os.path.expanduser(p) for p in path]
        self._load = False
        self.fps = []
        self._data_status = 0
        self.num_consecutive = num_consecutive
        self.seed = seed
        self.shuffle = shuffle
        self._combine_batch = combine_batch
        self.epoch = 0

    @property
    def combine_batch(self):
        return self._combine_batch

    @property
    def data_status(self):
        return self._data_status

    @data_status.setter
    def data_status(self, value):
        log.info(f"part-{self.part}-load_data_status: {value}")
        self._data_status = value

    def set_loaded(self, fps):
        """
        Set loaded fps
        """
        self._load = True
        self.int16_ds = True
        self.fps = fps

    def load(self):
        self._load = True
        self.int16_ds = True
        log.info("using int16 ds")

        for path in self.path:
            log.info(f"loading {path}, weights={self.weights}")
            ds = h5py.File(path, mode="r")
            assert "ds16_tokenwise_type_id" not in ds, f"this file maybe a multimodal H5, src={self.src}"

            fp = ds["ds16"]
            if "ds16_lossmask" in ds:
                fp_lossmask = ds["ds16_lossmask"]
                assert len(ds["ds16_lossmask"]) == len(ds["ds16"]), (len(ds["ds16_lossmask"]), len(ds["ds16"]))
            else:
                fp_lossmask = None
            # self.fp = self.fps[0]

            if "ds16_off" in ds:
                log.info(f"using ds w/ offset")
                off = ds["ds16_off"]
            else:
                off = None

            if "log_prob" in ds:
                log.info(f"using ds with log_prob")
                log_prob = ds["log_prob"]
            else:
                log_prob = None
            shape = fp.shape
            meta = {"shape": shape}
            if shape[0] <= 0 or shape[0] >= 1000000000000:  # 1000000000000 for max tokens of h5
                raise OSError
            self.fps.append({"fp": fp, "lossmask": fp_lossmask, "meta": meta, "off": off, "log_prob": log_prob})
            log.info(
                f"done loading {path}, shape:{shape}: int16:{self.int16_ds} "
                f"seqlen:{self.seqlen} stride:{self.stride}"
            )
        log.info(f"done loading part-{self.part}, file count: {len(self.fps)}")

    def __getitem__(self, idx):
        assert len(idx) == 2, f"idx format must be (`epoch, data_idx`), but got {idx} instead"
        epoch, idx = idx
        if idx == -1:
            return Example(ids=[], sids=None, task="lm", src=self.part, lossmask=None, log_prob=None)
        assert self._load
        fp = self.fps[epoch % len(self.fps)]
        off = fp["off"]
        if off:
            s = off[idx]
            e = off[idx + 1]
        else:
            s = max(idx * self.stride, 0)
            e = idx * self.stride + self.seqlen

        ids = fp["fp"][s:e].astype(np.int32)
        if fp["lossmask"]:
            lossmask = fp["lossmask"][s:e].astype(np.int32)
        else:
            lossmask = None
        if fp["log_prob"]:
            log_prob = fp["log_prob"][s:e].astype(np.float32)
        else:
            log_prob = None
        ret = Example(ids=ids, sids=None, task="lm", src=self.part, lossmask=lossmask, log_prob=log_prob)
        return ret

    def __len__(self):
        assert self._load
        fp = self.fps[self.epoch % len(self.fps)]
        if fp["off"]:
            return len(fp["off"])
        return int(np.ceil((fp["meta"]["shape"][0]) / self.stride))

    def __iter__(self):
        for i in range(len(self)):
            yield self[(0, i)]

    @property
    def example_id(self):
        example_id = range(0, len(self), self.num_consecutive)
        example_id = [(ii, min(ii + self.num_consecutive, len(self))) for ii in example_id]
        if self.shuffle:
            rng = random.Random(self.epoch + self.seed + self.part)
            rng.shuffle(example_id)
        return np.array(example_id)

    @property
    def num_examples(self):
        assert self.epoch == 0
        # return len(list(range(0, len(self), self.num_consecutive)))
        return (len(self) + self.num_consecutive - 1) // self.num_consecutive

    def sampler(self):
        # 在DistributedBatchSampler初始化 构造indices 时被调用。（主进程中）
        assert paddle.io.get_worker_info() is None

        self.epoch = 0
        while 1:
            if self._data_status >= len(self):
                self._data_status -= len(self)
            else:
                log.debug(f"...gen_index_from-[{self.part}]-[{self.epoch}]-offset-[{self.data_status}/{len(self)}]")
                for s, e in self.example_id:
                    _length = math.ceil((e - s) / self.combine_batch) * self.combine_batch
                    if self._data_status > 0:
                        if self._data_status >= _length:
                            self._data_status -= _length
                            continue
                        else:
                            s += self._data_status
                            self._data_status = 0
                    yield self.epoch, list(range(s, e))
            self.epoch += 1


class ExampleSet:
    """use to manage all h5 data"""

    def __init__(self, exs, fn, load_process_num=0):
        """
        __init__
        """
        self.exs = exs
        self.fn = fn
        self._load = False
        # 分shard之前 统计最大的part_id
        self.global_max_part_id = max([ex.part for ex in exs])
        self.partid2ex = {ex.part: ex for ex in exs}
        self.load_process_num = load_process_num

    def append(self, new_exs):
        log.info(f"updating exs, #new example: {len(new_exs)}")
        self.exs.append(new_exs)
        lens = [len(e) for e in self.exs]
        len_sum = sum(lens)
        log.info(f"multi task data portion")
        log.info("\n".join([f"{e.path}={l/len_sum}" for l, e in zip(lens, self.exs)]))

    def load(self, use_shard, dp_rank, dp_size):
        self._load = True
        log.info(f"loading h5... use_shard={use_shard}, {self._load} {id(self)}")

        log.info(f"loading h5 in dp_env:{dp_rank}/{dp_size}")
        if use_shard:
            log.info(f"#shard train file, before load")

            def keyfn(e):
                l = e.path.strip("/").split("/")
                return l[0]  # l[-3] if len(l) >= 3 else 1

            path_per_dp = equal_shard(self.exs, dp_rank, dp_size)
            log.debug(f"using source shard, # files before shard={len(self.exs)}, after shard={len(path_per_dp)}")
            self.exs = path_per_dp

        if self.load_process_num > 0:
            paths = []
            ranges = []
            start_idx = 0
            for i, ex in enumerate(self.exs):
                assert isinstance(ex, ExampleSetSingleDataSource), type(ex)
                cur_len = len(ex.path)
                paths.extend(ex.path)
                ranges.append((ex, start_idx, start_idx + cur_len))
                start_idx += cur_len

            fps = create_ipc_h5_resources(paths, self.load_process_num)
            for ex, start, end in ranges:
                ex.set_loaded(fps[start:end])
        else:
            loaded_exs, err_cnt = [], 0
            for ex in self.exs:
                try:
                    if isinstance(ex, ExampleSetSingleDataSource):
                        ex.load()
                except OSError as e:
                    log.warning(f"loading {ex.path} error:{e}, skip...")
                    err_cnt += 1
                    continue
                loaded_exs.append(ex)
            assert loaded_exs, f"data_dir {[e.path for e in self.exs]} empty, #err:{err_cnt}"
            self.exs = loaded_exs
            if err_cnt > 0:
                raise ValueError(f"some data load failed, #parts={len(self.exs)}, #err={err_cnt}")
            log.info(f"done loading h5 #parts={len(self.exs)}, #err={err_cnt}")

    def __getitem__(self, idx):
        # index 为三维坐标 (partid, part_epoch, part_data_idx)
        if isinstance(idx, int):
            # dev data
            s = 0
            for ex in self.exs:
                if s + len(ex) < idx:
                    s += len(ex)
                else:
                    ret = ex[(0, idx - s)]
                    break
        else:
            assert (
                len(idx) == 3
            ), f"idx format must be (`part_id`, `part_epoch`, `part_data_idx`), but got {idx} instead"
            part_id, epoch, idx = idx
            ret = self.partid2ex[part_id][(epoch, idx)]
        ret = self.fn(ret, idx)
        ret.update(data_id=idx)
        # log.info(f"index:{idx}, input_ids: {ret['input_ids'][0:10]}")
        return ret

    def __len__(self):
        assert self._load
        return sum(map(len, self.exs))

    def __iter__(self):
        # print(f"real len: {len(self)}")
        for i in range(len(self)):
            yield self[i]


class PretrainTask:
    def __init__(self, data_dir, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.mask_gen = None

    def train_data(
        self,
        max_seq_len=512,
        stride=None,
        overlap_len=0,
        rng=None,
        weights=None,
        evaluate=False,
        seed=0,
        num_consecutive=1,
        shuffle=True,
        combine_batch=1,
        load_process_num=0,
    ):
        if isinstance(self.data_dir[0][0], list):
            path = [i[0] for i in self.data_dir if not i[0][0].endswith("meta")]
        else:
            path = [i for i in self.data_dir if not i[0].endswith("meta")]
        if not weights:
            weights = [(None, None, i) for i in range(len(path))]
        # assert max_seq_len > 0, f'max_mask_num too big! seqlen={max_seq_len}, max_mask_num={mask_generator.special_token_num}'
        examples = ExampleSet(
            [
                ExampleSetSingleDataSource(
                    p,
                    max_seq_len,
                    stride=stride,
                    weights=w,
                    seed=seed,
                    num_consecutive=num_consecutive,
                    shuffle=shuffle,
                    combine_batch=combine_batch,
                )
                for p, w in zip(path, weights)
            ],
            partial(
                self.example_to_feature,
                rng=rng,
                overlap_len=overlap_len,
                evaluate=evaluate,
            ),
            load_process_num=load_process_num,
        )
        return examples

    def example_to_feature(
        self,
        example,
        idx,
        rng,
        overlap_len,
        evaluate,
    ):
        if not rng:
            rng = random
        if evaluate:
            # print(f"eval index: {idx}")
            rng = random.Random(idx)

        if example.lossmask is not None:
            labels = [self.tokenizer.ignored_index if j == 0 else i for i, j in zip(example.ids, example.lossmask)]
            tokens = example.ids[:-1]
            lm_labels = labels[1:]
        else:
            _tokens = example.ids
            tokens, lm_labels = _tokens[:-1], _tokens[1:]
        if example.log_prob is not None:
            log_prob = example.log_prob[1:]
        else:
            log_prob = None

        if overlap_len and idx != 0:  # do overlap
            # log.info(f"apply overlaping: overlap_len: {overlap_len}")
            if isinstance(lm_labels, np.ndarray):
                lm_labels = lm_labels.tolist()
            lm_labels = [self.tokenizer.ignored_index] * len(lm_labels[:overlap_len]) + lm_labels[overlap_len:]
            assert len(lm_labels) == len(tokens), f"lm_labels:{len(lm_labels)} vs tokens:{len(tokens)}"

        assert len(tokens) == len(lm_labels), f"tokens:{len(tokens)} != labels:{len(lm_labels)}"
        token_ids = np.array(tokens, dtype="int64")
        lm_labels = np.array(lm_labels, dtype="int64")

        features = OrderedDict(input_ids=token_ids, labels=lm_labels, src_id=example.src, log_prob=log_prob)
        return features


IDTYPES_2_ID = {"text": 0, "image": 1, "video": 2, "audio": 3}
DEBUG_PRINT_CNT = 0



def pad_sequence(sequences, padding_value=0, fix_len=None):
    """Fill sequences(np.ndarray) into a fixed-length matrix."""
    # don't use any paddle.Tensor in collate-fn
    #   which prevent leakage in multi-process
    max_size = sequences[0].shape
    trailing_dims = tuple(max_size[1:])
    # print("trailing_dims: ", trailing_dims)

    max_len = max([s.shape[0] for s in sequences])
    if fix_len is not None:
        if fix_len < max_len:
            logger.warning(f"truncating example from {max_len} to {fix_len}")
        max_len = fix_len
    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = np.full(out_dims, padding_value, dtype=sequences[0].dtype)
    for i, tensor in enumerate(sequences):
        tensor = tensor[:max_len]
        length = tensor.shape[0]
        out_tensor[i, :length, ...] = tensor
    return out_tensor





def smart_concat(tensor, axis=0):
    """_summary_

    Args:
        tensor (_type_): _description_
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    if isinstance(tensor[0], paddle.Tensor):
        return paddle.concat(tensor, axis=axis)
    else:
        return np.concatenate(tensor, axis=axis)


def merge_fn_group_batch(
    tokenizer,
    batch,
    pad_to_max_seqlen=None,
    debug_print=1,
    shift_label=False,
    combine_batch: int = 1,
    image_dtype="bfloat16",
    doc_pack_attn=False,
):
    """
    batch 内 n合一
    """
    bsz = len(batch)
    global DEBUG_PRINT_CNT
    if pad_to_max_seqlen and shift_label:
        pad_to_max_seqlen += 1

    keys = list(batch[0].keys())

    if combine_batch > 1:
        _batch = []
        for group in [batch[i : i + combine_batch] for i in range(0, len(batch), combine_batch)]:

            if "src_id" in group[0]:
                src_lst = list(set([b["src_id"] for b in group]))
                assert len(src_lst) == 1, f"src_lst: {src_lst}"

            item = {}
            for k in keys:
                if group[0][k] is None:
                    item[k] = None
                    continue
                if isinstance(group[0][k], (int, float)):
                    item[k] = np.stack([i[k] for i in group], 0)
                else:
                    item[k] = np.concatenate([i[k] for i in group])
            _batch.append(item)
        batch = _batch
    ret = {}
    for k in keys:
        if isinstance(batch[0][k], (int, float)):
            ret[k] = np.stack([b[k] for b in batch], 0)
        elif k in ["src_id", "data_id", "data_type"]:
            ret[k] = np.concatenate([b[k] for b in batch])
        elif k == "images":
            to_concat = [b[k] for b in batch if b[k] is not None]
            if len(to_concat) != 0:
                assert image_dtype != "bfloat16", f"Currently, not support {image_dtype} for numpy"
                ret[k] = np.concatenate(to_concat, axis=0).astype(image_dtype)
            else:
                ret[k] = None
        elif k == "grid_thw" and batch[0][k] is not None:
            ret[k] = np.concatenate([b[k] for b in batch], axis=0).astype("int64")
            if pad_to_max_seqlen:
                tmp = max(0, pad_to_max_seqlen * bsz - ret[k].shape[0])
                if tmp > 0:
                    ret[k] = np.concatenate([ret[k], np.zeros([tmp, 3])], axis=0).astype("int64")
        elif k in ["audio_input_ids", "audio_labels"]:
            to_concat = [b[k] for b in batch if b[k] is not None]
            if len(to_concat) != 0:
                concat_audio_ids = smart_concat(to_concat)
                assert len(concat_audio_ids.shape) == 2, f"拼接完的audio_ids必须是2维tensor，且shape=[sum(frames), depth]"
                ret[k] = pad_sequence(
                    [concat_audio_ids],
                    padding_value=tokenizer.ignored_index,
                    fix_len=pad_to_max_seqlen * bsz,
                )[0]
                assert len(ret[k].shape) == 2, f"padding完的audio_ids 必须是2维tensor，且shape=[bsz*pad_to_max_seqlen, depth]"
            else:
                ret[k] = None
        else:
            if k == "input_ids":
                pad_value = tokenizer.pad_token_id
            elif k == "labels" or k == "image_type_ids":
                pad_value = tokenizer.ignored_index
            elif k == "token_type_ids":
                pad_value = IDTYPES_2_ID["text"]  # pad is also considered as text
            else:
                pad_value = 0

            if batch[0][k] is not None:
                ret[k] = pad_sequence(
                    [b[k] for b in batch],
                    padding_value=pad_value,
                    fix_len=pad_to_max_seqlen if k != "token_type_ids" else pad_to_max_seqlen + 1,
                )

    batch = ret


    if shift_label:
        batch["labels"] = batch["labels"][:, 1:]
        batch["input_ids"] = batch["input_ids"][:, :-1]

    if doc_pack_attn:
        # 计算inbatch_pack_offset 来实现DovAtten
        doc_marks = (batch["input_ids"] == 2).astype(np.int64)
        doc_marks[:, -1] = 1  # 每条样本最后一个位置的doc_marks为1
        _offset = np.where(doc_marks.reshape([-1]))[0]
        _offset = (_offset + 1).tolist()
        # 开头补 一个 0
        offset = np.expand_dims(np.array([0] + _offset, dtype=np.int64), axis=0)
        # 使用 -1 pad到固定长度
        offset = pad_sequence(offset, padding_value=-1, fix_len=batch["input_ids"].shape[1])
        batch["inbatch_pack_offset"] = offset

    return batch
