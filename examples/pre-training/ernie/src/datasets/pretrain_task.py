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

import atexit
import os
import math
import re
import random
import logging
from functools import partial
import numpy as np
from collections import OrderedDict, namedtuple
from typing import List

import paddle
import h5py
from time import time
from src.utils.ipc_server import IPCServer


log = logging.getLogger(__name__)


class IPCH5Resource:

    def __init__(self, path, name, server):

        self.path = path
        self.name = name
        self.server = server
        self._length = None
        self._to_bool = None

    def __getitem__(self, key):

        return self.server.call(self.path, "get", (self.path, self.name, key))

    def __len__(self):

        if self._length is None:
            self._length = self.server.call(self.path, "len", (self.path, self.name))
        return self._length

    def __bool__(self):

        if self._to_bool is None:
            self._to_bool = self.server.call(
                self.path, "to_bool", (self.path, self.name)
            )
        return self._to_bool


class IPCH5MetaResource:

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

    def __init__(self, paths, server_idx, server_num):

        self.fps = {}
        path_num = len(paths)
        start_t = time()
        for idx, path in enumerate(paths):
            assert path not in self.fps, path

            ds = h5py.File(path, mode="r")
            fp = ds["ds16"]
            assert (
                "ds16_tokenwise_type_id" not in ds
            ), f"this file maybe a multimodal H5, path={path}"
            if "ds16_lossmask" in ds:
                fp_lossmask = ds["ds16_lossmask"]
                assert len(ds["ds16_lossmask"]) == len(ds["ds16"]), (
                    len(ds["ds16_lossmask"]),
                    len(ds["ds16"]),
                )
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
            self.fps[path] = {
                "fp": fp,
                "lossmask": fp_lossmask,
                "meta": meta,
                "off": off,
                "log_prob": log_prob,
            }
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

    def __init__(self, paths):
        """
        __init__
        """
        self.paths = paths

    def __call__(self, server_idx, server_num):

        return DatasetHolder(self.paths, server_idx, server_num)


def create_ipc_h5_resources(paths, num_server):

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
            assert (
                len(cols) >= 3
            ), f"配比文件至少要4列，格式为：pattern weight num_parts - {cols}"
            pattern, w, num_parts = cols[:3]
            if len(cols) >= 4 and cols[3] in ["lm", "mm", "audio"]:
                data_type = cols[3]
            else:
                data_type = "mm" if "multimodal" in i else "lm"

            num_parts = int(num_parts)
            pattern = re.compile(pattern)
            patterns.append((pattern, float(w) / num_parts, data_type))
    return patterns


def parse_data_weight(weights, filelist):

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

    assert (
        len(datasets) >= world_size
    ), f"#filelist={len(datasets)} < world_size{world_size}"
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
    datasets = sorted(datasets, key=lambda d: d.weights, reverse=True)
    for d in datasets:
        this_bucket = np.argmin(bucketsize)
        buckets[this_bucket].append(d)
        bucketsize[this_bucket] += d.weights

    log.info(
        f"sharding dataset according to prob, group vs probs={[sum([rr.weights for rr in r])for r in buckets]}"
    )
    bucketsize = bucketsize[rank]
    diff = bucketsize - (1 / world_size)
    log.info(
        f"unable to perfect shard. prob sum of this bucket:{bucketsize}, diff to perfect portion:{diff}"
    )
    assert (
        len(buckets) == world_size
    ), f"#ret={len(buckets)} prob not normalized:{[d.weights for d in datasets]}"
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
        self.combine_batch = combine_batch
        self.epoch = 0

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
            assert (
                "ds16_tokenwise_type_id" not in ds
            ), f"this file maybe a multimodal H5, src={self.src}"

            fp = ds["ds16"]
            if "ds16_lossmask" in ds:
                fp_lossmask = ds["ds16_lossmask"]
                assert len(ds["ds16_lossmask"]) == len(ds["ds16"]), (
                    len(ds["ds16_lossmask"]),
                    len(ds["ds16"]),
                )
            else:
                fp_lossmask = None
            # self.fp = self.fps[0]

            if "ds16_off" in ds:
                log.info("using ds w/ offset")
                off = ds["ds16_off"]
            else:
                off = None

            if "log_prob" in ds:
                log.info("using ds with log_prob")
                log_prob = ds["log_prob"]
            else:
                log_prob = None
            shape = fp.shape
            meta = {"shape": shape}
            if (
                shape[0] <= 0 or shape[0] >= 1000000000000
            ):  # 1000000000000 for max tokens of h5
                raise OSError
            self.fps.append(
                {
                    "fp": fp,
                    "lossmask": fp_lossmask,
                    "meta": meta,
                    "off": off,
                    "log_prob": log_prob,
                }
            )
            log.info(
                f"done loading {path}, shape:{shape}: int16:{self.int16_ds} "
                f"seqlen:{self.seqlen} stride:{self.stride}"
            )
        log.info(f"done loading part-{self.part}, file count: {len(self.fps)}")

    def __getitem__(self, idx):
        assert (
            len(idx) == 2
        ), f"idx format must be (`epoch, data_idx`), but got {idx} instead"
        epoch, idx = idx
        if idx == -1:
            return Example(
                ids=[],
                sids=None,
                task="lm",
                src=self.part,
                lossmask=None,
                log_prob=None,
            )
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
        ret = Example(
            ids=ids,
            sids=None,
            task="lm",
            src=self.part,
            lossmask=lossmask,
            log_prob=log_prob,
        )
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
        example_id = [
            (ii, min(ii + self.num_consecutive, len(self))) for ii in example_id
        ]
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
        assert paddle.io.get_worker_info() is None

        self.epoch = 0
        while 1:
            if self._data_status >= len(self):
                self._data_status -= len(self)
            else:
                log.debug(
                    f"...gen_index_from-[{self.part}]-[{self.epoch}]-offset-[{self.data_status}/{len(self)}]"
                )
                for s, e in self.example_id:
                    _length = (
                        math.ceil((e - s) / self.combine_batch) * self.combine_batch
                    )
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
        self.global_max_part_id = max([ex.part for ex in exs])
        self.partid2ex = {ex.part: ex for ex in exs}
        self.load_process_num = load_process_num

    def append(self, new_exs):
        log.info(f"updating exs, #new example: {len(new_exs)}")
        self.exs.append(new_exs)
        lens = [len(e) for e in self.exs]
        len_sum = sum(lens)
        log.info("multi task data portion")
        log.info(
            "\n".join([f"{e.path}={left/len_sum}" for left, e in zip(lens, self.exs)])
        )

    def load(self, use_shard, dp_rank, dp_size):
        self._load = True
        log.info(f"loading h5... use_shard={use_shard}, {self._load} {id(self)}")

        log.info(f"loading h5 in dp_env:{dp_rank}/{dp_size}")
        if use_shard:
            log.info("#shard train file, before load")

            def keyfn(e):
                left = e.path.strip("/").split("/")
                return left[0]

            path_per_dp = equal_shard(self.exs, dp_rank, dp_size)
            log.debug(
                f"using source shard, # files before shard={len(self.exs)}, after shard={len(path_per_dp)}"
            )
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
            assert (
                loaded_exs
            ), f"data_dir {[e.path for e in self.exs]} empty, #err:{err_cnt}"
            self.exs = loaded_exs
            if err_cnt > 0:
                raise ValueError(
                    f"some data load failed, #parts={len(self.exs)}, #err={err_cnt}"
                )
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
            labels = [
                self.tokenizer.ignored_index if j == 0 else i
                for i, j in zip(example.ids, example.lossmask)
            ]
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
            lm_labels = [self.tokenizer.ignored_index] * len(
                lm_labels[:overlap_len]
            ) + lm_labels[overlap_len:]
            assert len(lm_labels) == len(
                tokens
            ), f"lm_labels:{len(lm_labels)} vs tokens:{len(tokens)}"

        assert len(tokens) == len(
            lm_labels
        ), f"tokens:{len(tokens)} != labels:{len(lm_labels)}"
        token_ids = np.array(tokens, dtype="int64")
        lm_labels = np.array(lm_labels, dtype="int64")

        features = OrderedDict(
            input_ids=token_ids, labels=lm_labels, src_id=example.src, log_prob=log_prob
        )
        return features


class PretrainDummyDataset:
    """pretrain dummy dataset"""

    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx):
        return {
            "input_ids": np.array([1] * self.max_seq_len),
            "labels": np.array([1] * self.max_seq_len),
            "src_id": 0,
            "data_id": 0,
        }

    def __len__(self):
        return 10000

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
