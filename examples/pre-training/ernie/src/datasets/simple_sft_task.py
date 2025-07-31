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
"""
一个简单的SFT dataset，没有伪多轮，memory等策略。
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import math
import json
import logging
import os
import random
import re
from collections import OrderedDict, namedtuple
from functools import partial

import datasets
import numpy as np

log = logging.getLogger(__name__)


def parse_data_weight(weights, filelist):
    assert len(filelist) == len(set(filelist)), "duplicated filelist"
    weight_filelist = {}
    with open(weights) as inf:
        patterns = []
        for i in inf:
            pattern, w, num_parts = i.strip().split()[:3]
            num_parts = int(num_parts)
            pattern = re.compile(pattern)
            patterns.append((pattern, float(w) / num_parts))
    for f in filelist:
        for pattern, w in patterns:
            if pattern.search(f):
                weight_filelist[f] = float(w)
                break
        else:
            log.warning(f"{f} does not match any pattern")

    total_w = sum(weight_filelist.values())
    weight_filelist = {k: v / total_w for k, v in weight_filelist.items()}
    return list(zip(*weight_filelist.items()))


def equal_shard(datasets, rank, world_size):
    """
    如果有权重，根据权重概率累计概率相等的原则切分 train part.
    没有权重直接均分parts.
    args:
        datasets: List[ExampleSetSingleDataSource]
        rank: int
        world_size: int
    """
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

    def find_min():
        bucketsize = [sum([d.weights for d in b]) for b in buckets]
        minw = min(bucketsize)
        for i, s in enumerate(bucketsize):
            if minw == s:
                return i
        else:
            raise ValueError(f"min not find:{bucketsize}")

    datasets = sorted(
        datasets, key=lambda d: d.weights, reverse=True
    )  # 先分大part，或许有利于均匀分发？
    for d in datasets:
        this_bucket = find_min()
        buckets[this_bucket].append(d)

    log.info(
        f"sharding dataset according to prob, group vs probs={[sum([rr.weights for rr in r])for r in buckets]}"
    )
    bucketsize = sum([d.weights for d in buckets[rank]])
    diff = bucketsize - (1 / world_size)
    log.info(
        f"unable to perfect shard. prob sum of this bucket:{bucketsize}, diff to perfect portion:{diff}"
    )
    assert (
        len(buckets) == world_size
    ), f"#ret={len(buckets)} prob not normalized:{[d.weights for d in datasets]}"
    return buckets[rank]


Example = namedtuple("Example", ["ids", "sids", "task", "labels"])


class ExampleSetSingleDataSource:
    """Use to pick data from json"""

    trigger_pat = re.compile(
        r".*(\[<search>\]|\[<response>\]|\[<compute>\]|\[<kg>\]|\[<imagegen>\]|\[<code>\]|\[<tts>\]|\[<tts2>\]|\[<kpc>\]|\[<prompt>\])(.*)",
        re.DOTALL | re.M,
    )
    knowledge_pat = re.compile(
        r"\[<(search-res|kg-res|prompt-res)>\](.*)\[</(search-res|kg-res|prompt-res)>\]",
        re.DOTALL | re.M,
    )

    def __init__(
        self,
        path,
        seqlen,
        tokenizer,
        weights=None,
        ignore_index=-100,
        shuffle: bool = False,
        num_consecutive: int = 1,
        seed: int = 42,
        combine_batch: int = 1,
    ):
        self.seqlen = seqlen
        self.weights = weights
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.path = os.path.expanduser(path)
        self.part = weights[2] if weights is not None else 0
        self._load = False

        self.num_consecutive = num_consecutive
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.combine_batch = combine_batch
        self._data_status = 0

    def load(self):
        self._load = True
        log.info(f"loading {self.path}, weights={self.weights}")
        self.lines = []
        if os.path.isdir(self.path):
            self.lines = datasets.load_from_disk(self.path)
        else:
            with open(self.path, mode="r") as f:
                for line in f:
                    self.lines.append(json.loads(line))

        self.meta = {"shape": [len(self.lines), self.seqlen]}
        log.info(
            f'done loading {self.path}, shape:{self.meta["shape"]}'
            f"seqlen:{self.seqlen}"
        )

    def line_to_example_sharegpt(self, line):
        # code from vicuna
        # 处理shareGPT数据，代码来自：
        # https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py
        targets = []
        input_ids = []

        _sep = ["\n", "\n", "\n"]
        sep = [self.tokenizer._convert_token_to_id(s) for s in _sep]
        for conversation in line["conversation"]:
            message = self.tokenizer.encode(
                conversation["role"] + ": " + conversation["content"],
                add_special_tokens=False,
            )["input_ids"]
            input_ids.extend(message)
            input_ids.extend(sep)
            length = len(message) + len(sep)
            if conversation["role"] == "一言":
                targets.extend(message + sep)
            else:
                targets.extend([self.ignore_index] * length)
        # if len(input_ids) > 4096:
        #     log.info(f'super long example: txt={self.tokenizer.decode(input_ids)}')
        # targets = targets[1:] + [ self.ignore_index ] # shift_label in collate-fn
        return Example(ids=input_ids, sids=None, task="lm", labels=targets)

    @classmethod
    def make_ids_and_target(cls, ids, lossmask, ignored_index):
        return sum(ids, []), sum(
            [i if l else ([ignored_index] * len(i)) for i, l in zip(ids, lossmask)], []
        )

    def line_to_example_sft(self, line):
        # 简单的SFT训练代码，没有包含伪多轮。
        # 处理SFT格式数据，一行记录包含如下结构
        # {'src': ['xxx', 'yyy'], 'tgt: ['xxx', 'yyy']}

        src, tgt = line["src"], line["tgt"]
        if not isinstance(src, list):
            src = [src]
        if not isinstance(tgt, list):
            tgt = [tgt]

        knowledge = []
        knowledge_type = ""
        match_trigger = len(tgt) > 1 and self.trigger_pat.match(tgt[-2])
        if match_trigger:
            knowledge = self.knowledge_pat.match(src[-1]).group(2)
            knowledge_type = match_trigger.group(1)
            if knowledge_type == "[<search>]":
                knowledge = f"参考文章1：{knowledge.strip()}\n根据以上参考文章回答问题，补全对话[CLS]"
            elif knowledge_type == "[<prompt>]":
                knowledge = knowledge + "</s>"
            tgt = tgt[:-2] + [tgt[-1]]
            src = src[:-2] + [src[-2]]
        # log.info(src)
        src = [
            self.tokenizer.encode("user: " + s + "<s>", add_special_tokens=False)[
                "input_ids"
            ]
            for s in src
        ]
        tgt = [
            self.tokenizer.encode("bot: " + s + "</s>", add_special_tokens=False)[
                "input_ids"
            ]
            for s in tgt
        ]
        if knowledge:
            knowledge = self.tokenizer.encode(knowledge, add_special_tokens=False)[
                "input_ids"
            ]

        ids, lossmask = [], []
        for s, t in zip(src, tgt):
            ids.extend([s, t])
            lossmask.extend([0, 1])
        if knowledge:  # 知识必出
            ids = [knowledge] + ids  # knowledge in the head
            lossmask = [0] + lossmask  # only opt K

        ids, targets = self.make_ids_and_target(
            ids, lossmask, self.tokenizer.ignored_index
        )
        # ids, targets = ids[:-1], targets[1:] #shift label in collate-fn
        return Example(ids=ids, sids=None, task="lm", labels=targets)

    def __getitem__(self, idx):
        assert self._load
        assert (
            len(idx) == 2
        ), f"idx format must be (`epoch, data_idx`), but got {idx} instead"
        _, idx = idx
        return self.line_to_example_sft(self.lines[idx])

    @property
    def example_id(self):
        example_id = range(0, len(self), self.num_consecutive)
        example_id = [
            (ii, min(ii + self.num_consecutive, len(self))) for ii in example_id
        ]
        if self.shuffle:
            rng = random.Random(self.epoch + self.seed + self.part)
            rng.shuffle(example_id)
        return example_id

    @property
    def num_examples(self):
        assert self.epoch == 0
        return len(list(range(0, len(self), self.num_consecutive)))

    @property
    def data_status(self):
        return self._data_status

    @data_status.setter
    def data_status(self, value):
        log.info(f"part-{self.part}-load_data_status: {value}")
        self._data_status = value

    def __len__(self):
        assert self._load
        return len(self.lines)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def sampler(self):
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
                    yield self.epoch, (s, e)
            self.epoch += 1


class ExampleSet:
    """use to manage all h5 data"""

    def __init__(self, exs, fn):
        self.exs = exs
        self.fn = fn
        self._load = False
        self.global_max_part_id = max([ex.part for ex in exs])
        self.partid2ex = {ex.part: ex for ex in exs}

    def append(self, new_exs):
        log.info(f"updating exs, #new example: {len(new_exs)}")
        self.exs.append(new_exs)
        lens = [len(e) for e in self.exs]
        len_sum = sum(lens)
        log.info("multi task data portion")
        log.info("\n".join([f"{e.path}={l/len_sum}" for l, e in zip(lens, self.exs)]))

    def load(
        self,
    ):
        self._load = True
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

        log.info(f"done loading h5 #parts={len(self.exs)}, #err={err_cnt}")

    def __getitem__(self, idx):
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
        return ret

    def __len__(self):
        assert self._load
        return sum(map(len, self.exs))

    def __iter__(self):
        # print(f"real len: {len(self)}")
        for i in range(len(self)):
            yield self[i]


class ChatSftTask:
    def __init__(self, data_dir, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.data_dir = data_dir

    def train_data(
        self,
        max_seq_len=512,
        rng=None,
        weights=None,
        evaluate=False,
    ):
        log.info(f"loading {self.data_dir}")
        path = [i for i in self.data_dir if not i.endswith("meta")]
        if not weights:
            weights = [None for p in path]

        examples = ExampleSet(
            [
                ExampleSetSingleDataSource(
                    p,
                    max_seq_len,
                    tokenizer=self.tokenizer,
                    weights=w,
                )
                for p, w in zip(path, weights)
            ],
            partial(
                self.example_to_feature,
                rng=rng,
                evaluate=evaluate,
            ),
        )
        return examples

    def example_to_feature(
        self,
        example,
        idx,
        rng,
        evaluate,
    ):
        if not rng:
            rng = random
        if evaluate:
            # print(f"eval index: {idx}")
            rng = random.Random(idx)

        tokens = list(example.ids)
        lm_labels = list(example.labels)

        token_ids = np.array(tokens, dtype="int64")
        lm_labels = np.array(lm_labels, dtype="int64")

        features = OrderedDict(input_ids=token_ids, labels=lm_labels)
        return features
