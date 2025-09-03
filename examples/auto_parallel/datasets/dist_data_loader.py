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

from collections import OrderedDict
from itertools import groupby
from functools import reduce
from dataclasses import dataclass

import paddle
import paddle.distributed as dist
from paddle.utils.layers_utils import flatten, map_structure, pack_sequence_as
from paddleformers.data import DistDataLoader


class DistDataLoaderErnie(DistDataLoader):
    def __init__(
        self,
        dataset,
        batch_sampler=None,
        collate_fn=None,
        num_workers=0,
        prefetch_factor=2,
    ):
        super().__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        self._pp_data_group = self._hcg.get_pipe_parallel_group()

    def __next__(self):
        if self._need_data:
            data = next(self._dataloader_iter)
            input_ids = data["input_ids"]
            labels = data["labels"]
            assert {input_ids.dtype, labels.dtype} == {paddle.int64}, (
                f"Distloader requires dtype == `int64`, "
                f"got:{[input_ids.dtype, labels.dtype]}"
            )
        else:
            input_ids, labels = None, None
        pp_broadcast = (self._pp_data_group is None) or self.pp_rank == 0
        if self.mp_group is not None and self.mp_group.nranks > 1 and pp_broadcast:
            (
                input_ids,
                labels,
            ) = broadcast_data_obj(
                [
                    input_ids,
                    labels,
                ],
                self.mp_src_rank,
                self.mp_group,
            )

        if self._pp_data_group is not None and self._pp_data_group.nranks > 1:
            (
                input_ids,
                labels,
            ) = broadcast_data_obj(
                [
                    input_ids,
                    labels,
                ],
                self._pp_data_group.ranks[0],
                self._pp_data_group,
            )
        return OrderedDict([("input_ids", input_ids), ("labels", labels)])


@dataclass
class _DtypeSndShape:
    dtype: paddle.dtype
    shape: list

    def size(self):
        return reduce(lambda x, y: x * y, self.shape)


def split_group(grouped, split_size):
    ret = []
    while grouped:
        if sum([r[1].size() for r in ret]) > split_size:
            yield ret
            ret = []
        ret.append(grouped.pop())
    if ret:
        yield ret


def broadcast_data_obj(data, src_rank, group):
    this_rank = dist.get_rank()
    if this_rank == src_rank:
        template = [
            map_structure(
                lambda x: (
                    _DtypeSndShape(dtype=x.dtype, shape=x.shape)
                    if x is not None
                    else _DtypeSndShape(dtype="", shape=[0])
                ),
                data,
            )
        ]
    else:
        template = [None]
    dist.broadcast_object_list(template, src_rank, group)
    template = template[0]

    temp_flat = flatten(template)
    data_flat = flatten(data)

    def keyfn(i):
        return str(i[1].dtype)

    ret_flat = [-1 for _ in range(len(temp_flat))]
    for dtype, grouped in groupby(sorted(enumerate(temp_flat), key=keyfn), keyfn):
        grouped = list(grouped)
        for grouped_chunk in split_group(grouped, 2**18):
            idxs = [g[0] for g in grouped_chunk]
            if not dtype:
                for id in idxs:
                    ret_flat[id] = None
                continue

            data_buf_shapes = [
                reduce(lambda x, y: x * y, g[1].shape) for g in grouped_chunk
            ]
            if this_rank == src_rank:
                data_buf = paddle.concat([data_flat[i].reshape([-1]) for i in idxs], 0)
            else:
                data_buf = paddle.empty(
                    [sum(data_buf_shapes)], dtype=grouped_chunk[0][1].dtype
                )
            dist.broadcast(data_buf, src_rank, group)

            if this_rank != src_rank:
                if len(data_buf_shapes) == 1:
                    data_buf = [data_buf]
                else:
                    data_buf = data_buf.split(data_buf_shapes, axis=0)
                for g, data_chunk in zip(grouped_chunk, data_buf):
                    ret_flat[g[0]] = data_chunk.reshape(g[1].shape)

    if this_rank != src_rank:
        assert not [r for r in ret_flat if r is -1], ret_flat
        data = pack_sequence_as(template, ret_flat)
    return data
