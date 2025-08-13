# -*- coding: utf-8 -*-
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
DistDataLoader is a wrapper of paddle.io.DataLoader.
It is used to support hybrid parallelism.
It can replace paddle.io.DataLoader in most cases.
"""
import logging
from collections import OrderedDict
from itertools import groupby
from functools import reduce
from dataclasses import dataclass

import numpy as np
import os
import paddle
from paddle.distributed import fleet
import paddle.distributed as dist
from paddle.utils.layers_utils import flatten, map_structure, pack_sequence_as

from paddleformers.utils.batch_sampler import DistributedBatchSampler
from paddleformers.trainer.plugins.timer import get_timers
from paddleformers.utils.tools import get_env_device

from src.utils.misc import global_training_logs

logger = logging.getLogger(__name__)


log = logging.getLogger(__name__)

_MAX_DATA_DIM = 64

VOCAB_SIZE = os.getenv("VOCAB_SIZE")
G_DEBUG_DATA_MD5 = os.getenv("G_DEBUG_DATA_MD5")


class DummyDataset(paddle.io.Dataset):
    def __len__(self):
        return 0


class DistDataLoader(paddle.io.DataLoader):
    """
    DistDataLoader is a wrapper of paddle.io.DataLoader.
    """

    def __init__(
        self,
        dataset,
        feed_list=None,
        places=None,
        return_list=True,
        batch_sampler=None,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=None,
        num_workers=0,
        use_buffer_reader=True,
        prefetch_factor=2,
        use_shared_memory=True,
        timeout=0,
        worker_init_fn=None,
        persistent_workers=False,
        need_data=True,
        pp_broadcast=True,
        need_magic_trans=False,
    ):
        if dataset is None:
            dataset = DummyDataset()
            batch_sampler = DistributedBatchSampler(dataset, 1)
            log.info("rank has no data, use Dummpy dataset")
        super().__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        self.need_magic_trans = need_magic_trans
        # log.info(f'DistDataloader using image-dtype: {self.image_dtype}')
        self._hcg = fleet.get_hybrid_communicate_group()

        # init pp data comm group
        if self._hcg.get_pipe_parallel_world_size() > 1 and pp_broadcast:
            self._pp_data_group = self._init_dataloader_comm_group()
        else:
            log.info("skip pp broadcast")
            self._pp_data_group = None

        # tensor parallel message
        self.mp_rank = self._hcg.get_model_parallel_rank()
        self.mp_group = self._hcg.get_model_parallel_group()
        self.mp_src_rank = self._hcg.get_model_parallel_group_src_rank()

        self.pp_rank = self._hcg.get_stage_id()
        self.dp_rank = self._hcg.get_data_parallel_rank()
        sharding_rank = self._hcg.get_sharding_parallel_rank()
        self._need_data = need_data
        if self._need_data:
            self._dataloder = paddle.io.DataLoader(
                dataset,
                feed_list,
                places,
                return_list,
                batch_sampler,
                batch_size,
                shuffle,
                drop_last,
                collate_fn,
                num_workers,
                use_buffer_reader,
                prefetch_factor,
                use_shared_memory,
                timeout,
                worker_init_fn,
                persistent_workers,
            )

            # self._dataloder_iter = iter(self._dataloder)
            self._lazy_dataloader_iter = None
        else:
            log.info(
                "mp{}_pp{}_sharding{}_dp{} no data needed, "
                "skip init dataloader.".format(
                    self.mp_rank, self.pp_rank, sharding_rank, self.dp_rank
                )
            )

    @property
    def _dataloder_iter(self):
        if self._lazy_dataloader_iter is None:
            self._lazy_dataloader_iter = iter(self._dataloder)
        return self._lazy_dataloader_iter

    def __len__(self):
        if self._need_data:
            return super().__len__()
        else:
            raise ValueError(
                "raise error for `paddlenlp.trainer.trainer_utils.has_length`"
            )

    def _init_dataloader_comm_group(self):
        topo = self._hcg._topo
        parallel_comm_group = None
        parallel_groups = topo.get_comm_list("pipe")

        for group in parallel_groups:
            # only first rank and last rank
            if self.need_magic_trans:
                assert (
                    len(group) > 2
                ), f"magic_trans need ranks in group greater than 2, but get {len(group)}"
                ranks = [group[0], group[-2], group[-1]]
            else:
                ranks = [group[0], group[-1]]
            comm_group = paddle.distributed.new_group(ranks=ranks)
            if paddle.distributed.get_rank() in ranks:
                parallel_comm_group = comm_group
        return parallel_comm_group

    def __iter__(self):
        return self

    def __next__(self):
        get_timers() and get_timers()("read-raw-data").start()
        if self._need_data:
            # {'input_ids': int64, 'labels': int64, 'data_id': int64}
            data = next(self._dataloder_iter)
            if "data_not_valid" in data:
                global_training_logs.update(
                    data_not_valid=data["data_not_valid"].astype("float32").mean()
                )
            (
                input_ids,
                labels,
                data_id,
                src_id,
                data_type,
                images,
                token_type_ids,
                image_type_ids,
                audio_input_ids,
                audio_labels,
                grid_thw,
                inbatch_pack_offset,
                position_ids,
                log_prob,
            ) = (
                data["input_ids"],
                data["labels"],
                data["data_id"],
                data["src_id"],
                data.get("data_type", None),
                data.get("images", None),
                data.get("token_type_ids", None),
                data.get("image_type_ids", None),
                data.get("audio_input_ids", None),
                data.get("audio_labels", None),
                data.get("grid_thw", None),
                data.get("inbatch_pack_offset", None),
                data.get("position_ids", None),
                data.get("log_prob", None),
            )
            assert {input_ids.dtype, labels.dtype, data_id.dtype, src_id.dtype} == {
                paddle.int64
            }, (
                f"Distloader requires dtype == `int64`, "
                f"got:{[input_ids.dtype, labels.dtype, data_id.dtype, src_id.dtype]}"
            )
        else:
            (
                input_ids,
                labels,
                data_id,
                src_id,
                data_type,
                images,
                token_type_ids,
                image_type_ids,
                audio_input_ids,
                audio_labels,
                grid_thw,
                inbatch_pack_offset,
                position_ids,
                log_prob,
            ) = (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        get_timers() and get_timers()("read-raw-data").stop()

        # broadcast data
        pp_broadcast = (self._pp_data_group is None) or self.pp_rank == 0
        if self.mp_group is not None and self.mp_group.nranks > 1 and pp_broadcast:
            (
                input_ids,
                labels,
                data_id,
                src_id,
                data_type,
                images,
                token_type_ids,
                image_type_ids,
                audio_input_ids,
                audio_labels,
                grid_thw,
                inbatch_pack_offset,
                position_ids,
                log_prob,
            ) = broadcast_data_obj(
                [
                    input_ids,
                    labels,
                    data_id,
                    src_id,
                    data_type,
                    images,
                    token_type_ids,
                    image_type_ids,
                    audio_input_ids,
                    audio_labels,
                    grid_thw,
                    inbatch_pack_offset,
                    position_ids,
                    log_prob,
                ],
                self.mp_src_rank,
                self.mp_group,
            )

        if self._pp_data_group is not None and self._pp_data_group.nranks > 1:
            # NOTE(shenliang03): in last stage in pp, we don't need input_ids and data_id.
            # But it's only for paddle-new_model_7 compatible upgrade. It will remove in future.
            (
                input_ids,
                labels,
                data_id,
                src_id,
                data_type,
                images,
                token_type_ids,
                image_type_ids,
                audio_input_ids,
                audio_labels,
                grid_thw,
                inbatch_pack_offset,
                position_ids,
                log_prob,
            ) = broadcast_data_obj(
                [
                    input_ids,
                    labels,
                    data_id,
                    src_id,
                    data_type,
                    images,
                    token_type_ids,
                    image_type_ids,
                    audio_input_ids,
                    audio_labels,
                    grid_thw,
                    inbatch_pack_offset,
                    position_ids,
                    log_prob,
                ],
                self._pp_data_group.ranks[0],
                self._pp_data_group,
            )

        if VOCAB_SIZE is not None:
            if input_ids is not None:
                input_ids %= int(VOCAB_SIZE)
            if labels is not None:
                labels %= int(VOCAB_SIZE)

        to_return = OrderedDict(
            [
                ("input_ids", input_ids),
                ("labels", labels),
                ("data_id", data_id),
                ("src_id", src_id),
                ("data_type", data_type),
                ("images", images),
                ("token_type_ids", token_type_ids),
                ("image_type_ids", image_type_ids),
                ("audio_input_ids", audio_input_ids),
                ("audio_labels", audio_labels),
                ("grid_thw", grid_thw),
                ("inbatch_pack_offset", inbatch_pack_offset),
                ("position_ids", position_ids),
            ]
        )
        optional_keys = [
            "data_type",
            "images",
            "token_type_ids",
            "image_type_ids",
            "audio_input_ids",
            "audio_labels",
            "grid_thw",
            "inbatch_pack_offset",
            "position_ids",
            "log_prob",
        ]
        none_keys = [
            k for k, v in to_return.items() if v is None and k in optional_keys
        ]
        for k in none_keys:
            to_return.pop(k)
        return to_return


def broadcast_data_list(data_list, datatype, comm_rank=0, comm_group=None, src_rank=0):
    """
    Broadcast data from src_rank to all ranks in comm_group.
    """
    # Move to GPU and broadcast.
    size_cpu = []
    if comm_rank == 0:
        for data in data_list:
            size_cpu.append(len(data.shape))
            size_cpu += data.shape
    size_cpu = size_cpu + [0] * (_MAX_DATA_DIM - len(size_cpu))
    size_cuda = paddle.to_tensor(size_cpu)
    paddle.distributed.broadcast(size_cuda, src_rank, group=comm_group).wait()

    size_cpu = size_cuda.tolist()
    i = 0
    numel = 0
    sizes = []
    while size_cpu[i] > 0:
        rank = size_cpu[i]
        this_size = size_cpu[i + 1 : i + 1 + rank]
        numel += int(np.prod(this_size))
        sizes.append(this_size)
        i += 1 + rank

    if comm_rank == 0:
        assert (
            data.dtype == datatype
        ), "input has data type {} which " "is different than {}".format(
            data.dtype, datatype
        )
        data_b = paddle.concat(
            [d.to(get_env_device()).reshape([-1]) for d in data_list], 0
        )
        assert numel == sum([d.numel().item() for d in data_list]), (
            numel,
            [d.numel().item() for d in data_list],
        )
    else:
        data_b = paddle.empty([numel], dtype=datatype).to(get_env_device())

    # Broadcast
    paddle.distributed.broadcast(data_b, src_rank, group=comm_group).wait()

    ret = []
    offset = 0
    for size in sizes:
        numel = int(np.prod(size))
        ret.append(data_b[offset : offset + numel].reshape(size))
        offset += numel

    return ret


@dataclass
class _DtypeSndShape:
    """_summary_

    Returns:
        _type_: _description_
    """

    dtype: paddle.dtype
    shape: list

    def size(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return reduce(lambda x, y: x * y, self.shape)


def split_group(grouped, split_size):
    """_summary_

    Args:
        grouped (_type_): _description_
        split_size (_type_): _description_

    Yields:
        _type_: _description_
    """
    ret = []
    while grouped:
        if sum([r[1].size() for r in ret]) > split_size:
            yield ret
            ret = []
        ret.append(grouped.pop())
    if ret:
        yield ret


# Tea.chen congmin(葱明) brodcast
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
    # log.info(f'[rank={dist.get_rank()}]: {template}')

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
            # log.info(f'[rank={dist.get_rank()}]: done broadcast data:{data_buf.shape}')

            if this_rank != src_rank:
                # log.info(f'[rank={dist.get_rank()}] split:{data_buf_shapes}')
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


class DistDataLoaderAuto(DistDataLoader):

    def _init_dataloader_comm_group(self):
        return self._hcg.get_pipe_parallel_group()

    def __next__(self):
        data_dict = super().__next__()

        input_list = []
        if "token_type_ids" in data_dict.keys():
            (
                input_ids,
                labels,
                data_id,
                src_id,
                data_type,
                images,
                token_type_ids,
                image_type_ids,
                grid_thw,
            ) = (
                data_dict["input_ids"],
                data_dict["labels"],
                data_dict["data_id"],
                data_dict["src_id"],
                data_dict["data_type"],
                data_dict.get("images", None),
                data_dict["token_type_ids"],
                data_dict.get("image_type_ids", None),
                data_dict.get("grid_thw", None),
            )

            data_world_size = max(self._hcg.get_data_parallel_rank(), 1) * max(
                self._hcg.get_sharding_parallel_rank(), 1
            )
            if images is None:
                images = paddle.zeros([1, 64, 64], dtype="uint8")
                has_images = paddle.full([data_world_size, 1], False, dtype="bool")
            else:
                raise NotImplementedError
                has_images = paddle.full([data_world_size, 1], True, dtype="bool")
            if image_type_ids is None:
                image_type_ids = paddle.zeros_like(token_type_ids)  # padding for dy2st
            input_list = [
                input_ids,
                labels,
                data_id,
                src_id,
                data_type,
                images,
                token_type_ids,
                image_type_ids,
                has_images,
                grid_thw,
            ]
        else:
            for key, data in data_dict.items():
                input_list.append(data)
        return OrderedDict([("input_ids", input_list), ("labels", [])])
