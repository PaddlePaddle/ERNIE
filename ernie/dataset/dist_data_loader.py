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
import copy
import logging
import random
from collections import OrderedDict, defaultdict
from functools import reduce
from itertools import groupby

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.utils.layers_utils import flatten, map_structure, pack_sequence_as
from paddleformers.data import default_data_collator
from paddleformers.trainer.plugins.timer import get_timers
from paddleformers.utils.batch_sampler import DistributedBatchSampler

from ernie.dataset.vl_sft_reader.data_utils import merge_rope_3d_position
from ernie.modeling_moe_vl_pp import _DtypeSndShape
from ernie.utils.mm_data_utils import DATATYPE_2_ID

log = logging.getLogger(__name__)


class DummyDataset(paddle.io.Dataset):
    """
    A dummy Dataset class that has no elements.
    """

    def __len__(self):
        return 0


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


def broadcast_data_obj(data, src_rank, group):
    """
    Broadcast arbitrarily nested `data` structures, where each value must be a paddle.Tensor.
        Args:
            data: Arbitrarily nested `data` structure, where each value must be a paddle.Tensor.
            src_rank (int): Global rank of the sending node.
            this_rank (int): Local mp_rank of the current device.
            group (ProcessGroup): Communication group.

        Returns:
            data: The broadcasted `data`.
    """
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


def text_sft_collate_fn(batch):
    """
    Batch processing function for text SFT data.

    Args:
        batch (list): A list containing text SFT data,
        where each element is a dictionary consisting of text and SFT labels.

    Returns:
        np.ndarray: The processed batch data, returned as a numpy array.
    """
    batch = default_data_collator(batch, return_tensors="np")
    return batch


class MMDataloader(paddle.io.DataLoader):
    """
    MM dataloader
    """

    def __init__(
        self,
        dataset,
        tokenizer=None,
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
        multimodal_multiround_ratio=0.3,
    ):

        # dummy_dataset is a placeholder, not used
        if dataset is None:
            dataset = DummyDataset()
            batch_sampler = DistributedBatchSampler(dataset, 1)
            log.info("rank has no data, use Dummpy dataset")
        super().__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=1,
        )

        self._collate_fn = collate_fn
        self._dataloader = paddle.io.DataLoader(
            dataset,
            feed_list,
            places,
            return_list,
            batch_sampler,
            1,
            shuffle,
            drop_last,
            lambda x: x,  # collate_fn,
            num_workers,
            use_buffer_reader,
            prefetch_factor,
            use_shared_memory,
            timeout,
            worker_init_fn,
            persistent_workers,
        )
        self._lens_rcd = defaultdict(int)
        self._lens_images = defaultdict(int)
        self._sample_buffer = defaultdict(lambda: defaultdict(list))
        self._batch_buffer = defaultdict(list)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.eos_token = self.tokenizer.special_tokens_map.get("eos_token", "</s>")
        self.cls_token = self.tokenizer.special_tokens_map.get("cls_token", "<mask:0>")
        self.sep_token = self.tokenizer.special_tokens_map.get(
            "sep_token", "<|endofprompt|>"
        )
        log.info(
            f"cls token: {self.cls_token}, sep token: {self.sep_token}, eos token: {self.eos_token}"
        )
        self.cls_token_id = self.tokenizer._convert_token_to_id(self.cls_token)
        self.sep_token_id = self.tokenizer._convert_token_to_id(self.sep_token)
        self.eos_token_id = self.tokenizer._convert_token_to_id(self.eos_token)
        log.info(
            f"cls token: {self.cls_token_id}, sep token: {self.sep_token_id}, eos token: {self.eos_token_id}"
        )

        self._lazy_dataloader_iter = None
        self.rng = random.Random(2048)
        self.multimodal_multiround_ratio = multimodal_multiround_ratio
        self.need_multiround = self.rng.random() < self.multimodal_multiround_ratio

    def __len__(self):
        return super().__len__()

    @property
    def _dataloader_iter(self):
        if self._lazy_dataloader_iter is None:
            self._lazy_dataloader_iter = iter(self._dataloader)
        return self._lazy_dataloader_iter

    def sync_array_slices(self, buffer, remove_first=True):
        """
        Synchronize array slices.

        Args:
            buffer (dict): A dictionary containing data such as images, labels,
            input IDs, position IDs, and token type IDs.
            remove_first (bool, optional): Whether to remove the first element
            from the first slice. Defaults to True.

        Returns:
            dict: A dictionary containing the synchronized data, including input IDs,
            labels, position IDs, token type IDs,
                images, image type IDs, grid sizes, and data types.
        """

        # common operation
        cur_images = (
            paddle.concat(buffer["images"]) if len(buffer["images"]) > 0 else None
        )
        cur_image_type_ids = np.array(buffer["image_type_ids"])
        cur_data_type = (
            DATATYPE_2_ID["lm"] if cur_images is None else DATATYPE_2_ID["mm"]
        )
        cur_grid_thw = np.concatenate(buffer["grid_thw"], axis=0)

        # Helper function to slice arrays
        def slice_array(arr, remove_first, area, index):
            this_arr = arr[area[index][0] : area[index][1]]
            if remove_first and index == 0:
                return this_arr
            if not remove_first and index == len(area) - 1:
                return this_arr
            return this_arr[1:] if remove_first else this_arr[:-1]

        cur_input_ids = np.concatenate(buffer["input_ids"])[:-1]
        cur_labels = np.concatenate(buffer["labels"])[1:]
        buffer["input_ids"][-1] = buffer["input_ids"][-1][:-1]
        buffer["position_ids"][-1] = buffer["position_ids"][-1][:-1]

        # Apply the slicing consistently
        if len(buffer["input_ids"]) > 1:
            lengths = np.cumsum(
                np.array([0] + [len(arr) for arr in buffer["input_ids"]])
            )
            intervals = [(start, end) for start, end in zip(lengths[:-1], lengths[1:])]
            input_ids_batch = [
                slice_array(cur_input_ids, remove_first, intervals, idx)
                for idx in range(len(intervals))
            ]
            cur_input_ids = np.concatenate(input_ids_batch)
            cur_labels = np.concatenate(
                [
                    slice_array(cur_labels, remove_first, intervals, idx)
                    for idx in range(len(intervals))
                ]
            )
            cur_position_ids = np.concatenate(buffer["position_ids"], axis=0)
            # position ids
            cur_position_ids = [
                slice_array(cur_position_ids, remove_first, intervals, idx)
                for idx in range(len(intervals))
            ]
            if remove_first:
                cur_position_ids = [
                    position_id - 1 if idx != 0 else position_id
                    for idx, position_id in enumerate(cur_position_ids)
                ]
            cur_position_ids = merge_rope_3d_position(cur_position_ids)

            # token type ids
            cur_token_type_ids = np.concatenate(buffer["token_type_ids"], axis=0)
            cur_token_type_ids = np.concatenate(
                [
                    slice_array(cur_token_type_ids, remove_first, intervals, idx)
                    for idx in range(len(intervals))
                ]
            )

        else:
            input_ids_batch = np.array(copy.deepcopy(buffer["input_ids"]))
            cur_token_type_ids = np.concatenate(buffer["token_type_ids"])
            cur_image_type_ids = np.array(buffer["image_type_ids"])
            cur_grid_thw = np.concatenate(buffer["grid_thw"], axis=0)
            cur_position_ids = np.array(
                merge_rope_3d_position(buffer["position_ids"])[:-1]
            )

        return {
            "input_ids_batch": input_ids_batch,
            "input_ids": cur_input_ids,
            "labels": cur_labels,
            "position_ids": cur_position_ids,
            "token_type_ids": cur_token_type_ids,
            "images": cur_images,
            "image_type_ids": cur_image_type_ids,
            "grid_thw": cur_grid_thw,
            "data_type": cur_data_type,
        }

    def __iter__(self):
        while True:
            batch = next(self._dataloader_iter)
            for data in batch:
                data_not_valid = data.get("data_not_valid", 1)
                if data_not_valid:
                    log.info("[MMDataloader] mm data not valid.")
                    continue

                (
                    input_ids,
                    labels,
                    data_id,
                    part_id,
                    src_id,
                    example_id,
                    data_type,
                    images,
                    token_type_ids,
                    image_type_ids,
                    grid_thw,
                    position_ids,
                ) = (
                    data["input_ids"],
                    data["labels"],
                    data["data_id"],
                    data["part_id"],
                    data["src_id"],
                    data["example_id"],
                    data.get("data_type", None),
                    data.get("images", None),
                    data.get("token_type_ids", None),
                    data.get("image_type_ids", None),
                    data.get("grid_thw", None),
                    data.get("position_ids", None),
                )

                need_to_yield_sample = (
                    self._lens_rcd[src_id] + input_ids.shape[0]
                    > self.tokenizer.model_max_length
                )

                if need_to_yield_sample:
                    slice_result = self.sync_array_slices(
                        self._sample_buffer[src_id], self.need_multiround
                    )
                    example = {
                        "data_id": np.array(self._sample_buffer[src_id]["data_id"]),
                        "part_id": np.array(self._sample_buffer[src_id]["part_id"]),
                        "src_id": np.array(self._sample_buffer[src_id]["src_id"]),
                        "example_id": np.array(
                            self._sample_buffer[src_id]["example_id"]
                        ),
                        "need_multiround": self.need_multiround,
                    }
                    example.update(slice_result)

                    self._batch_buffer["cur_batch"].append(example)

                    self._lens_rcd[src_id] = 0
                    self._lens_images[src_id] = 0
                    self._sample_buffer[src_id] = defaultdict(list)
                    if len(self._batch_buffer["cur_batch"]) == self.batch_size:
                        batch_data = self._collate_fn(self._batch_buffer["cur_batch"])
                        for k in batch_data:
                            if batch_data[k] is not None:
                                batch_data[k] = paddle.to_tensor(batch_data[k])
                        self._batch_buffer["cur_batch"] = []
                        yield batch_data
                        self.need_multiround = (
                            self.rng.random() < self.multimodal_multiround_ratio
                        )
                self._sample_buffer[src_id]["input_ids"].append(input_ids)
                self._sample_buffer[src_id]["labels"].append(labels)
                self._sample_buffer[src_id]["data_id"].append(data_id)
                self._sample_buffer[src_id]["part_id"].append(part_id)
                self._sample_buffer[src_id]["src_id"].append(src_id)
                self._sample_buffer[src_id]["example_id"].append(example_id)
                self._sample_buffer[src_id]["data_type"].append(data_type)
                self._sample_buffer[src_id]["token_type_ids"].append(token_type_ids)
                self._sample_buffer[src_id]["image_type_ids"].extend(image_type_ids)
                self._sample_buffer[src_id]["images"].append(images)
                self._sample_buffer[src_id]["grid_thw"].append(grid_thw)
                self._sample_buffer[src_id]["position_ids"].append(position_ids)
                self._lens_rcd[src_id] += input_ids.shape[0]
                self._lens_images[src_id] += len(images)


class SFTDataLoader(paddle.io.DataLoader):
    """
    SFTDataLoader is a wrapper of paddle.io.DataLoader.
    """

    def __init__(self, dataset, num_workers=1, prefetch_factor=4):
        super().__init__(
            dataset=dataset,
            batch_size=None,
            collate_fn=text_sft_collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            use_shared_memory=True,
        )

        self.text_sft_dataloader = paddle.io.DataLoader(
            dataset,
            batch_size=None,
            collate_fn=text_sft_collate_fn,
            num_workers=num_workers,
            use_shared_memory=True,
            prefetch_factor=prefetch_factor,
        )

    def __iter__(self):
        while True:
            for batch in self.text_sft_dataloader:
                yield batch


class DistDataLoader(paddle.io.DataLoader):
    """
    DistDataLoader is a wrapper of paddle.io.DataLoader.
    """

    def __init__(
        self,
        dataset,
        tokenizer=None,
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
        is_train_text=False,
        is_train_mm=True,
        text_sft_dataset=None,
        gradient_accumulation_steps=1,
        multimodal_multiround_ratio=0.3,
        modality_ratio=[1, 1],
    ):

        if dataset is None:
            dataset = DummyDataset()
            batch_sampler = DistributedBatchSampler(dataset, 1)
            log.info("rank has no data, use Dummpy dataset")
        super().__init__(
            dataset=dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
        )
        self._lazy_dataloader_iter = None
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
        self.gradient_accumulation_steps = gradient_accumulation_steps
        log.info(
            f"[DistDataloader] gradient_accumulation_steps: {gradient_accumulation_steps}"
        )

        self.is_train_text = is_train_text
        self.is_train_mm = is_train_mm
        if self._need_data:
            self._dataloader = MMDataloader(
                dataset=dataset,
                tokenizer=tokenizer,
                feed_list=feed_list,
                places=places,
                return_list=return_list,
                batch_sampler=batch_sampler,
                batch_size=1,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=collate_fn,
                num_workers=num_workers,
                use_buffer_reader=use_buffer_reader,
                prefetch_factor=prefetch_factor,
                use_shared_memory=use_shared_memory,
                timeout=timeout,
                worker_init_fn=worker_init_fn,
                persistent_workers=persistent_workers,
                multimodal_multiround_ratio=0.3,
            )
            if text_sft_dataset is not None:
                self.text_sft_dataset = text_sft_dataset
                self.train_text = True
                text_sft_dataloader = SFTDataLoader(text_sft_dataset)
                self.text_sft_dataset_iter = iter(text_sft_dataloader)

        else:
            log.info(
                f"mp{self.mp_rank}_pp{self.pp_rank}_sharding{sharding_rank}_dp{self.dp_rank} no data needed, "
                "skip init dataloader."
            )

        self.counter = 0
        self.train_text = True
        self.count_mm = 0
        self.tokenizer = tokenizer
        self.modality_ratio = modality_ratio

    @property
    def _dataloader_iter(self):
        if self._lazy_dataloader_iter is None:
            self._lazy_dataloader_iter = iter(self._dataloader)
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
            ranks = [group[0], group[-1]]
            comm_group = paddle.distributed.new_group(ranks=ranks)
            if paddle.distributed.get_rank() in ranks:
                parallel_comm_group = comm_group
        return parallel_comm_group

    def __iter__(self):
        return self

    def position_id_assert(self, position_ids, input_ids, im_patch_id, grid_thw=None):
        """
        Test whether the position IDs for 3D RoPE are correct
        """
        accu_pos = 0
        accu_im_patch_id = 1
        if grid_thw is not None:
            grid_thw = grid_thw // 2
            grid_thw_list = [0] + [a[1] * a[2] for a in grid_thw]
        for idx, input_id in enumerate(input_ids):
            position_id = position_ids[idx]
            if input_id != im_patch_id:
                if position_id[0] != position_id[1] or position_id[2] != accu_pos:
                    return False

                accu_pos += 1
            if input_id == im_patch_id:
                assert position_id[0] == accu_pos
                for ik, thw_sum in enumerate(grid_thw_list[1:]):
                    if accu_im_patch_id <= sum(grid_thw_list[: 2 + ik]):
                        inner_image_idx = (
                            accu_im_patch_id - sum(grid_thw_list[: 1 + ik]) - 1
                        )
                        if (
                            position_id[1]
                            != accu_pos + inner_image_idx // grid_thw[ik][2]
                        ):
                            return False
                        if (
                            position_id[2]
                            != accu_pos
                            + inner_image_idx
                            - inner_image_idx // grid_thw[ik][2] * grid_thw[ik][2]
                        ):
                            return False

                        break
                accu_im_patch_id += 1
            if (
                idx < len(input_ids) - 1
                and input_id == im_patch_id
                and input_ids[idx + 1] != im_patch_id
            ):
                accu_pos = max(position_id) + 1
        return True

    def __next__(self):
        pp_broadcast = (self._pp_data_group is None) or self.pp_rank == 0
        get_timers() and get_timers()("read-raw-data").start()

        if self.is_train_mm and not self.is_train_text:
            self.train_text = False
        elif not self.is_train_mm and self.is_train_text:
            self.train_text = True
        else:
            if self.counter % self.gradient_accumulation_steps == 0:
                mm_ratio = int(self.modality_ratio[0])
                text_ratio = int(self.modality_ratio[1])
                sum_ratio = mm_ratio + text_ratio
                if self.count_mm % sum_ratio < text_ratio:
                    self.train_text = True
                else:
                    self.train_text = False
                self.count_mm += 1
        self.counter += 1

        if self.train_text:
            if self._need_data:
                data = next(self.text_sft_dataset_iter)
                log.info("[DistDataloader] train text.")
            else:
                data = None

            if self.mp_group is not None and self.mp_group.nranks > 1 and pp_broadcast:
                data = broadcast_data_obj(
                    data,
                    self.mp_src_rank,
                    self.mp_group,
                )

            if self._pp_data_group is not None and self._pp_data_group.nranks > 1:
                data = broadcast_data_obj(
                    data,
                    self._pp_data_group.ranks[0],
                    self._pp_data_group,
                )

            if data is not None:
                (
                    input_ids,
                    inbatch_pack_offset,
                    labels,
                    token_type_ids,
                    task_ids,
                    exact_total_task_ids,
                    data_type,
                    position_ids,
                ) = (
                    data["input_ids"],
                    data["inbatch_pack_offset"],
                    data["labels"],
                    data["token_type_ids"],
                    data["task_ids"],
                    data["exact_total_task_ids"],
                    data["data_type"],
                    data["position_ids"],
                )

                if inbatch_pack_offset is not None:
                    assert {inbatch_pack_offset.dtype} == {paddle.int64}, (
                        f"Distloader requires dtype == `int64`, "
                        f"got: inbatch_pack_offset.dtype: {inbatch_pack_offset.dtype}"
                    )
                    inbatch_pack_offset.stop_gradient = True
            else:
                (
                    input_ids,
                    inbatch_pack_offset,
                    labels,
                    token_type_ids,
                    task_ids,
                    exact_total_task_ids,
                    data_type,
                    position_ids,
                ) = (None, None, None, None, None, None, None, None)

            to_return = OrderedDict(
                [
                    ("input_ids", input_ids),
                    ("labels", labels),
                    ("inbatch_pack_offset", inbatch_pack_offset),
                    ("token_type_ids", token_type_ids),
                    ("task_ids", task_ids),
                    ("exact_total_task_ids", exact_total_task_ids),
                    ("data_type", data_type),
                    ("position_ids", position_ids),
                ]
            )
        else:
            if self._need_data:
                data = next(self._dataloader_iter)
                log.info("[DistDataloader] train multimodal.")
                images = data["images"]
                image_type_ids = data["image_type_ids"]
                grid_thw = data.get("grid_thw", None)
                position_ids = data.get("position_ids", None)
            else:
                data = None

            if self.mp_group is not None and self.mp_group.nranks > 1 and pp_broadcast:
                data = broadcast_data_obj(
                    data,
                    self.mp_src_rank,
                    self.mp_group,
                )

            if self._pp_data_group is not None and self._pp_data_group.nranks > 1:
                data = broadcast_data_obj(
                    data,
                    self._pp_data_group.ranks[0],
                    self._pp_data_group,
                )

            if data is not None:
                (
                    input_ids,
                    labels,
                    data_id,
                    part_id,
                    src_id,
                    data_type,
                    images,
                    token_type_ids,
                    image_type_ids,
                    inbatch_pack_offset,
                    grid_thw,
                    position_ids,
                ) = (
                    data["input_ids"],
                    data["labels"],
                    data["data_id"],
                    data["part_id"],
                    data["src_id"],
                    data.get("data_type", None),
                    data.get("images", None),
                    data.get("token_type_ids", None),
                    data.get("image_type_ids", None),
                    data.get("inbatch_pack_offset", None),
                    data.get("grid_thw", None),
                    data.get("position_ids", None),
                )
                assert {
                    input_ids.dtype,
                    labels.dtype,
                    data_id.dtype,
                    src_id.dtype,
                    part_id.dtype,
                } == {paddle.int64}, (
                    f"Distloader requires dtype == `int64`, "
                    f"got:{[input_ids.dtype, labels.dtype, data_id.dtype, part_id.dtype]}"
                )
                if inbatch_pack_offset is not None:
                    assert {inbatch_pack_offset.dtype} == {paddle.int64}, (
                        f"Distloader requires dtype == `int64`, "
                        f"got: inbatch_pack_offset.dtype: {inbatch_pack_offset.dtype}"
                    )
                    inbatch_pack_offset.stop_gradient = True
            else:
                (
                    input_ids,
                    labels,
                    data_id,
                    part_id,
                    src_id,
                    data_type,
                    images,
                    token_type_ids,
                    image_type_ids,
                    inbatch_pack_offset,
                    grid_thw,
                    position_ids,
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
                )
            to_return = OrderedDict(
                [
                    ("input_ids", input_ids),
                    ("labels", labels),
                    ("data_id", data_id),
                    ("part_id", part_id),
                    ("src_id", src_id),
                    ("data_type", data_type),
                    ("images", images),
                    ("token_type_ids", token_type_ids),
                    ("image_type_ids", image_type_ids),
                    ("inbatch_pack_offset", inbatch_pack_offset),
                    ("grid_thw", grid_thw),
                    ("position_ids", position_ids),
                ]
            )
            optional_keys = [
                "data_type",
                "images",
                "token_type_ids",
                "image_type_ids",
                "inbatch_pack_offset",
                "grid_thw",
                "position_ids",
            ]
            none_keys = [
                k for k, v in to_return.items() if v is None and k in optional_keys
            ]
            for k in none_keys:
                to_return.pop(k)
        get_timers() and get_timers()("read-raw-data").stop()

        return to_return
