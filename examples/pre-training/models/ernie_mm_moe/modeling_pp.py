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
""" ErniemmForCausalLMPipe """

import logging
import math
import heapq
import numpy as np
from typing import Dict, List, Union, Optional
from collections import defaultdict
import json
from dataclasses import dataclass
from itertools import accumulate
import functools
import contextlib
from copy import deepcopy
from functools import partial, reduce

import paddle

from paddle import nn
from paddle.nn import functional as F
import paddle.distributed as dist
from paddle.utils.layers_utils import flatten, map_structure, pack_sequence_as
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
)
from paddle.distributed.fleet.layers.mpu.mp_layers import (
    VocabParallelEmbedding,
    ColumnParallelLinear,
)
from paddle.distributed.communication.batch_isend_irecv import _coalescing_manager

from models.comm_utils import (
    gather_varlen,
    profile,
    mp_slice,
    all_gather_varlen,
)
from models.sequence_parallel_utils import (
    ScatterOp,
    SliceVarlenOp,
    AllGatherVarlenOpV2,
    mark_as_sequence_parallel_parameter,
)
from models.ernie.modeling import (
    ErnieModel,
    LayerNorm,
    RMSNorm,
    FusedLayerNorm,
    ErnieMLP,
)

from models.moe.moe_layer import MOELayer, DeepEPMOELayer, DeepEPDropTokenMOELayer
from models.moe.moe_all_gather_layer import MOEAllGatherLayer, MOEAllGatherLayerV2
from models.moe.moe_layer_uneven import MOELayer as MOELayerSizeAll2All
from models.ernie_moe.modeling_pp import (
    get_hcg,
    EmptyLayer,
    ErnieEmbeddingPipe,
    PipelinePretrainedModel,
    get_pp_vp_split_layers,
    create_skip_config_for_refined_recompute,
    _parse_moe_group,
)
from models.ernie_moe.modeling import ErnieDecoderLayer as ErnieMoEDecoderLayer
from models.ernie_moe.modeling import (
    moe_statedict_cherry_pick,
    moe_statedict_local_id_to_global,
)

from models.ernie_mm_moe.modeling import (
    ResamplerModel,
    VariableResolutionResamplerModel,
    ErniePretrainingCriterion,
    ErniemmMoEForCausalLM,
    ErniemmMoEConfig,
    TokenType,
    get_backbone_lm_param_regex,
    monkey_patch_param_hook,
    AudioEmbedding,
    AudioLayerNorm,
    ErniemmMoEHead,
)
from models.image_encoder.eva_vit_model import (
    EVAVisionTransformer,
    EVAVisionTransformerConfig,
)
from models.dfnrope.modeling import (
    DFNRopeVisionTransformerPretrainedModel,
    DFNRopeVisionTransformerConfig,
)
from models.openclip import (
    CLIPVisionTransformer,
)

try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}


logger = logging.getLogger(__name__)


def showmem(msg):
    """debug use"""
    logger.info(
        f"""
    {msg}
    memory_allocated: {paddle.device.cuda.memory_allocated()/1024/1024/1024:.3f} GB
    max_memory_allocated: {paddle.device.cuda.max_memory_allocated()/1024/1024/1024:.3f} GB
    """
    )


class ErniePretrainingCriterionPipe(ErniePretrainingCriterion):
    """
    ErniePretrainingCriterionPipe
    适配 `token_type_ids`
    """

    def __init__(self, config):
        if config.use_recompute_loss_fn:
            config = deepcopy(config)
            config.sequence_parallel = False  # Do GatherOp in LMHead
        super().__init__(config)

    def forward(self, logits, labels):
        """forward"""
        assert len(labels) in {2, 3}, labels
        assert len(logits) in {4, 8}, logits
        if len(labels) == 2:
            token_type_ids_untouched, labels = labels
            audio_labels = None
        else:
            token_type_ids_untouched, labels, audio_labels = labels
        if self.config.use_recompute_loss_fn:
            token_type_ids, logits_text, logits_image, logits_audio, *head_and_bias = (
                logits
            )
        else:
            token_type_ids, logits_text, logits_image, logits_audio = logits
            head_and_bias = ()
        token_type_ids_shifted = token_type_ids[:, 1:]
        loss, _ = super().forward(
            logits_text,
            logits_image,
            labels,
            token_type_ids_shifted,
            token_type_ids_untouched,
            logits_audio,
            audio_labels,
            *head_and_bias,
        )
        return loss


@dataclass
class _DtypeSndShape:
    dtype: paddle.dtype
    shape: list

    def size(self):
        """size"""
        return reduce(lambda x, y: x * y, self.shape)


def gather_tensors_list_in_pp_group(
    inputs, offload_pp_data_chunk_size=0, merge_output=True
):
    """
    gather `inputs` from all pp group, send to pp 0 and pp -1
    Args:
        `inputs`: (nested) lists of tensor
        `offload_pp_data_chunk_size`
        `merge_output`:
            if specified, return tensors.
            if no specified, return list of results gather from each pp rank.
    """
    hcg = get_hcg()
    dp_group = hcg.get_pipe_parallel_group()
    dp_worldsize = hcg.get_pipe_parallel_world_size()
    dp_src_rank = dp_group.ranks[0]
    dp_src_rank_last = dp_group.ranks[-1]
    this_rank = dist.get_rank()
    if dp_worldsize <= 1:
        return inputs

    template = map_structure(
        lambda x: (
            _DtypeSndShape(dtype=x.dtype, shape=x.shape)
            if x is not None
            else _DtypeSndShape(dtype="", shape=(0,))
        ),
        inputs,
    )
    tensor_flat = flatten(inputs)

    all_template = []
    dist.all_gather_object(all_template, template, group=dp_group)

    gather_meta = []
    dtype = [temp.dtype for temp in flatten(all_template) if temp.dtype != ""]
    if len(dtype) == 0:  # world all none
        nones = sum(map_structure(lambda i: None, all_template), [])
        if this_rank in (dp_src_rank, dp_src_rank_last):
            return nones
        return None

    assert len(set(dtype)) == 1, dtype
    dtype = dtype[0]
    for temps_per_rank in all_template:
        if all([temp.dtype == "" for temp in flatten(temps_per_rank)]):
            gather_meta.append((None, None))
        else:
            gather_meta.append(
                (
                    [
                        sum([np.prod(temp.shape) for temp in flatten(temps_per_rank)]),
                    ],
                    dtype,
                )
            )

    if all([t is None for t in tensor_flat]):
        tensor = None
    else:
        tensor = paddle.concat(
            [t.reshape([-1]) for t in tensor_flat if t is not None], 0
        )

    gathered_tensor_first = gather_varlen(
        tensor,
        dp_src_rank,
        dp_group,
        offload_pp_data_chunk_size,
        all_shape_and_dtype=gather_meta,
    )
    gathered_tensor_last = gather_varlen(
        tensor,
        dp_src_rank_last,
        dp_group,
        offload_pp_data_chunk_size,
        all_shape_and_dtype=gather_meta,
    )

    gathered_tensor = (
        gathered_tensor_first
        if len(gathered_tensor_first) > 0
        else gathered_tensor_last
    )
    if not len(gathered_tensor):
        return None

    start, end = 0, 0
    ret = []
    for template in all_template:
        ret_per_rank = []
        for temp in flatten(template):
            if np.prod(temp.shape) == 0:
                ret_per_rank.append(None)
                continue
            end += np.prod(temp.shape)
            r = (
                gathered_tensor[start:end].clone().reshape(temp.shape)
            )  # remove clone will trigger 719 error
            ret_per_rank.append(r)
            start = end
        ret_per_rank = pack_sequence_as(template, ret_per_rank)
        ret.append(ret_per_rank)
    if merge_output:
        return sum(ret, [])
    return ret


def exchange_images_meta(images, group):
    """
    exchange_images_meta
    """
    batch_size = paddle.to_tensor(images.shape[0], dtype=paddle.int32)
    batch_size_list = []
    dist.stream.all_gather(
        batch_size_list, batch_size, group=group, use_calc_stream=True
    )
    total_batch_size = sum(batch_size_list)
    avg = total_batch_size // len(batch_size_list)
    remain = total_batch_size % len(batch_size_list)
    unbalanced_rank2size = dict(enumerate(batch_size_list))
    sorted_unbalanced_rank2size = dict(
        sorted(unbalanced_rank2size.items(), key=lambda item: item[1])
    )
    balanced_rank2size = {key: avg for key in sorted_unbalanced_rank2size.keys()}
    if remain > 0:
        for key in list(sorted_unbalanced_rank2size.keys())[-remain:]:
            balanced_rank2size[key] += 1
    diff_rank2size = {
        key: sorted_unbalanced_rank2size[key] - balanced_rank2size[key]
        for key in balanced_rank2size
    }
    return diff_rank2size


def reshard_images(send_recv_pairs, group, images, reshard_size):
    """
    reshard_images
    """
    rank_in_group = group.rank
    if reshard_size > 0:
        send_meta = list()
        for pair in send_recv_pairs:
            if rank_in_group == pair[0]:
                send_meta.append((pair[1], pair[2]))
        sections = [images.shape[0] - reshard_size]
        for tup in send_meta:
            sections.append(tup[1])
        images_list = paddle.split(images, num_or_sections=sections)
        tasks = []
        with _coalescing_manager(group, tasks):
            for i in range(1, len(images_list)):
                task = dist.isend(
                    images_list[i], group.ranks[send_meta[i - 1][0]], group=group
                )
                tasks.append(task)
        for task in tasks:
            task.wait()
        return images_list[0]
    elif reshard_size < 0:
        recv_images_list = [images]
        tasks = []
        with _coalescing_manager(group, tasks):
            for pair in send_recv_pairs:
                if rank_in_group == pair[1]:
                    data_shape = images.shape
                    data_shape[0] = pair[2]
                    data = paddle.empty(data_shape, dtype=images.dtype)
                    task = dist.irecv(data, group.ranks[pair[0]], group=group)
                    tasks.append(task)
                    recv_images_list.append(data)
        for task in tasks:
            task.wait()
        return paddle.concat(recv_images_list)
    else:
        return images


def get_send_recv_pairs(diff_rank2size):
    """
    get_send_recv_pairs
    """
    send_rank_size_pairs = list()
    recv_rank_size_pairs = list()
    for key in diff_rank2size:
        if diff_rank2size[key] > 0:
            send_rank_size_pairs.append((-diff_rank2size[key].item(), key))
        elif diff_rank2size[key] < 0:
            recv_rank_size_pairs.append((diff_rank2size[key].item(), key))
    # max heap
    heapq.heapify(send_rank_size_pairs)
    # min heap
    heapq.heapify(recv_rank_size_pairs)

    send_recv_pairs = list()
    while len(send_rank_size_pairs) > 0 and len(recv_rank_size_pairs) > 0:
        src_pair = heapq.heappop(send_rank_size_pairs)
        src_size = -src_pair[0]
        src_rank = src_pair[1]
        dst_pair = heapq.heappop(recv_rank_size_pairs)
        dst_size = -dst_pair[0]
        dst_rank = dst_pair[1]
        if src_size > dst_size:
            send_recv_pairs.append((src_rank, dst_rank, dst_size))
            heapq.heappush(send_rank_size_pairs, (dst_size - src_size, src_rank))
        elif src_size == dst_size:
            send_recv_pairs.append((src_rank, dst_rank, src_size))
        else:
            send_recv_pairs.append((src_rank, dst_rank, src_size))
            heapq.heappush(recv_rank_size_pairs, (src_size - dst_size, dst_rank))
    assert (
        len(send_rank_size_pairs) == 0 and len(recv_rank_size_pairs) == 0
    ), f"send={send_rank_size_pairs} and recv={recv_rank_size_pairs} heap should be empty"
    return send_recv_pairs


def partition_numbers(nums, m):
    """
    Args:
    nums: List[int] - Sorted list of positive integers
    m: int - Number of piles to partition into.

    Returns:
    List[List[int]] - A list containing m lists
    """
    heap = [(0, 0, i) for i in range(m)]
    heapq.heapify(heap)
    piles = [[] for _ in range(m)]
    for num in reversed(nums):
        sum_pile, count_pile, pile_index = heapq.heappop(heap)
        piles[pile_index].append(num)
        sum_pile += num[1] * num[2]
        count_pile += 1
        heapq.heappush(heap, (sum_pile, count_pile, pile_index))

    return piles


def shard_data_in_pp_group(
    fn,
    fwd_batch_size=128,
    scatter_size=2048,
    input_is_parallel=False,
    feature_shape=None,
    is_balanced=False,
    offload_pp_data_chunk_size=0,
    patches_per_image=256,
):
    """
    Args:
        `fn`: lambda x -> x. x的 dtype 必须为 bfloat16
        `fwd_batch_size`: 每次 vit forward 的图片数。每个 pp 分到的总图片如果超过`fwd_batch_size` 会拆分成若干个 batch 进行 forward。
        `scatter_size`: 每次执行 scatter 的 batch 大小（图片数）,如果pp0 图片数大于`scatter_size`，会拆分成若干个 batch 进行 scatter。
        `input_is_parallel`, 为 true 的话不执行 scatter.
    Returns:
        在 pp group 内对 `x` 做分发(对 `x` 沿着 dim0 切分），在每个 pp-rank 内运行`fn`，对运行结果做聚合并累计到 pp0 上。
        运行逻辑：
        xs = scatter(x)
        xs = fn(xs)
        y = gather(xs)
    """

    @functools.wraps(fn)
    @paddle.no_grad()
    def _wrapper(*args):
        if len(args) == 2:
            images, grid_thw = args
        else:
            images, grid_thw = args[0], None
        if grid_thw is not None:
            assert (
                input_is_parallel
            ), "input_is_parallel must be true when grid_thw is not None"
            assert (
                not is_balanced
            ), "is_balanced must be false when grid_thw is not None"
        hcg = get_hcg()
        dp_group = hcg.get_pipe_parallel_group()
        dp_worldsize = hcg.get_pipe_parallel_world_size()
        dp_src_rank = dp_group.ranks[0]
        this_rank = dist.get_rank()

        if dp_worldsize <= 1:
            if images is None:
                return None
            with paddle.no_grad():
                out = fn(images)
            return out

        if is_balanced:
            pp_sd_group = hcg.pp_sd_group
            with profile("balanced_exchange_images_meta"):
                diff_rank2size = exchange_images_meta(images, pp_sd_group)
                send_recv_pairs = get_send_recv_pairs(diff_rank2size)
            with profile("balanced_reshared_images"):
                images = reshard_images(
                    send_recv_pairs,
                    pp_sd_group,
                    images,
                    diff_rank2size[pp_sd_group.rank].item(),
                )

        if not input_is_parallel:
            # showmem('before pp shard')
            if dp_src_rank == this_rank:
                assert images.ndim == 4, images.shape
                full_image_shape = paddle.shape(images).cuda().astype("int32")
            else:
                full_image_shape = paddle.empty([4], dtype="int32")
            dist.broadcast(full_image_shape, dp_src_rank, group=dp_group)
            full_image_shape = full_image_shape.tolist()
            assert scatter_size % dp_worldsize == 0 and scatter_size >= dp_worldsize, (
                scatter_size,
                dp_worldsize,
            )
            # pad to multiply of `scatter_size`
            pad_size = (
                full_image_shape[0] + scatter_size - 1
            ) // scatter_size * scatter_size - full_image_shape[0]
            # logger.info(f"full_image:{full_image_shape}, pad_size:{pad_size}")
            shareded_images = paddle.empty(
                [(full_image_shape[0] + pad_size) // dp_worldsize]
                + full_image_shape[1:],
                dtype="uint8",
            )  # images dtype bfloat16
            for ichunk in range((full_image_shape[0] + pad_size) // scatter_size):
                if images is not None:
                    i = images[ichunk * scatter_size : (ichunk + 1) * scatter_size]
                    assert len(i) <= scatter_size, (len(i), scatter_size)
                    if len(i) < scatter_size:
                        pad_len = int(scatter_size - len(i)) * int(
                            np.prod(images.shape[1:])
                        )
                        # shit hack
                        i = F.pad(
                            i.astype("bfloat16").reshape([-1]), (0, pad_len)
                        ).astype("uint8")
                        i = i.reshape([scatter_size] + images.shape[1:])
                else:
                    i = None
                o = shareded_images[
                    ichunk
                    * scatter_size
                    // dp_worldsize : (ichunk + 1)
                    * scatter_size
                    // dp_worldsize
                ]
                dist.stream.scatter(o, i, dp_src_rank, dp_group, use_calc_stream=True)
            images = shareded_images  # release mem

        # logger.info(f'sharded image :{images.shape}')
        # showmem('before call vit')
        if images is not None and len(images) > 0:
            out = []
            with paddle.no_grad():
                if grid_thw is not None:
                    grid_thw = grid_thw[grid_thw > 0].reshape([-1, 3])
                    # 把grid_thw 按t展开进vit。
                    grid_thw = F.pad(
                        paddle.repeat_interleave(grid_thw[:, 1:], grid_thw[:, 0], 0),
                        [0, 0, 1, 0],
                        value=1,
                    )
                    grid_thw_cumsum = F.pad(paddle.prod(grid_thw, -1).cumsum(0), [1, 0])

                    assert grid_thw_cumsum[-1] == len(images), (
                        grid_thw_cumsum[-1],
                        len(images),
                    )
                    # logger.info(f"GRID_THW_CUMSUM:{grid_thw_cumsum}")
                    s = 0
                    for i in range(1, len(grid_thw)):
                        if (
                            grid_thw_cumsum[i] - grid_thw_cumsum[s]
                        ) >= fwd_batch_size * patches_per_image:
                            # logger.info(f"{patches_cumsum[s]}--{patches_cumsum[i]}")
                            # logger.info(f"{grid_thw_cumsum[s]}--{grid_thw_cumsum[i]}")
                            # logger.info(f"images----{images[grid_thw_cumsum[s]: grid_thw_cumsum[i]]}")
                            o = fn(
                                images[grid_thw_cumsum[s] : grid_thw_cumsum[i]],
                                grid_thw[s:i],
                            )
                            s = i
                            out.append(o)
                    if s < len(grid_thw):
                        # logger.info(f"final-{patches_cumsum[s]}--{patches_cumsum[-1]}")
                        # logger.info(f"final-{grid_thw_cumsum[s]}--{grid_thw_cumsum[-1]}")
                        o = fn(images[grid_thw_cumsum[s] :], grid_thw[s:])
                        out.append(o)
                else:
                    s = 0
                    while s < len(images):
                        i = images[s : s + fwd_batch_size]
                        assert len(i) > 0, i.shape
                        o = fn(
                            i,
                            None,
                        )
                        s += fwd_batch_size
                        out.append(o)
            if len(out) == 1:
                if is_balanced:
                    with profile("balanced_reshard_images_fea"):
                        reverse_send_recv_pairs = [
                            (p[1], p[0], p[2]) for p in send_recv_pairs
                        ]
                        out = reshard_images(
                            reverse_send_recv_pairs,
                            pp_sd_group,
                            out[0],
                            -diff_rank2size[pp_sd_group.rank].item(),
                        )
                else:
                    (out,) = out
            else:
                # I dont know why can not release GPU memory, so I using `_clear_data` to clear underlaying GPU memory
                if offload_pp_data_chunk_size > 0:
                    args[0]._clear_data()
                    images._clear_data()
                out = paddle.concat(out, 0)
                if is_balanced:
                    with profile("balanced_reshard_images_fea"):
                        reverse_send_recv_pairs = [
                            (p[1], p[0], p[2]) for p in send_recv_pairs
                        ]
                        out = reshard_images(
                            reverse_send_recv_pairs,
                            pp_sd_group,
                            out,
                            -diff_rank2size[pp_sd_group.rank].item(),
                        )
            # self.offload()
            out = out.contiguous()
        else:
            out = None

        if input_is_parallel:
            # gather var len
            gathered = gather_varlen(
                out, dp_src_rank, dp_group, offload_pp_data_chunk_size
            )
        else:
            gathered = []
            dist.stream.gather(
                out, gathered, dp_src_rank, dp_group, use_calc_stream=True
            )
            if gathered:  # 非 pp0 无此参数
                gathered = paddle.concat(gathered, 0)
                if pad_size > 0:
                    gathered = gathered[:-pad_size]
        if not len(gathered):
            return None
        return gathered

    return _wrapper


class ErnieMoELMHeadPipe(ErniemmMoEHead):
    """
    支持 token-type-ids 的 `ErnieMoELMHeadPipe`
    """

    def __init__(self, config):
        super().__init__(config)

    @property
    def embedding_weight(self):
        """embedding_weight property"""
        return getattr(self, "weight")

    def forward(self, args):
        """forward"""
        if len(args) == 2:
            token_type_ids, hidden_states = args
            inbatch_pack_offset = None
        else:
            token_type_ids, hidden_states, inbatch_pack_offset = args
        token_type_ids_shifted = token_type_ids[:, 1:]

        logits_text, logits_image, logits_audio = super().forward(
            hidden_states, token_type_ids_shifted
        )
        token_type_ids = token_type_ids.detach()
        token_type_ids.stop_gradient = True
        if self.config.use_recompute_loss_fn:
            # `use_recompute_loss_fn` 时，logits实际为对应位置的 hidden-state
            mm_head_weight = self.mm_head.weight if self.mm_head is not None else None
            mm_head_bias = self.mm_head.bias if self.mm_head is not None else None
            return (
                token_type_ids,
                logits_text,
                logits_image,
                logits_audio,
                self.weight,
                self.bias,
                mm_head_weight,
                mm_head_bias,
            )
        return token_type_ids, logits_text, logits_image, logits_audio


class EVAVisionTransformerPipe(EVAVisionTransformer):
    """
    VisionModelPipe 在单个 Pipe 中完成整个 Vit 的计算
    即可以作为 流水线并行的一层，
    也可以作为独立的 feature 抽取模块附加在 PipelineModel 中(通过 `add_vision_model` 方法)
    """

    def __init__(self, config):
        logger.info(f"VISION-CONFIG-{config.vision_config}")
        super().__init__(config.vision_config)
        # logger.info(f"VISION-CONFIG-{config.vision_config}")

    def extract_feature(self, images, grid_thw=None):
        """
        Args: images: Tensor[B,H,W,C]
        Returns: image_feautres: Tensor[B,Hh*Hw,D]
        """
        # logger.info(f"INPUT_IMAGES-{images.shape}")
        # logger.info(f"md5 of image: {md5(images)}")
        ctx = (
            paddle.no_grad
            if getattr(self.config, "freeze_vision", False)
            else contextlib.nullcontext
        )
        with ctx():
            _, _, image_features, _ = super().forward(
                images,
                return_all_features=True,
            )
        # logger.info(f"md5 of imagefea: {md5(image_features)}")
        return image_features

    def forward(self, args):
        """Pipeline 模型入口"""
        assert (
            len(args) == 2
        ), f"The number of arguments must be (`input_ids`、`images`) but got {len(args)}-{args}"
        input_ids, images = args
        image_features = self.extract_feature(images)
        return (input_ids, image_features)

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        return super()._get_tensor_parallel_mappings(config.vision_config, is_split)


class InceptionPipe(EVAVisionTransformerPipe):
    """VisionModelPipe + LiteV, 全部放在一个 Pipe 中完成"""

    def __init__(self, config):
        super().__init__(config)
        assert config.inception_config is not None
        config.inception_config.sequence_parallel = (
            False  # should not use sp due to varlen
        )
        self.inception_resampler = nn.Linear(
            config.vision_config.width,
            config.inception_config.hidden_size,
        )
        self.inception_model = ErnieModel(config.inception_config)
        # self.inception_model.embed_tokens = None
        # `inception_resampler` is the resampler between litev & vit, which shoud be freeze
        # vit.width -> ernie.hidden_size # assume vit no `needhead`
        # hard code: inception-resampler not use `reduce-token`

    def extract_feature(self, images, grid_thw=None):
        """
        Args: images: Tensor[B,H,W,C]
        Returns: image_feautres: Tensor[B,Hh*Hw,D]
        """
        ctx = (
            paddle.no_grad
            if getattr(self.config, "freeze_vision", False)
            else contextlib.nullcontext
        )
        with ctx():
            image_features = super().extract_feature(images)
            image_features = self.inception_resampler(image_features)
            out = self.inception_model(inputs_embeds=image_features, return_dict=True)
            image_features = out.last_hidden_state
        return image_features

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        vision_actions = EVAVisionTransformer._get_tensor_parallel_mappings(
            config.vision_config, is_split
        )
        inception_actions = ErnieModel._get_tensor_parallel_mappings(
            config.inception_config, is_split
        )
        for k in list(inception_actions.keys()):
            inception_actions["inception_model." + k] = inception_actions.pop(k)
        return {**vision_actions, **inception_actions}


class DFNRopeVisionTransformerPipe(DFNRopeVisionTransformerPretrainedModel):
    """
    纯 MP 下可以用
    """

    def __init__(self, config, use_full_recompute=False):
        self.sorted_thw = None
        self.sorted_idx = None
        self.seq_list = None
        self.new_thw = []
        self.pp_data_balance = getattr(config.vision_config, "pp_data_balance", False)
        self.attn_sep = (
            getattr(config.vision_config, "attn_sep", False)
            and config.tensor_parallel_degree > 1
        )
        self.use_full_recompute = use_full_recompute
        if self.use_full_recompute:
            logger.info("use full recompute, vision model will NOT use recompute inner")
            config.vision_config.use_recompute = False
        super().__init__(config.vision_config)
        if self.config.tensor_parallel_degree > 1:
            logger.info(
                "use sp extract feature, vit parameter will be marked as sequence parallel"
            )
            for p in self.parameters():
                mark_as_sequence_parallel_parameter(p)

    def extract_feature(self, images, grid_thw, second_fwd=False):
        """_summary_

        Args:
            images (_type_): _description_
            grid_thw (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.config.tensor_parallel_degree <= 1:
            image_features = self._extract_feature(images, grid_thw)
            images_indices = None
            if second_fwd:
                return image_features, images_indices
            return image_features
        else:
            grid_thw = grid_thw.clone()
            # logger.info("use sp extract feature")
            images_indices = []
            parallelism = self.config.tensor_parallel_degree
            capacity = (grid_thw.prod(-1).sum(-1) + parallelism - 1) // parallelism
            crop_sizes = grid_thw.prod(-1)
            crop_offset = crop_sizes.cumsum(0)
            rank_per_crop = paddle.maximum(
                (crop_offset - 1) // capacity, paddle.to_tensor(0)
            )
            image_size_per_rank = paddle.zeros([parallelism], dtype="int64")
            num_crop_per_rank = paddle.bincount(rank_per_crop, minlength=parallelism)
            image_size_per_rank = paddle.scatter(
                image_size_per_rank, rank_per_crop, crop_sizes, overwrite=False
            )
            # split_size = grid_thw[:, 0].sum() // parallelism
            # (s, t, j) = (0, 0, -1)
            # thw_indices = []
            # logger.info(f'grid_thw-{grid_thw}, images={images.shape}')
            # for i in range(len(grid_thw)):
            #     if s + grid_thw[i][0] >= split_size or len(grid_thw) - i <= (parallelism - len(thw_indices)):
            #         thw_indices.append(i - j)
            #         images_indices.append(t + grid_thw[i].prod())
            #         (s, t, j) = (0, 0, i)
            #         t = 0
            #     else:
            #         s += grid_thw[i][0]
            #         t += grid_thw[i].prod()
            # if t > 0:
            #     thw_indices[-1] += i - j
            #     images_indices[-1] += t
            thw_indices = num_crop_per_rank
            images_indices = image_size_per_rank
            # feas = []
            # for _im, _grid in zip(images.split(images_indices.tolist()), grid_thw.split(thw_indices.tolist())):
            #     if not len(_im):
            #         continue
            #     fea = self._extract_feature(_im, _grid)
            #     feas.append(fea)
            # feas=paddle.concat(feas)

            num_pad = 0
            if self.attn_sep:
                seqlen = images.shape[0]
                num_pad = math.ceil(seqlen / parallelism) * parallelism - seqlen
                images = paddle.nn.functional.pad(images, [0, num_pad, 0, 0], value=0)
                images_indices = [
                    images.shape[0] // parallelism for _ in range(parallelism)
                ]
                images = SliceVarlenOp.apply(images, images_indices)
            else:
                images = SliceVarlenOp.apply(images, images_indices)
                images = images.detach()
                grid_thw = mp_slice(grid_thw, thw_indices)

            if len(images):
                image_features = self._extract_feature(
                    images, grid_thw, num_pad=num_pad
                )
            else:
                image_features = paddle.empty(
                    [0, self.config.hidden_size],
                    dtype=self.patch_embed.proj.weight.dtype,
                )
                image_features.stop_gradient = (
                    self.patch_embed.proj.weight.stop_gradient
                )
            # sanity check
            if not second_fwd:
                image_features = AllGatherVarlenOpV2.apply(
                    image_features, images_indices
                )
                if self.attn_sep:
                    image_features = image_features[:seqlen, :]
            # diff = (feas-image_features).abs().mean()
            # logger.info(f'shard vs not shard : {image_features.dtype} {image_features.stop_gradient} {diff}')
            if second_fwd:
                return image_features, images_indices
            return image_features

    def _extract_feature(self, images, grid_thw, num_pad=0):
        """_summary_

        Args:
            images (_type_): _description_
            grid_thw (_type_): _description_

        Returns:
            _type_: _description_
        """
        ctx = (
            paddle.no_grad
            if getattr(self.config, "freeze_vision", False)
            else contextlib.nullcontext
        )
        with ctx():
            image_features = super().forward(images, grid_thw, num_pad)
        return image_features

    def forward(self, args):
        """_summary_

        Args:
            args (_type_): _description_

        Returns:
            _type_: _description_
        """
        # TODO 当前只支持 非单独pp存在。
        raise NotImplementedError
        # mp 切features and


class CLIPVisionTransformerPipe(CLIPVisionTransformer):
    """_summary_

    Args:
        CLIPVisionTransformer (_type_): _description_
    """

    def __init__(self, config):
        """_summary_

        Args:
            config (_type_): _description_
        """
        super().__init__(config.vision_config)

    def extract_feature(self, images, grid_thw=None):
        """_summary_

        Args:
            images (_type_): _description_
            grid_thw (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        ctx = (
            paddle.no_grad
            if getattr(self.config, "freeze_vision", False)
            else contextlib.nullcontext
        )
        with ctx():
            image_forward_out = super().forward(
                images,
                output_hidden_states=True,
            )
            image_features = image_forward_out.hidden_states[-1][:, 1:, :]
        return image_features

    def forward(self, args):
        """_summary_

        Args:
            args (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        # TODO 当前只支持 非单独pp存在。
        raise NotImplementedError


class ErniemmEmbeddingPipe(ErnieEmbeddingPipe):
    """Embedding + Resampler"""

    def __init__(self, config, use_full_recompute=False):
        config = deepcopy(config)
        sequence_parallel = config.sequence_parallel
        config.sequence_parallel = False  # disable inner`ScatterOp`
        self.use_full_recompute = use_full_recompute
        self.offload_resamler = config.pp_recompute_offload_resampler
        self.resamper_empty_cache = config.resamper_empty_cache
        out_dim = config.hidden_size
        super().__init__(config)
        if config.mm_vocab_size > 0:
            self.mm_embed_tokens = VocabParallelEmbedding(
                config.mm_vocab_size, config.hidden_size
            )
        else:
            self.mm_embed_tokens = None
        resampler_cls = (
            VariableResolutionResamplerModel
            if getattr(config.vision_config, "variable_resolution", False)
            else ResamplerModel
        )
        self.resampler_model = resampler_cls(
            (
                config.inception_config.hidden_size
                if getattr(config, "inception_config", False)
                else config.pixel_hidden_size
            ),
            config.hidden_size,
            config.spatial_conv_size,
            config.temporal_conv_size,
            config=config,
        )
        if config.audio_config is not None:
            self.audio_embed_tokens = AudioEmbedding(config)
            with paddle.utils.unique_name.guard("audio_adapter_"):
                if self.config.tensor_parallel_degree > 1:
                    self.audio_adaptor = ColumnParallelLinear(
                        config.audio_config["audio_hidden_size"],
                        config.hidden_size,
                        gather_output=True,
                        has_bias=True,
                        fuse_matmul_bias=True,
                    )
                else:
                    self.audio_adaptor = nn.Linear(
                        config.audio_config["audio_hidden_size"],
                        config.hidden_size,
                    )
            Norm = RMSNorm if config.use_rmsnorm else LayerNorm
            if not config.use_rmsnorm and config.fuse_ln:
                Norm = FusedLayerNorm
            if not config.use_rmsnorm and not config.fuse_ln:
                self.audio_after_norm = AudioLayerNorm(config)
            else:
                self.audio_after_norm = Norm(config)
        else:
            self.audio_embed_tokens = None
            self.audio_after_norm = None
        self.config = config
        self.scatter_output = sequence_parallel  # outer `ScatterOp`
        self.use_mem_eff_attn = config.use_mem_eff_attn

    def forward(self, args):
        """forward lm embedding + mm embedding + resampler"""
        # assert len(args) == 4, args
        super_forward = super().forward
        token_type_ids, input_ids, *args = args

        def get_args(args, need_inbatch, need_image, need_varres, need_pos, need_audio):
            """
            按照如下优先级获取args: inbatch, position-id, image, image_type_ids, grid_thw, audio_ids
            """
            assert isinstance(args, (tuple, list)), type(args)
            keys = [
                i
                for i, j in zip(
                    [
                        "inbatch",
                        "images",
                        "image_type_ids",
                        "grid_thw",
                        "position_ids",
                        "audio_input_ids",
                    ],  # args 的出现顺序
                    [
                        need_inbatch,
                        need_image,
                        need_image,
                        need_varres,
                        need_pos,
                        need_audio,
                    ],
                )
                if j
            ]
            args = dict(zip(keys, args))
            return (
                args.get("inbatch"),
                args.get("images"),
                args.get("image_type_ids"),
                args.get("grid_thw"),
                args.get("position_ids"),
                args.get("audio_input_ids"),
            )

        (
            inbatch_pack_offset,
            image_features,
            image_type_ids,
            grid_thw,
            position_ids,
            audio_input_ids,
        ) = get_args(
            args,
            self.use_mem_eff_attn,  # inbatch,
            self.config.vision_config is not None,  # image-type-ids
            getattr(self.config.vision_config, "variable_resolution", False),  # varres
            self.config.rope_3d,  # position-ids
            self.config.audio_config is not None,
        )
        if inbatch_pack_offset is not None:
            inbatch_pack_offset.stop_gradient = True

        if position_ids is not None:
            position_ids.stop_gradient = True

        token_type_ids_input = token_type_ids[..., :-1]
        token_type_ids_input_ori = token_type_ids_input.clone()
        image_mask = input_ids == self.config.im_patch_id

        token_type_ids_input = token_type_ids_input.flatten()
        input_ids = input_ids.flatten()

        token_type_ids_input[token_type_ids_input == TokenType.video] = TokenType.image
        input_ids.stop_gradient = False  # make recompute happy
        if image_features is not None:
            image_features.stop_gradient = False

        lm_input_ids = input_ids.clone()
        mm_input_ids = input_ids.clone()
        if self.mm_embed_tokens is not None:
            lm_input_ids[token_type_ids_input == TokenType.image] = 0
            mm_input_ids[token_type_ids_input == TokenType.text] = (
                self.config.max_text_id
            )

        def fwd(image_features, _):
            nonlocal input_ids, lm_input_ids, mm_input_ids, token_type_ids_input, image_type_ids, image_mask
            nonlocal audio_input_ids
            """recompute 段"""
            assert lm_input_ids.max() < self.config.vocab_size, lm_input_ids.tolist()
            if self.resamper_empty_cache:
                paddle.device.cuda.empty_cache()  # 128k need

            inputs_embeds = super_forward(lm_input_ids)

            if image_features is not None:  # 纯文样本，不会过Vit
                # mapping_forward
                image_features = self.resampler_model(
                    image_features,
                    image_mask,
                    token_type_ids_input_ori,
                    image_type_ids,
                    grid_thw,
                )
                # B, N, C = image_features.shape
                # image_features = image_features.reshape([B * N, C])

                if self.mm_embed_tokens is not None:
                    if self.resamper_empty_cache:
                        paddle.device.cuda.empty_cache()  # 128k need
                    mm_ids_features = self.mm_embed_tokens(
                        mm_input_ids - self.config.max_text_id
                    )
                    mm_ids_features = mm_ids_features.astype(inputs_embeds.dtype)
                    image_indices = paddle.nonzero(
                        token_type_ids_input == TokenType.image
                    ).flatten()
                    inputs_embeds = paddle.scatter_(
                        inputs_embeds,
                        image_indices,
                        paddle.gather(mm_ids_features, image_indices, axis=0),
                        overwrite=True,
                    )
                # else:
                # assert (mm_input_ids <= self.config.max_text_id).all().item(), (
                #     f"found vistual token in ids, but `mm_vocab_size` == 0, "
                #     f"ids:{input_ids}, max_text_id={self.config.max_text_id} "
                # )

                image_indices = paddle.nonzero(image_mask.flatten()).flatten()
                image_features = image_features.reshape([-1, image_features.shape[-1]])
                inputs_embeds = paddle.scatter_(
                    inputs_embeds,
                    image_indices,
                    image_features.astype(inputs_embeds.dtype),
                    overwrite=True,
                )

            if audio_input_ids is not None:
                D = self.config.audio_config["audio_encode_frame_depth"]
                # audio_input_ids.shape = [F (音频帧数), D (音频每帧深度)]
                # audio_features.shape = [F, D, AH (音频hidden_size) ]
                # audio_features_reduce.shape = [F, AH]
                # audio_features_proj.shape = [F, LH (文本hidden_size)]
                audio_input_ids = audio_input_ids.reshape([-1, D])

                audio_pad_mask = (
                    audio_input_ids
                    == self.config.audio_config["audio_special_tokens"]["PAD"]
                ) | (audio_input_ids == self.config.ignored_index)
                audio_unpad_mask = ~audio_pad_mask
                audio_input_ids[audio_pad_mask] = 0
                audio_pad_mask = audio_pad_mask.astype("float32")
                audio_unpad_mask = audio_unpad_mask.astype("float32")
                audio_features = self.audio_embed_tokens(audio_input_ids)

                audio_feature_unpad_mask = audio_features * audio_unpad_mask.unsqueeze(
                    [-1]
                )
                audio_features_reduce = paddle.sum(audio_feature_unpad_mask, axis=1)

                audio_indices = token_type_ids_input == TokenType.audio
                audio_frame_num = paddle.sum(
                    audio_indices.astype("int64")
                )  # 计算实际音频帧数
                audio_features_reduce = audio_features_reduce[:audio_frame_num]
                audio_features_reduce = audio_features_reduce.astype(
                    inputs_embeds.dtype
                )
                audio_features_proj = self.audio_adaptor(audio_features_reduce)
                inputs_embeds[audio_indices] = audio_features_proj

            if self.scatter_output:
                inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
                inputs_embeds = ScatterOp.apply(inputs_embeds)
            else:
                inputs_embeds = inputs_embeds.reshape(
                    token_type_ids_input_ori.shape + [inputs_embeds.shape[-1]]
                )

            if self.resamper_empty_cache:
                paddle.device.cuda.empty_cache()  # 128k need
            return inputs_embeds

        # `image_features` could be none, add fake tensor to make recompute happy
        fake_tensor = paddle.zeros([])
        fake_tensor.stop_gradient = False

        if self.use_full_recompute and self.training:
            inputs_embeds = recompute(
                fwd,
                image_features,
                fake_tensor,
                offload_indices=[0, 1] if self.offload_resamler else [],
            )
        else:
            inputs_embeds = fwd(image_features, fake_tensor)

        # modify video token type to image token type for expert gating
        token_type_ids[token_type_ids == TokenType.video] = TokenType.image
        ret = (token_type_ids, inputs_embeds)
        if position_ids is not None:
            ret += (position_ids,)
        if inbatch_pack_offset is not None:
            ret += (inbatch_pack_offset,)
        return ret


class ErnieDecoderLayerPipe(ErnieMoEDecoderLayer):
    """_summary_

    Args:
        ErnieDecoderLayer (_type_): _description_
    """

    def __init__(self, config, layer_idx, use_full_recompute=False):
        """
        Args:
        - config (ConfigProto): 配置文件对象，包含模型的配置信息。
        - layer_idx (int): 当前层索引号。
        - use_full_recompute (bool, optional): 是否使用全新的计算方式进行权重更新，默认为 False。

        Returns:
        - None: 该方法没有返回值。
        """
        super().__init__(config, layer_idx)
        assert not config.moe_with_send_router_loss
        self.layer_idx = layer_idx
        self.use_full_recompute = use_full_recompute
        self.use_mem_eff_attn = config.use_mem_eff_attn
        self.sequence_parallel = config.sequence_parallel
        self.rope_3d = config.rope_3d

    def forward(self, args):
        """
        Args:
            args (tuple or Tensor): The input arguments to the layer.

        Returns:
            Union[Tuple, Tensor]: A tuple of outputs from the layer.
                If `use_moe` is set to False, a single Tensor will be returned instead.

        """
        # assert len(args) == 2, len(args)
        if len(args) == 2:
            token_type_ids, hidden_states = args
            inbatch_pack_offset = None
            position_ids = None
        elif len(args) == 3:
            if self.rope_3d:
                token_type_ids, hidden_states, position_ids = args
                inbatch_pack_offset = None
            else:
                token_type_ids, hidden_states, inbatch_pack_offset = args
                position_ids = None
                inbatch_pack_offset.stop_gradient = True
        elif len(args) == 4:
            token_type_ids, hidden_states, position_ids, inbatch_pack_offset = args

        token_type_ids = token_type_ids.clone()
        # logger.info(f'hidden_states: {hidden_states.shape}, {hidden_states.astype("float32").norm()}')
        # token_type_ids = token_type_ids.clone().detach()
        # token_type_ids.stop_gradient = True
        # hidden_states = PrintOp.apply(hidden_states.clone(), f'hidden_states@{self.layer_idx}')
        if self.training and self.use_full_recompute:
            decoderlayer_act_offload_settings = self.config.get(
                "decoderlayer_act_offload_settings", {"type": "", "value": ""}
            )
            setting_type = decoderlayer_act_offload_settings["type"]
            offload_value = decoderlayer_act_offload_settings["value"]
            offload_kwargs = {}
            if "mod" == setting_type:
                assert isinstance(offload_value, (list, tuple))
                v1, v2 = offload_value
                offload_kwargs["offload_indices"] = (
                    [0] if self.layer_idx % v1 == v2 else []
                )
            elif "layer_idxs" == setting_type:
                offload_kwargs["offload_indices"] = (
                    [0] if self.layer_idx in offload_value else []
                )

            hidden_states = recompute(
                super().forward,
                hidden_states,
                None,  # attention_mask,
                position_ids,  # position_ids,
                token_type_ids.clone(),  # token-type
                False,  # output-attention
                None,  # past-kv-cache
                False,  # use-cache
                inbatch_pack_offset,  # inbatch_pack_offset,
                False,  # output_gate_logits
                **offload_kwargs,
            )
        else:
            hidden_states = super().forward(
                hidden_states,
                None,  # attention_mask,
                position_ids,  # position_ids,
                token_type_ids.clone(),  # token-type
                False,  # output-attention
                None,  # past-kv-cache
                False,  # use-cache
                inbatch_pack_offset,  # inbatch_pack_offset,
                False,  # output_gate_logits
            )
        ret = (token_type_ids, hidden_states)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if inbatch_pack_offset is not None:
            ret += (inbatch_pack_offset.clone(),)
        return ret


class LayerNormPipe(LayerNorm):
    """LayerNormPipe"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        mark_as_sequence_parallel_parameter(self.weight)
        mark_as_sequence_parallel_parameter(self.bias)

    def forward(self, args):
        """forward"""
        token_type_ids, hidden_states, *_ = args
        hidden_states = super().forward(hidden_states)
        token_type_ids.stop_gradient = True
        return token_type_ids, hidden_states


class RMSNormPipe(RMSNorm):
    """RMSNormPipe"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        mark_as_sequence_parallel_parameter(self.weight)

    def forward(self, args):
        """forward"""
        token_type_ids, hidden_states, *_ = args
        hidden_states = super().forward(hidden_states)
        token_type_ids.stop_gradient = True
        return token_type_ids, hidden_states


def multimodal_data_provider(
    inputs,
    labels,
    image_index=2,
    split_image: Optional[List[int]] = None,
    use_async=False,
    image_fea_concated=True,
):
    """替换FakeMicroDataset 支持在prepare_inputs_func对acc个image_features进行offload

    Args:
        inputs (_type_):
        labels (_type_):
        image_index (int, optional): The index position of images in inputs
        split_image (List, optional): . Defaults to False.

    Yields:
        _type_:
    """
    hcg = get_hcg()
    pp_stages = hcg.get_pipe_parallel_world_size()
    pp_stage_id = hcg.get_stage_id()
    is_first_stage = pp_stage_id == 0
    is_last_stage = pp_stage_id == pp_stages - 1

    def check_len(list_of_ten, is_input, num_sample_per_pp_data=1):
        if not image_fea_concated and is_input:
            valid_lens = []
            for i, l in enumerate(list_of_ten):
                if isinstance(l, list):
                    valid_lens.append(
                        len(l) * num_sample_per_pp_data if i == 3 else len(l)
                    )
        else:
            valid_lens = [len(l) for l in list_of_ten if isinstance(l, list)]
        assert len(set(valid_lens)) == 1, valid_lens

    if image_fea_concated:
        check_len(inputs, is_input=True)
        check_len(labels, is_input=False)
        acc_steps = len(inputs[0])
    else:
        acc_steps = len(inputs[0])
        num_sample_per_pp_data = acc_steps // len(inputs[3])
        check_len(inputs, is_input=True, num_sample_per_pp_data=num_sample_per_pp_data)
        check_len(labels, is_input=False, num_sample_per_pp_data=num_sample_per_pp_data)

    if is_first_stage:
        labels = None
    if is_last_stage:
        inputs = None

    if not split_image:
        for micro_step in range(acc_steps):
            micro_inputs = (
                tuple(x[micro_step] if isinstance(x, list) else x for x in inputs)
                if inputs is not None
                else None
            )
            micro_labels = (
                tuple(l[micro_step] if isinstance(l, list) else l for l in labels)
                if labels is not None
                else None
            )
            yield micro_inputs, micro_labels
    else:

        def slice_image(x, start, end):
            if start == end:
                return None
            if image_fea_concated:
                return x.slice((0,), start, end).clone().cuda()
            return x.cuda()._slice(start, end)

        if image_fea_concated:
            split_offset = [
                0,
            ] + list(accumulate(split_image))
            for micro_step in range(acc_steps):
                if inputs is None:
                    micro_inputs = None
                else:
                    micro_inputs = tuple(
                        (
                            slice_image(
                                x,
                                split_offset[micro_step],
                                split_offset[micro_step + 1],
                            )
                            if i == image_index
                            else x[micro_step] if isinstance(x, list) else x
                        )
                        for i, x in enumerate(inputs)
                    )
                micro_labels = (
                    tuple(l[micro_step] if isinstance(l, list) else l for l in labels)
                    if labels is not None
                    else None
                )
                yield micro_inputs, micro_labels
        else:
            for micro_step in range(acc_steps):
                if inputs is None:
                    micro_inputs = None
                else:
                    micro_inputs = []
                    for i, x in enumerate(inputs):
                        if i == image_index:
                            pp_data_idx = micro_step // num_sample_per_pp_data
                            pp_data_idx_offset = micro_step % num_sample_per_pp_data
                            start = pp_data_idx * num_sample_per_pp_data
                            end = start + num_sample_per_pp_data
                            split_offset = [0] + list(
                                accumulate(split_image[start:end])
                            )
                            micro_inputs.append(
                                slice_image(
                                    x[pp_data_idx],
                                    split_offset[pp_data_idx_offset],
                                    split_offset[pp_data_idx_offset + 1],
                                )
                            )
                        elif isinstance(x, list):
                            micro_inputs.append(x[micro_step])
                        else:
                            micro_inputs.append(x)
                    micro_inputs = tuple(micro_inputs)

                micro_labels = (
                    tuple(l[micro_step] if isinstance(l, list) else l for l in labels)
                    if labels is not None
                    else None
                )
                yield micro_inputs, micro_labels


def exchange_pp_imgs_with_thw(
    images,
    img_thw,
    img_idx,
    recv_thw,
    recv_idx,
    cur_rank,
    src_rank_index,
    dst_rank_index,
    group,
):
    """
    根据给定的信息实际进行pp间img的数据交换并得到新的数据和相关信息
    """
    tasks = []
    with _coalescing_manager(group, tasks):
        for thw, idx in zip(img_thw, img_idx):
            if thw[src_rank_index] == cur_rank and thw[dst_rank_index] != cur_rank:
                size = thw[1] * thw[2]
                task = dist.isend(
                    images[idx : (idx + size), :],
                    group.ranks[thw[dst_rank_index]],
                    group=group,
                )
                tasks.append(task)
        new_images = []
        new_thw = []
        new_idx = [0]
        old_idx = []
        for thw, idx in zip(recv_thw, recv_idx):
            new_thw.append(thw)
            old_idx.append(idx)
            if thw[src_rank_index] != cur_rank:
                data_shape = thw[1] * thw[2]
                data = paddle.empty([data_shape, images.shape[1]], dtype=images.dtype)
                # dist.stream.recv(data, dp_group.ranks[src_rank], group=dp_group, use_calc_stream=True)
                task = dist.irecv(data, group.ranks[thw[src_rank_index]], group=group)
                tasks.append(task)
                new_images.append(data)
                new_idx.append(new_idx[-1] + data_shape)
            else:
                new_images.append(images[idx : (idx + thw[1] * thw[2]), :])
                new_idx.append(new_idx[-1] + thw[1] * thw[2])
    for task in tasks:
        task.wait()
    new_idx.pop()
    new_images = paddle.concat(new_images, axis=0)

    return new_images, new_thw, new_idx, old_idx


def get_len_and_offset(input_len, group):
    """
    获取一些长度信息和offset信息
    """
    input_len = paddle.to_tensor(input_len, dtype=paddle.int64)
    length_list = []
    dist.stream.all_gather(length_list, input_len, group=group)
    offset_list = [0]
    for length in length_list:
        offset_list.append(offset_list[-1] + length.item())
    offset_list.pop()
    return length_list, offset_list


class ErniemmMoEForCausalLMPipe(PipelinePretrainedModel, PipelineLayer):
    """支持Pipeline Parallel的ERNIE4 组网模型"""

    config_class = ErniemmMoEConfig
    _get_tensor_parallel_mappings = ErniemmMoEForCausalLM._get_tensor_parallel_mappings
    _resolve_prefix_keys = ErniemmMoEForCausalLM._resolve_prefix_keys

    def _prepare_pipeline_inputs_func(self, data: Union[List, Dict]):
        """_prepare_pipeline_inputs_func"""
        assert isinstance(data, list), type(data)
        if getattr(self.config.vision_config, "variable_resolution", False):
            assert (
                not self.balanced_image_preprocess
            ), "balanced_image_preprocess is not supported in variable_resolution"

        all_keys = [
            "images",
            "grid_thw",
            "input_ids",
            "audio_input_ids",
            "token_type_ids",
            "image_type_ids",
            "labels",
            "audio_labels",
            "position_ids",
        ]
        inputs = []
        for k in all_keys:
            temp = []
            for d in data:
                if k not in d:
                    temp.append(None)
                else:
                    temp.append(d[k])
            inputs.append(temp)

        hcg = get_hcg()
        dp_group = hcg.get_pipe_parallel_group()
        dp_worldsize = hcg.get_pipe_parallel_world_size()
        dp_src_rank = dp_group.ranks[0]
        dp_rank = hcg._get_pipe_parallel_id()
        this_rank = dist.get_rank()

        images, grid_thw, *other_inputs = inputs
        if self.pp_need_data_ranks:
            send_args = [
                grid_thw,
            ] + other_inputs
            recv_args = gather_tensors_list_in_pp_group(send_args, merge_output=False)
            if recv_args is not None:
                recv_args = list(zip(*recv_args))
                (
                    global_grid_thw,
                    ids,
                    audio_ids,
                    token_type_ids,
                    image_type_ids,
                    labels,
                    audio_labels,
                    position_ids,
                ) = [sum(args_from_all_pp, []) for args_from_all_pp in recv_args]
            else:
                # middle pp
                global_grid_thw = ids = audio_ids = token_type_ids = image_type_ids = (
                    labels
                ) = audio_labels = position_ids = None
        else:
            (
                ids,
                audio_ids,
                token_type_ids,
                image_type_ids,
                labels,
                audio_labels,
                position_ids,
            ) = other_inputs
            global_grid_thw = grid_thw
        if ids is not None:  # pp0, pp, -1
            token_type_ids = [t.astype("int32") for t in token_type_ids]
            token_type_ids_shifted = [t[:, 1:] for t in token_type_ids]
        else:
            ids = audio_ids = token_type_ids = image_type_ids = (
                token_type_ids_shifted
            ) = labels = audio_labels = None

        if self.vision_model is None:
            images = None  # 没有 vit，没法抽取视觉 feature # image 不算了
            global_grid_thw = None
            return multimodal_data_provider(
                (
                    token_type_ids,
                    ids,
                    images,
                    image_type_ids,
                    global_grid_thw,
                    position_ids,
                    audio_ids,
                ),
                (token_type_ids_shifted, labels, audio_labels),
                image_index=2,
            )

        if (self.pp_need_data_ranks and dp_rank not in self.pp_need_data_ranks) or (
            not self.pp_need_data_ranks and dp_rank != 0
        ):
            images = []

        image_len_before_concat = paddle.to_tensor(
            [len(n) if n is not None else 0 for i, n in enumerate(images)],
            dtype="int32",
        )
        images_is_all_none = paddle.to_tensor(
            all([i is None for i in images]), dtype="int32"
        )
        dist.broadcast(images_is_all_none, src=dp_src_rank, group=dp_group)
        if images_is_all_none.item():
            images = None  # 没有 images 没法抽取视觉 feature # image 不算了
            global_grid_thw = None
            return multimodal_data_provider(
                (
                    token_type_ids,
                    ids,
                    images,
                    image_type_ids,
                    global_grid_thw,
                    position_ids,
                    audio_ids,
                ),
                (token_type_ids_shifted, labels, audio_labels),
                image_index=2,
            )

        images = [i for i in images if i is not None]
        images = paddle.concat(images) if len(images) else None  # list -> tensor
        grid_thw = [i for i in grid_thw if i is not None]
        grid_thw = paddle.concat(grid_thw) if len(grid_thw) else None  # list -> tensor

        # 开始进行pp数据均衡
        pp_data_balance = getattr(self.vision_model, "pp_data_balance", False)

        if (
            self.balanced_image_preprocess
            or self.config.offload_pp_data_chunk_size > 0
            or pp_data_balance
        ):
            # 提前做alltoall是为了初始化batch send recv的通信组，提前建联
            if not hasattr(get_hcg(), "pp_sd_group"):
                # pp_sd_group = create_pp_sd_group()
                pp_sd_group = get_hcg().get_pipe_parallel_group()
                # alltoall to make p2p eager
                fake_data = paddle.ones([pp_sd_group.nranks, 1])
                fake_out = paddle.empty([pp_sd_group.nranks, 1])
                dist.alltoall(fake_out, fake_data, pp_sd_group)
                setattr(get_hcg(), "pp_sd_group", pp_sd_group)

        if pp_data_balance:
            # step1: 获取一些信息，如seqlen、grid_thw，用于当前排序以及后续恢复
            seq_list, seq_idx_list = get_len_and_offset(images.shape[0], dp_group)
            self.vision_model.seq_list = seq_idx_list

            # 把grid_thw 按t展开，否则这里会有0以及t>1的情况，不符合后面的排序代码逻辑
            grid_thw = grid_thw[grid_thw > 0].reshape([-1, 3])
            grid_thw = F.pad(
                paddle.repeat_interleave(grid_thw[:, 1:], grid_thw[:, 0], 0),
                [0, 0, 1, 0],
                value=1,
            )

            # 获取每张img的offset
            img_idx = paddle.cumsum(grid_thw[:, 1] * grid_thw[:, 2])
            thwsum = img_idx[-1]
            assert (
                thwsum == images.shape[0]
            ), f"thwsum {thwsum}, images.shape {images.shape}"
            img_idx = img_idx[:-1]
            img_idx = F.pad(img_idx, [1, 0], value=0)
            assert (
                img_idx.shape[0] == grid_thw.shape[0]
            ), f"img_idx.shape {img_idx.shape} , grid_thw.shape {grid_thw.shape}"

            # 给thw添加rank信息
            rank_column = paddle.full(
                shape=[grid_thw.shape[0], 1], fill_value=dp_rank, dtype=grid_thw.dtype
            )
            gridthw_withid = paddle.concat([grid_thw, rank_column], axis=-1)

            # 获得所有pp的thw和img offset信息
            thw_len = paddle.to_tensor(gridthw_withid.shape[0], dtype=paddle.int32)
            thw_len_list = []
            dist.stream.all_gather(thw_len_list, thw_len, group=dp_group)
            gathered_gridthw_withid = all_gather_varlen(
                gridthw_withid, thw_len_list, dp_group
            )
            gathered_img_idx = all_gather_varlen(img_idx, thw_len_list, dp_group)
            gridthw_withid = gathered_gridthw_withid
            img_idx = gathered_img_idx

            # pp间按照图片大小排序便于后续分桶
            gridthw_withid = np.array(gridthw_withid, dtype=np.int64)
            img_idx = np.array(img_idx, dtype=np.int64)
            # products = gridthw_withid[:, 1] * gridthw_withid[:, 2]
            # sorted_indices = np.argsort(products)
            sorted_indices = sorted(
                range(gridthw_withid.shape[0]),
                key=lambda i: gridthw_withid[i, 1] * gridthw_withid[i, 2],
            )
            sorted_thw = gridthw_withid[sorted_indices]
            sorted_idx = img_idx[sorted_indices]

            indices = np.arange(sorted_thw.shape[0]) % dp_worldsize
            indices = np.expand_dims(indices, axis=-1)
            sorted_thw = np.concatenate((sorted_thw, indices), axis=-1)
            sorted_thw = paddle.to_tensor(sorted_thw, dtype=gridthw_withid.dtype)
            sorted_idx = paddle.to_tensor(sorted_idx, dtype=img_idx.dtype)

            assert sorted_thw.shape[1] == 5, f"{sorted_thw.shape}"
            self.vision_model.sorted_thw = sorted_thw.clone()
            self.vision_model.sorted_idx = sorted_idx.clone()
            # 这里进行实际pp间数据交换
            new_images, new_thw, new_idx, old_idx = exchange_pp_imgs_with_thw(
                images,
                sorted_thw[sorted_thw[:, -2] == dp_rank],
                sorted_idx[sorted_thw[:, -2] == dp_rank],
                sorted_thw[sorted_thw[:, -1] == dp_rank],
                sorted_idx[sorted_thw[:, -1] == dp_rank],
                dp_rank,
                src_rank_index=-2,
                dst_rank_index=-1,
                group=dp_group,
            )

            # 用于输入vit的数据
            images = new_images
            # 5column保留了原始img原来的rank和排序后rank，需要保留后续出骨干网恢复使用
            grid_thw_5column = paddle.stack(new_thw, axis=0)
            new_idxes = paddle.to_tensor(new_idx, dtype=img_idx.dtype)
            old_idxes = paddle.to_tensor(old_idx, dtype=img_idx.dtype)
            grid_thw = grid_thw_5column[:, :-2]

        # I dont know why can not release GPU memory, so I using `_clear_data` to clear underlaying GPU memory
        if self.config.offload_pp_data_chunk_size > 0:
            for img in inputs[0]:
                if img is not None:
                    img._clear_data()
        image_len_before_concat_gathered = gather_varlen(
            image_len_before_concat, dst=dp_src_rank, group=dp_group
        )
        # logger.info(f"image_len_before_concat_gathered:{image_len_before_concat_gathered}")
        # image_len_before_concat_gathered = paddle.concat(image_len_before_concat_gathered, 0)

        def create_pp_sd_group():
            cur_rank = dist.get_rank()
            cur_world_size = dist.get_world_size()
            hcg = get_hcg()
            pp_world_size = hcg.get_pipe_parallel_world_size()
            sd_world_size = hcg.get_sharding_parallel_world_size()
            pp_sd_world_size = pp_world_size * sd_world_size
            tp_world_size = hcg.get_model_parallel_world_size()
            list_of_ranks = []
            for i in range(tp_world_size):
                ranks = np.arange(i, cur_world_size, tp_world_size)
                list_of_ranks.append(ranks)

            cur_group = None
            for i, ranks in enumerate(list_of_ranks):
                group = dist.new_group(ranks)
                if cur_rank % tp_world_size == i:
                    cur_group = group
            # alltoall to make p2p eager
            fake_data = paddle.ones([cur_group.nranks, 1])
            fake_out = paddle.empty([cur_group.nranks, 1])
            dist.alltoall(fake_out, fake_data, cur_group)
            return cur_group

        # pp内数据切片
        @partial(
            shard_data_in_pp_group,
            fwd_batch_size=getattr(self.config.vision_config, "vit_first_fwd_bsz", 128),
            input_is_parallel=len(self.pp_need_data_ranks) > 1,
            is_balanced=self.balanced_image_preprocess,
            offload_pp_data_chunk_size=self.config.offload_pp_data_chunk_size,
        )
        def fwd_image(images, grid_thw):
            # logger.info(f"# image inside shard : {images.shape}")
            with profile("extract_image_fea"):
                if self.image_preprocess is not None:
                    assert images.dtype == paddle.uint8, images.dtype
                    images = self.image_preprocess.rescale_factor * images.astype(
                        "float32"
                    )
                    images = (
                        images - self.image_preprocess.image_mean_tensor
                    ) / self.image_preprocess.image_std_tensor
                    images = images.astype("bfloat16")
                else:
                    assert images.dtype == paddle.bfloat16, images.dtype
                # logger.info(f"进网-{images}-{grid_thw}")
                image_fea = self.vision_model.extract_feature(images, grid_thw)
                if self.config.tensor_parallel_degree > 1:
                    if getattr(self.config.vision_config, "variable_resolution", False):
                        S, C = image_fea.shape
                        # scatterOp切fea时候 + 4合一，提前把4个token的feature并到一起。
                        image_fea = image_fea.reshape(
                            [-1, C * self.config.spatial_conv_size**2]
                        )
                    image_fea = ScatterOp.apply(image_fea, axis=-1)  # mp 切 Fea
                    if getattr(self.config.vision_config, "variable_resolution", False):
                        image_fea = image_fea.reshape([S, -1])
                # logger.info(f"# image-fea inside shard : {image_fea.shape}")

            return image_fea

        with profile("extract_image_fea_w_gather"):
            if self.balanced_image_preprocess:
                # broadcast image shape if needed
                if len(self.pp_need_data_ranks) < dp_worldsize:
                    if self.balanced_image_shape is None:
                        pp_sd_group = get_hcg().pp_sd_group
                        src_rank = pp_sd_group.ranks[0]
                        this_rank = dist.get_rank()
                        if src_rank == this_rank:
                            assert images.ndim == 4, images.shape
                            full_image_shape = (
                                paddle.shape(images).cuda().astype("int32")
                            )
                        else:
                            full_image_shape = paddle.empty([4], dtype="int32")
                        dist.broadcast(full_image_shape, src_rank, group=pp_sd_group)
                        full_image_shape = full_image_shape.tolist()
                        self.balanced_image_shape = full_image_shape
                    if dp_rank not in self.pp_need_data_ranks:
                        assert (
                            images is None
                        ), "pp rank exceed partial pp_need_data must be None"
                        full_image_shape = self.balanced_image_shape
                        full_image_shape[0] = 0
                        images = paddle.empty(full_image_shape, dtype=paddle.uint8)
                    else:
                        # check [c, h, w] must be equal
                        assert (
                            images.shape[1:] == self.balanced_image_shape[1:]
                        ), "image shape is not equal to the previous cache shape"

            image_fea = fwd_image(images, grid_thw)
            # if image_fea is not None:
            #     logger.info(f"# image-fea outside shard : {image_fea.shape}")

        if pp_data_balance:
            # gather后的img_fea还原
            # 注意这里从fwd_image出来后实际上已经进行了pp间的fea gather工作
            new_seq_list, new_seq_idx_list = get_len_and_offset(
                images.shape[0], dp_group
            )
            new_thw_len_list, new_thw_idx_list = get_len_and_offset(
                grid_thw_5column.shape[0], dp_group
            )

            new_gathered_gridthw_withid = all_gather_varlen(
                grid_thw_5column, new_thw_len_list, dp_group
            )
            new_gathered_img_idx = all_gather_varlen(
                new_idxes, new_thw_len_list, dp_group
            )
            new_gathered_old_idx = all_gather_varlen(
                old_idxes, new_thw_len_list, dp_group
            )
            assert (
                new_gathered_gridthw_withid.shape[0] == new_gathered_img_idx.shape[0]
            ), f"{new_gathered_gridthw_withid.shape[0]} != {new_gathered_img_idx.shape[0]}"
            # gather 每个pp 的img seq
            if image_fea is not None:
                new_gathered_gridthw_withid = np.array(
                    new_gathered_gridthw_withid, dtype=np.int64
                )
                new_gathered_img_idx = np.array(new_gathered_img_idx, dtype=np.int64)
                new_gathered_old_idx = np.array(new_gathered_old_idx, dtype=np.int64)
                new_seq_idx_list = np.array(new_seq_idx_list, dtype=np.int64)

                new_fea = []
                for rank in range(dp_group.nranks):
                    # 获取rank中的thw和offset，这些都是之前pp排序后的新结果，也拿到之前排序前img的offset
                    cur_thw = new_gathered_gridthw_withid[
                        new_gathered_gridthw_withid[:, -2] == rank
                    ]
                    cur_idx = new_gathered_img_idx[
                        new_gathered_gridthw_withid[:, -2] == rank
                    ]
                    old_idx = new_gathered_old_idx[
                        new_gathered_gridthw_withid[:, -2] == rank
                    ]

                    sorted_indices = np.argsort(old_idx)
                    sorted_fea_idx = cur_idx[sorted_indices]
                    sorted_fea_thw = cur_thw[sorted_indices]

                    # 依据原来offset排序，恢复fea的顺序
                    start_offset = (
                        new_seq_idx_list[sorted_fea_thw[:, -1]] + sorted_fea_idx
                    )
                    end_offset = (
                        new_seq_idx_list[sorted_fea_thw[:, -1]]
                        + sorted_fea_idx
                        + sorted_fea_thw[:, 1] * sorted_fea_thw[:, 2]
                    )
                    index_list = [
                        np.arange(start_offset[i], end_offset[i])
                        for i in range(len(start_offset))
                    ]
                    index_list = paddle.to_tensor(
                        np.concatenate(index_list, axis=-1), dtype=paddle.int64
                    )
                    fea = paddle.gather(image_fea, index_list)
                    new_fea.append(fea)
                new_fea = paddle.concat(new_fea, axis=0)
                image_fea = new_fea

        if image_fea is not None:  # pp 0 or LM batch
            return multimodal_data_provider(
                (
                    token_type_ids,
                    ids,
                    image_fea,
                    image_type_ids,
                    global_grid_thw,
                    position_ids,
                    audio_ids,
                ),
                (token_type_ids_shifted, labels, audio_labels),
                image_index=2,
                split_image=image_len_before_concat_gathered.tolist(),
                image_fea_concated=isinstance(image_fea, paddle.Tensor),
            )
        image_fea = None
        return multimodal_data_provider(
            (
                token_type_ids,
                ids,
                image_fea,
                image_type_ids,
                global_grid_thw,
                position_ids,
                audio_ids,
            ),
            (token_type_ids_shifted, labels, audio_labels),
            image_index=2,
        )

    def get_loss_fn(self, config):
        """get_loss_fn"""
        return ErniePretrainingCriterionPipe(config)  # 默认Ignored Index == 0, 并未传参

    def __init__(self, config, use_recompute=False):
        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(
            f"change initializer-range from {config.initializer_range} to {new_initializer_range}"
        )
        config.initializer_range = new_initializer_range
        if config.moe_group in {"mp", "model", "tp", "mpdp"}:
            assert config.sequence_parallel
            logger.info(
                f"disable FFN tensor model parallel, moe-group={config.moe_group}"
            )
            config.disable_ffn_model_parallel = True

        config.moe_group = _parse_moe_group(config.moe_group)
        config.moe_world_size = dist.get_world_size(config.moe_group)
        if config.moe_world_size < 0:
            config.moe_world_size = 1
        config.moe_rank = dist.get_rank(config.moe_group)
        config.moe_with_send_router_loss = False
        hcg = get_hcg()

        self.config = config
        self.image_preprocess = None
        self.pp_need_data_ranks = []  # default to all need data
        self.balanced_image_preprocess = (
            config.balanced_image_preprocess
            if hasattr(config, "balanced_image_preprocess")
            else False
        )
        self.balanced_image_shape = None

        tensor_parallel_degree = max(hcg.get_model_parallel_world_size(), 1)
        tensor_parallel_rank = max(hcg.get_model_parallel_rank(), 0)
        logger.info(f"using vpp={config.virtual_pp_degree}")
        if config.sequence_parallel:
            logger.info(f"using sequence_parallel, input seqlen={config.seqlen}")
            assert config.seqlen is not None
            assert (
                config.tensor_parallel_degree > 1
            ), f"sequence-parallel needs mp>1, got mp={config.tensor_parallel_degree}"
        config.tensor_parallel_degree = tensor_parallel_degree
        config.tensor_parallel_rank = tensor_parallel_rank

        if isinstance(config.vision_config, EVAVisionTransformerConfig):
            logger.info(f"update vision config {config.use_recompute_attn_vision}")
            config.vision_config.use_recompute_attn = config.use_recompute_attn_vision
            config.vision_config.tensor_parallel_degree = config.tensor_parallel_degree
            config.vision_config.tensor_parallel_rank = config.tensor_parallel_rank
        elif isinstance(config.vision_config, DFNRopeVisionTransformerConfig):
            logger.info("variable resolution vision model")
            config.vision_config.variable_resolution = True

        # DON'T call PipelinePretrainedModel `__ini__` due to mro issue
        PipelinePretrainedModel.init(self, config=config)

        # if config.inception_config is not None:
        #     self.add_sequential_layer(LayerDesc(
        #         InceptionPipe,
        #         config=config,
        #     ), "vision_model")
        # else:
        #     self.add_sequential_layer(LayerDesc(
        #             EVAVisionTransformerPipe,
        #             config=config,
        #     ), "vision_model")
        # self.add_sequential_layer(
        #     LayerDesc(ResamplerPipe, config=config, use_full_recompute=config.use_recompute), "resampler_model"
        # )

        insert_empty_layer = config.insert_empty_layer
        if len(insert_empty_layer) > 0:
            assert (
                min(insert_empty_layer) >= 0
            ), "cannot insert empty layer as first layer of the model"
            assert (
                max(insert_empty_layer) < config.num_hidden_layers
            ), "empty layers location exceed the num layers"
        logger.info(f"use insert_empty_layer: {insert_empty_layer}")

        if config.tie_word_embeddings:
            self.add_sequential_layer(
                SharedLayerDesc(
                    key="embed_weight_share",
                    layer_func=ErniemmEmbeddingPipe,
                    shared_weight_attr="embedding_weight",
                    use_full_recompute=config.use_recompute,
                    config=config,
                ),
                "ernie",
            )
        else:
            self.add_sequential_layer(
                LayerDesc(
                    ErniemmEmbeddingPipe,
                    config=config,
                    use_full_recompute=config.use_recompute,
                ),
                "ernie",
            )

        no_recompute_layers = get_pp_vp_split_layers(config)

        def _need_full_recompute(layer_idx):
            return layer_idx not in no_recompute_layers and config.use_recompute

        num_empty_layers = (
            config.remove_tail_layer if isinstance(config.remove_tail_layer, int) else 1
        )

        for i in range(config.num_hidden_layers - num_empty_layers):
            self.add_sequential_layer(
                LayerDesc(
                    ErnieDecoderLayerPipe,
                    config=create_skip_config_for_refined_recompute(i, config),
                    layer_idx=i,
                    use_full_recompute=_need_full_recompute(i),
                ),
                f"ernie.layers.{i}",
            )
            if i in insert_empty_layer:
                self.add_sequential_layer(
                    LayerDesc(
                        EmptyLayer,
                    ),
                    f"empty.layers.{i}",
                )

        if config.remove_tail_layer:
            for n in range(num_empty_layers):
                self.add_sequential_layer(
                    LayerDesc(
                        EmptyLayer,
                    ),
                    f"empty.layers.{n}",
                )
        else:
            for n in range(num_empty_layers):
                self.add_sequential_layer(
                    LayerDesc(
                        ErnieDecoderLayerPipe,
                        config=create_skip_config_for_refined_recompute(i, config),
                        layer_idx=i,
                        use_full_recompute=_need_full_recompute(i),
                    ),
                    f"ernie.layers.{n + config.num_hidden_layers - num_empty_layers}",
                )

        i = config.num_hidden_layers
        if i in insert_empty_layer:
            self.add_sequential_layer(
                LayerDesc(
                    EmptyLayer,
                ),
                f"empty.layers.{i}",
            )

        self.add_sequential_layer(
            LayerDesc(
                RMSNormPipe if config.use_rmsnorm else LayerNormPipe, config=config
            ),
            "ernie.norm",
        )

        if config.tie_word_embeddings:
            self.add_sequential_layer(
                SharedLayerDesc(
                    key="embed_weight_share",
                    layer_func=ErnieMoELMHeadPipe,
                    shared_weight_attr="embedding_weight",
                    config=config,
                ),
                "lm_head",
            )
        else:
            self.add_sequential_layer(
                LayerDesc(ErnieMoELMHeadPipe, config=config), "lm_head"
            )

        recompute_interval = 0

        if self.config.pp_first_stage_layers:  # 手动控制 pp-stage1 的 layer 数目.
            assert self.config.pp_first_stage_layers >= 2
            _num_layers = len(self.get_sequential_layers())
            _num_stages = get_hcg().topology().get_dim_size("pipe")
            part_size = (_num_layers - self.config.pp_first_stage_layers) // (
                _num_stages - 1
            )
            seg_method = [0, self.config.pp_first_stage_layers] + [
                part_size for i in range(_num_stages - 1)
            ]
            seg_method = list(accumulate(seg_method))
            seg_method[-1] = _num_layers
        else:
            seg_method = "layer:ErnieDecoderLayer|EmptyLayer"
            if (
                config.num_hidden_layers % get_hcg().topology().get_dim_size("pipe")
                != 0
            ):
                seg_method = "uniform"
        logger.info(
            f"using recompute_interval={recompute_interval}, seg_method={seg_method}"
        )

        PipelineLayer.__init__(
            self,
            layers=self.get_sequential_layers(),
            loss_fn=self.get_loss_fn(config),
            topology=get_hcg().topology(),
            seg_method=seg_method,
            recompute_interval=recompute_interval,
            recompute_ctx={
                "mp_group": get_hcg().get_model_parallel_group(),
                "offload": False,
                "partition": False,  # TODO：看看怎么 Partition recompute checkpoint。
            },
            num_virtual_pipeline_stages=config.virtual_pp_degree,
        )
        self.vision_model = None
        self._modality_param_mapping = None

    def add_vision_model(
        self,
        encoder: nn.Layer,
    ):
        """add_vision_model"""
        self.vision_model = encoder
        # 同时set_color 在sharding wrap model 之前
        self._set_modality_param_mapping()

    def add_image_preprocess(self, preprocess):
        logger.info("image preprocess is set")
        self.image_preprocess = preprocess

    def set_pp_need_data_degree(self, p):
        """刷新need-data 的 pp rank-id（group 内的相对 rank)"""
        if p == 1:
            logger.warning("you are trying to disable pp-need-data")
            return
        pp_world_size = get_hcg().get_pipe_parallel_world_size()
        no_need_data_range = list(range(p - 1, pp_world_size - 1))
        ranks = [i for i in range(pp_world_size) if i not in no_need_data_range]
        logger.info(f"set `pp_need_data_ranks` to {p}, {ranks}")
        self.pp_need_data_ranks = ranks

    def _set_modality_param_mapping(self, use_stop_grad=True):
        # if self._pipeline_name_mapping is None:
        self._set_pipeline_name_mapping()
        assert (
            len(self._pipeline_name_mapping) > 0
        ), "The pipeline stage must have parameters!"
        # pp_to_single_mapping = {v: k for k, v in self._pipeline_name_mapping.items()}
        pp_to_single_mapping = self._pp_to_single_mapping
        lm_pattern = get_backbone_lm_param_regex(self.config)
        self._modality_param_mapping = defaultdict(lambda: [])
        for name, param in self.named_parameters():
            name_split = name.split(".")
            if not name_split[0].isdigit():
                pipe = None
            else:
                if self.config.virtual_pp_degree > 1:
                    pipe = self._sub_layers[name_split[0]]._sub_layers[name_split[1]]
                else:
                    pipe = self._sub_layers[name_split[0]]
            expert_type = getattr(param, "expert_type", None)
            if not use_stop_grad:  # use hook
                monkey_patch_param_hook(param)
            name = pp_to_single_mapping[name]
            if "vision_model" in name:
                self._modality_param_mapping["vit"].append((name, param))
                setattr(param, "color", "vit")
            elif expert_type == "expert_type_3":
                self._modality_param_mapping["audio"].append((name, param))
                setattr(param, "color", "audio")
            elif lm_pattern.match(name) or expert_type == "expert_type_0":
                self._modality_param_mapping["lm"].append((name, param))
                setattr(param, "color", "lm")
            else:
                self._modality_param_mapping["mm"].append((name, param))
                setattr(param, "color", "mm")
        debug_msg = {
            k: [i[0] for i in v] for k, v in self._modality_param_mapping.items()
        }
        logger.info(
            f"modality_param_mapping: {json.dumps(debug_msg, ensure_ascii=False, indent=2)}"
        )

    def update_params_stat(self, param_group, stop_gradient):
        """freeze mm"""
        assert param_group in (
            "lm",
            "mm",
            "audio",
            "vit",
        ), "param_group must be in ('lm', 'mm', 'audio', 'vit')"
        if self._modality_param_mapping is None:
            self._set_modality_param_mapping()
        for name, param in self._modality_param_mapping.get(param_group, []):
            # logger.info(f"{param_group}: {name} set_stop_gradient to {stop_gradient}")
            param.stop_gradient = stop_gradient

    def freeze_vision(self):
        """freeze_vision"""
        if self._modality_param_mapping is None:
            self._set_modality_param_mapping()
        for name, param in self._modality_param_mapping.get("vit", []):
            logger.info("Freezing vision parameter: {}".format(name))
            param.stop_gradient = True
        self.vision_model.config.freeze_vision = True

    # Initialize weights and apply final processing
    def _post_init(self, original_init, *args, **kwargs):
        """_post_init"""
        super()._post_init(self, original_init, *args, **kwargs)
        with paddle.no_grad():
            if self.config.virtual_pp_degree > 1:
                pipe_layers = (
                    self._sub_layers[i]._sub_layers[j]
                    for i in self._sub_layers
                    for j in self._sub_layers[i]._sub_layers
                )
            else:
                pipe_layers = (self._sub_layers[i] for i in self._sub_layers)

            for i, layer in enumerate(pipe_layers):
                if isinstance(layer, ErniemmEmbeddingPipe):
                    if getattr(layer, "mm_embed_tokens", None) is not None:
                        layer.mm_embed_tokens.weight.expert_type = "expert_type_1"
                        if getattr(layer.mm_embed_tokens, "bias", None) is not None:
                            layer.mm_embed_tokens.bias.expert_type = "expert_type_1"
                    if getattr(layer, "audio_embed_tokens", None) is not None:
                        layer.audio_embed_tokens.weight.expert_type = "expert_type_3"
                        if getattr(layer.audio_embed_tokens, "bias", None) is not None:
                            layer.audio_embed_tokens.bias.expert_type = "expert_type_3"
                    if getattr(layer, "audio_after_norm", None) is not None:
                        layer.audio_after_norm.weight.expert_type = "expert_type_3"
                        if getattr(layer.audio_after_norm, "bias", None) is not None:
                            layer.audio_after_norm.bias.expert_type = "expert_type_3"

                if isinstance(layer, ErnieMoELMHeadPipe):
                    if getattr(layer, "mm_head", None) is not None:
                        layer.mm_head.weight.expert_type = "expert_type_1"
                        if getattr(layer.mm_head, "bias", None) is not None:
                            layer.mm_head.bias.expert_type = "expert_type_1"
                    if getattr(layer, "audio_out_module", None) is not None:
                        layer.audio_out_module.weight.expert_type = "expert_type_3"
                        if getattr(layer.audio_out_module, "bias", None) is not None:
                            layer.audio_out_module.bias.expert_type = "expert_type_3"

                if isinstance(layer, ErnieDecoderLayerPipe):
                    layer_id = layer.layer_idx  # skip_vocab
                    factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
                    logger.info(
                        f"using post init div: layer[{layer_id}].factor={factor}"
                    )
                    layer.self_attn.o_proj.weight.scale_(factor)
                    if isinstance(
                        layer.mlp,
                        (
                            DeepEPMOELayer,
                            DeepEPDropTokenMOELayer,
                            MOELayer,
                            MOEAllGatherLayer,
                            MOEAllGatherLayerV2,
                            MOELayerSizeAll2All,
                        ),
                    ):
                        for e in layer.mlp.experts:
                            if isinstance(e, ErnieMLP):
                                e.down_proj.weight.scale_(factor)
                        if getattr(layer.mlp, "dense_experts", None) and isinstance(
                            layer.mlp.dense_experts, ErnieMLP
                        ):
                            layer.mlp.dense_experts.down_proj.weight.scale_(factor)
                    else:
                        layer.mlp.down_proj.weight.scale_(factor)

        if not self.config.disable_pipeline_warmup:
            self.pipeline_warmup()

    def pipeline_warmup(self):
        """pipeline_warmup"""
        # warmup moe layer for pp
        try:
            input = None
            hcg = get_hcg()
            mp_size = hcg.get_model_parallel_world_size()
            with paddle.no_grad():
                ori_state = self.training
                self.eval()
                mbs = max(self.config.micro_batch_size, 1)
                token_type_ids = paddle.zeros([mbs, self.config.seqlen + 1]).astype(
                    "int32"
                )
                hidden_states = paddle.randn(
                    [mbs * self.config.seqlen // mp_size, self.config.hidden_size]
                )
                logger.info(f"mm-moe pp warmup w/ shape: {hidden_states.shape}")
                input_ids = paddle.zeros([mbs, self.config.max_sequence_length]).astype(
                    "int32"
                )
                input = (token_type_ids, hidden_states)
                if self._num_virtual_pipeline_stages <= 1:
                    chunk_id = None
                    if hcg.get_stage_id() == 0:
                        input = (token_type_ids, input_ids, None, None)
                else:
                    if hcg.get_stage_id() == 0:
                        chunk_id = 1
                    else:
                        chunk_id = 0
                output = self.forward(input, chunk_id=chunk_id)
            paddle.device.synchronize()
            if ori_state:
                self.train()
            logger.info("warmup moe layer for pp successfully")
            # NOTE(shenliang03): warmup阶段需要重置训练log，避免污染
        except Exception as e:
            logger.info(f"failed to warmup moe layer for pp: {e}")
        finally:
            if isinstance(global_training_logs, dict):
                global_training_logs.clear()
            else:
                global_training_logs.reset()

    def state_dict(self, *args, **kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        state_dict = super().state_dict(*args, **kwargs)
        # moe state dict 转换成全局id.
        return moe_statedict_local_id_to_global(state_dict, self.config)

    def set_state_dict(self, state_dict, *args, **kwargs):
        """
        mm_moe PP模型 和 mm_moe模型具备一样的 state-dict
        """
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()
        assert (
            len(self._pipeline_name_mapping) > 0
        ), "The pipeline stage must have parameters!"

        layer_idxs = []
        if self.config.virtual_pp_degree == 1:
            _layers = iter(self.run_function)
        else:
            _layers = (cc for c in self._model_chunks for cc in c.run_function)

        for layer in _layers:
            if isinstance(layer, ErnieDecoderLayerPipe):
                layer_idxs.append(layer.layer_idx)
        logger.info(f"this pipeline stage has ErnieDecoderLayers: {layer_idxs}")
        state_dict = moe_statedict_cherry_pick(state_dict, self.config)
        # 多模不走纯文部分自动upcycling逻辑。
        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if k not in self._pipeline_name_mapping:
                if f"ernie.{k}" in self._pipeline_name_mapping:
                    state_dict[self._pipeline_name_mapping[f"ernie.{k}"]] = v
                continue
            state_dict[self._pipeline_name_mapping[k]] = v
        res = super().set_state_dict(state_dict, *args, **kwargs)
        logger.info(f"ERNIE-MM-MOE-PP - set-state_dict- {res}")
        return res
