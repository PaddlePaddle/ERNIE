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

import paddle
import numpy as np
import h5py
import re
import paddle.distributed as dist
from glob import glob
from paddleformers.transformers.model_utils import _load_state_dict_into_model
from .logging import logger


def load_model(model, path, dtype=None):
    sd = paddle.load(str(path), return_numpy=True)
    for k, v in sd.items():  # TODO: should not force cast
        if k.endswith("position_ids"):
            sd[k] = v.astype(np.int64)
        if dtype and str(v.dtype) != dtype:
            sd[k] = v.astype(dtype)
    missing, unexpected = model.set_state_dict(sd)
    logger.warning(
        f"loading ckpt {path} done, unexpected={unexpected}, missing={missing}"
    )


def load_model_from_super_merge(model, h5dir):
    """_summary_

    Args:
        h5dir (_type_): _description_
        keys (_type_): _description_
        mp_rank (_type_): _description_

    Returns:
        _type_: _description_
    """
    h5 = {}
    out_sd = {}
    mp_rank = model.config.tensor_parallel_rank
    mp_degree = model.config.tensor_parallel_degree
    if mp_degree <= 1:
        mp_rank = None

    def process_key(h5, k, load_k, dtype):
        if load_k in h5.keys():
            logger.info(f"load {k} from h5, remap k: {load_k}, file:{h5}")
            return k, h5[load_k][:]
        logger.warning(f"key not found -- {k}, remap k:{load_k}, file:{h5}")
        return k, None

    futures = []
    pat = re.compile(r"ernie.layers\.(\d+)")
    expert_pat = re.compile(r"ernie.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.*)")
    for k, tensor in model.state_dict().items():
        if "vision" in k:
            continue
        load_k = k
        match = pat.match(k)
        expert_match = expert_pat.match(k)
        if expert_match:
            layer_id = int(expert_match.group(1))
            expert_id = int(expert_match.group(2))
        elif match:
            layer_id = int(match.group(1))
            expert_id = None
        else:
            layer_id = 0
            expert_id = None
        h5file = f"{h5dir}/mp_{mp_rank}.layer.{layer_id}.h5"
        moe_world_size = len(glob(f"{h5dir}/mp_{mp_rank}_moe*.layer.{layer_id}.h5"))
        if moe_world_size >= 1:  # h5 is dp-moe
            if expert_id is not None:
                if (
                    dist.get_world_size(model.config.moe_group) <= 1
                ):  # dummy load dp-moe
                    local_expert_id = expert_id % moe_world_size
                    moe_rank_to_load = expert_id // moe_world_size
                else:  # dp load dp-moe
                    local_expert_id = expert_id
                    moe_rank_to_load = dist.get_rank(model.config.moe_group)
                load_k = f"ernie.layers.{expert_match.group(1)}.mlp.experts.{local_expert_id}.{expert_match.group(3)}"
            else:
                moe_rank_to_load = 0
            # dp moe 下 ckpt 一定有 moe_* 中缀
            h5file = f"{h5dir}/mp_{mp_rank}_moe_{moe_rank_to_load}.layer.{layer_id}.h5"
        if h5file not in h5:
            h5[h5file] = h5py.File(h5file)
        futures.append(process_key(h5[h5file], k, load_k, tensor.dtype))
    for future in futures:
        k, tensor = future
        if tensor is not None:
            out_sd[k] = paddle.Tensor(tensor, zero_copy=True, place=paddle.CPUPlace())
    logger.info("load super merge shard sd DONE")
    res = _load_state_dict_into_model(model, out_sd, "")
    logger.info(f"set_state_dict res: {res}")
