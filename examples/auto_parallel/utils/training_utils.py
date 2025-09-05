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


import os
import yaml
import logging
import argparse
from omegaconf import OmegaConf
import paddle.distributed as dist
from paddle.distributed import fleet


logger = logging.getLogger(__name__)


def reset_per_device_batch_size(
    global_batch_size, per_device_train_batch_size, dataset_world_size
):
    assert (
        global_batch_size % dataset_world_size == 0
    ), f"global_bsz={global_batch_size} not evenly divided by world_size={dataset_world_size}"
    batch_per_device = global_batch_size // dataset_world_size
    if batch_per_device < per_device_train_batch_size:
        gradient_accumulation_steps = 1
        per_device_train_batch_size = batch_per_device
        logger.info(
            f"reset `per_device_train_batch_size` to {per_device_train_batch_size}, global_batch_size={global_batch_size }, "
            f"dp_worldsize={ dataset_world_size}, accumulate_steps={gradient_accumulation_steps} "
        )
    else:
        assert (
            batch_per_device % per_device_train_batch_size == 0
        ), f"global_bsz={global_batch_size} not evenly divided by world_size={dataset_world_size}, batch_per_device={batch_per_device}"
        gradient_accumulation_steps = batch_per_device // per_device_train_batch_size
        logger.info(
            f"per_device_train_batch_size={per_device_train_batch_size}, global_batch_size={global_batch_size }, "
            f"dp_worldsize={dataset_world_size}, accumulate_steps={gradient_accumulation_steps} "
        )
    return per_device_train_batch_size, gradient_accumulation_steps


def get_config(verbose=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs", action="store", nargs="+", required=True, help="config files"
    )
    parser.add_argument(
        "--kwargs", action="store", nargs="+", default=[], help="extra k-v configs"
    )
    opt = parser.parse_args()
    configs = [OmegaConf.load(p) for p in opt.configs]
    configs.append(OmegaConf.from_dotlist(opt.kwargs))
    config = OmegaConf.merge(*configs)

    if "env" in config:
        for key, value in OmegaConf.to_object(config.env).items():
            config.env[key] = os.environ.get(key, value)
    OmegaConf.resolve(config)
    if verbose:
        print(
            yaml.dump(
                OmegaConf.to_object(config),
                default_flow_style=False,
                indent=4,
                width=9999,
                allow_unicode=True,
            )
        )
    return config


def get_flatten_mesh(mesh):
    return dist.ProcessMesh(mesh.process_ids)


def is_pp_enable():
    mesh = fleet.auto.get_mesh()
    return "pp" in mesh.dim_names


def get_mesh(pp_idx=None):
    mesh = fleet.auto.get_mesh()
    if is_pp_enable():
        mesh = mesh.get_mesh_with_dim("pp", pp_idx)
    return mesh


def _reshard(tensor, mesh, placements):
    dst_tensor = dist.auto_parallel.moe_utils._dist_reshape(
        tensor, tensor.shape, mesh, placements
    )
    return dst_tensor
