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


import paddle.distributed as dist
from paddle.distributed import fleet


def get_flatten_mesh(mesh):

    return dist.ProcessMesh(mesh.process_ids)


def get_mesh(pp_idx=0):

    mesh = fleet.auto.get_mesh()
    if "pp" in mesh.dim_names:
        mesh = mesh.get_mesh_with_dim("pp", pp_idx)
    return mesh


def _reshard(tensor, mesh, placements):

    dst_tensor = dist.auto_parallel.moe_utils._dist_reshape(
        tensor, tensor.shape, mesh, placements
    )
    return dst_tensor
