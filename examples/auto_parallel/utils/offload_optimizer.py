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
import paddle.distributed as dist
from paddle import _C_ops
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
    HybridParallelOptimizer,
)
from paddle.optimizer import Optimizer
from paddleformers.utils.tools import get_env_device
from paddle.utils import unique_name
from paddle.base.layer_helper import LayerHelper
from paddle.base import core
from paddle.base.framework import device_guard


def to_device(tensor, place=None):
    if place is None:
        place = get_env_device()
    if isinstance(place, str):
        place = paddle.device._convert_to_place(place)
    if not tensor.place._equals(place):
        new_t = tensor._copy_to(place, True)
        dst_tensor = tensor.value().get_tensor()
        src_tensor = new_t.value().get_tensor()
        dst_tensor._share_data_with(src_tensor)
    return tensor


def offload(tensor):
    place = paddle.CUDAPinnedPlace()
    new_tensor = to_device(tensor, place)
    assert new_tensor is tensor, "to_device must be inplace operation"


def reload(tensor):
    new_tensor = to_device(tensor)
    assert new_tensor is tensor, "to_device must be inplace operation"


def mock_offload_optimizer():
    # Step 0: mock _create_master_weight
    def new_create_master_weight(self, param):
        if param.name in self._master_weights:
            var = self._master_weights[param.name]
        else:
            var_name = self._gen_master_weight_var_name(param)
            if param._is_initialized():
                param_cpu = param._copy_to(paddle.CPUPlace(), True)
                var = paddle.cast(param_cpu, "float32")
                var.name = var_name
                var.stop_gradient = param.stop_gradient
            else:
                var = paddle.cast(param, "float32")
                var.name = var_name
            self._master_weights[param.name] = var
        return var

    setattr(Optimizer, "_create_master_weight", new_create_master_weight)

    # Step 1: mock _add_accumulator
    def new_add_accumulator(
        self,
        name,
        param,
        dtype=None,
        fill_value=0.0,
        shape=None,
        type=None,
        device=None,
    ):
        if self._name is not None:
            name = self._name + "_" + name
        if name in self._accumulators and param.name in self._accumulators[name]:
            return self._accumulators[name][param.name]
        else:
            self.need_refuse()
        if shape is None:
            shape = param.shape
        var_name = param.name + "_" + name
        var_name = unique_name.generate(var_name)
        self._opti_name_list.append(var_name)

        if device is None:
            device = self._get_device_for_param(param.name)
        if self.helper is None:
            self.helper = LayerHelper(self.__class__.__name__)
        assert isinstance(self.helper, LayerHelper)
        var = self.helper.create_global_variable(
            name=var_name,
            persistable=True,
            dtype=dtype or param.dtype,
            type=core.VarDesc.VarType.DENSE_TENSOR,
            shape=shape,
            belong_to_optimizer=True,
        )
        with device_guard(device):
            self.helper.set_variable_initializer(
                var,
                initializer=paddle.nn.initializer.Constant(value=float(fill_value)),
            )
        if "beta" not in var_name:
            placements = param.placements
        else:
            placements = [
                dist.Replicate() for _ in range(len(param.process_mesh.shape))
            ]
        var = dist.shard_tensor(var, param.process_mesh, placements)
        if len(self._accumulators_holder) > 0:
            assert (
                var_name in self._accumulators_holder
            ), f"Optimizer set error, {var_name} should in state dict"
            var.set_value(self._accumulators_holder.pop(var_name))
        self._accumulators[name][param.name] = var
        offload(var)
        return var

    setattr(Optimizer, "_add_accumulator", new_add_accumulator)

    # Step 2: mock _C_ops.adamw_ and _C_ops.adamw
    for name in ["adam_", "adamw_"]:
        origin_op = getattr(_C_ops, name)

        def new_opt_op(*args):
            for arg in args:
                if isinstance(arg, paddle.Tensor):
                    reload(arg)
            ret = origin_op(*args)
            for i, arg in enumerate(args):
                if i >= 2 and isinstance(
                    arg, paddle.Tensor
                ):  # do not offload parameter and gradient
                    offload(arg)
            return ret

        setattr(_C_ops, name, new_opt_op)

    # Step 3: mock _insert_sync
    opt_type = HybridParallelOptimizer
    origin_insert_sync = getattr(opt_type, "_insert_sync")

    def new_insert_sync(self, sync_var, *args, **kwargs):
        origin_place = sync_var.place
        reload(sync_var)
        ret = origin_insert_sync(self, sync_var, *args, **kwargs)
        new_sync_var = to_device(sync_var, origin_place)
        assert new_sync_var is sync_var, "to_device must be inplace operation"
        return ret

    setattr(opt_type, "_insert_sync", new_insert_sync)
