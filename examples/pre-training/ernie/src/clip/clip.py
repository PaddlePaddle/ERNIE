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

import copy
from collections import defaultdict
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

try:
    from paddle.base import framework
except ImportError:
    from paddle.fluid import framework
from paddle.nn.clip import ClipGradBase, _squared_l2_norm
from src.utils import logger


class ClipGradByAdaptiveNorm(ClipGradBase):

    def __init__(
        self,
        clip_ratio=1.03,
        start_clip_steps=100,
        beta=0.98,
        epsilon=1e-8,
        shard_clip=False,
        enable_record=False,
        enable_record_clip_history=False,
        verbose=False,
    ):
        super().__init__()
        self.clip_ratio = clip_ratio
        self.beta = beta
        self.epsilon = epsilon
        self.state = defaultdict(dict)
        self.start_clip_steps = start_clip_steps
        self.shard_clip = shard_clip
        self.enable_record = enable_record
        self.steps = 0
        self.enable_record_clip_history = enable_record_clip_history
        self.verbose = verbose
        self.keys = [
            "clip_ratio",
            "beta",
            "epsilon",
            "start_clip_steps",
            "shard_clip",
            "enable_record",
            "steps",
            "enable_record_clip_history",
        ]

        if start_clip_steps < 0:
            raise ValueError(
                "start_clip_steps {}, please start_clip_steps >= 0.".format(
                    start_clip_steps
                )
            )

    def __str__(self):
        return "ClipGradByAdaptiveNorm, clip_ratio={}, beta={}, start_clip_steps={}, \
            shard_clip={}, enable_record={}".format(
            self.clip_ratio,
            self.beta,
            self.start_clip_steps,
            self.shard_clip,
            self.enable_record,
        )

    def clip_by_norm(self, param, grad, norm_value, global_norm):

        state = self.state[param.name]

        if "norm_value" not in state:
            state["norm_value"] = norm_value

        if "clip_times" not in state:
            state["clip_times"] = 0

        if self.enable_record_clip_history:
            if "clip_history" not in state:
                state["clip_history"] = {}

        avg_norm_value = state["norm_value"]

        if self.enable_record:
            if "norm_history" not in state:
                state["norm_history"] = {}
            state["norm_history"][self.steps] = [
                float(norm_value),
                float(avg_norm_value),
            ]

        if self.steps <= self.start_clip_steps:
            clip_coeff = 1.0 / (global_norm + self.epsilon)
            if clip_coeff < 1.0:
                grad.multiply_(clip_coeff)
                param._reset_grad_inplace_version(True)

            if norm_value < state["norm_value"]:
                state["norm_value"] = norm_value
        else:
            if norm_value > self.clip_ratio * avg_norm_value:
                # clip grad
                coef = (self.clip_ratio * avg_norm_value) / (norm_value + self.epsilon)
                grad.multiply_(coef)
                param._reset_grad_inplace_version(True)
                norm_value_old = norm_value
                norm_value = self.clip_ratio * avg_norm_value
                state["clip_times"] = state["clip_times"] + 1
                if self.enable_record_clip_history:
                    state["clip_history"][self.steps] = [
                        float(norm_value_old),
                        float(norm_value),
                    ]
                if self.verbose:
                    logger.info(
                        "{} gradclip {} times, clip from {} to {}".format(
                            param.name,
                            state["clip_times"],
                            float(norm_value_old),
                            float(norm_value),
                        )
                    )

                    logger.info(
                        "{} steps {}, gradclip {} times, clip_ratio {}, clip from {} to {}".format(
                            param.name,
                            self.steps,
                            state["clip_times"],
                            self.clip_ratio,
                            float(norm_value_old),
                            float(norm_value),
                        )
                    )
            state["norm_value"] = avg_norm_value * self.beta + norm_value * (
                1.0 - self.beta
            )

        return grad

    @paddle.no_grad()
    def _dygraph_clip(self, params_grads):
        global_norm_tensor = None
        if self.steps <= self.start_clip_steps:
            hcg = fleet.get_hybrid_communicate_group()
            mp_size = hcg.get_model_parallel_world_size()
            mp_group = hcg.get_model_parallel_group()
            pp_size = hcg.get_pipe_parallel_world_size()
            pp_group = hcg.get_pipe_parallel_group()
            sharding_size = hcg.get_sharding_parallel_world_size()
            sharding_group = hcg.get_sharding_parallel_group()

            norm_squared_values = []
            for p, g in params_grads:
                if g is None:
                    continue
                if getattr(p, "need_clip", True) is False:
                    continue
                norm_squared_value = _squared_l2_norm(g)
                if not p.is_distributed and mp_size > 1:
                    norm_squared_value = norm_squared_value / mp_size
                norm_squared_values.append(norm_squared_value)

            global_norm_squared_tensor = paddle.stack(norm_squared_values).sum()

            if mp_size > 1:
                dist.all_reduce(global_norm_squared_tensor, group=mp_group)
            if pp_size > 1:
                dist.all_reduce(global_norm_squared_tensor, group=pp_group)
            if sharding_size > 1:
                dist.all_reduce(global_norm_squared_tensor, group=sharding_group)
            global_norm_tensor = paddle.sqrt(global_norm_squared_tensor)

        if self.verbose and global_norm_tensor is not None:
            logger.info(
                "step: {}, global norm: {}".format(
                    self.steps, float(global_norm_tensor)
                )
            )

        if hasattr(self, "sharding_stage1_v2") and self.sharding_stage1_v2:
            need_sync = False
            if not self.shard_clip:
                hcg = fleet.get_hybrid_communicate_group()
                mp_size = hcg.get_model_parallel_world_size()
                mp_group = hcg.get_model_parallel_group()
                sharding_size = hcg.get_sharding_parallel_world_size()
                sharding_group = hcg.get_sharding_parallel_group()
                if mp_size > 1 or sharding_size > 1:
                    need_sync = True

            norm_squared_values = [
                paddle.zeros([1], dtype=params_grads[0][1].dtype)
                for _ in range(self.num_params)
            ]

            for p, g in params_grads:
                if g is None:
                    continue
                if getattr(p, "need_clip", True) is False:
                    continue
                norm_squared_value = _squared_l2_norm(g)
                if need_sync and not p.is_distributed:
                    norm_squared_values[self.pname_to_paramindex[p.name]] = (
                        1 / mp_size
                    ) * norm_squared_value
                else:
                    norm_squared_values[self.pname_to_paramindex[p.name]] = (
                        norm_squared_value
                    )

            num_has_grad = len(norm_squared_values)
            norm_squared_tensor = paddle.concat(norm_squared_values, axis=0)
            if need_sync:
                if mp_size > 1:
                    dist.all_reduce(norm_squared_tensor, group=mp_group)
                if sharding_size > 1:
                    dist.all_reduce(norm_squared_tensor, group=sharding_group)

            norm_tensor = paddle.sqrt(norm_squared_tensor)
            norm_values = paddle.split(norm_tensor, num_has_grad, axis=0)

            params_and_grads = []
            for p, g in params_grads:
                if g is None:
                    continue
                if getattr(p, "need_clip", True) is False:
                    params_and_grads.append((p, g))
                    continue
                new_grad = self.clip_by_norm(
                    p,
                    g,
                    norm_values[self.pname_to_paramindex[p.name]],
                    global_norm_tensor,
                )
                params_and_grads.append((p, new_grad))
        else:
            need_sync = False
            if not self.shard_clip:
                hcg = fleet.get_hybrid_communicate_group()
                mp_size = hcg.get_model_parallel_world_size()
                mp_group = hcg.get_model_parallel_group()
                if mp_size > 1:
                    need_sync = True

            norm_squared_values = []
            for p, g in params_grads:
                if g is None:
                    continue
                if getattr(p, "need_clip", True) is False:
                    continue
                norm_squared_value = _squared_l2_norm(g)
                if need_sync and not p.is_distributed:
                    norm_squared_values.append((1 / mp_size) * norm_squared_value)
                else:
                    norm_squared_values.append(norm_squared_value)

            num_has_grad = len(norm_squared_values)
            norm_squared_tensor = paddle.concat(norm_squared_values, axis=0)
            if need_sync:
                dist.all_reduce(norm_squared_tensor, group=mp_group)

            norm_tensor = paddle.sqrt(norm_squared_tensor)
            norm_values = paddle.split(norm_tensor, num_has_grad, axis=0)

            params_and_grads = []
            idx = 0
            for p, g in params_grads:
                if g is None:
                    continue
                if getattr(p, "need_clip", True) is False:
                    params_and_grads.append((p, g))
                    continue
                new_grad = self.clip_by_norm(p, g, norm_values[idx], global_norm_tensor)
                params_and_grads.append((p, new_grad))
                idx += 1

        self.steps += 1
        return params_and_grads

    @framework.dygraph_only
    def state_dict(self):

        state_dict = {k: v for k, v in self.state.items()}
        for key in self.keys:
            state_dict[key] = self.__dict__[key]
        return state_dict

    @framework.dygraph_only
    def set_state_dict(self, state_dict):

        if len(state_dict) == 0 or state_dict is None:
            logger.info("state_dict is empty, please check if it is right.")

        for key in self.keys:
            if key in state_dict:
                self.__dict__[key] = state_dict[key]
            else:
                logger.info("Can't find [ {} ] in state_dict".format(key))

        for k in state_dict:
            if k in self.keys:
                continue
            self.state[k] = copy.deepcopy(state_dict[k])
