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

"""全局随机数管理模块。

该模块提供了RandomManager类，用于管理分布式训练中的随机数生成，
确保不同进程间的随机数同步和一致性。
"""
import logging
import random
import paddle.distributed.fleet as fleet

logger = logging.getLogger(__name__)

try:
    from paddle.distributed.fleet.recompute import custom_state_manager
except (ImportError, ModuleNotFoundError):
    custom_state_manager = None
    logger.warning(
        "The random states may be wrong when using recompute. Please check your paddle version."
    )


class RandomManager(object):
    def __init__(self):
        """
        初始化方法。
        """
        self.global_random = random.Random(11)

    def init_random(self):
        """
        Initialize a 2D list of random.Random objects.

        Args:
            dp_size (int): Number of data parallel processes.
            sharding_size (int): Number of sharding processes.
        """
        hcg = fleet.get_hybrid_communicate_group()
        self.dp_size = hcg.get_data_parallel_world_size()
        self.sharding_size = hcg.get_sharding_parallel_world_size()
        self.dp_rank = hcg.get_data_parallel_rank()
        self.sharding_rank = hcg.get_sharding_parallel_rank()

        self.pp_rank = hcg.get_stage_id()
        self.pp_size = hcg.get_pipe_parallel_world_size()
        if hasattr(hcg, "get_moe_sharding_parallel_rank"):
            self.moe_sharding_rank = hcg.get_moe_sharding_parallel_rank()
        else:
            self.moe_sharding_rank = 0

        seed = (
            self.pp_rank * 1000000000
            + self.dp_rank * 100000000
            + self.sharding_rank * 10000000
        )
        self.global_random = random.Random(seed)

        ep_seed = self.pp_rank * 1000000000 + self.moe_sharding_rank * 100000000
        self.ep_random = random.Random(ep_seed)

        if custom_state_manager is not None:
            custom_state_manager.set_custom_get_state_func(self.get_states)
            custom_state_manager.set_custom_set_state_func(self.set_states)

    def get_states(self):
        """get_states, 用于recompute场景"""
        return (self.global_random.getstate(), self.ep_random.getstate())

    def set_states(self, packed_states):
        """set_states, 用于recompute场景"""
        assert isinstance(packed_states, tuple)
        assert len(packed_states) == 2
        self.global_random.setstate(packed_states[0])
        self.ep_random.setstate(packed_states[1])

    def seed_random(self, global_step):
        """
        Re-seed all random.Random objects with an additional global step.

        Args:
            global_step (int): The current global step to adjust the seed.
        """
        seed = (
            self.pp_rank * 1000000000
            + self.dp_rank * 100000000
            + self.sharding_rank * 10000000
            + global_step
        )
        self.global_random.seed(seed)

        ep_seed = (
            self.pp_rank * 1000000000 + self.moe_sharding_rank * 100000000 + global_step
        )
        self.ep_random = random.Random(ep_seed)


random_manager = RandomManager()
