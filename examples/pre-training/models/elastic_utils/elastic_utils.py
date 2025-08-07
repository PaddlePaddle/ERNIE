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
elastic utils
"""
# *_*coding:utf-8 *_*
from models.global_random import random_manager


def get_topk_random_uniform(topk_avg, data_num, max_topk, elastic_topk_prob=1.0):

    p = random_manager.global_random.random()
    total = topk_avg * data_num
    if p < (1 - elastic_topk_prob):
        return [topk_avg] * data_num
    if data_num == 1:
        return [min(total, max_topk)]

    while True:
        split_points = sorted(
            random_manager.global_random.sample(range(1, total), data_num - 1)
        )
        parts = (
            [split_points[0]]
            + [
                split_points[i] - split_points[i - 1]
                for i in range(1, len(split_points))
            ]
            + [total - split_points[-1]]
        )
        if all(1 <= x <= max_topk for x in parts):
            break

    return parts
