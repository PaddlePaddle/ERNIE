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
    """
    根据给定的平均值、数据量和最大值，生成一组随机的 topk 列表。

    Args:
        topk_avg (int): 平均 topk 数。
        data_num (int): 数据集中的样本数。
        max_topk (int, optional): 最大 topk 值。
        elastic_topk_prob (float, optional): 采用弹性TOPK的概率

    Returns:
        list of int: 包含 `data_num` 个元素的列表，每个元素代表对应样本的 topk 值，范围为 [1, max_topk]。
        如果 `data_num` 等于 1，则返回一个长度为 1 的列表，其中元素为 min(total, max_topk)，total 为 topk_avg * data_num。

    Raises:
        None
    """
    p = random_manager.global_random.random()
    total = topk_avg * data_num  # 总数
    if p < (1 - elastic_topk_prob):
        return [topk_avg] * data_num
    # **特殊情况：data_num=1，直接返回**
    if data_num == 1:
        return [min(total, max_topk)]  # 确保不超过 max_topk

    while True:
        # 1. 生成 `data_num-1` 个分割点，确保每个数都大于 0
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
        # 2. 检查是否所有数都 ≤ max_topk
        if all(1 <= x <= max_topk for x in parts):
            break  # 只有满足要求才退出循环

    return parts
