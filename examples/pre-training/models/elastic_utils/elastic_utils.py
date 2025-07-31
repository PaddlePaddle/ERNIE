#!/usr/bin/env python3

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
import paddle
import numpy as np
import paddle.distributed.fleet as fleet
from models.global_random import random_manager
from .vpp_simulator import ChunkType, VPPSimulator


# def generate_two_exclusive_lists(layer_num, min_ratio=0.25):
#     if not (0 < min_ratio <= 0.5):
#         raise ValueError("min_ratio 必须在 (0, 0.5] 之间，否则会导致分布错误")
#
#     if layer_num < int(1 / min_ratio):
#         raise ValueError(f"layer_num 必须 >= {int(1 / min_ratio)}，否则无法保证 {min_ratio * 100}% 的 1 分布")
#
#     min_ones = math.ceil(layer_num * min_ratio)
#     max_ones = math.floor(layer_num * (1 - min_ratio))
#     num_ones = random.randint(min_ones, max_ones)
#
#     indices = random.sample(range(layer_num), num_ones)
#     matrix = np.zeros((layer_num, 2), dtype=int)
#     matrix[indices, 0] = 1  # 在第一列填 1
#     matrix[:, 1] = 1 - matrix[:, 0]  # 第二列为互补值
#     return matrix


def get_elastic_num(
    elastic_model_list,
    elastic_model_ratio_list,
    data_accumulation_steps,
    gradient_accumulation_steps,
):
    """检查弹性相关配置,并返回相应的信息"""
    assert len(elastic_model_list) == len(
        elastic_model_ratio_list
    ), "elastic_model_list and elastic_model_ratio must have the same length"
    need_step_list = []
    is_full_model = []
    full_model_batch_size = elastic_model_list[0]
    elastic_batch_size = elastic_model_list[1]
    for i, mode_ratio in enumerate(elastic_model_ratio_list):
        model_num = mode_ratio * gradient_accumulation_steps
        assert model_num % 1 == 0, f"need_data must be an integer, but got {model_num}"
        need_step = [elastic_model_list[i]] * int(model_num)
        if i == 0:
            is_full = [True] * int(model_num)
        else:
            is_full = [False] * int(model_num)
        is_full_model.extend(is_full)
        need_step_list.extend(need_step)
    sum_need_step = sum(need_step_list)
    max_step = max(need_step_list)
    assert sum_need_step == data_accumulation_steps, (
        f"sum_need_data must be equal to data_accumulation_steps, "
        f"but got {sum_need_step} and {data_accumulation_steps}"
    )
    # print(f"elastic info need_step_list: {need_step_list}, max_step: {max_step}, is_full_model: {is_full_model}")
    return (
        need_step_list,
        max_step,
        is_full_model,
        full_model_batch_size,
        elastic_batch_size,
    )


def padding_input_for_eb5(input_dict, max_batch_size):
    """
    对 input_dict 里的指定键 ("input_ids", "attention_mask",
    "position_ids", "inbatch_pack_offset") 进行 batch 维度 (0 维) 填充到 max_batch_size
    """
    padded_dict = {}
    pad_keys = {
        "input_ids",
        "position_ids",
        "labels",
        "task_id",
        "startend_row_indices",
        "image_infinity_scale",
        "video_infinity_scale",
    }

    for key, value in input_dict.items():
        if key in pad_keys and value is not None:
            # 获取当前 batch_size
            current_batch_size = value.shape[0]

            # 如果当前 batch_size 已经等于 max_batch_size，不需要填充
            if current_batch_size == max_batch_size:
                padded_dict[key] = value
                continue

            # 计算需要填充的数量
            pad_size = max_batch_size - current_batch_size

            # 构造 padding tensor，形状和原 tensor 其他维度相同，但第 0 维度是 pad_size
            pad_shape = [pad_size] + list(value.shape[1:])
            if key == "labels":
                pad_tensor = (
                    paddle.ones(pad_shape, dtype=value.dtype) * -100
                )  # ignore_index
            else:
                pad_tensor = paddle.zeros(pad_shape, dtype=value.dtype)

            # 拼接填充后的 tensor
            padded_dict[key] = paddle.concat([value, pad_tensor], axis=0)
        else:
            # 其他 key 直接保留原值
            padded_dict[key] = value

    return padded_dict


def padding_input(input_dict, max_batch_size):
    """
    对 input_dict 里的指定键 ("input_ids", "attention_mask",
    "position_ids", "inbatch_pack_offset") 进行 batch 维度 (0 维) 填充到 max_batch_size
    """
    padded_dict = {}
    pad_keys = {"input_ids", "attention_mask", "position_ids", "inbatch_pack_offset"}

    for key, value in input_dict.items():
        if key in pad_keys and value is not None:
            # 获取当前 batch_size
            current_batch_size = value.shape[0]

            # 如果当前 batch_size 已经等于 max_batch_size，不需要填充
            if current_batch_size == max_batch_size:
                padded_dict[key] = value
                continue

            # 计算需要填充的数量
            pad_size = max_batch_size - current_batch_size

            # 构造 padding tensor，形状和原 tensor 其他维度相同，但第 0 维度是 pad_size
            pad_shape = [pad_size] + list(value.shape[1:])
            pad_tensor = paddle.zeros(pad_shape, dtype=value.dtype)

            # 拼接填充后的 tensor
            padded_dict[key] = paddle.concat([value, pad_tensor], axis=0)
        else:
            # 其他 key 直接保留原值
            padded_dict[key] = value

    return padded_dict


def merge_inputs(elastic_inputs, inputs):
    """
    将新输入和已有的 elastic_inputs 合并。如果 elastic_inputs 为 None，则初始化为新输入；否则，将新输入与已有的 elastic_inputs 进行拼接。

    Args:
        elastic_inputs (dict, optional): 以字典形式存储的输入，可能包含 None 值。默认为 None。
        inputs (dict): 以字典形式存储的新输入，必须包含所有新输入的键名。

    Returns:
        dict: 返回合并后的字典，包含所有输入，包括新输入和已有的 elastic_inputs。
    """
    if elastic_inputs is None:
        # 初始化 elastic_inputs，但保留 None 值
        elastic_inputs = {
            key: (value.clone() if value is not None else None)
            for key, value in inputs.items()
        }
    else:
        # 合并 inputs
        for key, value in inputs.items():
            if key in elastic_inputs:
                if value is None or elastic_inputs[key] is None:
                    # 只要有一个是 None，就不进行修改，保持原值
                    continue
                # 两者都非 None，进行拼接
                if key in [
                    "image_labels",
                    "image_features",
                    "video_labels",
                    "video_features",
                    "image_full_features",
                    "video_full_features",
                    "video_features_history",
                    "video_loss_reweight",
                    "audio_ids",
                    "audio_labels",
                    "audio_spk_embs",
                ]:
                    # EB5, 上述属性不存在bsz维度，需要特殊处理
                    pad_value = (
                        -100 if "labels" in key else -1
                    )  # label时默认pad为-100(ignore_idx)
                    # 找到第一个pad_value
                    first_pad_index_in_ori_value = paddle.where(
                        elastic_inputs[key].reshape([elastic_inputs[key].shape[0], -1])[
                            :, 0
                        ]
                        == pad_value
                    )[0][0]
                    first_pad_index_in_new_value = paddle.where(
                        value.reshape([value.shape[0], -1])[:, 0] == pad_value
                    )[0][0]
                    if first_pad_index_in_new_value.item() != 0:
                        # 填充新数值
                        elastic_inputs[key][
                            first_pad_index_in_ori_value : first_pad_index_in_ori_value
                            + first_pad_index_in_new_value
                        ] = value[:first_pad_index_in_new_value]

                else:
                    elastic_inputs[key] = paddle.concat(
                        [elastic_inputs[key], value], axis=0
                    )
            else:
                # 直接赋值（保持 None 的情况）
                elastic_inputs[key] = value.clone() if value is not None else None

    return elastic_inputs


def generate_choice_lists(layer_num, all_size, choice_size):
    """
    生成一个二维列表，其中每个元素都是一个长度为 all_size 的列表，列表中的每个元素都是一个长度为 layer_num 的列表。
    该函数会随机选择 all_size 个索引，并将这些索引对应的列表中的元素设置为 1，其他元素均为 0。

    Args:
        layer_num (int): 二维列表的第一维长度，即列表中每个列表的长度。
        all_size (int): 二维列表的第二维长度，即列表中每个列表中的元素的长度。
        choice_size (int): 需要随机选择的索引的数量。

    Returns:
        list of list of int: 返回一个二维列表，其中每个元素都是一个长度为 all_size 的列表，列表中的每个元素都是一个长度为 layer_num 的列表，
                              其中包含了随机选择的索引对应的列表中的元素设置为 1，其他元素均为 0。
    """
    result = [[0] * all_size for _ in range(layer_num)]  # 预分配 0
    for i in range(layer_num):
        indices = random_manager.global_random.sample(
            range(all_size), choice_size
        )  # 选择 choice_size 个索引
        for idx in indices:
            result[i][idx] = 1  # 设为 1
    return result


def generate_full_lists(layer_num, all_size, choice_size):
    """
    生成一个二维列表，其中每行都是一个长度为all_size的全0向量，但前choice_size个元素为1。

    Args:
        layer_num (int): 需要生成的二维列表的行数。
        all_size (int): 每行二维列表的大小，即所有元素的个数。
        choice_size (int): 需要设置为1的元素的个数。

    Returns:
        numpy.ndarray, shape=(layer_num, all_size): 返回一个二维列表，其中每行都是一个长度为all_size的向量，前choice_size个元素为1，
        其他元素为0。
    """
    result = np.zeros((layer_num, all_size), dtype=int)
    for i in range(layer_num):
        result[i, :choice_size] = 1

    return result


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


def get_topk_random(topk, max_topk, min_topk=1, elastic_topk_prob=1.0):
    """
    根据给定的平均值、数据量和最大值，生成一组随机的 topk 列表。

    Args:
        topk (int): topk 数。
        max_topk (int): 最大 topk 值。
        min_topk (int, optional): 最小 topk 值。
        elastic_topk_prob (float, optional): 采用弹性TOPK的概率

    Returns:
        返回topk随机值,是否和topk不相等的bool

    Raises:
        None
    """
    p = random_manager.global_random.random()
    if p < (1 - elastic_topk_prob):
        return topk, False
    # random_topk = random_manager.global_random.randint(min_topk, max_topk)
    choice_topk_list = [x for x in range(min_topk, max_topk + 1) if x != topk]
    random_topk = random_manager.global_random.choice(choice_topk_list)
    return random_topk, True


def get_valid_expert_num(min_val, moe_num_experts, moe_world_size, max_prob):
    """
    获取有效的专家数量和专家index列表
    :param min_val: 最少专家数
    :param moe_num_experts: 最大专家数
    :param moe_world_size: 专家切分份数
    :param max_prob: 选择最大专家数的概率
    :return:
    返回新 expert 数量、每 rank 的 local expert 列表、全局 expert gate 的 mask。
    """
    if min_val > moe_num_experts:
        raise ValueError(
            f"min_val ({min_val}) cannot be greater than moe_num_expert ({moe_num_experts})"
        )
    # print(f"yang debug min_val is {min_val}, moe_num_experts is {moe_num_experts},
    # moe_world_size is {moe_world_size}, max_prob is {max_prob}")

    # max_prob 概率选最大 expert 数
    p = random_manager.global_random.random()
    if p < max_prob:
        new_expert_num = moe_num_experts
    else:
        max_val = moe_num_experts
        candidates = [
            x
            for x in range(min_val, max_val - moe_world_size + 1)
            if x % moe_world_size == 0
        ]
        new_expert_num = (
            random_manager.global_random.choice(candidates)
            if candidates
            else moe_num_experts
        )

    if new_expert_num == moe_num_experts:
        new_expert_rank_list, global_gate_mask = None, None
    else:
        # 计算每个 rank 需要多少个 expert
        experts_per_rank = moe_num_experts // moe_world_size
        need_experts_per_rank = new_expert_num // moe_world_size

        new_expert_rank_list = []
        selected_indices = []
        # 生成 new_expert_rank_list，确保每个 rank 选取的专家是随机的
        for moe_rank in range(moe_world_size):
            one_rank_experts = sorted(
                random_manager.global_random.sample(
                    range(experts_per_rank), need_experts_per_rank
                )
            )
            new_expert_rank_list.append(one_rank_experts)
            selected_indices.extend(
                [moe_rank * experts_per_rank + i for i in one_rank_experts]
            )

        # 创建 bool mask，选中的为 True，其余为 False
        mask = paddle.zeros([moe_num_experts], dtype="bool")
        mask[paddle.to_tensor(selected_indices, dtype="int64")] = True

        global_gate_mask = paddle.where(
            mask,
            paddle.zeros_like(mask, dtype="float32"),  # 如果 mask 是 True，就设为 0.0
            paddle.full_like(mask, float("-inf"), dtype="float32"),  # 否则设为 -inf
        )
        global_gate_mask.stop_gradient = True

    return new_expert_num, new_expert_rank_list, global_gate_mask


def get_valid_eval_expert_num(min_val, moe_num_experts, moe_world_size, max_prob):
    """
    获取有效的专家数量和专家index列表
    :param min_val: 最少专家数
    :param moe_num_experts: 最大专家数
    :param moe_world_size: 专家切分份数
    :param max_prob: 选择最大专家数的概率
    :return:
    返回新 expert 数量、每 rank 的 local expert 列表、全局 expert gate 的 mask。
    """
    # 评估阶段强制设置expert数量
    new_expert_num = 8

    if new_expert_num == moe_num_experts:
        new_expert_rank_list, global_gate_mask = None, None
    else:
        # 计算每个 rank 需要多少个 expert
        experts_per_rank = moe_num_experts // moe_world_size
        need_experts_per_rank = new_expert_num // moe_world_size

        new_expert_rank_list = []
        selected_indices = []
        # 生成 new_expert_rank_list，确保每个 rank 选取的专家是随机的
        for moe_rank in range(moe_world_size):
            one_rank_experts = sorted(
                random_manager.global_random.sample(
                    range(experts_per_rank), need_experts_per_rank
                )
            )
            new_expert_rank_list.append(one_rank_experts)
            selected_indices.extend(
                [moe_rank * experts_per_rank + i for i in one_rank_experts]
            )

        # 创建 bool mask，选中的为 True，其余为 False
        mask = paddle.zeros([moe_num_experts], dtype="bool")
        mask[paddle.to_tensor(selected_indices, dtype="int64")] = True

        global_gate_mask = paddle.where(
            mask,
            paddle.zeros_like(mask, dtype="float32"),  # 如果 mask 是 True，就设为 0.0
            paddle.full_like(mask, float("-inf"), dtype="float32"),  # 否则设为 -inf
        )
        global_gate_mask.stop_gradient = True
    print(
        f"yang debug new_expert_num is {new_expert_num}, new_expert_rank_list is {new_expert_rank_list}, "
        f"global_gate_mask is {global_gate_mask}"
    )
    return new_expert_num, new_expert_rank_list, global_gate_mask


_pp_balance_elastic_layer_manager = None


def set_pp_balance_elastic_layer_manager(value):
    global _pp_balance_elastic_layer_manager
    _pp_balance_elastic_layer_manager = value


def get_pp_balance_elastic_layer_manager():
    global _pp_balance_elastic_layer_manager
    return _pp_balance_elastic_layer_manager


class PPBalanceElasticLayerManager:
    def __init__(
        self,
        pp_degree,
        vpp_degree,
        num_acc_steps,
        no_elastic_acc_step,
        num_hidden_layers,
        remove_head_layer,
        remove_tail_layer,
        retain_layer_prob,
    ):
        # 记录通信组相关信息
        self.hcg = fleet.get_hybrid_communicate_group()
        self.use_expert_group = hasattr(self.hcg, "get_expert_parallel_group")
        self.mp_group = self.hcg.get_model_parallel_group()
        self.pp_group = self.hcg.get_pipe_parallel_group()
        self.pp_rank = self.hcg.get_stage_id()
        self.mp_src_rank = self.hcg.get_model_parallel_group_src_rank()
        self.cur_rank = paddle.distributed.get_rank()
        if self.use_expert_group:
            self.expert_group = self.hcg.get_expert_parallel_group()
            self.expert_src_rank = self.hcg.get_expert_parallel_group_src_rank()

        # 记录模型与训练配置
        self.pp_degree = pp_degree
        self.vpp_degree = vpp_degree
        self.num_acc_steps = num_acc_steps
        self.no_elastic_acc_step = no_elastic_acc_step
        self.elastic_acc_step = num_acc_steps - no_elastic_acc_step
        self.remove_head_layer = remove_head_layer
        self.remove_tail_layer = remove_tail_layer
        self.retain_layer_prob = retain_layer_prob
        self.num_hidden_layers = (
            num_hidden_layers - remove_head_layer - remove_tail_layer
        )
        self.elastic_layer_ids = range(
            remove_head_layer, num_hidden_layers - remove_tail_layer
        )

        # 中间变量初始化
        self.acc_stamp = [
            0
        ] * self.num_hidden_layers  # 用于记录每个层的调用is_elastic_layer时所处的acc_step
        self.elastic_micro_step = self._get_elastic_micro_step()

        # 初始化弹性层掩码
        self.step()

    def _generate_elastic_layer_mask(self):
        """
        生成弹性层掩码，用于控制哪些层参与训练，如果当前迭代步被选中，合法的层均跳过训练
        Returns:
            np.ndarray: 二维布尔数组，shape为(elastic_acc_step, num_hidden_layers)
        """
        start_step_to_acc_and_layer_ids = self.elastic_micro_step
        acc_to_elastic_layer = {}

        for acc_and_layer_ids in start_step_to_acc_and_layer_ids.values():
            if random_manager.global_random.random() > self.retain_layer_prob:
                for acc_step, layer_index in acc_and_layer_ids:
                    if acc_step not in acc_to_elastic_layer:
                        acc_to_elastic_layer[acc_step] = []
                    acc_to_elastic_layer[acc_step].append(layer_index)

        elastic_layer_mask = np.zeros(
            (self.elastic_acc_step, self.num_hidden_layers), dtype=bool
        )
        for acc_step in acc_to_elastic_layer:
            elastic_layer_mask[
                acc_step - self.no_elastic_acc_step, acc_to_elastic_layer[acc_step]
            ] = True

        return elastic_layer_mask

    def _get_elastic_micro_step(self):
        """
        获取pp均衡弹性跳层策略的候选层，即流水并行中BACKWARD阶段处于相同迭代步，且acc_step 和 layer_id 处于弹性区间的层
        """
        vpp_simulator = VPPSimulator(
            pp_degree=self.pp_degree,
            vpp_degree=self.vpp_degree,
            num_acc_steps=self.num_acc_steps,
        )
        schedule_table = vpp_simulator.schedule()

        max_micro_step = schedule_table[0][-1].end
        start_step_to_acc_and_layer_ids = {}
        elastic_acc_steps = range(self.no_elastic_acc_step, self.num_acc_steps)
        for schedule in schedule_table:
            for chunk in schedule:
                if (
                    chunk.chunk_type == ChunkType.BACKWARD
                    and chunk.acc_step in elastic_acc_steps
                    and chunk.layer_id in self.elastic_layer_ids
                ):
                    start_step = chunk.start
                    if start_step_to_acc_and_layer_ids.get(start_step, None) is None:
                        start_step_to_acc_and_layer_ids[start_step] = []
                    start_step_to_acc_and_layer_ids[start_step].append(
                        (chunk.acc_step, chunk.layer_id - self.remove_head_layer)
                    )
        return start_step_to_acc_and_layer_ids

    def step(self):
        """
        一个训练step结束后，更新acc_stamp，并生成新的elastic_layer_mask，然后广播给其他rank
        """
        self.acc_stamp = [0] * self.num_hidden_layers

        # 为保证mp组内随机性一致, 所有卡都要推进random状态。但最后会将rank0的状态广播出去
        elastic_layer_mask = [self._generate_elastic_layer_mask()]

        if self.use_expert_group:
            # expert_group 广播
            if self.expert_group.nranks > 1 and self.pp_rank == 0:
                paddle.distributed.broadcast_object_list(
                    elastic_layer_mask,
                    src=self.expert_src_rank,
                    group=self.expert_group,
                )
        else:
            # mp_group 广播
            if self.mp_group.nranks > 1 and self.pp_rank == 0:
                paddle.distributed.broadcast_object_list(
                    elastic_layer_mask,
                    src=self.mp_src_rank,
                    group=self.mp_group,
                )
        # pp_group 广播
        if self.pp_group.nranks > 1:
            paddle.distributed.broadcast_object_list(
                elastic_layer_mask,
                src=self.pp_group.ranks[0],
                group=self.pp_group,
            )
        self.elastic_layer_mask = elastic_layer_mask[0]

    def is_elastic_layer(self, layer_id):
        """
        判断当前层是否可跳过
        """
        if (
            layer_id >= (self.num_hidden_layers + self.remove_head_layer)
            or layer_id < self.remove_head_layer
        ):
            return False

        acc_step = self.acc_stamp[layer_id - self.remove_head_layer]

        # 当前层的执行次数加1
        self.acc_stamp[layer_id - self.remove_head_layer] += 1

        if acc_step < self.no_elastic_acc_step:
            return False

        return self.elastic_layer_mask[acc_step - self.no_elastic_acc_step][
            layer_id - self.remove_head_layer
        ]


if __name__ == "__main__":
    # layer_num = 2
    # all_size = 5
    # choice_size = 4
    # a = generate_full_lists(layer_num, all_size, choice_size)
    # a = generate_choice_lists(layer_num, all_size, choice_size)
    # for i in range(20):
    #     a = get_topk_random_uniform(topk_avg=8, data_num=1, max_topk=15)
    new_expert_num, new_expert_rank_list, global_gate_mask = get_valid_expert_num(
        min_val=1, moe_num_experts=8, moe_world_size=2, max_prob=0.2
    )
    print("new_expert_num: ", new_expert_num)
    print("new_expert_rank_list: ", new_expert_rank_list)
    print("global_gate_mask: ", global_gate_mask)
