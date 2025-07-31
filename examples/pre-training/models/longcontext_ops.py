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

"""在SequenceParallel通信组中通过token_type_ids
重新均衡输入的tensor"""
import numpy as np
from collections import namedtuple
import heapq
import paddle
from paddle.autograd import PyLayer
import paddle.distributed.fleet as fleet
import paddle.distributed as dist
from paddle.distributed.communication.batch_isend_irecv import (
    _coalescing_manager as batch_isend_irecv_coalescing_manager,
)


class MaxHeap:
    """大根堆用于排序实现最小移动次数"""

    def __init__(self, data=None):
        """
        初始化大根堆，针对元组的最后一个元素进行排序。
        """
        if data is None:
            self.heap = []
        else:
            # 基于元组最后一个元素建堆
            self.heap = [(-item[-1], item) for item in data]
            heapq.heapify(self.heap)

    def push(self, item):
        """
        向堆中插入一个元素（元组）。
        """
        # 取反元组最后一个元素进行插入
        heapq.heappush(self.heap, (-item[-1], item))

    def pop(self):
        """
        弹出并返回堆中基于最后一个元素排序的最大元素。
        """
        if self.is_empty():
            raise IndexError("pop from an empty heap")
        # 返回原始元组
        return heapq.heappop(self.heap)[1]

    def top(self):
        """
        返回堆中基于最后一个元素排序的最大元素，不弹出。
        """
        if self.is_empty():
            raise IndexError("top from an empty heap")
        return self.heap[0][1]

    def is_empty(self):
        """
        检查堆是否为空。
        """
        return len(self.heap) == 0

    def __len__(self):
        """
        返回堆中元素的数量。
        """
        return len(self.heap)


def redistribute_tokens(piles):
    """给定一个 tokens 数量列表,实现最小移动次数,使得移动后每个rank
    所拥有的 tokens 尽量均衡，在极端情况下,只会多移动一次
    """
    Movement = namedtuple("Movement", ["src", "dst", "tokens"])

    total_tokens = sum(piles)
    n = len(piles)
    m = total_tokens // n
    r = total_tokens % n

    # 计算每堆的目标数量
    targets = [m] * n

    # 将token堆按数量从大到小排序，记录原始索引
    sorted_piles = sorted(enumerate(piles), key=lambda x: x[1], reverse=True)

    # 给前r堆多分配一个token
    for i in range(r):
        idx = sorted_piles[i][0]
        targets[idx] += 1

    # 计算每堆的剩余（正数表示有多的token，负数表示需要补充的token）
    surplus_piles = []
    deficit_piles = []
    for i in range(n):
        diff = piles[i] - targets[i]
        if diff > 0:
            surplus_piles.append([i, diff])
        elif diff < 0:
            deficit_piles.append([i, -diff])

    # 按照剩余token和需要token的堆，记录移动过程
    moves = []
    surplus_piles = MaxHeap(surplus_piles)
    deficit_piles = MaxHeap(deficit_piles)
    while not surplus_piles.is_empty() and not deficit_piles.is_empty():
        # 成对出堆,进行移动
        src_rank, surplus = surplus_piles.pop()
        dst_rank, deficit = deficit_piles.pop()
        move_amount = min(surplus, deficit)
        moves.append(Movement(src=src_rank, dst=dst_rank, tokens=move_amount))
        # print(f'src_rank {src_rank} to dst_rank {dst_rank}, tokens {move_amount}')

        # 因为 move_amount 是成对的堆顶元素,因此经过移动后,还没达到平衡的一个堆顶元素还需要进堆
        if (surplus - move_amount) != 0:
            surplus_piles.push([src_rank, surplus - move_amount])
        if (deficit - move_amount) != 0:
            deficit_piles.push([dst_rank, deficit - move_amount])

    return moves


class TensorBalanceByTokenType(PyLayer):
    """该PyLayer通过token_type_ids实现Tensor均衡"""

    @staticmethod
    def forward(
        ctx,
        tensor: paddle.Tensor,
        token_type_ids: paddle.Tensor,
        group=None,
        is_tensor_sharded=True,
        axis=0,
        is_token_type_ids_sharded=False,
        unique_tokens_type=None,
    ):
        """根据 token_type_ids 将输入 tensor 在 group 间均衡做均衡"""

        ctx.is_tensor_sharded = is_tensor_sharded
        ctx.axis = axis
        ctx.tensor_shape = tensor.shape
        ctx.tensor_dtype = tensor.dtype
        ctx.token_type_ids_shape = token_type_ids.shape
        ctx.token_type_ids_dtype = token_type_ids.dtype

        tensor_parallel_degree = (
            fleet.get_hybrid_communicate_group().get_model_parallel_world_size()
        )
        if tensor_parallel_degree > 1:
            ctx.group = (
                fleet.get_hybrid_communicate_group().get_model_parallel_group()
                if group is None
                else group
            )
            ctx.rank = ctx.group.rank
            ctx.world_size = ctx.group.nranks
        else:
            ctx.group = None
            ctx.rank = 0
            ctx.world_size = 1
        if len(ctx.tensor_shape) == 1:
            if not ctx.is_tensor_sharded:
                tensor = tensor.split(num_or_sections=ctx.world_size, axis=0)[ctx.rank]
            tensor = tensor.reshape([-1, 1])
        else:
            if (
                len(ctx.tensor_shape) == 2
                and not ctx.is_tensor_sharded
                and (axis == -1 or axis == len(ctx.tensor_shape) - 1)
            ):
                raise ValueError(
                    "Do not support len(ctx.tensor_shape) == 2 and not ctx.is_tensor_sharded"
                    + " and (axis == -1 or axis == len(ctx.tensor_shape) -1)"
                )
            assert (
                len(ctx.tensor_shape) <= 3
            ), f"len(tensor.shape) must <= 3, but got {len(tensor.shape)}"
            if len(ctx.tensor_shape) == 3:
                assert ctx.tensor_shape[0] == 1, "only support tensor.shape[0] == 1"

            if not ctx.is_tensor_sharded:
                tensor = tensor.split(num_or_sections=ctx.world_size, axis=ctx.axis)[
                    ctx.rank
                ]
            tensor = tensor.reshape([-1, tensor.shape[-1]])

        ctx.tensor_grad_shape = tensor.shape

        if is_token_type_ids_sharded:
            assert (
                unique_tokens_type is not None and len(unique_tokens_type) > 0
            ), "require len(unique_tokens_type) > 0 when is_token_type_ids_sharded=True"
            ctx.unique_tokens_type = unique_tokens_type
            token_type_ids_per_rank = token_type_ids.flatten()
        else:
            ctx.unique_tokens_type = token_type_ids.unique().tolist()
            token_type_ids_per_rank = token_type_ids.flatten().split(ctx.world_size)
        assert tensor.shape[0] == token_type_ids_per_rank[ctx.rank].shape[0], (
            f"tensor.shape[0]:{tensor.shape[0]} != "
            + f"token_type_ids_per_rank[ctx.rank].shape[0]:{token_type_ids_per_rank[ctx.rank].shape[0]}"
        )
        token_type_ids_per_rank_np = [
            chunk.numpy() for chunk in token_type_ids_per_rank
        ]
        tensor_list = []
        token_type_ids_list = []

        ctx.rend_recv_tokens_per_type = {}
        ctx.rend_recv_rank_per_type = {}
        ctx.indices_per_type = {}

        for token_type in ctx.unique_tokens_type:
            indices = paddle.nonzero(
                token_type_ids_per_rank[ctx.rank] == token_type
            ).flatten()
            tensor_cur_rank = paddle.gather(tensor, indices)
            token_type_ids_cur_rank = paddle.gather(
                token_type_ids_per_rank[ctx.rank], indices
            )

            ctx.indices_per_type[token_type] = indices

            type_counts_per_rank = [
                np.sum(chunk == token_type) for chunk in token_type_ids_per_rank_np
            ]
            move_records = redistribute_tokens(type_counts_per_rank)

            ## 在最小移动次数的语义下,当 tokens 大于平均值的 rank 只会往其他rank发,不可能再有收的情况
            ## 在这种背景下,下面代码是成立的
            rend_recv_tokens = {rank: [] for rank in range(ctx.world_size)}
            rend_recv_rank = {rank: [] for rank in range(ctx.world_size)}
            for move in move_records:
                rend_recv_tokens[move.src].append(move.tokens)
                rend_recv_tokens[move.dst].append(-move.tokens)
                rend_recv_rank[move.src].append(move.dst)
                rend_recv_rank[move.dst].append(move.src)

            ctx.rend_recv_tokens_per_type[token_type] = rend_recv_tokens
            ctx.rend_recv_rank_per_type[token_type] = rend_recv_rank

            if sum(rend_recv_tokens[ctx.rank]) > 0:
                # send
                sections = [
                    tensor_cur_rank.shape[0] - sum(rend_recv_tokens[ctx.rank])
                ] + rend_recv_tokens[ctx.rank]
                tensor_cur_rank = paddle.split(
                    tensor_cur_rank, num_or_sections=sections, axis=0
                )
                token_type_ids_cur_rank = paddle.split(
                    token_type_ids_cur_rank, num_or_sections=sections, axis=0
                )
                tasks = []
                with batch_isend_irecv_coalescing_manager(ctx.group, tasks):
                    for idx, rank in enumerate(rend_recv_rank[ctx.rank]):
                        task = dist.isend(
                            tensor_cur_rank[idx + 1],
                            ctx.group.ranks[rank],
                            group=ctx.group,
                        )
                        tasks.append(task)
                        task = dist.isend(
                            token_type_ids_cur_rank[idx + 1],
                            ctx.group.ranks[rank],
                            group=ctx.group,
                        )
                        tasks.append(task)
                for task in tasks:
                    task.wait()
                tensor_cur_rank = tensor_cur_rank[0]
                token_type_ids_cur_rank = token_type_ids_cur_rank[0]

            elif sum(rend_recv_tokens[ctx.rank]) < 0:
                # recv
                if tensor_cur_rank.shape[0] > 0:
                    recv_tensor_list = [tensor_cur_rank]
                    recv_token_type_ids_list = [token_type_ids_cur_rank]
                else:
                    recv_tensor_list = []
                    recv_token_type_ids_list = []
                tasks = []
                with batch_isend_irecv_coalescing_manager(ctx.group, tasks):
                    for idx, rank in enumerate(rend_recv_rank[ctx.rank]):
                        # rend_recv_tokens[ctx.rank][idx] 是一个负数,表示 recv 的大小,因此需要取负数得到正数
                        recv_tensor = paddle.empty(
                            shape=[
                                -rend_recv_tokens[ctx.rank][idx],
                                tensor_cur_rank.shape[-1],
                            ],
                            dtype=ctx.tensor_dtype,
                        )
                        recv_tensor_list.append(recv_tensor)
                        task = dist.irecv(
                            recv_tensor, ctx.group.ranks[rank], group=ctx.group
                        )
                        tasks.append(task)

                        recv_token_type_ids = paddle.empty(
                            shape=[-rend_recv_tokens[ctx.rank][idx]],
                            dtype=ctx.token_type_ids_dtype,
                        )
                        recv_token_type_ids_list.append(recv_token_type_ids)
                        task = dist.irecv(
                            recv_token_type_ids, ctx.group.ranks[rank], group=ctx.group
                        )
                        tasks.append(task)

                for task in tasks:
                    task.wait()
                if len(recv_tensor_list) > 1:
                    tensor_cur_rank = paddle.concat(recv_tensor_list, axis=0)
                    token_type_ids_cur_rank = paddle.concat(
                        recv_token_type_ids_list, axis=0
                    )
                else:
                    tensor_cur_rank = recv_tensor_list[0]
                    token_type_ids_cur_rank = recv_token_type_ids_list[0]
            else:
                # nothing to do
                pass

            # dist.barrier()

            tensor_list.append(tensor_cur_rank)
            token_type_ids_list.append(token_type_ids_cur_rank)

        ctx.output_concat_sections = [chunk.shape[0] for chunk in tensor_list]
        tensor = paddle.concat(tensor_list, axis=0)
        ctx.output_shape = tensor.shape
        token_type_ids = paddle.concat(token_type_ids_list, axis=0)

        if len(ctx.tensor_shape) == 1 and len(tensor.shape) != 1:
            tensor = tensor.reshape([-1])
        elif len(ctx.tensor_shape) == 2 and len(tensor.shape) != 2:
            tensor = tensor.reshape([-1, tensor.shape[-1]])
        elif len(ctx.tensor_shape) == 3 and len(tensor.shape) != 3:
            tensor = tensor.reshape([1, -1, tensor.shape[-1]])

        return tensor, token_type_ids

    @staticmethod
    def backward(ctx, tensor_grad, token_type_ids_grad):
        """doc"""

        tensor_grad = tensor_grad.reshape_(ctx.output_shape)
        tensor_grad_list = paddle.split(
            tensor_grad, num_or_sections=ctx.output_concat_sections, axis=0
        )
        tensor_grad = paddle.empty(ctx.tensor_grad_shape, dtype=ctx.tensor_dtype)
        for token_type_idx, token_type in enumerate(ctx.unique_tokens_type):

            tensor_grad_cur_rank = tensor_grad_list[token_type_idx]
            rend_recv_tokens = ctx.rend_recv_tokens_per_type[token_type]
            rend_recv_rank = ctx.rend_recv_rank_per_type[token_type]

            # 反向时前向的逆过程,因此是反过来的
            if sum(rend_recv_tokens[ctx.rank]) < 0:
                # send
                # rend_recv_tokens[ctx.rank][idx] 是一个负数,表示 send 的大小
                # 因此 tensor_grad_cur_rank.shape[0] + sum(rend_recv_tokens[ctx.rank] 表示剩下的是本身自己卡里的大小
                # [-x for x in rend_recv_tokens[ctx.rank]] 取负数转过来为正数
                sections = [
                    tensor_grad_cur_rank.shape[0] + sum(rend_recv_tokens[ctx.rank])
                ] + [-x for x in rend_recv_tokens[ctx.rank]]
                tensor_grad_cur_rank = paddle.split(
                    tensor_grad_cur_rank, num_or_sections=sections, axis=0
                )
                tasks = []
                with batch_isend_irecv_coalescing_manager(ctx.group, tasks):
                    for idx, rank in enumerate(rend_recv_rank[ctx.rank]):
                        task = dist.isend(
                            tensor_grad_cur_rank[idx + 1],
                            ctx.group.ranks[rank],
                            group=ctx.group,
                        )
                        tasks.append(task)
                for task in tasks:
                    task.wait()
                tensor_grad_cur_rank = tensor_grad_cur_rank[0]

            elif sum(rend_recv_tokens[ctx.rank]) > 0:
                # recv
                if tensor_grad_cur_rank.shape[0] > 0:
                    recv_tensor_grad_list = [tensor_grad_cur_rank]
                else:
                    recv_tensor_grad_list = []
                tasks = []
                with batch_isend_irecv_coalescing_manager(ctx.group, tasks):
                    for idx, rank in enumerate(rend_recv_rank[ctx.rank]):
                        recv_tensor_grad = paddle.empty(
                            shape=[
                                rend_recv_tokens[ctx.rank][idx],
                                tensor_grad_cur_rank.shape[-1],
                            ],
                            dtype=ctx.tensor_dtype,
                        )
                        recv_tensor_grad_list.append(recv_tensor_grad)
                        task = dist.irecv(
                            recv_tensor_grad, ctx.group.ranks[rank], group=ctx.group
                        )
                        tasks.append(task)

                for task in tasks:
                    task.wait()
                if len(recv_tensor_grad_list) > 1:
                    tensor_grad_cur_rank = paddle.concat(recv_tensor_grad_list, axis=0)
                else:
                    tensor_grad_cur_rank = recv_tensor_grad_list[0]
            else:
                # nothing to do
                pass

            indices = ctx.indices_per_type[token_type]
            paddle.scatter_(tensor_grad, indices, tensor_grad_cur_rank)

        if not ctx.is_tensor_sharded:
            tensor_grad_list = []
            dist.stream.all_gather(tensor_grad_list, tensor_grad, group=ctx.group)
            tensor_grad = paddle.concat(tensor_grad_list, axis=0)

        tensor_grad = tensor_grad.reshape_(ctx.tensor_shape)
        return tensor_grad, None


if __name__ == "__main__":
    mp_degree = 8
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "mp_degree": mp_degree,
    }
    fleet.init(is_collective=True, strategy=strategy)

    # tokens_type = paddle.load(f'token_type_ids_shifted_{paddle.distributed.get_rank()+6}.pd')
    # last_hidden_state = paddle.load(f'last_hidden_state_{paddle.distributed.get_rank()+6}.pd')
    #

    # warmup p2p init
    fake_data = paddle.ones([paddle.distributed.get_world_size(), 1])
    fake_out = paddle.empty([paddle.distributed.get_world_size(), 1])
    group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
    paddle.distributed.alltoall(fake_out, fake_data, group)

    last_hidden_state = (
        paddle.to_tensor(
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
            ]
        )
        .reshape([-1, 1])
        .split(mp_degree, axis=0)[paddle.distributed.get_rank()]
    )
    last_hidden_state.stop_gradient = False
    tokens_type = paddle.to_tensor(
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        dtype=paddle.int32,
    )
    last_hidden_state_remap, tokens_type_per_rank = TensorBalanceByTokenType.apply(
        last_hidden_state, tokens_type
    )

    last_hidden_state_grad = (
        paddle.to_tensor(
            [
                11,
                1,
                2,
                3,
                12,
                4,
                5,
                6,
                9,
                7,
                8,
                10,
                21,
                24,
                15,
                13,
                14,
                16,
                17,
                18,
                19,
                20,
                22,
                23,
            ]
        )
        .reshape([-1, 1])
        .split([4, 4, 3, 3, 3, 3, 2, 2], axis=0)[paddle.distributed.get_rank()]
    )
    paddle.autograd.backward([last_hidden_state_remap], [last_hidden_state_grad])
    np.testing.assert_equal(last_hidden_state.numpy(), last_hidden_state.grad.numpy())
    print("grad", last_hidden_state.grad)
