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
VPPSimulator, 模拟VPP编排，用于VPP均衡弹性跳层策略
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from enum import Enum


class ChunkType(Enum):
    """
    vpp chunk类型枚举
    """

    FORWARD = "F"
    BACKWARD = "B"
    BUBBLE = "Z"


class Chunk:
    """
    vpp chunk
    """

    def __init__(
        self,
        virtual_pp_rank,
        acc_step,
        pp_degree,
        vpp_degree,
        stage_id,
        chunk_type,
        start,
        end,
        barrier_step=-1,
    ):
        self.virtual_pp_rank = virtual_pp_rank
        self.acc_step = acc_step
        self.stage_id = stage_id
        self.chunk_type = chunk_type
        self.start = start
        self.end = end
        self.barrier_step = barrier_step
        self.layer_id = (
            self.virtual_pp_rank * pp_degree + self.stage_id
            if self.chunk_type != ChunkType.BUBBLE
            else None
        )

    def __str__(self):
        if self.chunk_type == ChunkType.BUBBLE:
            return f"{self.chunk_type.value}({self.start, self.end})"
        return f"{self.chunk_type.value}{self.layer_id}_{self.acc_step+1}{self.start, self.end}"

    def __repr__(self):
        return str(self)


class VPPSimulator:
    """
    VPPSimulator, 模拟VPP编排，用于VPP均衡弹性跳层策略
    """

    def __init__(
        self, pp_degree, vpp_degree, num_acc_steps, enable_batch_send_recv=True
    ):
        self.pp_degree = pp_degree
        self.vpp_degree = vpp_degree
        self.num_acc_steps = num_acc_steps
        self.chunks = []
        self.schedule_table = [[] for _ in range(self.pp_degree)]

        self.first_chunk_acc = (self.num_acc_steps % self.pp_degree) + self.pp_degree
        self.num_steps = self.num_acc_steps * self.vpp_degree
        self.layer_num = self.pp_degree * self.vpp_degree

        self.enable_batch_send_recv = enable_batch_send_recv

        self._is_scheduled = False

    def _add_bubble(self):
        num_chunks = self.num_acc_steps * self.pp_degree * self.vpp_degree * 2
        num_done_chunks = 0

        undone_micro_step = [0] * self.pp_degree  # 每个stage未处理的第一个chunk下标
        stage_index = 0  # 指向当前待处理stage的指针

        while num_done_chunks < num_chunks:
            micro_step = undone_micro_step[stage_index]
            num_chunks_each_stage = len(self.schedule_table[stage_index])

            if micro_step < num_chunks_each_stage:
                chunk = self.schedule_table[stage_index][micro_step]
                if micro_step > 0:
                    chunk.start = max(
                        chunk.start,
                        self.schedule_table[stage_index][micro_step - 1].end,
                    )

                preorder_chunk = self._get_preorder_chunk(chunk)

                if preorder_chunk is not None:
                    if preorder_chunk.end == 0:  # 前序chunk还没处理
                        # print(f"{stage_index}, {micro_step} : {preorder_chunk} -> {chunk} skiped")
                        stage_index = (stage_index + 1) % self.pp_degree
                        continue
                    chunk.start = max(chunk.start, preorder_chunk.end)

                chunk.end = chunk.start + self._get_consume_time(
                    chunk.virtual_pp_rank, chunk.acc_step, chunk.chunk_type
                )
                undone_micro_step[stage_index] += 1
                if chunk.chunk_type != ChunkType.BUBBLE:
                    num_done_chunks += 1
                # print(f"{stage_index}, {micro_step} : {preorder_chunk} -> {chunk} ")

            stage_index = (stage_index + 1) % self.pp_degree

        if self.enable_batch_send_recv:
            self._barrier()

    def _add_chunk(self, chunk):
        self.schedule_table[chunk.stage_id].append(chunk)

    def _barrier_two_chunk(self, c1, c2):
        assert (
            c1.chunk_type == c2.chunk_type
        ), f"{c1} and {c2} should have the same chunk type"
        c1.start = max(c1.start, c2.start)
        # print(f"{c1} <-- {c2} barrier")

    def _barrier(self):
        # barrier steady phase
        barrier_steps = []
        for stage_id in range(0, self.pp_degree - 1):
            warmup_steps, steady_steps = self._get_warmup_and_steady_steps(stage_id)
            barrier_steps.append(range(warmup_steps, warmup_steps + 2 * steady_steps))

        last_stage_warmup_steps, last_stage_steady_steps = (
            self._get_warmup_and_steady_steps(self.pp_degree - 1)
        )
        for micro_step in range(
            last_stage_warmup_steps,
            last_stage_warmup_steps + 2 * last_stage_steady_steps,
        ):
            barrier_chunk = self.schedule_table[-1][micro_step]
            for stage_id in range(0, self.pp_degree - 1):
                # 标准VPP编排对应的每个stage的step
                target_step = micro_step
                # VPPFhenBInBalancedMemory编排下对应的每个stage的step
                if (
                    self.num_acc_steps >= self.pp_degree
                    and self.num_acc_steps < self.pp_degree * 2
                ):
                    target_step = micro_step - (self.pp_degree - stage_id - 1)
                if target_step in barrier_steps[stage_id]:
                    self._barrier_two_chunk(
                        self.schedule_table[stage_id][target_step], barrier_chunk
                    )

        for stage_id in range(0, self.pp_degree - 1):
            for micro_step, chunk in enumerate(self.schedule_table[stage_id]):
                if micro_step > 0:
                    chunk.start = max(
                        chunk.start, self.schedule_table[stage_id][micro_step - 1].end
                    )
                    chunk.end = chunk.start + self._get_consume_time(
                        chunk.virtual_pp_rank, chunk.acc_step, chunk.chunk_type
                    )

        # barrier cooldown phase
        for barrier_stage_id in range(1, self.pp_degree):
            barrier_warmup_steps, barrier_steady_steps = (
                self._get_warmup_and_steady_steps(barrier_stage_id)
            )
            barrier_micro_steps = (
                barrier_warmup_steps + 2 * barrier_steady_steps - 1
            )  # steady阶段最后一个Backward
            barrier_chunk = self.schedule_table[barrier_stage_id][barrier_micro_steps]
            for stage_id in range(0, barrier_stage_id):
                warmup_steps, steady_steps = self._get_warmup_and_steady_steps(stage_id)
                target_step = (
                    warmup_steps + steady_steps * 2 + (barrier_stage_id - stage_id - 1)
                )
                self._barrier_two_chunk(
                    self.schedule_table[stage_id][target_step], barrier_chunk
                )

        for stage_id in range(0, self.pp_degree - 1):
            for micro_step, chunk in enumerate(self.schedule_table[stage_id]):
                if micro_step > 0:
                    chunk.start = max(
                        chunk.start, self.schedule_table[stage_id][micro_step - 1].end
                    )
                    chunk.end = chunk.start + self._get_consume_time(
                        chunk.virtual_pp_rank, chunk.acc_step, chunk.chunk_type
                    )

    def _get_consume_time(self, virtual_pp_rank, acc_step, chunk_type):
        return (
            1
            if chunk_type == ChunkType.FORWARD or chunk_type == ChunkType.BUBBLE
            else 2
        )

    def _get_preorder_chunk(self, chunk):
        if chunk.chunk_type == ChunkType.BUBBLE:
            return None
        stage_id = chunk.stage_id
        if chunk.chunk_type == ChunkType.FORWARD:
            return (
                None
                if chunk.layer_id == 0
                else self._find_preorder_chunk_from_stage(
                    chunk, (stage_id - 1 + self.pp_degree) % self.pp_degree
                )
            )
        elif chunk.chunk_type == ChunkType.BACKWARD:
            return (
                None
                if chunk.layer_id == self.vpp_degree * self.pp_degree - 1
                else self._find_preorder_chunk_from_stage(
                    chunk, (stage_id + 1) % self.pp_degree
                )
            )
        else:
            raise NotImplementedError

    def _find_preorder_chunk_from_stage(self, chunk, stage_id):
        for preorder_chunk in self.schedule_table[stage_id]:
            if (
                preorder_chunk.chunk_type == chunk.chunk_type
                and preorder_chunk.acc_step == chunk.acc_step
            ):
                if (
                    chunk.chunk_type == ChunkType.FORWARD
                    and chunk.layer_id == preorder_chunk.layer_id + 1
                ) or (
                    chunk.chunk_type == ChunkType.BACKWARD
                    and chunk.layer_id == preorder_chunk.layer_id - 1
                ):
                    return preorder_chunk
        raise ValueError(f"No pre-order chunks found for chunk: {chunk}")

    def _get_warmup_and_steady_steps(self, stage_id):
        # VPPFhenBInBalancedMemory
        if (
            self.num_acc_steps >= self.pp_degree
            and self.num_acc_steps < self.pp_degree * 2
        ):
            warmup_steps = (
                self.num_acc_steps * (self.vpp_degree - 1)
                + self.pp_degree
                - stage_id
                - 1
            )
            steady_steps = self.num_acc_steps - (self.pp_degree - stage_id - 1)
            return warmup_steps, steady_steps

        # PipelineParallelWithInterleave
        warmup_steps = (self.pp_degree - stage_id - 1) * 2
        warmup_steps += (self.vpp_degree - 1) * self.first_chunk_acc
        warmup_steps = min(warmup_steps, self.num_steps)
        steady_steps = self.num_steps - warmup_steps
        return warmup_steps, steady_steps

    def _get_virtual_pp_rank(self, micro_step, forward):
        first_chunk_steps = self.first_chunk_acc * self.vpp_degree
        if micro_step < first_chunk_steps:
            virtual_pp_rank = micro_step // self.first_chunk_acc
        else:
            origin_micro_step = micro_step
            micro_step -= first_chunk_steps
            virtual_pp_rank = micro_step % (self.pp_degree * self.vpp_degree)
            virtual_pp_rank = virtual_pp_rank // self.pp_degree

        if not forward:
            virtual_pp_rank = self.vpp_degree - virtual_pp_rank - 1
        return virtual_pp_rank

    def _schedule_without_bubble(self):
        # Structures to record the micro step for each layer chunk
        forward_micro_step_counter = {}
        backward_micro_step_counter = {}

        for stage_id in range(self.pp_degree):
            for i in range(self.vpp_degree):
                forward_micro_step_counter[i] = 0
                backward_micro_step_counter[i] = 0

            warmup_steps, steady_steps = self._get_warmup_and_steady_steps(stage_id)
            for micro_step in range(warmup_steps):
                virtual_pp_rank = self._get_virtual_pp_rank(micro_step, forward=True)
                real_micro_step = forward_micro_step_counter[virtual_pp_rank]
                forward_micro_step_counter[virtual_pp_rank] += 1
                self._add_chunk(
                    Chunk(
                        virtual_pp_rank=virtual_pp_rank,
                        acc_step=real_micro_step,
                        pp_degree=self.pp_degree,
                        vpp_degree=self.vpp_degree,
                        stage_id=stage_id,
                        chunk_type=ChunkType.FORWARD,
                        start=0,
                        end=0,
                    )
                )

            for micro_step in range(steady_steps):
                forward_micro_step_id = micro_step + warmup_steps
                forward_virtual_pp_rank = self._get_virtual_pp_rank(
                    forward_micro_step_id, forward=True
                )
                backward_micro_step_id = micro_step
                backward_virtual_pp_rank = self._get_virtual_pp_rank(
                    backward_micro_step_id, forward=False
                )

                real_forward_micro_step = forward_micro_step_counter[
                    forward_virtual_pp_rank
                ]
                forward_micro_step_counter[forward_virtual_pp_rank] += 1
                real_backward_micro_step = backward_micro_step_counter[
                    backward_virtual_pp_rank
                ]
                backward_micro_step_counter[backward_virtual_pp_rank] += 1
                barrier_step = micro_step + (self.pp_degree - stage_id - 1)

                self._add_chunk(
                    Chunk(
                        virtual_pp_rank=forward_virtual_pp_rank,
                        acc_step=real_forward_micro_step,
                        pp_degree=self.pp_degree,
                        vpp_degree=self.vpp_degree,
                        stage_id=stage_id,
                        chunk_type=ChunkType.FORWARD,
                        start=0,
                        end=0,
                        barrier_step=barrier_step,
                    )
                )
                self._add_chunk(
                    Chunk(
                        virtual_pp_rank=backward_virtual_pp_rank,
                        acc_step=real_backward_micro_step,
                        pp_degree=self.pp_degree,
                        vpp_degree=self.vpp_degree,
                        stage_id=stage_id,
                        chunk_type=ChunkType.BACKWARD,
                        start=0,
                        end=0,
                        barrier_step=barrier_step,
                    )
                )

            for micro_step in range(steady_steps, self.num_steps):
                virtual_pp_rank = self._get_virtual_pp_rank(micro_step, forward=False)
                real_micro_step = backward_micro_step_counter[virtual_pp_rank]
                backward_micro_step_counter[virtual_pp_rank] += 1
                self._add_chunk(
                    Chunk(
                        virtual_pp_rank=virtual_pp_rank,
                        acc_step=real_micro_step,
                        pp_degree=self.pp_degree,
                        vpp_degree=self.vpp_degree,
                        stage_id=stage_id,
                        chunk_type=ChunkType.BACKWARD,
                        start=0,
                        end=0,
                    )
                )

    def compute_bubble_rate(self):
        """
        计算bubble rate
        """
        if not self._is_scheduled:
            self.schedule()

        if not self.schedule_table:
            return 0.0  # 没有调度的chunk，视为无bubble

        # 计算时间跨度
        min_start = 1e8
        max_end = 0
        sum_exec = 0
        for i in range(self.pp_degree):
            min_start = min(min_start, min(c.start for c in self.schedule_table[i]))
            max_end = max(max_end, max(c.end for c in self.schedule_table[i]))
            sum_exec += sum(
                c.end - c.start
                for c in self.schedule_table[i]
                if c.chunk_type != ChunkType.BUBBLE
            )

        total_time = max_end - min_start
        total_possible_time = self.pp_degree * total_time
        bubble_time = total_possible_time - sum_exec
        bubble_rate = 1.0 * bubble_time / total_possible_time

        return bubble_rate

    def draw_chunks(self):
        """
        绘制vpp编排结果, 保存在pipeline_schedule.png
        """
        if not self._is_scheduled:
            self.schedule()

        fig, ax = plt.subplots(figsize=(12, 6))

        # 选择一个颜色映射，例如'viridis'
        cmap = plt.cm.get_cmap("viridis")

        # 生成颜色字典
        color_map = {i: cmap(i / (self.vpp_degree - 1)) for i in range(self.vpp_degree)}

        max_time = (
            max(chunk.end for stage in self.schedule_table for chunk in stage) + 1
            if self.schedule_table
            else 1
        )

        # 获取figure和axes的尺寸信息，用于计算自适应高度
        fig_width, fig_height = fig.get_size_inches()  # 获取figure的尺寸（英寸）
        ax_pos = ax.get_window_extent()  # 获取axes的位置和尺寸（像素）
        dpi = fig.dpi  # 获取分辨率

        # 设定一个固定的矩形高度比例
        rect_height_ratio = 1 / 20  # 这个比例可以根据需要调整
        stage_height = fig_height * rect_height_ratio / self.pp_degree

        # 绘制每个Chunk
        for stage_idx, chunks in enumerate(self.schedule_table):
            y_pos = (
                self.pp_degree - stage_idx - 1
            ) * stage_height  # Y轴位置对应stage_id，根据stage_height调整
            for chunk in chunks:
                if chunk.chunk_type == ChunkType.BUBBLE:
                    continue
                x = chunk.start
                width = chunk.end - x
                color = color_map[chunk.virtual_pp_rank]

                # 创建矩形，高度设置为stage_height
                rect = patches.Rectangle(
                    (x, y_pos),
                    width,
                    stage_height,
                    linewidth=1,
                    edgecolor="black",
                    facecolor=color,
                )
                ax.add_patch(rect)

                # text = f"{chunk.chunk_type.value}{chunk.layer_id}"
                text = f"{chunk.layer_id}"
                # 动态计算文本位置，确保位于矩形中心
                text_x = x + width / 2
                text_y = y_pos + stage_height / 2
                ax.text(
                    text_x,
                    text_y,
                    text,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )

        # 设置坐标轴
        ax.set_xlim(0, max_time)
        ax.set_ylim(
            0, fig_height * rect_height_ratio
        )  # 根据figure的高度和设定的比例设置y轴范围
        ax.set_yticks(range(self.pp_degree + 1) * stage_height)  # 调整yticks的位置
        ax.set_yticklabels(
            [f"Stage {i}" for i in range(self.pp_degree - 1, -1, -1)] + [""]
        )
        ax.set_xlabel("Time Step")
        ax.set_title("Pipeline Schedule Visualization")
        ax.grid(axis="y", alpha=0.5)

        plt.tight_layout()
        plt.savefig("pipeline_schedule.png")

    def draw_balls(self):
        """
        绘制vpp编排结果，所有处在相同迭代步的层在图中相连，保存到pipeline_balls.png
        """
        if not self._is_scheduled:
            self.schedule()

        fig, ax = plt.subplots()
        ax.set_xlim(-0.5, self.num_acc_steps - 0.5)
        ax.set_ylim(-0.5, self.layer_num - 0.5)
        ax.set_aspect("equal")  # 保持纵横比一致，使小球显示为圆形
        ax.axis("off")  # 隐藏坐标轴

        links = np.array([[-1] * (self.num_acc_steps - 1)] * self.layer_num)
        start_time_to_backward_chunks = {}
        for schedule in self.schedule_table:
            for chunk in schedule:
                if chunk.chunk_type == ChunkType.BACKWARD:
                    if start_time_to_backward_chunks.get(chunk.start, None) is None:
                        start_time_to_backward_chunks[chunk.start] = []
                    start_time_to_backward_chunks[chunk.start].append(chunk)

        for start_time, chunks in start_time_to_backward_chunks.items():
            chunks.sort(key=lambda chunk: chunk.acc_step)
            print(f"start_time: {start_time}, chunks: {chunks}")
            for i in range(len(chunks) - 1):
                chunk = chunks[i]
                next_chunk = chunks[i + 1]
                print(f"chunk: {chunk}, next_chunk: {next_chunk}")
                assert (
                    chunk.acc_step + 1 == next_chunk.acc_step
                ), f"{chunk.acc_step} + 1 != {next_chunk.acc_step}"
                links[chunk.layer_id][chunk.acc_step] = next_chunk.layer_id
                print(f"links: {links}")

        print(f"start_time_to_backward_chunks: {start_time_to_backward_chunks}")
        print(f"links: {links}")

        # 先绘制连接线，确保线在小球下方
        for j in range(self.num_acc_steps - 1):
            for i in range(self.layer_num):
                current_pos = (j, i)
                next_row = links[i][j]
                if next_row == -1:
                    continue
                next_pos = (j + 1, next_row)
                print(f"current_pos: {current_pos}, next_pos: {next_pos}")
                # 绘制线段
                ax.plot(
                    [current_pos[0], next_pos[0]],
                    [current_pos[1], next_pos[1]],
                    color="gray",
                    linewidth=1,
                    zorder=1,
                )

        # 绘制小球，使用zorder确保小球在上层
        for j in range(self.num_acc_steps):
            for i in range(self.layer_num):
                circle = plt.Circle(
                    (j, i), 0.2, edgecolor="black", facecolor="lightblue", zorder=2
                )
                ax.add_patch(circle)

                # 添加行编号文本（L0, L1, ...）

                ax.text(
                    x=j,  # x坐标与小球中心对齐
                    y=i,  # y坐标与小球中心对齐
                    s=f"{i}",  # 显示层号文本
                    ha="center",  # 水平居中
                    va="center",  # 垂直居中
                    fontsize=8,  # 根据小球大小调整字号
                    color="black",  # 确保与背景色对比
                    zorder=3,  # 确保文字在小球上方
                )

        plt.show()
        plt.savefig("balls.png")

    def schedule(self):
        """
        模拟vpp调度，返回编排表schedule_table
        """
        self._schedule_without_bubble()
        self._add_bubble()
        self._is_scheduled = True
        return self.schedule_table
