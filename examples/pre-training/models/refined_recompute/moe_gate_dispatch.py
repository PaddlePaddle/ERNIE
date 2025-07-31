""" refined_recompute for moe_gate_dispatch """

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import queue
import logging
from paddle import framework
from paddle.autograd import PyLayer
from models.refined_recompute.queue_check import global_rr_queue_log
from paddleformers.utils.tools import get_env_device

logger = logging.getLogger(__name__)

if get_env_device() == "xpu":
    try:
        from paddle_xpu_nn import moe_gate_dispatch as xpu_moe_gate_dispatch
    except ImportError:
        xpu_moe_gate_dispatch = None
        logger.warning("`xpu moe dispatch` not found")
else:
    try:
        import moe_ops
    except ImportError:
        moe_ops = None
        logger.warning(
            "`moe-ops` not found, run "
            "`python3  src/ernie_core/ops/moe/setup.py  install` to install"
        )


class MoEGateDispatchFunctor(PyLayer):
    """
    MoEGateDispatchFunctor is a fake layer to make the backward of MoEGateDispatch work.
    """

    @staticmethod
    def forward(ctx, x, prob, compat_args, k, capacity, use_pad, hold_tensors):
        """
        re-compute 中的第二次前向计算，直接获取第一次前向计算的结果作为输出，不再做重计算

        Args:
            ctx : 上下文对象，用于保存反向所需的数据。
            x (paddle.Tensor): 输入张量, dispatch 算子的输入。
            prob (paddle.Tensor): 数据分配到每个专家的概率, dispatch 算子的输入参数。
            compat_args (paddle.Tensor): 兼容性参数, dispatch 算子的输入参数。
            k (int): 每个 token 被分配到的专家数量, dispatch 算子的输入参数。
            capacity (int): 每个专家的最大容量, dispatch 算子的输入参数。
            use_pad (bool): 是否使用 padding, dispatch 算子的输入参数。
            hold_tensors (dict): 保存了第一次前向计算结果的字典。

        Returns:
            tuple: 返回值包含以下元素：
                1. dispatched_input (paddle.Tensor): 经过分派后的输入张量。
                2. combine_weights (paddle.Tensor): 组合权重。
                3. scatter_index (paddle.Tensor): 散列索引。
                4. expert_offset (paddle.Tensor): 专家偏移量。
                5. expert_id (paddle.Tensor): 专家 ID。
        """
        dispatched_input = hold_tensors["dispatched_input"]
        combine_weights = hold_tensors["combine_weights"]
        scatter_index = hold_tensors["scatter_index"]
        expert_offset = hold_tensors["expert_offset"]
        expert_id = hold_tensors["expert_id"]
        ctx.save_for_backward(
            x,
            prob,
            k,
            capacity,
            use_pad,
            dispatched_input,
            combine_weights,
            scatter_index,
            expert_offset,
            expert_id,
        )
        return (
            dispatched_input,
            combine_weights,
            scatter_index,
            expert_offset,
            expert_id,
        )

    @staticmethod
    def backward(ctx, *grads):
        """
        反向计算，调用moe_gate_dispatch_bwd函数计算梯度，并返回梯度, 并释放第一次前向计算的结果。

        Args:
            ctx : 上下文对象，用于保存反向所需的数据。
            grads : 输出梯度张量。

        Returns:
            tuple: 返回两个梯度张量，分别是输入x和分配概率的梯度。
        """
        (
            x,
            prob,
            k,
            capacity,
            use_pad,
            dispatched_input,
            combine_weights,
            scatter_index,
            expert_offset,
            expert_id,
        ) = ctx.saved_tensor()
        y_grad = grads[0]
        combine_weights_grad = grads[1]
        x_grad, gate_logits_grad = moe_ops.moe_gate_dispatch_bwd(
            combine_weights.detach(),
            scatter_index.detach(),
            expert_id.detach(),
            y_grad,
            combine_weights_grad,
            k=k,
            capacity=capacity,
            use_pad=use_pad,
        )

        dispatched_input._clear_dataptr()
        combine_weights._clear_dataptr()
        scatter_index._clear_dataptr()
        expert_id._clear_dataptr()
        expert_offset._clear_dataptr()

        return x_grad, gate_logits_grad


class RefinedRcomputeMoEGateDispatch(object):
    """
    Wrapper class when skipping recompute for MoEGateDispatch op.
    """

    def __init__(self):
        self._hold_tensors_queue = queue.Queue()
        global_rr_queue_log.update(self._hold_tensors_queue, "moe_gate_dispatch")

    def forward(self, x, prob, compat_args, k, capacity, use_pad):
        """
        Function for the first and second forward phase in re-compute.

        Args:
            x (paddle.Tensor): input tensor of moe_gate_dispatch op.
            prob (paddle.Tensor): probability tensor of moe_gate_dispatch op.
            compat_args (list): compatibility arguments of moe_gate_dispatch op.
            k (int): number of experts per token.
            capacity (int): maximum capacity of each expert.
            use_pad (bool): whether to use padding.

        Returns:
            tuple: output tensors of moe_gate_dispatch op.
        """

        assert k is not None, "k should not be None"
        assert capacity is not None, "capacity should not be None"
        assert use_pad is not None, "use_pad should not be None"
        if not framework._dygraph_tracer()._has_grad:
            (
                dispatched_input,
                combine_weights,
                scatter_index,
                expert_offset,
                expert_id,
            ) = self._first_fwd(x, prob, compat_args, k, capacity, use_pad)
        else:
            assert not self._hold_tensors_queue.empty(), "queue should not be empty"
            (
                dispatched_input,
                combine_weights,
                scatter_index,
                expert_offset,
                expert_id,
            ) = self._second_fwd(
                x, prob, compat_args, k=k, capacity=capacity, use_pad=use_pad
            )

        return (
            dispatched_input,
            combine_weights,
            scatter_index,
            expert_offset,
            expert_id,
        )

    @paddle.no_grad()
    def _first_fwd(self, x, prob, compat_args, k, capacity, use_pad):

        dispatched_input, combine_weights, scatter_index, expert_offset, expert_id = (
            moe_ops.moe_gate_dispatch(
                x, prob, *compat_args, k=k, capacity=capacity, use_pad=use_pad
            )
        )

        hold_tensors = {
            "dispatched_input": dispatched_input,
            "combine_weights": combine_weights,
            "scatter_index": scatter_index,
            "expert_offset": expert_offset,
            "expert_id": expert_id,
        }

        self._hold_tensors_queue.put(hold_tensors)
        return (
            dispatched_input,
            combine_weights,
            scatter_index,
            expert_offset,
            expert_id,
        )

    def _second_fwd(self, x, prob, compat_args, k, capacity, use_pad):
        hold_tensors = self._hold_tensors_queue.get()
        return MoEGateDispatchFunctor.apply(
            x, prob, compat_args, k, capacity, use_pad, hold_tensors
        )

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
