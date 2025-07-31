""" refined_recompute for moe_combine """

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
    try:
        import moe_combine
    except ImportError:
        moe_combine = None
        logger.warning(
            "`moe-combine` not found, run "
            "`python3  src/ernie_core/ops/moe/setup.py  install` to install"
        )


class MoECombineFunctor(PyLayer):
    """
    MoECombineFunctor is a fake layer to make the backward of MoECombine work.
    """

    @staticmethod
    def forward(ctx, x, combine_weights, scatter_index, hold_tensors):
        """
        直接获取第一次前向计算的结果作为输出，不再做重计算

        Args:
            ctx : 上下文对象，用于保存反向所需的数据。
            x (paddle.Tensor): 输入张量, combine 算子的输入。
            combine_weights (paddle.Tensor): 组合权重张量, combine 算子的输入参数。
            scatter_index (paddle.Tensor): 索引张量, combine 算子的输入参数。
            hold_tensors (dict): 包含第一次前向计算结果的字典。

        Returns:
            paddle.Tensor: 输出张量。

        """
        combined_out = hold_tensors["combined_out"].detach()
        ctx.save_for_backward(x, combine_weights, scatter_index, combined_out)
        return combined_out

    @staticmethod
    def backward(ctx, grad_y, *_):
        """
        反向计算，调用moe_combine_bwd函数计算梯度，并返回梯度, 并释放第一次前向计算的结果。

        Args:
            ctx : 上下文对象，用于保存反向所需的数据。
            grad_y (paddle.Tensor): 输出梯度张量。
            *_ : 其他梯度，不使用。

        Returns:
            tuple: 返回两个梯度张量，分别是输入x和组合权重的梯度。
        """
        x, combine_weights, scatter_index, combined_out = ctx.saved_tensor()

        grad_x, grad_combine_weight_helper = moe_combine.moe_combine_bwd(
            x, combine_weights, scatter_index, grad_y
        )
        grad_combine_weight = grad_combine_weight_helper.sum(-1)
        grad_combine_weight = grad_combine_weight.reshape(combine_weights.shape)

        # release memory
        combined_out._clear_dataptr()

        return grad_x, grad_combine_weight, None


class RefinedRcomputeMoECombine(object):
    """
    Wrapper class when skipping recompute for MoECombine op.
    """

    def __init__(self):
        self._hold_tensors_queue = queue.Queue()
        global_rr_queue_log.update(self._hold_tensors_queue, "moe_combine")

    def forward(self, expert_output, combine_weights, scatter_index):
        """
        Function for the first and second forward phase in re-compute.

        Args:
            expert_output (paddle.Tensor): 1st input arg of moe_combine op.
            combine_weights (paddle.Tensor): 2nd input arg of moe_combine op.
            scatter_index (paddle.Tensor): 3rd input arg of moe_combine op.

        Returns:
            paddle.Tensor: output tensor of moe_combine op.
        """

        if not framework._dygraph_tracer()._has_grad:
            combined_output = self._first_fwd(
                expert_output, combine_weights, scatter_index
            )
        else:
            assert not self._hold_tensors_queue.empty(), "queue should not be empty"
            combined_output = self._second_fwd(
                expert_output, combine_weights, scatter_index
            )

        return combined_output

    @paddle.no_grad()
    def _first_fwd(self, expert_output, combine_weights, scatter_index):

        combined_out = moe_combine.moe_combine(
            expert_output, combine_weights, scatter_index
        )

        hold_tensors = {
            "combined_out": combined_out,
        }

        self._hold_tensors_queue.put(hold_tensors)
        return combined_out

    def _second_fwd(self, expert_output, combine_weights, scatter_index):
        hold_tensors = self._hold_tensors_queue.get()
        return MoECombineFunctor.apply(
            expert_output, combine_weights, scatter_index, hold_tensors
        )

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
