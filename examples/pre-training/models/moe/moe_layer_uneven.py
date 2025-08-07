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

from paddleformers.utils.tools import get_env_device

from typing import Any, Tuple, List
import logging
import inspect
from collections import namedtuple

import numpy as np
import paddle
from paddle import nn
from paddle.distributed.communication import stream

from paddle.autograd import PyLayer
from paddle.distributed.communication.group import Group
from paddle.distributed.fleet.utils import recompute
import paddle.distributed as dist
from paddle import Tensor
from paddle.nn import functional as F
from paddleformers.trainer.plugins.timer import get_timers
from paddle.distributed import fleet


# from models.moe.moe_layer import _AllToAll
from models.moe.top2_gate import TopKGateFused
from models.moe.sinkhorn_gate import SinkHornGateFused
from models.moe.round_robin_gate import RoundRobinGateFused

try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}

logger = logging.getLogger(__name__)

GateOutput = namedtuple(
    "GateOutput",
    [
        "aux",
        "z",
        "logits",
    ],
)


if get_env_device() == "xpu":
    try:
        from paddle_xpu_nn import moe_combine as xpu_moe_combine
        from paddle_xpu_nn import moe_combine_bwd as xpu_moe_combine_bwd
    except ImportError:
        xpu_moe_combine = None
        xpu_moe_combine_bwd = None
        logger.warning("`xpu moe combine` not found")
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


try:
    import moe_ops_no_softmaxtopk
except ImportError:
    moe_ops_no_softmaxtopk = None
    logger.warning(
        "moe-ops-no-softmaxtopk` not found, run "
        "`python3  src/ernie_core/ops/moe/setup.py  install` to install"
    )


def average_grad(x, y, dy, eps=1e-12):

    s, k = x.shape
    xsum = x.sum(axis=-1, keepdim=True)  # [s,1]
    maskpos = (xsum == 0.0).expand_as(x)

    xsum_square = xsum.square()  # [s,1]
    left = paddle.triu(
        paddle.tril((1 / xsum).unsqueeze(-1).expand([s, k, k]))
    )  # aka diag-emb [s,k,k]
    right = (-x / xsum_square).unsqueeze(-1).expand([s, k, k])
    dydx = left + right
    dx = paddle.matmul(dy.unsqueeze(-2).cast(dydx.dtype), dydx).squeeze(
        -2
    )  # [s,1,k] @[s,k,k] -> [s,1,k]
    dx = paddle.where(maskpos, paddle.zeros_like(dx), dx)
    return dx


mask = paddle.to_tensor(
    [
        [1, -1],
        [-1, 1],
    ]
).unsqueeze(0)


def average_grad_bi(x, y, dy, eps=1e-12):

    s, k = x.shape
    assert k == 2, k
    xsum = paddle.clip(x.sum(axis=-1, keepdim=True), min=eps)  # [s,1]
    dydx = (
        x.flip(axis=1).unsqueeze(-2).tile([1, 2, 1])
        * mask.cast(x.dtype)
        / xsum.square().unsqueeze(-1)
    )
    dx = paddle.matmul(dy.unsqueeze(-2).cast(dydx.dtype), dydx).squeeze(
        -2
    )  # [s,1,k] @[s,k,k] -> [s,1,k]
    return dx


def topk_grad(x, dy, indicies):

    s, e = x.shape
    _, k = dy.shape
    dx = paddle.scatter_nd(
        paddle.stack(
            [
                paddle.arange(s).repeat_interleave(k).cast(indicies.dtype),
                indicies.reshape([-1]),
            ],
            -1,
        ),
        dy.reshape([-1]),
        shape=[s, e],
    )  # [s,k] -> [s,e]
    return dx  # dx 保持高精度


class GateDispatch(PyLayer):

    @staticmethod
    def forward(ctx, x, gate_prob, k, capacity, use_pad, eps=1e-12):

        ctx.k = k
        ctx.eps = eps
        ctx.capacity = capacity
        ctx.gate_prob = gate_prob
        if "corr_bias" in inspect.signature(moe_ops.moe_gate_dispatch).parameters:
            compat_args = (None,)
        else:
            compat_args = ()
        y, combine_weights, scatter_index, expert_offset, expert_id = (
            moe_ops.moe_gate_dispatch(
                x, gate_prob, *compat_args, k=k, capacity=capacity, use_pad=use_pad
            )
        )
        ctx.combine_weights = combine_weights
        scatter_index = scatter_index.transpose([1, 0])  # [k,s] ->[s,k]
        ctx.scatter_index = scatter_index
        ctx.expert_id = expert_id
        num_experts = gate_prob.shape[-1]

        ctx.num_experts = num_experts
        ctx.seqlen = gate_prob.shape[0]

        return y, combine_weights, scatter_index, expert_offset, expert_id

    @staticmethod
    def backward(ctx, dy, dw, *_):

        s, k = ctx.combine_weights.shape
        grad = F.embedding(ctx.scatter_index, dy)  # [s, k,d]
        mask = (ctx.combine_weights > 0.0).astype(grad.dtype)  # [s,k]
        dx = paddle.matmul(mask.unsqueeze(1), grad).squeeze(
            1
        )  # [s,1,k] @ [s,k,d] -> [s,1,d]
        if ctx.gate_prob.stop_gradient:
            return dx, None

        combine_weights_unnorm = ctx.combine_weights
        dw = dw.astype(combine_weights_unnorm.dtype)
        d_prob = topk_grad(ctx.gate_prob, dw, ctx.expert_id)
        return dx, d_prob


class HardGateDispatch(PyLayer):

    @staticmethod
    def forward(ctx, x, expert_id, k, capacity, num_experts, eps=1e-12):

        ctx.k = k
        assert k == 1, f"k must be 1 in hard gate, got k={k}"
        ctx.eps = eps
        ctx.capacity = capacity
        y, scatter_index, expert_offset = (
            moe_ops_no_softmaxtopk.moe_gate_dispatch_no_softmax_topk(
                x, expert_id, k, capacity, num_experts, False
            )
        )
        scatter_index = scatter_index.transpose([1, 0])  # [k,s] ->[s,k]
        ctx.scatter_index = scatter_index
        ctx.expert_id = expert_id
        ctx.num_experts = num_experts

        return y, scatter_index, expert_offset

    @staticmethod
    def backward(ctx, dy, dw, *_):
        grad = F.embedding(ctx.scatter_index, dy)  # [s, k,d]
        assert grad.shape[1] == 1, f"k must == 1, got {grad.shape}"
        dx = grad
        return dx, None


class GateCombine(PyLayer):
    """GateCombine"""

    @staticmethod
    def forward(ctx, x, combine_weights, scatter_index):
        """
        Input:
            x:  [seqlen * k, hidden_size]
            combine_weights: [seqlen, k]
            scatter_index: [seqlen, k]
        Output:
            y: [seqlen, hidden_size]
        """
        ctx.x = x
        ctx.combine_weights = combine_weights
        ctx.scatter_index = scatter_index
        if get_env_device() == "xpu":
            assert xpu_moe_combine is not None
            return xpu_moe_combine(x, combine_weights, scatter_index)
        else:
            assert moe_combine is not None
            ret = moe_combine.moe_combine(x, combine_weights, scatter_index)
            return ret

    @staticmethod
    def backward(ctx, grad_y, *_):
        """
        Input:
            grad_y:  [seqlen, hidden_size]
            combine_weights: [seqlen, k]
            scatter_index: [seqlen, k]
        Output:
            grad_x: [seqlen * k, hidden_size]
            grad_combine_weight: [seqlen, k]

        """

        if get_env_device() == "xpu":
            assert xpu_moe_combine_bwd is not None
            grad_x, grad_combine_weight_helper = xpu_moe_combine_bwd(
                ctx.x, ctx.combine_weights, ctx.scatter_index, grad_y
            )
        else:
            assert moe_combine is not None
            grad_x, grad_combine_weight_helper = moe_combine.moe_combine_bwd(
                ctx.x, ctx.combine_weights, ctx.scatter_index, grad_y
            )

        grad_combine_weight = grad_combine_weight_helper.sum(-1)
        return grad_x, grad_combine_weight.reshape(ctx.combine_weights.shape), None


def rr_combining_wrapper(x, combine_weights, scatter_index, rr_func, hard_gate=False):
    """
    Args:
        x: Tensor[seq, dim]
        combine_weights: [s, k]
        scatter_index:  ** [k, s] **
        rr_func: refined recompute function for moe_combine
    Returns:
        y: Tensor[s, dim]
    """
    if hard_gate:
        x_gatherd = F.embedding(scatter_index, x)  # [s,k,dim]
        return x_gatherd.squeeze(-2)
    ret = rr_func(x, combine_weights, scatter_index)
    ret.stop_gradient = False
    return ret


def combining(x, combine_weights, scatter_index, hard_gate=False):
    """
    Args:
        x: Tensor[seq, dim]
        combine_weights: [s, k]
        scatter_index:  ** [k, s] **

    Returns:
        y: Tensor[s, dim]
    """
    if hard_gate:
        x_gatherd = F.embedding(scatter_index, x)  # [s,k,dim]
        return x_gatherd.squeeze(-2)
    ret = GateCombine.apply(x, combine_weights, scatter_index)
    ret.stop_gradient = False
    return ret


class _AllToAllSized(PyLayer):

    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        input_indices: Tensor,
        group: Group,
    ) -> Tensor:  # type: ignore

        ctx.group = group
        assert input.ndim == 2, input.shape

        if get_timers() is not None:
            get_timers()("moe-all2all").start()
        input_indices = input_indices.astype("int64")

        world_size = dist.get_world_size(group)
        expert_per_device = len(input_indices) // world_size
        input_indices = input_indices.reshape([-1, expert_per_device])

        output_indices = paddle.zeros_like(input_indices)
        stream.alltoall_single(
            output_indices, input_indices, None, None, group, True, True
        )

        input_indices.stop_gradient = True
        output_indices.stop_gradient = True
        ctx.input_indices = input_indices
        ctx.output_indices = output_indices

        output = paddle.empty(
            [output_indices.sum().item()] + input.shape[1:], dtype=input.dtype
        )

        stream.alltoall_single(
            output,
            input,
            output_indices.sum(-1).tolist(),
            input_indices.sum(-1).tolist(),
            group,
            True,
            True,
        )
        if get_timers() is not None:
            get_timers()("moe-all2all").stop()
        return output, output_indices.reshape([-1]).cast("float32")

    @staticmethod
    def backward(ctx: Any, dy: Tensor, _) -> Tuple[Tensor]:

        dx, _ = _AllToAllSized.apply(
            dy,
            ctx.output_indices.reshape([-1]),
            ctx.group,
        )

        return dx, None


class MOELayer(nn.Layer):

    def __init__(
        self,
        gate: nn.Layer,
        experts: List[nn.Layer],
        layer_idx,
        group: Group = None,
        recompute=False,
        enable_logging=False,
    ) -> None:

        super().__init__()
        self.gate = gate
        if gate is not None:
            assert isinstance(
                gate, (TopKGateFused, SinkHornGateFused, RoundRobinGateFused)
            ), type(gate)
            for p in self.gate.parameters():
                p.is_gate = True
        self.recompute = recompute
        self.layer_idx = layer_idx
        self.enable_logging = enable_logging

        logger.info(f"using moe recompute={recompute}")

        if isinstance(experts, nn.LayerList):
            self.experts = experts
        else:
            self.experts = nn.LayerList([experts])
        self.group = group
        is_mp_moe = (
            hasattr(fleet.fleet, "_hcg")
            and group is fleet.get_hybrid_communicate_group().get_model_parallel_group()
        )
        is_dummy_moe = dist.get_world_size(group) == 1

        for p in experts.parameters():
            p.expert = not (is_mp_moe or is_dummy_moe)  # type: ignore
            p.no_sync = not (is_mp_moe or is_dummy_moe)
            if is_mp_moe or is_mp_moe:
                p.is_distributed = True
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.num_local_experts = len(self.experts)
        self.use_hard_gate = gate is None

    def forward(
        self,
        input: Tensor,
        token_type_ids=None,
    ) -> Tensor:
        """
        Args:
            input: Tensor, shape [batch, s, d]

        Returns:
            combined_output: Tensor, shape [batch, s, d]
            combine_weights: Tensor, shape [batch, 2*n, n+1]
            l_aux: float
            l_zloss: float
        """
        if (
            self.use_hard_gate
            and (not self.training)
            and paddle.all(token_type_ids == 0).numpy()
        ):
            return (
                self.experts[0](input),
                None,
                paddle.zeros([1], dtype="float32"),
                None,
            )

        if input.ndim == 3:
            orig_shape = input.shape
            input = input.reshape([-1, input.shape[-1]])
        else:
            orig_shape = None
        assert (
            len(input.shape) == 2
        ), f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"

        seqlen, d_model = input.shape

        def fwdfn(dispatched_input, expert_count_out):

            expert_outputs = []
            expert_count_out = expert_count_out.astype("int64")
            assert (len(expert_count_out) // self.world_size) == len(self.experts), (
                expert_count_out,
                len(self.experts),
            )
            # [worldsize, local_experts]
            expert_offset_r = expert_count_out.cumsum(0).numpy()
            expert_offset_l = np.pad(expert_offset_r, (1, 0), constant_values=0)[:-1]
            expert_offset = np.stack([expert_offset_l, expert_offset_r], axis=-1)

            expert_offset = np.transpose(
                expert_offset.reshape([self.world_size, len(self.experts), 2]),
                [1, 0, 2],
            )

            for iexpert, (expert, off_list) in enumerate(
                zip(self.experts, expert_offset)
            ):
                ins = []
                ins_offset, cnt = [], 0
                for left, right in off_list:
                    ins.append(dispatched_input[left:right])
                    ins_offset.append((cnt, cnt + (right - left)))
                    cnt += right - left
                if not sum(map(len, ins)):
                    expert_outputs.append([None for _ in ins_offset])
                    continue

                ins = paddle.concat(ins, 0)
                out = expert(ins)
                out = [out[left:right] for left, right in ins_offset]
                expert_outputs.append(out)
            expert_outputs = list(zip(*expert_outputs))  # transpose
            expert_outputs = [
                o for outputs in expert_outputs for o in outputs if o is not None
            ]

            if not expert_outputs:
                expert_outputs = paddle.zeros_like(dispatched_input)
                expert_outputs.stop_gradient = False
            else:
                expert_outputs = paddle.concat(expert_outputs, 0)

            return expert_outputs

        if get_timers() is not None:
            get_timers()("moe-gate").start()
        gate_args = ()
        if self.use_hard_gate:
            assert token_type_ids is not None

        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape([-1])
            gate_args = (token_type_ids,)

        if self.use_hard_gate:
            logits, capacity, routerloss = (
                None,
                None,
                paddle.zeros([1], dtype="float32"),
            )
        else:
            logits, capacity, routerloss = self.gate(input, *gate_args)

        # capacity no use
        if self.use_hard_gate:
            dispatched_input, scatter_index, expert_offset = HardGateDispatch.apply(
                input,
                token_type_ids.astype("int32").unsqueeze(-1),
                k=1,
                capacity=input.shape[0],
                num_experts=len(self.experts),
            )
            combine_weights = None
            routerloss = paddle.zeros([1], dtype="float32")
        else:
            if get_timers() is not None:
                get_timers()("moe-gate").stop()
            k = 1 if isinstance(self.gate, SinkHornGateFused) else 2
            gates = F.softmax(logits, axis=-1)
            (
                dispatched_input,
                combine_weights,
                scatter_index,
                expert_offset,
                expert_id,
            ) = GateDispatch.apply(
                input,
                gates,
                k=k,
                capacity=capacity,
                use_pad=False,
            )

            if get_timers() is not None:
                get_timers()("moe-gate").start()
            gates = F.softmax(logits, axis=-1)
            expert_mask = expert_id[:, 0]
            valid_pos = expert_mask < self.gate.num_experts
            expert_mask[~valid_pos] = 0
            mask1 = F.one_hot(
                expert_mask, num_classes=self.gate.num_experts
            ) * valid_pos.astype("float32").unsqueeze(-1)
            l_aux = self.gate._cal_aux_loss(gates, mask1)
            routerloss = routerloss + self.gate.config.moe_aux_loss_lambda * l_aux
            if self.enable_logging:
                _log = {
                    f"aux_loss_layer_{self.layer_idx}": l_aux.item(),
                }
                global_training_logs.update(
                    **_log,
                    **{
                        k.replace(f"_layer_{self.layer_idx}", ""): v
                        for k, v in _log.items()
                    },
                )
            if get_timers() is not None:
                get_timers()("moe-gate").stop()

        expert_offset_pad = F.pad(expert_offset, [1, 0], value=0)
        expert_count = (
            paddle.diff(expert_offset_pad).astype(expert_offset.dtype).detach()
        )

        if self.world_size > 1:
            dispatched_input, expert_count_out = _AllToAllSized.apply(
                dispatched_input, expert_count, self.group
            )  # [ecm]
        else:
            expert_count_out = expert_count

        expert_output = (
            recompute(fwdfn, dispatched_input, expert_count_out)
            if self.recompute and self.training
            else fwdfn(dispatched_input, expert_count_out)
        )

        if self.world_size > 1:
            expert_output, expert_count_rec = _AllToAllSized.apply(
                expert_output, expert_count_out, self.group
            )
        combined_output = combining(
            expert_output, combine_weights, scatter_index, hard_gate=self.use_hard_gate
        )
        if orig_shape:
            combined_output = combined_output.reshape(
                orig_shape[:-1] + [combined_output.shape[-1]]
            )
        return combined_output, combine_weights, routerloss, logits
