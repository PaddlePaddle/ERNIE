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
Utility
"""
import os
from typing import Any, List, Callable
import logging

import paddle
from paddle import framework

logger = logging.getLogger(__name__)

try:
    import moe_permutation

except ImportError:
    moe_permutation = None
    logger.warning("moe_permutation is not installed.")

try:
    from paddle import scatter_add_
except ImportError:
    scatter_add_ = None

g_has_print_recovery_log = False


def has_recovered():
    """has recovered"""
    recover_step = os.getenv("RECOVER_STEP")
    if recover_step is None:
        return True
    recover_step = int(recover_step)
    current_step = int(os.getenv("TRAINER_GLOBAL_STEP"))
    if current_step > recover_step:
        global g_has_print_recovery_log
        if not g_has_print_recovery_log:
            logger.info(f"Recovery would be enabled in the step {current_step}")
            g_has_print_recovery_log = True
        return True
    else:
        return False


def get_global_training_logs():
    """
    全局方法，获取 global-training-logs 单例，支持在 erniebot 和 RL new-infra 中使用
    """
    try:
        from src.utils.misc import global_training_logs

        return global_training_logs
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from rl.utils.stat_utils import global_training_logs

        return global_training_logs
    except (ImportError, ModuleNotFoundError):
        pass
    return {}


def global_training_logs_enabled():
    """
    global_training_logs_enabled
    """
    global_training_logs = get_global_training_logs()
    return isinstance(global_training_logs, dict) or global_training_logs.is_enabled()


def inplace_offload(tensor):
    """
    inplace offload tensor
    """
    tmp = tensor.pin_memory() if paddle.is_compiled_with_cuda() else tensor.cpu()
    tmp._share_buffer_to(tensor)


def detach_and_requires_grad_(*args):
    """detach_and_requires_grad_"""
    ret = [a.detach() if a is not None else None for a in args]
    for r, a in zip(ret, args):
        if a is not None:
            r.stop_gradient = a.stop_gradient
    return ret


class FakeClone(paddle.autograd.PyLayer):
    """
    manual_backward中, 为了保留局部的计算图做临时反向计算
    需要把manual_backward的output给clone出来, 这个clone
    本质上不需要output的值, 而是需要拿到output身上的计算图

    但调用paddle.clone会做一次额外的数据拷贝, 这是没必要的
    FakeClone可以免去这个数据拷贝, 实现摘取计算图的目的
    """

    @staticmethod
    def forward(ctx, input):
        """forward"""
        if input.is_contiguous():
            fake_output = paddle.empty_like(input)
            input._share_buffer_to(fake_output)
        else:
            fake_output = input.clone()
        return fake_output

    @staticmethod
    def backward(ctx, grad_output):
        """backward"""
        return grad_output


def manual_backward(f: Callable, is_first_fwd: bool, *args: List[Any]):
    """
    Args:
        f(callable)
        args(*Any)
    Returns
        bw_f(callable): manual backward fn
        out(List[Tensor]): output of f(*args)
    """
    tracer = framework._dygraph_tracer()
    orig = tracer._has_grad
    if not is_first_fwd:
        tracer._has_grad = True  # turn on grad trace so we can manual backward

    detached_args = detach_and_requires_grad_(*args)
    detached_args_clone = [
        FakeClone.apply(a) if a is not None else None for a in detached_args
    ]
    out = f(*detached_args_clone)
    if isinstance(out, list):
        out = tuple(out)
    elif not isinstance(out, tuple):
        out = (out,)

    if is_first_fwd:
        tracer._has_grad = orig
        return None, out

    out_cached = [
        FakeClone.apply(o) for o in out if o is not None
    ]  # do not cache stop_gradient output

    for o in out_cached:
        o._clear_dataptr()  # free mem
    tracer._has_grad = orig

    def bwd_f(*grad):
        nonlocal out_cached, detached_args, f
        grad = list(grad)
        grad = [g for g in grad if g is not None]
        assert grad and out_cached, (len(grad), len(out_cached))
        # out 中的 stop_graident 参数，也会收到 gradient，在这里过滤掉
        grad, out_cached = zip(
            *[(g, o) for g, o in zip(grad, out_cached) if not o.stop_gradient]
        )

        assert len(grad) == len(out_cached), (len(grad), len(out_cached), f)
        # out, grad = zip(*[(o, g) for o, g in zip(out, grad) if g is not None])
        paddle.autograd.backward(out_cached, grad)
        return tuple([t.grad for t in detached_args if t is not None])

    return bwd_f, out


class FakeGather(paddle.autograd.PyLayer):
    """
    临时绕开gather 0size索引的coredump问题
    """

    @staticmethod
    def forward(ctx, input, indices):
        """forward"""
        assert len(indices.shape) == 1
        ctx.save_for_backward(indices, input.shape)
        if indices.shape[0] == 0:
            out_shape = input.shape
            out_shape[0] = 0
            return paddle.zeros(out_shape, dtype=input.dtype)
        return paddle.index_select(input, axis=0, index=indices)

    @staticmethod
    def backward(ctx, grad_output):
        """backward"""
        indices, input_shape = ctx.saved_tensor()

        grad_input = paddle.zeros(input_shape, dtype=grad_output.dtype)
        if indices.shape[0] != 0:
            if scatter_add_ is not None:
                scatter_add_(grad_input, indices.unsqueeze(-1), grad_output)
            else:
                paddle.scatter_(
                    grad_input, indices.unsqueeze(-1), grad_output, overwrite=False
                )
        return grad_input, None


class FusedUnpermutation(paddle.autograd.PyLayer):
    """FusedUnpermutation"""

    @staticmethod
    def forward(
        ctx,
        output_tokens,  # inplace mutable
        permuted_tokens,
        token_permuted_indices,
        dispatched_probs,
        prob_permuted_indices,
    ):
        """forward"""
        # NOTE(zhangyuqin): 确保在调用FusedUnpermutation之前, output_tokens没被任何算子调用过,
        # 否则可能会因为output_tokens被修改而导致梯度错误。case可见:
        # https://docs.pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.mark_dirty.html
        #
        # 原本的操作是inplace, 计算图如下:
        #      t1   t2   t3
        #        \  |  /
        #      output_tokens
        # 但由于pylayer不支持inplace, 自定义算子的计算图也有问题。
        # 所以现在通过trick将计算图变为如下, 但实际图中的output_tokens是同块物理内存:
        #         t1   output_tokens
        #           \    /
        #    t2   output_tokens
        #      \    /
        #  t3   output_tokens
        #   \  /
        #  output_tokens
        assert (
            token_permuted_indices.stop_gradient
        ), "token_permuted_indices must be stop_gradient"
        if dispatched_probs is not None:
            assert (
                prob_permuted_indices is not None
                and prob_permuted_indices.stop_gradient
            ), "dispatched_probs must be stop_gradient"

        output_tokens.stop_gradient = False

        src_token_num = permuted_tokens.shape[0]
        if src_token_num > 0:
            output_tokens = moe_permutation.unpermute(
                output_tokens,  # inplace mutable
                permuted_tokens,
                token_permuted_indices,
                dispatched_probs,
                prob_permuted_indices,
            )
        else:
            output_tokens = FakeClone.apply(output_tokens)

        ctx.save_for_backward(
            permuted_tokens,
            token_permuted_indices,
            dispatched_probs,
            prob_permuted_indices,
        )
        return output_tokens

    @staticmethod
    def backward(ctx, output_tokens_grad):
        """backward"""
        (
            permuted_tokens,
            token_permuted_indices,
            dispatched_probs,
            prob_permuted_indices,
        ) = ctx.saved_tensor()

        src_token_num = permuted_tokens.shape[0]
        if src_token_num > 0:
            permuted_tokens_grad, dispatched_probs_grad = (
                moe_permutation.unpermute_grad(
                    output_tokens_grad,  # const
                    permuted_tokens,
                    token_permuted_indices,
                    dispatched_probs,
                    prob_permuted_indices,
                )
            )
        else:
            permuted_tokens_grad = paddle.zeros_like(permuted_tokens)
            if dispatched_probs is not None:
                dispatched_probs_grad = paddle.zeros_like(dispatched_probs)

        if dispatched_probs is None:
            return output_tokens_grad, permuted_tokens_grad, None
        else:
            return (
                output_tokens_grad,
                permuted_tokens_grad,
                None,
                dispatched_probs_grad,
                None,
            )
