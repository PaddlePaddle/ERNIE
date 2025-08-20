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

import logging
from typing import Any, Callable, List

import paddle
from paddle import framework

logger = logging.getLogger(__name__)

try:
    import moe_permutation

except ImportError:
    moe_permutation = None
    logger.warning("moe_permutation is not installed.")


def detach_and_requires_grad_(*args):
    ret = [a.detach() if a is not None else None for a in args]
    for r, a in zip(ret, args):
        if a is not None:
            r.stop_gradient = a.stop_gradient
    return ret


class FakeClone(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input):
        if input.is_contiguous():
            fake_output = paddle.empty_like(input)
            input._share_buffer_to(fake_output)
        else:
            fake_output = input.clone()
        return fake_output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def manual_backward(f: Callable, is_first_fwd: bool, *args: List[Any]):
    tracer = framework._dygraph_tracer()
    orig = tracer._has_grad
    if not is_first_fwd:
        tracer._has_grad = True

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

    out_cached = [FakeClone.apply(o) for o in out if o is not None]

    for o in out_cached:
        o._clear_dataptr()
    tracer._has_grad = orig

    def bwd_f(*grad):
        nonlocal out_cached, detached_args, f
        grad = list(grad)
        grad = [g for g in grad if g is not None]
        assert grad and out_cached, (len(grad), len(out_cached))
        grad, out_cached = zip(
            *[(g, o) for g, o in zip(grad, out_cached) if not o.stop_gradient]
        )

        assert len(grad) == len(out_cached), (len(grad), len(out_cached), f)
        paddle.autograd.backward(out_cached, grad)
        return tuple([t.grad for t in detached_args if t is not None])

    return bwd_f, out
