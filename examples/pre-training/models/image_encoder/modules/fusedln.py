# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""
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

# import sys
# your should add the path to sys.path if the root isn't in the system ENV "PATH"
# sys.path.insert(
#     0,
#     '/usr/local/lib/python3.7/site-packages/fast_ln-0.0.0-py3.7-linux-x86_64.egg/'
# )
# sys.path.insert(
#     0,
#     '/usr/local/lib/python3.7/site-packages/fused_ln-0.0.0-py3.7-linux-x86_64.egg/'
# )
"""
import distutils.util
import importlib
import os

import paddle
from paddle import _C_ops

OriginLayerNorm = paddle.nn.LayerNorm
origin_linear = paddle.incubate.nn.functional.fused_linear


def try_import(module_name, func_name=None):
    """_summary_

    Args:
        module_name (_type_): _description_
        func_name (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if func_name is None:
        func_name = module_name
    try:
        m = importlib.import_module(module_name)
        # return m
        return getattr(m, func_name)
    except ImportError:
        print("import module error: {}".format(module_name))
        return None


fast_ln = try_import("fast_ln")
fused_ln = try_import("fused_ln")


def check_normalized_shape(normalized_shape):
    """_summary_

    Args:
        normalized_shape (_type_): _description_
    """
    if isinstance(normalized_shape, (list, tuple)):
        assert len(normalized_shape) == 1


class FusedLayerNorm(OriginLayerNorm):
    """_summary_

    Args:
        OriginLayerNorm (_type_): _description_
    """

    def __init__(
        self,
        normalized_shape,
        epsilon=1e-05,
        weight_attr=None,
        bias_attr=None,
        name=None,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            epsilon=epsilon,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
        )
        check_normalized_shape(self._normalized_shape)

    def forward(self, input):
        """_summary_

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        return fused_ln(input, self.weight, self.bias, self._epsilon)[0]


class FastLayerNorm(OriginLayerNorm):
    """_summary_

    Args:
        OriginLayerNorm (_type_): _description_
    """

    def __init__(
        self,
        normalized_shape,
        epsilon=1e-05,
        weight_attr=None,
        bias_attr=None,
        name=None,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            epsilon=epsilon,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
        )
        check_normalized_shape(self._normalized_shape)

    def forward(self, input):
        """_summary_

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        return fast_ln(input, self.weight, self.bias, self._epsilon)[0]


class FusedLinearWithGradAdd(paddle.autograd.PyLayer):
    """_summary_

    Args:
        paddle (_type_): _description_

    Returns:
        _type_: _description_
    """

    @staticmethod
    def forward(ctx, x, weight, bias=None, name=None):
        """_summary_

        Args:
            ctx (_type_): _description_
            x (_type_): _description_
            weight (_type_): _description_
            bias (_type_, optional): _description_. Defaults to None.
            name (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        y = origin_linear(x, weight, bias)
        ctx.save_for_backward(x, weight, bias)
        return y

    @staticmethod
    def backward(ctx, y_grad):
        """_summary_

        Args:
            ctx (_type_): _description_
            y_grad (_type_): _description_

        Returns:
            _type_: _description_
        """
        x, weight, bias = ctx.saved_tensor()
        x_grad = paddle.matmul(y_grad, weight, transpose_y=True)

        if bias is None:
            if hasattr(weight, "main_grad"):
                weight.main_grad, _ = _C_ops.fused_linear_param_grad_add(
                    x, y_grad, weight.main_grad, None, True
                )
                return x_grad, None
            else:
                weight_grad, _ = _C_ops.fused_linear_param_grad_add(
                    x, y_grad, None, None, False
                )
                return x_grad, weight_grad

        if hasattr(weight, "main_grad") and hasattr(bias, "main_grad"):
            weight.main_grad, bias.main_grad = _C_ops.fused_linear_param_grad_add(
                x, y_grad, weight.main_grad, bias.main_grad, True
            )
            return x_grad, None, None
        else:
            weight_grad, bias_grad = _C_ops.fused_linear_param_grad_add(
                x, y_grad, None, None, False
            )
            return x_grad, weight_grad, bias_grad


def strtobool(s):
    """_summary_

    Args:
        s (_type_): _description_

    Returns:
        _type_: _description_
    """
    return True if distutils.util.strtobool(s) else False


def get_env(env_name, default_value=False):
    """_summary_

    Args:
        env_name (_type_): _description_
        default_value (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    return strtobool(os.getenv(env_name, str(default_value)))
