# -*- coding: utf-8 -*-
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

"""
@author: kebo
@contact: kebo01@baidu.com

@version: 1.0
@file: modeling_pp.py
@time: 2024/03/21 17:04:24
@Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved

这一行开始写关于本文件的说明与解释


"""
import paddle
from .eva_vit_model_pp import (
    EVAVisionTransformerPipe,
    get_hcg,
)
from .modeling import ImageEncoderCriterion


class ImageEncoderPipe(EVAVisionTransformerPipe):
    """_summary_"""

    def get_loss_fn(self, config):
        """pp stage -1 的时候进行criterion初始化

        Args:
            config (_type_): _description_

        Returns:
            _type_: _description_
        """
        hcg = get_hcg()
        if hcg.stage_id == hcg._pp_degree - 1:
            return ImageEncoderCriterion(config)

        return lambda x: paddle.zeros([])
