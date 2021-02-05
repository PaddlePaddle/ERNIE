#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

"""parameters init function implementations"""


from __future__ import print_function

import os
import six

import numpy as np
import paddle.fluid as fluid

def circle_loss(sp, sn, m, scale_circle):
    """
    sp: score list of positive samples, shape [B * L]
    sn: score list of negative samples, shape [B * K]
    m: relaxation factor in circle loss function
    scale:  scale factor in circle loss function

    return: circle loss value, shape [1]
    """
    op = 1. + m
    on = 0. - m

    delta_p = 1 - m
    delta_n = m

    ap = fluid.layers.relu(op - sp)
    ap.stop_gradient = True
    an = fluid.layers.relu(sn - on)
    an.stop_gradient = True

    logit_p = ap * (sp - delta_p)
    logit_p = -1. * scale_circle * logit_p
    logit_p = fluid.layers.cast(x=logit_p, dtype=np.float64)
    loss_p = fluid.layers.reduce_sum(fluid.layers.exp(logit_p), dim=1, keep_dim=False)

    logit_n = an * (sn - delta_n)
    logit_n = scale_circle * logit_n
    logit_n = fluid.layers.cast(x=logit_n, dtype=np.float64)
    loss_n = fluid.layers.reduce_sum(fluid.layers.exp(logit_n), dim=1, keep_dim=False)

    circle_loss = fluid.layers.log(1 + loss_n * loss_p)
    circle_loss = fluid.layers.cast(x=circle_loss, dtype=np.float32)
    return fluid.layers.mean(circle_loss)


