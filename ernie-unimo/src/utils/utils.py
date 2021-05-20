#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""utils"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


def visualdl_log(metrics_output, train_loss, steps, phase):
    """log visualization
    """
    print("{phase} log: steps {steps}, loss {loss}, metrics: {metrics}".format(
        phase=phase, steps=steps, loss=train_loss, metrics=metrics_output))


def print_eval_log(ret):
    """print log"""
    prefix_log = "[%s evaluation] ave loss: %.4f," % (ret['phase'], ret['loss'])
    postfix_log = "data_num: %d, elapsed time: %.4f s" % (ret['data_num'], ret['used_time'])
    mid_log = " "
    for k, v in ret.items():
        if k not in ['phase', 'loss', 'data_num', 'used_time', 'key_eval']:
            mid_log = mid_log + "%s: %.4f, " % (k, round(v, 4))
    log = prefix_log + mid_log + postfix_log
    print(log)


def get_time():
    """get time"""
    res = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    return res
