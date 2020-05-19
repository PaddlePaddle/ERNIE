#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""record summary tensor in a collection scope"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import sys

import paddle.fluid as F
from propeller.paddle.collection import default_collection, Key


def scalar(name, tensor):
    """scalar summary"""
    if not isinstance(tensor, F.framework.Variable):
        raise ValueError('expect paddle Variable, got %s' % repr(tensor))
    tensor.persistable = True
    default_collection().add(Key.SUMMARY_SCALAR, (name, tensor))


def histogram(name, tensor):
    """histogram summary"""
    if not isinstance(tensor, F.framework.Variable):
        raise ValueError('expect paddle Variable, got %s' % repr(tensor))
    tensor.persistable = True
    default_collection().add(Key.SUMMARY_HISTOGRAM, (name, tensor))
