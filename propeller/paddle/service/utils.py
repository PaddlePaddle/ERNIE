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

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import struct

from propeller.service import interface_pb2
from propeller.service import interface_pb2_grpc

import paddle.fluid.core as core


def slot_to_paddlearray(slot):
    if slot.type == interface_pb2.Slot.FP32:
        type_str = 'f'
        dtype = core.PaddleDType.FLOAT32
    elif slot.type == interface_pb2.Slot.INT32:
        type_str = 'i'
        dtype = core.PaddleDType.INT32
    elif slot.type == interface_pb2.Slot.INT64:
        type_str = 'q'
        dtype = core.PaddleDType.INT64
    else:
        raise RuntimeError('know type %s' % slot.type)
    ret = core.PaddleTensor()
    ret.shape = slot.dims
    ret.dtype = dtype
    num = len(slot.data) // struct.calcsize(type_str)
    arr = struct.unpack('%d%s' % (num, type_str), slot.data)
    ret.data = core.PaddleBuf(arr)
    return ret


def paddlearray_to_slot(arr):
    if arr.dtype == core.PaddleDType.FLOAT32:
        dtype = interface_pb2.Slot.FP32
        type_str = 'f'
        arr_data = arr.data.float_data()
    elif arr.dtype == core.PaddleDType.INT32:
        dtype = interface_pb2.Slot.INT32
        type_str = 'i'
        arr_data = arr.data.int32_data()
    elif arr.dtype == core.PaddleDType.INT64:
        dtype = interface_pb2.Slot.INT64
        type_str = 'q'
        arr_data = arr.data.int64_data()
    else:
        raise RuntimeError('know type %s' % arr.dtype)
    data = struct.pack('%d%s' % (len(arr_data), type_str), *arr_data)
    pb = interface_pb2.Slot(type=dtype, dims=list(arr.shape), data=data)
    return pb
