#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
"""utils for server"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import struct

from propeller.service import interface_pb2


def slot_to_numpy(slot):
    """doc"""
    if slot.type == interface_pb2.Slot.FP32:
        dtype = np.float32
        type_str = 'f'
    elif slot.type == interface_pb2.Slot.INT32:
        type_str = 'i'
        dtype = np.int32
    elif slot.type == interface_pb2.Slot.INT64:
        dtype = np.int64
        type_str = 'q'
    else:
        raise RuntimeError('know type %s' % slot.type)
    num = len(slot.data) // struct.calcsize(type_str)
    arr = struct.unpack('%d%s' % (num, type_str), slot.data)
    shape = slot.dims
    ret = np.array(arr, dtype=dtype).reshape(shape)
    return ret


def numpy_to_slot(arr):
    """doc"""
    if arr.dtype == np.float32:
        dtype = interface_pb2.Slot.FP32
    elif arr.dtype == np.int32:
        dtype = interface_pb2.Slot.INT32
    elif arr.dtype == np.int64:
        dtype = interface_pb2.Slot.INT64
    else:
        raise RuntimeError('know type %s' % arr.dtype)
    pb = interface_pb2.Slot(
        type=dtype, dims=list(arr.shape), data=arr.tobytes())
    return pb


def slot_to_paddlearray(slot):
    """doc"""
    import paddle.fluid.core as core
    if slot.type == interface_pb2.Slot.FP32:
        dtype = np.float32
        type_str = 'f'
    elif slot.type == interface_pb2.Slot.INT32:
        dtype = np.int32
        type_str = 'i'
    elif slot.type == interface_pb2.Slot.INT64:
        dtype = np.int64
        type_str = 'q'
    else:
        raise RuntimeError('know type %s' % slot.type)
    num = len(slot.data) // struct.calcsize(type_str)
    arr = struct.unpack('%d%s' % (num, type_str), slot.data)
    ret = core.PaddleTensor(data=np.array(arr, dtype=dtype).reshape(slot.dims))
    return ret


def paddlearray_to_slot(arr):
    """doc"""
    import paddle.fluid.core as core
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


def nparray_list_serialize(arr_list):
    """doc"""
    slot_list = [numpy_to_slot(arr) for arr in arr_list]
    slots = interface_pb2.Slots(slots=slot_list)
    return slots.SerializeToString()


def nparray_list_deserialize(string):
    """doc"""
    slots = interface_pb2.Slots()
    slots.ParseFromString(string)
    return [slot_to_numpy(slot) for slot in slots.slots]
