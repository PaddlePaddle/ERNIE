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
import asyncio
import threading
import math

import zmq
import zmq.asyncio
import numpy as np

from propeller import log
import propeller.service.utils as serv_utils


class InferenceBaseClient(object):
    def __init__(self, address):
        self.context = zmq.Context()
        self.address = address
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(address)
        log.info("Connecting to server... %s" % address)

    def __call__(self, *args):
        for arg in args:
            if not isinstance(arg, np.ndarray):
                raise ValueError('expect ndarray slot data, got %s' %
                                 repr(arg))
        request = serv_utils.nparray_list_serialize(args)

        self.socket.send(request)
        reply = self.socket.recv()
        ret = serv_utils.nparray_list_deserialize(reply)
        return ret


class InferenceClient(InferenceBaseClient):
    def __init__(self, address, batch_size=128, num_coroutine=10, timeout=10.):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        context = zmq.asyncio.Context()
        self.socket_pool = [
            context.socket(zmq.REQ) for _ in range(num_coroutine)
        ]
        log.info("Connecting to server... %s" % address)
        for socket in self.socket_pool:
            socket.connect(address)
        self.num_coroutine = num_coroutine
        self.batch_size = batch_size
        self.timeout = int(timeout * 1000)

    #yapf: disable
    def __call__(self, *args):
        for arg in args:
            if not isinstance(arg, np.ndarray):
                raise ValueError('expect ndarray slot data, got %s' %
                                 repr(arg))

        num_tasks = math.ceil(1. * args[0].shape[0] / self.batch_size)
        rets = [None] * num_tasks

        async def get(coroutine_idx=0, num_coroutine=1):
            socket = self.socket_pool[coroutine_idx]
            while coroutine_idx < num_tasks:
                begin = coroutine_idx * self.batch_size
                end = (coroutine_idx + 1) * self.batch_size

                arr_list = [arg[begin:end] for arg in args]
                request = serv_utils.nparray_list_serialize(arr_list)
                try:
                    await socket.send(request)
                    await socket.poll(self.timeout, zmq.POLLIN)
                    reply = await socket.recv(zmq.NOBLOCK)
                    ret = serv_utils.nparray_list_deserialize(reply)
                except Exception as e:
                    log.exception(e)
                    ret = None
                rets[coroutine_idx] = ret
                coroutine_idx += num_coroutine

        futures = [
            get(i, self.num_coroutine) for i in range(self.num_coroutine)
        ]
        self.loop.run_until_complete(asyncio.wait(futures))
        for r in rets:
            if r is None:
                raise RuntimeError('Client call failed')
        return [np.concatenate(col, 0) for col in zip(*rets)]
    #yapf: enable
