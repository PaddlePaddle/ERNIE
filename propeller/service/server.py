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
""" 
Never Never Never import paddle.fluid in main process, or any module would import fluid.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging
import six
from time import sleep, time
import multiprocessing

import zmq

log = logging.getLogger(__name__)


def _profile(msg):
    def _decfn(fn):
        def _retfn(*args, **kwargs):
            start = time()
            ret = fn(*args, **kwargs)
            end = time()
            log.debug('%s timecost: %.5f' % (msg, end - start))
            return ret

        return _retfn

    return _decfn


class Predictor(object):
    """paddle predictor wrapper"""

    def __init__(self, model_dir, device_idx=0):
        import paddle.fluid as F
        log.debug('create predictor on card %d' % device_idx)
        config = F.core.AnalysisConfig(model_dir)
        config.enable_use_gpu(5000, device_idx)
        self._predictor = F.core.create_paddle_predictor(config)

    @_profile('paddle')
    def __call__(self, args):
        for i, a in enumerate(args):
            a.name = 'placeholder_%d' % i
        res = self._predictor.run(args)
        return res


def run_worker(model_dir, device_idx, endpoint="ipc://worker.ipc"):
    """worker process entrence"""
    try:
        log.debug("run_worker %s" % device_idx)
        os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv(
            "CUDA_VISIBLE_DEVICES").split(",")[device_idx]
        log.debug('cuda_env %s' % os.environ["CUDA_VISIBLE_DEVICES"])
        import paddle.fluid as F
        from propeller.service import interface_pb2
        import propeller.service.utils as serv_utils
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.connect(endpoint)
        #socket.bind(endpoint)
        log.debug("Predictor building %s" % device_idx)
        predictor = Predictor(model_dir, 0)
        log.debug("Predictor %s" % device_idx)
    except Exception as e:
        log.exception(e)

    while True:
        #  Wait for next request from client
        try:
            message = socket.recv()
            log.debug("get message %s" % device_idx)
            slots = interface_pb2.Slots()
            slots.ParseFromString(message)
            pts = [serv_utils.slot_to_paddlearray(s) for s in slots.slots]
            ret = predictor(pts)
            slots = interface_pb2.Slots(
                slots=[serv_utils.paddlearray_to_slot(r) for r in ret])
            socket.send(slots.SerializeToString())
        except Exception as e:
            log.exception(e)
            socket.send(e.message)


class InferencePredictor(object):
    """control Predictor for multi gpu card"""

    def __init__(self, backend_addr, model_dir, n_devices=1):
        self.backend_addr = backend_addr
        self.model_dir = model_dir
        self.n_devices = n_devices
        self.children = []

    def start(self):
        """doc"""
        for device_idx in range(self.n_devices):
            p = multiprocessing.Process(
                target=run_worker,
                args=(self.model_dir, device_idx, self.backend_addr))
            p.start()
            self.children.append(p)
        return self

    def join(self):
        """doc"""
        for p in self.children:
            p.join()

    def term(self):
        """doc"""
        for p in self.children:
            log.debug("terminating children %s" % repr(p))
            p.terminate()


class InferenceProxy(object):
    """zmq proxy"""

    def __init__(self):
        """doc"""
        self.backend = None
        self.frontend = None

    def listen(self, frontend_addr, backend_addr):
        """doc"""
        log.info("InferenceProxy starting...")
        try:
            context = zmq.Context(1)
            # Socket facing clients
            self.frontend = context.socket(zmq.ROUTER)
            self.frontend.bind(frontend_addr)
            # Socket facing services
            self.backend = context.socket(zmq.DEALER)
            self.backend.bind(backend_addr)
            log.info("Queue init done")
            zmq.device(zmq.QUEUE, self.frontend, self.backend)
        except Exception as e:
            log.exception(e)
            log.info("Bringing down zmq device")
        finally:
            log.debug('terminating proxy')
            if self.frontend is not None:
                self.frontend.close()
            if self.backend is not None:
                self.backend.close()
            context.term()


class InferenceServer(object):
    """start InferencePredictor and InferenceProxy"""

    def __init__(self, model_dir, n_devices):
        """doc"""
        self.model_dir = model_dir
        self.n_devices = n_devices

    def listen(self, port):
        """doc"""
        frontend_addr = "tcp://*:%s" % port
        backend_addr = "ipc://backend.ipc"
        predictor = InferencePredictor(backend_addr, self.model_dir,
                                       self.n_devices).start()
        try:
            proxy = InferenceProxy()
            proxy.listen(frontend_addr, backend_addr)
            predictor.join()
        except KeyboardInterrupt:
            log.debug('terminating  server')
            predictor.term()
