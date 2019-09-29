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

import logging
import six
import asyncio
import threading

import grpc
from propeller.service import interface_pb2
from propeller.service import interface_pb2_grpc
import propeller.paddle.service.utils as serv_utils

from concurrent.futures import ThreadPoolExecutor

import paddle.fluid as F

from time import sleep, time

log = logging.getLogger(__name__)


def profile(msg):
    def decfn(fn):
        def retfn(*args, **kwargs):
            start = time()
            ret = fn(*args, **kwargs)
            end = time()
            log.debug('%s timecost: %.5f' % (msg, end - start))
            return ret

        return retfn

    return decfn


def serve(model_dir, host, num_concurrent=None):
    if six.PY2:
        raise RuntimeError('propeller service work in python3 only')
    num_worker = len(F.cuda_places(
    )) if num_concurrent is None else num_concurrent
    pool = ThreadPoolExecutor(num_worker)

    class Predictor(object):
        def __init__(self, did):
            log.debug('create predictor on card %d' % did)
            config = F.core.AnalysisConfig(model_dir)
            config.enable_use_gpu(5000, did)
            self._predictor = F.core.create_paddle_predictor(config)

        @profile('paddle')
        def __call__(self, args):
            for i, a in enumerate(args):
                a.name = 'placeholder_%d' % i
            res = self._predictor.run(args)
            return res

    predictor_context = {}

    class InferenceService(interface_pb2_grpc.InferenceServicer):
        @profile('service')
        def Infer(self, request, context):
            try:
                slots = request.slots
                current_thread = threading.current_thread()
                log.debug('%d slots received dispatch to thread %s' %
                          (len(slots), current_thread))
                if current_thread not in predictor_context:
                    did = list(pool._threads).index(current_thread)
                    log.debug('spawning worker thread %d' % did)
                    predictor = Predictor(did)
                    predictor_context[current_thread] = predictor
                else:
                    predictor = predictor_context[current_thread]
                slots = [serv_utils.slot_to_paddlearray(s) for s in slots]
                ret = predictor(slots)
                response = [serv_utils.paddlearray_to_slot(r) for r in ret]
            except Exception as e:
                log.exception(e)
                raise e
            return interface_pb2.Slots(slots=response)

    server = grpc.server(pool)
    interface_pb2_grpc.add_InferenceServicer_to_server(InferenceService(),
                                                       server)
    server.add_insecure_port(host)
    server.start()
    log.info('server started on %s...' % host)
    try:
        while True:
            sleep(100000)
    except KeyboardInterrupt as e:
        pass
    log.info('server stoped...')


if __name__ == '__main__':
    from propeller import log
    log.setLevel(logging.DEBUG)
    serve(
        '/home/work/chenxuyi/playground/grpc_play/ernie2.0/',
        '10.255.138.19:8334',
        num_concurrent=3)
