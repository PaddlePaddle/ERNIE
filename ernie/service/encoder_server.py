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

import sys
import os
import argparse
import logging
import logging.handlers
import re
from propeller.service.server import InferenceServer
from propeller import log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, required=True)
    parser.add_argument('-p', '--port', type=int, default=8888)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--encode_layer', type=str, choices=[
        'pooler', 
        'layer12', 
        'layer11',
        'layer10',
        'layer9',
        'layer8',
        'layer7',
        'layer6',
        'layer5',
        'layer4',
        'layer3',
        'layer2',
        'layer1',
        ], default='pooler')
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)
    cuda_env = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_env is None:
        raise RuntimeError('CUDA_VISIBLE_DEVICES not set')
    if not os.path.exists(args.model_dir):
        raise ValueError('model_dir not found: %s' % args.model_dir)
    if not os.path.exists(args.model_dir):
        raise ValueError('model_dir not found: %s' % args.model_dir)
    n_devices = len(cuda_env.split(","))
    if args.encode_layer.lower() == 'pooler':
        model_dir = os.path.join(args.model_dir, 'pooler')
    else:
        pat = re.compile(r'layer(\d+)')
        match = pat.match(args.encode_layer.lower())
        layer = int(match.group(1))
        model_dir = os.path.join(args.model_dir, 'enc%d' % layer)

    server = InferenceServer(model_dir, n_devices)
    log.info('propeller server listent on port %d' % args.port)
    server.listen(args.port)
