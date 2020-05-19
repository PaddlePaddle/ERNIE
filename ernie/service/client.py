#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from propeller.service.client import InferenceClient
from propeller import log
import six
import utils.data
from time import time
import numpy as np

class ErnieClient(InferenceClient):
    def __init__(self, 
            vocab_file, 
            host='localhost', 
            port=8888, 
            batch_size=32, 
            num_coroutine=1, 
            timeout=10., 
            max_seqlen=128):
        host_port = 'tcp://%s:%d' % (host, port) 
        client = super(ErnieClient, self).__init__(host_port, batch_size=batch_size, num_coroutine=num_coroutine, timeout=timeout)
        self.vocab = {j.strip().split(b'\t')[0].decode('utf8'): i for i, j in enumerate(open(vocab_file, 'rb'))}
        self.tokenizer = utils.data.CharTokenizer(self.vocab.keys())
        self.max_seqlen = max_seqlen
        self.cls_id = self.vocab['[CLS]']
        self.sep_id = self.vocab['[SEP]']

    def txt_2_id(self, text):
        ids = np.array([self.vocab[i] for i in self.tokenizer(text)])
        return ids

    def pad_and_batch(self, ids):
        max_len = max(map(len, ids))
        padded = np.stack([np.pad(i, [[0, max_len - len(i)]], mode='constant')for i in ids])
        padded = np.expand_dims(padded, axis=-1)
        return padded

    def __call__(self, text_a, text_b=None):
        if text_b is not None and len(text_a) != len(text_b):
            raise ValueError('text_b %d has different size than text_a %d' % (text_b, text_a))
        text_a = [i.encode('utf8') if isinstance(i, six.string_types) else i for i in text_a]
        if text_b is not None:
            text_b = [i.encode('utf8') if isinstance(i, six.string_types) else i for i in text_b]

        ids_a = map(self.txt_2_id, text_a)
        if text_b is not None:
            ids_b = map(self.txt_2_id, text_b)
            ret = [utils.data.build_2_pair(a, b, self.max_seqlen, self.cls_id, self.sep_id) for a, b in zip(ids_a, ids_b)]
        else:
            ret = [utils.data.build_1_pair(a, self.max_seqlen, self.cls_id, self.sep_id) for a in ids_a]
        sen_ids, token_type_ids = zip(*ret)
        sen_ids = self.pad_and_batch(sen_ids)
        token_type_ids = self.pad_and_batch(token_type_ids)
        ret, = super(ErnieClient, self).__call__(sen_ids, token_type_ids)
        return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ernie_encoder_client')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-p', '--port', type=int, default=8888)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_coroutine', type=int, default=1)
    parser.add_argument('--vocab', type=str, required=True)
    args = parser.parse_args()

    client = ErnieClient(args.vocab, args.host, args.port, batch_size=args.batch_size, num_coroutine=args.num_coroutine)
    inputs = [i.strip().split(b'\t') for i in open(args.input, 'rb').readlines()]
    if len(inputs) == 0:
        raise ValueError('empty input')
    send_batch = args.num_coroutine * args.batch_size
    send_num = len(inputs) // send_batch + 1
    rets = []
    start = time()
    for i in range(send_num):
        slice = inputs[i * send_batch: (i + 1) * send_batch]
        if len(slice) == 0:
            continue
        columns = list(zip(*slice))
        if len(columns) > 2:
            raise ValueError('inputs file has more than 2 columns')
        ret = client(*columns)
        if len(ret.shape) == 3:
            ret = ret[:, 0, :] # take cls
        rets.append(ret)
    end = time()
    with open(args.output, 'wb') as outf:
        arr = np.concatenate(rets, 0)
        np.save(outf, arr)
        log.info('query num: %d average latency %.5f' % (len(inputs), (end - start)/len(inputs)))

