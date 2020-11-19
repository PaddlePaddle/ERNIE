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

import os
import re
import time
import logging
import json
from random import random
from functools import reduce, partial

import numpy as np
import logging
import argparse

import paddle as P

from propeller import log
import propeller.paddle as propeller

log.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

#from model.bert import BertConfig, BertModelLayer
from ernie.modeling_ernie import ErnieModel, ErnieModelForSequenceClassification
from ernie.tokenizing_ernie import ErnieTokenizer, ErnieTinyTokenizer
from ernie.optimization import AdamW, LinearDecay
from demo.utils import UnpackDataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser('classify model with ERNIE')
    parser.add_argument(
        '--from_pretrained',
        type=str,
        required=True,
        help='pretrained model directory or tag')
    parser.add_argument(
        '--max_seqlen',
        type=int,
        default=128,
        help='max sentence length, should not greater than 512')
    parser.add_argument('--bsz', type=int, default=32, help='batchsize')
    parser.add_argument('--epoch', type=int, default=3, help='epoch')
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='data directory includes train / develop data')
    parser.add_argument(
        '--use_lr_decay',
        action='store_true',
        help='if set, learning rate will decay to zero at `max_steps`')
    parser.add_argument(
        '--warmup_proportion',
        type=float,
        default=0.1,
        help='if use_lr_decay is set, '
        'learning rate will raise to `lr` at `warmup_proportion` * `max_steps` and decay to 0. at `max_steps`'
    )
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument(
        '--inference_model_dir',
        type=str,
        default=None,
        help='inference model output directory')
    parser.add_argument(
        '--save_dir', type=str, default=None, help='model output directory')
    parser.add_argument(
        '--max_steps',
        type=int,
        default=None,
        help='max_train_steps, set this to EPOCH * NUM_SAMPLES / BATCH_SIZE')
    parser.add_argument(
        '--wd',
        type=float,
        default=0.01,
        help='weight decay, aka L2 regularizer')
    parser.add_argument(
        '--init_checkpoint',
        type=str,
        default=None,
        help='checkpoint to warm start from')

    args = parser.parse_args()

    tokenizer = ErnieTokenizer.from_pretrained(args.from_pretrained)
    #tokenizer = ErnieTinyTokenizer.from_pretrained(args.from_pretrained)

    feature_column = propeller.data.FeatureColumns([
        propeller.data.TextColumn(
            'seg_a',
            unk_id=tokenizer.unk_id,
            vocab_dict=tokenizer.vocab,
            tokenizer=tokenizer.tokenize),
        propeller.data.TextColumn(
            'seg_b',
            unk_id=tokenizer.unk_id,
            vocab_dict=tokenizer.vocab,
            tokenizer=tokenizer.tokenize),
        propeller.data.LabelColumn(
            'label',
            vocab_dict={
                b"contradictory": 0,
                b"contradiction": 0,
                b"entailment": 1,
                b"neutral": 2,
            }),
    ])

    def map_fn(seg_a, seg_b, label):
        seg_a, seg_b = tokenizer.truncate(seg_a, seg_b, seqlen=args.max_seqlen)
        sentence, segments = tokenizer.build_for_ernie(seg_a, seg_b)
        return sentence, segments, label


    train_ds = feature_column.build_dataset('train', data_dir=os.path.join(args.data_dir, 'train'), shuffle=True, repeat=False, use_gz=False) \
                                   .map(map_fn) \
                                   .padded_batch(args.bsz, (0, 0, 0))

    dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(args.data_dir, 'dev'), shuffle=False, repeat=False, use_gz=False) \
                                   .map(map_fn) \
                                   .padded_batch(args.bsz, (0, 0, 0))

    shapes = ([-1, args.max_seqlen], [-1, args.max_seqlen], [-1])
    types = ('int64', 'int64', 'int64')

    train_ds.data_shapes = shapes
    train_ds.data_types = types
    dev_ds.data_shapes = shapes
    dev_ds.data_types = types

    place = P.CUDAPlace(0)
    model = ErnieModelForSequenceClassification.from_pretrained(
        args.from_pretrained, num_labels=3, name='')

    if args.init_checkpoint is not None:
        log.info('loading checkpoint from %s' % args.init_checkpoint)
        sd = P.load(args.init_checkpoint)
        model.set_state_dict(sd)

    g_clip = P.nn.ClipGradByGlobalNorm(1.0)  #experimental
    if args.use_lr_decay:
        opt = AdamW(
            learning_rate=LinearDecay(args.lr,
                                      int(args.warmup_proportion *
                                          args.max_steps), args.max_steps),
            parameter_list=model.parameters(),
            weight_decay=args.wd,
            grad_clip=g_clip)
    else:
        opt = AdamW(
            args.lr,
            parameter_list=model.parameters(),
            weight_decay=args.wd,
            grad_clip=g_clip)

    for epoch in range(args.epoch):
        for step, (
                ids, sids, label
        ) in enumerate(UnpackDataLoader(
                train_ds, places=P.CUDAPlace(0))):
            loss, _ = model(ids, sids, labels=label)
            loss.backward()
            if step % 10 == 0:
                log.debug('train loss %.5f lr %.3e' %
                          (loss.numpy(), opt.current_step_lr()))
            opt.minimize(loss)
            model.clear_gradients()
            if step % 100 == 0:
                acc = []
                with P.no_grad():
                    model.eval()
                    for step, d in enumerate(
                            UnpackDataLoader(
                                dev_ds, places=P.CUDAPlace(0))):
                        ids, sids, label = d
                        loss, logits = model(ids, sids, labels=label)
                        #print('\n'.join(map(str, logits.numpy().tolist())))
                        a = (logits.argmax(-1) == label)
                        acc.append(a.numpy())
                    model.train()
                log.debug('acc %.5f' % np.concatenate(acc).mean())
    if args.save_dir is not None:
        P.save(model.state_dict(), args.save_dir)
    if args.inference_model_dir is not None:
        log.debug('saving inference model')

        class InferemceModel(ErnieModelForSequenceClassification):
            @P.jit.to_static(input_spec=[
                P.static.InputSpec(
                    shape=[None, None], name='ids'), P.static.InputSpec(
                        shape=[None, None], name='sids')
            ])
            def forward(self, ids, sids):
                _, logits = super(InferemceModel, self).forward(ids, sids)
                return logits

        model.__class__ = InferemceModel  #dynamic change model type, to make sure forward output doesn't contain `None`
        src_placeholder = P.zeros([2, 2], dtype='int64')
        sent_placehodler = P.zeros([2, 2], dtype='int64')
        model(src_placeholder, sent_placehodler)
        P.jit.save(model, args.inference_model_dir)
        log.debug('done')
