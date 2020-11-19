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
from tqdm import tqdm
from functools import reduce, partial

import numpy as np
import logging
import argparse

import paddle
import paddle.fluid as F
import paddle.fluid.dygraph as FD
import paddle.fluid.layers as L

from propeller import log
import propeller.paddle as propeller

log.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
log = logging.getLogger()

#from model.bert import BertConfig, BertModelLayer
from ernie.modeling_ernie import ErnieModel, ErnieModelForSequenceClassification
from ernie.tokenizing_ernie import ErnieTokenizer, ErnieTinyTokenizer
from ernie.optimization import AdamW, LinearDecay

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
        '--max_steps',
        type=int,
        required=True,
        help='max_train_steps, set this to EPOCH * NUM_SAMPLES / BATCH_SIZE')
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument(
        '--save_dir', type=str, default=None, help='model output directory')
    parser.add_argument(
        '--wd',
        type=float,
        default=0.01,
        help='weight decay, aka L2 regularizer')

    args = parser.parse_args()

    tokenizer = ErnieTokenizer.from_pretrained(args.from_pretrained)
    #tokenizer = ErnieTinyTokenizer.from_pretrained(args.from_pretrained)

    place = F.CUDAPlace(0)
    with FD.guard(place):
        model = ErnieModelForSequenceClassification.from_pretrained(
            args.from_pretrained, num_labels=3, name='')
        if not args.eval:
            feature_column = propeller.data.FeatureColumns([
                propeller.data.TextColumn(
                    'seg_a',
                    unk_id=tokenizer.unk_id,
                    vocab_dict=tokenizer.vocab,
                    tokenizer=tokenizer.tokenize),
                propeller.data.LabelColumn('label'),
            ])

            def map_fn(seg_a, label):
                seg_a, _ = tokenizer.truncate(
                    seg_a, [], seqlen=args.max_seqlen)
                sentence, segments = tokenizer.build_for_ernie(seg_a, [])
                return sentence, segments, label


            train_ds = feature_column.build_dataset('train', data_dir=os.path.join(args.data_dir, 'train'), shuffle=True, repeat=False, use_gz=False) \
                                           .map(map_fn) \
                                           .padded_batch(args.bsz)

            dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(args.data_dir, 'dev'), shuffle=False, repeat=False, use_gz=False) \
                                           .map(map_fn) \
                                           .padded_batch(args.bsz)

            shapes = ([-1, args.max_seqlen], [-1, args.max_seqlen], [-1])
            types = ('int64', 'int64', 'int64')

            train_ds.data_shapes = shapes
            train_ds.data_types = types
            dev_ds.data_shapes = shapes
            dev_ds.data_types = types

            g_clip = F.clip.GradientClipByGlobalNorm(1.0)  #experimental
            opt = AdamW(
                learning_rate=LinearDecay(
                    args.lr,
                    int(args.warmup_proportion * args.max_steps),
                    args.max_steps),
                parameter_list=model.parameters(),
                weight_decay=args.wd,
                grad_clip=g_clip)

            for epoch in range(args.epoch):
                for step, d in enumerate(
                        tqdm(
                            train_ds.start(place), desc='training')):
                    ids, sids, label = d
                    loss, _ = model(ids, sids, labels=label)
                    loss.backward()
                    if step % 10 == 0:
                        log.debug('train loss %.5f lr %.3e' %
                                  (loss.numpy(), opt.current_step_lr()))
                    opt.minimize(loss)
                    model.clear_gradients()
                    if step % 100 == 0:
                        acc = []
                        with FD.base._switch_tracer_mode_guard_(
                                is_train=False):
                            model.eval()
                            for step, d in enumerate(
                                    tqdm(
                                        dev_ds.start(place),
                                        desc='evaluating %d' % epoch)):
                                ids, sids, label = d
                                loss, logits = model(ids, sids, labels=label)
                                #print('\n'.join(map(str, logits.numpy().tolist())))
                                a = L.argmax(logits, -1) == label
                                acc.append(a.numpy())
                            model.train()
                        log.debug('acc %.5f' % np.concatenate(acc).mean())
            if args.save_dir is not None:
                F.save_dygraph(model.state_dict(), args.save_dir)
        else:
            feature_column = propeller.data.FeatureColumns([
                propeller.data.TextColumn(
                    'seg_a',
                    unk_id=tokenizer.unk_id,
                    vocab_dict=tokenizer.vocab,
                    tokenizer=tokenizer.tokenize),
            ])

            assert args.save_dir is not None
            sd, _ = FD.load_dygraph(args.save_dir)
            model.set_dict(sd)
            model.eval()

            def map_fn(seg_a):
                seg_a, _ = tokenizer.truncate(
                    seg_a, [], seqlen=args.max_seqlen)
                sentence, segments = tokenizer.build_for_ernie(seg_a, [])
                return sentence, segments

            predict_ds = feature_column.build_dataset_from_stdin('predict') \
                                           .map(map_fn) \
                                           .padded_batch(args.bsz)
            shapes = ([-1, args.max_seqlen], [-1, args.max_seqlen])
            types = ('int64', 'int64')
            predict_ds.data_shapes = shapes
            predict_ds.data_types = types

            for step, (ids, sids) in enumerate(predict_ds.start(place)):
                _, logits = model(ids, sids)
                pred = logits.numpy().argmax(-1)
                print('\n'.join(map(str, pred.tolist())))
