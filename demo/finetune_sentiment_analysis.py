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
from pathlib import Path
from visualdl import LogWriter

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
#from ernie.optimization import AdamW, LinearDecay
from demo.utils import create_if_not_exists, get_warmup_and_linear_decay

parser = argparse.ArgumentParser('classify model with ERNIE')
parser.add_argument(
    '--from_pretrained',
    type=Path,
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
    '--save_dir', type=Path, required=True, help='model output directory')
parser.add_argument(
    '--init_checkpoint',
    type=str,
    default=None,
    help='checkpoint to warm start from')
parser.add_argument(
    '--wd', type=float, default=0.01, help='weight decay, aka L2 regularizer')
parser.add_argument(
    '--use_amp',
    action='store_true',
    help='only activate AMP(auto mixed precision accelatoin) on TensorCore compatible devices'
)

args = parser.parse_args()

tokenizer = ErnieTokenizer.from_pretrained(args.from_pretrained)
#tokenizer = ErnieTinyTokenizer.from_pretrained(args.from_pretrained)

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
        seg_a, _ = tokenizer.truncate(seg_a, [], seqlen=args.max_seqlen)
        sentence, segments = tokenizer.build_for_ernie(seg_a, [])
        return sentence, segments, label


    train_ds = feature_column.build_dataset('train', data_dir=os.path.join(args.data_dir, 'train'), shuffle=True, repeat=False, use_gz=False) \
                                   .map(map_fn) \
                                   .padded_batch(args.bsz)

    dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(args.data_dir, 'dev'), shuffle=False, repeat=False, use_gz=False) \
                                   .map(map_fn) \
                                   .padded_batch(args.bsz)

    g_clip = P.nn.ClipGradByGlobalNorm(1.0)  #experimental
    lr_scheduler = P.optimizer.lr.LambdaDecay(
        args.lr,
        get_warmup_and_linear_decay(
            args.max_steps, int(args.warmup_proportion * args.max_steps)))

    param_name_to_exclue_from_weight_decay = re.compile(
        r'.*layer_norm_scale|.*layer_norm_bias|.*b_0')

    opt = P.optimizer.AdamW(
        lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.wd,
        apply_decay_param_fun=lambda n: not param_name_to_exclue_from_weight_decay.match(n),
        grad_clip=g_clip)
    scaler = P.amp.GradScaler(enable=args.use_amp)
    with LogWriter(logdir=str(create_if_not_exists(args.save_dir /
                                                   'vdl'))) as log_writer:
        with P.amp.auto_cast(enable=args.use_amp):
            for epoch in range(args.epoch):
                for step, d in enumerate(
                        P.io.DataLoader(
                            train_ds, places=P.CUDAPlace(0), batch_size=None)):
                    ids, sids, label = d
                    loss, _ = model(ids, sids, labels=label)
                    loss = scaler.scale(loss)
                    loss.backward()
                    scaler.minimize(opt, loss)
                    model.clear_gradients()
                    lr_scheduler.step()

                    if step % 10 == 0:
                        _lr = lr_scheduler.get_lr()
                        if args.use_amp:
                            _l = (loss / scaler._scale).numpy()
                            msg = '[step-%d] train loss %.5f lr %.3e scaling %.3e' % (
                                step, _l, _lr, scaler._scale.numpy())
                        else:
                            _l = loss.numpy()
                            msg = '[step-%d] train loss %.5f lr %.3e' % (
                                step, _l, _lr)
                        log.debug(msg)
                        log_writer.add_scalar('loss', _l, step=step)
                        log_writer.add_scalar('lr', _lr, step=step)

                    if step % 100 == 0:
                        acc = []
                        with P.no_grad():
                            model.eval()
                            for step, d in enumerate(
                                    P.io.DataLoader(
                                        dev_ds,
                                        places=P.CUDAPlace(0),
                                        batch_size=None)):
                                ids, sids, label = d
                                loss, logits = model(ids, sids, labels=label)
                                a = (logits.argmax(-1) == label)
                                acc.append(a.numpy())
                            model.train()
                        acc = np.concatenate(acc).mean()
                        log_writer.add_scalar('eval/acc', acc, step=step)
                        log.debug('acc %.5f' % acc)
                        if args.save_dir is not None:
                            P.save(model.state_dict(),
                                   args.save_dir / 'ckpt.bin')
        if args.save_dir is not None:
            P.save(model.state_dict(), args.save_dir / 'ckpt.bin')
else:
    feature_column = propeller.data.FeatureColumns([
        propeller.data.TextColumn(
            'seg_a',
            unk_id=tokenizer.unk_id,
            vocab_dict=tokenizer.vocab,
            tokenizer=tokenizer.tokenize),
    ])

    sd = P.load(args.init_checkpoint)
    model.set_dict(sd)
    model.eval()

    def map_fn(seg_a):
        seg_a, _ = tokenizer.truncate(seg_a, [], seqlen=args.max_seqlen)
        sentence, segments = tokenizer.build_for_ernie(seg_a, [])
        return sentence, segments

    predict_ds = feature_column.build_dataset_from_stdin('predict') \
                                   .map(map_fn) \
                                   .padded_batch(args.bsz)

    for step, (ids, sids) in enumerate(
            P.io.DataLoader(
                predict_ds, places=P.CUDAPlace(0), batch_size=None)):
        _, logits = model(ids, sids)
        pred = logits.numpy().argmax(-1)
        print('\n'.join(map(str, pred.tolist())))
