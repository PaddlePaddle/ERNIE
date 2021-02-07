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
from visualdl import LogWriter

import numpy as np
import logging
import argparse
from pathlib import Path
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
parser.add_argument(
    '--bsz',
    type=int,
    default=128,
    help='global batch size for each optimizer step')
parser.add_argument(
    '--micro_bsz',
    type=int,
    default=32,
    help='batch size for each device. if `--bsz` > `--micro_bsz` * num_device, will do grad accumulate'
)
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
    type=Path,
    default=None,
    help='inference model output directory')
parser.add_argument(
    '--save_dir', type=Path, required=True, help='model output directory')
parser.add_argument(
    '--max_steps',
    type=int,
    default=None,
    help='max_train_steps, set this to EPOCH * NUM_SAMPLES / BATCH_SIZE')
parser.add_argument(
    '--wd', type=float, default=0.01, help='weight decay, aka L2 regularizer')
parser.add_argument(
    '--init_checkpoint',
    type=str,
    default=None,
    help='checkpoint to warm start from')
parser.add_argument(
    '--use_amp',
    action='store_true',
    help='only activate AMP(auto mixed precision accelatoin) on TensorCore compatible devices'
)

args = parser.parse_args()

if args.bsz > args.micro_bsz:
    assert args.bsz % args.micro_bsz == 0, 'cannot perform gradient accumulate with bsz:%d micro_bsz:%d' % (
        args.bsz, args.micro_bsz)
    acc_step = args.bsz // args.micro_bsz
    log.info(
        'performing gradient accumulate: global_bsz:%d, micro_bsz:%d, accumulate_steps:%d'
        % (args.bsz, args.micro_bsz, acc_step))
    args.bsz = args.micro_bsz
else:
    acc_step = 1

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

place = P.CUDAPlace(0)
model = ErnieModelForSequenceClassification.from_pretrained(
    args.from_pretrained, num_labels=3, name='')

if args.init_checkpoint is not None:
    log.info('loading checkpoint from %s' % args.init_checkpoint)
    sd = P.load(args.init_checkpoint)
    model.set_state_dict(sd)

g_clip = P.nn.ClipGradByGlobalNorm(1.0)  #experimental
param_name_to_exclue_from_weight_decay = re.compile(
    r'.*layer_norm_scale|.*layer_norm_bias|.*b_0')
if args.use_lr_decay:
    lr_scheduler = P.optimizer.lr.LambdaDecay(
        args.lr,
        get_warmup_and_linear_decay(
            args.max_steps, int(args.warmup_proportion * args.max_steps)))
    opt = P.optimizer.AdamW(
        lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.wd,
        apply_decay_param_fun=lambda n: not param_name_to_exclue_from_weight_decay.match(n),
        grad_clip=g_clip)
else:
    lr_scheduler = None
    opt = P.optimizer.AdamW(
        args.lr,
        parameters=model.parameters(),
        weight_decay=args.wd,
        apply_decay_param_fun=lambda n: not param_name_to_exclue_from_weight_decay.match(n),
        grad_clip=g_clip)

scaler = P.amp.GradScaler(enable=args.use_amp)
step, inter_step = 0, 0
with LogWriter(
        logdir=str(create_if_not_exists(args.save_dir / 'vdl'))) as log_writer:
    with P.amp.auto_cast(enable=args.use_amp):
        for epoch in range(args.epoch):
            for ids, sids, label in P.io.DataLoader(
                    train_ds, places=P.CUDAPlace(0), batch_size=None):
                inter_step += 1
                loss, _ = model(ids, sids, labels=label)
                loss /= acc_step
                loss = scaler.scale(loss)
                loss.backward()
                if inter_step % acc_step != 0:
                    continue
                step += 1
                scaler.minimize(opt, loss)
                model.clear_gradients()
                lr_scheduler and lr_scheduler.step()

                if step % 10 == 0:
                    _lr = lr_scheduler.get_lr(
                    ) if args.use_lr_decay else args.lr
                    if args.use_amp:
                        _l = (loss / scaler._scale).numpy()
                        msg = '[step-%d] train loss %.5f lr %.3e scaling %.3e' % (
                            step, _l, _lr, scaler._scale.numpy())
                    else:
                        _l = loss.numpy()
                        msg = '[step-%d] train loss %.5f lr %.3e' % (step, _l,
                                                                     _lr)
                    log.debug(msg)
                    log_writer.add_scalar('loss', _l, step=step)
                    log_writer.add_scalar('lr', _lr, step=step)
                if step % 100 == 0:
                    acc = []
                    with P.no_grad():
                        model.eval()
                        for ids, sids, label in P.io.DataLoader(
                                dev_ds, places=P.CUDAPlace(0),
                                batch_size=None):
                            loss, logits = model(ids, sids, labels=label)
                            #print('\n'.join(map(str, logits.numpy().tolist())))
                            a = (logits.argmax(-1) == label)
                            acc.append(a.numpy())
                        model.train()
                    acc = np.concatenate(acc).mean()
                    log_writer.add_scalar('eval/acc', acc, step=step)
                    log.debug('acc %.5f' % acc)
                    if args.save_dir is not None:
                        P.save(model.state_dict(), args.save_dir / 'ckpt.bin')
if args.save_dir is not None:
    P.save(model.state_dict(), args.save_dir / 'ckpt.bin')
if args.inference_model_dir is not None:

    class InferenceModel(ErnieModelForSequenceClassification):
        def forward(self, ids, sids):
            _, logits = super(InferenceModel, self).forward(ids, sids)
            return logits

    model.__class__ = InferenceModel
    log.debug('saving inference model')
    src_placeholder = P.zeros([2, 2], dtype='int64')
    sent_placehodler = P.zeros([2, 2], dtype='int64')
    _, static = P.jit.TracedLayer.trace(
        model, inputs=[src_placeholder, sent_placehodler])
    static.save_inference_model(str(args.inference_model_dir))

    #class InferenceModel(ErnieModelForSequenceClassification):
    #    @P.jit.to_static
    #    def forward(self, ids, sids):
    #        _, logits =  super(InferenceModel, self).forward(ids, sids, labels=None)
    #        return logits
    #model.__class__ = InferenceModel
    #src_placeholder = P.zeros([2, 2], dtype='int64')
    #sent_placehodler = P.zeros([2, 2], dtype='int64')
    #P.jit.save(model, args.inference_model_dir, input_var=[src_placeholder, sent_placehodler])
    log.debug('done')
