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
import time
import logging
import json
import re
from random import random
from functools import reduce, partial

import numpy as np
import logging
#from visualdl import LogWriter

from pathlib import Path
import paddle as P
from propeller import log
import propeller.paddle as propeller

#from model.bert import BertConfig, BertModelLayer
from ernie.modeling_ernie import ErnieModel, ErnieModelForSequenceClassification
from ernie.tokenizing_ernie import ErnieTokenizer, ErnieTinyTokenizer
#from ernie.optimization import AdamW, LinearDecay
from demo.utils import UnpackDataLoader, create_if_not_exists, get_warmup_and_linear_decay

log.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

if __name__ == '__main__':
    parser = propeller.ArgumentParser('classify model with ERNIE')
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
    parser.add_argument(
        '--save_dir', type=Path, default=None, help='model output directory')
    parser.add_argument(
        '--wd',
        type=int,
        default=0.01,
        help='weight decay, aka L2 regularizer')
    parser.add_argument(
        '--init_checkpoint',
        type=str,
        default=None,
        help='checkpoint to warm start from')
    parser.add_argument(
        '--max_bsz',
        type=str,
        default=32,
        help='maximum batch size for each rank, if bsz > max_bsz, '
        'will performe gradient accumulate')

    parser.add_argument(
        '--use_amp',
        action='store_true',
        help='only activate AMP(auto mixed precision accelatoin) on TensorCore compatible devices'
    )

    args = parser.parse_args()
    env = P.distributed.ParallelEnv()

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
            'label', vocab_dict={
                b"0": 0,
                b"1": 1,
                b"2": 2,
            }),
    ])

    def map_fn(seg_a, seg_b, label):
        seg_a, seg_b = tokenizer.truncate(seg_a, seg_b, seqlen=args.max_seqlen)
        sentence, segments = tokenizer.build_for_ernie(seg_a, seg_b)
        return sentence, segments, label

    assert args.bsz % env.nranks == 0,  \
        'assume batch_size %d to be equally split on nranks %d' % (args.bsz, env.nranks)
    bsz_per_rank = args.bsz // env.nranks
    assert bsz_per_rank > args.max_bsz or bsz_per_rank % args.max_bsz == 0,  \
        'cannot do gradient accumulate, bsz:%d, max_bsz:%d nranks:%d' % (args.bsz, args.max_bsz, env.nrank)
    args.bsz = min(bsz_per_rank, args.max_bsz)
    acc_steps = bsz_per_rank // args.max_bsz
    if acc_steps > 1:
        log.info('doing gradient accumulate, acc_steps:%d, actual_bsz:%d' %
                 (acc_steps, args.bsz))

    train_ds = feature_column.build_dataset('train', data_dir=os.path.join(args.data_dir, 'train'),
                                                shuffle=False, repeat=True, use_gz=False) \
                                   .map(map_fn) \
                                   .padded_batch(args.bsz, (0, 0, 0))
    train_ds = train_ds.shard(env.nranks, env.dev_id)
    log.debug('sharding %d/%d' % (env.nranks, env.dev_id))
    train_ds = train_ds.shuffle(10000)

    dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(args.data_dir, 'dev'),
                                            shuffle=False, repeat=False, use_gz=False) \
                                   .map(map_fn) \
                                   .padded_batch(args.bsz, (0, 0, 0))

    shapes = ([-1, args.max_seqlen], [-1, args.max_seqlen], [-1])
    types = ('int64', 'int64', 'int64')

    place = P.CUDAPlace(env.dev_id)
    P.distributed.init_parallel_env()
    env = P.distributed.ParallelEnv()
    model = ErnieModelForSequenceClassification.from_pretrained(
        args.from_pretrained, num_labels=3, name='')

    if args.init_checkpoint is not None:
        log.info('loading checkpoint from %s' % args.init_checkpoint)
        sd, _ = P.load(args.init_checkpoint)
        model.set_state_dict(sd)

    model = P.DataParallel(model)

    g_clip = P.nn.ClipGradByGlobalNorm(1.0)  #experimental
    param_name_to_exclue_from_weight_decay = re.compile(
        r'.*layer_norm_scale|.*layer_norm_bias|.*b_0')

    lr_scheduler = P.optimizer.lr.LambdaDecay(
        args.lr,
        get_warmup_and_linear_decay(
            args.max_steps, int(args.warmup_proportion * args.max_steps)))
    opt = P.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        apply_decay_param_fun=lambda n: param_name_to_exclue_from_weight_decay.match(n),
        weight_decay=args.wd,
        grad_clip=g_clip)
    scaler = P.amp.GradScaler(enable=args.use_amp)
    inner_step, step = 0, 0
    create_if_not_exists(args.save_dir)
    #with LogWriter(logdir=str(create_if_not_exists(args.save_dir / 'vdl-%d' % env.dev_id))) as log_writer:

    for ids, sids, label in UnpackDataLoader(
            train_ds, places=P.CUDAPlace(env.dev_id)):
        inner_step += 1
        loss, _ = model(ids, sids, labels=label)
        loss /= acc_steps
        loss.backward()
        if inner_step % acc_steps != 0:
            continue  # do grad average
        lr_scheduler.step()
        opt.step()
        step += 1
        model.clear_gradients()

        # do logging
        if step % 10 == 0:
            _lr = lr_scheduler.get_lr()
            if args.use_amp:
                _l = (loss / scaler._scale).numpy()
                msg = '[rank-%d][step-%d] train loss %.5f lr %.3e scaling %.3e' % (
                    env.dev_id, step, _l, _lr, scaler._scale.numpy())
            else:
                _l = loss.numpy()
                msg = '[rank-%d][step-%d] train loss %.5f lr %.3e' % (
                    env.dev_id, step, _l, _lr)
            log.debug(msg)
            #log_writer.add_scalar('loss', _l, step=step)
            #log_writer.add_scalar('lr', _lr, step=step)

        # do saving
        if step % 100 == 0 and env.dev_id == 0:
            acc = []
            with P.no_grad():
                model.eval()
                for d in UnpackDataLoader(
                        dev_ds, places=P.CUDAPlace(env.dev_id)):
                    ids, sids, label = d
                    loss, logits = model(ids, sids, labels=label)
                    a = (logits.argmax(-1) == label)
                    acc.append(a.numpy())
                model.train()
            acc = np.concatenate(acc).mean()
            #log_writer.add_scalar('eval/acc', acc, step=step)
            log.debug('acc %.5f' % acc)
            if args.save_dir is not None:
                P.save(model.state_dict(), args.save_dir / 'ckpt.bin')
        # exit 
        if step > args.max_steps:
            break

    if args.save_dir is not None and env.dev_id == 0:
        P.save(model.state_dict(), args.save_dir / 'ckpt.bin')
    log.debug('done')
