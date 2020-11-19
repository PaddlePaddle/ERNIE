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

import os
import re
import time
import logging
import json
from random import random
from tqdm import tqdm
from functools import reduce, partial
import pickle
import argparse

import numpy as np
import logging

import paddle
import paddle.fluid as F
import paddle.fluid.dygraph as D
import paddle.fluid.layers as L

from propeller import log
import propeller.paddle as propeller

from ernie.modeling_ernie import ErnieModel, ErnieModelForQuestionAnswering
from ernie.tokenizing_ernie import ErnieTokenizer, ErnieTinyTokenizer
from ernie.optimization import AdamW, LinearDecay

from demo.mrc import mrc_reader
from demo.mrc import mrc_metrics

log.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)


def evaluate(model, ds, all_examples, all_features, tokenizer, args):
    dev_file = json.loads(open(args.dev_file).read())
    with D.base._switch_tracer_mode_guard_(is_train=False):
        log.debug('start eval')
        model.eval()
        all_res = []
        for step, (uids, token_ids, token_type_ids, _,
                   __) in enumerate(ds.start(place)):
            _, start_logits, end_logits = model(token_ids, token_type_ids)
            res = [
                mrc_metrics.RawResult(
                    unique_id=u, start_logits=s, end_logits=e)
                for u, s, e in zip(uids.numpy(),
                                   start_logits.numpy(), end_logits.numpy())
            ]
            all_res += res
        open('all_res', 'wb').write(pickle.dumps(all_res))
        all_pred, all_nbests = mrc_metrics.make_results(
            tokenizer,
            all_examples,
            all_features,
            all_res,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            do_lower_case=tokenizer.lower)
        f1, em, _, __ = mrc_metrics.evaluate(dev_file, all_pred)
        model.train()
        log.debug('done eval')
        return f1, em


def train(model, train_dataset, dev_dataset, dev_examples, dev_features,
          tokenizer, args):
    ctx = D.parallel.prepare_context()
    model = D.parallel.DataParallel(model, ctx)

    max_steps = len(train_features) * args.epoch // args.bsz
    g_clip = F.clip.GradientClipByGlobalNorm(1.0)  #experimental
    opt = AdamW(
        learning_rate=args.lr,
        parameter_list=model.parameters(),
        weight_decay=args.wd,
        grad_clip=g_clip)

    train_dataset = train_dataset \
            .repeat() \
            .shard(D.parallel.Env().nranks, D.parallel.Env().dev_id) \
            .shuffle(1000) \
            .padded_batch(args.bsz)

    log.debug('init training with args: %s' % repr(args))
    for step, (_, token_ids, token_type_ids, start_pos,
               end_pos) in enumerate(train_dataset.start(place)):
        loss, _, __ = model(
            token_ids, token_type_ids, start_pos=start_pos, end_pos=end_pos)
        scaled_loss = model.scale_loss(loss)
        scaled_loss.backward()
        model.apply_collective_grads()
        opt.minimize(scaled_loss)
        model.clear_gradients()
        if D.parallel.Env().dev_id == 0 and step % 10 == 0:
            log.debug('[step %d] train loss %.5f lr %.3e' %
                      (step, loss.numpy(), opt.current_step_lr()))
        if D.parallel.Env().dev_id == 0 and step % 100 == 0:
            f1, em = evaluate(model, dev_dataset, dev_examples, dev_features,
                              tokenizer, args)
            log.debug('[step %d] eval result: f1 %.5f em %.5f' %
                      (step, f1, em))
        if step > max_steps:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser('MRC model with ERNIE')
    parser.add_argument(
        '--from_pretrained',
        type=str,
        required=True,
        help='pretrained model directory or tag')
    parser.add_argument(
        '--max_seqlen',
        type=int,
        default=512,
        help='max sentence length, should not greater than 512')
    parser.add_argument('--bsz', type=int, default=8, help='batchsize')
    parser.add_argument('--epoch', type=int, default=2, help='epoch')
    parser.add_argument(
        '--train_file',
        type=str,
        required=True,
        help='data directory includes train / develop data')
    parser.add_argument(
        '--dev_file',
        type=str,
        required=True,
        help='data directory includes train / develop data')
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument(
        '--save_dir', type=str, default=None, help='model output directory')
    parser.add_argument(
        '--n_best_size', type=int, default=20, help='nbest prediction to keep')
    parser.add_argument(
        '--max_answer_length', type=int, default=100, help='max answer span')
    parser.add_argument(
        '--wd',
        type=float,
        default=0.00,
        help='weight decay, aka L2 regularizer')

    args = parser.parse_args()

    tokenizer = ErnieTokenizer.from_pretrained(args.from_pretrained)

    if not os.path.exists(args.train_file):
        raise RuntimeError('input data not found at %s' % args.train_file)
    if not os.path.exists(args.dev_file):
        raise RuntimeError('input data not found at %s' % args.dev_file)

    log.info('making train/dev data...')
    train_examples = mrc_reader.read_files(args.train_file, is_training=True)
    train_features = mrc_reader.convert_example_to_features(
        train_examples, args.max_seqlen, tokenizer, is_training=True)

    dev_examples = mrc_reader.read_files(args.dev_file, is_training=False)
    dev_features = mrc_reader.convert_example_to_features(
        dev_examples, args.max_seqlen, tokenizer, is_training=False)

    log.info('train examples: %d, features: %d' %
             (len(train_examples), len(train_features)))

    def map_fn(unique_id, example_index, doc_span_index, tokens,
               token_to_orig_map, token_is_max_context, token_ids,
               position_ids, text_type_ids, start_position, end_position):
        if start_position is None:
            start_position = 0
        if end_position is None:
            end_position = 0
        return np.array(unique_id), np.array(token_ids), np.array(
            text_type_ids), np.array(start_position), np.array(end_position)

    train_dataset = propeller.data.Dataset.from_list(train_features).map(
        map_fn)

    dev_dataset = propeller.data.Dataset.from_list(dev_features).map(
        map_fn).padded_batch(args.bsz)
    shapes = ([-1], [-1, args.max_seqlen], [-1, args.max_seqlen], [-1], [-1])
    types = ('int64', 'int64', 'int64', 'int64', 'int64')

    train_dataset.name = 'train'
    dev_dataset.name = 'dev'

    train_dataset.data_shapes = shapes
    train_dataset.data_types = types
    dev_dataset.data_shapes = shapes
    dev_dataset.data_types = types

    place = F.CUDAPlace(D.parallel.Env().dev_id)
    D.guard(place).__enter__()
    model = ErnieModelForQuestionAnswering.from_pretrained(
        args.from_pretrained, name='')

    train(model, train_dataset, dev_dataset, dev_examples, dev_features,
          tokenizer, args)

    if D.parallel.Env().dev_id == 0:
        f1, em = evaluate(model, dev_dataset, dev_examples, dev_features,
                          tokenizer, args)
        log.debug('final eval result: f1 %.5f em %.5f' % (f1, em))
    if D.parallel.Env().dev_id == 0 and args.save_dir is not None:
        F.save_dygraph(model.state_dict(), args.save_dir)
