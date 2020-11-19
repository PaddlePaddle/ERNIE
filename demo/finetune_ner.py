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
import six
import json
from random import random
from tqdm import tqdm
from collections import OrderedDict
from functools import reduce, partial

import numpy as np
import multiprocessing
import pickle
import logging

from sklearn.metrics import f1_score
import paddle
import paddle.fluid as F
import paddle.fluid.dygraph as FD
import paddle.fluid.layers as L

from propeller import log
import propeller.paddle as propeller

log.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

from ernie.modeling_ernie import ErnieModel, ErnieModelForSequenceClassification, ErnieModelForTokenClassification
from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.optimization import AdamW, LinearDecay

if __name__ == '__main__':
    parser = propeller.ArgumentParser('NER model with ERNIE')
    parser.add_argument('--max_seqlen', type=int, default=256)
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=6)
    parser.add_argument(
        '--warmup_proportion',
        type=float,
        default=0.1,
        help='if use_lr_decay is set, '
        'learning rate will raise to `lr` at `warmup_proportion` * `max_steps` and decay to 0. at `max_steps`'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        required=True,
        help='max_train_steps, set this to EPOCH * NUM_SAMPLES / BATCH_SIZE, used in learning rate scheduler'
    )
    parser.add_argument('--from_pretrained', type=str, required=True)
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument(
        '--save_dir', type=str, default=None, help='model output directory')
    parser.add_argument(
        '--wd',
        type=float,
        default=0.01,
        help='weight decay, aka L2 regularizer')
    args = parser.parse_args()

    tokenizer = ErnieTokenizer.from_pretrained(args.from_pretrained)

    def tokenizer_func(inputs):
        ret = inputs.split(b'\2')
        tokens, orig_pos = [], []
        for i, r in enumerate(ret):
            t = tokenizer.tokenize(r)
            for tt in t:
                tokens.append(tt)
                orig_pos.append(i)
        assert len(tokens) == len(orig_pos)
        return tokens + orig_pos

    def tokenizer_func_for_label(inputs):
        return inputs.split(b'\2')

    feature_map = {
        b"B-PER": 0,
        b"I-PER": 1,
        b"B-ORG": 2,
        b"I-ORG": 3,
        b"B-LOC": 4,
        b"I-LOC": 5,
        b"O": 6,
    }
    other_tag_id = feature_map[b'O']

    feature_column = propeller.data.FeatureColumns([
        propeller.data.TextColumn(
            'text_a',
            unk_id=tokenizer.unk_id,
            vocab_dict=tokenizer.vocab,
            tokenizer=tokenizer_func), propeller.data.TextColumn(
                'label',
                unk_id=other_tag_id,
                vocab_dict=feature_map,
                tokenizer=tokenizer_func_for_label, )
    ])

    def before(seg, label):
        seg, orig_pos = np.split(seg, 2)
        aligned_label = label[orig_pos]
        seg, _ = tokenizer.truncate(seg, [], args.max_seqlen)
        aligned_label, _ = tokenizer.truncate(aligned_label, [],
                                              args.max_seqlen)
        orig_pos, _ = tokenizer.truncate(orig_pos, [], args.max_seqlen)

        sentence, segments = tokenizer.build_for_ernie(
            seg
        )  #utils.data.build_1_pair(seg, max_seqlen=args.max_seqlen, cls_id=cls_id, sep_id=sep_id)
        aligned_label = np.concatenate([[0], aligned_label, [0]], 0)
        orig_pos = np.concatenate([[0], orig_pos, [0]])

        assert len(aligned_label) == len(sentence) == len(orig_pos), (
            len(aligned_label), len(sentence), len(orig_pos))  # alinged
        return sentence, segments, aligned_label, label, orig_pos

    train_ds = feature_column.build_dataset('train', data_dir=os.path.join(args.data_dir, 'train'), shuffle=True, repeat=False, use_gz=False) \
                                   .map(before) \
                                   .padded_batch(args.bsz, (0,0,0, other_tag_id + 1, 0)) \

    dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(args.data_dir, 'dev'), shuffle=False, repeat=False, use_gz=False) \
                                   .map(before) \
                                   .padded_batch(args.bsz, (0,0,0, other_tag_id + 1,0)) \

    test_ds = feature_column.build_dataset('test', data_dir=os.path.join(args.data_dir, 'test'), shuffle=False, repeat=False, use_gz=False) \
                                   .map(before) \
                                   .padded_batch(args.bsz, (0,0,0, other_tag_id + 1,0)) \

    shapes = ([-1, args.max_seqlen], [-1, args.max_seqlen],
              [-1, args.max_seqlen])
    types = ('int64', 'int64', 'int64')

    train_ds.data_shapes = shapes
    train_ds.data_types = types
    dev_ds.data_shapes = shapes
    dev_ds.data_types = types
    test_ds.data_shapes = shapes
    test_ds.data_types = types

    place = F.CUDAPlace(0)

    @FD.no_grad
    def evaluate(model, dataset):
        model.eval()
        chunkf1 = propeller.metrics.ChunkF1(None, None, None, len(feature_map))
        for step, (ids, sids, aligned_label, label,
                   orig_pos) in enumerate(tqdm(dataset.start(place))):
            loss, logits = model(ids, sids)
            #print('\n'.join(map(str, logits.numpy().tolist())))

            assert orig_pos.shape[0] == logits.shape[0] == ids.shape[
                0] == label.shape[0]
            for pos, lo, la, id in zip(orig_pos.numpy(),
                                       logits.numpy(),
                                       label.numpy(), ids.numpy()):
                _dic = OrderedDict()
                assert len(pos) == len(lo) == len(id)
                for _pos, _lo, _id in zip(pos, lo, id):
                    if _id > tokenizer.mask_id:  # [MASK] is the largest special token
                        _dic.setdefault(_pos, []).append(_lo)
                merged_lo = np.array(
                    [np.array(l).mean(0) for _, l in six.iteritems(_dic)])
                merged_preds = np.argmax(merged_lo, -1)
                la = la[np.where(la != (other_tag_id + 1))]  #remove pad
                if len(la) > len(merged_preds):
                    log.warn(
                        'accuracy loss due to truncation: label len:%d, truncate to %d'
                        % (len(la), len(merged_preds)))
                    merged_preds = np.pad(merged_preds,
                                          [0, len(la) - len(merged_preds)],
                                          mode='constant',
                                          constant_values=7)
                else:
                    assert len(la) == len(
                        merged_preds
                    ), 'expect label == prediction, got %d vs %d' % (
                        la.shape, merged_preds.shape)
                chunkf1.update((merged_preds, la, np.array(len(la))))
        #f1 = f1_score(np.concatenate(all_label), np.concatenate(all_pred), average='macro')
        f1 = chunkf1.eval()
        model.train()
        return f1

    with FD.guard(place):
        model = ErnieModelForTokenClassification.from_pretrained(
            args.from_pretrained,
            num_labels=len(feature_map),
            name='',
            has_pooler=False)

        g_clip = F.clip.GradientClipByGlobalNorm(1.0)  #experimental
        opt = AdamW(
            learning_rate=LinearDecay(args.lr,
                                      int(args.warmup_proportion *
                                          args.max_steps), args.max_steps),
            parameter_list=model.parameters(),
            weight_decay=args.wd,
            grad_clip=g_clip)
        #opt = F.optimizer.AdamOptimizer(learning_rate=LinearDecay(args.lr, args.warmup_steps, args.max_steps), parameter_list=model.parameters())
        for epoch in range(args.epoch):
            for step, (ids, sids, aligned_label, label,
                       orig_pos) in enumerate(tqdm(train_ds.start(place))):
                loss, logits = model(
                    ids,
                    sids,
                    labels=aligned_label,
                    loss_weights=L.cast(
                        ids > tokenizer.mask_id,
                        'float32'))  # [MASK] is the largest special token
                loss.backward()
                if step % 10 == 0:
                    log.debug('train loss %.5f, lr %.3e' %
                              (loss.numpy(), opt.current_step_lr()))
                opt.minimize(loss)
                model.clear_gradients()
                if step % 100 == 0:
                    f1 = evaluate(model, dev_ds)
                    log.debug('eval f1: %.5f' % f1)

        f1 = evaluate(model, dev_ds)
        log.debug('final eval f1: %.5f' % f1)
        if args.save_dir is not None:
            F.save_dygraph(model.state_dict(), args.save_dir)
