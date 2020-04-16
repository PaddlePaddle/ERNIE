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
import multiprocessing
import pickle
import jieba
import logging

import paddle
import paddle.fluid as F
import paddle.fluid.dygraph as FD
import paddle.fluid.layers as L

import utils.data


from propeller import log
import propeller.paddle as propeller

log.setLevel(logging.DEBUG)
logging.getLogger().addHandler(log.handlers[0])
logging.getLogger().setLevel(logging.DEBUG)


#from model.bert import BertConfig, BertModelLayer
from modeling_ernie import ErnieModel, ErnieModelForSequenceClassification
from tokenizing_ernie import ErnieTokenizer
from optimization import AdamW


class LinearDecay(FD.learning_rate_scheduler.LearningRateDecay):
    def __init__(self,
                 learning_rate,
                 warmup_steps,
                 decay_steps,
                 end_learning_rate=0,
                 power=1.0,
                 cycle=False,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(LinearDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle

    def step(self):
        if self.step_num < self.warmup_steps:
            decayed_lr = self.learning_rate * (self.step_num /
                                               self.warmup_steps)
            decayed_lr = self.create_lr_var(decayed_lr)
        else:
            tmp_step_num = self.step_num
            tmp_decay_steps = self.decay_steps
            if self.cycle:
                div_res = fluid.layers.ceil(
                    self.create_lr_var(tmp_step_num / float(self.decay_steps)))
                if tmp_step_num == 0:
                    div_res = self.create_lr_var(1.0)
                tmp_decay_steps = self.decay_steps * div_res
            else:
                tmp_step_num = self.create_lr_var(
                    tmp_step_num
                    if tmp_step_num < self.decay_steps else self.decay_steps)
                decayed_lr = (self.learning_rate - self.end_learning_rate) * \
                    ((1 - tmp_step_num / tmp_decay_steps) ** self.power) + self.end_learning_rate

        return decayed_lr


if __name__ == '__main__':
    parser = propeller.ArgumentParser('classify model with ERNIE')
    parser.add_argument('--max_seqlen', type=int, default=256)
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warm_start_from', type=str)
    parser.add_argument('--sentence_piece_model', type=str, default=None)
    args = parser.parse_args()

    vocab = {j.strip().decode('utf8'): i for i, j in enumerate(open(os.path.join(args.data_dir, 'vocab.txt'), 'rb'))}
    sep_id = vocab['[SEP]']
    cls_id = vocab['[CLS]']
    unk_id = vocab['[UNK]']

    if args.sentence_piece_model is not None:
        tokenizer = utils.data.JBSPTokenizer(args.sentence_piece_model, jb=True, lower=True)
    else:
        tokenizer = utils.data.CharTokenizer(vocab.keys())

    #tokenizer = ErnieTokenizer.from_pretrained('./pretrained/')

    def tokenizer_func(inputs):
        '''avoid pickle error'''
        ret = tokenizer(inputs)
        return ret

    feature_column = propeller.data.FeatureColumns([
        propeller.data.TextColumn('title',unk_id=unk_id, vocab_dict=vocab, tokenizer=tokenizer_func),
        propeller.data.LabelColumn('label'),
    ])

    def before(seg_a, label):
        sentence, segments = utils.data.build_1_pair(seg_a, max_seqlen=args.max_seqlen, cls_id=cls_id, sep_id=sep_id)
        return sentence, segments, label


    log.debug(os.path.join(args.data_dir, 'train'))
    train_ds = feature_column.build_dataset('train', data_dir=os.path.join(args.data_dir, 'train'), shuffle=True, repeat=False, use_gz=False) \
                                   .map(before) \
                                   .padded_batch(args.bsz, (0, 0, 0)) 

    dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(args.data_dir, 'dev'), shuffle=False, repeat=False, use_gz=False) \
                                   .map(before) \
                                   .padded_batch(args.bsz, (0, 0, 0)) 

    test_ds = feature_column.build_dataset('test', data_dir=os.path.join(args.data_dir, 'test'), shuffle=False, repeat=False, use_gz=False) \
                                   .map(before) \
                                   .padded_batch(args.bsz, (0, 0, 0)) 



    shapes = ([-1, args.max_seqlen], [-1, args.max_seqlen], [-1, 1])
    types = ('int64', 'int64', 'int64')

    train_ds.data_shapes = shapes
    train_ds.data_types = types
    dev_ds.data_shapes = shapes
    dev_ds.data_types = types
    test_ds.data_shapes = shapes
    test_ds.data_types = types


    with FD.guard():
        model = ErnieModelForSequenceClassification.from_pretrained('./pretrained/', num_labels=2, name='')

        opt = AdamW(learning_rate=LinearDecay(args.lr, args.warmup_steps, args.max_steps), parameter_list=model.parameters(), weight_decay=0.01)
        for epoch in range(args.epoch):
            for step, d in enumerate(tqdm(train_ds.start())):
                ids, sids, label = d
                loss, _ = model(ids, sids, labels=label)
                loss.backward()
                if step % 10 == 0 :
                    log.debug(loss.numpy())
                opt.minimize(loss)
                model.clear_gradients()
                if step % 100 == 0 :
                    acc = []
                    with FD.base._switch_tracer_mode_guard_(is_train=False):
                        for step, d in enumerate(tqdm(dev_ds.start())):
                            ids, sids, label = d
                            loss, logits = model(ids, sids, labels=label)
                            #print('\n'.join(map(str, logits.numpy().tolist())))
                            a = L.argmax(logits, -1) == L.squeeze(label, axes=[-1])
                            acc.append(a.numpy())
                    log.debug(np.concatenate(acc).mean())

        #F.save_dygraph(model.state_dict(), './saved')
        ##F.save_dygraph(opt.state_dict(), './saved')



