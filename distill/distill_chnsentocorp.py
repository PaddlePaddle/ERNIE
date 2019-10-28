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
from random import random
from functools import reduce, partial
import logging

import numpy as np
import multiprocessing

import paddle
import paddle.fluid as F
import paddle.fluid.layers as L
from propeller import log
import propeller.paddle as propeller
from propeller.paddle.data import Dataset

from optimization import optimization
import utils.data

log.setLevel(logging.DEBUG)

class ClassificationBowModel(propeller.train.Model):
    """propeller Model wraper for paddle-ERNIE """
    def __init__(self, config, mode, run_config):
        self.config = config
        self.mode = mode
        self.run_config = run_config
        self._param_initializer = F.initializer.TruncatedNormal(
            scale=config.initializer_range)
        self._emb_dtype = "float32"
        self._word_emb_name = "word_embedding"

    def forward(self, features):
        text_ids_a, = features

        def bow(ids):
            embed = L.embedding(
                input=ids,
                size=[self.config.vocab_size, self.config.emb_size],
                dtype=self._emb_dtype,
                param_attr=F.ParamAttr(
                    name=self._word_emb_name, initializer=self._param_initializer),
                is_sparse=False)

            zero = L.fill_constant(shape=[1], dtype='int64', value=0)
            pad = L.cast(L.logical_not(L.equal(ids, zero)), 'float32')
            sumed = L.reduce_sum(embed * pad, dim=1)
            sumed = L.softsign(sumed)
            return sumed

        sumed = bow(text_ids_a)

        fced = L.fc(
            input=sumed,
            size=self.config.emb_size,
            act='tanh',
            param_attr=F.ParamAttr(
                name="middle_fc.w_0", initializer=self._param_initializer),
            bias_attr="middle_fc.b_0")

        logits = L.fc(
            input=fced,
            size=self.config.num_label,
            act=None,
            param_attr=F.ParamAttr(
                name="pooler_fc.w_0", initializer=self._param_initializer),
            bias_attr="pooler_fc.b_0")


        if self.mode is propeller.RunMode.PREDICT:
            probs = L.softmax(logits)
            return probs
        else:
            return logits


    def loss(self, predictions, labels):
        labels = L.softmax(labels)
        loss = L.softmax_with_cross_entropy(predictions, labels, soft_label=True)
        loss = L.mean(loss)
        return loss

    def backward(self, loss):
        scheduled_lr, _ = optimization(
            loss=loss,
            warmup_steps=int(self.run_config.max_steps * self.config.warmup_proportion),
            num_train_steps=self.run_config.max_steps,
            learning_rate=self.config.learning_rate,
            train_program=F.default_main_program(), 
            startup_prog=F.default_startup_program(),
            weight_decay=self.config.weight_decay,
            scheduler="linear_warmup_decay",)
        propeller.summary.scalar('lr', scheduled_lr)


    def metrics(self, predictions, labels):
        predictions = L.argmax(predictions, axis=1)
        labels = L.argmax(labels, axis=1)
        #predictions = L.unsqueeze(predictions, axes=[1])
        acc = propeller.metrics.Acc(labels, predictions)
        #auc = propeller.metrics.Auc(labels, predictions)
        return {'acc': acc}

if __name__ == '__main__':
    parser = propeller.ArgumentParser('Distill model with Paddle')
    parser.add_argument('--max_seqlen', type=int, default=128)
    parser.add_argument('--vocab_file', type=str, required=True)
    parser.add_argument('--unsupervise_data_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str)
    args = parser.parse_args()
    run_config = propeller.parse_runconfig(args)
    hparams = propeller.parse_hparam(args)

    vocab = {j.strip().split(b'\t')[0].decode('utf8'): i for i, j in enumerate(open(args.vocab_file, 'rb'))}
    unk_id = vocab['[UNK]']

    char_tokenizer = utils.data.CharTokenizer(vocab.keys())
    space_tokenizer = utils.data.SpaceTokenizer(vocab.keys())

    supervise_feature_column = propeller.data.FeatureColumns([
        propeller.data.TextColumn('text_a', unk_id=unk_id, vocab_dict=vocab, tokenizer=space_tokenizer),
        propeller.data.LabelColumn('label'),
    ])

    def before(text_a, label):
        sentence_a = text_a[: args.max_seqlen]
        return sentence_a, label

    def after(sentence_a, label):
        batch_size = sentence_a.shape[0]
        onehot_label = np.zeros([batch_size, hparams.num_label], dtype=np.float32)
        onehot_label[np.arange(batch_size), label] = 9999.
        sentence_a, = utils.data.expand_dims(sentence_a)
        return sentence_a, onehot_label


    train_ds = supervise_feature_column.build_dataset('train', data_dir=os.path.join(args.data_dir, 'train'), shuffle=True, repeat=True, use_gz=False) \
                                   .map(before) \
                                   .padded_batch(hparams.batch_size, (0, 0)) \
                                   .map(after) \

    unsup_train_ds = supervise_feature_column.build_dataset('unsup_train', data_dir=args.unsupervise_data_dir, shuffle=True, repeat=True, use_gz=False) \
                                   .map(before) \
                                   .padded_batch(hparams.batch_size, (0, 0)) \
                                   .map(after) 

    dev_ds = supervise_feature_column.build_dataset('dev', data_dir=os.path.join(args.data_dir, 'dev'), shuffle=False, repeat=False, use_gz=False) \
                                   .map(before) \
                                   .padded_batch(hparams.batch_size, (0, 0)) \
                                   .map(after)

    train_ds = utils.data.interleave(train_ds, unsup_train_ds)

    shapes = ([-1, args.max_seqlen, 1], [-1, hparams.num_label]) 
    types = ('int64', 'float32')

    train_ds.data_shapes = shapes
    train_ds.data_types = types

    dev_ds.data_shapes = shapes
    dev_ds.data_types = types

    '''
    from tqdm import tqdm
    for slots in tqdm(train_ds):
        pass
    '''

    best_exporter = propeller.train.exporter.BestExporter(os.path.join(run_config.model_dir, 'best'), cmp_fn=lambda old, new: new['dev']['acc'] > old['dev']['acc'])
    propeller.train.train_and_eval(
            model_class_or_model_fn=ClassificationBowModel, 
            params=hparams, 
            run_config=run_config, 
            train_dataset=train_ds, 
            eval_dataset={'dev': dev_ds}, 
            exporters=[best_exporter])
    print('dev_acc3\t%.5f' % (best_exporter._best['dev']['acc']))

