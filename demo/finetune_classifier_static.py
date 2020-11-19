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
from __future__ import absolute_import

import os
import re
import time
import logging
from random import random
import json
from functools import reduce, partial

import numpy as np
import multiprocessing
import tempfile
import re

import paddle
import paddle.fluid as F
import paddle.fluid.layers as L

from ernie.modeling_ernie import ErnieModel, ErnieModelForSequenceClassification
from ernie.tokenizing_ernie import ErnieTokenizer, ErnieTinyTokenizer
from ernie.optimization import optimization
#import utils.data

from propeller import log
import propeller.paddle as propeller

log.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)


def model_fn(features, mode, params, run_config):
    ernie = ErnieModelForSequenceClassification(params, name='')
    if not params is propeller.RunMode.TRAIN:
        ernie.eval()

    metrics, loss = None, None
    if mode is propeller.RunMode.PREDICT:
        src_ids, sent_ids = features
        _, logits = ernie(src_ids, sent_ids)
        predictions = [logits, ]
    else:
        src_ids, sent_ids, labels = features
        if mode is propeller.RunMode.EVAL:
            loss, logits = ernie(src_ids, sent_ids, labels=labels)
            pred = L.argmax(logits, axis=1)
            acc = propeller.metrics.Acc(labels, pred)
            metrics = {'acc': acc}
            predictions = [pred]
        else:
            loss, logits = ernie(src_ids, sent_ids, labels=labels)
            scheduled_lr, _ = optimization(
                loss=loss,
                warmup_steps=int(run_config.max_steps *
                                 params['warmup_proportion']),
                num_train_steps=run_config.max_steps,
                learning_rate=params['learning_rate'],
                train_program=F.default_main_program(),
                startup_prog=F.default_startup_program(),
                use_fp16=params.use_fp16,
                weight_decay=params['weight_decay'],
                scheduler="linear_warmup_decay", )
            propeller.summary.scalar('lr', scheduled_lr)
            predictions = [logits, ]

    return propeller.ModelSpec(
        loss=loss, mode=mode, metrics=metrics, predictions=predictions)


if __name__ == '__main__':
    parser = propeller.ArgumentParser('DAN model with Paddle')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--max_seqlen', type=int, default=128)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--from_pretrained', type=str, required=True)
    parser.add_argument('--warm_start_from', type=str)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--use_fp16', action='store_true')

    args = parser.parse_args()

    if not os.path.exists(args.from_pretrained):
        raise ValueError('--from_pretrained not found: %s' %
                         args.from_pretrained)
    cfg_file_path = os.path.join(args.from_pretrained, 'ernie_config.json')
    param_path = os.path.join(args.from_pretrained, 'params')
    vocab_path = os.path.join(args.from_pretrained, 'vocab.txt')

    assert os.path.exists(cfg_file_path) and os.path.exists(
        param_path) and os.path.exists(vocab_path)

    hparams_cli = propeller.parse_hparam(args)
    hparams_config_file = json.loads(open(cfg_file_path).read())
    default_hparams = propeller.HParams(
        batch_size=32,
        num_labels=3,
        warmup_proportion=0.1,
        learning_rate=5e-5,
        weight_decay=0.01,
        use_task_id=False,
        use_fp16=args.use_fp16, )

    hparams = default_hparams.join(propeller.HParams(
        **hparams_config_file)).join(hparams_cli)

    default_run_config = dict(
        max_steps=args.epoch * 390000 / hparams.batch_size,
        save_steps=1000,
        log_steps=10,
        max_ckpt=1,
        skip_steps=0,
        model_dir=tempfile.mkdtemp(),
        eval_steps=100)
    run_config = dict(default_run_config, **json.loads(args.run_config))
    run_config = propeller.RunConfig(**run_config)

    tokenizer = ErnieTokenizer.from_pretrained(args.from_pretrained)
    #tokenizer = ErnieTinyTokenizer.from_pretrained(args.from_pretrained)
    unk_id = tokenizer.vocab['[UNK]']

    shapes = ([-1, args.max_seqlen], [-1, args.max_seqlen], [-1])
    types = ('int64', 'int64', 'int64')
    if not args.do_predict:
        feature_column = propeller.data.FeatureColumns([
            propeller.data.TextColumn(
                'title',
                unk_id=unk_id,
                vocab_dict=tokenizer.vocab,
                tokenizer=tokenizer.tokenize),
            propeller.data.TextColumn(
                'comment',
                unk_id=unk_id,
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
            seg_a, seg_b = tokenizer.truncate(
                seg_a, seg_b, seqlen=args.max_seqlen)
            sentence, segments = tokenizer.build_for_ernie(seg_a, seg_b)
            #label = np.expand_dims(label, -1) #
            return sentence, segments, label

        train_ds = feature_column.build_dataset('train', data_dir=os.path.join(args.data_dir, 'train'), shuffle=True, repeat=True, use_gz=False) \
                                       .map(map_fn) \
                                       .padded_batch(hparams.batch_size)

        dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(args.data_dir, 'dev'), shuffle=False, repeat=False, use_gz=False) \
                                       .map(map_fn) \
                                       .padded_batch(hparams.batch_size)

        test_ds = feature_column.build_dataset('test', data_dir=os.path.join(args.data_dir, 'test'), shuffle=False, repeat=False, use_gz=False) \
                                       .map(map_fn) \
                                       .padded_batch(hparams.batch_size) \

        train_ds.data_shapes = shapes
        train_ds.data_types = types
        dev_ds.data_shapes = shapes
        dev_ds.data_types = types
        test_ds.data_shapes = shapes
        test_ds.data_types = types

        varname_to_warmstart = re.compile(
            r'^encoder.*[wb]_0$|^.*embedding$|^.*bias$|^.*scale$|^pooled_fc.[wb]_0$'
        )

        ws = propeller.WarmStartSetting(
                predicate_fn=lambda v: varname_to_warmstart.match(v.name) and os.path.exists(os.path.join(param_path, v.name)),
                from_dir=param_path,
            )

        best_exporter = propeller.train.exporter.BestExporter(
            os.path.join(run_config.model_dir, 'best'),
            cmp_fn=lambda old, new: new['dev']['acc'] > old['dev']['acc'])
        propeller.train.train_and_eval(
            model_class_or_model_fn=model_fn,
            params=hparams,
            run_config=run_config,
            train_dataset=train_ds,
            eval_dataset={'dev': dev_ds,
                          'test': test_ds},
            warm_start_setting=ws,
            exporters=[best_exporter])

        print('dev_acc3\t%.5f\ntest_acc3\t%.5f' %
              (best_exporter._best['dev']['acc'],
               best_exporter._best['test']['acc']))

    else:
        feature_column = propeller.data.FeatureColumns([
            propeller.data.TextColumn(
                'title',
                unk_id=unk_id,
                vocab_dict=tokenizer.vocab,
                tokenizer=tokenizer.tokenize),
            propeller.data.TextColumn(
                'comment',
                unk_id=unk_id,
                vocab_dict=tokenizer.vocab,
                tokenizer=tokenizer.tokenize),
        ])

        def map_fn(seg_a, seg_b):
            seg_a, seg_b = tokenizer.truncate(
                seg_a, seg_b, seqlen=args.max_seqlen)
            sentence, segments = tokenizer.build_for_ernie(seg_a, seg_b)
            return sentence, segments


        predict_ds = feature_column.build_dataset_from_stdin('predict') \
                               .map(map_fn) \
                               .padded_batch(hparams.batch_size) \

        predict_ds.data_shapes = shapes[:-1]
        predict_ds.data_types = types[:-1]

        est = propeller.Learner(model_fn, run_config, hparams)
        for res, in est.predict(predict_ds, ckpt=-1):
            print('%d\t%.5f\t%.5f\t%.5f' %
                  (np.argmax(res), res[0], res[1], res[2]))
