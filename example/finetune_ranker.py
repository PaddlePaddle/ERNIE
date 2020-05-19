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
import sys
import io

from random import random
from functools import reduce, partial, wraps

import numpy as np
import multiprocessing
import re

import paddle
import paddle.fluid as F
import paddle.fluid.layers as L


from model.ernie import ErnieModel
from optimization import optimization
import utils.data

from propeller import log
import propeller.paddle as propeller
log.setLevel(logging.DEBUG)

class RankingErnieModel(propeller.train.Model):
    """propeller Model wraper for paddle-ERNIE """
    def __init__(self, hparam, mode, run_config):
        self.hparam = hparam
        self.mode = mode
        self.run_config = run_config

    def forward(self, features):
        src_ids, sent_ids, qid = features

        zero = L.fill_constant([1], dtype='int64', value=0)
        input_mask = L.cast(L.logical_not(L.equal(src_ids, zero)), 'float32') # assume pad id == 0
        #input_mask = L.unsqueeze(input_mask, axes=[2])
        d_shape = L.shape(src_ids)
        seqlen = d_shape[1]
        batch_size = d_shape[0]
        pos_ids = L.unsqueeze(L.range(0, seqlen, 1, dtype='int32'), axes=[0])
        pos_ids = L.expand(pos_ids, [batch_size, 1])
        pos_ids = L.unsqueeze(pos_ids, axes=[2])
        pos_ids = L.cast(pos_ids, 'int64')
        pos_ids.stop_gradient = True
        input_mask.stop_gradient = True
        task_ids = L.zeros_like(src_ids) + self.hparam.task_id #this shit wont use at the moment
        task_ids.stop_gradient = True

        ernie = ErnieModel(
            src_ids=src_ids,
            position_ids=pos_ids,
            sentence_ids=sent_ids,
            task_ids=task_ids,
            input_mask=input_mask,
            config=self.hparam,
            use_fp16=self.hparam['use_fp16']
        )

        cls_feats = ernie.get_pooled_output()

        cls_feats = L.dropout(
            x=cls_feats,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train"
        )

        logits = L.fc(
            input=cls_feats,
            size=self.hparam['num_label'],
            param_attr=F.ParamAttr(
                name="cls_out_w",
                initializer=F.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=F.ParamAttr(
                name="cls_out_b", initializer=F.initializer.Constant(0.))
        )

        propeller.summary.histogram('pred', logits)

        if self.mode is propeller.RunMode.PREDICT:
            probs = L.softmax(logits)
            return qid, probs
        else:
            return qid, logits

    def loss(self, predictions, labels):
        qid, predictions = predictions
        ce_loss, probs = L.softmax_with_cross_entropy(
            logits=predictions, label=labels, return_softmax=True)
        #L.Print(ce_loss, message='per_example_loss')
        loss = L.mean(x=ce_loss)
        return loss

    def metrics(self, predictions, label):
        qid, logits = predictions

        positive_class_logits = L.slice(logits, axes=[1], starts=[1], ends=[2])
        mrr = propeller.metrics.Mrr(qid, label, positive_class_logits)

        predictions = L.argmax(logits, axis=1)
        predictions = L.unsqueeze(predictions, axes=[1])
        f1 = propeller.metrics.F1(label, predictions)
        acc = propeller.metrics.Acc(label, predictions)
        #auc = propeller.metrics.Auc(label, predictions)

        return {'acc': acc, 'f1': f1, 'mrr': mrr}

    def backward(self, loss):
        scheduled_lr, _ = optimization(
            loss=loss,
            warmup_steps=int(self.run_config.max_steps * self.hparam['warmup_proportion']),
            num_train_steps=self.run_config.max_steps,
            learning_rate=self.hparam['learning_rate'],
            train_program=F.default_main_program(), 
            startup_prog=F.default_startup_program(),
            weight_decay=self.hparam['weight_decay'],
            scheduler="linear_warmup_decay",)
        propeller.summary.scalar('lr', scheduled_lr)



if __name__ == '__main__':
    parser = propeller.ArgumentParser('ranker model with ERNIE')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--predict_model', type=str, default=None)
    parser.add_argument('--max_seqlen', type=int, default=128)
    parser.add_argument('--vocab_file', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--warm_start_from', type=str)
    parser.add_argument('--sentence_piece_model', type=str, default=None)
    parser.add_argument('--word_dict', type=str, default=None)
    args = parser.parse_args()
    run_config = propeller.parse_runconfig(args)
    hparams = propeller.parse_hparam(args)


    vocab = {j.strip().split(b'\t')[0].decode('utf8') : i for i, j in enumerate(open(args.vocab_file, 'rb'))}
    sep_id = vocab['[SEP]']
    cls_id = vocab['[CLS]']
    unk_id = vocab['[UNK]']

    if args.sentence_piece_model is not None:
        if args.word_dict is None:
            raise ValueError('--word_dict no specified in subword Model')
        tokenizer = utils.data.WSSPTokenizer(args.sentence_piece_model, args.word_dict, ws=True, lower=True)
    else:
        tokenizer = utils.data.CharTokenizer(vocab.keys())

    def tokenizer_func(inputs):
        '''avoid pickle error'''
        ret = tokenizer(inputs)
        return ret


    shapes = ([-1, args.max_seqlen, 1], [-1, args.max_seqlen, 1], [-1, 1], [-1, 1]) 
    types = ('int64', 'int64', 'int64', 'int64')


    if not args.do_predict:
        feature_column = propeller.data.FeatureColumns([
            propeller.data.LabelColumn('qid'),
            propeller.data.TextColumn('title', vocab_dict=vocab, tokenizer=tokenizer_func, unk_id=unk_id),
            propeller.data.TextColumn('comment', vocab_dict=vocab, tokenizer=tokenizer_func, unk_id=unk_id),
            propeller.data.LabelColumn('label'),
        ])

        def before(qid, seg_a, seg_b, label):
            sentence, segments = utils.data.build_2_pair(seg_a, seg_b, max_seqlen=args.max_seqlen, cls_id=cls_id, sep_id=sep_id)
            return sentence, segments, qid, label

        def after(sentence, segments, qid, label):
            sentence, segments, qid, label = utils.data.expand_dims(sentence, segments, qid, label)
            return sentence, segments, qid, label



        train_ds = feature_column.build_dataset('train', data_dir=os.path.join(args.data_dir, 'train'), shuffle=True, repeat=True, use_gz=False) \
                                       .map(before) \
                                       .padded_batch(hparams.batch_size, (0, 0, 0, 0)) \
                                       .map(after) 

        dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(args.data_dir, 'dev'), shuffle=False, repeat=False, use_gz=False) \
                                       .map(before) \
                                       .padded_batch(hparams.batch_size, (0, 0, 0, 0)) \
                                       .map(after)

        test_ds = feature_column.build_dataset('test', data_dir=os.path.join(args.data_dir, 'test'), shuffle=False, repeat=False, use_gz=False) \
                                       .map(before) \
                                       .padded_batch(hparams.batch_size, (0, 0, 0, 0)) \
                                       .map(after) 

        train_ds.data_shapes = shapes
        train_ds.data_types = types
        dev_ds.data_shapes = shapes
        dev_ds.data_types = types
        test_ds.data_shapes = shapes
        test_ds.data_types = types

        varname_to_warmstart = re.compile(r'^encoder.*[wb]_0$|^.*embedding$|^.*bias$|^.*scale$|^pooled_fc.[wb]_0$')
        warm_start_dir = args.warm_start_from
        ws = propeller.WarmStartSetting(
                predicate_fn=lambda v: varname_to_warmstart.match(v.name) and os.path.exists(os.path.join(warm_start_dir, v.name)),
                from_dir=warm_start_dir
            )

        best_exporter = propeller.train.exporter.BestInferenceModelExporter(os.path.join(run_config.model_dir, 'best'), cmp_fn=lambda old, new: new['dev']['f1'] > old['dev']['f1'])
        propeller.train_and_eval(
                model_class_or_model_fn=RankingErnieModel, 
                params=hparams, 
                run_config=run_config, 
                train_dataset=train_ds, 
                eval_dataset={'dev': dev_ds, 'test': test_ds}, 
                warm_start_setting=ws, 
                exporters=[best_exporter])

        print('dev_mrr\t%.5f\ntest_mrr\t%.5f\ndev_f1\t%.5f\ntest_f1\t%.5f' % (
            best_exporter._best['dev']['mrr'], best_exporter._best['test']['mrr'],
            best_exporter._best['dev']['f1'], best_exporter._best['test']['f1'],
        ))
    else:
        feature_column = propeller.data.FeatureColumns([
            propeller.data.LabelColumn('qid'),
            propeller.data.TextColumn('title', unk_id=unk_id, vocab_dict=vocab, tokenizer=tokenizer_func),
            propeller.data.TextColumn('comment', unk_id=unk_id, vocab_dict=vocab, tokenizer=tokenizer_func),
        ])

        def before(qid, seg_a, seg_b):
            sentence, segments = utils.data.build_2_pair(seg_a, seg_b, max_seqlen=args.max_seqlen, cls_id=cls_id, sep_id=sep_id)
            return sentence, segments, qid

        def after(sentence, segments, qid):
            sentence, segments, qid = utils.data.expand_dims(sentence, segments, qid)
            return sentence, segments, qid

        predict_ds = feature_column.build_dataset_from_stdin('predict') \
                               .map(before) \
                               .padded_batch(hparams.batch_size, (0, 0, 0)) \
                               .map(after) 

        predict_ds.data_shapes = shapes[: -1]
        predict_ds.data_types = types[: -1]

        est = propeller.Learner(RankingErnieModel, run_config, hparams)
        for qid, res in est.predict(predict_ds, ckpt=-1):
            print('%d\t%d\t%.5f\t%.5f' % (qid[0], np.argmax(res), res[0], res[1]))

        #for i in predict_ds:
        #    sen = i[0]
        #    for ss in np.squeeze(sen):
        #        print(' '.join(map(str, ss)))

