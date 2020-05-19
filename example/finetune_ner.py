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

import sys
import os
import re
import time
from random import random
from functools import reduce, partial

import numpy as np
import multiprocessing
import logging
import six
import re

import paddle
import paddle.fluid as F
import paddle.fluid.layers as L


from model.ernie import ErnieModel
from optimization import optimization
import utils.data

from propeller import log
log.setLevel(logging.DEBUG)
import propeller.paddle as propeller

class SequenceLabelErnieModel(propeller.train.Model):
    """propeller Model wraper for paddle-ERNIE """
    def __init__(self, hparam, mode, run_config):
        self.hparam = hparam
        self.mode = mode
        self.run_config = run_config
        self.num_label = len(hparam['label_list'])

    def forward(self, features):
        src_ids, sent_ids, input_seqlen = features
        zero = L.fill_constant([1], dtype='int64', value=0)
        input_mask = L.cast(L.equal(src_ids, zero), 'float32') # assume pad id == 0
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

        model = ErnieModel(
            src_ids=src_ids,
            position_ids=pos_ids,
            sentence_ids=sent_ids,
            task_ids=task_ids,
            input_mask=input_mask,
            config=self.hparam,
            use_fp16=self.hparam['use_fp16']
        )

        enc_out = model.get_sequence_output()
        logits = L.fc(
            input=enc_out,
            size=self.num_label,
            num_flatten_dims=2,
            param_attr= F.ParamAttr(
                name="cls_seq_label_out_w",
                initializer= F.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=F.ParamAttr(
                name="cls_seq_label_out_b",
                initializer=F.initializer.Constant(0.)))

        propeller.summary.histogram('pred', logits)

        return logits, input_seqlen

    def loss(self, predictions, labels):
        logits, input_seqlen = predictions
        logits = L.flatten(logits, axis=2)
        labels = L.flatten(labels, axis=2)
        ce_loss, probs = L.softmax_with_cross_entropy(
            logits=logits, label=labels, return_softmax=True)
        loss = L.mean(x=ce_loss)
        return loss

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

    def metrics(self, predictions, label):
        pred, seqlen = predictions
        pred = L.argmax(pred, axis=-1)
        pred = L.unsqueeze(pred, axes=[-1])
        f1 = propeller.metrics.ChunkF1(label, pred, seqlen, self.num_label)
        return {'f1': f1}

def make_sequence_label_dataset(name, input_files, label_list, tokenizer, batch_size, max_seqlen, is_train):
    label_map = {v: i for i, v in enumerate(label_list)}
    no_entity_id = label_map['O']
    delimiter = b''

    def read_bio_data(filename):
        ds = propeller.data.Dataset.from_file(filename)
        iterable = iter(ds)
        def gen():
            buf, size = [], 0
            iterator = iter(ds)
            while 1:
                line = next(iterator)
                cols = line.rstrip(b'\n').split(b'\t')
                tokens = cols[0].split(delimiter)
                labels = cols[1].split(delimiter)
                if len(cols) != 2:
                    continue
                if len(tokens) != len(labels) or len(tokens) == 0:
                    continue
                yield [tokens, labels]

        return propeller.data.Dataset.from_generator_func(gen)

    def reseg_token_label(dataset):
        def gen():
            iterator = iter(dataset)
            while True:
                tokens, labels = next(iterator)
                assert len(tokens) == len(labels)
                ret_tokens = []
                ret_labels = []
                for token, label in zip(tokens, labels):
                    sub_token = tokenizer(token)
                    label = label.decode('utf8')
                    if len(sub_token) == 0:
                        continue
                    ret_tokens.extend(sub_token)
                    ret_labels.append(label)
                    if len(sub_token) < 2:
                        continue
                    sub_label = label
                    if label.startswith("B-"):
                        sub_label = "I-" + label[2:]
                    ret_labels.extend([sub_label] * (len(sub_token) - 1))

                assert len(ret_tokens) == len(ret_labels)
                yield ret_tokens, ret_labels

        ds = propeller.data.Dataset.from_generator_func(gen)
        return ds

    def convert_to_ids(dataset):
        def gen():
            iterator = iter(dataset)
            while True:
                tokens, labels = next(iterator)
                if len(tokens) > max_seqlen - 2:
                    tokens = tokens[: max_seqlen - 2]
                    labels = labels[: max_seqlen - 2]

                tokens = ['[CLS]'] + tokens + ['[SEP]']
                token_ids = [vocab[t] for t in tokens]
                label_ids = [no_entity_id] + [label_map[x] for x in labels] + [no_entity_id]
                token_type_ids = [0] * len(token_ids)
                input_seqlen = len(token_ids)

                token_ids = np.array(token_ids, dtype=np.int64)
                label_ids = np.array(label_ids, dtype=np.int64)
                token_type_ids = np.array(token_type_ids, dtype=np.int64)
                input_seqlen = np.array(input_seqlen, dtype=np.int64)

                yield token_ids, token_type_ids, input_seqlen, label_ids

        ds = propeller.data.Dataset.from_generator_func(gen)
        return ds

    def after(*features):
        return utils.data.expand_dims(*features)

    dataset = propeller.data.Dataset.from_list(input_files)
    if is_train:
        dataset = dataset.repeat().shuffle(buffer_size=len(input_files))
    dataset = dataset.interleave(map_fn=read_bio_data, cycle_length=len(input_files), block_length=1)
    if is_train:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = reseg_token_label(dataset)
    dataset = convert_to_ids(dataset)
    dataset = dataset.padded_batch(batch_size).map(after)
    dataset.name = name
    return dataset


def make_sequence_label_dataset_from_stdin(name, tokenizer, batch_size, max_seqlen):
    delimiter = b''

    def stdin_gen():
        if six.PY3:
            source = sys.stdin.buffer 
        else:
            source = sys.stdin
        while True:
            line = source.readline()
            if len(line) == 0:
                break
            yield line,

    def read_bio_data(ds):
        iterable = iter(ds)
        def gen():
            buf, size = [], 0
            iterator = iter(ds)
            while 1:
                line, = next(iterator)
                cols = line.rstrip(b'\n').split(b'\t')
                tokens = cols[0].split(delimiter)
                if len(cols) != 1:
                    continue
                if len(tokens) == 0:
                    continue
                yield tokens, 
        return propeller.data.Dataset.from_generator_func(gen)

    def reseg_token_label(dataset):
        def gen():
            iterator = iter(dataset)
            while True:
                tokens, = next(iterator)
                ret_tokens = []
                for token in tokens:
                    sub_token = tokenizer(token)
                    if len(sub_token) == 0:
                        continue
                    ret_tokens.extend(sub_token)
                    if len(sub_token) < 2:
                        continue
                yield ret_tokens, 
        ds = propeller.data.Dataset.from_generator_func(gen)
        return ds

    def convert_to_ids(dataset):
        def gen():
            iterator = iter(dataset)
            while True:
                tokens, = next(iterator)
                if len(tokens) > max_seqlen - 2:
                    tokens = tokens[: max_seqlen - 2]

                tokens = ['[CLS]'] + tokens + ['[SEP]']
                token_ids = [vocab[t] for t in tokens]
                token_type_ids = [0] * len(token_ids)
                input_seqlen = len(token_ids)

                token_ids = np.array(token_ids, dtype=np.int64)
                token_type_ids = np.array(token_type_ids, dtype=np.int64)
                input_seqlen = np.array(input_seqlen, dtype=np.int64)
                yield token_ids, token_type_ids, input_seqlen

        ds = propeller.data.Dataset.from_generator_func(gen)
        return ds

    def after(*features):
        return utils.data.expand_dims(*features)

    dataset = propeller.data.Dataset.from_generator_func(stdin_gen)
    dataset = read_bio_data(dataset)
    dataset = reseg_token_label(dataset)
    dataset = convert_to_ids(dataset)
    dataset = dataset.padded_batch(batch_size).map(after)
    dataset.name = name
    return dataset


if __name__ == '__main__':
    parser = propeller.ArgumentParser('NER model with ERNIE')
    parser.add_argument('--max_seqlen', type=int, default=128)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--vocab_file', type=str, required=True)
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--use_sentence_piece_vocab', action='store_true')
    parser.add_argument('--warm_start_from', type=str)
    args = parser.parse_args()
    run_config = propeller.parse_runconfig(args)
    hparams = propeller.parse_hparam(args)


    vocab = {j.strip().split('\t')[0]: i for i, j in enumerate(open(args.vocab_file, 'r', encoding='utf8'))}
    tokenizer = utils.data.CharTokenizer(vocab, sentencepiece_style_vocab=args.use_sentence_piece_vocab)
    sep_id = vocab['[SEP]']
    cls_id = vocab['[CLS]']
    unk_id = vocab['[UNK]']
    pad_id = vocab['[PAD]']

    label_list = ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'O']
    hparams['label_list'] = label_list

    if not args.do_predict:
        train_data_dir = os.path.join(args.data_dir, 'train')
        train_input_files = [os.path.join(train_data_dir, filename) for filename in os.listdir(train_data_dir)]
        dev_data_dir = os.path.join(args.data_dir, 'dev')
        dev_input_files = [os.path.join(dev_data_dir, filename) for filename in os.listdir(dev_data_dir)]
        test_data_dir = os.path.join(args.data_dir, 'test')
        test_input_files = [os.path.join(test_data_dir, filename) for filename in os.listdir(test_data_dir)]

        train_ds = make_sequence_label_dataset(name='train', 
                                               input_files=train_input_files, 
                                               label_list=label_list, 
                                               tokenizer=tokenizer, 
                                               batch_size=hparams.batch_size, 
                                               max_seqlen=args.max_seqlen,
                                               is_train=True)
        dev_ds = make_sequence_label_dataset(name='dev',
                                             input_files=dev_input_files,
                                             label_list=label_list,
                                             tokenizer=tokenizer,
                                             batch_size=hparams.batch_size,
                                             max_seqlen=args.max_seqlen,
                                             is_train=False)
        test_ds = make_sequence_label_dataset(name='test',
                                              input_files=test_input_files,
                                              label_list=label_list,
                                              tokenizer=tokenizer,
                                              batch_size=hparams.batch_size,
                                              max_seqlen=args.max_seqlen,
                                              is_train=False)

        shapes = ([-1, args.max_seqlen, 1], [-1, args.max_seqlen, 1], [-1, 1], [-1, args.max_seqlen, 1]) 
        types = ('int64', 'int64', 'int64', 'int64')

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
        propeller.train.train_and_eval(
                model_class_or_model_fn=SequenceLabelErnieModel,
                params=hparams, 
                run_config=run_config, 
                train_dataset=train_ds, 
                eval_dataset={'dev': dev_ds, 'test': test_ds}, 
                warm_start_setting=ws, 
                exporters=[best_exporter])

        for k in best_exporter._best['dev'].keys():
            if 'loss' in k:
                continue
            dev_v = best_exporter._best['dev'][k]
            test_v = best_exporter._best['test'][k]
            print('dev_%s\t%.5f\ntest_%s\t%.5f' % (k, dev_v, k, test_v))
    else:
        predict_ds = make_sequence_label_dataset_from_stdin(name='pred', 
                                               tokenizer=tokenizer, 
                                               batch_size=hparams.batch_size, 
                                               max_seqlen=args.max_seqlen)

        shapes = ([-1, args.max_seqlen, 1], [-1, args.max_seqlen, 1], [-1, 1]) 
        types = ('int64', 'int64', 'int64')

        predict_ds.data_shapes = shapes
        predict_ds.data_types = types

        rev_label_map = {i: v for i, v in enumerate(label_list)}
        learner = propeller.Learner(SequenceLabelErnieModel, run_config, hparams)
        for pred, _  in learner.predict(predict_ds, ckpt=-1):
            pred_str = ' '.join([rev_label_map[idx] for idx in np.argmax(pred, 1).tolist()])
            print(pred_str)
            

