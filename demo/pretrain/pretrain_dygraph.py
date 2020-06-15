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
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import sys
import io
import os
import time
import numpy as np
import re
import logging
import six
from glob import glob
from functools import reduce, partial
import itertools

import paddle
import paddle.fluid as F
import paddle.fluid.dygraph as D
import paddle.fluid.layers as L
import sentencepiece as spm
import json

from tqdm import tqdm

import random as r

from ernie.modeling_ernie import ErnieModelForPretraining
from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.optimization import AdamW, LinearDecay

import propeller.paddle as propeller
from propeller.paddle.data import Dataset

from propeller import log

log.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

if six.PY3:
    from itertools import accumulate
else:
    import operator
    def accumulate(iterable, func=operator.add, initial=None):
        'Return running totals'
        # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
        # accumulate([1,2,3,4,5], initial=100) --> 100 101 103 106 110 115
        # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
        it = iter(iterable)
        total = initial
        if initial is None:
            try:
                total = next(it)
            except StopIteration:
                return
        yield total
        for element in it:
            total = func(total, element)
            yield total


def truncate_sentence(seq, from_length, to_length):
    random_begin = np.random.randint(0, np.maximum(0, from_length - to_length) + 1)
    return seq[random_begin: random_begin + to_length]


def build_pair(seg_a, seg_b, max_seqlen, vocab):
    #log.debug('pair %s \n %s' % (seg_a, seg_b))
    cls_id = vocab['[CLS]']
    sep_id = vocab['[SEP]']
    a_len = len(seg_a)
    b_len = len(seg_b)
    ml = max_seqlen - 3
    half_ml = ml // 2
    if a_len > b_len:
        a_len_truncated, b_len_truncated = np.maximum(half_ml, ml - b_len), np.minimum(half_ml, b_len)
    else:
        a_len_truncated, b_len_truncated = np.minimum(half_ml, a_len), np.maximum(half_ml, ml - a_len)

    seg_a = truncate_sentence(seg_a, a_len, a_len_truncated)
    seg_b = truncate_sentence(seg_b, b_len, b_len_truncated)

    seg_a_txt, seg_a_info = seg_a[:, 0], seg_a[:, 1]
    seg_b_txt, seg_b_info = seg_b[:, 0], seg_b[:, 1]

    token_type_a = np.ones_like(seg_a_txt, dtype=np.int64) * 0
    token_type_b = np.ones_like(seg_b_txt, dtype=np.int64) * 1
    sen_emb = np.concatenate([[cls_id], seg_a_txt, [sep_id], seg_b_txt, [sep_id]], 0)
    info_emb = np.concatenate([[-1], seg_a_info, [-1], seg_b_info, [-1]], 0)
    token_type_emb = np.concatenate([[0], token_type_a, [0], token_type_b, [1]], 0)

    return sen_emb, info_emb, token_type_emb


def apply_mask(sentence, seg_info, mask_rate, vocab_size, vocab):
    pad_id = vocab['[PAD]']
    mask_id = vocab['[MASK]']
    shape = sentence.shape
    batch_size, seqlen = shape

    invalid_pos = np.where(seg_info == -1)
    seg_info += 1 #no more =1
    seg_info_flatten = seg_info.reshape([-1])
    seg_info_incr = seg_info_flatten - np.roll(seg_info_flatten, shift=1)
    seg_info = np.add.accumulate(np.array([0 if s == 0 else 1 for s in seg_info_incr])).reshape(shape)
    seg_info[invalid_pos] = -1

    u_seginfo = np.array([i for i in np.unique(seg_info) if i != -1])
    np.random.shuffle(u_seginfo)
    sample_num = max(1, int(len(u_seginfo) * mask_rate))
    u_seginfo = u_seginfo[: sample_num]
    mask = reduce(np.logical_or, [seg_info == i for i in u_seginfo])

    mask[:, 0] = False # ignore CLS head

    rand = np.random.rand(*shape)
    choose_original = rand < 0.1                   # 
    choose_random_id = (0.1 < rand) & (rand < 0.2) # 
    choose_mask_id = 0.2 < rand                    # 
    random_id = np.random.randint(1, vocab_size, size=shape)

    replace_id = mask_id * choose_mask_id + \
                 random_id * choose_random_id + \
                 sentence * choose_original

    mask_pos = np.where(mask)
    #mask_pos_flatten = list(map(lambda idx: idx[0] * seqlen + idx[1], zip(*mask_pos))) #transpose
    mask_label = sentence[mask_pos]
    sentence[mask_pos] = replace_id[mask_pos] #overwrite
    #log.debug(mask_pos_flatten)
    return sentence, np.stack(mask_pos, -1), mask_label


def make_pretrain_dataset(name, dir, vocab, args):
    gz_files = glob(dir)
    if not gz_files:
        raise ValueError('train data not found in %s' % gz_files)

    log.info('read from %s' % '\n'.join(gz_files))
    max_input_seqlen = args.max_seqlen 
    max_pretrain_seqlen = lambda: max_input_seqlen if r.random() > 0.15 else r.randint(1, max_input_seqlen) # short sentence rate

    def _parse_gz(record_str): # function that takes python_str as input
        ex = propeller.data.example_pb2.SequenceExample()
        ex.ParseFromString(record_str)
        doc = [np.array(f.int64_list.value, dtype=np.int64) for f in ex.feature_lists.feature_list['txt'].feature]
        doc_seg = [np.array(f.int64_list.value, dtype=np.int64) for f in ex.feature_lists.feature_list['segs'].feature]
        return doc, doc_seg

    def bb_to_segments(filename):
        ds = Dataset.from_record_file(filename).map(_parse_gz)
        iterable = iter(ds)
        def gen():
            buf, size = [], 0
            iterator = iter(ds)
            while 1:
                doc, doc_seg = next(iterator)
                for line, line_seg in zip(doc, doc_seg):
                    #line = np.array(sp_model.SampleEncodeAsIds(line, -1, 0.1), dtype=np.int64) # 0.1 means large variance on sentence piece result
                    if len(line) == 0:
                        continue
                    line = np.array(line) # 0.1 means large variance on sentence piece result
                    line_seg = np.array(line_seg)
                    size += len(line)
                    buf.append(np.stack([line, line_seg]).transpose())
                    if size > max_input_seqlen:
                        yield buf,
                        buf, size = [], 0
                if len(buf) != 0:
                    yield buf, 
                    buf, size = [], 0
        return Dataset.from_generator_func(gen)

    def sample_negative(dataset):
        def gen():
            iterator = iter(dataset)
            while True:
                chunk_a, = next(iterator)
                #chunk_b, = next(iterator)

                seqlen = max_pretrain_seqlen()
                seqlen_a = r.randint(1, seqlen)
                seqlen_b = seqlen - seqlen_a
                len_a = list(accumulate([len(c) for c in chunk_a]))
                buf_a = [c for c, l in zip(chunk_a, len_a) if l < seqlen_a] #always take the first one
                buf_b = [c for c, l in zip(chunk_a, len_a) if seqlen_a <= l < seqlen]

                if r.random() < 0.5: #pos or neg
                    label = np.int64(1)
                else:
                    label = np.int64(0)
                    buf_a, buf_b = buf_b, buf_a

                if not (len(buf_a) and len(buf_b)):
                    continue
                a = np.concatenate(buf_a)
                b = np.concatenate(buf_b)
                #log.debug(a)
                #log.debug(b)
                sample, seg_info, token_type = build_pair(a, b, args.max_seqlen, vocab) #negative sample might exceed max seqlen
                yield sample, seg_info, token_type, label

        ds = propeller.data.Dataset.from_generator_func(gen)
        return ds

    def after(sentence, seg_info, segments, label):
        batch_size, seqlen = sentence.shape
        sentence, mask_pos, mlm_label = apply_mask(sentence, seg_info, args.mask_rate, len(vocab), vocab)

        ra = r.random()
        if ra < args.check:
            print('***')
            print('\n'.join([str(j) + '\t' + '|'.join(map(str, i)) for i, j in zip(sentence.tolist(), label)]))
            print('***')
            print('\n'.join(['|'.join(map(str, i)) for i in seg_info.tolist()]))
            print('***')
            print('|'.join(map(str, mlm_label.tolist())))
            print('***')

        return sentence, segments, mlm_label, mask_pos, label

    # pretrain pipeline
    dataset = Dataset.from_list(gz_files)
    if propeller.train.distribution.status.mode == propeller.train.distribution.DistributionMode.NCCL:
        log.info('Apply sharding in distribution env')
        if len(gz_files) < propeller.train.distribution.status.num_replica:
            raise ValueError('not enough train file to shard: # of train files: %d, # of workers %d' % (len(gz_files), propeller.train.distribution.status.num_replica))
        dataset = dataset.shard(propeller.train.distribution.status.num_replica, propeller.train.distribution.status.replica_id)
    dataset = dataset.repeat().shuffle(buffer_size=len(gz_files))

    dataset = dataset.interleave(map_fn=bb_to_segments, cycle_length=len(gz_files), block_length=1)
    dataset = dataset.shuffle(buffer_size=1000) #must shuffle to ensure negative sample randomness
    dataset = sample_negative(dataset)
    dataset = dataset.padded_batch(args.bsz, (0, 0, 0, 0)).map(after)
    dataset.name = name
    return dataset


if __name__ == '__main__':
    if six.PY3:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    parser = propeller.ArgumentParser('DAN model with Paddle')
    parser.add_argument('--max_seqlen', type=int, default=256, help='max sequence length, documents from pretrain data will expand to this length')
    parser.add_argument('--data_dir', type=str, required=True, help='protobuf pretrain data directory')
    parser.add_argument('--mask_rate', type=float, default=0.15, help='probability of input token tobe masked')
    parser.add_argument('--check', type=float, default=0., help='probability of debug info')
    parser.add_argument('--warmup_steps', type=int, default=10000, help='warmups steps')
    parser.add_argument('--max_steps', type=int, default=1000000, help='max pretrian steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning_rate')
    parser.add_argument('--from_pretrained', type=str, required=True, help='pretraind model dir')
    parser.add_argument('--save_dir', type=str, default=None, help='model output_dir')
    parser.add_argument('--bsz', type=int, default=50)


    args = parser.parse_args()

    tokenizer = ErnieTokenizer.from_pretrained(args.from_pretrained)

    train_ds = make_pretrain_dataset('train', args.data_dir,
            vocab=tokenizer.vocab, args=args)

    seq_shape = [-1, args.max_seqlen]
    ints_shape = [-1,]
    shapes = (seq_shape, seq_shape, ints_shape, [-1, 2], ints_shape) 
    types = ('int64', 'int64', 'int64', 'int64', 'int64')

    train_ds.data_shapes = shapes
    train_ds.data_types = types

    place = F.CUDAPlace(D.parallel.Env().dev_id)
    with D.guard(place):
        model = ErnieModelForPretraining.from_pretrained(args.from_pretrained)
        opt = AdamW(learning_rate=LinearDecay(args.lr, args.warmup_steps, args.max_steps), parameter_list=model.parameters(), weight_decay=0.01)

        ctx = D.parallel.prepare_context()
        model = D.parallel.DataParallel(model, ctx)

        for step, samples in enumerate(tqdm(train_ds.start(place))):
            (src_ids, sent_ids, mlm_label, mask_pos, nsp_label) = samples
            loss, mlmloss, nsploss = model(src_ids, sent_ids, labels=mlm_label, mlm_pos=mask_pos, nsp_labels=nsp_label)
            scaled_loss = model.scale_loss(loss)
            scaled_loss.backward()
            model.apply_collective_grads()
            opt.minimize(scaled_loss)
            model.clear_gradients()
            if step % 10 == 0:
                log.debug('train loss %.5f scaled loss %.5f' % (loss.numpy(), scaled_loss.numpy()))
            if step % 10000 == 0 and D.parallel.Env().dev_id == 0 and args.save_dir is not None:
                F.save_dygraph(model.state_dict(), args.save_dir)



