#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
# File: test_decode.py
# Author: chenxuyi(chenxuyi@baidu.com)
# Date: 2020/05/08 20:55:15
#
########################################################################
"""
    Comment.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sys
import argparse
import logging
import json
import os
import numpy as np

import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D

from tqdm import tqdm

from ernie.modeling_ernie import ErnieModel, ErnieModelForPretraining
from ernie.modeling_ernie import _build_linear, _build_ln, append_name
from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.optimization import AdamW, LinearDecay

from experimental.seq2seq.decode import beam_search_infilling

from propeller import log
import propeller.paddle as propeller

np.set_printoptions(edgeitems=10)

logging.getLogger().handlers[0]=log.handlers[0]
logging.getLogger().setLevel(logging.DEBUG)
log = logging.getLogger() 

class ErnieModelForGeneration(ErnieModel):
    def __init__(self, cfg, name=None):
        cfg['return_additional_info'] = True
        cfg['has_pooler'] = False
        super(ErnieModelForGeneration, self).__init__(cfg, name=name)
        initializer = F.initializer.TruncatedNormal(scale=cfg['initializer_range'])
        d_model = cfg['hidden_size']
        d_vocab = cfg['vocab_size']

        self.mlm = _build_linear(d_model, d_model, append_name(name, 'mask_lm_trans_fc'), initializer, act=cfg['hidden_act'])
        self.mlm_ln = _build_ln(d_model, name = append_name(name, 'mask_lm_trans'))
        self.mlm_bias = L.create_parameter(
                dtype='float32',
                shape=[d_vocab], 
                attr=F.ParamAttr(
                    name=append_name(name, 'mask_lm_out_fc.b_0'), 
                    initializer=F.initializer.Constant(value=0.0)
                    ),
                is_bias=True,
            )

    def forward(self, src_ids, *args, **kwargs):
        tgt_labels = kwargs.pop('tgt_labels', None)
        tgt_pos = kwargs.pop('tgt_pos', None)
        encode_only = kwargs.pop('encode_only', False)
        _, encoded, info = ErnieModel.forward(self, src_ids, *args, **kwargs)
        #log.debug('hidden_-1 %r'% L.reduce_mean(info['hiddens'][0]).numpy())
        #log.debug('hidden_0 %r'% L.reduce_mean(info['hiddens'][1]).numpy())
        if encode_only:
            return None, None, info
        elif tgt_labels is None:
            encoded = self.mlm(encoded)
            encoded = self.mlm_ln(encoded)
            logits = L.matmul(encoded, self.word_emb.weight, transpose_y=True) + self.mlm_bias
            output_ids = L.argmax(logits, -1)
            return output_ids, logits, info
        else:
            encoded_2d = L.gather_nd(encoded, tgt_pos)
            #log.debug('input shape %s' % repr(src_ids.shape))
            #log.debug(L.gather_nd(src_ids, tgt_pos).numpy())
            encoded_2d = self.mlm(encoded_2d)
            encoded_2d = self.mlm_ln(encoded_2d)
            logits_2d = L.matmul(encoded_2d, self.word_emb.weight, transpose_y=True) + self.mlm_bias
            if len(tgt_labels.shape) == 1:
                tgt_labels = L.reshape(tgt_labels, [-1, 1])
            loss = L.reduce_mean(L.softmax_with_cross_entropy(logits_2d, tgt_labels))
            return loss, logits_2d, info

@np.vectorize
def rev_lookup(i):
    return rev_dict[i]


def evaluate(model, datasets, step, args):
    did = D.parallel.Env().dev_id
    place = F.CUDAPlace(D.parallel.Env().dev_id)
    with open(os.path.join(args.predict_output_dir, 'pred.step%d.%d' % (step, did)), 'w') as outf:
        for step, data in enumerate(datasets.start(place)):
            (example_id, src_ids, src_sids, src_pids,
             _, _, _,
             _, 
             _, _, _, _) = data # never use target when infer
            output_ids = beam_search_infilling(model, src_ids, src_sids,
                    eos_id=tokenizer.sep_id,
                    sos_id=tokenizer.cls_id,
                    attn_id=tokenizer.vocab['[ATTN]'],
                    max_decode_len=args.max_decode_len, 
                    max_encode_len=args.max_encode_len, 
                    beam_width=args.beam_width)
            output_str = rev_lookup(output_ids.numpy())
            for ostr in output_str.tolist():
                if '[SEP]' in ostr:
                    ostr = ostr[: ostr.index('[SEP]')]
                ostr = ' '.join(ostr)
                print('%d\t%s' % (example_id.numpy(), ostr), file=outf)

    model.train()


def seq2seq(model, tokenizer, args):
    attn_id = tokenizer.vocab['[ATTN]']
    def gen_mask(batch_ids, mask_type='bidi', query_len=None, pad_value=0):
        if query_len is None:
            query_len = batch_ids.shape[1]
        if mask_type != 'empty':
            mask = (batch_ids != pad_value).astype(np.float32)
            mask = np.tile(np.expand_dims(mask, 1), [1, query_len, 1])
            if mask_type == 'causal':
                assert query_len == batch_ids.shape[1]
                mask = np.tril(mask)
            elif mask_type == 'causal_without_diag':
                assert query_len == batch_ids.shape[1]
                mask = np.tril(mask, -1)
            elif mask_type == 'diag':
                assert query_len == batch_ids.shape[1]
                mask = np.stack([np.diag(np.diag(m)) for m in mask], 0)
        else:
            mask_type == 'empty'
            mask = np.zeros_like(batch_ids).astype(np.float32)
            mask = np.tile(np.expand_dims(mask, 1), [1, query_len, 1])
        return mask


    def map_fn(example_id, src_ids, tgt_ids):
        src_ids = src_ids[: args.max_encode_len]
        tgt_ids = tgt_ids[: args.max_decode_len]
        src_ids, src_sids = tokenizer.build_for_ernie(src_ids)
        src_pids = np.arange(len(src_ids))

        tgt_ids, tgt_sids = tokenizer.build_for_ernie(tgt_ids)
        tgt_pids = np.arange(len(tgt_ids)) + len(src_ids) # continues position
        tgt_sids = np.ones_like(tgt_sids) * args.tgt_type_id

        attn_ids = np.ones_like(tgt_ids) * attn_id

        return (example_id, src_ids, src_pids, src_sids,
                tgt_ids, tgt_pids, tgt_sids,
                attn_ids)

    def after_padding(example_id, src_ids, src_pids, src_sids,
                      tgt_ids, tgt_pids, tgt_sids,
                      attn_ids):
        '''
        attention mask:
        ***  src,  tgt, attn
        src  00,   01,   11
        tgt  10,   11,   12
        attn 20,   21,   22

        ***   s1, s2 | t1 t2 t3| attn1 attn2 attn3
        s1    1,  1  | 0, 0, 0,| 0,    0,    0,
        s2    1,  1  | 0, 0, 0,| 0,    0,    0,
        -
        t1    1,  1, | 1, 0, 0,| 0,    0,    0,
        t2    1,  1, | 1, 1, 0,| 0,    0,    0,
        t3    1,  1, | 1, 1, 1,| 0,    0,    0,
        -
        attn1 1,  1, | 0, 0, 0,| 1,    0,    0,
        attn2 1,  1, | 1, 0, 0,| 0,    1,    0,
        attn3 1,  1, | 1, 1, 0,| 0,    0,    1,

        for details, see Fig3. https://arxiv.org/abs/2001.11314
        '''

        src_len = src_ids.shape[1]
        tgt_len = tgt_ids.shape[1]
        mask_00 = gen_mask(src_ids, 'bidi', query_len=src_len)
        mask_01 = gen_mask(tgt_ids, 'empty', query_len=src_len)
        mask_02 = gen_mask(attn_ids,'empty', query_len=src_len)

        mask_10 = gen_mask(src_ids, 'bidi', query_len=tgt_len)
        mask_11 = gen_mask(tgt_ids, 'causal', query_len=tgt_len)
        mask_12 = gen_mask(attn_ids, 'empty', query_len=tgt_len)

        mask_20 = gen_mask(src_ids, 'bidi', query_len=tgt_len)
        mask_21 = gen_mask(tgt_ids, 'causal_without_diag', query_len=tgt_len)
        mask_22 = gen_mask(attn_ids, 'diag', query_len=tgt_len)

        '''
        mask = np.concatenate([
            np.concatenate([mask_00, mask_01, mask_02], 2),
            np.concatenate([mask_10, mask_11, mask_12], 2),
            np.concatenate([mask_20, mask_21, mask_22], 2),
        ], 1)

        ids = np.concatenate([src_ids, tgt_ids, attn_ids], 1)
        pids = np.concatenate([src_pids, tgt_pids, tgt_pids], 1)
        sids = np.concatenate([src_sids, tgt_sids, tgt_sids], 1)

        '''

        mask_src_2_src = mask_00 
        mask_tgt_2_srctgt = np.concatenate([mask_10, mask_11], 2)
        mask_attn_2_srctgtattn = np.concatenate([mask_20, mask_21, mask_22], 2)


        tgt_labels = tgt_ids[np.where(tgt_ids != 0)]
        #log.debug(src_ids)
        #log.debug(src_pids)
        #log.debug(src_sids)
        #log.debug(tgt_ids)
        #log.debug(tgt_pids)
        #log.debug(tgt_sids)
        #log.debug(attn_ids)

        return (example_id, src_ids, src_sids, src_pids,
                tgt_ids, tgt_sids, tgt_pids,
                attn_ids, 
                mask_src_2_src, mask_tgt_2_srctgt, mask_attn_2_srctgtattn, tgt_labels)

    bytes_vocab = {k.encode('utf8'): v for k, v in tokenizer.vocab.items()}
    feature_column = propeller.data.FeatureColumns([
        propeller.data.LabelColumn('id'),
        propeller.data.TextColumn('src', unk_id=tokenizer.unk_id, vocab_dict=bytes_vocab),
        propeller.data.TextColumn('tgt', unk_id=tokenizer.unk_id, vocab_dict=bytes_vocab),
    ])
    
    train_ds = feature_column.build_dataset('train', data_dir=os.path.join(args.data_dir, 'train'), shuffle=False, repeat=True, use_gz=False) \
                                   .map(map_fn) 

    dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(args.data_dir, 'dev'), shuffle=False, repeat=False, use_gz=False) \
                                   .map(map_fn) \
                                   .padded_batch(args.eval_bsz) \
                                   .map(after_padding)

    log.debug('shard %d of %d'%(D.parallel.Env().dev_id, D.parallel.Env().nranks))
    train_ds = train_ds.shard(D.parallel.Env().nranks, D.parallel.Env().dev_id).shuffle(10000).padded_batch(args.bsz).map(after_padding)
    dev_ds = dev_ds.shard(D.parallel.Env().nranks, D.parallel.Env().dev_id)

    shapes = [[None, None]] * 7 + [[None, None, None]] * 3 +[[None]]
    types = ['int64'] * 11

    train_ds.data_shapes = shapes
    train_ds.data_types = types
    dev_ds.data_shapes = shapes
    dev_ds.data_types = types

    ctx = D.parallel.prepare_context()
    model = D.parallel.DataParallel(model, ctx)
    opt = AdamW(learning_rate=LinearDecay(args.lr, int(args.warmup_proportion * args.max_steps), args.max_steps), parameter_list=model.parameters(), weight_decay=args.wd)
    g_clip = F.dygraph_grad_clip.GradClipByGlobalNorm(1.0)
    attn_id = tokenizer.vocab['[ATTN]']
    for step, data in enumerate(train_ds.start(place)):
        (example_id, src_ids, src_sids, src_pids,
         tgt_ids, tgt_sids, tgt_pids,
         attn_ids, 
         mask_src_2_src, mask_tgt_2_srctgt, mask_attn_2_srctgtattn, tgt_labels) = data

        _, __, info = model(src_ids, sent_ids=src_sids, pos_ids=src_pids, attn_bias=mask_src_2_src, encode_only=True)
        cached_k, cached_v = info['caches']
        _, __, info = model(tgt_ids, sent_ids=tgt_sids, pos_ids=tgt_pids, attn_bias=mask_tgt_2_srctgt, past_cache=(cached_k, cached_v), encode_only=True)
        cached_k2, cached_v2 = info['caches']
        past_cache_k = [L.concat([k, k2], 1) for k, k2 in zip(cached_k, cached_k2)]
        past_cache_v = [L.concat([v, v2], 1) for v, v2 in zip(cached_v, cached_v2)]
        loss, _, __ = model(attn_ids, sent_ids=tgt_sids, pos_ids=tgt_pids, attn_bias=mask_attn_2_srctgtattn, 
                past_cache=(past_cache_k, past_cache_v), 
                tgt_labels=tgt_labels, 
                tgt_pos=L.where(attn_ids==attn_id))

        scaled_loss = model.scale_loss(loss)
        scaled_loss.backward()
        model.apply_collective_grads()
        opt.minimize(scaled_loss, grad_clip=g_clip)
        model.clear_gradients()
        if step % 10 == 0:
            loss = loss.numpy()
            ppl = np.exp(loss)
            log.debug('[step %d]train loss %.5f, ppl %.5f, lr %.3e' % (step, loss, ppl, opt.current_step_lr()))
        if args.save_dir is not None and step % 1000 == 0 and D.parallel.Env().dev_id == 0:
            F.save_dygraph(model.state_dict(), args.save_dir)

        if args.predict_output_dir is not None and (step + 1) % args.eval_steps == 0:
            assert os.path.exists(args.predict_output_dir), 'predict_output_dir not found: %s' % args.predict_output_dir
            log.debug('doing predict on gpu %d...' % D.parallel.Env().dev_id)
            evaluate(model, dev_ds, step, args)
        if step > args.max_steps:
            break

    if args.save_dir is not None:
        F.save_dygraph(model.state_dict(), args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('seq2seq model with ERNIE')
    parser.add_argument('--from_pretrained', type=str, required=True, help='pretrained model directory or tag')
    parser.add_argument('--bsz', type=int, default=8, help='batchsize')
    parser.add_argument('--eval_bsz', type=int, default=20, help='batchsize')
    parser.add_argument('--epoch', type=int, default=30, help='epoch')
    parser.add_argument('--data_dir', type=str, required=True, help='data directory includes train / develop data')
    parser.add_argument('--max_steps', type=int, required=True, help='max_train_steps, set this to EPOCH * NUM_SAMPLES / BATCH_SIZE')
    parser.add_argument('--eval_steps', type=int, default='10000000', help='evaluation frequency')
    parser.add_argument('--max_encode_len', type=int, default=640)
    parser.add_argument('--max_decode_len', type=int, default=120)
    parser.add_argument('--tgt_type_id', type=int, default=3)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--beam_width', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--predict_output_dir', type=str, default=None, help='predict file output directory')
    parser.add_argument('--inference_model_dir', type=str, default=None, help='inference model output directory')
    parser.add_argument('--save_dir', type=str, default=None, help='model output directory')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay, aka L2 regularizer')

    args = parser.parse_args()

    sd = F.io.load_program_state(os.path.join(args.from_pretrained, 'params'))

    place = F.CUDAPlace(D.parallel.Env().dev_id)
    D.guard(place).__enter__()

    cfg = json.loads(open(os.path.join(args.from_pretrained, 'ernie_config.json')).read())
    ernie = ErnieModelForGeneration(cfg, name='')
    sd['sent_embedding'] = ernie.state_dict()['sent_emb.weight']
    ernie.set_dict(sd, use_structured_name=False)
    tokenizer = ErnieTokenizer.from_pretrained(args.from_pretrained, mask_token=None)
    rev_dict = {v: k for k, v in tokenizer.vocab.items()}
    rev_dict[tokenizer.pad_id] = '' # replace [PAD]
    rev_dict[tokenizer.unk_id] = '' # replace [PAD]

    seq2seq(ernie, tokenizer, args)
