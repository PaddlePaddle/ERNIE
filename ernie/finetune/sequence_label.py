#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import


import os
import time
import argparse
import numpy as np
import multiprocessing

import paddle
import logging
import paddle.fluid as fluid

from six.moves import xrange

from model.ernie import ErnieModel

log = logging.getLogger(__name__)

def create_model(args, pyreader_name, ernie_config, is_prediction=False):
    src_ids = fluid.layers.data(name='1', shape=[-1, args.max_seq_len, 1], dtype='int64')
    sent_ids = fluid.layers.data(name='2', shape=[-1, args.max_seq_len, 1], dtype='int64')
    pos_ids = fluid.layers.data(name='3', shape=[-1, args.max_seq_len, 1], dtype='int64')
    task_ids = fluid.layers.data(name='4', shape=[-1, args.max_seq_len, 1], dtype='int64')
    input_mask = fluid.layers.data(name='5', shape=[-1, args.max_seq_len, 1], dtype='float32')
    labels = fluid.layers.data(name='7', shape=[-1, args.max_seq_len, 1], dtype='int64')
    seq_lens = fluid.layers.data(name='8', shape=[-1], dtype='int64')

    pyreader = fluid.io.DataLoader.from_generator(feed_list=[src_ids, sent_ids, pos_ids, task_ids, input_mask, labels, seq_lens], 
            capacity=70,
            iterable=False)

    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        task_ids=task_ids,
        input_mask=input_mask,
        config=ernie_config,
        use_fp16=args.use_fp16)

    enc_out = ernie.get_sequence_output()
    enc_out = fluid.layers.dropout(
        x=enc_out, dropout_prob=0.1, dropout_implementation="upscale_in_train")
    logits = fluid.layers.fc(
        input=enc_out,
        size=args.num_labels,
        num_flatten_dims=2,
        param_attr=fluid.ParamAttr(
            name="cls_seq_label_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_seq_label_out_b",
            initializer=fluid.initializer.Constant(0.)))
    infers = fluid.layers.argmax(logits, axis=2)

    ret_infers = fluid.layers.reshape(x=infers, shape=[-1, 1])
    lod_labels = fluid.layers.sequence_unpad(labels, seq_lens)
    lod_infers = fluid.layers.sequence_unpad(infers, seq_lens)

    (_, _, _, num_infer, num_label, num_correct) = fluid.layers.chunk_eval(
         input=lod_infers,
         label=lod_labels,
         chunk_scheme=args.chunk_scheme,
         num_chunk_types=((args.num_labels-1)//(len(args.chunk_scheme)-1)))

    labels = fluid.layers.flatten(labels, axis=2)
    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=fluid.layers.flatten(
            logits, axis=2),
        label=labels,
        return_softmax=True)
    input_mask = fluid.layers.flatten(input_mask, axis=2)
    ce_loss = ce_loss * input_mask
    loss = fluid.layers.mean(x=ce_loss)

    graph_vars = {
        "inputs": src_ids,
        "loss": loss,
        "probs": probs,
        "seqlen": seq_lens,
        "num_infer": num_infer,
        "num_label": num_label,
        "num_correct": num_correct,
    }

    for k, v in graph_vars.items():
        v.persistable = True

    return pyreader, graph_vars


def calculate_f1(num_label, num_infer, num_correct):
    if num_infer == 0:
        precision = 0.0
    else:
        precision = num_correct * 1.0 / num_infer

    if num_label == 0:
        recall = 0.0
    else:
        recall = num_correct * 1.0 / num_label

    if num_correct == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate(exe,
             program,
             pyreader,
             graph_vars,
             tag_num,
             dev_count=1):
    fetch_list = [
        graph_vars["num_infer"].name, graph_vars["num_label"].name,
        graph_vars["num_correct"].name
    ]

    total_label, total_infer, total_correct = 0.0, 0.0, 0.0
    time_begin = time.time()
    pyreader.start()
    while True:
        try:
            np_num_infer, np_num_label, np_num_correct = exe.run(program=program,
                                                    fetch_list=fetch_list)
            total_infer += np.sum(np_num_infer)
            total_label += np.sum(np_num_label)
            total_correct += np.sum(np_num_correct)

        except fluid.core.EOFException:
            pyreader.reset()
            break

    precision, recall, f1 = calculate_f1(total_label, total_infer,
                                         total_correct)
    time_end = time.time()
    return  \
        "[evaluation] f1: %f, precision: %f, recall: %f, elapsed time: %f s" \
        % (f1, precision, recall, time_end - time_begin)


def chunk_predict(np_inputs, np_probs, np_lens, dev_count=1):
    inputs = np_inputs.reshape([-1]).astype(np.int32)
    probs = np_probs.reshape([-1, np_probs.shape[-1]])

    all_lens = np_lens.reshape([dev_count, -1]).astype(np.int32).tolist()

    base_index = 0
    out = []
    for dev_index in xrange(dev_count):
        lens = all_lens[dev_index]
        max_len = 0
        for l in lens:
            max_len = max(max_len, l)

        for i in xrange(len(lens)):
            seq_st = base_index + i * max_len + 1
            seq_en = seq_st + (lens[i] - 2)
            prob = probs[seq_st:seq_en, :]
            infers = np.argmax(prob, -1)
            out.append((
                    inputs[seq_st:seq_en].tolist(), 
                    infers.tolist(),
                    prob.tolist()))
        base_index += max_len * len(lens)
    return out


def predict(exe,
            test_program,
            test_pyreader,
            graph_vars,
            dev_count=1):
    fetch_list = [
        graph_vars["inputs"].name,
        graph_vars["probs"].name,
        graph_vars["seqlen"].name,
    ]

    test_pyreader.start()
    res = []
    while True:
        try:
            inputs, probs, np_lens = exe.run(program=test_program,
                                        fetch_list=fetch_list)
            r = chunk_predict(inputs, probs, np_lens, dev_count)
            res += r
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    return res

