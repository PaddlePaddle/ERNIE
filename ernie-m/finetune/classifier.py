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
"""Model for classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import time
import json
import numpy as np

import logging
import paddle.fluid as fluid

from model.ernie import ErnieModel

log = logging.getLogger(__name__)

def create_model(args, ernie_config, is_prediction=False):
    src_ids = fluid.layers.data(name="src_ids", shape=[-1, args.max_seq_len, 1], dtype="int64")
    pos_ids = fluid.layers.data(name="pos_ids", shape=[-1, args.max_seq_len, 1], dtype="int64")
    input_mask = fluid.layers.data(name="input_mask", shape=[-1, args.max_seq_len, 1], dtype="float32")
    labels = fluid.layers.data(name="labels", shape=[-1, 1], dtype="int64")
    qids = fluid.layers.data(name="qids", shape=[-1, 1], dtype="int64")

    pyreader = fluid.io.DataLoader.from_generator(
            feed_list=[src_ids, pos_ids, input_mask, labels, qids],
            capacity=70, 
            iterable=False)

    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        input_mask=input_mask,
        config=ernie_config,
        use_fp16=args.use_fp16)

    cls_feats = ernie.get_pooled_output()
    cls_feats = fluid.layers.dropout(
        x=cls_feats,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")
    logits = fluid.layers.fc(
        input=cls_feats,
        size=args.num_labels,
        param_attr=fluid.ParamAttr(
            name="cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_out_b", initializer=fluid.initializer.Constant(0.)))

    if is_prediction:
        probs = fluid.layers.softmax(logits)
        feed_targets_name = [
            src_ids.name, pos_ids.name, input_mask.name
        ]
        return pyreader, probs, feed_targets_name

    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
            logits=logits, label=labels, return_softmax=True)
    loss = fluid.layers.mean(x=ce_loss)

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=labels, total=num_seqs)
    
    graph_vars = {
        "loss": loss,
        "probs": probs,
        "accuracy": accuracy,
        "labels": labels,
        "num_seqs": num_seqs,
        "qids": qids
    }

    for k, v in graph_vars.items():
        v.persistable = True

    return pyreader, graph_vars

def evaluate(exe,
        test_program,
        test_pyreader,
        graph_vars,
        eval_phase,
        lang=None):
    train_fetch_list = [
        graph_vars["loss"].name,
        graph_vars["accuracy"].name,
        graph_vars["num_seqs"].name
    ]

    if eval_phase == "train":
        if "learning_rate" in graph_vars:
            train_fetch_list.append(graph_vars["learning_rate"].name)
        outputs = exe.run(fetch_list=train_fetch_list)
        ret = {"loss": np.mean(outputs[0]), "accuracy": np.mean(outputs[1])}
        if "learning_rate" in graph_vars:
            ret["learning_rate"] = float(outputs[3][0])
        return ret

    fetch_list = [
        graph_vars["loss"].name,
        graph_vars["accuracy"].name,
        graph_vars["probs"].name,
        graph_vars["labels"].name,
        graph_vars["num_seqs"].name,
        graph_vars["qids"].name,
    ]

    test_pyreader.start()
    time_begin = time.time()
    total_cost, total_num_seqs = 0.0, 0.0
    qids, labels, preds = [], [], []
    while True:
        try:
            np_loss, np_acc, np_probs, np_labels, np_num_seqs, np_qids = exe.run(
                    program=test_program, fetch_list=fetch_list)
            total_cost += np.sum(np_loss * np_num_seqs)
            total_num_seqs += np.sum(np_num_seqs)
            if np_qids is None:
                np_qids = np.array([])
            qids.extend(np_qids.reshape(-1).tolist())
            labels.extend(np_labels.reshape(-1).tolist())
            np_preds = np.argmax(np_probs, axis=1).astype(np.float32)
            preds.extend(np_preds.tolist())
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()

    cost = total_cost / total_num_seqs
    elapsed_time = time_end - time_begin

    acc = simple_accuracy(preds, labels)
    evaluate_info = "[%s evaluation] ave loss: %f, %s acc: %f, data_num: %d, elapsed time: %f s" \
            % (eval_phase, cost, lang, acc, total_num_seqs, elapsed_time)
        
    return evaluate_info


def simple_accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean()


def predict(exe,
        test_program,
        test_pyreader,
        graph_vars):

    test_pyreader.start()
    qids, probs, preds = [], [], []

    fetch_list = [
        graph_vars["probs"].name, 
        graph_vars["qids"].name,
    ]

    while True:
        try:
            np_probs, np_qids = exe.run(
                    program=test_program, fetch_list=fetch_list)
            if np_qids is None:
                np_qids = np.array([])
            qids.extend(np_qids.reshape(-1).tolist())
            np_preds = np.argmax(np_probs, axis=1).astype(np.float32)
            preds.extend(np_preds.tolist())
            probs.append(np_probs)

        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    
    probs = np.concatenate(probs, axis=0).reshape([len(preds), -1])

    return qids, preds, probs

