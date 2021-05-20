#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import time
import numpy as np

from six.moves import xrange
import paddle.fluid as fluid
from model.unimo_finetune import UNIMOModel
from eval import glue_eval
from collections import OrderedDict
from utils.utils import print_eval_log


def create_model(args, pyreader_name, config):
    """create_model"""
    stype = 'int64'
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, args.max_seq_len], [-1, 1],
                [-1, 1]],
        dtypes=[stype, stype, stype, 'float32', stype, stype],
        lod_levels=[0, 0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)

    (src_ids, sent_ids, pos_ids, input_mask, labels,
     qids) = fluid.layers.read_file(pyreader)

    emb_ids = {"word_embedding": src_ids, "sent_embedding": sent_ids, "pos_embedding": pos_ids}
    model = UNIMOModel(
        emb_ids=emb_ids,
        input_mask=input_mask,
        config=config)

    cls_feats = model.get_pooled_text_output()
    cls_feats = fluid.layers.dropout(
        x=cls_feats,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")

    cls_params_name = ["cls_out_%d_w" % args.num_labels, "cls_out_%d_b" % args.num_labels]
    logits = fluid.layers.fc(
        input=cls_feats,
        size=args.num_labels,
        param_attr=fluid.ParamAttr(
            name=cls_params_name[0],
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name=cls_params_name[1], initializer=fluid.initializer.Constant(0.)))

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
    return pyreader, graph_vars


def predict(exe, test_program, test_pyreader, graph_vars, dev_count=1):
    """predict"""
    qids, scores, probs, preds = [], [], [], []
    fetch_list = [graph_vars["probs"].name, graph_vars["qids"].name]
    test_pyreader.start()
    while True:
        try:
            if dev_count == 1:
                np_probs, np_qids = exe.run(program=test_program, fetch_list=fetch_list)
            else:
                np_probs, np_qids = exe.run(fetch_list=fetch_list)
            qids.extend(np_qids.reshape(-1).tolist())
            np_preds = np.argmax(np_probs, axis=1).astype(np.float32)
            preds.extend(np_preds)
            probs.append(np_probs)
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    probs = np.concatenate(probs, axis=0).reshape([len(qids), -1])
    return qids, preds, probs


def evaluate(args, exe, test_program, test_pyreader, graph_vars, eval_phase):
    """evaluate"""
    total_cost, total_num_seqs = 0.0, 0.0
    qids, labels, scores, preds = [], [], [], []
    time_begin = time.time()
    fetch_list = [
        graph_vars["loss"].name,
        graph_vars["probs"].name, graph_vars["labels"].name,
        graph_vars["num_seqs"].name, graph_vars["qids"].name
    ]
    test_pyreader.start()
    while True:
        try:
            np_loss, np_probs, np_labels, np_num_seqs, np_qids = exe.run(
                program=test_program, fetch_list=fetch_list) \
                        if not args.use_multi_gpu_test else exe.run(fetch_list=fetch_list)
            total_cost += np.sum(np_loss * np_num_seqs)
            total_num_seqs += np.sum(np_num_seqs)
            labels.extend(np_labels.reshape((-1)).tolist())
            if np_qids is not None:
                qids.extend(np_qids.reshape(-1).tolist())
            scores.extend(np_probs[:, 1].reshape(-1).tolist())
            np_preds = list(np.argmax(np_probs, axis=1).astype(np.float32))
            preds.extend([float(val) for val in np_preds])
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()
    ret = OrderedDict()
    ret['phase'] = eval_phase
    ret['loss'] = round(total_cost / total_num_seqs, 4)
    ret['data_num'] = total_num_seqs
    ret['used_time'] = round(time_end - time_begin, 4)

    metrics = OrderedDict()
    metrics["acc_and_f1"] = glue_eval.acc_and_f1
    metrics["simple_accuracy"] = glue_eval.simple_accuracy
    metrics["matthews_corrcoef"] = glue_eval.matthews_corrcoef

    if args.eval_mertrics in metrics:
        ret_metric = metrics[args.eval_mertrics](preds, labels)
        ret.update(ret_metric)
        print_eval_log(ret)
    else:
        raise ValueError('unsupported metric {}'.format(args.eval_mertrics))
    return ret
