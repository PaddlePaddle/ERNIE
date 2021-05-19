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

import time
import numpy as np
import collections
from collections import namedtuple

import paddle.fluid as fluid

from model.static.ernie import ErnieDocModel
from utils.multi_process_eval import MultiProcessEvalForErnieDoc
from utils.metrics import Acc

def create_model(args, ernie_config, mem_len=128, is_infer=False):
    """create model for classifier"""
    shapes = [[-1, args.max_seq_len, 1], [-1, 2 * args.max_seq_len + mem_len, 1], [-1, args.max_seq_len, 1],
            [-1, args.max_seq_len, 1], [-1, 1], [-1, 1], [-1, 1], []]
    dtypes = ['int64', 'int64', 'int64', 'float32', 'int64', 'int64', 'int64', 'int64']
    names = ["src_ids", "pos_ids", "task_ids", "input_mask", "labels", "qids", "gather_idx", "need_cal_loss"]

    inputs = []
    for shape, dtype, name in zip(shapes, dtypes, names):
        inputs.append(fluid.layers.data(name=name, shape=shape, dtype=dtype, append_batch_size=False))

    src_ids, pos_ids, task_ids, input_mask, labels, qids, \
            gather_idx, need_cal_loss = inputs
    pyreader = fluid.io.DataLoader.from_generator(
            feed_list=inputs,
            capacity=70, iterable=False)

    ernie_doc = ErnieDocModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        task_ids=task_ids,
        input_mask=input_mask,
        config=ernie_config,
        number_instance=args.batch_size,
        rel_pos_params_sharing=args.rel_pos_params_sharing,
        use_vars=args.use_vars)
    
    mems, new_mems = ernie_doc.get_mem_output()

    cls_feats = ernie_doc.get_pooled_output()
    checkpoints = ernie_doc.get_checkpoints()
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

    if is_infer:
        probs = fluid.layers.softmax(logits)
        feed_targets_name = [
            src_ids.name, pos_ids.name, task_ids.name, input_mask.name
        ]
        return pyreader, probs, feed_targets_name
    
    # filter
    qids, logits, labels = list(map(lambda x: fluid.layers.gather(x, gather_idx), [qids, logits, labels]))
    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=logits, label=labels, return_softmax=True)
    loss = fluid.layers.mean(x=ce_loss)

    num_seqs = fluid.layers.create_tensor(dtype='int32')
    accuracy = fluid.layers.accuracy(input=probs, label=labels, total=num_seqs)
    
    loss, num_seqs, accuracy = list(map(lambda x: x * need_cal_loss, [loss, num_seqs, accuracy]))
    graph_vars = collections.OrderedDict()

    fetch_names = ['loss', 'accuracy', 'probs', 'labels', 'num_seqs', 'qids', 'need_cal_loss']
    fetch_vars = [loss, accuracy, probs, labels, num_seqs, qids, need_cal_loss]
    for name, var in zip(fetch_names, fetch_vars):
        graph_vars[name] = var

    for k, v in graph_vars.items():
        v.persistable = True
    mems_vars = {'mems': mems, 'new_mems': new_mems}
    return pyreader, graph_vars, checkpoints, mems_vars


def evaluate(exe,  
             program, 
             pyreader, 
             graph_vars,
             mems_vars,
             tower_mems_np, 
             phase, 
             steps=None,
             trainers_id=None, 
             trainers_num=None,
             scheduled_lr=None,
             use_vars=False):
    """evaluate interface"""
    fetch_names = [k for k, v in graph_vars.items()]
    fetch_list = [v for k, v in graph_vars.items()]

    if phase == "train":
        fetch_names += ['scheduled_lr']
        fetch_list += [scheduled_lr]

    if not use_vars:
        feed_dict = {}
        for m, m_np in zip(mems_vars['mems'], tower_mems_np):
            feed_dict[m.name] = m_np
        
        fetch_list += mems_vars['new_mems']
        fetch_names += [m.name for m in mems_vars['new_mems']]
        

    if phase == "train":
        if use_vars:
            outputs = exe.run(fetch_list=fetch_list, program=program, use_program_cache=True)
        else:
            outputs = exe.run(feed=feed_dict, fetch_list=fetch_list, program=program, use_program_cache=True)
            tower_mems_np = outputs[-len(mems_vars['new_mems']):]

        outputs_dict = {}
        for var_name, output_var in zip(fetch_names, outputs):
            outputs_dict[var_name] = output_var

        ret = {"loss": np.mean(outputs_dict['loss']), 
               "accuracy": np.mean(outputs_dict['accuracy']), 
               "learning_rate": np.mean(outputs_dict['scheduled_lr']),
               "tower_mems_np": tower_mems_np}
        return ret

    if phase == "eval" or phase == "test":
        pyreader.start()
        qids, labels, scores = [], [], []
        time_begin = time.time()
        
        all_results = []
        total_cost, total_num_seqs= 0.0, 0.0
        RawResult = namedtuple("RawResult", ["unique_id", "prob", "label"])
        while True:
            try:
                if use_vars:
                    outputs = exe.run(
                        program=program, fetch_list=fetch_list, use_program_cache=True)
                else:
                    feed_dict = {}
                    for m, m_np in zip(mems_vars['mems'], tower_mems_np):
                        feed_dict[m.name] = m_np
                    outputs = exe.run(feed=feed_dict, fetch_list=fetch_list, program=program, use_program_cache=True)
                    tower_mems_np = outputs[-len(mems_vars['new_mems']):]
                    outputs = outputs[:-len(mems_vars['new_mems'])]

                np_loss, np_acc, np_probs, np_labels, np_num_seqs, np_qids, np_need_cal_loss = outputs

                if int(np_need_cal_loss) == 1:
                    total_cost += np.sum(np_loss * np_num_seqs)
                    total_num_seqs += np.sum(np_num_seqs)
                    for idx in range(np_qids.shape[0]):
                        if len(all_results) % 1000 == 0 and len(all_results):
                            print("processining example: %d" % len(all_results))
                        qid_each = int(np_qids[idx])
                        probs_each = [float(x) for x in np_probs[idx].flat]
                        label_each = int(np_labels[idx])
                        all_results.append(
                        RawResult(
                            unique_id=qid_each,
                            prob=probs_each,
                            label=label_each))

            except fluid.core.EOFException:
                pyreader.reset()
                break
        time_end = time.time()
        
        output_path = "./tmpout"
        mul_pro_test = MultiProcessEvalForErnieDoc(output_path, phase, trainers_num, trainers_id)
        is_print = True
        if mul_pro_test.dev_count > 1:
            is_print = False
            mul_pro_test.write_result(all_results)
            if trainers_id == 0:
                is_print = True
                all_results = mul_pro_test.concat_result(RawResult)

        if is_print:
            num_seqs, all_labels, all_probs = mul_pro_test.write_predictions(all_results)
            acc_func = Acc()
            accuracy = acc_func.eval([all_probs, all_labels])
            time_cost = time_end - time_begin
            print("[%d_%s evaluation] ave loss: %f, ave acc: %f, data_num: %d, elapsed time: %f s"
                % (steps, phase, total_cost / total_num_seqs, accuracy, num_seqs, time_cost))
 
