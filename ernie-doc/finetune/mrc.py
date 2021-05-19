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
"""Model for MRC."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
from collections import namedtuple

import paddle.fluid as fluid
from model.static.ernie import ErnieDocModel
from utils.metrics import EM_AND_F1 
from reader.tokenization import BasicTokenizer 
from utils.multi_process_eval import MultiProcessEvalForMrc

def create_model(args, ernie_config, mem_len=128, is_infer=False):
    """create model for mrc"""
    shapes = [[-1, args.max_seq_len, 1], [-1, 2 * args.max_seq_len + mem_len, 1], [-1, args.max_seq_len, 1],
            [-1, args.max_seq_len, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], []]
    dtypes = ['int64', 'int64', 'int64', 'float32', 'int64', 'int64', 'int64', 'int64', 'int64']
    names = ["src_ids", "pos_ids", "task_ids", "input_mask", "start_positions", \
             "end_positions", "qids", "gather_idx", "need_cal_loss"]

    inputs = []
    for shape, dtype, name in zip(shapes, dtypes, names):
        inputs.append(fluid.layers.data(name=name, shape=shape, dtype=dtype, append_batch_size=False))

    src_ids, pos_ids, task_ids, input_mask, start_positions, \
            end_positions, qids, gather_idx, need_cal_loss = inputs
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

    enc_out = ernie_doc.get_sequence_output()
    checkpoints = ernie_doc.get_checkpoints()
    mems, new_mems = ernie_doc.get_mem_output()
    enc_out = fluid.layers.dropout(
        x=enc_out,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")

    logits = fluid.layers.fc(
        input=enc_out,
        size=2,
        num_flatten_dims=2,
        param_attr=fluid.ParamAttr(
            name="cls_mrc_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_mrc_out_b", initializer=fluid.initializer.Constant(0.)))
    
    if is_infer:
        probs = fluid.layers.softmax(logits)
        feed_targets_name = [
            src_ids.name, pos_ids.name, task_ids.name, input_mask.name
        ]
        return pyreader, probs, feed_targets_name

    logits = fluid.layers.transpose(x=logits, perm=[2, 0, 1])
    start_logits, end_logits = fluid.layers.unstack(x=logits, axis=0)

    filter_output = list(map(lambda x: fluid.layers.gather(x, gather_idx), \
                                    [qids, start_logits, end_logits, start_positions, end_positions])) 
    qids, start_logits, end_logits, start_positions, end_positions = filter_output

    def compute_loss(logits, positions):
        """compute loss"""
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=logits, label=positions)
        loss = fluid.layers.mean(x=loss)
        return loss

    start_loss = compute_loss(start_logits, start_positions)
    end_loss = compute_loss(end_logits, end_positions)
    loss = (start_loss + end_loss) / 2.0
    loss *= need_cal_loss
    
    mems_vars = {'mems': mems, 'new_mems': new_mems}
    graph_vars = {
        "loss": loss,
        "qids": qids,
        "start_logits": start_logits,
        "end_logits": end_logits,
        "need_cal_loss": need_cal_loss
    }

    for k, v in graph_vars.items():
        v.persistable = True

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
             use_vars=False,
             examples=None,
             features=None,
             args=None):
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
               "learning_rate": np.mean(outputs_dict['scheduled_lr']),
               "tower_mems_np": tower_mems_np}
        return ret
    
    if phase == "eval" or phase == "test":
        output_dir = args.checkpoints
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_prediction_file = os.path.join(output_dir, phase + "_predictions.json")
        output_nbest_file = os.path.join(output_dir, phase + "_nbest_predictions.json")

        RawResult = namedtuple("RawResult",
                ["unique_id", "start_logits", "end_logits"])

        pyreader.start()
        all_results = []
        time_begin = time.time()
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
                np_loss, np_qids, np_start_logits, np_end_logits, np_need_cal_loss = outputs 

                if int(np_need_cal_loss) == 1:
                    for idx in range(np_qids.shape[0]):
                        if len(all_results) % 1000 == 0:
                            print("Processing example: %d" % len(all_results))
                        qid_each = int(np_qids[idx])
                        start_logits_each = [float(x) for x in np_start_logits[idx].flat]
                        end_logits_each = [float(x) for x in np_end_logits[idx].flat]
                        all_results.append(
                            RawResult(
                                unique_id=qid_each,
                                start_logits=start_logits_each,
                                end_logits=end_logits_each))
            except fluid.core.EOFException:
                pyreader.reset()
                break
        time_end = time.time()

        output_path = "./tmpout"
        tokenizer = BasicTokenizer(do_lower_case=args.do_lower_case)
        mul_pro_test = MultiProcessEvalForMrc(output_path, phase, trainers_num,
                                                  trainers_id, tokenizer)
        
        is_print = True
        if mul_pro_test.dev_count > 1:
            is_print = False
            mul_pro_test.write_result(all_results)
            if trainers_id == 0:
                is_print = True
                all_results = mul_pro_test.concat_result(RawResult)

        if is_print:
            mul_pro_test.write_predictions(examples,
                                           features,
                                           all_results,
                                           args.n_best_size,
                                           args.max_answer_length,
                                           args.do_lower_case,
                                           mul_pro_test.output_prediction_file,
                                           mul_pro_test.output_nbest_file)

            if phase == "eval":
                data_file = args.dev_set
            elif phase == "test":
                data_file = args.test_set
            
            elapsed_time = time_end - time_begin
            em_and_f1 = EM_AND_F1()
            em, f1, avg, total = em_and_f1.eval_file(data_file, mul_pro_test.output_prediction_file) 
            
            print("[%d_%s evaluation] em: %f, f1: %f, avg: %f, questions: %d, elapsed time: %f"
                % (steps, phase, em, f1, avg, total, elapsed_time))
 
                       
