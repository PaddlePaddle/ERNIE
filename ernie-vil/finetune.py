#    Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" finetuning vison-language task """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import datetime
import argparse
import numpy as np
import multiprocessing
import json
import math
import pickle 

from reader.vcr_finetuning import VCRDataJointReader
from reader.refcoco_plus_finetuning import RefcocoPlusDataReader
from reader.flickr_finetuning import FlickrDataReader
from reader.vqa_finetuning import VQADataReader
from model.ernie_vil import ErnieVilModel, ErnieVilConfig
from optim.optimization import optimization
from utils.args import print_arguments
from utils.init import init_checkpoint, init_pretraining_params
from utils.loss import circle_loss
from args.finetune_args import parser

import paddle.fluid as fluid

args = parser.parse_args()

# yapf: enable.

#READERS = {"vcr": VCRDataJointReader, "vqa": VQADataReader, "refcoco_plus": RefcocoPlusReader, "flickr": FlickrReader}
READERS = {"vcr": VCRDataJointReader, "refcoco_plus": RefcocoPlusDataReader, 
           "flickr": FlickrDataReader, "vqa": VQADataReader}


def write_result_file(res_arr, qids, labels, ans_arr):
    """ trans batch results into json format (for VQA test)
    """
    for i in range(len(qids)):
        #print(int(qids[i]))
        res = {
             'question_id': int(qids[i]),
             'answer': ans_arr[labels[i]]
            }
        res_arr.append(res)
    return res_arr


def format_result(res_arr, qids, pred, labels, scores):
    """
        trans batch results into json format
    """
    for i in range(len(qids)):
        res="\t".join([str(qids[i]), str(pred[i]), str(labels[i]), " ".join(["%.5f" % s for s in scores[i]])])
        res_arr.append(res)
    return res_arr


def vqa_classifier_loss(emb_fuse, hidden_size, label, is_test):
    """
       classifier loss for vqa
    """
    co_emb_size = emb_fuse.shape[-1]
    num_class = label.shape[-1]

    weight_init_0 = fluid.initializer.UniformInitializer(
        low = - math.sqrt(3 / (co_emb_size + hidden_size)), high = math.sqrt(3 / (co_emb_size + hidden_size)))

    weight_init_1 = fluid.initializer.UniformInitializer(
        low = - math.sqrt(3 / (hidden_size + num_class)), high =  math.sqrt(3 / (hidden_size + num_class)))


    hidden_emb = fluid.layers.fc(input=emb_fuse, size=hidden_size,
                                 param_attr = fluid.ParamAttr(
                                 initializer = weight_init_0, name = "vqa_fc_w_0"),
                                 bias_attr = "vqa_fc_b_0", act='relu')

    hidden_emb = fluid.layers.dropout(hidden_emb, 0.5, dropout_implementation="upscale_in_train")

    pred = fluid.layers.fc(input=hidden_emb,
                           param_attr = fluid.ParamAttr(
                           initializer = weight_init_1, name = "vqa_fc_w_1"),
                           bias_attr = "vqa_fc_b_1", size=num_class)

    pred = fluid.layers.cast(x=pred, dtype='float32')
    cost = fluid.layers.sigmoid_cross_entropy_with_logits(pred, label, name="cross_entropy_loss")
    cost = fluid.layers.reduce_sum(cost, -1)
    max_conf_label = fluid.layers.argmax(pred, axis=1)
    max_conf_label_re = fluid.layers.reshape(max_conf_label, [-1, 1])
    one_hot_label = fluid.layers.one_hot(input=max_conf_label_re, depth=num_class)
    acc = fluid.layers.reduce_sum(one_hot_label * label, -1)
    return max_conf_label, fluid.layers.reduce_mean(cost), fluid.layers.reduce_mean(acc)


def create_vqa_model(pyreader_name, ernie_config, task_group, is_prediction=False):
    """
        detail model arch for vqa task
    """
    num_class = task_group[0]["num_class"]
    classifier_hid_size = task_group[0]["classifier_hid_size"]
    shapes=[[-1, args.max_seq_len, 1],    #src_id 
            [-1, args.max_seq_len, 1],    #pos_id
            [-1, args.max_seq_len, 1],    #sent_id
            [-1, args.max_seq_len, 1],    #input_mask
            [-1, args.max_img_len, args.feature_size],  #image_embedding
            [-1, args.max_img_len, 5],     #image_loc
            [-1, args.max_img_len, 1],    #image_mask
            [-1, num_class],     #soft_labels
            [-1],                     #q_id
            ]
    dtypes = ['int64', 'int64', 'int64', 'float32', 'float32', 'float32', 'float32', 'float32', 'int64']
              #srd_id   pos_id   sent_id  input_mask image_emb image_loc image_mask, labels
    lod_levels = [0] * len(dtypes)

    pyreader = fluid.layers.py_reader(
        capacity=30,
        shapes=shapes,
        dtypes=dtypes,
        lod_levels=lod_levels,
        name=pyreader_name,
        use_double_buffer=True)

    inputs = fluid.layers.read_file(pyreader)
    src_ids, pos_ids, sent_ids, input_mask, image_embeddings, \
        image_loc, image_mask, labels, q_ids = inputs[: 11]
    ernie_vil = ErnieVilModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        image_embeddings=image_embeddings,
        image_loc=image_loc,
        input_image_mask=image_mask,
        config=ernie_config
        )

    h_cls, h_img = ernie_vil.get_pooled_output()
    score = ernie_vil.get_match_score(h_cls, h_img, "mul")
    pred_label, loss, acc = vqa_classifier_loss(score, classifier_hid_size, labels, args.do_test)

    task_vars = [loss, acc, pred_label, q_ids]
    for var in task_vars:
        var.persistable = True

    return pyreader, task_vars


def create_refcoco_plus_model(pyreader_name, ernie_config, task_group, is_prediction=False):
    """
        detail model arch for refcoco_plus task
    """
    shapes=[[-1, args.max_seq_len, 1],                  #src_id
            [-1, args.max_seq_len, 1],                  #pos_id
            [-1, args.max_seq_len, 1],                  #sent_id
            [-1, args.max_seq_len, 1],                  #input_mask
            [-1, 1],                                    #seq_lens
            [-1, args.max_img_len, args.feature_size],  #image_embedding
            [-1, args.max_img_len, 5],                  #image_loc
            [-1, args.max_img_len, 1],                  #image_mask
            [-1, args.max_img_len, 1],                  #labels
            [-1, 1],                                    #add_items
            ]
    dtypes = ['int64', 'int64', 'int64', 'float', 'int64', \
            'float32', 'float32', 'float32', 'float32', 'float32']

    lod_levels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    pyreader = fluid.layers.py_reader(
        capacity=30,
        shapes=shapes,
        dtypes=dtypes,
        lod_levels=lod_levels,
        name=pyreader_name,
        use_double_buffer=True)
    inputs = fluid.layers.read_file(pyreader)
    src_ids, pos_ids, sent_ids, input_mask, seq_lens, \
            image_embeddings, image_loc, image_mask, labels, add_item = inputs[: 10]

    ernie_vil = ErnieVilModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        image_embeddings=image_embeddings,
        image_loc=image_loc,
        input_image_mask=image_mask,
        config=ernie_config
        )

    enc_l_out, enc_vl_out = ernie_vil.get_sequence_output()
    pred_fc = enc_vl_out
    if args.seq_dropout > 0.0:
        pred_fc = fluid.layers.dropout(pred_fc, args.seq_dropout, dropout_implementation="upscale_in_train")

    logits = fluid.layers.fc(
        input=pred_fc, size=1,
        num_flatten_dims=2,
        param_attr=fluid.ParamAttr(
            name="cls_seq_label_vl_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02),
            learning_rate=1.0),
        bias_attr=fluid.ParamAttr(
            name="cls_seq_label_vl_out_b",
            initializer=fluid.initializer.Constant(0.),
            learning_rate=1.0))

    logits_re = fluid.layers.reduce_mean(logits, -1)
    labels_re = fluid.layers.reduce_mean(labels, -1)
    input_image_mask = fluid.layers.reduce_mean(image_mask, -1)
    ce_loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits_re, labels_re)
    ce_loss = ce_loss * input_image_mask
    loss = fluid.layers.reduce_sum(ce_loss) / fluid.layers.reduce_sum(input_image_mask)
    loss = fluid.layers.mean(x = loss) * args.batch_size
    with_mask_loss = fluid.layers.mean(ce_loss) * args.batch_size
    if is_prediction:
        task_vars = [logits, image_loc, labels, add_item]
    else:
        task_vars = [loss, with_mask_loss] 
    for var in task_vars:
        var.persistable = True
    return pyreader, task_vars


def create_flickr_model(pyreader_name, ernie_config, task_group, is_prediction=False):
    """
       detailed  model arch for flickr task
    """
    shapes=[[-1, args.max_seq_len, 1],    #src_id 
            [-1, args.max_seq_len, 1],    #pos_id
            [-1, args.max_seq_len, 1],    #sent_id
            [-1, args.max_seq_len, 1],    #input_mask
            [-1, args.max_img_len, args.feature_size],  #image_embedding
            [-1, args.max_img_len, 5],     #image_loc
            [-1, args.max_img_len, 1],  #image_mask
            [-1, 1],     #labels
            [-1, 1],     #ids
            ]
    dtypes = ['int64', 'int64', 'int64', 'float', 'float32', 'float32', 'float32', 'int64', 'int64']
              #srd_id   pos_id   sent_id  input_mask image_emb image_loc image_mask, labels, ids
    #lod_levels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lod_levels = [0] * len(dtypes)

    pyreader = fluid.layers.py_reader(
        capacity=30,
        shapes=shapes,
        dtypes=dtypes,
        lod_levels=lod_levels,
        name=pyreader_name,
        use_double_buffer=True)

    inputs = fluid.layers.read_file(pyreader)
    src_ids, pos_ids, sent_ids, input_mask, image_embeddings, \
        image_loc, image_mask, labels, ids = inputs[: 9]
    ernie = ErnieVilModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        image_embeddings=image_embeddings,
        image_loc=image_loc,
        input_image_mask=image_mask,
        config=ernie_config
        )

    h_cls, h_img = ernie.get_pooled_output()
    match_emb = ernie.get_match_score(h_cls, h_img)

    match_score = fluid.layers.fc(
        input=match_emb,
        size=1,
        act=None,
        param_attr=fluid.ParamAttr(
            name='match_fc.w_0',
            initializer=fluid.initializer.Xavier()),
        bias_attr=fluid.ParamAttr(name='match_fc.b_0',
            initializer=fluid.initializer.UniformInitializer()))

    if not is_prediction:
        outs = len(task_group[0]["negative_schema"]) + 1
        match_score = fluid.layers.reshape(match_score, [-1, outs])
        match_score = fluid.layers.sigmoid(match_score)
        positive_score = match_score[:, 0]
        image_neg_score = match_score[:, 1:int((outs + 1) / 2)]
        caption_neg_score = match_score[:, int((outs + 1) / 2):]

        positive_score = fluid.layers.reshape(x=positive_score, shape=[-1, 1])
        loss_c = circle_loss(positive_score, caption_neg_score, args.margin, args.scale_circle)
        loss_i = circle_loss(positive_score, image_neg_score, args.margin, args.scale_circle)
        #total_loss = fluid.layers.mean(loss_c + loss_i)
        total_loss = (loss_c + loss_i) / 2
        acc = fluid.layers.accuracy(match_score, labels, k=1)
        task_vars = [total_loss, acc, match_score, ids]
    else:
        outs = 1
        match_score = fluid.layers.reshape(match_score, [-1, outs])
        task_vars = [match_score, ids]
    for var in task_vars:
        var.persistable = True

    return pyreader, task_vars

def create_vcr_model(pyreader_name, ernie_config, task_group, is_prediction=False):
    """
        create model arc for vcr tasks
    """
    shapes = [[-1, args.max_seq_len, 1],    #src_id 
             [-1, args.max_seq_len, 1],    #pos_id
             [-1, args.max_seq_len, 1],    #sent_id
             [-1, args.max_seq_len, 1],    #input_mask
             [-1, args.max_img_len, args.feature_size],  #image_embedding
             [-1, args.max_img_len, 5],     #image_loc
             [-1, args.max_img_len, 1],    #image_mask
             [-1, 1],     #labels
             [-1, 1],     #qids
             [],          #task_index
             [-1, 1],     #binary_labels
             ]
    dtypes = ['int64', 'int64', 'int64', 'float32', 'float32', 'float32', 'float32', 
                       'int64', 'int64', 'int64', 'float32']
    lod_levels = [0] * len(dtypes)

    for _ in task_group:
        shapes.append([])
        dtypes.append('float')
        lod_levels.append(0)

    pyreader = fluid.layers.py_reader(
        capacity=30,
        shapes=shapes,
        dtypes=dtypes,
        lod_levels=lod_levels,
        name=pyreader_name,
        use_double_buffer=False)

    inputs = fluid.layers.read_file(pyreader)
    src_ids, pos_ids, sent_ids, input_mask, image_embeddings, \
         image_loc, image_mask, labels, q_ids, task_index, binary_labels = inputs[: 11]

    ernie_vil = ErnieVilModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        image_embeddings=image_embeddings,
        image_loc=image_loc,
        input_image_mask=image_mask,
        config=ernie_config
        )

    h_cls, h_img = ernie_vil.get_pooled_output()
    task_conf = task_group[0]
    fusion_method = task_conf["fusion_method"]
    fusion_fea = ernie_vil.get_match_score(text=h_cls, image=h_img,         \
                                           dropout_rate=task_conf["dropout_rate"],
                                           mode=fusion_method)
    if is_prediction:
        num_choice = int(task_conf['num_choice'])
        task_name = task_conf.get('task_prefix', 'vcr')
        score = fluid.layers.fc(fusion_fea, 1,
                                param_attr = fluid.ParamAttr(name = task_name + "_fc.w_0",
                                                    initializer = fluid.initializer.TruncatedNormal(scale = 0.02)),
                                                    bias_attr = task_name + "_fc.b_0")
        score = fluid.layers.reshape(score, shape = [-1, num_choice])
        _loss, _softmax = fluid.layers.softmax_with_cross_entropy(logits = score,
                                                                  label = labels, return_softmax = True)
        _acc = fluid.layers.accuracy(input = _softmax, label = labels)
        pred = fluid.layers.argmax(score, axis = 1)
        mean_loss = fluid.layers.mean(_loss)
        task_vars = [mean_loss, _acc, pred, q_ids, labels, _softmax]
        for var in task_vars:
            var.persistable = True
        return pyreader, task_vars
    else:
        start_ind = 11
        mean_loss = fluid.layers.zeros(shape = [1], dtype = 'float32')
        mean_acc = fluid.layers.zeros(shape = [1], dtype = 'float32')
        for task_conf in task_group:
            task_weight = inputs[start_ind]
            start_ind += 1
            num_choice = int(task_conf['num_choice'])
            task_name = task_conf.get('task_prefix', 'vcr')
            score = fluid.layers.fc(fusion_fea, 1,
                                    param_attr = fluid.ParamAttr(name = task_name + "_fc.w_0",
                                    initializer = fluid.initializer.TruncatedNormal(scale = 0.02)),
                                    bias_attr = task_name + "_fc.b_0")

            _loss = fluid.layers.sigmoid_cross_entropy_with_logits(score,
                                                                    binary_labels, name = "cross_entropy_loss")
            tmp_score = fluid.layers.reshape(score, shape = [-1, num_choice])
            _softmax = fluid.layers.softmax(tmp_score)
            _acc = fluid.layers.accuracy(input = _softmax, label = labels)
            _mean_loss = fluid.layers.mean(_loss)
            mean_loss += _mean_loss * task_weight
            mean_acc += _acc * task_weight
        task_vars = [fluid.layers.reduce_mean(mean_loss), mean_acc]
        for var in task_vars:
            var.persistable = True

        return pyreader, task_vars

#MODELS = {"vcr": create_vcr_model, "vqa": create_vqa_model, "refcoco+": create_refcoco_model}
MODELS = {"vcr": create_vcr_model, "refcoco_plus": create_refcoco_plus_model, 
          "flickr": create_flickr_model, "vqa": create_vqa_model}

def predict_wrapper(args,
                    exe,
                    ernie_config,
                    task_group,
                    test_prog=None,
                    pyreader=None,
                    graph_vars=None):
    """Context to do validation.
    """
    reader_name = READERS[args.task_name]
    data_reader = reader_name(
        task_group,
        split=args.test_split,
        vocab_path=args.vocab_path,
        is_test=True,
        batch_size=args.batch_size,
        epoch=args.epoch)
    if args.do_test:
        assert args.init_checkpoint is not None, "[FATAL] Please use --init_checkpoint '/path/to/checkpoints' \
                                                  to specify you pretrained model checkpoints"

        init_pretraining_params(exe, args.init_checkpoint, test_prog)
        print(("testing on %s %s split") % (args.task_name, args.test_split))

    def predict_vcr(exe=exe, pyreader=pyreader):
        """
            inference for vcr tasks
        """
        pyreader.decorate_tensor_provider(data_reader.data_generator())
        pyreader.start()

        task_acc = {}
        task_steps = {}
        steps = 0
        time_begin = time.time()
        task_name_list = [v.name for v in graph_vars]
        fetch_list = task_name_list

        print('task name list : ', task_name_list)
        sum_acc = 0
        res_arr = []
        while True:
            try:
                outputs = exe.run(fetch_list=fetch_list, program=test_prog)
                each_acc = outputs[1][0]
                preds = np.reshape(outputs[2], [-1])
                qids = np.reshape(outputs[3], [-1])
                labels = np.reshape(outputs[4], [-1])
                scores = np.reshape(outputs[5], [-1, 4])
                sum_acc += each_acc
                steps += 1
                if steps % 10 == 0:
                    print('cur_step:', steps, 'cur_acc:', sum_acc / steps)
                format_result(res_arr, qids.tolist(), preds.tolist(), labels.tolist(), scores.tolist())
            except fluid.core.EOFException:
                pyreader.reset()
                break

        used_time = time.time() - time_begin

        with open(args.result_file, "w") as f:
            for r in res_arr:
                f.write(r + "\n")

        print("average_acc:", sum_acc / steps)
        ret = {}
        ret["acc"] = "acc: %f" % (sum_acc / steps)  
        for item in ret:
            try:
                ret[item] = ret[item].split(':')[-1]
            except:
                pass
        return ret

    def predict_flickr(exe=exe, pyreader=pyreader):
        """
            inference for flickr tasks
        """
        pyreader.decorate_tensor_provider(data_reader.data_generator())
        pyreader.start()

        task_acc = {}
        task_steps = {} 
        steps = 0
        time_begin = time.time()
        task_name_list = [v.name for v in graph_vars]
        fetch_list = task_name_list
        print('task name list : ', task_name_list)
        out_file = open(args.result_file, 'w')
        sum_acc = 0
        res_arr = []
        while True:
            try:
                outputs = exe.run(fetch_list=fetch_list, program=test_prog)
                score = outputs[0]
                ids = outputs[1]
                for i in range(len(score)):
                    out_list = [str(score[i][0]), str(ids[i][0]), str(ids[i][1])]
                    out_file.write('\t'.join(out_list) + '\n')
                steps += 1

            except fluid.core.EOFException:
                pyreader.reset()
                break
        out_file.close()
        used_time = time.time() - time_begin
        return None
    
    def predict_vqa(exe=exe, pyreader=pyreader):
        """
            inference for vqa tasks
        """
        pyreader.decorate_tensor_provider(data_reader.data_generator())
        pyreader.start()

        appear_step = 0
        task_acc = {}
        task_steps = {}
        steps = 0
        time_begin = time.time()
        task_name_list = [v.name for v in graph_vars]
        fetch_list =  task_name_list

        print('task name list : ', task_name_list)
        sum_acc = 0
        total_data = 0
        res_arr = []
        pickle_file = task_group[0]["pickle_file"]
        pkl_file = open(pickle_file)
        ans_arr = pickle.load(pkl_file)
        while True:
            try:
                outputs = exe.run(fetch_list=fetch_list, program=test_prog)
                each_acc = outputs[1][0]
                labels = outputs[2]
                qids = outputs[3]
                total_data += len(qids.tolist())
                sum_acc += each_acc * len(qids.tolist())
                steps += 1
                if steps % 10 == 0:
                    print('cur_step:', steps, 'cur_acc:', sum_acc / total_data)
                write_result_file(res_arr, qids.tolist(), labels.tolist(), ans_arr)
            except fluid.core.EOFException:
                pyreader.reset()
                break

        used_time = time.time() - time_begin

        with open(args.result_file, "w") as f:
            json.dump(res_arr, f)
        print("step:", steps)
        print("average_acc:", sum_acc / total_data)
        ret = {}
        ret["acc"] = "acc: %f" % (sum_acc / total_data)
        for item in ret:
            try:
                ret[item] = ret[item].split(':')[-1]
            except:
                pass
        return ret
    
    def predict_refcoco_plus(exe=exe, pyreader=pyreader):
        """
            inference for refcoco_plus tasks
        """
        pyreader.decorate_tensor_provider(data_reader.data_generator())
        pyreader.start()

        task_acc = {}
        task_steps = {}
        steps = 0
        time_begin = time.time()
        task_name_list = [v.name for v in graph_vars]
        fetch_list = task_name_list

        print('task name list : ', task_name_list)
        res_arr = []
        acc_all = 0
        sample_all = 0
        while True:
            try:
                outputs = exe.run(fetch_list=fetch_list, program=test_prog)
                logits, image_locs, labels, items = outputs[0:]
                for i in range(len(items)):
                    acc = 0
                    logit, loc, label, item = logits[i], image_locs[i], labels[i], items[i]
                    number_box, width, height, w1, h1, w2, h2 = item
                    start_idx = 1
                    list_label = list(label[:, 0])[start_idx: int(number_box)]
                    list_logit = list(logit[:, 0])[start_idx: int(number_box)]
                    logit = logit[start_idx:int(number_box), 0]
                    pred = np.argmax(logit)
                    if label[pred + start_idx, 0] >= 0.5:
                        acc = 1
                    acc_all += acc
                    sample_all += 1

                print(acc_all * 1.0 / sample_all)
                steps += 1
            except fluid.core.EOFException:
                pyreader.reset()
                break
        print('all', sample_all, acc_all * 1.0 / sample_all) 

    if args.task_name == "vcr":
        return predict_vcr
    elif args.task_name == "refcoco_plus":
        return predict_refcoco_plus
    elif args.task_name == "flickr":
        return predict_flickr
    else:
        return predict_vqa


def get_optimizer(total_loss, train_program, startup_prog, args):
    """
        optimization func
    """
    decay_steps_str=args.decay_steps
    if decay_steps_str == "":
        decay_steps = []
    else:
        decay_steps = [int(s) for s in decay_steps_str.split(";")]
    scheduled_lr = optimization(
         loss=total_loss,
         warmup_steps=args.warmup_steps,
         num_train_steps=args.num_train_steps,
         learning_rate=args.learning_rate,
         train_program=train_program,
         startup_prog=startup_prog,
         weight_decay=args.weight_decay,
         scheduler=args.lr_scheduler,
         decay_steps=decay_steps,
         lr_decay_ratio=args.lr_decay_ratio,
         layer_decay_rate=args.layer_decay_rate,
         text_init_layers=args.text_init_layers,
         n_layers=args.n_layers)
    return scheduled_lr


def main(args):
    """
       Main func for downstream tasks
    """
    print("finetuning tasks start")
    ernie_config = ErnieVilConfig(args.ernie_config_path)
    ernie_config.print_config()

    with open(args.task_group_json) as f:
        task_group = json.load(f)
        print('task: ', task_group)

    startup_prog = fluid.Program()
    if args.do_train and args.do_test:
        print("can not set both do_train and do_test as True")
        return 

    model_name = MODELS[args.task_name]
    if args.do_train:
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, model_outputs = model_name(
                    pyreader_name='train_reader', ernie_config=ernie_config, task_group=task_group)

                total_loss = model_outputs[0]
                scheduled_lr = get_optimizer(total_loss, train_program, startup_prog, args)
    if args.do_test:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, model_outputs  = model_name(
                    pyreader_name='test_reader', ernie_config=ernie_config, task_group=task_group, is_prediction=True)
                total_loss = model_outputs[0]

        test_prog = test_prog.clone(for_test=True)
    
    if args.use_gpu:
        gpu_id = 0
        if os.getenv("FLAGS_selected_gpus"):
            gpu_id = int(os.getenv("FLAGS_selected_gpus"))
    place = fluid.CUDAPlace(gpu_id) if args.use_gpu else fluid.CPUPlace()

    print("theoretical memory usage: ")
    if args.do_train:
        print(fluid.contrib.memory_usage(
             program=train_program, batch_size=args.batch_size))
    if args.do_test:
        print(fluid.contrib.memory_usage(
            program=test_prog, batch_size=args.batch_size))

    nccl2_num_trainers = 1
    nccl2_trainer_id = 0
    print("args.is_distributed:", args.is_distributed)
    trainer_id = 0
    if args.is_distributed:
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        worker_endpoints_env = os.getenv("PADDLE_TRAINER_ENDPOINTS")
        current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
        worker_endpoints = worker_endpoints_env.split(",")
        trainers_num = len(worker_endpoints)

        print("worker_endpoints:{} trainers_num:{} current_endpoint:{} \
              trainer_id:{}".format(worker_endpoints, trainers_num,
                                    current_endpoint, trainer_id))

        # prepare nccl2 env.
        config = fluid.DistributeTranspilerConfig()
        config.mode = "nccl2"
        if args.nccl_comm_num > 1:
            config.nccl_comm_num = args.nccl_comm_num
        if args.use_hierarchical_allreduce and trainers_num > args.hierarchical_allreduce_inter_nranks:
            config.use_hierarchical_allreduce=args.use_hierarchical_allreduce
            config.hierarchical_allreduce_inter_nranks=args.hierarchical_allreduce_inter_nranks

            assert config.hierarchical_allreduce_inter_nranks > 1
            assert trainers_num % config.hierarchical_allreduce_inter_nranks == 0

            config.hierarchical_allreduce_exter_nranks = \
                trainers_num / config.hierarchical_allreduce_inter_nranks

        t = fluid.DistributeTranspiler(config=config)
        t.transpile(
            trainer_id,
            trainers=worker_endpoints_env,
            current_endpoint=current_endpoint,
            program=train_program,
            startup_program=startup_prog)

        nccl2_num_trainers = trainers_num
        nccl2_trainer_id = trainer_id

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if args.do_train:
        if args.init_checkpoint and args.init_checkpoint != "":
            sys.stderr.write('############################WARNING############################')
            sys.stderr.write('####### using init_pretraining_params, not init_checkpoint ####')
            sys.stderr.write('## meaning hyper param e.g. lr won\'t inherit from checkpoint##')
            sys.stderr.write('###############################################################')
            init_pretraining_params(exe, args.init_checkpoint, train_program)

        reader_name=READERS[args.task_name]
        data_reader = reader_name(
            task_group,
            split="train",
            vocab_path=args.vocab_path,
            batch_size=args.batch_size,
            epoch=args.epoch)

    exec_strategy = fluid.ExecutionStrategy()
    if args.use_fast_executor:
        exec_strategy.use_experimental_executor = True
    exec_strategy.num_threads = 2
    
    exec_strategy.num_iteration_per_drop_scope = min(10, args.skip_steps)

    build_strategy = fluid.compiler.BuildStrategy()
    build_strategy.fuse_all_reduce_ops = False

    if args.use_fuse:
        build_strategy.fuse_all_reduce_ops = True

    if args.do_train:
        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            loss_name=total_loss.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy,
            main_program=train_program,
            num_trainers=nccl2_num_trainers,
            trainer_id=nccl2_trainer_id)

    if args.do_test: 
        predict = predict_wrapper(
            args,
            exe,
            ernie_config,
            task_group,
            test_prog=test_prog,
            pyreader=test_pyreader,
            graph_vars=model_outputs)
        result = predict()

    if args.do_train:
        train_pyreader.decorate_tensor_provider(data_reader.data_generator())
        train_pyreader.start()
        steps = 0
        time_begin = time.time()
        node_nums = 1 #int(os.getenv("PADDLE_NODES_NUM"))
        used_time_all = 0 
        
        if args.task_name == "refcoco_plus":
            metr = "all image loss"
        else:
            metr = "acc"
        
        while steps < args.num_train_steps:
            try:
                steps += node_nums
                skip_steps = args.skip_steps * node_nums
                fetch_list = []
                if nccl2_trainer_id == 0 and steps % skip_steps == 0:
                    task_name_list = [v.name for v in model_outputs]
                    fetch_list = task_name_list
                    fetch_list.append(scheduled_lr.name)
                
                time_begin = time.time()
                outputs = train_exe.run(fetch_list=fetch_list)
                
                if outputs:
                    print("feed_queue size", train_pyreader.queue.size())
                    progress_file = data_reader.get_progress()
                    epoch = progress_file["current_epoch"]
                    current_file_index = progress_file["current_file_index"]
                    total_file =  progress_file["total_file"]
                    current_file = progress_file["current_file"]
                    print(
                        "epoch: %d, progress: %d/%d, step: %d, loss: %f, "
                        "%s : %f"
                        % (epoch, current_file_index, total_file, steps,
                           outputs[0][0],
                           metr,
                           outputs[1][0]))

                    np_lr = outputs[-1:]

                    date_str = datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")

                    np_lr = float(np.mean(np_lr[0]))
                    print("%s current learning_rate:%.8f" % (date_str, np_lr))

                    if steps % args.save_steps == 0:
                        save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                        print("save_path:", save_path)
                        fluid.io.save_persistables(exe, save_path, train_program)
                    time_end = time.time()
                    used_time = time_end - time_begin
                    time_end = time_begin
                    print("used_time:", used_time)  
            except fluid.core.EOFException:
                train_pyreader.reset()
                break


if __name__ == '__main__':
    print_arguments(args)
    main(args)

