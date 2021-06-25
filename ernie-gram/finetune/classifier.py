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
import six
import subprocess
import numpy as np
import json
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
from six.moves import xrange
import paddle.fluid as fluid

from model.ernie import ErnieModel

if six.PY2:
    import commands as subprocess


def create_model(args, pyreader_name, ernie_config, is_prediction=False, is_classify=False, is_regression=False, for_race=False, has_fc=True):

    shapes = [[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1], [-1, 1],
                [-1, 1], [-1, args.max_seq_len, args.max_seq_len, 1]]
    dtypes=['int64', 'int64', 'int64', 'int64', 'float32', 'int64', 'int64', 'int64']
    lod_levels=[0, 0, 0, 0, 0, 0, 0, 0]
    if is_regression:
        dtypes[-3] = 'float32'

    if for_race:
        shapes.append([-1, 1])
        dtypes.append('float32')
        lod_levels.append(0)
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=shapes,
        dtypes=dtypes,
        lod_levels=lod_levels,
        name=pyreader_name,
        use_double_buffer=True)
    if for_race:
        (src_ids, sent_ids, pos_ids, task_ids, input_mask, labels,
            qids, rel_pos_scaler, labels_pair) = fluid.layers.read_file(pyreader)
    else:
        (src_ids, sent_ids, pos_ids, task_ids, input_mask, labels,
            qids, rel_pos_scaler) = fluid.layers.read_file(pyreader)

    checkpoints = []

    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=[pos_ids, rel_pos_scaler],
        sentence_ids=sent_ids,
        task_ids=task_ids,
        input_mask=input_mask,
        config=ernie_config,
        use_fp16=args.use_fp16)
    checkpoints.extend(ernie.get_checkpoints())

    cls_feats = ernie.get_pooled_output(has_fc)
    cls_feats = fluid.layers.dropout(
        x=cls_feats,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")
    size = 1 if for_race else args.num_labels # for race dataset
    logits = fluid.layers.fc(
        input=cls_feats,
        size=size,
        param_attr=fluid.ParamAttr(
            name="cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_out_b", initializer=fluid.initializer.Constant(0.)))
    if for_race:
        loss_pair = fluid.layers.sigmoid_cross_entropy_with_logits(logits, labels_pair)
        logits = fluid.layers.reshape(logits, [-1, 4])


    if is_prediction:
        probs = fluid.layers.softmax(logits)
        feed_targets_name = [
            src_ids.name, pos_ids.name, sent_ids.name, task_ids.name, input_mask.name
        ]
        return pyreader, probs, feed_targets_name

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    if is_classify:
        ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
                logits=logits, label=labels, return_softmax=True)

        loss = fluid.layers.mean(x=ce_loss)
        if for_race:
            loss += 0.5 * fluid.layers.mean(x=loss_pair)

        accuracy = fluid.layers.accuracy(input=probs, label=labels, total=num_seqs)

        graph_vars = {
            "loss": loss,
            "probs": probs,
            "accuracy": accuracy,
            "labels": labels,
            "num_seqs": num_seqs,
            "qids": qids,
            "checkpoints": checkpoints
        }
    elif is_regression:
        if False:
            logits = fluid.layers.sigmoid(logits)
        cost = fluid.layers.square_error_cost(input=logits, label=labels)
        loss = fluid.layers.mean(x=cost)
        graph_vars = {
            "loss": loss,
            "probs": logits,
            "labels": labels,
            "num_seqs": num_seqs,
            "qids": qids
        }


    for k, v in graph_vars.items():
        if k != "checkpoints":
            v.persistable = True

    return pyreader, graph_vars

def write_result(output_path, eval_phase, gpu_id, eval_index, save_lists=None):
    outfile = output_path + "/" + eval_phase
    if eval_index is not None:
        outfile_part = outfile + ".part" + str(gpu_id)
        writer = open(outfile_part, "w")
        write_content = "\t".join([str(i) for i in eval_index]) + "\n"
        writer.write(write_content)
        writer.close()
    if save_lists is not None:
        save_list_name = ["qids", "labels", "scores"]
        for idx in range(len(save_list_name)):
            save_list = json.dumps(save_lists[idx])
            savefile_part = outfile + "." + save_list_name[idx] + ".part." + str(gpu_id)
            list_writer = open(savefile_part, "w")
            list_writer.write(save_list)
            list_writer.close()
    tmp_writer = open(output_path + "/" + eval_phase + "_dec_finish." + str(gpu_id), "w")
    tmp_writer.close()


def concat_result(output_path, eval_phase, dev_count, num_eval_index, num_list=None, eval_span=None):
    outfile = output_path + "/" + eval_phase
    eval_index_all = [0.0] * num_eval_index
    eval_list_all = defaultdict(list)
    while True:
        _, ret = subprocess.getstatusoutput('find ' + output_path + \
            ' -maxdepth 1 -name ' + eval_phase + '"_dec_finish.*"')
        ret = ret.split("\n")
        if len(ret) != dev_count:
            time.sleep(1)
            continue

        for dev_cnt in range(dev_count):
            if not eval_span:
                fin = open(outfile + ".part" + str(dev_cnt))
                cur_eval_index_all = fin.readline().strip().split("\t")
                cur_eval_index_all = [float(i) for i in cur_eval_index_all]
                eval_index_all = list(map(lambda x :x[0]+x[1], zip(eval_index_all, cur_eval_index_all)))
                
            if num_list is not None:
                save_list_name = ["qids", "labels", "scores"]
                for idx in range(len(save_list_name)):    
                    fin_list = open(outfile + "." + save_list_name[idx] + ".part." + str(dev_cnt), "r")
                    eval_list_all[save_list_name[idx]].extend(json.loads(fin_list.read()))
            
        subprocess.getstatusoutput("rm " + outfile + ".*part*")
        subprocess.getstatusoutput("rm " + output_path + "/" + eval_phase + "_dec_finish.*")
        break
    if num_list is not None:
        return eval_list_all
    return eval_index_all

def merge_results(qids, labels, scores):
    dic = {}
    corr = 0
    for ind, qid in enumerate(map(str, qids)):
        if qid in dic:
            dic[qid]["scores"].append(scores[ind])
        else:
            dic[qid] = {}
            dic[qid]["labels"] = labels[ind]
            dic[qid]["scores"] = [scores[ind]]
    for qid in dic.keys():
        score = dic[qid]["scores"]
        pred = list(map(lambda i:(max(score[i]), score[i].index(max(score[i]))), range(len(score))))
        pred = sorted(pred, key=lambda x:x[0], reverse=True)[0][1]

        if pred == dic[qid]["labels"]:
            corr += 1
    return float(corr) / len(dic.keys()), len(dic.keys())

def evaluate_regression(exe, 
             test_program,
             test_pyreader,
             graph_vars,
             eval_phase,
             tag_num=None,
             dev_count=1, 
             metric='pearson_and_spearman'):

    if eval_phase == "train":
        # train_fetch_list = [graph_vars["loss"].name, graph_vars["num_seqs"].name]
        # if "learning_rate" in graph_vars:
        #    train_fetch_list.append(graph_vars["learning_rate"].name)
        train_fetch_list = [graph_vars["loss"].name]
        if "learning_rate" in graph_vars:
            train_fetch_list.append(graph_vars["learning_rate"].name)
        outputs = exe.run(fetch_list=train_fetch_list)
        ret = {"loss": np.mean(outputs[0])}
        if "learning_rate" in graph_vars:
            ret["learning_rate"] = float(outputs[1][0])
        return ret

    test_pyreader.start()
    total_cost, total_num_seqs = 0.0, 0.0
    qids, labels, scores = [], [], []

    fetch_list = [
        graph_vars["loss"].name,
        graph_vars["probs"].name, 
        graph_vars["labels"].name,
        graph_vars["qids"].name
    ]

    time_begin = time.time()
    while True:
        try:
            if dev_count == 1:
                np_loss, np_probs, np_labels, np_qids = exe.run(
                    program=test_program, fetch_list=fetch_list)
            else:
                np_loss, np_probs, np_labels, np_qids = exe.run(
                    fetch_list=fetch_list)
            #total_cost += np.sum(np_loss * np_num_seqs)
            #total_num_seqs += np.sum(np_num_seqs)
            labels.extend(np_labels.reshape((-1)).tolist())
            if np_qids is None:
                qids.extend(list(range(len(np_labels))))
            else: 
                qids.extend(np_qids.reshape(-1).tolist())
            scores.extend(np_probs.reshape(-1).tolist())
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()

    #cost = total_cost / total_num_seqs
    elapsed_time = time_end - time_begin

    meta = {}
    best_thre = None
    if metric == 'pearson_and_spearman':
        ret = pearson_and_spearman(scores, labels)
        meta['score'] = ret['pearson']
        print("[%s evaluation] ave loss: %f, pearson: %f, spearman: %f, corr: %f, elapsed time: %f s" \
            % (eval_phase, 0.0, ret['pearson'], ret['spearmanr'], ret['corr'], elapsed_time))
    elif metric == 'matthews_corrcoef':
        best_score = -1000000
        best_thresh = None
        scores = np.array(scores)
        scores = 1 / (1 + np.exp(-scores))
        for s in range(0, 1000):
            T = s / 1000.0
            pred = (scores > T).astype('int')
            matt_score = matthews_corrcoef(pred, labels)
            if matt_score > best_score:
                best_score = matt_score
                best_thre = T
        print("[%s evaluation] ave loss: %f, matthews_corrcoef: %f, data_num: %d, elapsed time: %f s, best_thres: %f" % (eval_phase, 0.0, best_score, total_num_seqs, elapsed_time, best_thre))
    else:
        raise ValueError('unsupported metric {}'.format(metric))

    #return {'best_thre': best_thre}, evaluate_info



def evaluate_classify(exe, test_program, test_pyreader, graph_vars, eval_phase, 
            use_multi_gpu_test=False, gpu_id=0, output_path="./tmpout", dev_count=1, metric='simple_accuracy', eval_span=False):
    train_fetch_list = [
        graph_vars["loss"].name, graph_vars["accuracy"].name,
        graph_vars["num_seqs"].name
    ]

    if eval_phase == "train":
        outputs = exe.run(fetch_list=train_fetch_list)
        ret = {"loss": np.mean(outputs[0]), "accuracy": np.mean(outputs[1])}
        return ret

    test_pyreader.start()
    total_cost, total_acc, total_num_seqs, tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    qids, labels, scores, preds = [], [], [], []
    time_begin = time.time()

    fetch_list = [
        graph_vars["loss"].name, graph_vars["accuracy"].name,
        graph_vars["probs"].name, graph_vars["labels"].name,
        graph_vars["num_seqs"].name, graph_vars["qids"].name
    ]
    while True:
        try:
            np_loss, np_acc, np_probs, np_labels, np_num_seqs, np_qids = exe.run(
                program=test_program, fetch_list=fetch_list)
            total_cost += np.sum(np_loss * np_num_seqs)
            total_acc += np.sum(np_acc * np_num_seqs)
            total_num_seqs += np.sum(np_num_seqs)
            labels.extend(np_labels.reshape((-1)).tolist())
            if np_qids is None:
                qids.extend(list(range(len(np_labels))))
            else: 
                qids.extend(np_qids.reshape(-1).tolist())
            scores.extend(np_probs.tolist())
            np_preds = np.argmax(np_probs, axis=1).astype(np.float32)
            preds.extend(np_preds.reshape((-1)).tolist())
            tp += np.sum((np_labels == 1) & (np_preds == 1))
            tn += np.sum((np_labels == 0) & (np_preds == 0))
            fp += np.sum((np_labels == 0) & (np_preds == 1))
            fn += np.sum((np_labels == 1) & (np_preds == 0))

        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()
    if True:
        if dev_count == 1:
            meta = {}
            evaluate_info = ""
            if metric == 'acc_and_f1':
                ret = acc_and_f1(preds, labels)
                print("[%s evaluation] ave loss: %f, ave_acc: %f, f1: %f, data_num: %d, elapsed time: %f s" \
                    % (eval_phase, total_cost / total_num_seqs, ret['acc'], ret['f1'], total_num_seqs, time_end - time_begin))
                meta['score'] = ret['f1']
                
            elif metric == 'matthews_corrcoef':
                ret = matthews_corrcoef(preds, labels)
                print("[%s evaluation] ave loss: %f, matthews_corrcoef: %f, data_num: %d, elapsed time: %f s" \
                    % (eval_phase, total_cost / total_num_seqs, ret, total_num_seqs, time_end - time_begin))
                meta['score'] = ret

            elif metric == 'matthews_corrcoef_and_accf1':

                mat_ret = matthews_corrcoef(preds, labels)
                sim_ret = acc_and_f1(preds, labels)

                evaluate_info = "[%s evaluation] ave loss: %f, matthews_corrcoef: %f, acc: %f, f1: %f, data_num: %d, elapsed time: %f s" \
                    % (eval_phase, cost, mat_ret, sim_ret['acc'], sim_ret['f1'], total_num_seqs, elapsed_time)

                meta['score'] = mat_ret
                    
            elif metric == 'pearson_and_spearman':
                ret = pearson_and_spearman(scores, labels)
                print("[%s evaluation] ave loss: %f, pearson:%f, spearman:%f, corr:%f, data_num: %d, elapsed time: %f s" \
                    % (eval_phase, total_cost / total_num_seqs, ret['pearson'], ret['spearman'], ret['corr'], total_num_seqs, time_end - time_begin))
                meta['score'] = (ret['pearson'] + ret['spearman']) / 2.0

            elif metric == 'simple_accuracy':
                ret = simple_accuracy(preds, labels)
                print("[%s evaluation] ave loss: %f, acc:%f, data_num: %d, elapsed time: %f s" \
                    % (eval_phase, total_cost / total_num_seqs, ret, total_num_seqs, time_end - time_begin))
                meta['score'] = ret

            elif metric == "acc_and_f1_and_mrr":
                ret_a = acc_and_f1(preds, labels)
                preds = sorted(zip(qids, scores, labels), key=lambda elem: (elem[0], -elem[1]))
                ret_b = evaluate_mrr(preds)
                evaluate_info = "[%s evaluation] ave loss: %f, acc: %f, f1: %f, mrr: %f, data_num: %d, elapsed time: %f s" \
                    % (eval_phase, cost, ret_a['acc'], ret_a['f1'], ret_b, total_num_seqs, elapsed_time)
                meta['score'] = ret_a['f1']
            else:
                raise ValueError('unsupported metric {}'.format(metric))
            
        else:
            if metric== 'simple_accuracy':
                if not eval_span:
                    write_result(output_path, eval_phase, gpu_id, [total_acc, total_num_seqs])
                    if gpu_id == 0:
                        acc_sum, data_num = concat_result(output_path, eval_phase, dev_count, 2)
                        print(
                            "[%s evaluation] ave loss: %f, ave acc: %f, data_num: %d, elapsed time: %f s"
                            % (eval_phase, total_cost / total_num_seqs, acc_sum / data_num, 
                               int(data_num), time_end - time_begin))
                else:
                    write_result(output_path, eval_phase, gpu_id, None, [qids, labels, scores])
                    if gpu_id == 0:
                        ret = concat_result(output_path, eval_phase, dev_count, 0, 1, True)
                        qids, labels, scores = ret["qids"], ret["labels"], ret["scores"]
                        acc, data_num = merge_results(qids, labels, scores)
                        print(
                            "[%s evaluation] ave loss: %f, ave acc: %f, data_num: %d, elapsed time: %f s"
                            % (eval_phase, total_cost / total_num_seqs, acc, 
                               int(data_num), time_end - time_begin))


            elif metric== 'matthews_corrcoef':
                write_result(output_path, eval_phase, gpu_id, [tp, tn, fp, fn])
                if gpu_id == 0:
                    tp, tn, fp, fn = concat_result(output_path, eval_phase, dev_count, 2)
                    mcc = ( (tp*tn)-(fp*fn)) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) )
                    print(
                        "[%s evaluation] ave loss: %f, ave mcc: %f, elapsed time: %f s"
                        % (eval_phase, total_cost / total_num_seqs, mcc, time_end - time_begin))
    else:
        is_print = True
        if dev_count > 1:
            is_print = False
            write_result(output_path, eval_phase, gpu_id, [total_correct_num, total_label_pos_num, total_pred_pos_num], [qids, labels, scores])
            if gpu_id == 0:
                is_print = True
                eval_index_all, eval_list_all =  concat_result(output_path, eval_phase, dev_count, 3, 3)
                total_correct_num, total_label_pos_num, total_pred_pos_num = eval_index_all
                qids, labels, scores = [eval_list_all[name] for name in ["qids", "labels", "scores"]]
                       
        if is_print:
            r = total_correct_num / total_label_pos_num
            p = total_correct_num / total_pred_pos_num
            f = 2 * p * r / (p + r)

            assert len(qids) == len(labels) == len(scores)
            preds = sorted(
                zip(qids, scores, labels), key=lambda elem: (elem[0], -elem[1]))
            mrr = evaluate_mrr(preds)
            map = evaluate_map(preds)

            print(
                "[%s evaluation] ave loss: %f, ave_acc: %f, mrr: %f, map: %f, p: %f, r: %f, f1: %f, data_num: %d, elapsed time: %f s"
                % (eval_phase, total_cost / total_num_seqs,
                   total_acc / total_num_seqs, mrr, map, p, r, f, total_num_seqs,
                   time_end - time_begin))

def simple_accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean()


def matthews_corrcoef(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))

    mcc = ( (tp*tn)-(fp*fn)) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) )
    return mcc

def pearson_and_spearman(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def evaluate_mrr(preds):
    last_qid = None
    total_mrr = 0.0
    qnum = 0.0
    rank = 0.0
    correct = False
    for qid, score, label in preds:
        if qid != last_qid:
            rank = 0.0
            qnum += 1
            correct = False
            last_qid = qid

        rank += 1
        if not correct and label != 0:
            total_mrr += 1.0 / rank
            correct = True

    return total_mrr / qnum


def evaluate_map(preds):
    def singe_map(st, en):
        total_p = 0.0
        correct_num = 0.0
        for index in xrange(st, en):
            if int(preds[index][2]) != 0:
                correct_num += 1
                total_p += correct_num / (index - st + 1)
        if int(correct_num) == 0:
            return 0.0
        return total_p / correct_num

    last_qid = None
    total_map = 0.0
    qnum = 0.0
    st = 0
    for i in xrange(len(preds)):
        qid = preds[i][0]
        if qid != last_qid:
            qnum += 1
            if last_qid != None:
                total_map += singe_map(st, i)
            st = i
            last_qid = qid

    total_map += singe_map(st, len(preds))
    return total_map / qnum

def acc_and_f1(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    acc = simple_accuracy(preds, labels)
    f1 = f1_score(preds, labels)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }
    
def f1_score(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    p = tp / (tp+fp)
    r = tp / (tp+fn)
    f1 = (2*p*r) / (p+r+1e-8)
    return f1


