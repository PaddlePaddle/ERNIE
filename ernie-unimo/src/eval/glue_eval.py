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
"""ultis help and eval functions for glue ."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

from scipy.stats import pearsonr, spearmanr
from six.moves import xrange
import paddle.fluid as fluid
from functools import partial
from collections import OrderedDict


def matthews_corrcoef(preds, labels):
    """matthews_corrcoef"""
    preds = np.array(preds)
    labels = np.array(labels)
    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))

    mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    ret = OrderedDict()
    ret['mat_cor'] = mcc
    ret['key_eval'] = "mat_cor"
    return ret


def f1_score(preds, labels):
    """f1_score"""
    preds = np.array(preds)
    labels = np.array(labels)

    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = (2 * p * r) / (p + r + 1e-8)
    ret = OrderedDict()
    ret['f1'] = f1
    ret['key_eval'] = "f1"
    return ret


def pearson_and_spearman(preds, labels):
    """pearson_and_spearman"""
    preds = np.array(preds)
    labels = np.array(labels)

    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    ret = OrderedDict()
    ret['pearson'] = pearson_corr
    ret['spearmanr'] = spearman_corr
    ret['p_and_sp'] = (pearson_corr + spearman_corr) / 2
    ret['key_eval'] = "p_and_sp"
    return ret


def acc_and_f1(preds, labels):
    """acc_and_f1"""
    preds = np.array(preds)
    labels = np.array(labels)

    acc = simple_accuracy(preds, labels)['acc']
    f1 = f1_score(preds, labels)['f1']

    ret = OrderedDict()
    ret['acc'] = acc
    ret['f1'] = f1
    ret['acc_and_f1'] = (acc + f1) / 2
    ret['key_eval'] = "acc_and_f1"
    return ret


def simple_accuracy(preds, labels):
    """simple_accuracy"""
    preds = np.array(preds)
    labels = np.array(labels)
    acc = (preds == labels).mean()
    ret = OrderedDict()
    ret['acc'] = acc
    ret['key_eval'] = "acc"
    return ret


def evaluate_mrr(preds):
    """evaluate_mrr"""
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
    """evaluate_map"""

    def singe_map(st, en):
        """singe_map"""
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
            if last_qid is not None:
                total_map += singe_map(st, i)
            st = i
            last_qid = qid

    total_map += singe_map(st, len(preds))
    return total_map / qnum