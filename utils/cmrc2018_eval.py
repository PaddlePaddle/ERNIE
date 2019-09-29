# -*- coding: utf-8 -*-
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
'''
Evaluation script for CMRC 2018
version: v5
Note: 
v5 formatted output, add usage description
v4 fixed segmentation issues
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

from collections import Counter, OrderedDict
import string
import re
import argparse
import json
import sys
import nltk
import pdb


# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
    in_str = in_str.lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = [
        '-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', '，', '。', '：',
        '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、', '「', '」', '（',
        '）', '－', '～', '『', '』'
    ]
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    #handling last part
    if temp_str != "":
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


# remove punctuation
def remove_punctuation(in_str):
    in_str = in_str.lower().strip()
    sp_char = [
        '-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', '，', '。', '：',
        '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、', '「', '」', '（',
        '）', '－', '～', '『', '』'
    ]
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


# find longest common string
def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax


#
def evaluate(ground_truth_file, prediction_file):
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    for instances in ground_truth_file["data"]:
        for instance in instances["paragraphs"]:
            context_text = instance['context'].strip()
            for qas in instance['qas']:
                total_count += 1
                query_id = qas['id'].strip()
                query_text = qas['question'].strip()
                answers = [ans["text"] for ans in qas["answers"]]

                if query_id not in prediction_file:
                    sys.stderr.write('Unanswered question: {}\n'.format(
                        query_id))
                    skip_count += 1
                    continue

                prediction = prediction_file[query_id]
                f1 += calc_f1_score(answers, prediction)
                em += calc_em_score(answers, prediction)

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return f1_score, em_score, total_count, skip_count


def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision = 1.0 * lcs_len / len(prediction_segs)
        recall = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em


def eval_file(dataset_file, prediction_file):
    ground_truth_file = json.load(open(dataset_file, 'r'))
    prediction_file = json.load(open(prediction_file, 'r'))
    F1, EM, TOTAL, SKIP = evaluate(ground_truth_file, prediction_file)
    AVG = (EM + F1) * 0.5
    return EM, F1, AVG, TOTAL


if __name__ == '__main__':
    EM, F1, AVG, TOTAL = eval_file(sys.argv[1], sys.argv[2])
    print(EM)
    print(F1)
    print(TOTAL)
