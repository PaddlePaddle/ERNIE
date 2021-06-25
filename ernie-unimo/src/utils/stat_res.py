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
"""results statistics"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import argparse
from args import ArgumentGroup

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "stat", "stat configuration")
model_g.add_arg("log_dir", str, None, "stat log dir")
model_g.add_arg("random_slot", int, 2, "random slot in log file")
model_g.add_arg("key_words", str, "lanch.log", "key words indentify log file")
model_g.add_arg("line_prefix", str, "Best validation result:", "key words indentify final score to stat")
model_g.add_arg("score_slot", int, -1, "score slot in stat line")
model_g.add_arg("final_res_file", str, "final_res.txt", "the file to save final stat score")

args = parser.parse_args()


def get_res(infile):
    """get results"""
    acc = 0
    with open(infile) as fr:
        for line in fr.readlines():
            line = line.strip('\r\n')
            if line.startswith(args.line_prefix):
                acc = float(line.split(' ')[args.score_slot])
    return acc


def print_stat(score_files):
    """print statistics"""
    nums = len(score_files)
    max_score, max_score_file = score_files[0]
    min_score, min_score_file = score_files[-1]
    median_score, median_score_file = score_files[int(nums / 2)]
    mean_score = np.average([s for s, f in score_files])

    log = 'tot_random_seed %d\nmax_score %f max_file %s\nmin_score %f min_file %s' \
          '\nmedian_score %f median_file %s\navg_score %f' % \
          (nums, max_score, max_score_file, min_score, min_score_file,
           median_score, median_score_file, mean_score)
    print(log)


score_file = {}
for file in os.listdir(args.log_dir):
    if args.key_words in file:
        randint = file.split('_')[args.random_slot]
        acc = get_res(os.path.join(args.log_dir, file))
        if randint in score_file:
            score_file[randint].append((acc, file))
        else:
            score_file[randint] = [(acc, file)]

best_randint_score_file = []
for randint, s_f in score_file.items():
    sorted_s_f = sorted(s_f, key=lambda a: a[0], reverse=True)
    best_randint_score_file.append(sorted_s_f[0])

best_randint_score_file = sorted(best_randint_score_file, key=lambda a: a[0], reverse=True)

sys.stdout = open(os.path.join(args.log_dir, args.final_res_file), 'w')
print_stat(best_randint_score_file)
