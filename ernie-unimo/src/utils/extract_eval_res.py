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
"""extract eval results"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import re
import argparse
from args import ArgumentGroup


# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "stat", "stat configuration")
model_g.add_arg("log_dir", str, None, "stat log dir")
model_g.add_arg("file_name", str, "job.log.0", "key words indentify log file")
model_g.add_arg("final_res_file", str, "final_res.txt", "the file to save final stat score")
args = parser.parse_args()


def extract_res(infile):
    """extract eval results"""
    res = []
    with open(infile) as fr:
        for line in fr.readlines():
            line = line.strip('\r\n')
            pattern = re.compile(r'\[\w+_step_\d+ evaluation\]')
            if pattern.match(line):
                res.append(line)
    return res


eval_res = {}
log_file = os.path.join(args.log_dir, args.file_name)
if os.path.exists(log_file):
    eval_res[args.file_name] = extract_res(log_file)
else:
    for sub_dir in os.listdir(args.log_dir):
        cur_log_dir = os.path.join(args.log_dir, sub_dir)
        log_file = os.path.join(cur_log_dir, args.file_name)
        if os.path.exists(log_file):
            res = extract_res(log_file)
            eval_res[sub_dir] = res


sys.stdout = open(os.path.join(args.log_dir, args.final_res_file), 'w')
for name, all_res in eval_res.items():
    print(name)
    for val in all_res:
        print(val)
    print('\n')
