#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import print_function

import sys
import subprocess
import commands
import os
import six
import copy
import argparse
import time
import random

sys.path.append("./")

from utils.args import ArgumentGroup, print_arguments, inv_arguments
from utils.finetune_args import parser as finetuning_parser

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
multip_g = ArgumentGroup(parser, "multiprocessing", 
        "start paddle training using multi-processing mode.")
multip_g.add_arg("node_ips", str, None, 
        "paddle trainer ips")
multip_g.add_arg("node_id", int, None, 
        "the trainer id of the node for multi-node distributed training.")
multip_g.add_arg("print_config", bool, True, 
        "print the config of multi-processing mode.")
multip_g.add_arg("current_node_ip", str, None, 
        "the ip of current node.")
multip_g.add_arg("split_log_path", str, "log",
        "log path for each trainer.")
multip_g.add_arg("log_prefix", str, "",
        "the prefix name of job log.")
multip_g.add_arg("training_script", str, None, "the program/script to be lauched "
        "in parallel followed by all the arguments", positional_arg=True)
multip_g.add_arg("training_script_args", str, None,
        "training script args", positional_arg=True, nargs=argparse.REMAINDER)

# yapf: enable


def start_procs(args):
    procs = []
    log_fns = []

    selected_gpus=os.getenv("CUDA_VISIBLE_DEVICES", "")
    if selected_gpus == "":
        nproc_per_node = len(os.popen("lspci | grep -i nvidia").readlines())
    else:
        nproc_per_node = len(selected_gpus.split(","))
    selected_gpus = [i for i in range(nproc_per_node)]

    default_env = os.environ.copy()

    node_id = args.node_id
    node_ips = [x.strip() for x in args.node_ips.split(',')]
    current_ip = args.current_node_ip
    num_nodes = len(node_ips)
    selected_gpu_num = len(selected_gpus)

    all_trainer_endpoints = ""
    for ip in node_ips:
        for i in range(nproc_per_node):
            if all_trainer_endpoints != "":
                all_trainer_endpoints += ","
            all_trainer_endpoints += "%s:617%d" % (ip, i)

    nranks = num_nodes * nproc_per_node
    gpus_per_proc = nproc_per_node % selected_gpu_num 
    if gpus_per_proc == 0:
        gpus_per_proc =  selected_gpu_num / nproc_per_node
    else:
        gpus_per_proc =  selected_gpu_num / nproc_per_node + 1

    selected_gpus_per_proc = [selected_gpus[i:i + gpus_per_proc] for i in range(0, len(selected_gpus), gpus_per_proc)]

    if args.print_config:
        print("all_trainer_endpoints: ", all_trainer_endpoints, 
              ", node_id: ", node_id,
              ", current_ip: ", current_ip,
              ", num_nodes: ", num_nodes,
              ", node_ips: ", node_ips,
              ", gpus_per_proc: ", gpus_per_proc,
              ", selected_gpus_per_proc: ", selected_gpus_per_proc,
              ", nranks: ", nranks)

    current_env = copy.copy(default_env)
    procs = []
    cmds = []
    log_fns = []
    for i in range(0, nproc_per_node):
        trainer_id = node_id * nproc_per_node + i
        current_env.update({
            "FLAGS_selected_gpus": "%s" % ",".join([str(s) for s in selected_gpus_per_proc[i]]),
            "PADDLE_TRAINER_ID" : "%d" % trainer_id,
            "PADDLE_CURRENT_ENDPOINT": "%s:617%d" % (current_ip, i),
            "PADDLE_TRAINERS_NUM": "%d" % nranks,
            "PADDLE_TRAINER_ENDPOINTS": all_trainer_endpoints,
            "PADDLE_NODES_NUM": "%d" % num_nodes,
            "PADDLE_PROC_PER_NODE": "%d" % nproc_per_node
        })

        cmd = [sys.executable, "-u",
               args.training_script] + args.training_script_args
        cmds.append(cmd)

        if args.split_log_path:
            fn = open("%s/%sjob.log.%d" % (args.split_log_path, args.log_prefix, trainer_id), "a")
            log_fns.append(fn)
            process = subprocess.Popen(cmd, env=current_env, stdout=fn, stderr=fn)
        else:
            process = subprocess.Popen(cmd, env=current_env)
        procs.append(process)

    for i in range(len(procs)):
        proc = procs[i]
        proc.wait()
        if len(log_fns) > 0:
            log_fns[i].close()
        if proc.returncode != 0:    
            raise subprocess.CalledProcessError(returncode=procs[i].returncode,
                                                cmd=cmds[i])
        else:
            print("proc %d finsh" % i)


def main(args):

    def get_param(name):
        key = "--" + name
        if key not in args.training_script_args:
            return None
        index = args.training_script_args.index(key) + 1
        return args.training_script_args[index]

    rs_index = -1
    rs_name = "--random_seed"
    has_rs = False
    if rs_name in args.training_script_args:
        rs_index = args.training_script_args.index(rs_name) + 1
        if args.training_script_args[rs_index] != '-1':
            has_rs = True
    else:
        args.training_script_args += [rs_name, '-1']
        rs_index = args.training_script_args.index(rs_name) + 1

    if not has_rs:
        args.training_script_args[rs_index] = str(random.randint(0, 100000))
    if args.print_config:
        print_arguments(args)
    start_procs(args)
    if not has_rs:
        args.training_script_args[rs_index] = '-1'

if __name__ == "__main__":
    lanch_args = parser.parse_args()
    finetuning_args = finetuning_parser.parse_args(
            lanch_args.training_script_args)
    
    main(lanch_args)
