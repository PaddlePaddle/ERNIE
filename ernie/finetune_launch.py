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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import sys
import subprocess
import os
import six
import copy
import argparse
import time
import logging

from utils.args import ArgumentGroup, print_arguments, prepare_logger
from finetune_args import parser as worker_parser

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
multip_g = ArgumentGroup(parser, "multiprocessing", 
        "start paddle training using multi-processing mode.")
multip_g.add_arg("node_ips", str, None, 
        "paddle trainer ips")
multip_g.add_arg("node_id", int, 0, 
        "the trainer id of the node for multi-node distributed training.")
multip_g.add_arg("print_config", bool, True, 
        "print the config of multi-processing mode.")
multip_g.add_arg("current_node_ip", str, None, 
        "the ip of current node.")
multip_g.add_arg("split_log_path", str, "log",
        "log path for each trainer.")
multip_g.add_arg("log_prefix", str, "",
        "the prefix name of job log.")
multip_g.add_arg("nproc_per_node", int, 8, 
        "the number of process to use on each node.")
multip_g.add_arg("selected_gpus", str, "0,1,2,3,4,5,6,7", 
        "the gpus selected to use.")
multip_g.add_arg("training_script", str, None, "the program/script to be lauched "
        "in parallel followed by all the arguments", positional_arg=True)
multip_g.add_arg("training_script_args", str, None,
        "training script args", positional_arg=True, nargs=argparse.REMAINDER)
# yapf: enable


log = logging.getLogger()

def start_procs(args):
    procs = []
    log_fns = []

    default_env = os.environ.copy()

    node_id = args.node_id
    node_ips = [x.strip() for x in args.node_ips.split(',')]
    current_ip = args.current_node_ip
    if args.current_node_ip is None:
        assert len(node_ips) == 1
        current_ip = node_ips[0]
        log.info(current_ip)

    num_nodes = len(node_ips)
    selected_gpus = [x.strip() for x in args.selected_gpus.split(',')]
    selected_gpu_num = len(selected_gpus)

    all_trainer_endpoints = ""
    for ip in node_ips:
        for i in range(args.nproc_per_node):
            if all_trainer_endpoints != "":
                all_trainer_endpoints += ","
            all_trainer_endpoints += "%s:617%d" % (ip, i)

    nranks = num_nodes * args.nproc_per_node
    gpus_per_proc = args.nproc_per_node % selected_gpu_num 
    if gpus_per_proc == 0:
        gpus_per_proc =  selected_gpu_num // args.nproc_per_node
    else:
        gpus_per_proc =  selected_gpu_num // args.nproc_per_node + 1

    selected_gpus_per_proc = [selected_gpus[i:i + gpus_per_proc] for i in range(0, len(selected_gpus), gpus_per_proc)]

    if args.print_config:
        log.info("all_trainer_endpoints: %s"
              ", node_id: %s"
              ", current_ip: %s"
              ", num_nodes: %s"
              ", node_ips: %s"
              ", gpus_per_proc: %s"
              ", selected_gpus_per_proc: %s"
              ", nranks: %s" % (
                all_trainer_endpoints, 
                node_id,
                current_ip,
                num_nodes,
                node_ips,
                gpus_per_proc,
                selected_gpus_per_proc,
                nranks))


    current_env = copy.copy(default_env)
    procs = []
    cmds = []
    log_fns = []
    for i in range(0, args.nproc_per_node):
        trainer_id = node_id * args.nproc_per_node + i
        assert current_ip is not None
        current_env.update({
            "FLAGS_selected_gpus": "%s" % ",".join([str(s) for s in selected_gpus_per_proc[i]]),
            "PADDLE_TRAINER_ID" : "%d" % trainer_id,
            "PADDLE_CURRENT_ENDPOINT": "%s:617%d" % (current_ip, i),
            "PADDLE_TRAINERS_NUM": "%d" % nranks,
            "PADDLE_TRAINER_ENDPOINTS": all_trainer_endpoints,
            "PADDLE_NODES_NUM": "%d" % num_nodes
        })

        try:
            idx = args.training_script_args.index('--is_distributed')
            args.training_script_args[idx + 1] = 'true'
        except ValueError:
            args.training_script_args += ['--is_distributed', 'true']

        cmd = [sys.executable, "-u",
               args.training_script] + args.training_script_args
        cmds.append(cmd)
        
        if args.split_log_path:
            logdir = "%s/%sjob.log.%d" % (args.split_log_path, args.log_prefix, trainer_id)
            try:
                os.mkdir(os.path.dirname(logdir))
            except OSError:
                pass
            fn = open(logdir, "a")
            log_fns.append(fn)
            process = subprocess.Popen(cmd, env=current_env, stdout=fn, stderr=fn)
            log.info('subprocess launched, check log at %s' % logdir)
        else:
            process = subprocess.Popen(cmd, env=current_env)
            log.info('subprocess launched')
        procs.append(process)

    try:
        for i in range(len(procs)):
            proc = procs[i]
            proc.wait()
            if len(log_fns) > 0:
                log_fns[i].close()
            if proc.returncode != 0:    
                raise subprocess.CalledProcessError(returncode=procs[i].returncode,
                                                    cmd=cmds[i])
            else:
                log.info("proc %d finsh" % i)
    except KeyboardInterrupt as e:
        for p in procs:
            log.info('killing %s' % p)
            p.terminate()


def main(args):
    if args.print_config:
        print_arguments(args)
    start_procs(args)


if __name__ == "__main__":
    prepare_logger(log)
    lanch_args = parser.parse_args()
    finetuning_args = worker_parser.parse_args(
            lanch_args.training_script_args)
    init_path = finetuning_args.init_pretraining_params 
    log.info("init model: %s" % init_path)
    if not finetuning_args.use_fp16:
        os.system('rename .master "" ' + init_path + '/*.master') 
    main(lanch_args)
