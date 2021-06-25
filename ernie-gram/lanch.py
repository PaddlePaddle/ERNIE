import sys
import subprocess
import os
import six
import copy
import argparse
import time
import random

from utils.args import ArgumentGroup, print_arguments, inv_arguments
from finetune.finetune_args import parser as finetuning_parser


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
multip_g.add_arg("nproc_per_node", int, 8, 
        "the number of process to use on each node.")
multip_g.add_arg("training_script", str, None, "the program/script to be lauched "
        "in parallel followed by all the arguments", positional_arg=True)
multip_g.add_arg("training_script_args", str, None,
        "training script args", positional_arg=True, nargs=argparse.REMAINDER)
grid_g = ArgumentGroup(parser, "grid_search", 
        "finetuning by grid searching.")
grid_g.add_arg("grid_lr", str, "1e-4", "learning rate.")
grid_g.add_arg("grid_bsz", str, "32", "barch size.")
grid_g.add_arg("grid_epoch", str, "3", "epoch.")


def start_procs(args, grid_search_config):
    procs = []
    log_fns = []

    default_env = os.environ.copy()

    node_id = args.node_id
    node_ips = [x.strip() for x in args.node_ips.split(',')]
    current_ip = args.current_node_ip
    num_nodes = len(node_ips)
    selected_gpus = list(map(str, range(args.nproc_per_node)))
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
        gpus_per_proc =  selected_gpu_num / args.nproc_per_node
    else:
        gpus_per_proc =  selected_gpu_num / args.nproc_per_node + 1

    selected_gpus_per_proc = [selected_gpus[i:i + int(gpus_per_proc)] for i in range(0, len(selected_gpus), int(gpus_per_proc))]

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
    for i in range(0, args.nproc_per_node):
        trainer_id = node_id * args.nproc_per_node + i
        current_env.update({
            "FLAGS_selected_gpus": "%s" % ",".join([str(s) for s in selected_gpus_per_proc[i]]),
            "PADDLE_TRAINER_ID" : "%d" % trainer_id,
            "PADDLE_CURRENT_ENDPOINT": "%s:617%d" % (current_ip, i),
            "PADDLE_TRAINERS_NUM": "%d" % nranks,
            "PADDLE_TRAINER_ENDPOINTS": all_trainer_endpoints,
            "PADDLE_NODES_NUM": "%d" % num_nodes,
            "GRID_SEARCH_LR": "%f" % grid_search_config["lr"],
            "GRID_SEARCH_EPOCH": "%d" % grid_search_config["epoch"],
            "GRID_SEARCH_BSZ": "%d" % grid_search_config["bsz"],
            "RANDSEED": "%d" % grid_search_config["random_seed"]
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


def main(lanch_args):
    if lanch_args.print_config:
        print_arguments(lanch_args)
    grid_lr = list(map(float, lanch_args.grid_lr.split(",")))
    grid_bsz = list(map(int, lanch_args.grid_bsz.split(",")))
    grid_epoch = list(map(int, lanch_args.grid_epoch.split(",")))
    for bsz in grid_bsz:
        for epoch in grid_epoch:
            for lr in grid_lr:
                lanch_args.log_prefix = ".".join([str(bsz), str(epoch), str(lr), ""])
                grid_search_config = {"bsz":bsz, "lr":lr, "epoch":epoch, "random_seed":random.randint(0, 10000)}
                start_procs(lanch_args, grid_search_config)


if __name__ == "__main__":
    lanch_args = parser.parse_args()
    
    while True:
        main(lanch_args)
