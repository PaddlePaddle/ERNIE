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
"""Finetuning on mrc tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import multiprocessing
import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy

import reader.task_reader as task_reader
from model.optimization import optimization
from model.static.ernie import ErnieConfig, ErnieDocModel
from finetune.mrc import create_model, evaluate
from utils.args import print_arguments
from utils.init import init_model
from utils.finetune_args import parser

paddle.enable_static()
args = parser.parse_args()

def main(args):
    """main function"""
    print("""finetuning start""")
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()
    memory_len = ernie_config["memory_len"]
    d_dim = ernie_config["hidden_size"]
    n_layers = ernie_config["num_hidden_layers"]
    print("args.is_distributed:", args.is_distributed)

    exec_strategy = fluid.ExecutionStrategy()
    if args.use_fast_executor:
        exec_strategy.use_experimental_executor = True
    exec_strategy.num_threads = 4 if args.use_amp else 2
    exec_strategy.num_iteration_per_drop_scope = min(1, args.skip_steps)

    if args.is_distributed:
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        trainer_id = fleet.worker_index()
        current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
        worker_endpoints = fleet.worker_endpoints()
        trainers_num = len(worker_endpoints)
        print("worker_endpoints:{} trainers_num:{} current_endpoint:{} trainer_id:{}"
              .format(worker_endpoints, trainers_num, current_endpoint, trainer_id))

        dist_strategy = DistributedStrategy()
        dist_strategy.exec_strategy = exec_strategy
        dist_strategy.remove_unnecessary_lock = False
        dist_strategy.fuse_all_reduce_ops = False
        dist_strategy.nccl_comm_num = 1

        if args.use_amp:
            dist_strategy.use_amp = True
            dist_strategy.amp_loss_scaling = args.init_loss_scaling
        if args.use_recompute:
            dist_strategy.forward_recompute = True
            dist_strategy.enable_sequential_execution=True
    else:
        dist_strategy=None

    gpu_id = 0 
    gpus = fluid.core.get_cuda_device_count()
    if args.is_distributed:
        gpus = os.getenv("FLAGS_selected_gpus").split(",")
        gpu_id = int(gpus[0])
    
    if args.use_cuda:
        place = fluid.CUDAPlace(gpu_id)
        dev_count = fleet.worker_num() if args.is_distributed else gpus
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    trainer_id = fleet.worker_index()
    print("Device count %d, trainer_id:%d" % (dev_count, trainer_id))
    print('args.vocab_path', args.vocab_path)
    reader = task_reader.MRCReader(
        trainer_id=fleet.worker_index(),
        trainer_num=dev_count,
        vocab_path=args.vocab_path,
        memory_len=memory_len,
        repeat_input=args.repeat_input,
        train_all=args.train_all,
        eval_all=args.eval_all,
        label_map_config=args.label_map_config,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        in_tokens=args.in_tokens,
        random_seed=args.random_seed,
        tokenizer=args.tokenizer,
        is_zh=args.is_zh, 
        for_cn=args.for_cn,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length)

    if not (args.do_train or args.do_val or args.do_test):
        raise ValueError("For args `do_train`, `do_val` and `do_test`, at "
                         "least one of them must be True.")

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    if args.do_train:
        train_data_generator = reader.data_generator(
            input_file=args.train_set,
            batch_size=args.batch_size,
            epoch=args.epoch,
            shuffle=True,
            phase="train")

        num_train_examples = reader.get_num_examples("train", args.train_set)
        if args.in_tokens:
            max_train_steps = args.epoch * num_train_examples // (
                args.batch_size // args.max_seq_len) // dev_count
        else:
            max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count

        warmup_steps = int(max_train_steps * args.warmup_proportion)
        print("Device count: %d, trainer_id: %d" % (dev_count, trainer_id))
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)
        print("Num warmup steps: %d" % warmup_steps)

        train_program = fluid.Program()

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, graph_vars, checkpoints, train_mems_vars = create_model(
                    args,
                    ernie_config=ernie_config, 
                    mem_len=memory_len)
                if args.use_recompute:
                    dist_strategy.recompute_checkpoints = checkpoints
                scheduled_lr, loss_scaling = optimization(
                    loss=graph_vars['loss'],
                    warmup_steps=warmup_steps,
                    num_train_steps=max_train_steps,
                    learning_rate=args.learning_rate,
                    train_program=train_program,
                    startup_prog=startup_prog,
                    weight_decay=args.weight_decay,
                    scheduler=args.lr_scheduler,
		    use_amp=args.use_amp,
                    init_loss_scaling=args.init_loss_scaling,
                    layer_decay_rate=args.layer_decay_ratio,
                    n_layers=n_layers,
                    dist_strategy=dist_strategy)
        
        origin_train_program = train_program
        if args.is_distributed:
            train_program = fleet.main_program
            origin_train_program = fleet._origin_program

        # add data pe
        # exec_strategy = fluid.ExecutionStrategy()
        # exec_strategy.num_threads = 1
        # exec_strategy.num_iteration_per_drop_scope = 10000
        # build_strategy = fluid.BuildStrategy()
        # train_program_dp = fluid.CompiledProgram(train_program).with_data_parallel(
        #         loss_name=graph_vars["loss"].name,
        #         exec_strategy=exec_strategy,
        #         build_strategy=build_strategy)
        
    if args.do_val or args.do_test:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, test_graph_vars, _, test_mems_vars = create_model(
                    args,
                    ernie_config=ernie_config,
                    mem_len=memory_len)
        test_prog = test_prog.clone(for_test=True)
    
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    init_model(args, exe, startup_prog)
    
    train_exe = exe if args.do_train else None
    steps = 0
    def init_memory():
        return [np.zeros([args.batch_size, memory_len, d_dim], dtype="float32")
                     for _ in range(n_layers)]

    if args.do_train:
        train_pyreader.set_batch_generator(train_data_generator) 
        train_pyreader.start()
        time_begin = time.time()
        tower_mems_np = init_memory()
        while True:
            try:
                steps += 1
                if steps % args.skip_steps == 0:
                    outputs = evaluate(train_exe, train_program, train_pyreader, graph_vars, 
                                        train_mems_vars, tower_mems_np, "train", steps, trainer_id, 
                                       dev_count, scheduled_lr, use_vars=args.use_vars)
                    tower_mems_np = outputs['tower_mems_np']

                    time_end = time.time()
                    used_time = time_end - time_begin
                    current_example, current_epoch = reader.get_train_progress()
                    print("train pyreader queue size: %d, " % train_pyreader.queue.size())
                    print("epoch: %d, worker_index: %d, progress: %d/%d, step: %d, ave loss: %f, "
                          "time cost: %f, speed: %f steps/s, learning_rate: %f" %
                          (current_epoch, trainer_id, current_example, num_train_examples,
                           steps, outputs["loss"], used_time, args.skip_steps / used_time, 
                           outputs["learning_rate"]))
                    time_begin = time.time()
                else:
                    if args.use_vars:
                        # train_exe.run(fetch_list=[])
                        train_exe.run(program=train_program, use_program_cache=True)
                    else:
                        outputs = evaluate(train_exe, train_program, train_pyreader, graph_vars, 
                                        train_mems_vars, tower_mems_np, "train", steps, trainer_id, 
                                           dev_count, scheduled_lr, use_vars=args.use_vars)
                        tower_mems_np = outputs['tower_mems_np']

                if steps % args.validation_steps == 0:
                    # evaluate dev set
                    if args.do_val:
                        test_pyreader.set_batch_generator(
                            reader.data_generator(
                                args.dev_set,
                                batch_size=args.batch_size,
                                epoch=1,
                                shuffle=False,
                                phase="eval"))
                        num_dev_examples = reader.get_num_examples("eval", args.dev_set)
                        print("the example number of dev file is {}".format(num_dev_examples))
                        evaluate(exe, test_prog, test_pyreader, test_graph_vars, test_mems_vars, 
                                 init_memory(), "eval", steps, trainer_id, dev_count, 
                                 use_vars=args.use_vars, examples=reader.get_examples("eval"),
                                 features=reader.get_features("eval"), args=args)
                    # evaluate test set
                    if args.do_test:
                        test_pyreader.set_batch_generator(
                            reader.data_generator(
                                args.test_set,
                                batch_size=args.batch_size,
                                epoch=1,
                                shuffle=False,
                                phase="test"))
                        num_test_examples = reader.get_num_examples("test", args.test_set)
                        print("the example number of test file is {}".format(num_test_examples))
                        evaluate(exe, test_prog, test_pyreader, test_graph_vars, test_mems_vars, 
                                 init_memory(), "test", steps, trainer_id, dev_count, 
                                 use_vars=args.use_vars, examples=reader.get_examples("test"),
                                features=reader.get_features("test"), args=args)
                # save model
                if trainer_id == 0 and steps % args.save_steps == 0:
                    save_path = os.path.join(args.checkpoints,
                                                "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, origin_train_program)

            except fluid.core.EOFException:
                if trainer_id == 0:
                    save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, origin_train_program)
                train_pyreader.reset()
                break

    # final eval on dev set
    if args.do_val:
        test_pyreader.set_batch_generator(
            reader.data_generator(
                args.dev_set,
                batch_size=args.batch_size,
                epoch=1,
                shuffle=False,
                phase="eval"))
        print("Final validation result:")
        num_dev_examples = reader.get_num_examples("eval", args.dev_set)
        print("the example number of dev file is {}".format(num_dev_examples))                
        evaluate(exe, test_prog, test_pyreader, test_graph_vars, 
                 test_mems_vars, init_memory(), "eval", steps, 
                 trainer_id, dev_count, use_vars=args.use_vars, 
                 examples=reader.get_examples("eval"), 
                 features=reader.get_features("eval"), args=args)

    # final eval on test set
    if args.do_test:
        test_pyreader.set_batch_generator(
            reader.data_generator(
                args.test_set,
                batch_size=args.batch_size,
                epoch=1,
                shuffle=False, 
                phase="test"))
        print("Final test result:")
        num_test_examples = reader.get_num_examples("test", args.test_set)
        print("the example number of test file is {}".format(num_test_examples)) 
        evaluate(exe, test_prog, test_pyreader, test_graph_vars, 
                 test_mems_vars, init_memory(), "test", steps, 
                 trainer_id, dev_count, use_vars=args.use_vars, 
                 examples=reader.get_examples("test"), 
                 features=reader.get_features("test"), args=args)

if __name__ == '__main__':
    print_arguments(args)
    main(args)
