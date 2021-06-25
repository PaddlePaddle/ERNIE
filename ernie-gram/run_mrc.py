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
"""Finetuning on classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import multiprocessing

# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect.
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import paddle.fluid as fluid

import paddle
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import reader.task_reader as task_reader
from model.ernie import ErnieConfig
from finetune.mrc import create_model, evaluate
from model.optimization import optimization
from utils.args import print_arguments
from utils.init import init_pretraining_params, init_checkpoint
from finetune.finetune_args import parser

args = parser.parse_args()

def create_strategy(args):
    """
    Create build strategy and exec strategy.
    Args:

    Returns:
        build_strategy: build strategy
        exec_strategy: exec strategy
    """
    build_strategy = paddle.static.BuildStrategy()
    exec_strategy = paddle.static.ExecutionStrategy()

    build_strategy.enable_addto = True if args.use_fp16 else False
    build_strategy.enable_sequential_execution = True

    if args.use_fast_executor:
        exec_strategy.use_experimental_executor = True
    exec_strategy.num_threads = 4 if args.use_fp16 else 2
    exec_strategy.num_iteration_per_drop_scope = max(1000, args.skip_steps)
    
    return build_strategy, exec_strategy

def create_distributed_strategy(args,
                                build_strategy=None,
                                exec_strategy=None):
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)
    trainer_id = fleet.worker_index()
    current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
    worker_endpoints = fleet.worker_endpoints()
    num_trainers = len(worker_endpoints)
    print("worker_endpoints:{} trainers_num:{} current_endpoint:{} trainer_id:{}"
          .format(worker_endpoints, num_trainers, current_endpoint, trainer_id))

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.execution_strategy = exec_strategy \
            if exec_strategy else paddle.static.ExecutionStrategy()
    dist_strategy.build_strategy = build_strategy \
            if build_strategy else paddle.static.ExecutionStrategy()

    dist_strategy.fuse_all_reduce_ops = True if args.use_fuse else False
    
    dist_strategy.nccl_comm_num = args.nccl_comm_num
    if args.nccl_comm_num > 1:
        dist_strategy.sync_nccl_allreduce=False
    
    if args.use_hierarchical_allreduce \
        and num_trainers > args.hierarchical_allreduce_inter_nranks:
            dist_strategy.use_hierarchical_allreduce = \
                    args.use_hierarchical_allreduce
            dist_strategy.hierarchical_allreduce_inter_nranks = \
                    args.hierarchical_allreduce_inter_nranks
    
    if args.use_fp16:
        print("use ammmmmmmmmmmmmmmmp")
        dist_strategy.amp = True
        #custom_black_list
        custom_white_list = ['softmax', 'layer_norm', 'gelu', 'relu']
        dist_strategy.amp_configs = {
                'custom_white_list': custom_white_list,
                'init_loss_scaling': args.init_loss_scaling
            }
    
    if args.use_recompute:
        dist_strategy.recompute = True
    
    return num_trainers, trainer_id, dist_strategy


def main(args):
    args.epoch = int(os.getenv("GRID_SEARCH_EPOCH"))
    args.learning_rate = float(os.getenv("GRID_SEARCH_LR"))
    args.random_seed = int(os.getenv("RANDSEED"))
    args.batch_size = int(os.getenv("GRID_SEARCH_BSZ"))
    print("Modified -> bsz: %d, epoch: %d, lr: %5f, randseed: %d"%
            (args.batch_size, args.epoch, args.learning_rate, args.random_seed))

    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()
    # Initialize the paddle execute enviroment
    paddle.enable_static()
    
    build_strategy, exec_strategy = create_strategy(args)
    
    node_nums = int(os.getenv("PADDLE_NODES_NUM"))
    
    trainers_num = 1
    trainer_id = 0
    #num_train_steps = args.num_train_steps
    #warmup_steps = args.warmup_steps
    trainers_num, trainer_id, dist_strategy = \
                create_distributed_strategy(args, build_strategy, exec_strategy)

    gpu_id = 0
    gpus = fluid.core.get_cuda_device_count()
    if args.is_distributed:
        gpus = os.getenv("FLAGS_selected_gpus").split(",")
        gpu_id = int(gpus[0])

    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    reader = task_reader.MRCReader(
        vocab_path=args.vocab_path,
        label_map_config=args.label_map_config,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        in_tokens=args.in_tokens,
        random_seed=args.random_seed,
        tokenizer=args.tokenizer,
        is_classify=args.is_classify,
        is_regression=args.is_regression,
        for_cn=args.for_cn,
        task_id=args.task_id,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        version_2_with_negative=args.version_2)

    if not (args.do_train or args.do_val or args.do_test):
        raise ValueError("For args `do_train`, `do_val` and `do_test`, at "
                         "least one of them must be True.")
    
    #if args.do_test:
    #    assert args.test_save is not None
    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    if args.predict_batch_size == None:
        args.predict_batch_size = args.batch_size
    if args.do_train:
        train_data_generator = reader.data_generator(
            input_file=args.train_set,
            batch_size=args.batch_size,
            epoch=args.epoch,
            dev_count=trainers_num,
            shuffle=True,
            phase="train")

        num_train_examples = reader.get_num_examples("train")

        if args.in_tokens:
            max_train_steps = args.epoch * num_train_examples // (
                args.batch_size // args.max_seq_len) // trainers_num
        else:
            max_train_steps = args.epoch * num_train_examples // args.batch_size // trainers_num

        warmup_steps = int(max_train_steps * args.warmup_proportion)
        print("Device count: %d, gpu_id: %d" % (trainers_num, gpu_id))
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)
        print("Num warmup steps: %d" % warmup_steps)

        train_program = fluid.Program()

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, graph_vars = create_model(
                    args,
                    pyreader_name='train_reader',
                    ernie_config=ernie_config,
                    is_training=True)
                if args.use_recompute:
                    dist_strategy.recompute_configs = {
                        "checkpoints": graph_vars["checkpoints"],
                        "enable_offload": False,
                        }

                scheduled_lr, loss_scaling = optimization(
                    loss=graph_vars["loss"],
                    warmup_steps=warmup_steps,
                    num_train_steps=max_train_steps,
                    learning_rate=args.learning_rate,
                    train_program=train_program,
                    startup_prog=startup_prog,
                    weight_decay=args.weight_decay,
                    scheduler=args.lr_scheduler,
                    dist_strategy=dist_strategy,
		            use_amp=args.use_fp16,
                    layer_decay_rate=args.layer_wise_decay_rate,
                    n_layers=ernie_config['num_hidden_layers'])

    if args.do_val or args.do_test:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, test_graph_vars = create_model(
                    args,
                    pyreader_name='test_reader',
                    ernie_config=ernie_config,
                    is_training=False)

        test_prog = test_prog.clone(for_test=True)

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if args.do_train:
        if args.init_checkpoint and args.init_pretraining_params:
            print(
                "WARNING: args 'init_checkpoint' and 'init_pretraining_params' "
                "both are set! Only arg 'init_checkpoint' is made valid.")
        if args.init_checkpoint:
            init_checkpoint(
                exe,
                args.init_checkpoint,
                main_program=startup_prog)
        elif args.init_pretraining_params:
            init_pretraining_params(
                exe,
                args.init_pretraining_params,
                main_program=startup_prog)
    elif args.do_val or args.do_test:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing validation or testing!")
        init_checkpoint(
            exe,
            args.init_checkpoint,
            main_program=startup_prog)

    if args.do_train:

        train_exe = exe
        train_pyreader.decorate_tensor_provider(train_data_generator)
    else:
        train_exe = None
    
    test_exe = exe
    test_dev_count = 1
    if args.do_val or args.do_test:
        if args.use_multi_gpu_test:
            test_dev_count = min(trainers_num, 8)

    if args.do_train:
        train_pyreader.start()
        steps = 0
        current_epoch = 0
        last_epoch = 0
        time_begin = time.time()
        while steps < max_train_steps:
            try:
                steps += 1
                if steps % args.skip_steps != 0:
                    train_exe.run(fetch_list=[], program=train_program)
                else:
                    outputs = evaluate(train_exe, train_program, train_pyreader,
                                       graph_vars, "train", version_2_with_negative=args.version_2)
                    current_epoch = steps * args.batch_size * trainers_num // num_train_examples
                    current_example = steps * args.batch_size * trainers_num % num_train_examples
                    time_end = time.time()
                    used_time = time_end - time_begin
                    print("epoch: %d, progress: %d/%d, step: %d, ave loss: %f, "
                          "speed: %f steps/s lr: %.6f" %
                          (current_epoch, current_example, num_train_examples,
                           steps, outputs["loss"], args.skip_steps / used_time, scheduled_lr.get_lr()))
                    time_begin = time.time()
                scheduled_lr.step()
                
                if trainer_id == 0:
                    if steps % args.save_steps == 0:
                        save_path = os.path.join(args.checkpoints,
                                                 "step_" + str(steps))
                        fluid.io.save_persistables(exe, save_path, train_program)
                current_epoch = steps * args.batch_size * trainers_num // num_train_examples
               
                if trainer_id < 8:
                    if last_epoch != current_epoch:
                        if args.do_val:
                            test_pyreader.decorate_tensor_provider(
                                reader.data_generator(
                                    args.dev_set,
                                    batch_size=args.batch_size,
                                    epoch=1,
                                    dev_count=test_dev_count,
                                    shuffle=False,
                                    phase="dev"))
                            evaluate(
                                exe,
                                test_prog,
                                test_pyreader,
                                test_graph_vars,
                                str(steps) + "_dev",
                                examples=reader.get_examples("dev"),
                                features=reader.get_features("dev"),
                                args=args, 
                                use_multi_gpu_test=args.use_multi_gpu_test,
                                gpu_id=gpu_id, 
                                dev_count=test_dev_count, 
                                tokenizer=reader.tokenizer,
                                version_2_with_negative=args.version_2)

                        if args.do_test:
                            test_pyreader.decorate_tensor_provider(
                                reader.data_generator(
                                    args.test_set,
                                    batch_size=args.batch_size,
                                    epoch=1,
                                    dev_count=test_dev_count,
                                    shuffle=False,
                                    phase="test"))
                            evaluate(
                                exe,
                                test_prog,
                                test_pyreader,
                                test_graph_vars,
                                str(steps) + "_test",
                                examples=reader.get_examples("test"),
                                features=reader.get_features("test"),
                                args=args,
                                use_multi_gpu_test=args.use_multi_gpu_test,
                                gpu_id=gpu_id, 
                                dev_count=test_dev_count,tokenizer=reader.tokenizer,
                                version_2_with_negative=args.version_2)
                if last_epoch != current_epoch:
                    last_epoch = current_epoch
                
            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break

        train_pyreader.reset()
    # final eval on dev set
    if args.do_val:
        print("Final validation result:")
        test_pyreader.decorate_tensor_provider(
            reader.data_generator(
                args.dev_set,
                batch_size=args.batch_size,
                epoch=1,
                dev_count=test_dev_count,
                shuffle=False,
                phase="dev"))
        evaluate(
            exe,
            test_prog,
            test_pyreader,
            test_graph_vars,
            "dev",
            examples=reader.get_examples("dev"),
            features=reader.get_features("dev"),
            args=args,
            use_multi_gpu_test=args.use_multi_gpu_test,
            gpu_id=gpu_id, 
            dev_count=test_dev_count,tokenizer=reader.tokenizer,
            version_2_with_negative=args.version_2)

    # final eval on test set
    if args.do_test:
        print("Final test result:")
        test_pyreader.decorate_tensor_provider(
            reader.data_generator(
                args.test_set,
                batch_size=args.batch_size,
                epoch=1,
                dev_count=test_dev_count,
                shuffle=False,
                phase="test"))
        evaluate(
            exe,
            test_prog,
            test_pyreader,
            test_graph_vars,
            "test",
            examples=reader.get_examples("test"),
            features=reader.get_features("test"),
            args=args,
            use_multi_gpu_test=args.use_multi_gpu_test,
            gpu_id=gpu_id, 
            dev_count=test_dev_count,tokenizer=reader.tokenizer,
            version_2_with_negative=args.version_2)

if __name__ == '__main__':
    print_arguments(args)
    main(args)
