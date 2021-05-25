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
"""visual entailment tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import multiprocessing
import numpy as np
import paddle.fluid as fluid
from utils.optimization import optimization
from utils.utils import get_time
from utils.init import init_pretraining_params, init_checkpoint
from utils.args import print_arguments
from model.tokenization import GptBpeTokenizer
from args.visual_entailment_args import parser
from collections import OrderedDict
from model.unimo_finetune import UNIMOConfig
from finetune.visual_entailment import create_model, evaluate
from reader.visual_entailment_reader import ClassifyReader


args = parser.parse_args()


def main(args):
    """main"""
    model_config = UNIMOConfig(args.unimo_config_path)
    model_config.print_config()

    gpu_id = 0
    gpus = fluid.core.get_cuda_device_count()
    if args.is_distributed and os.getenv("FLAGS_selected_gpus") is not None:
        gpu_list = os.getenv("FLAGS_selected_gpus").split(",")
        gpus = len(gpu_list)
        gpu_id = int(gpu_list[0])

    if args.use_cuda:
        place = fluid.CUDAPlace(gpu_id)
        dev_count = gpus
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    tokenizer = GptBpeTokenizer(vocab_file=args.unimo_vocab_file,
                                encoder_json_file=args.encoder_json_file,
                                vocab_bpe_file=args.vocab_bpe_file,
                                do_lower_case=args.do_lower_case)

    if not (args.do_train or args.do_val or args.do_test or args.do_test_hard):
        raise ValueError("For args `do_train`, `do_val`, `do_test`, `do_test_hard`, at "
                         "least one of them must be True.")

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    trainers_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))

    if args.do_train:
        train_data_reader = ClassifyReader(args.train_filelist, args.max_seq_len, tokenizer)
        train_data_generator = train_data_reader.data_generator(
            batch_size=args.batch_size,
            epoch=args.epoch,
            phase="train")

        if args.num_train_examples:
            num_train_examples = args.num_train_examples
        else:
            num_train_examples = train_data_reader.get_num_examples()
        step_num_per_epoch = num_train_examples // args.batch_size // trainers_num
        max_train_steps = args.epoch * step_num_per_epoch

        warmup_steps = int(max_train_steps * args.warmup_proportion)
        print("Device count: %d, gpu_id: %d" % (dev_count, gpu_id))
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)
        print("Num warmup steps: %d" % warmup_steps)

        train_program = fluid.Program()

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, graph_vars = create_model(
                    args,
                    config=model_config,
                    pyreader_name="train_reader",
                    is_train=True)

                scheduled_lr, loss_scaling = optimization(
                    loss=graph_vars["loss"],
                    warmup_steps=warmup_steps,
                    num_train_steps=max_train_steps,
                    learning_rate=args.learning_rate,
                    train_program=train_program,
                    weight_decay=args.weight_decay,
                    scheduler=args.lr_scheduler,
                    use_fp16=args.use_fp16,
                    use_dynamic_loss_scaling=args.use_dynamic_loss_scaling,
                    init_loss_scaling=args.init_loss_scaling,
                    beta1=args.beta1,
                    beta2=args.beta2,
                    epsilon=args.epsilon)

    if args.do_val or args.do_test or args.do_test_hard:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, test_graph_vars = create_model(
                    args,
                    config=model_config,
                    pyreader_name="dev_reader",
                    is_train=False)
        test_prog = test_prog.clone(for_test=True)
        if args.do_val:
            dev_data_reader = ClassifyReader(args.dev_filelist, args.max_seq_len, tokenizer)
            dev_data_generator = dev_data_reader.data_generator(
                batch_size=args.test_batch_size,
                epoch=1,
                phase="dev")

        if args.do_test:
            test_data_reader = ClassifyReader(args.test_filelist, args.max_seq_len, tokenizer)
            test_data_generator = test_data_reader.data_generator(
                batch_size=args.test_batch_size,
                epoch=1,
                phase="test")

        if args.do_test_hard:
            test_hard_data_reader = ClassifyReader(args.test_hard_filelist, args.max_seq_len, tokenizer)
            test_hard_data_generator = test_hard_data_reader.data_generator(
                batch_size=args.test_batch_size,
                epoch=1,
                phase="test_hard")

    nccl2_num_trainers = 1
    nccl2_trainer_id = 0
    print("args.is_distributed:", args.is_distributed)
    if args.is_distributed:
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        worker_endpoints_env = os.getenv("PADDLE_TRAINER_ENDPOINTS")
        current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
        worker_endpoints = worker_endpoints_env.split(",")
        trainers_num = len(worker_endpoints)

        print("worker_endpoints:{} trainers_num:{} current_endpoint:{} \
              trainer_id:{}".format(worker_endpoints, trainers_num,
                                    current_endpoint, trainer_id))

        # prepare nccl2 env.
        config = fluid.DistributeTranspilerConfig()
        config.mode = "nccl2"
        if args.nccl_comm_num > 1:
            config.nccl_comm_num = args.nccl_comm_num
        if args.use_hierarchical_allreduce and trainers_num > args.hierarchical_allreduce_inter_nranks:
            config.use_hierarchical_allreduce = args.use_hierarchical_allreduce
            config.hierarchical_allreduce_inter_nranks = args.hierarchical_allreduce_inter_nranks

            assert config.hierarchical_allreduce_inter_nranks > 1
            assert trainers_num % config.hierarchical_allreduce_inter_nranks == 0

            config.hierarchical_allreduce_exter_nranks = \
                trainers_num / config.hierarchical_allreduce_inter_nranks

        t = fluid.DistributeTranspiler(config=config)
        t.transpile(
            trainer_id,
            trainers=worker_endpoints_env,
            current_endpoint=current_endpoint,
            program=train_program if args.do_train else test_prog,
            startup_program=startup_prog)
        nccl2_num_trainers = trainers_num
        nccl2_trainer_id = trainer_id

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
                main_program=train_program)
        elif args.init_pretraining_params:
            init_pretraining_params(
                exe,
                args.init_pretraining_params,
                main_program=train_program)
    elif args.do_val or args.do_test or args.do_test_hard:
        args.init_checkpoint = args.init_pretraining_params
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing validation or testing!")
        init_checkpoint(
            exe,
            args.init_checkpoint,
            main_program=startup_prog)

    if args.do_train:
        exec_strategy = fluid.ExecutionStrategy()
        if args.use_fast_executor:
            exec_strategy.use_experimental_executor = True
        exec_strategy.num_threads = 4 if args.use_fp16 else 2
        exec_strategy.num_iteration_per_drop_scope = min(args.num_iteration_per_drop_scope, args.skip_steps)

        build_strategy = fluid.BuildStrategy()
        build_strategy.remove_unnecessary_lock = False

        if args.use_fuse:
            build_strategy.fuse_all_reduce_ops = True

        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            loss_name=graph_vars["loss"].name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy,
            main_program=train_program,
            num_trainers=nccl2_num_trainers,
            trainer_id=nccl2_trainer_id)
        train_pyreader.decorate_tensor_provider(train_data_generator)
    else:
        train_exe = None

    if args.do_val or args.do_test or args.do_test_hard:
        test_exe = fluid.ParallelExecutor(use_cuda=args.use_cuda,
                main_program=test_prog,
                share_vars_from=train_exe)

    dev_ret_history = [] # (steps, key_eval, eval)
    test_ret_history = [] # (steps, key_eval, eval)
    test_hard_ret_history = []  # (steps, key_eval, eval)
    steps = 0

    if args.do_train:
        train_pyreader.start()
        time_begin = time.time()
        skip_steps = args.skip_steps
        while True:
            try:
                steps += 1
                if steps % skip_steps == 0:
                    train_fetch_list = [graph_vars["loss"].name, scheduled_lr.name]
                    res = train_exe.run(fetch_list=train_fetch_list)
                    outputs = {"loss": np.mean(res[0]), 'learning_rate': float(res[1][0])}
                    if args.verbose:
                        verbose = "train pyreader queue size: %d, learning_rate: %.10f" % \
                                (train_pyreader.queue.size(), outputs['learning_rate'])
                        print(verbose)
                    current_epoch, current_example, current_file_index, total_file, current_file = \
                            train_data_reader.get_progress()

                    time_end = time.time()
                    used_time = time_end - time_begin
                    print("%s - epoch: %d, progress: %d/%d, %d/%d, step: %d, ave loss: %f, speed: %f steps/s" % \
                          (get_time(), current_epoch, current_example, num_train_examples, current_file_index, \
                          total_file, steps, outputs["loss"], args.skip_steps / used_time))
                    time_begin = time.time()
                else:
                    train_exe.run(fetch_list=[])

                if nccl2_trainer_id == 0:
                    if steps % args.save_steps == 0 and args.save_checkpoints:
                        save_path = os.path.join(args.checkpoints,
                                                 "step_" + str(steps))
                        fluid.io.save_persistables(exe, save_path, train_program)

                if steps % args.validation_steps == 0:
                    # evaluate dev set
                    if args.do_val:
                        test_pyreader.decorate_tensor_provider(dev_data_generator)
                        outputs = evaluate(args, test_exe, test_pyreader, test_graph_vars, \
                                "dev", trainers_num, nccl2_trainer_id)
                        if nccl2_trainer_id == 0:
                            dev_ret_history.append((steps, outputs['key_eval'], outputs[outputs['key_eval']]))

                    # evaluate test set
                    if args.do_test:
                        test_pyreader.decorate_tensor_provider(test_data_generator)
                        outputs = evaluate(args, test_exe, test_pyreader, test_graph_vars, \
                                "test", trainers_num, nccl2_trainer_id)
                        if nccl2_trainer_id == 0:
                            test_ret_history.append((steps, outputs['key_eval'], outputs[outputs['key_eval']]))

                    # evaluate test set
                    if args.do_test_hard:
                        test_pyreader.decorate_tensor_provider(test_hard_data_generator)
                        outputs = evaluate(args, test_exe, test_pyreader, test_graph_vars, \
                                "test_hard", trainers_num, nccl2_trainer_id)
                        if nccl2_trainer_id == 0:
                            test_hard_ret_history.append((steps, outputs['key_eval'], outputs[outputs['key_eval']]))

            except fluid.core.EOFException:
                if args.save_checkpoints:
                    save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break

    # final eval on dev set
    if args.do_val:
        test_pyreader.decorate_tensor_provider(dev_data_generator)
        outputs = evaluate(args, test_exe, test_pyreader, test_graph_vars, "dev", trainers_num, nccl2_trainer_id)
        if nccl2_trainer_id == 0:
            dev_ret_history.append((steps, outputs['key_eval'], outputs[outputs['key_eval']]))

    # final eval on test set
    if args.do_test:
        test_pyreader.decorate_tensor_provider(test_data_generator)
        outputs = evaluate(args, test_exe, test_pyreader, test_graph_vars, "test", trainers_num, nccl2_trainer_id)
        if nccl2_trainer_id == 0:
            test_ret_history.append((steps, outputs['key_eval'], outputs[outputs['key_eval']]))

    # final eval on test_hard set
    if args.do_test_hard:
        test_pyreader.decorate_tensor_provider(test_hard_data_generator)
        outputs = evaluate(args, test_exe, test_pyreader, test_graph_vars, "test_hard", trainers_num, nccl2_trainer_id)
        if nccl2_trainer_id == 0:
            test_hard_ret_history.append((steps, outputs['key_eval'], outputs[outputs['key_eval']]))

    if nccl2_trainer_id == 0:
        if args.do_val:
            dev_ret_history = sorted(dev_ret_history, key=lambda a: a[2], reverse=True)
            print("Best validation result: step %d %s %f" % \
                    (dev_ret_history[0][0], dev_ret_history[0][1], dev_ret_history[0][2]))


if __name__ == '__main__':
    print_arguments(args)
    main(args)
