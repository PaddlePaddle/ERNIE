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
"""Finetuning on classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import multiprocessing

import paddle.fluid as fluid

from reader.seq2seq_reader import Seq2SeqReader
from model.ernie import ErnieConfig
from model.optimization import optimization
from utils.init import init_model
from utils.args import print_arguments
from finetune.seq2seq import ErnieGenFinetune
from utils.finetune_args import parser
from functools import partial

args = parser.parse_args()


def evaluate_datasets(pyreader, reader, eval_func, data_generator, do_pred=False, suffix="out"):
    def evaluate_dataset(phase, path):
        pyreader.set_batch_generator(data_generator(input_file=path, phase=phase))
        eval_func(eval_phase="%s_%s" % (phase, suffix), features=reader.get_features(phase))

    if args.do_val:
        evaluate_dataset("dev", args.dev_set)
    if args.do_test:
        evaluate_dataset("test", args.test_set)
    if args.do_pred and do_pred:
        evaluate_dataset("pred", args.pred_set)


def save_checkpoint(program, exe, suffix):
    save_path = os.path.join(args.checkpoints, suffix)
    fluid.io.save_persistables(exe, save_path, program)


def main(args):
    ernie_config = ErnieConfig(args.ernie_config_path)
    if args.task_type == "dialog":
        ernie_config["role_type_size"] = args.role_type_size
        ernie_config["turn_type_size"] = args.turn_type_size
    if args.hidden_dropout_prob >= 0:
        ernie_config["hidden_dropout_prob"] = args.hidden_dropout_prob
    if args.attention_probs_dropout_prob >= 0:
        ernie_config["attention_probs_dropout_prob"] = args.attention_probs_dropout_prob
    ernie_config.print_config()

    if args.pred_batch_size <= 0:
        args.pred_batch_size = args.batch_size

    gpu_id = 0 
    gpus = fluid.core.get_cuda_device_count()
    if args.is_distributed:
        gpus = os.getenv("FLAGS_selected_gpus").split(",")
        gpu_id = int(gpus[0])
    
    if args.use_cuda:
        place = fluid.CUDAPlace(gpu_id)
        dev_count = len(gpus) if args.is_distributed else gpus
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    reader = Seq2SeqReader(args)
    ernie_gen = ErnieGenFinetune(args, ernie_config, reader.tokenizer)

    if not (args.do_train or args.do_val or args.do_test or args.do_pred):
        raise ValueError("For args `do_train`, `do_val` and `do_test`, at "
                         "least one of them must be True.")

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    if args.do_train:
        trainers_num = int(os.getenv("PADDLE_TRAINERS_NUM"))
        train_data_generator = reader.data_generator(
            input_file=args.train_set,
            batch_size=args.batch_size,
            epoch=args.epoch,
            dev_count=trainers_num,
            shuffle=True,
            phase="train")

        num_train_examples = reader.get_num_examples(args.train_set)

        if args.in_tokens:
            max_train_steps = args.epoch * num_train_examples // (
                args.batch_size // args.max_seq_len) // trainers_num
        else:
            max_train_steps = args.epoch * num_train_examples // args.batch_size // trainers_num

        warmup_steps = int(max_train_steps * args.warmup_proportion)
        print("Device count: %d, gpu_id: %d" % (dev_count, gpu_id))
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)
        print("Num warmup steps: %d" % warmup_steps)

        train_program = fluid.Program()

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, graph_vars = ernie_gen.create_model()
                scheduled_lr, loss_scaling = optimization(
                    loss=graph_vars["loss"],
                    warmup_steps=warmup_steps,
                    num_train_steps=max_train_steps,
                    learning_rate=args.learning_rate,
                    train_program=train_program,
                    startup_prog=startup_prog,
                    weight_decay=args.weight_decay,
                    scheduler=args.lr_scheduler,
                    use_fp16=args.use_fp16,
                    use_dynamic_loss_scaling=args.use_dynamic_loss_scaling,
                    init_loss_scaling=args.init_loss_scaling,
                    incr_every_n_steps=args.incr_every_n_steps,
                    decr_every_n_nan_or_inf=args.decr_every_n_nan_or_inf,
                    incr_ratio=args.incr_ratio,
                    decr_ratio=args.decr_ratio)

        if args.verbose:
            if args.in_tokens:
                lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                    program=train_program,
                    batch_size=args.batch_size // args.max_seq_len)
            else:
                lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                    program=train_program, batch_size=args.batch_size)
            print("Theoretical memory usage in training: %.3f - %.3f %s" %
                  (lower_mem, upper_mem, unit))

    if args.do_val or args.do_test or args.do_pred:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, test_graph_vars = ernie_gen.create_model(decoding=args.do_decode)
        test_prog = test_prog.clone(for_test=True)
    
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
    init_model(args, exe, startup_prog)

    if args.do_train:
        exec_strategy = fluid.ExecutionStrategy()
        if args.use_fast_executor:
            exec_strategy.use_experimental_executor = True
        exec_strategy.num_threads = dev_count
        exec_strategy.num_iteration_per_drop_scope = args.num_iteration_per_drop_scope
        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            loss_name=graph_vars["loss"].name,
            exec_strategy=exec_strategy,
            main_program=train_program,
            num_trainers=nccl2_num_trainers,
            trainer_id=nccl2_trainer_id)
        train_pyreader.set_batch_generator(train_data_generator)
        train_resource = {"exe": train_exe,
                          "program": train_program,
                          "pyreader": train_pyreader}
        save_model = partial(save_checkpoint, program=train_program, exe=exe)

    test_dev_count = 1
    if args.do_val or args.do_test or args.do_pred:
        test_exe = exe
        if args.use_multi_gpu_test:
            test_dev_count = min(trainers_num, int(os.getenv("PADDLE_PROC_PER_NODE", "1")))
        test_resource = {"exe": test_exe,
                         "program": test_prog,
                         "pyreader": test_pyreader}
        eval_data_generator = partial(reader.data_generator, batch_size=args.pred_batch_size, 
            epoch=1, dev_count=test_dev_count, shuffle=False, do_decode=args.do_decode, place=place)
        eval_func = partial(ernie_gen.evaluate, resource=test_resource, graph_vars=test_graph_vars,
            dev_count=test_dev_count, output_path=args.checkpoints, gpu_id=trainer_id)
        evaluate = partial(evaluate_datasets, pyreader=test_pyreader, reader=reader,
            eval_func=eval_func, data_generator=eval_data_generator)

    if args.do_train:
        train_pyreader.start()
        steps = 0
        last_epoch = 0
        if warmup_steps > 0:
            graph_vars["learning_rate"] = scheduled_lr

        time_begin = time.time()

        skip_steps = args.skip_steps
        while True:
            try:
                steps += 1
                if args.save_and_valid_by_epoch:
                    suffix = "epoch_" + str(last_epoch)
                else:
                    suffix ="step_" + str(steps)
                if steps % skip_steps == 0:
                    outputs = ernie_gen.evaluate(train_resource, "train", graph_vars)
                    if args.verbose:
                        verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size()
                        verbose += "learning rate: %f" % (
                            outputs["learning_rate"] if warmup_steps > 0 else args.learning_rate)
                        print(verbose)

                    if args.in_tokens:
                        current_example, current_epoch = reader.get_train_progress()
                    else:
                        current_epoch = steps * args.batch_size * trainers_num // num_train_examples
                        current_example = steps * args.batch_size * trainers_num % num_train_examples

                    time_end = time.time()
                    used_time = time_end - time_begin
                    print("epoch: %d, progress: %d/%d, step: %d, loss: %f, "
                          "ppl: %f, speed: %f steps/s"
                          % (current_epoch, current_example, num_train_examples,
                             steps, outputs["loss"], outputs["ppl"],
                             args.skip_steps / used_time))
                    time_begin = time.time()
                else:
                    train_exe.run(fetch_list=[])

                if nccl2_trainer_id >= test_dev_count:
                    continue
 
                do_save = False
                do_eval = False
                if not args.save_and_valid_by_epoch:
                    if steps % args.save_steps == 0 and nccl2_trainer_id == 0:
                        do_save = True
                    if steps % args.validation_steps == 0:
                        do_eval = True
                else:
                    if args.in_tokens:
                        current_example, current_epoch = reader.get_train_progress()    
                    else:
                        current_epoch = steps * args.batch_size * trainers_num // num_train_examples
                    if current_epoch != last_epoch:
                        if nccl2_trainer_id == 0:
                            do_save =True
                        do_eval = True

                if do_save:
                    save_model(suffix=suffix)
                if do_eval:
                    evaluate(suffix=suffix)

                last_epoch = current_epoch

            except fluid.core.EOFException:
                save_model(suffix=suffix)
                train_pyreader.reset()
                break

    if nccl2_trainer_id >= test_dev_count:
        return

    if args.do_val or args.do_test or args.do_pred:
        suffix = "output"
        if args.do_train:
            if not args.save_and_valid_by_epoch:
                suffix = "step_" + str(steps)
            else:
                suffix = "epoch_" + str(last_epoch)

        evaluate(suffix=suffix, do_pred=True)

if __name__ == '__main__':
    print_arguments(args)
    main(args)
