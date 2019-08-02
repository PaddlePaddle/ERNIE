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

import reader.task_reader as task_reader
from model.ernie import ErnieConfig
from finetune.classifier import create_model, evaluate, predict
from optimization import optimization
from utils.args import print_arguments, check_cuda
from utils.init import init_pretraining_params, init_checkpoint
from utils.cards import get_cards
from finetune_args import parser

args = parser.parse_args()


def main(args):
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)

    reader = task_reader.ClassifyReader(
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
        task_id=args.task_id)

    if not (args.do_train or args.do_val or args.do_test):
        raise ValueError("For args `do_train`, `do_val` and `do_test`, at "
                         "least one of them must be True.")

    if args.do_test:
        assert args.test_save is not None
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
            dev_count=dev_count,
            shuffle=True,
            phase="train")

        num_train_examples = reader.get_num_examples(args.train_set)

        if args.in_tokens:
            max_train_steps = args.epoch * num_train_examples // (
                args.batch_size // args.max_seq_len) // dev_count
        else:
            max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count

        warmup_steps = int(max_train_steps * args.warmup_proportion)
        print("Device count: %d" % dev_count)
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)
        print("Num warmup steps: %d" % warmup_steps)

        train_program = fluid.Program()
        if args.random_seed is not None and args.enable_ce:
            train_program.random_seed = args.random_seed

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, graph_vars = create_model(
                    args,
                    pyreader_name='train_reader',
                    ernie_config=ernie_config,
                    is_classify=args.is_classify,
                    is_regression=args.is_regression)
                scheduled_lr, loss_scaling = optimization(
                    loss=graph_vars["loss"],
                    warmup_steps=warmup_steps,
                    num_train_steps=max_train_steps,
                    learning_rate=args.learning_rate,
                    train_program=train_program,
                    startup_prog=startup_prog,
                    weight_decay=args.weight_decay,
                    scheduler=args.lr_scheduler,
                    use_fp16=args.use_fp16)

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

    if args.do_val or args.do_test:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, graph_vars = create_model(
                    args,
                    pyreader_name='test_reader',
                    ernie_config=ernie_config,
                    is_classify=args.is_classify,
                    is_regression=args.is_regression)

        test_prog = test_prog.clone(for_test=True)
    nccl2_num_trainers = 1
    nccl2_trainer_id = 0
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
                main_program=startup_prog,
                use_fp16=args.use_fp16)
        elif args.init_pretraining_params:
            init_pretraining_params(
                exe,
                args.init_pretraining_params,
                main_program=startup_prog,
                use_fp16=args.use_fp16)
    elif args.do_val or args.do_test:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing validation or testing!")
        init_checkpoint(
            exe,
            args.init_checkpoint,
            main_program=startup_prog,
            use_fp16=args.use_fp16)

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

        train_pyreader.decorate_tensor_provider(train_data_generator)
    else:
        train_exe = None

    test_exe = exe
    if args.do_val or args.do_test:
        if args.use_multi_gpu_test:
            test_exe = fluid.ParallelExecutor(
                use_cuda=args.use_cuda,
                main_program=test_prog,
                share_vars_from=train_exe)

    if args.do_train:
        train_pyreader.start()
        steps = 0
        if warmup_steps > 0:
            graph_vars["learning_rate"] = scheduled_lr

        ce_info = []
        time_begin = time.time()
        last_epoch = 0
        current_epoch = 0
        while True:
            try:
                steps += 1
                if steps % args.skip_steps != 0:
                    train_exe.run(fetch_list=[])
                else:
                    outputs = evaluate(
                        train_exe,
                        train_program,
                        train_pyreader,
                        graph_vars,
                        "train",
                        metric=args.metric,
                        is_classify=args.is_classify,
                        is_regression=args.is_regression)

                    if args.verbose:
                        verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size(
                        )
                        verbose += "learning rate: %f" % (
                            outputs["learning_rate"]
                            if warmup_steps > 0 else args.learning_rate)
                        print(verbose)

                    current_example, current_epoch = reader.get_train_progress()
                    time_end = time.time()
                    used_time = time_end - time_begin

                    if args.is_classify:
                        print(
                            "epoch: %d, progress: %d/%d, step: %d, ave loss: %f, "
                            "ave acc: %f, speed: %f steps/s" %
                            (current_epoch, current_example, num_train_examples,
                             steps, outputs["loss"], outputs["accuracy"],
                             args.skip_steps / used_time))
                        ce_info.append(
                            [outputs["loss"], outputs["accuracy"], used_time])
                    if args.is_regression:
                        print(
                            "epoch: %d, progress: %d/%d, step: %d, ave loss: %f, "
                            " speed: %f steps/s" %
                            (current_epoch, current_example, num_train_examples,
                             steps, outputs["loss"],
                             args.skip_steps / used_time))
                    time_begin = time.time()

                if steps % args.save_steps == 0:
                    save_path = os.path.join(args.checkpoints,
                                             "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, train_program)

                if steps % args.validation_steps == 0 or last_epoch != current_epoch:
                    # evaluate dev set
                    if args.do_val:
                        evaluate_wrapper(args, reader, exe, test_prog,
                                         test_pyreader, graph_vars,
                                         current_epoch, steps)

                    if args.do_test:
                        predict_wrapper(args, reader, exe, test_prog,
                                        test_pyreader, graph_vars,
                                        current_epoch, steps)

                if last_epoch != current_epoch:
                    last_epoch = current_epoch

            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break
        if args.enable_ce:
            card_num = get_cards()
            ce_loss = 0
            ce_acc = 0
            ce_time = 0
            try:
                ce_loss = ce_info[-2][0]
                ce_acc = ce_info[-2][1]
                ce_time = ce_info[-2][2]
            except:
                print("ce info error")
            print("kpis\ttrain_duration_card%s\t%s" % (card_num, ce_time))
            print("kpis\ttrain_loss_card%s\t%f" % (card_num, ce_loss))
            print("kpis\ttrain_acc_card%s\t%f" % (card_num, ce_acc))

    # final eval on dev set
    if args.do_val:
        evaluate_wrapper(args, reader, exe, test_prog, test_pyreader,
                         graph_vars, current_epoch, steps)

    # final eval on test set
    if args.do_test:
        predict_wrapper(args, reader, exe, test_prog, test_pyreader, graph_vars,
                        current_epoch, steps)

    # final eval on dianostic, hack for glue-ax
    if args.diagnostic:
        test_pyreader.decorate_tensor_provider(
            reader.data_generator(
                args.diagnostic,
                batch_size=args.batch_size,
                epoch=1,
                dev_count=1,
                shuffle=False))

        print("Final diagnostic")
        qids, preds, probs = predict(
            test_exe,
            test_prog,
            test_pyreader,
            graph_vars,
            is_classify=args.is_classify,
            is_regression=args.is_regression)
        assert len(qids) == len(preds), '{} v.s. {}'.format(
            len(qids), len(preds))
        with open(args.diagnostic_save, 'w') as f:
            for id, s, p in zip(qids, preds, probs):
                f.write('{}\t{}\t{}\n'.format(id, s, p))

        print("Done final diagnostic, saving to {}".format(
            args.diagnostic_save))


def evaluate_wrapper(args, reader, exe, test_prog, test_pyreader, graph_vars,
                     epoch, steps):
    # evaluate dev set
    for ds in args.dev_set.split(','):
        test_pyreader.decorate_tensor_provider(
            reader.data_generator(
                ds,
                batch_size=args.predict_batch_size,
                epoch=1,
                dev_count=1,
                shuffle=False))
        print("validation result of dataset {}:".format(ds))
        evaluate_info = evaluate(
            exe,
            test_prog,
            test_pyreader,
            graph_vars,
            "dev",
            metric=args.metric,
            is_classify=args.is_classify,
            is_regression=args.is_regression)
        print(evaluate_info + ', file: {}, epoch: {}, steps: {}'.format(
            ds, epoch, steps))


def predict_wrapper(args, reader, exe, test_prog, test_pyreader, graph_vars,
                    epoch, steps):
    test_sets = args.test_set.split(',')
    save_dirs = args.test_save.split(',')
    assert len(test_sets) == len(save_dirs)

    for test_f, save_f in zip(test_sets, save_dirs):
        test_pyreader.decorate_tensor_provider(
            reader.data_generator(
                test_f,
                batch_size=args.predict_batch_size,
                epoch=1,
                dev_count=1,
                shuffle=False))

        save_path = save_f + '.' + str(epoch) + '.' + str(steps)
        print("testing {}, save to {}".format(test_f, save_path))
        qids, preds, probs = predict(
            exe,
            test_prog,
            test_pyreader,
            graph_vars,
            is_classify=args.is_classify,
            is_regression=args.is_regression)

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(save_path, 'w') as f:
            for id, s, p in zip(qids, preds, probs):
                f.write('{}\t{}\t{}\n'.format(id, s, p))


if __name__ == '__main__':
    print_arguments(args)
    check_cuda(args.use_cuda)
    main(args)
