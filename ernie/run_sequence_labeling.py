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
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import time
import six
import logging
import multiprocessing
from io import open

# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect.
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import paddle.fluid as fluid

import reader.task_reader as task_reader
from model.ernie import ErnieConfig
from optimization import optimization
from utils.init import init_pretraining_params, init_checkpoint
from utils.args import print_arguments, check_cuda, prepare_logger
from finetune.sequence_label import create_model, evaluate, predict, calculate_f1
from finetune_args import parser

args = parser.parse_args()
log = logging.getLogger()


def main(args):
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    if args.use_cuda:
        dev_list = fluid.cuda_places()
        place = dev_list[0]
        dev_count = len(dev_list)
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    reader = task_reader.SequenceLabelReader(
        vocab_path=args.vocab_path,
        label_map_config=args.label_map_config,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        in_tokens=args.in_tokens,
        random_seed=args.random_seed,
        task_id=args.task_id)

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

        num_train_examples = reader.get_num_examples(args.train_set)

        if args.in_tokens:
            if args.batch_size < args.max_seq_len:
                raise ValueError('if in_tokens=True, batch_size should greater than max_sqelen, got batch_size:%d seqlen:%d' % (args.batch_size, args.max_seq_len))

            max_train_steps = args.epoch * num_train_examples // (
                args.batch_size // args.max_seq_len) // dev_count
        else:
            max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count

        warmup_steps = int(max_train_steps * args.warmup_proportion)
        log.info("Device count: %d" % dev_count)
        log.info("Num train examples: %d" % num_train_examples)
        log.info("Max train steps: %d" % max_train_steps)
        log.info("Num warmup steps: %d" % warmup_steps)

        train_program = fluid.Program()

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, graph_vars = create_model(
                    args,
                    pyreader_name='train_reader',
                    ernie_config=ernie_config)
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
            log.info("Theoretical memory usage in training: %.3f - %.3f %s" %
                  (lower_mem, upper_mem, unit))

    if args.do_val or args.do_test:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, graph_vars = create_model(
                    args,
                    pyreader_name='test_reader',
                    ernie_config=ernie_config)

        test_prog = test_prog.clone(for_test=True)

    nccl2_num_trainers = 1
    nccl2_trainer_id = 0
    if args.is_distributed:
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        worker_endpoints_env = os.getenv("PADDLE_TRAINER_ENDPOINTS")
        current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
        worker_endpoints = worker_endpoints_env.split(",")
        trainers_num = len(worker_endpoints)
        
        log.info("worker_endpoints:{} trainers_num:{} current_endpoint:{} \
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

    if args.do_train:
        if args.init_checkpoint and args.init_pretraining_params:
            log.info(
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

        train_pyreader.set_batch_generator(train_data_generator)
    else:
        train_exe = None

    if args.do_val or args.do_test:
        test_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            main_program=test_prog,
            share_vars_from=train_exe)

    if args.do_train:
        train_pyreader.start()
        steps = 0
        graph_vars["learning_rate"] = scheduled_lr

        time_begin = time.time()
        while True:
            try:
                steps += 1
                if steps % args.skip_steps != 0:
                    train_exe.run(fetch_list=[])
                else:
                    fetch_list = [
                        graph_vars["num_infer"].name, graph_vars["num_label"].name,
                        graph_vars["num_correct"].name,
                        graph_vars["loss"].name,
                        graph_vars['learning_rate'].name,
                    ]
                    
                    out = train_exe.run(fetch_list=fetch_list)
                    num_infer, num_label, num_correct, np_loss, np_lr = out
                    lr = float(np_lr[0])
                    loss = np_loss.mean()
                    precision, recall, f1 = calculate_f1(num_label, num_infer, num_correct)
                    if args.verbose:
                        log.info("train pyreader queue size: %d, learning rate: %f" % (train_pyreader.queue.size(),
                                lr if warmup_steps > 0 else args.learning_rate))

                    current_example, current_epoch = reader.get_train_progress()
                    time_end = time.time()
                    used_time = time_end - time_begin
                    log.info("epoch: %d, progress: %d/%d, step: %d, loss: %f, "
                          "f1: %f, precision: %f, recall: %f, speed: %f steps/s"
                          % (current_epoch, current_example, num_train_examples,
                             steps, loss, f1, precision, recall,
                             args.skip_steps / used_time))
                    time_begin = time.time()

                if nccl2_trainer_id == 0 and steps % args.save_steps == 0:
                    save_path = os.path.join(args.checkpoints,
                                             "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, train_program)

                if nccl2_trainer_id == 0 and steps % args.validation_steps == 0:
                    # evaluate dev set
                    if args.do_val:
                        evaluate_wrapper(reader, exe, test_prog, test_pyreader, graph_vars,
                                current_epoch, steps)
                    # evaluate test set
                    if args.do_test:
                        predict_wrapper(reader, exe, test_prog, test_pyreader, graph_vars,
                                current_epoch, steps)


            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break

    # final eval on dev set
    if nccl2_trainer_id ==0 and args.do_val:
        evaluate_wrapper(reader, exe, test_prog, test_pyreader, graph_vars,
                current_epoch, 'final')

    if nccl2_trainer_id == 0 and args.do_test:
        predict_wrapper(reader, exe, test_prog, test_pyreader, graph_vars,
                current_epoch, 'final')


def evaluate_wrapper(reader, exe, test_prog, test_pyreader, graph_vars,
                     epoch, steps):
    # evaluate dev set
    batch_size = args.batch_size if args.predict_batch_size is None else args.predict_batch_size
    for ds in args.dev_set.split(','): #single card eval
        test_pyreader.set_batch_generator(
            reader.data_generator(
                ds,
                batch_size=batch_size,
                epoch=1,
                dev_count=1,
                shuffle=False))
        log.info("validation result of dataset {}:".format(ds))
        info = evaluate(exe, test_prog, test_pyreader, graph_vars,
                 args.num_labels)
        log.info(info + ', file: {}, epoch: {}, steps: {}'.format(
            ds, epoch, steps))


def predict_wrapper(reader, exe, test_prog, test_pyreader, graph_vars,
                    epoch, steps):
    test_sets = args.test_set.split(',')
    save_dirs = args.test_save.split(',')
    assert len(test_sets) == len(save_dirs), 'number of test_sets & test_save not match, got %d vs %d' % (len(test_sets), len(save_dirs))

    batch_size = args.batch_size if args.predict_batch_size is None else args.predict_batch_size
    for test_f, save_f in zip(test_sets, save_dirs):
        test_pyreader.set_batch_generator(reader.data_generator(
                    test_f,
                    batch_size=batch_size,
                    epoch=1,
                    dev_count=1,
                    shuffle=False))

        save_path = save_f + '.' + str(epoch) + '.' + str(steps)
        log.info("testing {}, save to {}".format(test_f, save_path))
        res = predict(exe, test_prog, test_pyreader, graph_vars, dev_count=1)
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        tokenizer = reader.tokenizer
        rev_label_map = {v: k for k, v in six.iteritems(reader.label_map)}
        with open(save_path, 'w', encoding='utf8') as f:
            for id, s, p in res:
                id = ' '.join(tokenizer.convert_ids_to_tokens(id))
                p = ' '.join(['%.5f' % pp[ss] for ss, pp in zip(s, p)])
                s = ' '.join([rev_label_map[ss]for ss in s])
                f.write('{}\t{}\t{}\n'.format(id, s, p))

if __name__ == '__main__':
    prepare_logger(log)
    print_arguments(args)
    check_cuda(args.use_cuda)
    main(args)
