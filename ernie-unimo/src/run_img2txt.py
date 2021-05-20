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
"""Finetuning on image-to-text generation tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import multiprocessing
import paddle.fluid as fluid

from reader.img2txt_reader import Img2TxtReader
from model.tokenization import GptBpeTokenizer
from model.unimo_finetune import UNIMOConfig
from utils.optimization import optimization
from utils.init import init_model
from utils.args import print_arguments
from utils.utils import visualdl_log
from finetune.img2txt import Img2Txt
from args.img2txt_args import parser
from functools import partial
from collections import OrderedDict

args = parser.parse_args()


def get_time():
    """get time"""
    res = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    return res


def evaluate_datasets(pyreader, reader, eval_func, data_generator,
                      do_pred=False, suffix="out"):
    """evaluate"""
    def evaluate_dataset(phase, path):
        """run evaluation"""
        pyreader.set_batch_generator(data_generator(filelist=path, phase=phase))
        eval_func(eval_phase="%s_%s" % (phase, suffix))

    if args.do_val:
        evaluate_dataset("dev", args.valid_filelist)
    if args.do_test:
        evaluate_dataset("test", args.test_filelist)
    if args.do_pred and do_pred:
        evaluate_dataset("pred", args.test_filelist)


def save_checkpoint(program, exe, suffix):
    """save model checkpoint"""
    save_path = os.path.join(args.checkpoints, suffix)
    fluid.io.save_persistables(exe, save_path, program)


def main(args):
    """main func"""
    unimo_config = UNIMOConfig(args.unimo_config_path)
    if args.hidden_dropout_prob >= 0:
        unimo_config["hidden_dropout_prob"] = args.hidden_dropout_prob
    if args.attention_probs_dropout_prob >= 0:
        unimo_config["attention_probs_dropout_prob"] = args.attention_probs_dropout_prob
    unimo_config.print_config()

    if args.pred_batch_size <= 0:
        args.pred_batch_size = args.batch_size

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

    """load vocabulary"""
    tokenizer = GptBpeTokenizer(vocab_file=args.unimo_vocab_file,
                                encoder_json_file=args.encoder_json_file,
                                vocab_bpe_file=args.vocab_bpe_file,
                                do_lower_case=True)

    reader = Img2TxtReader(tokenizer, args)
    img2txt = Img2Txt(args, unimo_config, tokenizer)

    if not (args.do_train or args.do_val or args.do_test or args.do_pred):
        raise ValueError("For args `do_train`, `do_val` and `do_test`, at "
                         "least one of them must be True.")

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    if args.do_train:
        trainers_num = int(os.getenv("PADDLE_TRAINERS_NUM", 1))
        train_data_generator = reader.data_generator(
            filelist=args.train_filelist,
            batch_size=args.batch_size,
            epoch=args.epoch,
            dev_count=trainers_num,
            shuffle=True,
            phase="train")

        num_train_examples = reader.get_num_examples(args.train_filelist)  # 566747
        max_train_steps = args.epoch * num_train_examples // args.batch_size // trainers_num

        warmup_steps = int(max_train_steps * args.warmup_proportion)
        print("Device count: %d, gpu_id: %d" % (dev_count, gpu_id))
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)
        print("Num warmup steps: %d" % warmup_steps)

        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                print("using adv_type is ", args.adv_type)
                if args.adv_type == "freelb_text":
                    train_pyreader, graph_vars = img2txt.create_model_freelb_text()
                elif args.adv_type == "freelb_image":
                    train_pyreader, graph_vars = img2txt.create_model_freelb_image()
                elif args.adv_type == "villa":
                    train_pyreader, graph_vars = img2txt.create_model_villa()
                else:
                    print("Unsupported adv_type, run model without adversial training")
                    train_pyreader, graph_vars = img2txt.create_model()

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

    if args.do_val or args.do_test or args.do_pred:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, test_graph_vars = img2txt.create_model(decoding=args.do_decode)
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
    init_model(args, exe, train_program if args.do_train else test_prog)

    if args.do_train:
        exec_strategy = fluid.ExecutionStrategy()
        if args.use_fast_executor:
            exec_strategy.use_experimental_executor = True
        exec_strategy.num_threads = 4 if args.use_fp16 else 2  # 2 for fp32 4 for fp16
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
        train_pyreader.set_batch_generator(train_data_generator)
        train_resource = {"exe": train_exe,
                          "program": train_program,
                          "pyreader": train_pyreader}
        save_model = partial(save_checkpoint, program=train_program, exe=exe)

    test_dev_count = 1
    if args.do_val or args.do_test or args.do_pred:
        test_exe = exe
        if args.use_multi_gpu_test:
            test_dev_count = nccl2_num_trainers
        test_resource = {"exe": test_exe,
                         "program": test_prog,
                         "pyreader": test_pyreader}
        eval_data_generator = partial(reader.data_generator, batch_size=args.pred_batch_size,
                                      epoch=1, dev_count=test_dev_count, shuffle=False, do_decode=args.do_decode,
                                      place=place)
        eval_func = partial(img2txt.evaluate, resource=test_resource, graph_vars=test_graph_vars,
                            dev_count=test_dev_count, output_path=args.checkpoints, gpu_id=nccl2_trainer_id)
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
                    suffix = "step_" + str(steps)
                if steps % skip_steps == 0:
                    outputs = img2txt.evaluate(train_resource, "train", graph_vars)
                    if args.verbose:
                        verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size()
                        verbose += "learning rate: %.8f" % (
                            outputs["learning_rate"] if warmup_steps > 0 else args.learning_rate)
                        print(verbose)

                    current_epoch = steps * args.batch_size * trainers_num // num_train_examples
                    current_example = steps * args.batch_size * trainers_num % num_train_examples

                    time_end = time.time()
                    used_time = time_end - time_begin
                    print("%s - epoch: %d, progress: %d/%d, step: %d, loss: %f, "
                          "ppl: %f, speed: %f steps/s"
                          % (get_time(), current_epoch, current_example, num_train_examples,
                             steps, outputs["loss"], outputs["ppl"],
                             args.skip_steps / used_time))
                    time_begin = time.time()

                    if args.visualdl_log and nccl2_trainer_id == 0:
                        visuallog_dict = OrderedDict()
                        visuallog_dict["ppl"] = outputs["ppl"]
                        visualdl_log(visuallog_dict, outputs["ppl"], steps, phase='train')
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
                    current_epoch = steps * args.batch_size * trainers_num // num_train_examples
                    if current_epoch != last_epoch:
                        if nccl2_trainer_id == 0:
                            do_save = True
                        do_eval = True

                if do_save:
                    save_model(suffix=suffix)
                if do_eval:
                    if args.do_val or args.do_test or args.do_pred:
                        evaluate(suffix=suffix)

                if args.save_and_valid_by_epoch:
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
