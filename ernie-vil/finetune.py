#    Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" finetuning vison-language task """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import datetime
import argparse
import numpy as np
import multiprocessing
import json

from reader.vcr_finetuning import VCRDataJointReader
from model.ernie_vil import ErnieVilModel, ErnieVilConfig
from optim.optimization import optimization
from utils.args import print_arguments
from utils.init import init_checkpoint, init_pretraining_params
from args.finetune_args import parser

import paddle.fluid as fluid

args = parser.parse_args()

# yapf: enable.

#READERS = {"vcr": VCRDataJointReader, "vqa": VQADataReader, "refcoco+": RefcocoReader, "flickr": FlickrReader}
READERS = {"vcr": VCRDataJointReader}

def format_result(res_arr, qids, pred, labels, scores):
    """
        trans batch results into json format
    """
    for i in range(len(qids)):
        res="\t".join([str(qids[i]), str(pred[i]), str(labels[i]), " ".join(["%.5f" % s for s in scores[i]])])
        res_arr.append(res)
    return res_arr


def create_vcr_model(pyreader_name, ernie_config, task_group, is_prediction=False):
    """
        create model arc for vcr tasks
    """
    shapes = [[-1, args.max_seq_len, 1],    #src_id 
             [-1, args.max_seq_len, 1],    #pos_id
             [-1, args.max_seq_len, 1],    #sent_id
             [-1, args.max_seq_len, 1],    #task_id
             [-1, args.max_seq_len, 1],    #input_mask
             [-1, args.max_img_len, args.feature_size],  #image_embedding
             [-1, args.max_img_len, 5],     #image_loc
             [-1, args.max_img_len, 1],    #image_mask
             [-1, 1],     #labels
             [-1, 1],     #qids
             [],          #task_index
             [-1, 1],     #binary_labels
             ]
    dtypes = ['int64', 'int64', 'int64', 'int64', 'float32', 'float32', 'float32', 'float32', 
                       'int64', 'int64', 'int64', 'float32']
    lod_levels = [0] * len(dtypes)

    for _ in task_group:
        shapes.append([])
        dtypes.append('float')
        lod_levels.append(0)

    pyreader = fluid.layers.py_reader(
        capacity=30,
        shapes=shapes,
        dtypes=dtypes,
        lod_levels=lod_levels,
        name=pyreader_name,
        use_double_buffer=False)

    inputs = fluid.layers.read_file(pyreader)
    src_ids, pos_ids, sent_ids, task_ids, input_mask, image_embeddings, \
         image_loc, image_mask, labels, q_ids, task_index, binary_labels = inputs[: 12]

    ernie_vil = ErnieVilModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        task_ids=task_ids,
        input_mask=input_mask,
        image_embeddings=image_embeddings,
        image_loc=image_loc,
        input_image_mask=image_mask,
        config=ernie_config
        )

    h_cls, h_img = ernie_vil.get_pooled_output()
    task_conf = task_group[0]
    fusion_method = task_conf["fusion_method"]
    fusion_fea = ernie_vil.get_match_score(text=h_cls, image=h_img,         \
                                           dropout_rate=task_conf["dropout_rate"],
                                           mode=fusion_method)
    if is_prediction:
        num_choice = int(task_conf['num_choice'])
        task_name = task_conf.get('task_prefix', 'vcr')
        score = fluid.layers.fc(fusion_fea, 1,
                                param_attr = fluid.ParamAttr(name = task_name + "_fc.w_0",
                                                    initializer = fluid.initializer.TruncatedNormal(scale = 0.02)),
                                                    bias_attr = task_name + "_fc.b_0")
        score = fluid.layers.reshape(score, shape = [-1, num_choice])
        _loss, _softmax = fluid.layers.softmax_with_cross_entropy(logits = score,
                                                                  label = labels, return_softmax = True)
        _acc = fluid.layers.accuracy(input = _softmax, label = labels)
        pred = fluid.layers.argmax(score, axis = 1)
        mean_loss = fluid.layers.mean(_loss)
        task_vars = [mean_loss, _acc, pred, q_ids, labels, _softmax]
        for var in task_vars:
            var.persistable = True
        return pyreader, task_vars
    else:
        start_ind = 12
        mean_loss = fluid.layers.zeros(shape = [1], dtype = 'float32')
        mean_acc = fluid.layers.zeros(shape = [1], dtype = 'float32')
        for task_conf in task_group:
            task_weight = inputs[start_ind]
            start_ind += 1
            num_choice = int(task_conf['num_choice'])
            task_name = task_conf.get('task_prefix', 'vcr')
            score = fluid.layers.fc(fusion_fea, 1,
                                    param_attr = fluid.ParamAttr(name = task_name + "_fc.w_0",
                                    initializer = fluid.initializer.TruncatedNormal(scale = 0.02)),
                                    bias_attr = task_name + "_fc.b_0")

            _loss = fluid.layers.sigmoid_cross_entropy_with_logits(score,
                                                                    binary_labels, name = "cross_entropy_loss")
            tmp_score = fluid.layers.reshape(score, shape = [-1, num_choice])
            _softmax = fluid.layers.softmax(tmp_score)
            _acc = fluid.layers.accuracy(input = _softmax, label = labels)
            _mean_loss = fluid.layers.mean(_loss)
            mean_loss += _mean_loss * task_weight
            mean_acc += _acc * task_weight
        task_vars = [fluid.layers.reduce_mean(mean_loss), mean_acc]
        for var in task_vars:
            var.persistable = True

        return pyreader, task_vars


#MODELS = {"vcr": create_vcr_model, "vqa": create_vqa_model, "refcoco+": create_refcoco_model}
MODELS = {"vcr": create_vcr_model}

def predict_wrapper(args,
                    exe,
                    ernie_config,
                    task_group,
                    test_prog=None,
                    pyreader=None,
                    graph_vars=None):
    """Context to do validation.
    """
    reader_name = READERS[args.task_name]
    data_reader = reader_name(
        task_group,
        split=args.test_split,
        vocab_path=args.vocab_path,
        is_test=True,
        shuffle=False,
        batch_size=args.batch_size,
        epoch=args.epoch)
    if args.do_test:
        assert args.init_checkpoint is not None, "[FATAL] Please use --init_checkpoint '/path/to/checkpoints' \
                                                  to specify you pretrained model checkpoints"

        init_pretraining_params(exe, args.init_checkpoint, test_prog)
        print(("testing on %s %s split") % (args.task_name, args.test_split))

    def predict(exe=exe, pyreader=pyreader):
        """
            inference for downstream tasks
        """
        pyreader.decorate_tensor_provider(data_reader.data_generator())
        pyreader.start()

        cost = 0
        appear_step = 0
        task_acc = {}
        task_steps = {}
        steps = 0
        case_f1 = 0
        appear_f1 = 0
        time_begin = time.time()
        task_name_list = [v.name for v in graph_vars]
        fetch_list = task_name_list

        print('task name list : ', task_name_list)
        sum_acc = 0
        res_arr = []
        while True:
            try:
                outputs = exe.run(fetch_list=fetch_list, program=test_prog)
                each_acc = outputs[1][0]
                preds = np.reshape(outputs[2], [-1])
                qids = np.reshape(outputs[3], [-1])
                labels = np.reshape(outputs[4], [-1])
                scores = np.reshape(outputs[5], [-1, 4])
                sum_acc += each_acc
                steps += 1
                if steps % 10 == 0:
                    print('cur_step:', steps, 'cur_acc:', sum_acc / steps)
                format_result(res_arr, qids.tolist(), preds.tolist(), labels.tolist(), scores.tolist())
            except fluid.core.EOFException:
                pyreader.reset()
                break

        used_time = time.time() - time_begin

        with open(args.result_file, "w") as f:
            for r in res_arr:
                f.write(r + "\n")

        print("average_acc:", sum_acc / steps)
        ret = {}
        ret["acc"] = "acc: %f" % (sum_acc / steps)  
        for item in ret:
            try:
                ret[item] = ret[item].split(':')[-1]
            except:
                pass
        return ret
    return predict


def get_optimizer(total_loss, train_program, startup_prog, args):
    """
        optimization func
    """
    decay_steps_str=args.decay_steps
    if decay_steps_str == "":
        decay_steps = []
    else:
        decay_steps = [int(s) for s in decay_steps_str.split(";")]
    scheduled_lr = optimization(
         loss=total_loss,
         warmup_steps=args.warmup_steps,
         num_train_steps=args.num_train_steps,
         learning_rate=args.learning_rate,
         train_program=train_program,
         startup_prog=startup_prog,
         weight_decay=args.weight_decay,
         scheduler=args.lr_scheduler,
         decay_steps=decay_steps,
         lr_decay_ratio=args.lr_decay_ratio)
    return scheduled_lr


def main(args):
    """
       Main func for downstream tasks
    """
    print("finetuning tasks start")
    ernie_config = ErnieVilConfig(args.ernie_config_path)
    ernie_config.print_config()

    with open(args.task_group_json) as f:
        task_group = json.load(f)
        print('task: ', task_group)

    startup_prog = fluid.Program()
    if args.do_train and args.do_test:
        print("can not set both do_train and do_test as True")
        return 

    model_name = MODELS[args.task_name]
    if args.do_train:
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, model_outputs = model_name(
                    pyreader_name='train_reader', ernie_config=ernie_config, task_group=task_group)

                total_loss = model_outputs[0]
                scheduled_lr = get_optimizer(total_loss, train_program, startup_prog, args)
    if args.do_test:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, model_outputs  = model_name(
                    pyreader_name='test_reader', ernie_config=ernie_config, task_group=task_group, is_prediction=True)
                total_loss = model_outputs[0]

        test_prog = test_prog.clone(for_test=True)
    
    if args.use_gpu:
        gpu_id = 0
        if os.getenv("FLAGS_selected_gpus"):
            gpu_id = int(os.getenv("FLAGS_selected_gpus"))
    place = fluid.CUDAPlace(gpu_id) if args.use_gpu else fluid.CPUPlace()

    print("theoretical memory usage: ")
    if args.do_train:
        print(fluid.contrib.memory_usage(
             program=train_program, batch_size=args.batch_size))
    if args.do_test:
        print(fluid.contrib.memory_usage(
            program=test_prog, batch_size=args.batch_size))

    nccl2_num_trainers = 1
    nccl2_trainer_id = 0
    print("args.is_distributed:", args.is_distributed)
    trainer_id = 0
    if args.is_distributed:
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
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
            config.use_hierarchical_allreduce=args.use_hierarchical_allreduce
            config.hierarchical_allreduce_inter_nranks=args.hierarchical_allreduce_inter_nranks

            assert config.hierarchical_allreduce_inter_nranks > 1
            assert trainers_num % config.hierarchical_allreduce_inter_nranks == 0

            config.hierarchical_allreduce_exter_nranks = \
                trainers_num / config.hierarchical_allreduce_inter_nranks

        t = fluid.DistributeTranspiler(config=config)
        t.transpile(
            trainer_id,
            trainers=worker_endpoints_env,
            current_endpoint=current_endpoint,
            program=train_program,
            startup_program=startup_prog)

        nccl2_num_trainers = trainers_num
        nccl2_trainer_id = trainer_id

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if args.do_train:
        if args.init_checkpoint and args.init_checkpoint != "":
            sys.stderr.write('############################WARNING############################')
            sys.stderr.write('####### using init_pretraining_params, not init_checkpoint ####')
            sys.stderr.write('## meaning hyper param e.g. lr won\'t inherit from checkpoint##')
            sys.stderr.write('###############################################################')
            init_pretraining_params(exe, args.init_checkpoint, train_program)

        reader_name=READERS[args.task_name]
        data_reader = reader_name(
            task_group,
            split="train",
            vocab_path=args.vocab_path,
            batch_size=args.batch_size,
            epoch=args.epoch,)

    exec_strategy = fluid.ExecutionStrategy()
    if args.use_fast_executor:
        exec_strategy.use_experimental_executor = True
    exec_strategy.num_threads = 2
    
    exec_strategy.num_iteration_per_drop_scope = min(10, args.skip_steps)

    build_strategy = fluid.compiler.BuildStrategy()
    build_strategy.fuse_all_reduce_ops = False

    if args.use_fuse:
        build_strategy.fuse_all_reduce_ops = True

    if args.do_train:
        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            loss_name=total_loss.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy,
            main_program=train_program,
            num_trainers=nccl2_num_trainers,
            trainer_id=nccl2_trainer_id)

    if args.do_test: 
        predict = predict_wrapper(
            args,
            exe,
            ernie_config,
            task_group,
            test_prog=test_prog,
            pyreader=test_pyreader,
            graph_vars=model_outputs)
        result = predict()

    if args.do_train:
        train_pyreader.decorate_tensor_provider(data_reader.data_generator())
        train_pyreader.start()
        steps = 0
        time_begin = time.time()
        node_nums = 1 #int(os.getenv("PADDLE_NODES_NUM"))
        used_time_all = 0 
        while steps < args.num_train_steps:
            try:
                steps += node_nums
                skip_steps = args.skip_steps * node_nums
                fetch_list = []
                if nccl2_trainer_id == 0 and steps % skip_steps == 0:
                    task_name_list = [v.name for v in model_outputs]
                    fetch_list = task_name_list
                    fetch_list.append(scheduled_lr.name)
                
                time_begin = time.time()
                outputs = train_exe.run(fetch_list=fetch_list)
                if outputs:
                    print("feed_queue size", train_pyreader.queue.size())
                    progress_file = data_reader.get_progress()
                    epoch = progress_file["current_epoch"]
                    current_file_index = progress_file["current_file_index"]
                    total_file =  progress_file["total_file"]
                    current_file = progress_file["current_file"]
                    print(
                        "epoch: %d, progress: %d/%d, step: %d, loss: %f, "
                        "acc: %f"
                        % (epoch, current_file_index, total_file, steps,
                           outputs[0][0],
                           outputs[1][0]))
                    print("steps:", steps)
                    print("save_steps:", args.save_steps)

                    np_lr = outputs[-1:]

                    date_str = datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")

                    np_lr = float(np.mean(np_lr[0]))
                    print("%s current learning_rate:%.8f" % (date_str, np_lr))

                    if steps % args.save_steps == 0:
                        save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                        print("save_path:", save_path)
                        fluid.io.save_persistables(exe, save_path, train_program)
                    time_end = time.time()
                    used_time = time_end - time_begin
                    time_end = time_begin
                    print("used_time:", used_time)  
            except fluid.core.EOFException:
                train_pyreader.reset()
                break


if __name__ == '__main__':
    print_arguments(args)
    main(args)

