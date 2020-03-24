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

import os
import time
import argparse
import numpy as np
import logging
import multiprocessing

# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect.
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import paddle.fluid as fluid
from paddle.fluid.core import PaddleBuf
from paddle.fluid.core import PaddleDType
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

from reader.task_reader import ClassifyReader
from model.ernie import ErnieConfig
from finetune.classifier import create_model

from utils.args import print_arguments, check_cuda, prepare_logger, ArgumentGroup
from utils.init import init_pretraining_params
from finetune_args import parser

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "options to init, resume and save model.")
model_g.add_arg("ernie_config_path",            str,  None,  "Path to the json file for ernie model config.")
model_g.add_arg("init_checkpoint",              str,  None,  "Init checkpoint to resume training from.")
model_g.add_arg("save_inference_model_path",    str,  "inference_model",  "If set, save the inference model to this path.")
model_g.add_arg("use_fp16",                     bool, False, "Whether to resume parameters from fp16 checkpoint.")
model_g.add_arg("num_labels",                   int,  2,     "num labels for classify")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options.")
data_g.add_arg("predict_set",         str,  None,  "Predict set file")
data_g.add_arg("vocab_path",          str,  None,  "Vocabulary path.")
data_g.add_arg("label_map_config",    str,  None,  "Label_map_config json file.")
data_g.add_arg("max_seq_len",         int,  128,   "Number of words of the longest seqence.")
data_g.add_arg("batch_size",          int,  32,    "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("do_lower_case",       bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",          bool,   True,  "If set, use GPU for training.")
run_type_g.add_arg("do_prediction",     bool,   True,  "Whether to do prediction on test set.")

args = parser.parse_args()
log = logging.getLogger()
# yapf: enable.

def main(args):
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    reader = ClassifyReader(
        vocab_path=args.vocab_path,
        label_map_config=args.label_map_config,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        in_tokens=False,
        is_inference=True)

    predict_prog = fluid.Program()
    predict_startup = fluid.Program()
    with fluid.program_guard(predict_prog, predict_startup):
        with fluid.unique_name.guard():
            predict_pyreader, probs, feed_target_names = create_model(
                args,
                pyreader_name='predict_reader',
                ernie_config=ernie_config,
                is_classify=True,
                is_prediction=True)

    predict_prog = predict_prog.clone(for_test=True)

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    place = fluid.CUDAPlace(0) if args.use_cuda == True else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(predict_startup)

    if args.init_checkpoint:
        init_pretraining_params(exe, args.init_checkpoint, predict_prog)
    else:
        raise ValueError("args 'init_checkpoint' should be set for prediction!")

    assert args.save_inference_model_path, "args save_inference_model_path should be set for prediction"
    _, ckpt_dir = os.path.split(args.init_checkpoint.rstrip('/'))
    dir_name = ckpt_dir + '_inference_model'
    model_path = os.path.join(args.save_inference_model_path, dir_name)
    log.info("save inference model to %s" % model_path)
    fluid.io.save_inference_model(
        model_path,
        feed_target_names, [probs],
        exe,
        main_program=predict_prog)

    # Set config
    #config = AnalysisConfig(args.model_dir)
    #config = AnalysisConfig(os.path.join(model_path, "__model__"), os.path.join(model_path, ""))
    config = AnalysisConfig(model_path)
    if not args.use_cuda:
        log.info("disable gpu")
        config.disable_gpu()
        config.switch_ir_optim(True) 
    else:
        log.info("using gpu")
        config.enable_use_gpu(1024)

    # Create PaddlePredictor
    predictor = create_paddle_predictor(config)

    predict_data_generator = reader.data_generator(
        input_file=args.predict_set,
        batch_size=args.batch_size,
        epoch=1,
        shuffle=False)

    log.info("-------------- prediction results --------------")
    np.set_printoptions(precision=4, suppress=True)
    index = 0
    total_time = 0
    for sample in predict_data_generator():
        src_ids    = sample[0]
        sent_ids   = sample[1]
        pos_ids    = sample[2]
        task_ids   = sample[3]
        input_mask = sample[4]

        inputs = [array2tensor(ndarray) for ndarray in [src_ids, sent_ids, pos_ids, input_mask]]
        begin_time = time.time()
        outputs = predictor.run(inputs)
        end_time = time.time()
        total_time += end_time - begin_time

        # parse outputs
        output = outputs[0]
        batch_result  = output.as_ndarray()
        for single_example_probs in batch_result:
            print('\t'.join(map(str, single_example_probs.tolist())))
            index += 1
    log.info("qps:{}\ttotal_time:{}\ttotal_example:{}\tbatch_size:{}".format(index/total_time, total_time, index, args.batch_size))


def array2tensor(ndarray):
    """ convert numpy array to PaddleTensor"""
    assert isinstance(ndarray, np.ndarray), "input type must be np.ndarray"
    tensor = PaddleTensor(data=ndarray)
    return tensor

if __name__ == '__main__':
    prepare_logger(log)
    print_arguments(args)
    main(args)
