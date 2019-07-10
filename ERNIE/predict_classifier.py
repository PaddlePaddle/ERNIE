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
"""Load classifier's checkpoint to do prediction or save inference model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import multiprocessing
import paddle.fluid as fluid

from reader.task_reader import ClassifyReader
from model.ernie import ErnieConfig
from finetune.classifier import create_model

from utils.args import ArgumentGroup, print_arguments
from utils.init import init_pretraining_params
from finetune_args import parser

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "options to init, resume and save model.")
model_g.add_arg("ernie_config_path",            str,  None,  "Path to the json file for bert model config.")
model_g.add_arg("init_checkpoint",              str,  None,  "Init checkpoint to resume training from.")
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
# yapf: enable.

def main(args):
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    reader = ClassifyReader(
        vocab_path=args.vocab_path,
        label_map_config=args.label_map_config,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        in_tokens=False)

    predict_prog = fluid.Program()
    predict_startup = fluid.Program()
    with fluid.program_guard(predict_prog, predict_startup):
        with fluid.unique_name.guard():
            predict_pyreader, probs, feed_target_names = create_model(
                args,
                pyreader_name='predict_reader',
                ernie_config=ernie_config,
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

    predict_exe = fluid.Executor(place)

    predict_data_generator = reader.data_generator(
        input_file=args.predict_set,
        batch_size=args.batch_size,
        epoch=1,
        shuffle=False)

    predict_pyreader.decorate_tensor_provider(predict_data_generator)

    predict_pyreader.start()
    all_results = []
    time_begin = time.time()
    while True:
        try:
            results = predict_exe.run(program=predict_prog, fetch_list=[probs.name])
            all_results.extend(results[0])
        except fluid.core.EOFException:
            predict_pyreader.reset()
            break
    time_end = time.time()

    np.set_printoptions(precision=4, suppress=True)
    print("-------------- prediction results --------------")
    for index, result in enumerate(all_results):
        print(str(index) + '\t{}'.format(result))


if __name__ == '__main__':
    print_arguments(args)
    main(args)
