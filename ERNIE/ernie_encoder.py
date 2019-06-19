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
"""extract embeddings from ERNIE encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import multiprocessing

import paddle.fluid as fluid

import reader.task_reader as task_reader
from model.ernie import ErnieConfig, ErnieModel
from utils.args import ArgumentGroup, print_arguments
from utils.init import init_pretraining_params

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("ernie_config_path",         str,  None, "Path to the json file for ernie model config.")
model_g.add_arg("init_pretraining_params",   str,  None,
                "Init pre-training params which preforms fine-tuning from. If the "
                 "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
model_g.add_arg("output_dir",                str,  "embeddings", "path to save embeddings extracted by ernie_encoder.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("data_set",            str,  None,  "Path to data for calculating ernie_embeddings.")
data_g.add_arg("vocab_path",          str,  None,  "Vocabulary path.")
data_g.add_arg("max_seq_len",         int,  512,   "Number of words of the longest seqence.")
data_g.add_arg("batch_size",          int,  32,    "Total examples' number in batch for training.")
data_g.add_arg("do_lower_case",       bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",                     bool,   True,  "If set, use GPU for training.")
# yapf: enable


def create_model(args, pyreader_name, ernie_config):
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1], [-1, 1]],
        dtypes=['int64', 'int64', 'int64', 'float', 'int64'],
        lod_levels=[0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)

    (src_ids, sent_ids, pos_ids, input_mask,
     seq_lens) = fluid.layers.read_file(pyreader)

    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        config=ernie_config)

    enc_out = ernie.get_sequence_output()
    unpad_enc_out = fluid.layers.sequence_unpad(enc_out, length=seq_lens)
    cls_feats = ernie.get_pooled_output()

    # set persistable = True to avoid memory opimizing
    enc_out.persistable = True
    unpad_enc_out.persistable = True
    cls_feats.persistable = True

    graph_vars = {
        "cls_embeddings": cls_feats,
        "top_layer_embeddings": unpad_enc_out,
    }

    return pyreader, graph_vars


def main(args):
    args = parser.parse_args()
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    exe = fluid.Executor(place)

    reader = task_reader.ExtractEmbeddingReader(
        vocab_path=args.vocab_path,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case)

    startup_prog = fluid.Program()

    data_generator = reader.data_generator(
        input_file=args.data_set,
        batch_size=args.batch_size,
        epoch=1,
        shuffle=False)

    total_examples = reader.get_num_examples(args.data_set)

    print("Device count: %d" % dev_count)
    print("Total num examples: %d" % total_examples)

    infer_program = fluid.Program()

    with fluid.program_guard(infer_program, startup_prog):
        with fluid.unique_name.guard():
            pyreader, graph_vars = create_model(
                args, pyreader_name='reader', ernie_config=ernie_config)

            fluid.memory_optimize(input_program=infer_program)

    infer_program = infer_program.clone(for_test=True)

    exe.run(startup_prog)

    if args.init_pretraining_params:
        init_pretraining_params(
            exe, args.init_pretraining_params, main_program=startup_prog)
    else:
        raise ValueError(
            "WARNING: args 'init_pretraining_params' must be specified")

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = dev_count

    pyreader.decorate_tensor_provider(data_generator)
    pyreader.start()

    total_cls_emb = []
    total_top_layer_emb = []
    total_labels = []
    while True:
        try:
            cls_emb, unpad_top_layer_emb = exe.run(
                program=infer_program,
                fetch_list=[
                    graph_vars["cls_embeddings"].name, graph_vars[
                        "top_layer_embeddings"].name
                ],
                return_numpy=False)
            # batch_size * embedding_size
            total_cls_emb.append(np.array(cls_emb))
            total_top_layer_emb.append(np.array(unpad_top_layer_emb))
        except fluid.core.EOFException:
            break

    total_cls_emb = np.concatenate(total_cls_emb)
    total_top_layer_emb = np.concatenate(total_top_layer_emb)

    with open(os.path.join(args.output_dir, "cls_emb.npy"),
              "w") as cls_emb_file:
        np.save(cls_emb_file, total_cls_emb)
    with open(os.path.join(args.output_dir, "top_layer_emb.npy"),
              "w") as top_layer_emb_file:
        np.save(top_layer_emb_file, total_top_layer_emb)


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    main(args)
