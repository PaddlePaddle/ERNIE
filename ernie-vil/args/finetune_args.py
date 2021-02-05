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

""" args defination and default value """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse

from utils.args import ArgumentGroup, print_arguments

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("ernie_config_path", str, "./config/ernie_config.json", "json file path for ernie model config.")
model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from.")
model_g.add_arg("checkpoints", str, "checkpoints", "Path to save checkpoints.")
model_g.add_arg("task_name", str, "vcr", "Task to finetune on ERNIE-ViL")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 100, "Number of epoches for training.")
train_g.add_arg("learning_rate", float, 0.0001, "Learning rate used to train with warmup.")
train_g.add_arg("seq_dropout", float, 0.0, "dropout rate after the sequence output.")
train_g.add_arg("lr_scheduler", str, "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay', 'manual_warmup_decay'])
train_g.add_arg("layer_decay_rate", float, 0.0, "layer wise decay, 0.0 denote no layer decay")
train_g.add_arg("text_init_layers", int, 18, "diff from text and image layer, base:12-6=6, large:24-6=18")
train_g.add_arg("n_layers", int, 30, "max layers of text and image, base:12 + 6 , large:24 + 6")

train_g.add_arg("decay_steps", str, "", "learning rate decay steps, list with ;")
train_g.add_arg("lr_decay_ratio", float, 0.1, "learning rate decay ratio, used with manual_warmup_decay")
train_g.add_arg("weight_decay", float, 0.01, "Weight decay rate for L2 regularizer.")
train_g.add_arg("num_train_steps", int, 1000000, "Total steps to perform pretraining.")
train_g.add_arg("warmup_steps", int, 0, "Total steps to perform warmup when pretraining.")
train_g.add_arg("save_steps", int, 100, "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps", int, 6000, "The steps interval to evaluate model performance.")
train_g.add_arg("use_fuse", bool, False, "Whether to use fuse_allreduce_ops.")
train_g.add_arg("nccl_comm_num", int, 1, "NCCL comm num.")
train_g.add_arg("hierarchical_allreduce_inter_nranks", int, 8, "Hierarchical allreduce inter ranks.")
train_g.add_arg("use_hierarchical_allreduce", bool, False, "Use hierarchical allreduce or not.")
train_g.add_arg("use_gpu", bool, True, "Whether to gpu.")

log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("skip_steps", int, 10, "The steps interval to print loss.")
log_g.add_arg("verbose", bool, False, "Whether to output verbose log.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("result_file", str, "./res_tmp", "file to storage results")
data_g.add_arg("lr_decay_dict_file", str, "", "learning rate decay files.")
data_g.add_arg("train_filelist", str, "", "Path to training filelist.")
data_g.add_arg("valid_filelist", str, "", "Path to valid filelist.")
data_g.add_arg("test_filelist", str, "", "Path to test filelist.")
data_g.add_arg("vocab_path", str, "./config/vocab.txt", "Vocabulary path.")
data_g.add_arg("test_split", str, "val", "test of sub tasks, val or test")
data_g.add_arg("max_seq_len", int, 128, "Number of words of the longest seqence.")
data_g.add_arg("max_img_len", int, 100, "Number of image rois of the longest seqence.")
data_g.add_arg("feature_size", int, 2048, "Number of roi feature size of image.")
data_g.add_arg("fusion_method", str, "sum", "Number of roi feature size of image.")
data_g.add_arg("batch_size", int, 16, "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("task_group_json", str, "", "Path to task json")
data_g.add_arg("scale_circle", float, "1.0", "The scale factor in circle loss function, only use in circle loss mode")
data_g.add_arg("use_sigmoid", bool, False, "Whether to use sigmoid to match score, use for explode problem")
data_g.add_arg("margin", float, "0.2", "The margin value in triplet loss function")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("is_distributed", bool, False, "If set, then start distributed training.")
run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
run_type_g.add_arg("use_fast_executor", bool, False, "If set, use fast parallel executor (in experiment).")
run_type_g.add_arg("do_train", bool, False, "Whether to perform evaluation on test data set.")
run_type_g.add_arg("do_test", bool, False, "Whether to perform evaluation on test data set.")
run_type_g.add_arg("output_file", str, "", "The output file to save model output.")
# yapf: enable
