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

import os
import time
import argparse

from utils.args import ArgumentGroup, print_arguments

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("ernie_config_path",      str,  "./config/ernie_config.json",  "Path to the json file for ernie model config.")
model_g.add_arg("init_checkpoint",        str,  None,                         "Init checkpoint to resume training from.")
model_g.add_arg("checkpoints",            str,  "checkpoints",                "Path to save checkpoints.")
model_g.add_arg("weight_sharing",         bool, True,                         "If set, share weights between word embedding and masked lm.")
model_g.add_arg("generate_neg_sample",    bool, False,                         "If set, randomly generate negtive samples by positive samples.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch",             int,    100,     "Number of epoches for training.")
train_g.add_arg("learning_rate",     float,  0.0001,  "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler",      str,    "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("weight_decay",      float,  0.01,    "Weight decay rate for L2 regularizer.")
train_g.add_arg("num_train_steps",   int,    1000000, "Total steps to perform pretraining.")
train_g.add_arg("warmup_steps",      int,    5000,    "Total steps to perform warmup when pretraining.")
train_g.add_arg("save_steps",        int,    10000,   "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps",  int,    1000,    "The steps interval to evaluate model performance.")
train_g.add_arg("use_fp16",          bool,   False,   "Whether to use fp16 mixed precision training.")
train_g.add_arg("loss_scaling",      float,  1.0,
                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")

log_g = ArgumentGroup(parser,     "logging", "logging related.")
log_g.add_arg("skip_steps",          int,    10,    "The steps interval to print loss.")
log_g.add_arg("verbose",             bool,   False, "Whether to output verbose log.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("train_filelist",           str,  "",  "Path to training filelist.")
data_g.add_arg("valid_filelist",           str,  "",  "Path to valid filelist.")
data_g.add_arg("test_filelist",            str,  "",  "Path to test filelist.")
data_g.add_arg("vocab_path",          str,  "./config/vocab.txt",  "Vocabulary path.")
data_g.add_arg("max_seq_len",         int,  512,                   "Number of words of the longest seqence.")
data_g.add_arg("batch_size",          int,  16,                    "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("in_tokens",           bool, False,
              "If set, the batch size will be the maximum number of tokens in one batch. "
              "Otherwise, it will be the maximum number of examples in one batch.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("is_distributed",    bool,   False,  "If set, then start distributed training.")
run_type_g.add_arg("use_cuda",          bool,   True,   "If set, use GPU for training.")
run_type_g.add_arg("use_fast_executor", bool,   False,  "If set, use fast parallel executor (in experiment).")
run_type_g.add_arg("do_test",           bool,   False,  "Whether to perform evaluation on test data set.")
# yapf: enable
