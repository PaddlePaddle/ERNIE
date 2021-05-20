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
"""args for seq2seq generation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
from utils.args import ArgumentGroup


class CustomAction(argparse.Action):
    """custom action"""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, " ".join(values))


# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from.")
model_g.add_arg("init_pretraining_params", str, None,
                "Init pre-training params which preforms fine-tuning from. If the "
                "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
model_g.add_arg("checkpoints", str, "checkpoints", "Path to save checkpoints.")
model_g.add_arg("weight_sharing", bool, True, "If set, share weights between word embedding and masked lm.")
model_g.add_arg("unimo_vocab_file", str, './model_files/dict/unimo_en.vocab.txt', "unimo vocab")
model_g.add_arg("encoder_json_file", str, './model_files/dict/unimo_en.encoder.json', 'bpt map')
model_g.add_arg("vocab_bpe_file", str, './model_files/dict/unimo_en.vocab.bpe', "vocab bpe")
model_g.add_arg("unimo_config_path", str, "./model_files/config/unimo_base_en.json",
                "The file to save unimo configuration.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 50, "Number of epoches for fine-tuning.")
train_g.add_arg("learning_rate", float, 4e-5, "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler", str, "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("weight_decay", float, 0.01, "Weight decay rate for L2 regularizer.")
train_g.add_arg("warmup_proportion", float, 0.02,
                "Proportion of training steps to perform linear learning rate warmup for.")
train_g.add_arg("save_steps", int, 100000, "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps", int, 100000, "The steps interval to evaluate model performance.")
train_g.add_arg("use_fuse", bool, False, "Whether to use fuse_allreduce_ops.")
train_g.add_arg("nccl_comm_num", int, 1, "NCCL comm num.")
train_g.add_arg("hierarchical_allreduce_inter_nranks", int, 8, "Hierarchical allreduce inter ranks.")
train_g.add_arg("use_hierarchical_allreduce", bool, False, "Use hierarchical allreduce or not.")
train_g.add_arg("use_fp16", bool, False, "Whether to use fp16 mixed precision training.")
train_g.add_arg("use_dynamic_loss_scaling", bool, False, "Whether to use dynamic loss scaling.")
train_g.add_arg("init_loss_scaling", float, 128.0,
                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")
train_g.add_arg("incr_every_n_steps", int, 100, "Increases loss scaling every n consecutive.")
train_g.add_arg("decr_every_n_nan_or_inf", int, 2,
                "Decreases loss scaling every n accumulated steps with nan or inf gradients.")
train_g.add_arg("incr_ratio", float, 2.0,
                "The multiplier to use when increasing the loss scaling.")
train_g.add_arg("decr_ratio", float, 0.8,
                "The less-than-one-multiplier to use when decreasing.")
train_g.add_arg("beta1", float, 0.9, "beta1 for adam")
train_g.add_arg("beta2", float, 0.98, "beta2 for adam.")
train_g.add_arg("epsilon", float, 1e-06, "epsilon for adam.")
train_g.add_arg("tgt_type_id", int, 1, "for seq2seq task.")
train_g.add_arg("do_decode", bool, False, "for seq2seq task.")
train_g.add_arg("label_smooth", float, 0.1, "label smooth")
train_g.add_arg("hidden_dropout_prob", float, 0.1, "hidden_dropout_prob")
train_g.add_arg("attention_probs_dropout_prob", float, 0.1, "attention_probs_dropout_prob")

log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("skip_steps", int, 100, "The steps interval to print loss.")
log_g.add_arg("verbose", bool, True, "Whether to output verbose log.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("task_type", str, "normal", "is task type")
data_g.add_arg("train_set", str, None, "Path to training data.")
data_g.add_arg("test_set", str, None, "Path to test data.")
data_g.add_arg("dev_set", str, None, "Path to validation data.")
data_g.add_arg("pred_set", str, None, "Path to pred data.")
data_g.add_arg("max_seq_len", int, 512, "Number of words of the longest seqence.")
data_g.add_arg("max_tgt_len", int, 512, "for seq2seq task.")
data_g.add_arg("max_src_len", int, 512, "for seq2seq task.")
data_g.add_arg("max_out_len", int, 512, "for seq2seq task.")
data_g.add_arg("min_out_len", int, 20, "for seq2seq task.")
data_g.add_arg("block_trigram", bool, True, "utilize trigram blocking during beam search")
data_g.add_arg("beam_size", int, 5, "for seq2seq task.")
data_g.add_arg("batch_size", int, 32, "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("pred_batch_size", int, 0, "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("in_tokens", bool, False,
               "If set, the batch size will be the maximum number of tokens in one batch. "
               "Otherwise, it will be the maximum number of examples in one batch.")
data_g.add_arg("do_lower_case", bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
data_g.add_arg("tokenized_input", bool, True, "input is tokenized")
data_g.add_arg("length_penalty", float, 0.6, "length_penalty")
data_g.add_arg("continuous_position", bool, False, "position is continuous")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
run_type_g.add_arg("visualdl_log", bool, False, "If set, use visualdl_log on paddlecloud.")
run_type_g.add_arg("is_distributed", bool, True, "If set, then start distributed training.")
run_type_g.add_arg("use_fast_executor", bool, True, "If set, use fast parallel executor (in experiment).")
run_type_g.add_arg("num_iteration_per_drop_scope", int, 1, "Iteration intervals to drop scope.")
run_type_g.add_arg("do_train", bool, True, "Whether to perform training.")
run_type_g.add_arg("do_val", bool, True, "Whether to perform evaluation on dev data set.")
run_type_g.add_arg("do_test", bool, True, "Whether to perform evaluation on test data set.")
run_type_g.add_arg("do_pred", bool, True, "Whether to perform evaluation on pred data set.")
run_type_g.add_arg("use_multi_gpu_test", bool, True, "Whether to perform evaluation using multiple gpu cards")
run_type_g.add_arg("save_and_valid_by_epoch", bool, False, "save_and_valid_by_epoch")
run_type_g.add_arg("eval_script", action=CustomAction, type=str, nargs='+', help="eval_script", default=None)
run_type_g.add_arg("eval_mertrics", str, "", "eval_mertrics")
run_type_g.add_arg("random_seed", int, 0, "Random seed.")

dialo_g = ArgumentGroup(parser, "dialogue", "for dialogue task.")
dialo_g.add_arg("role_type_size", int, 2, "role type size")
dialo_g.add_arg("turn_type_size", int, 16, "turn type size")
# yapf: enable
