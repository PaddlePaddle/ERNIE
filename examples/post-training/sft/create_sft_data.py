# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
from dataclasses import fields

import numpy as np
from paddleformers.data.indexed_dataset import SFTMMapIndexedDatasetBuilder
from paddleformers.trainer import PdArgumentParser, RuntimeTimer
from paddleformers.utils.log import logger
from sft_utils import (
    BuildDataArgument,
    BuildSFTTrainingArguments,
    DataGenerator,
)
from train import ModelArgument

from ernie.configuration import Ernie4_5_Config
from ernie.dataset.finetuning import Sequence, create_dataset
from ernie.tokenizer import Ernie4_5_Tokenizer
from ernie.utils.common_utils import estimate_training


def main():
    """
    Convert the dataset to the MapDataset format that can be used by the SFT training.
    """
    runtime_timer = RuntimeTimer("Creating SFT MapDataset")

    parser = PdArgumentParser((ModelArgument, BuildDataArgument, BuildSFTTrainingArguments))

    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file_and_cmd_lines()
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Avoid initializing tp, pp, and sdp to -1 in CPU environment.
    training_args.pipeline_parallel_degree = training_args.pp_degree
    training_args.tensor_parallel_degree = training_args.tp_degree
    training_args.sharding_parallel_degree = training_args.sdp_degree

    training_args.data_parallel_degree = (
        training_args.num_of_gpus
        // training_args.tensor_parallel_degree
        // training_args.sharding_parallel_degree
        // training_args.pipeline_parallel_degree
    )
    global_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.data_parallel_degree
        * training_args.sharding_parallel_degree
    )

    tokenizer = Ernie4_5_Tokenizer.from_pretrained(model_args.model_name_or_path)
    config = Ernie4_5_Config.from_pretrained(model_args.model_name_or_path)

    if tokenizer.vocab_size < 2**16 - 1:
        save_dtype = np.uint16
    else:
        save_dtype = np.int32

    dataset_config = {
        "tokenizer": tokenizer,
        "max_seq_len": data_args.max_seq_len,
        "random_seed": training_args.seed,
        "num_samples_each_epoch": data_args.num_samples_each_epoch,
        "random_shuffle": data_args.random_shuffle,
        "greedy_intokens": data_args.greedy_intokens,
    }
    dataclass = Sequence

    if training_args.do_train and data_args.train_dataset_path:
        runtime_timer.start("Create SFT Train MapDataset")
        os.makedirs(os.path.join(data_args.dataset_output_dir, 'train'), exist_ok=True)

        train_output_idx_files = os.path.join(data_args.dataset_output_dir, 'train', 'index.idx')
        train_dataset = create_dataset(
            task_group=data_args.train_dataset_path,
            task_group_prob=data_args.train_dataset_prob,
            sub_dataset_type=data_args.train_dataset_type,
            is_valid=False,
            **dataset_config,
        )
        if training_args.max_steps == -1:
            training_args.estimation_output_file = (
                'estimate_training.json'
                if training_args.estimation_output_file is None
                else training_args.estimation_output_file
            )
            training_args.max_steps = estimate_training(train_dataset, data_args, training_args, model_args)

        train_samples = training_args.max_steps * global_batch_size

        output_file_dict = {}
        train_dir = os.path.join(data_args.dataset_output_dir, 'train')
        for field in fields(dataclass):
            output_path = os.path.join(train_dir, f"{field.name}.bin")
            output_file_dict[field.name] = output_path

        train_builder = SFTMMapIndexedDatasetBuilder(output_file_dict, save_dtype)
        train_sample_generator = DataGenerator(train_dataset)

        used_samples = 0
        while used_samples < train_samples:
            train_sample = next(train_sample_generator)
            for sequence in train_sample:
                train_builder.add_item(sequence)

            train_builder.end_document()
            used_samples += 1
        train_builder.finalize(train_output_idx_files)
        logger.info(f"{runtime_timer.log()}")

    if training_args.do_eval and data_args.eval_task_config:
        runtime_timer.start("Create SFT Eval MapDataset")
        os.makedirs(os.path.join(data_args.dataset_output_dir, 'eval'), exist_ok=True)
        eval_output_idx_files = os.path.join(data_args.dataset_output_dir, 'eval', 'index.idx')
        eval_dataset = create_dataset(
            task_group=data_args.eval_dataset_path,
            task_group_prob=data_args.eval_dataset_prob,
            sub_dataset_type=data_args.eval_dataset_type,
            is_valid=True,
            **dataset_config,
        )
        output_file_dict = {}
        eval_dir = os.path.join(data_args.dataset_output_dir, 'eval')
        for field in fields(dataclass):
            output_path = os.path.join(eval_dir, f"{field.name}.bin")
            output_file_dict[field.name] = output_path
        eval_builder = SFTMMapIndexedDatasetBuilder(output_file_dict, save_dtype)

        for sequences in eval_dataset:
            for sequence in sequences:
                eval_builder.add_item(sequence)
            eval_builder.end_document()
        eval_builder.finalize(eval_output_idx_files)
        logger.info(f"{runtime_timer.log()}")


if __name__ == "__main__":
    main()
