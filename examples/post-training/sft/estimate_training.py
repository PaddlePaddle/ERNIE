# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os

import numpy as np
from paddleformers.trainer.argparser import strtobool
from paddleformers.utils.log import logger

from ernie.dataset.finetuning import create_dataset
from ernie.tokenizer import Ernie4_5_Tokenizer
from ernie.utils.download_utils import check_download_repo


def parse_arguments():
    """Parse command line arguments using the argparse library and return a Namespace object containing parameter values.

    Args:
        None

    Returns:
        Namespace object containing the following parameters:
            --train_dataset_path:
                str type. Used to specify the path of training dataset.
                Default: examples/data/sft-train.jsonl.
            --train_dataset_type:
                str type. Used to specify the type of training dataset. Default: erniekit.
            --train_dataset_prob:
                str type. Used to specify the prob of training dataset. Default: 1.0.
            --model_name_or_path: str type. Used to specify the model directory or filename. Default: ./inference.
            --max_seq_len: int type. Used to specify the maximum input sequence length after tokenization. Default: 4096.
            --num_epochs: int type. Number of epochs to train for. No default value provided.
            --per_device_train_batch_size: int type. Batch size per device for training. Default: 1.
            --out_file: str type. Filename to save results. Default: estimate_training.json.
            --num_of_gpus: int type. Number of GPUs to use. Default: 8.
            --tensor_parallel_degree: int type. Tensor parallelism degree. Default: 8.
            --gradient_accumulation_steps:
                int type. (Not yet implemented, to be done)
                Number of steps to accumulate gradients before backward pass and parameter update.
                Default: 16.

    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dataset_path",
        default="examples/data/sft-train.jsonl",
        help="path of training datasets.",
    )
    parser.add_argument(
        "--train_dataset_prob",
        default="1.0",
        help="probabilities of training datasets.",
    )
    parser.add_argument(
        "--train_dataset_type",
        default="erniekit",
        help="type of training datasets.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="./inference",
        help="The directory of model.",
    )
    parser.add_argument(
        "--max_seq_len",
        default=4096,
        type=int,
        help="The maximum total input sequence length after tokenization",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=1,
        type=int,
        help="Batch size per GPU for training.",
    )
    parser.add_argument(
        "--out_file",
        default="estimate_training.json",
        help="The file to save results.",
    )
    parser.add_argument(
        "--num_of_gpus", type=int, default=8, help="The number of GPUs."
    )
    parser.add_argument(
        "--tensor_parallel_degree",
        type=int,
        default=8,
        help="The degree of tensor_parallel.",
    )
    parser.add_argument(
        "--pipeline_parallel_degree",
        type=int,
        default=1,
        help="The degree of pipeline.",
    )
    parser.add_argument(
        "--sharding_parallel_degree",
        type=int,
        default=1,
        help="The degree of sharding parallel.",
    )
    # TODO(gongenlei): support gradient_accumulation_steps args
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=0,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--from_hf_hub",
        type=bool,
        default=False,
        help="Whether to download model from huggingface hub",
    )
    parser.add_argument(
        "--from_aistudio",
        type=bool,
        default=False,
        help="Whether to download model from aistudio",
    )
    parser.add_argument(
        "--from_modelscope",
        type=bool,
        default=False,
        help="Whether to download model from modelscope",
    )

    # Data args, should be same with training.
    parser.add_argument("--seed", type=int, default=23, help="Random seed.")
    parser.add_argument(
        "--num_samples_each_epoch",
        type=int,
        default=6000000,
        help="Number of samples per epoch. Used for SFT.",
    )
    parser.add_argument(
        "--max_estimate_samples",
        type=int,
        default=1e5,
        help="Maximum number of samples used in estimation.",
    )
    parser.add_argument(
        "--greedy_intokens",
        type=strtobool,
        default="True",
        help="Whether to use greedy intokens.",
    )
    parser.add_argument(
        "--random_shuffle",
        type=strtobool,
        default="True",
        help="Whether to shuffle data.",
    )
    return parser.parse_args()


def estimate_training(args):
    """
    Estimate the number of steps required for training based on the training data.

    Args:
        This arguments for the function is defined in parse_arguments().

    Returns:
        dict: Returns a dictionary containing information about the number of training steps required. This function has similar functionality to the function of the same name in utils.py.

    """
    if len(args.train_dataset_path) > 1:
        logger.warning(
            "Suggest to use max_steps instead of num_train_epochs for multi source dataset."
        )
        logger.info(
            "Multi source dataset detected, number of samples will be estimated by following rule. "
            "num_samples = (source1_num_samples * prob1 + source2_num_samples * prob2 + ...) * epochs)"
        )

    # convert paddle model repo id
    args.model_name_or_path = check_download_repo(
        args.model_name_or_path,
        from_hf_hub=args.from_hf_hub,
        from_aistudio=args.from_aistudio,
        from_modelscope=args.from_modelscope,
    )

    if getattr(args, "from_modelscope", False):
        os.environ["from_modelscope"] = "True"

    tokenizer = Ernie4_5_Tokenizer.from_pretrained(
        args.model_name_or_path,
        from_hf_hub=args.from_hf_hub,
        from_aistudio=args.from_aistudio,
        convert_from_torch=False,
    )
    logger.info("Start to estimate max training steps...")
    dataset_config = {
        "tokenizer": tokenizer,
        "max_seq_len": args.max_seq_len,
        "random_seed": args.seed,
        "num_samples_each_epoch": args.num_samples_each_epoch,
        "random_shuffle": args.random_shuffle,
        "greedy_intokens": args.greedy_intokens,
    }
    train_dataset = create_dataset(
        task_group=args.train_dataset_path,
        task_group_prob=args.train_dataset_prob,
        sub_dataset_type=args.train_dataset_type,
        **dataset_config,
    )
    train_dataset.estimate = True

    max_samples = train_dataset.max_estimate_samples

    if args.max_estimate_samples != -1:
        # Set estimate samples to max_estimate_samples
        logger.warning(
            "The results between sampling and non-sampling methods may differ."
        )
        train_dataset.max_estimate_samples = min(
            args.max_estimate_samples, train_dataset.max_estimate_samples
        )

    if train_dataset.max_estimate_samples > 0:
        train_batches = 0
        train_tokens = 0
        for sequences in train_dataset:
            if not train_dataset.estimate:
                break
            train_batches += 1
            for sequence in sequences:
                train_tokens += len(sequence.token_ids)

        train_tokens *= args.num_train_epochs
        train_batches *= args.num_train_epochs
        if args.gradient_accumulation_steps > 0:
            grad_acc_steps = args.gradient_accumulation_steps
        else:
            grad_acc_steps = np.round(min(max(train_tokens / 1e5, 1), 16))

        data_parallel_degree = (
            args.num_of_gpus
            // args.tensor_parallel_degree
            // args.sharding_parallel_degree
            // args.pipeline_parallel_degree
        )
        global_batch_size = (
            args.per_device_train_batch_size
            * grad_acc_steps
            * data_parallel_degree
            * args.sharding_parallel_degree
        )
        max_steps = np.ceil(train_batches / global_batch_size)

        if max_samples != train_dataset.max_estimate_samples:
            max_steps *= max_samples / train_dataset.max_estimate_samples
            train_tokens *= max_samples / train_dataset.max_estimate_samples
            train_dataset.used_samples *= (
                max_samples / train_dataset.max_estimate_samples
            )
            train_dataset.unused_samples *= (
                max_samples / train_dataset.max_estimate_samples
            )

        res = {
            "num_train_epochs": int(args.num_train_epochs),
            "max_steps": int(np.ceil(max_steps)),
            "train_tokens": int(train_tokens),
            "global_batch_size": int(global_batch_size),
            "gradient_accumulation_steps": int(grad_acc_steps),
            "warmup_steps": int(np.ceil(0.1 * max_steps)),
            "num_of_gpus": int(args.num_of_gpus),
            "per_device_train_batch_size": int(args.per_device_train_batch_size),
            "tensor_parallel_degree": int(args.tensor_parallel_degree),
            "pipeline_parallel_degree": int(args.pipeline_parallel_degree),
            "sharding_parallel_degree": int(args.sharding_parallel_degree),
            "seed": args.seed,
            "num_samples_each_epoch": args.num_samples_each_epoch,
            "max_seq_len": int(args.max_seq_len),
            "valid": True,
            "train_samples": int(max_samples * args.num_train_epochs),
            "estimate_samples": int(train_dataset.max_estimate_samples),
            "actual_train_samples": int(
                train_dataset.used_samples * args.num_train_epochs
            ),
            "skip_samples": int(train_dataset.unused_samples * args.num_train_epochs),
        }
        if train_batches / args.num_train_epochs / global_batch_size < 1:
            logger.warning(
                "This dataset is too small, you'd better enlarge your dataset."
            )
            res["valid"] = False
    else:
        logger.error("No valid data found, please check your dataset format.")
        res = {
            "num_train_epochs": int(args.num_train_epochs),
            "max_steps": 0,
            "train_tokens": 0,
            "num_of_gpus": int(args.num_of_gpus),
            "per_device_train_batch_size": int(args.per_device_train_batch_size),
            "tensor_parallel_degree": int(args.tensor_parallel_degree),
            "pipeline_parallel_degree": int(args.pipeline_parallel_degree),
            "sharding_parallel_degree": int(args.sharding_parallel_degree),
            "seed": args.seed,
            "num_samples_each_epoch": args.num_samples_each_epoch,
            "max_seq_len": int(args.max_seq_len),
            "valid": False,
            "train_samples": 0,
        }
    out_file = getattr(args, "out_file", None)
    if out_file:
        with open(args.out_file, "w", encoding="utf-8") as f:
            json.dump(res, f)

    return res


if __name__ == "__main__":
    args = parse_arguments()
    enable_auth = False
    if enable_auth:
        from encryption.auth import auth_product

        product_name = auth_product(args.model_name_or_path)

    training_hp = estimate_training(args)
    print(training_hp)
