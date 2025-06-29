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
import time
from typing import Any, Optional

import paddle
from paddleformers.mergekit import MergeConfig, MergeModel
from paddleformers.trainer import get_last_checkpoint
from paddleformers.utils.log import logger

from ..hparams import get_export_args, read_args


def logger_merge_config(merge_config, lora_merge):
    """
    Logs the merge configuration details to debug output, with different formatting
    for LoRA merges versus standard model merges.

    Args:
        merge_config (object): Configuration object containing merge parameters.
                              Expected to have attributes accessible via __dict__.
        lora_merge (bool): Flag indicating whether this is a LoRA merge operation.
                           When True, logs only LoRA-specific parameters.
                           When False, logs standard merge parameters.

    Outputs:
        Writes formatted configuration details to the logger at DEBUG level.
        For LoRA merges: Displays centered "LoRA Merge Info" header and specific paths.
        For standard merges: Displays centered "Mergekit Config Info" header and all
        parameters except excluded ones.
    """
    if lora_merge:
        logger.debug("{:^40}".format("LoRA Merge Info"))
        for k, v in merge_config.__dict__.items():
            if k in ["lora_model_path", "base_model_path"]:
                logger.debug(f"{k:30}: {v}")
    else:
        logger.debug("{:^40}".format("Mergekit Config Info"))
        for k, v in merge_config.__dict__.items():
            if k in ["model_path_str", "device", "tensor_type", "merge_preifx"]:
                continue
            logger.debug(f"{k:30}: {v}")


def run_export(args: Optional[dict[str, Any]] = None) -> None:
    """_summary_

    Args:
        args (Optional[dict[str, Any]], optional): _description_. Defaults to None.
    """

    args = read_args(args)
    model_args, data_args, generating_args, finetuning_args, export_args = get_export_args(args)

    paddle.set_device(finetuning_args.device)
    tensor_type = "np" if finetuning_args.device == "cpu" else "pd"

    last_checkpoint = None
    if os.path.isdir(finetuning_args.output_dir):
        last_checkpoint = get_last_checkpoint(finetuning_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected, beginning export {last_checkpoint}.")
        else:
            raise FileNotFoundError(f"No checkpoint detected in {finetuning_args.output_dir}.")

    if model_args.lora:
        start = time.time()
        logger.info("***** Start merging LoRA model *****")
        config = {}
        config["base_model_path"] = model_args.model_name_or_path
        config["lora_model_path"] = last_checkpoint
        config["output_path"] = os.path.join(finetuning_args.output_dir, 'export')
        if export_args.copy_tokenizer:
            config["copy_file_list"] = [
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "config.json",
            ]

        merge_config = MergeConfig(**config)
        mergekit = MergeModel(merge_config)
        logger_merge_config(merge_config, model_args.lora)
        mergekit.merge_model()
        logger.info(f"***** Successfully finished merging LoRA model. Time cost: {time.time()-start} s *****")
    else:
        raise ValueError("Only support merge lora checkpoint, but get model_args.lora is False.")
