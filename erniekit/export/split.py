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
from typing import Any, Optional

import paddle
from paddleformers.utils.log import logger

from ernie.modeling_moe import Ernie4_5_MoeForCausalLM
from ernie.modeling_moe_pp import Ernie4_5_MoeForCausalLMPipe

from ..hparams import get_export_args, read_args


def run_split(args: Optional[dict[str, Any]] = None) -> None:
    """_summary_

    Args:
        args (Optional[dict[str, Any]], optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
    """
    # read args
    args = read_args(args)
    model_args, data_args, generating_args, finetuning_args, export_args = get_export_args(args)

    paddle.set_device(finetuning_args.device)

    if finetuning_args.sequence_parallel:
        if finetuning_args.pipeline_parallel_degree > 1:
            assert (
                hasattr(finetuning_args, "pipeline_parallel_config")
                and "disable_partial_send_recv" in finetuning_args.pipeline_parallel_config
            ), "Should set '--pipeline_parallel_config disable_partial_send_recv' in bash script for pp with sp."
        if finetuning_args.tensor_parallel_degree <= 1:
            finetuning_args.sequence_parallel = False
            logger.info("Tensor_parallel_degree = 1. Set sequence_parallel to False.")

    # Set the dtype for loading model
    dtype = paddle.get_default_dtype()
    if finetuning_args.fp16_opt_level == "O2":
        if finetuning_args.fp16:
            dtype = "float16"
        if finetuning_args.bf16:
            dtype = "bfloat16"

    logger.info("Start to load model ...")
    model_class = Ernie4_5_MoeForCausalLM
    if finetuning_args.pipeline_parallel_degree > 1:
        model_class = Ernie4_5_MoeForCausalLMPipe
    if model_args.moe_group.lower() in {"data", "dp"} and finetuning_args.data_parallel_degree > 1:
        finetuning_args.use_expert_parallel = True

    if finetuning_args.weight_quantize_algo is not None:
        if finetuning_args.weight_quantize_algo == "weight_only_mix":
            weight_quantize_algo = {
                "weight_only_int4": [".*mlp.experts.*"],
                "weight_only_int8": [
                    ".*self_attn.qkv_proj.*",
                    ".*self_attn.o_proj.*",
                    ".*mlp.up_gate_proj.*",
                    ".*mlp.down_proj.*",
                ],
            }
        else:
            weight_quantize_algo = finetuning_args.weight_quantize_algo
        quantization_config = dict(
            weight_quantize_algo=weight_quantize_algo,
            ignore_modules=[".*out_linear.*"],
            apply_hadamard=finetuning_args.apply_hadamard,
            hadamard_block_size=finetuning_args.hadamard_block_size,
            quant_input_grad=finetuning_args.quant_input_grad,
            quant_weight_grad=finetuning_args.quant_weight_grad,
            apply_online_actscale_step=finetuning_args.apply_online_actscale_step,
            actscale_moving_rate=finetuning_args.actscale_moving_rate,
            fp8_format_type=finetuning_args.fp8_format_type,
        )
    else:
        quantization_config = dict(weight_quantize_algo=finetuning_args.weight_quantize_algo)

    from ernie.configuration import Ernie4_5_MoeConfig

    split_model_path = finetuning_args.output_dir
    model_config = Ernie4_5_MoeConfig.from_pretrained(
        split_model_path,
        dtype=dtype,
        quantization_config=quantization_config,
    )
    model_config.tensor_parallel_degree = finetuning_args.tensor_parallel_degree
    model_config.tensor_parallel_rank = finetuning_args.tensor_parallel_rank
    model_config.pipeline_parallel_degree = finetuning_args.pipeline_parallel_degree
    model_config.sharding_parallel_degree = finetuning_args.sharding_parallel_degree
    model_config.sharding = finetuning_args.sharding
    model_config.pipeline_parallel_config = finetuning_args.pipeline_parallel_config
    model_config.sequence_parallel = finetuning_args.sequence_parallel
    model_config.moe_group = "dummy"
    model = model_class.from_pretrained(
        split_model_path,
        config=model_config,
    )
    paddle.device.cuda.empty_cache()
    logger.info("Succeed to load model ...")

    logger.info("Begin to split model ...")
    model.save_pretrained(
        save_dir=os.path.join(split_model_path, 'split_export'),
        max_shard_size=f"{export_args.max_shard_size}GB",
        safe_serialization=True,
    )
    logger.info("Succeed to split model ...")

    # logger.info("Begin to upload model to hugging face...")
    # try:
    #     model.save_to_hf_hub(
    #         repo_id="paddle",
    #         private=True,
    #         subfolder="ernie",
    #     )
    #     logger.info("Succeed to upload model to hugging face...")
    # except MaxRetryError:
    #     logger.info(f'Please check your network. Error occurs while uploading model to huggingface.')
    # except Exception as e:
    #     logger.info(f"Error while uploading model to huggingface: {e}")
