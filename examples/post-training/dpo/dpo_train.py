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

""" Training DPO """
import gc
import importlib.util
import os
import sys
import time
import json
from functools import partial

if importlib.util.find_spec("triton") is not None:
    try:
        import use_triton_in_paddle

        use_triton_in_paddle.make_triton_compatible_with_paddle()
    except Exception as _:
        raise RuntimeError(
            "Triton is installed, but not yet compatible with Paddle. "
            "Please run 'python -m pip install use-triton-in-paddle' to enable Triton support in Paddle."
        )

import paddle
from paddleformers.peft import LoRAConfig, LoRAModel
from paddleformers.trainer import (
    IntervalStrategy,
    PdArgumentParser,
    get_last_checkpoint,
    set_seed,
)
from paddleformers.trainer.trainer_utils import ShardingOption
from paddleformers.utils.log import logger

from ernie.callbacks import LayerwiseDropoutCallback
from ernie.configuration import Ernie4_5_MoeConfig
from ernie.dataset.dpo import collate_fn, create_dataset
from ernie.modeling_moe import Ernie4_5_MoeForCausalLM
from ernie.modeling_moe_pp import Ernie4_5_MoeForCausalLMPipe
from ernie.tokenizer import Ernie4_5_Tokenizer
from ernie.utils.common_utils import check_refined_recompute

# isort: off
from dpo_estimate_training import dpo_estimate_training
from dpo_trainer import ErnieMoEDPOTrainer
from dpo_utils import (
    DataArgument,
    DPOConfig,
    DPOTrainingArguments,
    ModelArgument,
    calculate_effective_tokens,
)

# isort: on


def main():
    """main"""
    parser = PdArgumentParser(
        (ModelArgument, DataArgument, DPOTrainingArguments, DPOConfig)
    )
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, dpo_config = (
            parser.parse_json_file_and_cmd_lines()
        )
    else:
        model_args, data_args, training_args, dpo_config = (
            parser.parse_args_into_dataclasses()
        )

    if not model_args.use_sparse_head_and_loss_fn:
        model_args.use_sparse_head_and_loss_fn = True
        logger.warning(
            "Dpo training requires use_sparse_head_and_loss_fn=True. Set use_sparse_head_and_loss_fn to True"
        )

    if data_args.max_seq_len < 16:
        data_args.max_seq_len = 16
        logger.warning(
            f"max_seq_len must be greater than 16, set max_seq_len to {data_args.max_seq_len}."
        )
    if data_args.max_seq_len < data_args.max_prompt_len + 10:
        data_args.max_prompt_len = data_args.max_seq_len - 10
        logger.warning(
            "max_seq_len must be greater than max_prompt_len + 10, "
            "set max_prompt_len to {data_args.max_prompt_len}."
        )
    if dpo_config.loss_type == "orpo":
        dpo_config.reference_free = True
        dpo_config.sft_loss_ratio = 1.0
        dpo_config.loss_type = "or"
        logger.info("orpo loss_type is equal to sft_loss + pref_loss_ratio * or_loss.")
    if dpo_config.loss_type in ["or", "simpo"] and not dpo_config.reference_free:
        dpo_config.reference_free = True
        logger.warning(
            f"{dpo_config.loss_type} loss_type only supports reference_free. Set reference_free to True."
        )
    if dpo_config.lora:
        assert model_args.continue_training, "Continue training is required for LoRA."
    if training_args.pipeline_parallel_degree > 1:
        assert (
            hasattr(training_args, "pipeline_parallel_config")
            and "enable_clear_every_step_cache"
            in training_args.pipeline_parallel_config
        ), "Should set '--pipeline_parallel_config enable_clear_every_step_cache' in bash script for pp."
    if training_args.sequence_parallel:
        if training_args.pipeline_parallel_degree > 1:
            assert (
                hasattr(training_args, "pipeline_parallel_config")
                and "disable_partial_send_recv"
                in training_args.pipeline_parallel_config
            ), "Should set '--pipeline_parallel_config disable_partial_send_recv' in bash script for pp with sp."
        if training_args.tensor_parallel_degree <= 1:
            training_args.sequence_parallel = False
            logger.info("Tensor_parallel_degree = 1. Set sequence_parallel to False.")

    if dpo_config.lora and model_args.fuse_linear:
        model_args.fuse_linear = False
        logger.info("LoRA does not support fuse_linear. Set fuse_linear to False.")
    if dpo_config.lora:
        dpo_config.ref_model_update_steps = -1
        logger.warning(
            "LoRA does not support ref_model_update_steps. Set ref_model_update_steps to -1."
        )

    if training_args.sharding_parallel_degree > 1:
        if (
            ShardingOption.SHARD_GRAD_OP in training_args.sharding
            or ShardingOption.FULL_SHARD in training_args.sharding
        ):
            if training_args.release_grads is True:
                training_args.release_grads = False

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.print_config(dpo_config, "DPOConfig")

    paddle.set_device(training_args.device)

    set_seed(training_args.seed)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: "
        f"{training_args.world_size}, distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        uc_async_save = (
            training_args.unified_checkpoint
            and "async_save" in training_args.unified_checkpoint_config
        )
        last_checkpoint = get_last_checkpoint(
            training_args.output_dir,
            signal_folder=training_args.output_signal_dir,
            uc_async_save=uc_async_save,
        )

        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set the dtype for loading model
    dtype = paddle.get_default_dtype()
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        if training_args.bf16:
            dtype = "bfloat16"

    logger.info("Start to load model ...")

    # Detect torch model.
    config_path = os.path.join(model_args.model_name_or_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    if "torch_dtype" in config_dict:
        raise ValueError(
            "Unsupported weight format: Torch weights are not compatible with Paddle model currently."
        )

    # fuse_softmax_mask only support for rocm.
    if not paddle.is_compiled_with_rocm():
        if model_args.fuse_softmax_mask:
            logger.warning(
                "The fuse_softmax_mask flag is only available when using the ROCM version of paddlepaddle. "
            )
            model_args.fuse_softmax_mask = False

    check_refined_recompute(
        training_args.refined_recompute,
        training_args.sequence_parallel,
        lora=dpo_config.lora,
    )

    if model_args.weight_quantize_algo is not None:
        if model_args.weight_quantize_algo == "weight_only_mix":
            quantization_config = dict(
                weight_quantize_algo={
                    "weight_only_int4": [".*mlp.experts.*"],
                    "weight_only_int8": [
                        ".*self_attn.qkv_proj.*",
                        ".*self_attn.o_proj.*",
                        ".*mlp.up_gate_proj.*",
                        ".*mlp.down_proj.*",
                    ],
                },
                ignore_modules=[".*out_linear.*"],
            )
        else:
            quantization_config = dict(
                weight_quantize_algo=model_args.weight_quantize_algo,
                ignore_modules=[".*out_linear.*"],
            )
    else:
        quantization_config = dict(weight_quantize_algo=model_args.weight_quantize_algo)

    model_kwargs = dict(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        dtype=dtype,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        virtual_pp_degree=model_args.virtual_pp_degree,
        pp_seg_method=model_args.pp_seg_method,
        recompute=training_args.recompute,
        recompute_granularity=model_args.recompute_granularity,
        use_flash_attention=model_args.use_flash_attention,
        tensor_parallel_output=model_args.tensor_parallel_output,
        fuse_linear=model_args.fuse_linear,
        fuse_softmax_mask=model_args.fuse_softmax_mask,
        fuse_rms_norm=model_args.fuse_rms_norm,
        fuse_swiglu=model_args.fuse_swiglu,
        fuse_gate_detach_matmul=model_args.fuse_gate_detach_matmul,
        dpo_config=dpo_config,
        sequence_parallel=training_args.sequence_parallel,
        max_sequence_length=data_args.max_seq_len,
        use_sparse_head_and_loss_fn=model_args.use_sparse_head_and_loss_fn,
        no_recompute_layers=model_args.no_recompute_layers,
        quantization_config=quantization_config,
        use_fused_head_and_loss_fn=model_args.use_fused_head_and_loss_fn,
        recompute_use_reentrant=model_args.recompute_use_reentrant,
        use_sparse_flash_attn=model_args.use_sparse_flash_attn,
        refined_recompute=training_args.refined_recompute,
        fuse_rope=model_args.fuse_rope,
        moe_group=model_args.moe_group,
        hidden_dropout_prob=training_args.hidden_dropout_prob,
        attention_probs_dropout_prob=training_args.attention_probs_dropout_prob,
        moe_multimodal_dispatch_use_allgather=model_args.moe_multimodal_dispatch_use_allgather,
        moe_group_experts=model_args.moe_group_experts,
        moe_aux_loss_lambda=model_args.moe_aux_loss_lambda,
        moe_orthogonal_loss_lambda=model_args.moe_orthogonal_loss_lambda,
        moe_z_loss_lambda=model_args.moe_z_loss_lambda,
        moe_use_hard_gate=model_args.moe_use_hard_gate,
        num_acc_steps=training_args.gradient_accumulation_steps,
        add_tail_layers=model_args.add_tail_layers,
        num_nextn_predict_layers=0,
    )
    if model_args.moe_use_aux_free is False:
        model_kwargs.update({"moe_use_aux_free": False})
    config = Ernie4_5_MoeConfig.from_pretrained(**model_kwargs)

    if (
        training_args.pipeline_parallel_degree > 1
        and model_args.weight_quantize_algo is not None
        and config.tie_word_embeddings
    ):
        raise NotImplementedError(
            "Quantization is not supported for models with tied lm_head and word_embedding \
            weights when using Pipeline Parallelism (PP)."
        )

    if config.moe_num_experts is None or config.moe_num_experts == 0:
        config.moe_group = (
            "dummy" if model_args.moe_group == "mp" else model_args.moe_group
        )

    if training_args.pipeline_parallel_degree > 1:
        model_class = Ernie4_5_MoeForCausalLMPipe
    else:
        model_class = Ernie4_5_MoeForCausalLM
    if model_args.continue_training:
        model = model_class.from_pretrained(
            model_args.model_name_or_path, config=config
        )
    else:
        model = model_class._from_config(config, dtype=dtype)

    if not dpo_config.reference_free and not dpo_config.lora:
        ref_config = Ernie4_5_MoeConfig.from_pretrained(**model_kwargs)
        if ref_config.moe_num_experts is None or ref_config.moe_num_experts == 0:
            ref_config.moe_group = (
                "dummy" if model_args.moe_group == "mp" else model_args.moe_group
            )
        ref_model = model_class._from_config(ref_config, dtype=dtype)
        # make sure the state_dict is the same to get the same loss for first step
        ref_model.set_state_dict(model.state_dict())
    else:
        ref_model = None

    model.config.dpo_config = None

    if model.config.head_dim is None:
        del model.config.head_dim
    if ref_model is not None and ref_model.config.head_dim is None:
        del ref_model.config.head_dim

    if dpo_config.lora:
        logger.info("Start to wrap model with LoRA config ...")
        if model_args.lora_path is None:
            target_modules = [
                ".*qkv_proj.*",
                ".*out_proj.*",
                ".*linear1.*",
                ".*linear2.*",
            ]
            if model_args.rslora_plus:
                model_args.rslora = True
                model_args.lora_plus_scale = 4
                model_args.lora_alpha = 4
            if model_args.weight_quantize_algo is not None:
                if model_args.rslora or model_args.lora_plus_scale != 1.0:
                    logger.info(
                        "Weight quantization is not supported in LoRA+ and RsLoRA."
                    )
            if model_args.lora_alpha == -1:
                if model_args.rslora:
                    model_args.lora_alpha = 4
                else:
                    model_args.lora_alpha = 2 * model_args.lora_rank
            lora_config = LoRAConfig(
                target_modules=target_modules,
                r=model_args.lora_rank,
                lora_alpha=model_args.lora_alpha,
                rslora=model_args.rslora,
                lora_plus_scale=model_args.lora_plus_scale,
                tensor_parallel_degree=training_args.tensor_parallel_degree,
                dtype=dtype,
                head_dim=model.config.hidden_size // model.config.num_attention_heads,
                base_model_name_or_path=model_args.model_name_or_path,
            )
            model = LoRAModel(model, lora_config)
        else:
            model = LoRAModel.from_pretrained(
                model=model, lora_path=model_args.lora_path
            )
        model.print_trainable_parameters()
        logger.info("Wraping model with LoRA config successfully !")

    tokenizer = Ernie4_5_Tokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    logger.info("Loading model & tokenizer successfully !")

    logger.info("Start to create dataset ...")
    dataset_config = {
        "tokenizer": tokenizer,
        "max_seq_len": data_args.max_seq_len,
        "max_prompt_len": data_args.max_prompt_len,
        "random_seed": training_args.seed,
        "num_replicas": training_args.dataset_world_size,
        "rank": training_args.dataset_rank,
        "num_samples_each_epoch": data_args.num_samples_each_epoch,
        "random_shuffle": data_args.random_shuffle,
        "greedy_intokens": data_args.greedy_intokens,
        "buffer_size": data_args.buffer_size,
        "use_attn_mask_start_row_indices": model_args.use_attn_mask_start_row_indices,
        "mask_out_eos_token": data_args.mask_out_eos_token,
    }

    if training_args.max_steps == -1:
        if training_args.should_load_dataset and paddle.distributed.get_rank() == 0:
            # NOTE(gongenlei): not to feed train_dataset, or the data will be wrong in next training.
            training_args, _ = dpo_estimate_training(
                tokenizer, data_args, training_args, config=model.config
            )

        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.barrier()
            pd_max_steps = paddle.to_tensor([training_args.max_steps])
            paddle.distributed.broadcast(pd_max_steps, src=0)
            training_args.max_steps = int(pd_max_steps.item())
        logger.info(
            f"Re-setting training_args.max_steps to {training_args.max_steps} ({training_args.num_train_epochs})"
        )
        if training_args.max_steps <= 0:
            raise ValueError(
                f"Invalid max_steps: {training_args.max_steps}. Please check your dataset"
            )
    if training_args.save_strategy == IntervalStrategy.EPOCH:
        training_args.save_strategy = IntervalStrategy.STEPS
        training_args.save_steps = int(
            training_args.max_steps / training_args.num_train_epochs
        )
    if training_args.evaluation_strategy == IntervalStrategy.EPOCH:
        training_args.evaluation_strategy = IntervalStrategy.STEPS
        training_args.eval_steps = int(
            training_args.max_steps / training_args.num_train_epochs
        )
    if training_args.logging_strategy == IntervalStrategy.EPOCH:
        training_args.logging_strategy = IntervalStrategy.STEPS
        training_args.logging_steps = int(
            training_args.max_steps / training_args.num_train_epochs
        )

    if training_args.should_load_dataset:
        train_dataset = create_dataset(
            task_group=data_args.train_dataset_path,
            task_group_prob=data_args.train_dataset_prob,
            sub_dataset_type=data_args.train_dataset_type,
            **dataset_config,
        )

    if training_args.do_eval and training_args.should_load_dataset:
        eval_dataset = create_dataset(
            task_group=data_args.eval_dataset_path,
            task_group_prob=data_args.eval_dataset_prob,
            sub_dataset_type=data_args.eval_dataset_type,
            is_valid=True,
            **dataset_config,
        )
    logger.info("Creating dataset successfully ...")

    trainer = ErnieMoEDPOTrainer(
        model=model,
        ref_model=ref_model,
        dpo_config=dpo_config,
        args=training_args,
        train_dataset=(
            train_dataset
            if training_args.do_train and training_args.should_load_dataset
            else None
        ),
        eval_dataset=(
            eval_dataset
            if training_args.do_eval and training_args.should_load_dataset
            else None
        ),
        tokenizer=tokenizer,
        data_collator=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_seq_len=data_args.max_seq_len,
            use_sparse_head_and_loss_fn=model_args.use_sparse_head_and_loss_fn,
            use_fused_head_and_loss_fn=model_args.use_fused_head_and_loss_fn,
            use_response_score_delta=dpo_config.offset_alpha > 0.0,
        ),
        model_with_dpo_criterion=True,
    )

    if training_args.hidden_dropout_prob or training_args.attention_probs_dropout_prob:
        trainer.add_callback(LayerwiseDropoutCallback())

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        if (
            training_args.dpo_benchmark
            and training_args.should_load_dataset
            and paddle.distributed.get_rank() == 0
        ):
            del train_dataset
            gc.collect()
            train_dataset = create_dataset(
                task_group=data_args.train_dataset_path,
                task_group_prob=data_args.train_dataset_prob,
                sub_dataset_type=data_args.train_dataset_type,
                **dataset_config,
            )
            total_effective_tokens, total_tokens = calculate_effective_tokens(
                training_args, train_dataset, data_args.max_seq_len
            )
            effective_tokens_per_second = (
                total_effective_tokens / train_result.metrics["train_runtime"]
            )
            total_tokens_per_second = (
                total_tokens / train_result.metrics["train_runtime"]
            )
            effective_ratio = 100 * total_effective_tokens / total_tokens
            logger.info(
                "[timelog] {}: {:.2f} % ({}) ".format(
                    "Effective ratio",
                    effective_ratio,
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
            logger.info(
                "[timelog] {}: {:.2f} token/s ({}) ".format(
                    "Effective tokens per second",
                    effective_tokens_per_second,
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
            logger.info(
                "[timelog] {}: {:.2f} token/s ({}) ".format(
                    "Tokens per second",
                    total_tokens_per_second,
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            )

        if not training_args.dpo_benchmark:
            trainer.save_model(
                merge_tensor_parallel=training_args.tensor_parallel_degree > 1
            )
            if paddle.distributed.get_world_size() > 1:
                paddle.distributed.barrier()
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()

    if training_args.do_eval:
        eval_result = trainer.evaluate()
        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result, combined=False)


if __name__ == "__main__":
    with paddle.amp.auto_cast(enable=False):
        main()
