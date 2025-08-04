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
"""Training Ernie Model."""

import gc
import importlib.util
import math
import os
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

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
from paddleformers.trainer import (
    IntervalStrategy,
    PdArgumentParser,
    RuntimeTimer,
    TrainingArguments,
    get_last_checkpoint,
    set_seed,
)
from paddleformers.trainer.trainer_utils import ShardingOption
from paddleformers.transformers.model_utils import unwrap_model
from paddleformers.utils.log import logger

from ernie.callbacks import LayerwiseDropoutCallback
from ernie.configuration import Ernie4_5_MoeConfig
from ernie.modeling_moe import Ernie4_5_MoeForCausalLM
from ernie.modeling_moe_pp import Ernie4_5_MoeForCausalLMPipe
from ernie.tokenizer import Ernie4_5_Tokenizer
from ernie.utils.common_utils import (
    add_start_docstrings,
    calculate_effective_tokens,
    check_refined_recompute,
    estimate_training,
    save_stop_info,
)
from ernie.utils.download_utils import check_download_repo

# isort: off
from trainer import ErnieMoETrainer

# isort: on


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class SFTTrainingArguments(TrainingArguments):
    """SFT Training Arguments"""

    unified_checkpoint: bool = field(
        default=True,
        metadata={
            "help": "Enable fused linear grad add strategy, which will reduce elementwise "
            "add for grad accumulation in the backward of nn.Linear ."
        },
    )
    unified_checkpoint_config: Optional[str] = field(
        default="",
        metadata={
            "help": (
                "Configs to unify hybrid parallel checkpoint.\n"
                "Following options are supports:\n"
                "- skip_save_model_weight: do not save model weights when the masters weight exist\n"
                "- master_weight_compatible: 1. if the master weights exist, only load when needed\n"
                "                            2. if master weights does not exist, convert model weights"
                " to master weights when needed\n"
                "- async_save: enable asynchronous saving checkpoints to disk\n"
                "- enable_all_options: enable all optimization configurations\n"
            )
        },
    )
    decay_steps: int = field(
        default=None,
        metadata={
            "help": "The steps use to control the learing rate. If the step > decay_steps, "
            "will use the min_learning_rate."
        },
    )
    max_estimate_samples: int = field(
        default=1e5,
        metadata={"help": "Maximum number of samples used in estimation."},
    )
    dropout_warmup_steps: int = field(
        default=0,
        metadata={"help": "dropout warmup steps"},
    )
    hidden_dropout_prob: float = field(
        default=0.0,
        metadata={"help": "dropout probability for hidden layers"},
    )
    attention_probs_dropout_prob: float = field(
        default=0.0,
        metadata={"help": "dropout probability for attention layers"},
    )
    disable_ckpt_quant: bool = field(
        default=False,
        metadata={"help": "Whether disable checkpoint quantization."},
    )
    sequence_parallel: bool = field(
        default=True, metadata={"help": "Whether to use sequence_parallel"}
    )
    layerwise_lr_decay_bound: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Use a large learning rate for the top layers and "
            "a small learning rate for the bottom layers. 1.0: Do not use this strategy."
        },
    )
    use_sp_callback: bool = field(
        default=False,
        metadata={
            "help": "Using the SP callback will skip the implementation of SPHook "
            "to avoid redundant gradient computation."
        },
    )
    # Quantiztaion
    weight_quantize_algo: str = field(
        default=None,
        metadata={
            "help": "Model weight quantization algorithm including 'nf4'(qlora), 'weight_only_int8'."
        },
    )


@dataclass
class DataArgument:
    """Data Argument"""

    train_dataset_type: str = field(
        default="erniekit",
        metadata={"help": "List contains type of training datasets."},
    )
    train_dataset_path: str = field(
        default="examples/data/sft-train.jsonl",
        metadata={"help": "List contains path of training data sources."},
    )
    train_dataset_prob: str = field(
        default="1.0",
        metadata={"help": "List contains probabilities of training data sources."},
    )
    eval_dataset_type: str = field(
        default="erniekit", metadata={"help": "List contains type of eval datasets."}
    )
    eval_dataset_path: str = field(
        default="examples/data/sft-eval.jsonl",
        metadata={"help": "List contains path of eval data sources."},
    )
    eval_dataset_prob: str = field(
        default="1.0",
        metadata={"help": "List contains probabilities of eval data sources."},
    )
    max_seq_len: int = field(
        default=4096, metadata={"help": "Maximum sequence length."}
    )
    in_tokens_batching: bool = field(
        default=True,
        metadata={"help": "Whether to using in tokens batching strategy."},
    )
    num_samples_each_epoch: int = field(
        default=100000,
        metadata={"help": "Number of samples per epoch. Used for SFT."},
    )
    num_comparisons: int = field(
        default=6, metadata={"help": "Number of candidate responses."}
    )
    use_cls: bool = field(
        default=True,
        metadata={"help": "Whether to use cls to predict RM score."},
    )
    sft_benchmark: bool = field(
        default=False,
        metadata={"help": "Whether to calculate effective token per second"},
    )
    random_shuffle: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable authorize code for privatization. Defaults to False."
        },
    )
    greedy_intokens: bool = field(
        default=True,
        metadata={"help": "Whether to use greedy_intokens packing method."},
    )
    dataset_type: str = field(
        default="iterable",
        metadata={
            "help": (
                "Specify the type of dataset to use. Options are 'iterable' "
                "for 'IterableDataset' and 'map' for 'MapDataset'."
            )
        },
    )
    offline_dataset_path: str = field(
        default=None,
        metadata={
            "help": (
                "If 'dataset_type' is set to 'map', this field is required to "
                "specify the path to the offline dataset."
            )
        },
    )


@dataclass
class ModelArgument:
    """Model Argument"""

    model_name_or_path: str = field(
        default="ernie-bot",
        metadata={"help": "Pretrained model name or path to local directory."},
    )
    tensor_parallel_output: bool = field(
        default=True,
        metadata={
            "help": "If set to True, this option is used with fleet.meta_parallel. "
            "ParallelCrossEntropy to calculate cross-entropy loss for parallel model."
        },
    )
    from_hf_hub: bool = field(
        default=False,
        metadata={"help": "Whether to download model from huggingface hub"},
    )
    from_aistudio: bool = field(
        default=False,
        metadata={"help": "Whether to download model from aistudio"},
    )
    from_modelscope: bool = field(
        default=False,
        metadata={"help": "Whether to download model from modelscope"},
    )
    # LoRA
    lora: bool = field(
        default=False, metadata={"help": "Whether to use LoRA technique."}
    )
    lora_rank: int = field(default=8, metadata={"help": "Lora rank."})
    lora_path: str = field(
        default=None, metadata={"help": "Initialize lora state dict."}
    )
    rslora: bool = field(default=False, metadata={"help": "Whether to use RsLoRA"})
    lora_plus_scale: float = field(
        default=1.0, metadata={"help": "Lora B scale in LoRA+ technique"}
    )
    lora_alpha: int = field(default=-1, metadata={"help": "lora_alpha"})
    rslora_plus: bool = field(
        default=False, metadata={"help": "Strengthen lora performance"}
    )
    use_flash_attention: bool = field(
        default=True, metadata={"help": "Whether to use flash attention"}
    )
    use_sparse_head_and_loss_fn: bool = field(
        default=False,
        metadata={"help": "Whether to use sparse LM Head and loss function."},
    )
    use_fused_head_and_loss_fn: bool = field(
        default=False,
        metadata={"help": "Whether to fuse LM Head and loss function."},
    )
    recompute_granularity: str = field(
        default="full",
        metadata={
            "help": "The granularity of recompute training can be selected as `full` or `full_attn` or `core_attn`. "
            " `full` means complete all transformers, `full_attn` indicates only recompute all self attention parts,"
            " `core_attn` indicates that only the `softmax (qkT) v` part is recomputed. Note: In terms of memory usage,"
            " `core_attn` > `full_attn` > `full`, if the selected policy generates an OOM error, the recompute can be"
            " changed appropriately recompute_granularity. (default: `full`)"
        },
    )
    no_recompute_layers: Optional[int] = field(
        default=None,
        metadata={
            "help": "Specify the full transformer layers that should not be recomputed."
        },
    )
    offload_recompute_inputs: bool = field(
        default=False,
        metadata={
            "help": "Whether to offload input Tensors of recompute to Pinned-Memory/CPU."
        },
    )
    virtual_pp_degree: int = field(
        default=1,
        metadata={"help": "virtual_pp_degree"},
    )
    pp_seg_method: str = field(
        default="layer:Ernie4_5_DecoderLayer|EmptyLayer",
        metadata={
            "help": (
                "The method used to segment the pipeline layers among pipeline stages. "
                "Possible values include `layer:Ernie4_5_DecoderLayer`, "
                "`layer:Ernie4_5_DecoderLayer|Empty`, `uniform`, `[0, 30, 59]`."
            )
        },
    )
    fuse_linear: bool = field(
        default=False, metadata={"help": "Whether to use fused_gemm_epilogue"}
    )
    fuse_rope: bool = field(
        default=False,
        metadata={"help": "Whether to fuse rotary postition embedding"},
    )
    fuse_softmax_mask: bool = field(
        default=False, metadata={"help": "Whether to fuse softmax and add"}
    )
    fuse_rms_norm: bool = field(
        default=True, metadata={"help": "Whether to fuse RMSNorm for efficiency"}
    )
    fuse_swiglu: bool = field(
        default=True,
        metadata={
            "help": "Whether to fuse SwiGLU projection and activation for efficiency"
        },
    )
    fuse_gate_detach_matmul: bool = field(
        default=True,
        metadata={
            "help": "Whether to use the fused gate-detach matmul implementation."
        },
    )
    use_attn_mask_start_row_indices: bool = field(
        default=True,
        metadata={
            "help": "Whether to use attn_mask_start_row_indices in flash attention."
        },
    )
    use_sparse_flash_attn: bool = field(
        default=True,
        metadata={
            "help": "Under use attn_mask_start_row_indices=True, whether use sparse flash attention or not."
        },
    )
    recompute_use_reentrant: bool = field(
        default=False,
        metadata={"help": "recompute_use_reentrant"},
    )
    continue_training: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to train from existing paddleformers model weights.\n"
                "If set True, the model_name_or_path argument must exist in the paddleformers models."
            )
        },
    )
    add_tail_layers: int = field(
        default=False,
        metadata={
            "help": (
                "Add EmptyLayer after Ernie4_5_DecoderLayerPipe. Only for Pipeline Parallel"
            )
        },
    )

    # MoE
    use_recompute_moe: Optional[bool] = field(
        default=False, metadata={"help": "Whether to apply recompute to MoE layers."}
    )
    moe_group: Optional[str] = field(
        default="dummy",
        metadata={"help": "MoE communication group. Supported values: 'mp', 'dummy'."},
    )
    moe_multimodal_dispatch_use_allgather: Optional[str] = field(
        default="v2-alltoall-unpad",
        metadata={"help": "moe dispatch use unpad allgather strategy."},
    )
    moe_group_experts: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to apply group-wise processing to expert gate logits."
        },
    )
    moe_aux_loss_lambda: Optional[float] = field(
        default=1e-5,
        metadata={"help": "Lambda value for moe aux loss."},
    )
    moe_orthogonal_loss_lambda: Optional[float] = field(
        default=0.0,
        metadata={"help": "Lambda value for moe orthogonal loss."},
    )
    moe_z_loss_lambda: Optional[float] = field(
        default=0.0,
        metadata={"help": "Lambda value for moe z loss."},
    )
    moe_use_hard_gate: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use hard gate. If `moe_use_hard_gate` is True, a hard "
            "routing strategy is used instead of a learned gating network."
        },
    )
    moe_use_aux_free: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to use auxiliary‑loss‑free routing. If True, "
            "load balancing (using expert bias adjustments) is used instead "
            "of traditional auxiliary loss for MoE."
        },
    )

    apply_hadamard: bool = field(
        default=True, metadata={"help": "Whether to apply hadamard"}
    )
    hadamard_block_size: int = field(
        default=32, metadata={"help": "hadamard block size"}
    )
    quant_input_grad: bool = field(
        default=False, metadata={"help": "Whether to quantize input grad"}
    )
    quant_weight_grad: bool = field(
        default=False, metadata={"help": "Whether to quantize weight grad"}
    )
    apply_online_actscale_step: int = field(
        default=200,
        metadata={
            "help": "Use online activation scale for first N step to keep stable training."
        },
    )
    actscale_moving_rate: float = field(
        default=0.01, metadata={"help": "EMA moving_rate for activation scale"}
    )
    fp8_format_type: str = field(default="hybrid", metadata={"help": "FP8 Format"})
    num_nextn_predict_layers: int = field(
        default=0, metadata={"help": "Number of nextn predict layers."}
    )
    multi_token_pred_lambda: float = field(
        default=0.3, metadata={"help": "multi token pred lambda"}
    )
    use_recompute_mtp: bool = field(
        default=False, metadata={"help": "Whether to use recompute_mtp"}
    )


def main():
    """
    The main function that creates a model with parameters configured from pretrained settings,
    arguments, and training the sft/lora model.

    Args:
        None

    Returns:
        None
    """
    parser = PdArgumentParser((ModelArgument, DataArgument, SFTTrainingArguments))
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file_and_cmd_lines()
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
    if model_args.lora and model_args.fuse_linear:
        model_args.fuse_linear = False
        logger.info("LoRA does not support fuse_linear. Set fuse_linear to False.")
    if training_args.recompute and model_args.offload_recompute_inputs:
        assert (
            model_args.recompute_use_reentrant
        ), "offload_recompute_inputs can only be enabled along with reentrant recompute."
        assert (
            model_args.recompute_granularity == "full"
        ), "To save device memory, please try higher recompute_granularity before enabling offload_recompute_inputs."
        if training_args.pipeline_parallel_degree > 1:
            logger.debug(
                "offload_recompute_inputs is not supported in pipeline parallel. Set offload_recompute_inputs to False."
            )
            model_args.offload_recompute_inputs = False

    runtime_timer = RuntimeTimer("Training")

    if training_args.sharding_parallel_degree > 1:
        if (
            ShardingOption.SHARD_GRAD_OP in training_args.sharding
            or ShardingOption.FULL_SHARD in training_args.sharding
        ):
            if training_args.release_grads is True:
                training_args.release_grads = False

    # checkpoint O1 quantization is open by default.
    if (
        not training_args.disable_ckpt_quant
        and training_args.ckpt_quant_stage == "O0"
        and not model_args.lora
    ):
        training_args.ckpt_quant_stage = "O1"
    elif training_args.disable_ckpt_quant:
        training_args.ckpt_quant_stage = "O0"

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    if data_args.sft_benchmark:
        training_args.do_train = True
        training_args.do_export = False
        training_args.do_predict = False
        training_args.do_eval = False
        training_args.overwrite_output_dir = True
        training_args.load_best_model_at_end = False
        training_args.report_to = []
        training_args.save_strategy = IntervalStrategy.NO
        training_args.evaluation_strategy = IntervalStrategy.NO
        if not training_args.disable_tqdm:
            training_args.logging_steps = 1
            training_args.logging_strategy = IntervalStrategy.STEPS

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

    if (
        last_checkpoint is not None
        and model_args.continue_training
        and not model_args.lora
    ):
        model_args.continue_training = False
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. Set `continue_training` to False."
        )

    # Set the dtype for loading model
    dtype = paddle.get_default_dtype()
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        if training_args.bf16:
            dtype = "bfloat16"

    logger.info("Start to load model ...")

    model_args.model_name_or_path = check_download_repo(
        model_args.model_name_or_path,
        from_hf_hub=model_args.from_hf_hub,
        from_aistudio=model_args.from_aistudio,
        from_modelscope=model_args.from_modelscope,
    )

    if getattr(model_args, "from_modelscope", False):
        os.environ["from_modelscope"] = "True"

    model_class = Ernie4_5_MoeForCausalLM
    if training_args.pipeline_parallel_degree > 1:
        model_class = Ernie4_5_MoeForCausalLMPipe
    if (
        model_args.moe_group.lower() in {"data", "dp"}
        and training_args.data_parallel_degree > 1
    ):
        training_args.use_expert_parallel = True

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
        lora=model_args.lora,
    )

    runtime_timer.start("basemodel loading time")
    if training_args.weight_quantize_algo is not None:
        if training_args.weight_quantize_algo == "weight_only_mix":
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
            weight_quantize_algo = training_args.weight_quantize_algo
        quantization_config = dict(
            weight_quantize_algo=weight_quantize_algo,
            ignore_modules=[".*out_linear.*"],
            apply_hadamard=model_args.apply_hadamard,
            hadamard_block_size=model_args.hadamard_block_size,
            quant_input_grad=model_args.quant_input_grad,
            quant_weight_grad=model_args.quant_weight_grad,
            apply_online_actscale_step=model_args.apply_online_actscale_step,
            actscale_moving_rate=model_args.actscale_moving_rate,
            fp8_format_type=model_args.fp8_format_type,
        )
        if training_args.weight_quantize_algo == "fp8linear":
            quantization_config.update(
                {
                    "dense_quant_type": "tensor_wise_fp8",
                    "moe_quant_type": "tensor_wise_fp8",
                    "quantization": "mix_quant",
                }
            )
    else:
        quantization_config = dict(
            weight_quantize_algo=training_args.weight_quantize_algo
        )

    model_config = Ernie4_5_MoeConfig.from_pretrained(
        model_args.model_name_or_path,
        dtype=dtype,
        quantization_config=quantization_config,
        from_hf_hub=model_args.from_hf_hub,
        from_aistudio=model_args.from_aistudio,
        convert_from_torch=False,
    )
    model_config.tensor_parallel_degree = training_args.tensor_parallel_degree
    model_config.tensor_parallel_rank = training_args.tensor_parallel_rank
    model_config.recompute = training_args.recompute
    model_config.recompute_granularity = model_args.recompute_granularity
    model_config.no_recompute_layers = model_args.no_recompute_layers
    model_config.refined_recompute = training_args.refined_recompute
    model_config.offload_recompute_inputs = model_args.offload_recompute_inputs
    model_config.use_flash_attention = model_args.use_flash_attention
    model_config.sequence_parallel = training_args.sequence_parallel
    model_config.use_sparse_head_and_loss_fn = model_args.use_sparse_head_and_loss_fn
    model_config.use_fused_head_and_loss_fn = model_args.use_fused_head_and_loss_fn
    model_config.tensor_parallel_output = model_args.tensor_parallel_output
    model_config.virtual_pp_degree = model_args.virtual_pp_degree
    model_config.pp_seg_method = model_args.pp_seg_method
    model_config.add_tail_layers = model_args.add_tail_layers
    model_config.fuse_linear = model_args.fuse_linear
    model_config.fuse_rope = model_args.fuse_rope
    model_config.fuse_softmax_mask = model_args.fuse_softmax_mask
    model_config.fuse_rms_norm = model_args.fuse_rms_norm
    model_config.fuse_swiglu = model_args.fuse_swiglu
    model_config.fuse_gate_detach_matmul = model_args.fuse_gate_detach_matmul
    model_config.max_sequence_length = data_args.max_seq_len
    model_config.recompute_use_reentrant = model_args.recompute_use_reentrant
    model_config.use_sparse_flash_attn = model_args.use_sparse_flash_attn
    model_config.use_recompute_moe = model_args.use_recompute_moe
    model_config.moe_group = model_args.moe_group
    model_config.moe_group_experts = model_args.moe_group_experts
    model_config.moe_aux_loss_lambda = model_args.moe_aux_loss_lambda
    model_config.moe_orthogonal_loss_lambda = model_args.moe_orthogonal_loss_lambda
    model_config.moe_z_loss_lambda = model_args.moe_z_loss_lambda
    model_config.moe_use_hard_gate = model_args.moe_use_hard_gate
    model_config.moe_multimodal_dispatch_use_allgather = (
        model_args.moe_multimodal_dispatch_use_allgather
    )
    if model_args.moe_use_aux_free is False:
        model_config.moe_use_aux_free = model_args.moe_use_aux_free
    model_config.hidden_dropout_prob = training_args.hidden_dropout_prob
    model_config.attention_probs_dropout_prob = (
        training_args.attention_probs_dropout_prob
    )
    model_config.num_acc_steps = training_args.gradient_accumulation_steps
    model_config.num_nextn_predict_layers = model_args.num_nextn_predict_layers
    model_config.multi_token_pred_lambda = model_args.multi_token_pred_lambda
    model_config.use_recompute_mtp = model_args.use_recompute_mtp
    if model_config.moe_num_experts is None or model_config.moe_num_experts == 0:
        model_config.moe_group = (
            "dummy" if model_args.moe_group == "mp" else model_args.moe_group
        )

    if (
        training_args.pipeline_parallel_degree > 1
        and training_args.weight_quantize_algo is not None
        and model_config.tie_word_embeddings
    ):
        raise NotImplementedError(
            "Quantization is not supported for models with tied lm_head and word_embedding \
            weights when using Pipeline Parallelism (PP)."
        )

    if model_args.continue_training or training_args.weight_quantize_algo is not None:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            from_hf_hub=model_args.from_hf_hub,
            from_aistudio=model_args.from_aistudio,
            convert_from_torch=False,
        )
    else:
        model = model_class.from_config(model_config, dtype=dtype)

    if model.config.head_dim is None:
        del model.config.head_dim

    paddle.device.cuda.empty_cache()
    logger.info("Loading model successfully !")
    logger.debug(f"Model config: {model.config}")
    logger.info(f"{runtime_timer.log()}")

    tokenizer = Ernie4_5_Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        from_hf_hub=model_args.from_hf_hub,
        from_aistudio=model_args.from_aistudio,
        convert_from_torch=False,
    )

    logger.info("Start to create dataset ...")
    dataset_config = {
        "tokenizer": tokenizer,
        "max_seq_len": data_args.max_seq_len,
        "random_seed": training_args.seed,
        "num_replicas": training_args.dataset_world_size,
        "rank": training_args.dataset_rank,
    }
    from ernie.dataset.finetuning import collate_fn

    if data_args.dataset_type == "map":
        from ernie.dataset.finetuning import create_indexed_dataset as create_dataset
    else:
        from ernie.dataset.finetuning import create_dataset
    dataset_config.update(
        {
            "num_samples_each_epoch": data_args.num_samples_each_epoch,
            "random_shuffle": data_args.random_shuffle,
            "greedy_intokens": data_args.greedy_intokens,
        }
    )

    if training_args.should_load_dataset:
        if data_args.dataset_type == "map":
            train_file_path = os.path.join(data_args.offline_dataset_path, "train")
            train_dataset = create_dataset(data_file_prefix=train_file_path)
        else:
            train_dataset = create_dataset(
                task_group=data_args.train_dataset_path,
                task_group_prob=data_args.train_dataset_prob,
                sub_dataset_type=data_args.train_dataset_type,
                **dataset_config,
            )

    if training_args.do_eval and training_args.should_load_dataset:
        if data_args.dataset_type == "map":
            eval_file_path = os.path.join(data_args.offline_dataset_path, "eval")
            eval_dataset = create_dataset(data_file_prefix=eval_file_path)
        else:
            eval_dataset = create_dataset(
                task_group=data_args.eval_dataset_path,
                task_group_prob=data_args.eval_dataset_prob,
                sub_dataset_type=data_args.eval_dataset_type,
                is_valid=True,
                **dataset_config,
            )

    logger.info("Creating dataset successfully ...")

    data_collator = partial(
        collate_fn,
        tokenizer=tokenizer,
        model_args=model_args,
        max_seq_len=data_args.max_seq_len + model_config.num_nextn_predict_layers,
    )

    if model_args.lora:
        logger.info("Start to wrap model with LoRA config ...")

        from ernie.utils.peft_utils import initialize_lora_model

        model = initialize_lora_model(
            model=model,
            training_args=training_args,
            model_args=model_args,
            resume_from_checkpoint=last_checkpoint is not None,
            dtype=dtype,
        )

    if training_args.max_steps == -1:
        if training_args.should_load_dataset and paddle.distributed.get_rank() == 0:
            if data_args.dataset_type != "map":
                training_args.max_steps = estimate_training(
                    train_dataset, data_args, training_args, model_args
                )
                del train_dataset
                gc.collect()
                train_dataset = create_dataset(
                    task_group=data_args.train_dataset_path,
                    task_group_prob=data_args.train_dataset_prob,
                    sub_dataset_type=data_args.train_dataset_type,
                    **dataset_config,
                )
            else:
                global_batch_size = (
                    training_args.per_device_train_batch_size
                    * training_args.gradient_accumulation_steps
                    * training_args.dataset_world_size
                )
                training_args.max_steps = math.ceil(
                    len(train_dataset) / global_batch_size
                )

        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.barrier()
            max_steps = paddle.to_tensor([training_args.max_steps])
            paddle.distributed.broadcast(max_steps, src=0)
            training_args.max_steps = int(max_steps.item())
        if training_args.max_steps <= 0:
            raise ValueError(
                f"Invalid max_steps: {training_args.max_steps}. Please check your dataset"
            )

        logger.info(f"Re-setting training_args.max_steps to {training_args.max_steps}.")
    # Create the learning_rate sheduler and optimizer
    if training_args.decay_steps is None:
        training_args.decay_steps = training_args.max_steps

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

    if (
        not model_args.use_sparse_head_and_loss_fn
        and not training_args.prediction_loss_only
    ):
        unwraped_model = unwrap_model(model)
        if hasattr(model, "compute_metrics"):
            compute_metrics = model.compute_metrics
        elif hasattr(unwraped_model, "compute_metrics"):
            # NOTE(liuting): if model is LoRAModel, we need to unwrap it first.
            compute_metrics = unwraped_model.compute_metrics
        else:
            compute_metrics = None
    else:
        compute_metrics = None

    trainer = ErnieMoETrainer(
        model=model,
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
        do_generation=False,
        data_args=data_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainable_parameters = [
        p
        for p in model.parameters()
        if not p.stop_gradient or ("quantization_linear" in p.name and "w_1" in p.name)
    ]
    trainer.set_optimizer_grouped_parameters(trainable_parameters)

    if training_args.hidden_dropout_prob or training_args.attention_probs_dropout_prob:
        trainer.add_callback(LayerwiseDropoutCallback())

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        if not data_args.sft_benchmark:
            runtime_timer.start("model saving time")
            trainer.save_model(
                merge_tensor_parallel=training_args.tensor_parallel_degree > 1
            )
            if paddle.distributed.get_world_size() > 1:
                paddle.distributed.barrier()
            logger.info(f"{runtime_timer.log()}")
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()

        if (
            training_args.should_load_dataset
            and data_args.sft_benchmark
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

    if training_args.do_eval:
        eval_result = trainer.evaluate()
        trainer.log_metrics("eval", eval_result)
        # NOTE(gongenlei): set combined=False to avoid overwriting errors on AFS
        trainer.save_metrics("eval", eval_result, combined=False)

    save_stop_info(
        training_args,
        trainer.state.global_step,
        outside_eval=training_args.do_eval,
        outside_predict=0,
    )


if __name__ == "__main__":
    with paddle.amp.auto_cast(enable=False):
        main()
