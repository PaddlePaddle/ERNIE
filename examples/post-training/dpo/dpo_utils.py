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

"""DPO utils"""

from dataclasses import dataclass, field
from typing import Optional

from paddleformers.trainer import IntervalStrategy, TrainingArguments


def add_start_docstrings(*docstr):
    """Adds docstrings for a function."""

    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class DPOTrainingArguments(TrainingArguments):
    """DPOTrainingArguments"""

    unified_checkpoint: bool = field(
        default=True,
        metadata={"help": "Enable fused linear grad add strategy."},
    )
    unified_checkpoint_config: Optional[str] = field(
        default="",
        metadata={"help": "Configs to unify hybrid parallel checkpoint.\n"},
    )
    num_of_gpus: int = field(default=-1, metadata={"help": "Number of gpus."})
    pipeline_degree: int = field(
        default=1, metadata={"help": "pipeline_degree for estimate"}
    )
    tensor_degree: int = field(
        default=1, metadata={"help": "tensor_degree for estimate"}
    )
    sharding_degree: int = field(
        default=1, metadata={"help": "sharding_degree for estimate"}
    )
    dpo_benchmark: bool = field(
        default=False,
        metadata={
            "help": "Whether to run benchmark by autotuner. True for from_scratch."
        },
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

    def __post_init__(self):
        super().__post_init__()
        if self.dpo_benchmark:
            self.do_train = True
            self.do_export = False
            self.do_predict = False
            self.do_eval = False
            self.overwrite_output_dir = True
            self.load_best_model_at_end = False
            self.save_strategy = IntervalStrategy.NO
            self.evaluation_strategy = IntervalStrategy.NO
            if not self.disable_tqdm:
                self.logging_steps = 1
                self.logging_strategy = IntervalStrategy.STEPS


@dataclass
class DataArgument:
    """DataArgument"""

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
    max_prompt_len: int = field(
        default=2048, metadata={"help": "Maximum prompt length."}
    )
    num_samples_each_epoch: int = field(
        default=6000000,
        metadata={"help": "Number of samples per epoch. Used for SFT."},
    )
    random_shuffle: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable authorize code for privatization. Defaults to False."
        },
    )
    greedy_intokens: bool = field(
        default=True,
        metadata={"help": "Whether apply greedy intokens."},
    )
    buffer_size: int = field(
        default=500,
        metadata={"help": "Buffer size for greedy_intokens strategy."},
    )
    mask_out_eos_token: bool = field(
        default=True, metadata={"help": "Mask out eos token"}
    )


@dataclass
class ModelArgument:
    """ModelArgument"""

    model_name_or_path: str = field(
        default="ernie-bot",
        metadata={"help": "Pretrained model name or path to local directory."},
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
    use_flash_attention: bool = field(
        default=True, metadata={"help": "Whether to use flash attention"}
    )
    recompute_granularity: str = field(
        default="full",
        metadata={
            "help": "The granularity of recompute training can be selected as `full` or `full_attn` or `core_attn`."
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
                "`layer:Ernie4_5_DecoderLayer|EmptyLayer`, `uniform`, `[0, 30, 59]`."
            )
        },
    )
    fuse_linear: bool = field(
        default=True, metadata={"help": "Whether to use fuse_linear"}
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
    tensor_parallel_output: bool = field(
        default=True, metadata={"help": "tensor_parallel_output"}
    )
    use_sparse_head_and_loss_fn: bool = field(
        default=False,
        metadata={"help": "Whether to use sparse LM Head and loss function."},
    )
    use_sparse_flash_attn: bool = field(
        default=True,
        metadata={
            "help": "Under use attn_mask_start_row_indices=True, whether use sparse flash attention or not."
        },
    )
    use_attn_mask_start_row_indices: bool = field(
        default=True,
        metadata={
            "help": "Whether to use attn_mask_start_row_indices in flash attention."
        },
    )
    no_recompute_layers: Optional[int] = field(
        default=None,
        metadata={
            "help": "Specify the full transformer layers that should not be recomputed."
        },
    )
    weight_quantize_algo: str = field(
        default=None,
        metadata={
            "help": "Model weight quantization algorithm including 'nf4'(qlora), 'weight_only_int8'."
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
    # LoRA
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
    use_fused_head_and_loss_fn: bool = field(
        default=False,
        metadata={"help": "Whether to fuse LM Head and loss function."},
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
    fuse_rope: bool = field(
        default=False,
        metadata={"help": "Whether to fuse rotary postition embedding"},
    )
    # MoE
    use_recompute_moe: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use recompute moe"}
    )
    moe_group: Optional[str] = field(
        default="dummy",
        metadata={"help": "MoE communication group, currently support 'mp|dummy'"},
    )
    moe_multimodal_dispatch_use_allgather: Optional[str] = field(
        default="v2-alltoall-unpad", metadata={"help": "moe dispatch use allgather"}
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


@dataclass
class DPOConfig:
    """DPOConfig"""

    beta: float = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )
    offset_alpha: float = field(
        default=0.0,
        metadata={"help": "the offset coefficient for score-based DPO loss"},
    )
    simpo_gamma: float = field(
        default=0.5, metadata={"help": "the gamma parameter for SimPO loss"}
    )
    normalize_logps: bool = field(
        default=True,
        metadata={"help": "Apply logprobs normalization."},
    )
    label_smoothing: float = field(
        default=0.0, metadata={"help": "label_smoothing ratio"}
    )
    loss_type: str = field(default="sigmoid", metadata={"help": "DPO loss type"})
    pref_loss_ratio: float = field(default=1.0, metadata={"help": "DPO loss ratio"})
    sft_loss_ratio: float = field(default=0.0, metadata={"help": "SFT loss ratio"})
    dpop_lambda: float = field(default=50, metadata={"help": "dpop_lambda"})
    ref_model_update_steps: int = field(
        default=-1, metadata={"help": "Update ref model state dict "}
    )
    reference_free: bool = field(
        default=False, metadata={"help": "No reference model."}
    )
    lora: bool = field(default=False, metadata={"help": "Use LoRA model."})

    def __post_init__(self):
        if self.offset_alpha > 0.0:
            if self.loss_type != "sigmoid":
                raise ValueError(
                    "Only sigmoid loss_type supports score-based loss (offset_alpha > 0), "
                    "please set loss_type to sigmoid or set offset_alpha to 0."
                )


def calculate_effective_tokens(training_args, train_dataset, max_seq_len):
    """
    Caculate the effective tokens during training.

    Args:
        training_args (TrainingArguments): Configuration object containing:
            - data_parallel_degree (int): Number of data parallel partitions
            - sharding_parallel_degree (int): Number of sharding partitions
            - max_steps (int): Total training iterations
            - per_device_train_batch_size (int): Batch size per GPU/device
            - gradient_accumulation_steps (int): Grad accumulation steps
        train_dataset (IterableDataset): Training dataset with input_ids fields
        max_seq_len (int): Padded sequence length

    Returns:
        tuple: (effective_tokens, total_possible_tokens) where:
            - effective_tokens (int): Actual processed tokens (excludes padding)
            - total_possible_tokens (int): Theoretical maximum (batch_size * seq_len)
    """
    total_effective_tokens = 0
    try:
        data_parallel_degree = training_args.data_parallel_degree
    except Exception:
        data_parallel_degree = 1
    if training_args.sharding_parallel_degree > 1:
        sharding_parallel_degree = training_args.sharding_parallel_degree
    else:
        sharding_parallel_degree = 1

    total_batch = (
        training_args.max_steps
        * training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * sharding_parallel_degree
        * data_parallel_degree
    )
    for i, data in enumerate(train_dataset):
        if i == total_batch:
            break
        for dd in data:
            total_effective_tokens += len(dd.input_ids)
    total_tokens = total_batch * max_seq_len

    return total_effective_tokens, total_tokens
