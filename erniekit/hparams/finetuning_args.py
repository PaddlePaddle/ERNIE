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

from dataclasses import dataclass, field
from typing import Optional

from paddleformers.trainer import TrainingArguments


@dataclass
class CheckPointArguments(TrainingArguments):
    """CheckPoint Arguments"""

    disable_ckpt_quant: bool = field(
        default=False,
        metadata={"help": "Whether disable checkpoint quantization."},
    )


@dataclass
class DistributedArguments(TrainingArguments):
    use_sp_callback: bool = field(
        default=False,
        metadata={
            "help": "Using the SP callback will skip the implementation of SPHook "
            "to avoid redundant gradient computation."
        },
    )
    # server deploy
    server_tp_degree: int = field(
        default=1,
        metadata={"help": "Tensor parallelism degree use for server deploy"},
    )


@dataclass
class SFTTrainingArguments(TrainingArguments):
    """SFT Training Arguments"""

    max_estimate_samples: int = field(
        default=1e5,
        metadata={"help": "Maximum number of samples used in estimation."},
    )
    sft_benchmark: bool = field(
        default=False,
        metadata={"help": "Whether to calculate effective token per second"},
    )


@dataclass
class DPOTrainingArguments(TrainingArguments):
    """DPOTrainingArguments"""

    # dpo estimate parameters
    num_of_gpus: int = field(
        default=-1,
        metadata={"help": "Number of gpus used in dpo estimate training."},
    )
    # base
    normalize_logps: bool = field(
        default=True,
        metadata={"help": "Apply logprobs normalization."},
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "label_smoothing ratio"},
    )
    dpo_benchmark: bool = field(
        default=False,
        metadata={
            "help": "Whether to run benchmark by autotuner. True for from_scratch."
        },
    )
    # reference model
    ref_model_update_steps: int = field(
        default=-1,
        metadata={"help": "Update ref model state dict "},
    )
    reference_free: bool = field(
        default=False,
        metadata={"help": "No reference model."},
    )
    # dpo loss
    loss_type: str = field(
        default="sigmoid",
        metadata={"help": "DPO loss type"},
    )
    pref_loss_ratio: float = field(
        default=1.0,
        metadata={"help": "DPO loss ratio"},
    )
    sft_loss_ratio: float = field(
        default=0.0,
        metadata={"help": "SFT loss ratio"},
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "the beta parameter for DPO loss"},
    )
    offset_alpha: float = field(
        default=0.0,
        metadata={"help": "the offset coefficient for score-based DPO loss"},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "the gamma parameter for SimPO loss"},
    )
    dpop_lambda: float = field(
        default=50,
        metadata={"help": "dpop_lambda"},
    )


@dataclass
class FinetuningArguments(
    SFTTrainingArguments,
    DPOTrainingArguments,
    CheckPointArguments,
    DistributedArguments,
):
    """Finetuning Argument"""

    # base
    batch_size: int = field(default=1, metadata={"help": "Batch size per GPU."})
    layerwise_lr_decay_bound: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Use a large learning rate for the top layers and "
            "a small learning rate for the bottom layers. 1.0: Do not use this strategy."
        },
    )
    decay_steps: int = field(
        default=None,
        metadata={
            "help": "The steps use to control the learing rate. If the step > decay_steps, "
            "will use the min_learning_rate."
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

    # performance
    compute_type: str = field(
        default="bf16",
        metadata={"help": "The compute type."},
    )
    weight_quantize_algo: str = field(
        default=None,
        metadata={
            "help": "Model weight quantization algorithm including 'nf4'(qlora), 'weight_only_int8'."
        },
    )

    # fp8
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
    multi_token_pred_lambda: float = field(
        default=0.3, metadata={"help": "multi token pred lambda"}
    )
    use_recompute_mtp: bool = field(
        default=False, metadata={"help": "Whether to use recompute_mtp"}
    )

    def __post_init__(self):
        self.bf16 = True
        if self.compute_type == "bf16":
            self.fp16 = False
            self.weight_quantize_algo = None
        elif self.compute_type == "fp16":
            self.bf16 = False
            self.fp16 = True
            self.weight_quantize_algo = None
        elif self.compute_type == "fp8":
            self.weight_quantize_algo = "fp8linear"
            self.apply_hadamard = True
            self.optim = "adamw_custom"
            self.use_lowprecision_moment = True
            self.tensorwise_offload_optimizer = True
            self.optim_shard_num = 8
            self.unified_checkpoint_config = "ignore_merge_optimizer"
        elif self.compute_type == "wint8":
            self.weight_quantize_algo = "weight_only_int8"
        elif self.compute_type == "wint4/8":
            self.weight_quantize_algo = "weight_only_mix"
        else:
            raise ValueError(f"Unknown compute_type: {self.compute_type}")
        self.per_device_train_batch_size = self.batch_size
        self.per_device_eval_batch_size = self.batch_size
        self.server_tp_degree = self.tensor_parallel_degree

        super().__post_init__()
