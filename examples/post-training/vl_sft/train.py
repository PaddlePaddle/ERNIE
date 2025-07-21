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
""" Training Ernie VL Model. """

import json
import os
import random
from dataclasses import dataclass, field, asdict
from functools import partial
from typing import Optional

import numpy as np
import paddle
from typing import List, Dict

from paddle.distributed import fleet
from paddleformers.datasets import IterDataset
from paddleformers.trainer import PdArgumentParser, get_last_checkpoint
from paddleformers.utils.log import logger
from paddleformers.utils.tools import get_env_device
from pretraining_trainer import PreTrainingArguments
from trainer import SFTTrainer

from ernie.callbacks import (
    GlobalRNGCallback,
    MoECorrectionBiasAdjustCallback,
    MultiModalInterleaveCallback,
    OrthogonalCallback,
    PPNeedDataCallback,
    VitTrainableCallback,
)
from ernie.configuration import Ernie4_5_VLMoeConfig
from ernie.dataset.text_sft_reader.sft_task import KnoverDataset, create_pyreader
from ernie.dataset.vl_sft_reader import MixExampleSetJson, SFTMultimodalDatasetJson
from ernie.dataset.vl_sft_reader.data_utils import merge_fn_group_batch

from ernie.modeling_moe_vl import Ernie4_5_VLMoeForConditionalGeneration
from ernie.tokenizer_vl import Ernie4_5_VLTokenizer
from ernie.modeling_moe_vl_pp import Ernie4_5_VLMoeForConditionalGenerationPipe
from ernie.utils.misc import global_training_logs
from ernie.utils.mm_data_utils import MMSpecialTokensConfig
from ernie.utils.seed_utils import set_seed

from data_processor.steps.end2end_processing import (
    End2EndProcessor,
    End2EndProcessorArguments,
)
from data_processor.image_preprocessor.image_preprocessor_adaptive import (
    AdaptiveImageProcessor,
)


@dataclass
class ChatSFTArguments(PreTrainingArguments):
    """Chat SFT Arguments"""

    train_dataset_path: str = field(default=None, metadata={"help": "sft vl data path"})
    train_dataset_prob: str = field(default=None, metadata={"help": "sft vl data prob"})
    random_seed: int = field(default=42, metadata={"help": "random seed"})

    text_sft_task_config: str = field(
        default=None, metadata={"help": "text sft data config"}
    )
    eval_task_config: str = field(
        default=None, metadata={"help": "Path to eval_task_config."}
    )
    dataset_name: str = field(default="KnowledgeBasedSFTReader", metadata={"help": "."})
    factor: int = field(
        default=20, metadata={"help": "Pretrained model name or path to local model."}
    )
    number_of_samples_each_epoch: int = field(default=50000, metadata={"help": "."})
    pseudo_strategy: int = field(default=0, metadata={"help": "."})
    example_from_same_task_prob: float = field(default=0.0, metadata={"help": "."})
    pseudo_sampling_prob: float = field(default=0.5, metadata={"help": "."})
    trigger_data_prob: float = field(default=0.5, metadata={"help": "."})
    drop_history_with_k: bool = field(default=False, metadata={"help": "drop history"})
    add_sys_token: bool = field(
        default=False, metadata={"help": "use <sys> </sys> tokens segment system info"}
    )
    add_header_eos_token: bool = field(
        default=False,
        metadata={
            "help": "use <header_start> <header_end> </s> tokens segment system info"
        },
    )
    min_shot: int = field(default=2, metadata={"help": "min shot"})
    max_shot: int = field(default=8, metadata={"help": "max shot"})

    all_use_skip_align: bool = field(
        default=False, metadata={"help": "use skip align strategy"}
    )
    skip_align_s_policy: str = field(
        default="default", metadata={"help": "skip align strategy"}
    )
    skip_align_target_length: int = field(
        default=100000, metadata={"help": "skip align max length"}
    )
    skip_align_sub_sample_prob: float = field(
        default=0.5, metadata={"help": "skip align with prob"}
    )
    skip_align_method: str = field(
        default="skip_outter", metadata={"help": "skip align method"}
    )

    follow_deepseek: bool = field(
        default=True, metadata={"help": "use the new concat of deepseek"}
    )

    sampling_wo_replacement_data_resuming: Optional[bool] = field(
        default=True,
        metadata={
            "help": "save and load state of SFT data, support resuming without replacement"
        },
    )

    in_tokens: bool = field(
        default=False,
        metadata={
            "help": "use in-batch merge strategy, bsz=1, need to set `global_batch_size`, "
        },
    )

    use_sft_data: int = field(
        default=0,
        metadata={"help": "use standard online SFT data stream"},
    )
    hidden_dropout_prob: float = field(
        default=0.0, metadata={"help": "hidden dropout rate"}
    )
    moe_dropout_prob: float = field(default=0.0, metadata={"help": "moe dropout rate"})
    token_balance_loss: bool = field(
        default=False, metadata={"help": "use token_loss_equal_weight or not."}
    )

    pre_alloc_memory: float = field(
        default=0.0,
        metadata={
            "help": "Pre-allocate one specific-capacity empty tensor "
            "and release it for avoiding memory fragmentation"
        },
    )

    use_flash_attention: Optional[bool] = field(
        default=True, metadata={"help": "use flash attention"}
    )
    use_mem_eff_attn: Optional[bool] = field(
        default=True, metadata={"help": "use use_mem_eff_attn"}
    )
    use_flash_attn_with_mask: Optional[bool] = field(
        default=True, metadata={"help": "use use_flash_attn_with_mask"}
    )
    offload_optim: Optional[bool] = field(
        default=True, metadata={"help": "use offload_optim"}
    )
    use_train_part_sharding: Optional[bool] = field(
        default=True, metadata={"help": "use_train_part_sharding"}
    )
    text_use_train_part_sharding: Optional[bool] = field(
        default=True, metadata={"help": "text dataset use_train_part_sharding"}
    )
    rope_3d: Optional[bool] = field(default=True, metadata={"help": "use rope3d"})
    moe_with_send_router_loss: bool = field(
        default=False, metadata={"help": "use send router loss"}
    )

    resume_from_ptx_model: Optional[bool] = field(
        default=True, metadata={"help": "load pretrained weights from ptx_upload_dir"}
    )


@dataclass
class VisionArguments:
    attn_implementation: str = field(
        default="eager", metadata={"help": "Attention implementation"}
    )
    attn_sep: bool = field(
        default=True, metadata={"help": "Whether to separate attention"}
    )
    depth: int = field(default=32, metadata={"help": "Depth of the vision model"})
    embed_dim: int = field(default=1280, metadata={"help": "Embedding dimension"})
    hidden_act: str = field(
        default="quick_gelu", metadata={"help": "Hidden activation function"}
    )
    hidden_size: int = field(default=1280, metadata={"help": "Hidden size"})
    in_channels: int = field(default=3, metadata={"help": "Input channels"})
    in_chans: int = field(default=3, metadata={"help": "Input channels (alias)"})
    mlp_ratio: int = field(default=4, metadata={"help": "MLP ratio"})
    model_type: str = field(
        default="DFNRope_vision_transformer", metadata={"help": "Vision model type"}
    )
    num_heads: int = field(default=16, metadata={"help": "Number of attention heads"})
    patch_size: int = field(default=14, metadata={"help": "Patch size"})
    spatial_merge_size: int = field(default=2, metadata={"help": "Spatial merge size"})
    spatial_patch_size: int = field(default=14, metadata={"help": "Spatial patch size"})
    tensor_parallel_degree: int = field(
        default=4, metadata={"help": "Tensor parallel degree"}
    )
    use_recompute: bool = field(
        default=True, metadata={"help": "Whether to use recompute"}
    )
    vit_num_recompute_layers: int = field(
        default=10000, metadata={"help": "Number of recompute layers"}
    )


@dataclass
class ModelArguments:
    add_tail_layers: bool = field(
        default=False, metadata={"help": "Whether to add tail layers"}
    )
    attention_probs_dropout_prob: float = field(
        default=0.0, metadata={"help": "Attention probability dropout rate"}
    )
    bos_token_id: int = field(
        default=0, metadata={"help": "Beginning of sentence token ID"}
    )
    cachekv_quant: bool = field(
        default=False, metadata={"help": "Whether to quantize cache key/value"}
    )
    compression_ratio: float = field(
        default=1.0, metadata={"help": "Compression ratio"}
    )
    dpo_config: Optional[Dict] = field(
        default=None, metadata={"help": "DPO configuration"}
    )
    enable_delay_scale_loss: bool = field(
        default=True, metadata={"help": "Whether to enable delay scale loss"}
    )
    eos_token_id: int = field(default=1, metadata={"help": "End of sentence token ID"})
    freq_allocation: int = field(default=20, metadata={"help": "Frequency allocation"})
    fuse_gate_detach_matmul: bool = field(
        default=True, metadata={"help": "Whether to fuse gate detach matmul"}
    )
    fuse_linear: bool = field(
        default=True, metadata={"help": "Whether to fuse linear layers"}
    )
    fuse_ln: bool = field(
        default=False, metadata={"help": "Whether to fuse layer normalization"}
    )
    fuse_rms_norm: bool = field(
        default=False, metadata={"help": "Whether to fuse RMS normalization"}
    )
    fuse_rope: bool = field(default=False, metadata={"help": "Whether to fuse RoPE"})
    fuse_softmax_mask: bool = field(
        default=False, metadata={"help": "Whether to fuse softmax mask"}
    )
    fuse_swiglu: bool = field(default=True, metadata={"help": "Whether to fuse SwiGLU"})
    global_aux_loss: bool = field(
        default=False, metadata={"help": "Whether to use global auxiliary loss"}
    )
    head_dim: Optional[int] = field(default=None, metadata={"help": "Head dimension"})
    hidden_act: str = field(
        default="silu", metadata={"help": "Hidden activation function"}
    )
    hidden_size: int = field(default=2560, metadata={"help": "Hidden size"})
    ignored_index: int = field(default=-100, metadata={"help": "Ignored index"})
    initializer_range: float = field(
        default=0.011410316056095904, metadata={"help": "Initializer range"}
    )
    intermediate_size: int = field(
        default=12288, metadata={"help": "Intermediate size"}
    )
    max_position_embeddings: int = field(
        default=4096, metadata={"help": "Maximum position embeddings"}
    )
    mm_vocab_size: int = field(
        default=0, metadata={"help": "Multimodal vocabulary size"}
    )
    modality_detach: bool = field(
        default=False, metadata={"help": "Whether to detach modalities"}
    )
    model_type: str = field(default="ernie4_5_moe_vl", metadata={"help": "Model type"})
    moe_all_to_all_dropout: float = field(
        default=0.0, metadata={"help": "MoE all-to-all dropout rate"}
    )
    moe_aux_loss_lambda: float = field(
        default=0.0, metadata={"help": "MoE auxiliary loss lambda"}
    )
    moe_capacity: List[int] = field(
        default_factory=lambda: [128, 128, 128], metadata={"help": "MoE capacity"}
    )
    moe_dense_experts_token_type_id: int = field(
        default=3, metadata={"help": "MoE dense experts token type ID"}
    )
    moe_fuse_experts: bool = field(
        default=False, metadata={"help": "Whether to fuse MoE experts"}
    )
    moe_gate: str = field(default="top2_fused", metadata={"help": "MoE gate type"})
    moe_gate_act: str = field(
        default="softmax", metadata={"help": "MoE gate activation"}
    )
    moe_group_experts: bool = field(
        default=False, metadata={"help": "Whether to group MoE experts"}
    )
    moe_group_orthogonal_loss: bool = field(
        default=True, metadata={"help": "Whether to use MoE group orthogonal loss"}
    )
    moe_intermediate_size: List[int] = field(
        default_factory=lambda: [1536, 512], metadata={"help": "MoE intermediate sizes"}
    )
    moe_k: int = field(default=6, metadata={"help": "MoE top-k value"})
    moe_layer_end_index: List[int] = field(
        default_factory=lambda: [29, 28], metadata={"help": "MoE layer end indices"}
    )
    moe_layer_feed_fake_token: bool = field(
        default=False, metadata={"help": "Whether MoE layers feed fake tokens"}
    )
    moe_layer_interval: int = field(default=1, metadata={"help": "MoE layer interval"})
    moe_layer_start_index: List[int] = field(
        default_factory=lambda: [1, 1], metadata={"help": "MoE layer start indices"}
    )
    moe_multimodal_dispatch_use_allgather: str = field(
        default="v2-alltoall-unpad-text",
        metadata={"help": "MoE multimodal dispatch method"},
    )
    moe_norm_gate_logits: bool = field(
        default=True, metadata={"help": "Whether to normalize MoE gate logits"}
    )
    moe_num_experts: List[int] = field(
        default_factory=lambda: [64, 64], metadata={"help": "Number of MoE experts"}
    )
    moe_num_shared_experts: int = field(
        default=2, metadata={"help": "Number of shared MoE experts"}
    )
    moe_orthogonal_loss_lambda: float = field(
        default=0.0, metadata={"help": "MoE orthogonal loss lambda"}
    )
    moe_reverse_token_drop: bool = field(
        default=False, metadata={"help": "Whether to reverse MoE token drop"}
    )
    moe_use_aux_free: bool = field(
        default=True, metadata={"help": "Whether MoE uses auxiliary free"}
    )
    moe_use_hard_gate: bool = field(
        default=True, metadata={"help": "Whether MoE uses hard gate"}
    )
    moe_use_size_all2all: bool = field(
        default=False, metadata={"help": "Whether MoE uses size all-to-all"}
    )
    moe_use_token_type_bias: bool = field(
        default=False, metadata={"help": "Whether MoE uses token type bias"}
    )
    moe_z_loss_lambda: float = field(
        default=0.0, metadata={"help": "MoE z loss lambda"}
    )
    multi_token_pred_lambda: float = field(
        default=0.3, metadata={"help": "Multi-token prediction lambda"}
    )
    num_acc_steps: int = field(
        default=4, metadata={"help": "Number of accumulation steps"}
    )
    num_attention_heads: int = field(
        default=20, metadata={"help": "Number of attention heads"}
    )
    num_hidden_layers: int = field(
        default=28, metadata={"help": "Number of hidden layers"}
    )
    num_key_value_heads: int = field(
        default=4, metadata={"help": "Number of key/value heads"}
    )
    num_nextn_predict_layers: int = field(
        default=0, metadata={"help": "Number of next-n prediction layers"}
    )
    paddleformers_version: Optional[str] = field(
        default=None, metadata={"help": "PaddleFormers version"}
    )
    pp_seg_method: str = field(
        default="layer:Ernie4_5_DecoderLayer|EmptyLayer",
        metadata={"help": "Pipeline parallel segmentation method"},
    )
    recompute_granularity: str = field(
        default="core_attn", metadata={"help": "Recompute granularity"}
    )
    resampler_fuse_rms_norm: bool = field(
        default=False, metadata={"help": "Whether to fuse RMS norm in resampler"}
    )
    rms_norm_eps: float = field(
        default=1e-05, metadata={"help": "RMS normalization epsilon"}
    )
    rope_theta: int = field(default=500000, metadata={"help": "RoPE theta value"})
    sinkhorn_2gate: bool = field(
        default=True, metadata={"help": "Whether to use Sinkhorn 2-gate"}
    )
    sinkhorn_temp: float = field(
        default=0.03, metadata={"help": "Sinkhorn temperature"}
    )
    skip_recompute_ops: Dict = field(
        default_factory=dict, metadata={"help": "Skip recompute operations"}
    )
    temporal_conv_size: int = field(
        default=2, metadata={"help": "Temporal convolution size"}
    )
    tie_word_embeddings: bool = field(
        default=True, metadata={"help": "Whether to tie word embeddings"}
    )
    use_bias: bool = field(default=False, metadata={"help": "Whether to use bias"})
    use_fused_head_and_loss_fn: bool = field(
        default=False, metadata={"help": "Whether to use fused head and loss function"}
    )
    use_recompute_lm_head: bool = field(
        default=False, metadata={"help": "Whether to recompute LM head"}
    )
    use_recompute_loss_fn: bool = field(
        default=True, metadata={"help": "Whether to recompute loss function"}
    )
    use_recompute_moe: bool = field(
        default=False, metadata={"help": "Whether to recompute MoE"}
    )
    use_recompute_mtp: bool = field(
        default=False, metadata={"help": "Whether to recompute MTP"}
    )
    use_recompute_resampler: bool = field(
        default=False, metadata={"help": "Whether to recompute resampler"}
    )
    use_rmsnorm: bool = field(
        default=True, metadata={"help": "Whether to use RMS normalization"}
    )
    use_sparse_flash_attn: bool = field(
        default=True, metadata={"help": "Whether to use sparse flash attention"}
    )
    use_sparse_head_and_loss_fn: bool = field(
        default=False, metadata={"help": "Whether to use sparse head and loss function"}
    )
    use_temporal_conv: bool = field(
        default=True, metadata={"help": "Whether to use temporal convolution"}
    )
    use_var_len_flash_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use variable length flash attention"},
    )
    vision_config: VisionArguments = field(
        default_factory=VisionArguments, metadata={"help": "Vision configuration"}
    )
    vocab_size: int = field(default=103424, metadata={"help": "Vocabulary size"})
    weight_share_add_bias: bool = field(
        default=True, metadata={"help": "Whether to share weight and add bias"}
    )


def update_model_config_from_args(config: Ernie4_5_VLMoeConfig, model_args: dict):
    """update model config from args

    Args:
        config (ErnieConfig): _description_
        model_args (dict): _description_

    Returns:
        _type_: _description_
    """
    if "vision_config" in model_args:
        for k, v in model_args.pop("vision_config").items():
            if hasattr(config, "vision_config") and hasattr(config.vision_config, k):
                setattr(config.vision_config, k, v)
            else:
                logger.warning(f"vision_config key: {k} does not exist")
    for k, v in model_args.items():
        if hasattr(config, k):
            setattr(config, k, v)
        else:
            logger.warning(f"model config key: {k} does not exist")
    return config


def get_resume_checkpoint_path(config):
    """
    get resume checkpoint path from mpirun env
    """
    pdc_init_step = os.getenv("PDC_INIT_STEP")
    if (
        not hasattr(config, "resume_from_checkpoint")
        or config.resume_from_checkpoint is None
        or config.resume_from_checkpoint == ""
        or config.resume_from_checkpoint == "auto"
    ):
        if pdc_init_step is None:
            logger.info(
                "launching training process from scratch with no resume step defined."
            )
            return None
        elif pdc_init_step == "0":
            # from_scratch train process launched by pdc longjob
            logger.info(
                f"resume training process by pdc longjob with resume step: {pdc_init_step}"
            )
            return None
        else:
            # injected with mpirun by pdc longjob
            logger.info(
                f"resume training process by pdc longjob with resume step: {pdc_init_step}"
            )
            return os.path.join(config.output_dir, f"checkpoint-{pdc_init_step}")
    else:
        assert pdc_init_step is None, (
            "setting resume_from_checkpoint by yaml is deprecated in longjob, "
            + "please remove resume_from_checkpoint from yaml "
            + "and use script/restart.sh or mpirun -x PDC_INIT_STEP=<value> bash script/train.sh ..."
        )
        # user defined resume_from_checkpoint
        user_defined_resume_from_checkpoint = getattr(
            config, "resume_from_checkpoint", None
        )
        logger.info(
            f"user has defined resume_from_checkpoint: {user_defined_resume_from_checkpoint}"
        )
        return user_defined_resume_from_checkpoint


def main():
    """
    main function
    """

    parser = PdArgumentParser(
        (ChatSFTArguments, ModelArguments, *End2EndProcessorArguments)
    )
    args = parser.parse_args_into_dataclasses()
    data_processor_args = args[2:]
    model_config = asdict(args[1])
    args = args[0]

    # create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    if not hasattr(args, "pipeline_parallel_config"):
        args.pipeline_parallel_config = ""

    if getattr(args, "sequence_parallel", 0):
        logger.warning(
            "disabling `disable_partial_send_recv` when using sequence parallel"
        )
        args.pipeline_parallel_config += " disable_partial_send_recv"

    if (
        getattr(args, "bf16", False)
        and "enable_delay_scale_loss" not in args.pipeline_parallel_config
    ):
        logger.warning(
            "It is recommended to enable delay_scale_loss for better performance "
            "of precision when using bf16 in training"
        )
        args.pipeline_parallel_config += " enable_delay_scale_loss"

    if "enable_dp_comm_overlap" in args.pipeline_parallel_config:
        logger.warning(
            "Pipeline dp_comm_overlap and FusedLinearWithGradAdd can not be used at "
            "the same time."
        )

    if "enable_timer" in args.pipeline_parallel_config:
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallel,
        )

        PipelineParallel.timer_printer = lambda _: None

    args.resume_from_checkpoint = get_resume_checkpoint_path(args)
    if args.resume_from_checkpoint is not None and args.resume_from_checkpoint != "":
        assert os.path.exists(
            args.resume_from_checkpoint
        ), f"{args.resume_from_checkpoint} does not exist."
    logger.info(f"setting resume_from_checkpoint: {args.resume_from_checkpoint}")

    # hacking data processor
    data_processor_args[2].max_seq_length = args.max_seq_length
    data_processor_args[2].data_filelist = None
    data_processor_args[2].serialize_output = False
    logger.info(data_processor_args)

    args.enable_delay_scale_loss = (
        "enable_delay_scale_loss" in args.pipeline_parallel_config
    )

    if args.modality_ratio is not None:
        args.modality_ratio = (
            "".join(args.modality_ratio).replace("[", "").replace("]", "")
        )
        args.modality_ratio = args.modality_ratio.split(",")
        args.modality_ratio = [int(i) for i in args.modality_ratio]
        args.modality_interleave = (
            sum(args.modality_ratio)
            if args.modality_interleave == "acc"
            else sum(args.modality_ratio) * args.gradient_accumulation_steps
        )
        args.modality_ratio = [
            i / sum(args.modality_ratio) for i in args.modality_ratio
        ]

    if args.resume_from_ptx_model:
        args.ignore_load_lr_and_optim = True

    # same_data is set to "" and modifed here by default, but can be set to True/False explicitly
    if (
        not hasattr(args, "same_data")
        or args.same_data is None
        or args.same_data == ""
        or args.same_data == "auto"
    ):
        args.same_data = True
    logger.info(f"setting same_data: {args.same_data}")

    image_preprocess_save = AdaptiveImageProcessor.from_pretrained(
        args.model_name_or_path
    )
    for i, x in enumerate(data_processor_args):
        print("data_processor_args:\n", i, x)

    tokenizer = Ernie4_5_VLTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
        model_max_length=args.max_seq_length,
    )
    data_processor = End2EndProcessor(
        data_processor_args, tokenizer, image_preprocess_save
    )
    data_processor.train().sft()
    logger.info(f"[DEBUG] data_processor_args: {data_processor_args}")

    paddle.set_device(args.device)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    set_seed(args.random_seed)

    if get_env_device() == "gpu":
        prop = paddle.device.cuda.get_device_properties()
        if prop.total_memory < args.pre_alloc_memory * 1024 * 1024 * 1024:
            logger.warning(
                "Invalid value for `pre_alloc_memory`, so pre-allocating just failed."
            )
        elif args.pre_alloc_memory > 0:
            logger.warning(
                f"pre-allocating a tensor whose memory capacity is {args.pre_alloc_memory} GB "
                "and then release it."
            )
            memory_size = int(args.pre_alloc_memory * 1024 * 1024 * 1024)
            x = paddle.empty([memory_size], dtype=paddle.uint8)
            del x

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Define the metrics of tasks.
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        output = paddle.to_tensor(preds)
        labels = paddle.to_tensor(p.label_ids)
        output = [t.astype("float32").cuda() for t in output]
        labels = [t[t != tokenizer.ignored_index] for t in labels]
        labels = [t.cuda() for t in labels]
        all_numel = (
            (paddle.concat(labels, 0) != tokenizer.ignored_index).astype("int64").sum()
        )
        ignored = (paddle.concat(labels, 0) == -100).astype("int64").sum()
        labels = all_numel - ignored
        output = sum(output)
        logger.info(f"output : {output.item()}, labels : {labels.item()}")
        nll_loss = output / (labels + 1.0e-6)  # nll_loss is global loss
        ppl = paddle.exp(nll_loss)

        return {
            "nll_loss": nll_loss.item(),
            "ppl": ppl.item(),
            "num_token": labels.item(),
        }

    # model
    dtype = "float32"
    if args.fp16 and args.fp16_opt_level == "O2":
        paddle.set_default_dtype("float16")
        dtype = "float16"
    elif args.bf16:
        paddle.set_default_dtype("bfloat16")
        dtype = "bfloat16"

    if args.use_moe:
        if args.multimodal:
            if args.pipeline_parallel_degree > 1:
                assert args.pp_need_data

    if args.moe_group.lower() in {"mp", "tp", "model", "dummy"}:
        logger.info(f"disable moe flag when using moe-group={args.moe_group}")
        args.use_moe = False

    cfg = Ernie4_5_VLMoeConfig.from_pretrained(os.path.join(args.model_name_or_path))
    cfg.use_cache = False
    cfg.max_sequence_length = args.max_seq_length
    cfg.seqlen = args.max_seq_length
    cfg.token_balance_seqlen = args.max_seq_length * args.per_device_train_batch_size
    cfg.fp16_opt_level = args.fp16_opt_level
    cfg.moe_group = args.moe_group  # pp mp use sharding group as moe group
    cfg.dtype = dtype
    cfg.use_fp8 = args.use_fp8
    cfg.disable_pipeline_warmup = args.disable_pipeline_warmup
    cfg.enable_mtp_magic_send = args.enable_mtp_magic_send
    if args.tensor_parallel_degree > 1:
        cfg.sequence_parallel = args.sequence_parallel
        cfg.tensor_parallel_degree = max(
            fleet.get_hybrid_communicate_group().get_model_parallel_world_size(), 1
        )
        cfg.tensor_parallel_rank = max(
            fleet.get_hybrid_communicate_group().get_model_parallel_rank(), 0
        )
    else:
        cfg.sequence_parallel = False
        cfg.tensor_parallel_degree = 1
        cfg.tensor_parallel_rank = 0

    cfg.vision_config.tensor_parallel_degree = cfg.tensor_parallel_degree
    cfg.vision_config.tensor_parallel_rank = cfg.tensor_parallel_rank
    cfg.pixel_hidden_size = cfg.vision_config.hidden_size
    cfg.im_patch_id = tokenizer.get_vocab()[
        MMSpecialTokensConfig.get_special_tokens_info()["image_placeholder"]
    ]
    cfg.max_text_id = cfg.im_patch_id

    image_preprocess = AdaptiveImageProcessor.from_pretrained(args.model_name_or_path)
    image_preprocess.image_mean_tensor = paddle.to_tensor(
        image_preprocess.image_mean, dtype="float32"
    ).reshape([1, 3, 1, 1])
    image_preprocess.image_std_tensor = paddle.to_tensor(
        image_preprocess.image_std, dtype="float32"
    ).reshape([1, 3, 1, 1])
    image_preprocess.rescale_factor = paddle.to_tensor(
        image_preprocess.rescale_factor, dtype="float32"
    )
    image_preprocess.image_mean_tensor = image_preprocess.image_mean_tensor.squeeze(
        [-2, -1]
    ).repeat_interleave(cfg.vision_config.patch_size**2 * 1, -1)
    image_preprocess.image_std_tensor = image_preprocess.image_std_tensor.squeeze(
        [-2, -1]
    ).repeat_interleave(cfg.vision_config.patch_size**2 * 1, -1)

    cfg = update_model_config_from_args(cfg, model_config)
    cfg.use_flash_attention = args.use_flash_attention
    cfg.use_mem_eff_attn = args.use_mem_eff_attn
    cfg.use_flash_attn_with_mask = args.use_flash_attn_with_mask
    cfg.hidden_dropout_prob = args.hidden_dropout_prob
    cfg.moe_dropout_prob = args.moe_dropout_prob
    cfg.token_balance_loss = args.token_balance_loss
    cfg.token_balance_seqlen = args.max_seq_length * args.per_device_train_batch_size

    if args.pipeline_parallel_degree > 1:  # pp
        print(f"[sft-debug]: virtual_pp_degree={args.virtual_pp_degree}")
        cfg.virtual_pp_degree = args.virtual_pp_degree
        cfg.num_acc_steps = args.gradient_accumulation_steps
        cfg.moe_with_send_router_loss = args.moe_with_send_router_loss
        cfg.enable_delay_scale_loss = args.enable_delay_scale_loss
        cfg.balanced_image_preprocess = args.balanced_image_preprocess

        if args.pp_need_data and not args.pp_need_data_degree:
            args.pp_need_data_degree = args.pipeline_parallel_degree

        if cfg.balanced_image_preprocess:
            assert (
                args.pp_need_data
            ), "balanced image preprocess must use with pp_need_data"

        if args.from_scratch:
            model = Ernie4_5_VLMoeForConditionalGenerationPipe(cfg)

        else:
            model = Ernie4_5_VLMoeForConditionalGenerationPipe.from_pretrained(
                args.model_name_or_path,
                config=cfg,
            )
        if args.pp_need_data_degree:
            model.set_pp_need_data_degree(args.pp_need_data_degree)
    else:
        if args.from_scratch:
            model = Ernie4_5_VLMoeForConditionalGeneration(cfg)
        else:
            model = Ernie4_5_VLMoeForConditionalGeneration.from_pretrained(
                args.model_name_or_path,
                config=cfg,
            )
    logger.info(f"vision_model: {model.vision_model}")

    if image_preprocess is not None and hasattr(model, "add_image_preprocess"):
        model.add_image_preprocess(image_preprocess)

    cfg = model.config
    logger.info(f"using model type:{type(model)}")
    paddle.set_default_dtype("float32")

    logger.info(f"using model={type(model)}, cfg={cfg}")
    ortho_loss_lambda = (
        cfg.moe_orthogonal_loss_lambda
        if hasattr(cfg, "moe_orthogonal_loss_lambda")
        else 0.0
    )
    if args.use_ortho_loss_callback:
        logger.info("using orthogonal loss callback")
        cfg.moe_orthogonal_loss_lambda = 0.0

    freeze_config = set(args.freeze_config.split(" "))
    if "freeze_vision" in freeze_config and hasattr(model, "freeze_vision"):
        logger.info("Freeze model vision module")
        model.freeze_vision()

    # data
    logger.info("loading data...")
    logger.info(f"args.need_data: {args.need_data}")

    if args.do_train:
        hcg = fleet.get_hybrid_communicate_group()
        dp_rank = hcg.get_data_parallel_rank()
        dp_size = hcg.get_data_parallel_world_size()
        sharding_rank = hcg.get_sharding_parallel_rank()
        sharding_size = hcg.get_sharding_parallel_world_size()
        logger.info(
            f"""[main] hcg: dp_rank: {dp_rank},
            dp_size: {dp_size},
            sharding_rank: {sharding_rank},
            sharding_size: {sharding_size}"""
        )

        if args.need_data:
            if args.multimodal:
                train_dataset = SFTMultimodalDatasetJson(
                    train_dataset_path=args.train_dataset_path,
                    train_dataset_prob=args.train_dataset_prob,
                    tokenizer=tokenizer,
                    image_preprocess=image_preprocess,
                    seed=args.random_seed,
                    image_token_len=64,
                    seqlen=args.max_seq_length,
                    use_prompt=True,
                    mix_resolution=False,
                    special_token_loss_mask_ratio=0.5,
                    adaptive_resolution=False,
                    im_patch_id=tokenizer.get_vocab()[
                        MMSpecialTokensConfig.get_special_tokens_info()[
                            "image_placeholder"
                        ]
                    ],
                    use_crop=True,
                    crop_tile_option="9",  # args.crop_tile_option,
                    crop_tile_rate="1.0",  # args.crop_tile_rate,
                    batch_size=args.per_device_train_batch_size,
                    dp_rank=args.reeao_dataset_rank,
                    dp_size=args.reeao_dataset_world_size,
                    data_processor=data_processor,
                )
                train_dataset._load(shuffle_json=False)
                train_dataset = IterDataset(
                    MixExampleSetJson(
                        lm_weights=0.0,
                        mm_weights=1.0,
                        lm_example_set=None,
                        mm_example_set=train_dataset,
                    )
                )
            else:
                train_dataset = None
        else:
            logger.info(
                f"mp_{args.pipeline_parallel_rank}_pp{args.tensor_parallel_rank} no data needed, \
                            skip init train_dataset"
            )
            train_dataset = None

        text_sft_dataset = None
        args.is_train_text = getattr(args, "text_sft_task_config", False)
        if args.need_data and args.is_train_text:
            # text SFT close multi-thread processing data
            logger.info(
                f"[TEXT SFT] training pure text sft with {args.text_sft_task_config}."
            )
            # text_sft_task_list = args.text_sft_task_config.split(",")
            train_task_group_text = json.load(open(args.text_sft_task_config))
            train_task_group_text = [
                task for task in train_task_group_text if task["prob"] > 0
            ]
            logger.info(f"train_task_group_text: {train_task_group_text}")

            dataset_seed = (
                args.random_seed * args.factor * (args.reeao_dataset_rank + 1)
                if args.text_use_train_part_sharding
                else args.random_seed * args.factor
            )
            config_dataset_text = {
                "dataset_name": args.dataset_name,
                "is_valid": False,
                "max_seq_len": args.max_seq_length,
                "random_seed": dataset_seed,
                "dp_worldrank": args.reeao_dataset_rank,
                "dp_worldsize": args.reeao_dataset_world_size,
                "worker_index": paddle.distributed.get_rank(),
                "prefetch_factor": args.prefetch_factor,
                "task_group": train_task_group_text,
                "in_tokens": True,  # True for Text SFT
                "tokenizer": tokenizer,
                "number_of_samples_each_epoch": args.number_of_samples_each_epoch,
                "pseudo_strategy": args.pseudo_strategy,
                "example_from_same_task_prob": args.example_from_same_task_prob,
                "pseudo_sampling_prob": args.pseudo_sampling_prob,
                "trigger_data_prob": args.trigger_data_prob,
                "drop_history_with_k": args.drop_history_with_k,
                "add_sys_token": args.add_sys_token,
                "add_header_eos_token": args.add_header_eos_token,
                "ignore_load_lr_and_optim": args.ignore_load_lr_and_optim,
                "load_optimizer_and_scheduler": (
                    False if args.resume_from_ptx_model else True
                ),
                "resume_from_checkpoint": args.resume_from_checkpoint,
                "sampling_wo_replacement_data_resuming": args.sampling_wo_replacement_data_resuming,
                "min_shot": args.min_shot,
                "max_shot": args.max_shot,
                "follow_deepseek": args.follow_deepseek,
                "all_use_skip_align": args.all_use_skip_align,
                "skip_align_s_policy": args.skip_align_s_policy,
                "skip_align_target_length": args.skip_align_target_length,
                "skip_align_sub_sample_prob": args.skip_align_sub_sample_prob,
                "skip_align_method": args.skip_align_method,
                "use_train_part_sharding": args.text_use_train_part_sharding,
                "rope_3d": args.rope_3d,
            }

            text_sft_train_reader = create_pyreader(config_dataset_text)
            text_sft_generator = text_sft_train_reader.data_generator()
            # update each task's sample number
            for task_in_reader, task in zip(
                text_sft_train_reader.task_group, train_task_group_text
            ):
                if "total_num_examples" in task_in_reader:
                    task["total_num_examples"] = task_in_reader["total_num_examples"]

            text_sft_dataset = KnoverDataset(
                text_sft_generator,
                args.per_device_train_batch_size,
                ignored_index=tokenizer.ignored_index,
                task_group=train_task_group_text,
                use_mem_eff_attn=True,
                use_flash_attention=False,
            )
        else:
            logger.info("[TEXT SFT] not training pure text sft.")

    else:
        train_dataset = None
        text_sft_dataset = None

    eval_dataset = None

    data_collator = partial(
        merge_fn_group_batch,
        tokenizer=tokenizer,
        pad_to_max_seqlen=args.max_seq_length,
        im_prefix_length=256,
        rng=random.Random(2024),
        combine_batch=1,
    )

    callbacks = []
    callbacks += [GlobalRNGCallback()]
    if "freeze_lm" in freeze_config:
        if args.modality_ratio is not None:
            callbacks += [MultiModalInterleaveCallback()]
        elif hasattr(model, "update_params_stat"):
            logger.info("Freeze model lm module")
            model.update_params_stat("lm", stop_gradient=True)
    if args.pp_need_data:
        callbacks += [PPNeedDataCallback()]

    callbacks += (
        [OrthogonalCallback(ortho_loss_lambda)] if args.use_ortho_loss_callback else []
    )
    if args.pp_need_data_degree:
        callbacks += [PPNeedDataCallback()]

    if getattr(cfg, "moe_use_aux_free", 0.0) > 0.0:
        logger.info("adding aux free callback")
        callbacks += [
            MoECorrectionBiasAdjustCallback(
                args.moe_use_aux_free_update_coef, args.sequence_parallel
            )
        ]

    vit_trainable_callback = None
    if (
        args.pipeline_parallel_degree > 1
        and "freeze_vision" not in freeze_config
        and args.multimodal
    ):
        # train VIT
        vit_trainable_callback = VitTrainableCallback(args, model)

    trainer = SFTTrainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        is_train_text=args.is_train_text,
        text_sft_dataset=text_sft_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        processing_class=image_preprocess_save,
    )
    if vit_trainable_callback is not None:
        vit_trainable_callback.auto_cast_func = trainer.autocast_smart_context_manager
        trainer.add_callback(vit_trainable_callback)

    global_training_logs.accumulate = args.gradient_accumulation_steps
    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Training
    if args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model(args.output_dir)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate and tests model
    if args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()
