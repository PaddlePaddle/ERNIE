# Training Configuration

This document provides a comprehensive reference for all training and evaluation parameters available in ERNIEKit. It covers:

- Basic model configuration and training setup
- Evaluation metrics and strategies
- Performance optimization techniques
- Distributed training configurations
- Memory optimization options
- Checkpoint saving strategies
- Acceleration methods
- Mixed precision training settings
- Specialized configurations for SFT, LoRA, DPO and FP8 training

Each parameter is documented with its type, default value and detailed description to help developers properly configure their training jobs.

## 1. General Configuration

## 1. General Configuration

### 1.1 Basic Configuration

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name_or_path` | str | Required | Model name or local model path for the model and tokenizer |
| `hidden_dropout_prob` | float | 0.0 | Dropout probability for hidden layers |
| `attention_probs_dropout_prob` | float | 0.0 | Dropout probability for attention layers |
| `dropout_warmup_steps` | int | 0 | Warmup steps for dropout. Dropout probability increases linearly during warmup and disables afterward. Set to 0 to disable dropout. |
| `weight_quantize_algo` | str | Required | Model quantization algorithm. Options: `weight_only_mix` (expert weights as int4, other linear layers as int8) or `weight_only_int8` (all linear layers as int8) or `fp8_linear` |
| `output_dir` | str | Required | Directory to save model files, checkpoints, tokenizers, and evaluation results |
| `logging_steps` | int | Required | Logging interval. Decrease for more frequent updates. |
| `logging_dir` | str | Required | Log directory (defaults to `output_dir` if unspecified) |
| `do_eval` | bool | False | Enable model evaluation |
| `do_train` | bool | False | Enable training |
| `disable_tqdm` | bool | False | Disable tqdm progress bar for estimating total training time |
| `continue_training` | bool | True | Load pretrained weights to continue training |

### 1.2 Evaluation

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `per_device_eval_batch_size` | int | Required | Evaluation batch size (micro batch size) |
| `eval_dataset_path` | str | Required | Path to evaluation dataset (see [sft-eval.jsonl](../examples/data/sft-eval.jsonl) |
| `eval_dataset_prob` | str | 1.0 | Evaluation dataset sampling probability. |
| `eval_dataset_type` | str | erniekit | Evaluation dataset type. |
| `eval_steps` | int | Required | Evaluation interval steps |
| `evaluation_strategy` | str | "steps" | Evaluation strategy. "steps" enables periodic evaluation |
| `max_evaluate_steps` | int | 1 | Maximum steps per evaluation (if positive) |

### 1.3 Training Performance

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `train_dataset_path` | str | Required | Training dataset path (see [sft-train.jsonl](../examples/data/sft-train.jsonl)) |
| `train_dataset_prob` | str | 1.0 | Training dataset sampling probability. |
| `train_dataset_type` | str | erniekit | Training dataset type. |
| `max_steps` | int | Required | Maximum training steps (overrides `num_train_epochs` if set) |
| `num_train_epochs` | int | Required | Training epochs |
| `per_device_train_batch_size` | int | Required | Training batch size (micro batch size). Global batch size = DP * sharding * micro_batch_size * gradient_accumulation_steps |
| `gradient_accumulation_steps` | int | Required | Gradient accumulation steps |
| `weight_decay` | float | 0.0 | AdamW optimizer weight decay |
| `seed` | int | 42 | Random seed |
| `max_seq_len` | int | Required | Maximum token length. Reduce if OOM occurs when increasing GBS. |
| `learning_rate` | float | Required | Learning rate (SFT: 3e-5, DPO: 1e-6, SFT-LoRA: 3e-4, DPO-LoRA: 1e-5) |
| `warmup_steps` | int | Required | Warmup steps (typically 1%-10% of max_steps) |
| `lr_scheduler_type` | str | linear | Learning rate scheduler (linear/cosine/polynomial/constant/constant_with_warmup) |
| `min_lr` | float | 0.0 | Minimum learning rate (cosine scheduler only) |
| `layerwise_lr_decay_bound` | float | 1.0 | Layerwise LR decay factor (0,1]. 1 means no decay. |
| `random_shuffle` | bool | True | Enable dataset shuffling |
| `num_cycles` | float | 0.5 | Cosine scheduler: number of waves |
| `lr_end` | float | 1e-7 | Polynomial scheduler: final LR |
| `power` | float | 1.0 | Polynomial scheduler: power |
| `adam_beta1` | float | 0.9 | AdamW beta1 |
| `adam_beta2` | float | 0.999 | AdamW beta2 |
| `adam_epsilon` | float | 1e-8 | AdamW epsilon |

### 1.4 Distributed Training

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `tensor_parallel_degree` | int | Required | Tensor parallelism degree |
| `tensor_parallel_config` | str | Required | Recommended: "sync_param sync_grad sync_moment" |
| `tensor_parallel_output` | bool | True | Enable parallel output for last Transformer layer to save memory |
| `pipeline_parallel_degree` | int | Required | Pipeline parallelism degree |
| `pipeline_parallel_config` | str | Required | Recommended: "disable_partial_send_recv enable_clear_every_step_cache enable_delay_scale_loss enable_overlap_p2p_comm best_unbalanced_scheduler" |
| `pp_seg_method` | str | Required | Pipeline layer segmentation method |
| `virtual_pp_degree` | int | 1 | Virtual pipeline degree (effective when pipeline_parallel_degree > 1) |
| `add_tail_layers` | int | 0 | Add EmptyLayers after DecodeLayer for virtual pipeline requirements |
| `sharding_parallel_degree` | int | Required | Sharding parallelism degree |
| `sharding_parallel_config` | str | Required | Recommended: "enable_stage1_overlap enable_release_grads" |
| `sharding` | str | Required | Sharding stage (stage1: optimizer, stage2: gradients, stage3: parameters) |
| `sequence_parallel` | bool | True | Enable sequence parallelism |
| `moe_group` | str | "dummy" | MoE communication group ("mp" for training, "dummy" for inference) |

### 1.5 Memory Optimization

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `release_grads` | bool | False | Release gradients after each iteration to reduce peak memory |
| `use_sparse_head_and_loss_fn` | bool | False | Use sparse LM Head and loss function |
| `use_fused_head_and_loss_fn` | bool | False | Fuse LM head and CrossEntropyLoss to save memory |
| `use_attn_mask_start_row_indices` | bool | True | Use sparse mask representation with start row indices |
| `recompute_use_reentrant` | bool | False | Recompute implementation (PyLayer if True, hooks if False) |
| `recompute` | bool | False | Enable gradient checkpointing |
| `recompute_granularity` | str | "full" | Recompute granularity ("full"/"full_attn"/"core_attn") |
| `offload_optim` | bool | False | Offload optimizer to CPU |

### 1.6 Checkpoint

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `save_steps` | int | Required | Checkpoint save interval (when save_strategy=="steps") |
| `save_strategy` | str | "no" | Checkpoint save strategy |
| `unified_checkpoint` | bool | True | Use unified checkpoint format |
| `unified_checkpoint_config` | str | "" | See [Unified Checkpoint](./unified_checkpoint.md) |
| `disable_ckpt_quant` | bool | False | See [Unified Checkpoint](./unified_checkpoint.md) |
| `ignore_save_lr_and_optim` | bool | False | Skip saving optimizer states |
| `ignore_load_lr_and_optim` | bool | False | Skip loading optimizer states |
| `save_total_limit` | int | None | Maximum number of checkpoints to keep |

### 1.7 Acceleration

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `use_flash_attention` | bool | True | Enable FlashAttention |
| `use_sparse_flash_attn` | bool | True | Enable FlashMask (requires `use_attn_mask_start_row_indices`) |
| `fuse_rope` | bool | False | Fuse rotary position embedding |
| `fuse_linear` | bool | False | F fuse linear operations |
| `greedy_intokens` | bool | True | Enable greedy token-based packing. <br>Instead of sequential sampling, a global buffer of samples is maintained <br>and greedily packed into sequences to maximize token utilization and minimize padding. |
| `dataloader_num_workers` | int | 1 | Dataloader subprocess count (0 to disable) |
| `distributed_dataloader` | int | 0 | Use distributed dataloader for large datasets |
| `moe_multimodal_dispatch_use_allgather` | str | v2-alltoall-unpad | Optimize MoE layer with allgather+unpad |

### 1.8 Mixed Precision (Recommended Defaults)

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `bf16` | bool | False | Enable BF16 training |
| `fp16_opt_level` | str | O1 | AMP level (O2 converts params to float16/bfloat16) |
| `scale_loss` | int | 2 ** 15 | Loss scaling factor for float16 |
| `amp_custom_white_list` | str | Required | AMP O2 whitelist (e.g., "lookup_table flash_attn matmul") |
| `amp_custom_black_list` | str | Required | AMP O2 blacklist (e.g., "reduce_sum elementwise_div") |
| `amp_master_grad` | bool | False | Maintain float32 gradients for AMP O2 |

## 2. Specialized Configurations

### 2.1 SFT Configuration

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `num_samples_each_epoch` | int | 6000000 | Virtual epoch size (recommend keeping default) |

### 2.2 LoRA

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `lora_rank` | int | 8 | LoRA rank (typical: 8/16/32. Higher improves quality but increases memory) |
| `lora_alpha` | float | -1 | LoRA scaling factor (scaling = alpha/rank or alpha/sqrt(rank) for rslora) |
| `rslora` | bool | False | Enable rslora scaling (recommended for rank ≥64) |
| `lora_plus_scale` | float | 1 | LoRA+ learning rate multiplier (recommended: 4-16) |
| `rslora_plus` | bool | False | Enhanced LoRA (improves performance but may cause forgetting) |
| `lora` | bool | False | Enable LoRA training |

### 2.3 DPO

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `beta` | float | 0.1 | DPO loss temperature |
| `simpo_gamma` | float | 0.5 | SimPO loss gamma |
| `offset_alpha` | float | 0.0 | Score-based DPO loss offset |
| `max_prompt_len` | int | 2048 | Maximum prompt length (truncated beyond max_seq_len-10) |
| `loss_type` | str | sigmoid | Preference loss type (sigmoid/ipo/kto_pair) |
| `pref_loss_ratio` | float | 1.0 | Preference loss weight |
| `sft_loss_ratio` | float | 0.0 | Chosen data SFT loss weight |
| `label_smoothing` | float | 0.0 | Label smoothing for sigmoid loss |
| `reference_free` | bool | False | Disable reference model |
| `ref_model_update_steps` | int | -1 | Reference model update interval (-1 to disable) |

### 2.4 FP8 Training

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `apply_hadamard` | bool | True | Use Hadamard transform for FP8 precision |
| `use_lowprecision_moment` | bool | False | Use BF16 optimizer momentum (recommended for FP8) |
| `tensorwise_offload_optimizer` | bool | False | Offload optimizer to reduce memory |
| `apply_online_actscale_step` | bool | 200 | Dynamic quantization scale steps |
| `optim_shard_num` | int | 1 | Split optimizer state files during saving to avoid memory OOM. Works only when `unified_checkpoint_config: ignore_merge_optimizer`. |
