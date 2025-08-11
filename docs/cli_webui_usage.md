# CLI / WebUI

## Overview

CLI (Command Line Interface) provides terminal-based interaction with the program, enabling efficient and flexible execution of model training, inference, and evaluation tasks through parameterized configurations.

WebUI (Web User Interface) offers a browser-based visual interface that allows users to perform model training, chatting, and deployment without coding or complex commands, making it ideal for non-technical users and rapid prototyping.

## Features
This document details the usage of CLI tools and WebUI in the ERNIE model toolkit, covering core functionalities:

- 📈 Model Fine-tuning: SFT/LoRA/DPO fine-tuning with built-in/custom datasets
- 🗣️ Chat Interaction: Load models for multi-turn conversation testing
- 📊 Performance Evaluation: Validate models on built-in/custom datasets
- 📁 Model Export: Convert trained models to deployable formats

Whether you're a developer seeking script-based customization or prefer graphical interfaces for quick experimentation, both approaches are supported.

## Quick Start

**Installation**

Run in the erniekit root directory:
```bash
python -m pip install -e .
```

Verify installation:
```bash
erniekit help
```

Expected output:
```
------------------------------------------------------------
| Usage:                                                     |
|   erniekit train -h: model finetuning                      |
|   erniekit export -h: model export                         |
|   erniekit split -h: model split                           |
|   erniekit eval -h: model evaluation                       |
|   erniekit server -h: model deployment                     |
|   erniekit chat -h: launch a chat interface in CLI         |
|   erniekit webui -h: launch webui                          |
|   erniekit version: show version info                      |
|   erniekit help: show helping info                         |
------------------------------------------------------------
```

**GPU Configuration**

By default, all available gpus are used in CLI/WebUI.
If you wan to specify certain gpus, please set CUDA_VISIBLE_DEVICES before running CLI/WebUI:

```bash
# Single GPU
export CUDA_VISIBLE_DEVICES=0
# Multi GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Single XPU
export XPU_VISIBLE_DEVICES=0
# Multi XPUs
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Single NPU
export ASCEND_RT_VISIBLE_DEVICES=0
# Multi NPUs
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

* Note: In `Chat` module, the number of gpus configured by CUDA_VISIBLE_DEVICES should be equal to `tensor_parallel_degree` in the config.
Alternatively, you can also unset CUDA_VISIBLE_DEVICES.

# 1. CLI Usage

Examples using **ERNIE-4.5-0.3B** model:

## 1.1. Chat
```bash
# download model from huggingface
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# Load model and start service
erniekit server examples/configs/ERNIE-4.5-0.3B/run_chat.yaml
# Launch CLI chat interface
erniekit chat examples/configs/ERNIE-4.5-0.3B/run_chat.yaml
```

* Note: the command-line dialogue for VL-model only supports pure text input.

## 1.2. Model Fine-tuning

### 1.2.1. SFT & LoRA Fine-tuning
```bash
# download model from huggingface
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# Example 1: 8K seq length, SFT
erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_sft_8k.yaml
# Example 2: 32K seq length, SFT
erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_sft_32k.yaml
# Example 3: 8K seq length, SFT-LoRA
erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_sft_lora_8k.yaml
# Example 4: 32K seq length, SFT-LoRA
erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_sft_lora_32k.yaml
```

### 1.2.2. DPO & LoRA Fine-tuning
```bash
# download model from huggingface
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# Example 1: 8K seq length, DPO
erniekit train examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_8k.yaml
# Example 2: 32K seq length, DPO
erniekit train examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_32k.yaml
# Example 3: 8K seq length, DPO-LoRA
erniekit train examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_lora_8k.yaml
# Example 4: 32K seq length, DPO-LoRA
erniekit train examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_lora_32k.yaml
```

## 1.3. Model Evaluation
```bash
erniekit eval examples/configs/ERNIE-4.5-0.3B/run_eval.yaml
```

## 1.4. Model Export
```bash
erniekit export examples/configs/ERNIE-4.5-0.3B/run_export.yaml
```

## 1.5. Multi-Node Training
```bash
NNODES={num_nodes} MASTER_ADDR={your_master_addr} MASTER_PORT={your_master_port} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 erniekit train examples/configs/ERNIE-4.5-300B-A47B/sft/run_sft_lora_8k.yaml
```

# 2. WebUI Examples

Launch WebUI:
```bash
erniekit webui
# Specify port: GRADIO_SERVER_PORT=8080 erniekit webui
```

WebUI contains five modules: Basic Info, Training, Chat, Evaluation, and Export.

## 2.1. Basic Info

### 2.1.1 Model
Default model name is "Customization". Custom models support local paths (relative/absolute).

### 2.1.2 Export Directory
If empty, training will auto-generate paths like `./output/ERNIE-4.5-0.3B_SFT_LoRA_2025_06_29_12_03_36`. Evaluation/chat/export default to `./output`.

### 2.1.3 Available GPUs
Displays GPU count (read-only).

### 2.1.4 Training Method

| WebUI Param | Variable | Description |
|--------|-------|-------|
| Fine-tuning | fine_tuning | LoRA or Full-parameter |
| Compute Type | compute_type | bf16, fp16, fp8 (NVIDIA H-series only), wint8, wint4/8 |
| AMP Master Grad | amp_master_grad | For AMP O2, uses fp32 weight gradients (default: keep unchanged) |
| Disable CKPT Quant | disable_ckpt_quant | Disables weight quantization |
| LoRA Rank | lora_rank | LoRA rank dimension |
| LoRA Alpha | lora_alpha | LoRA scaling factor |
| LoRA+ Scale | lora_plus_scale | LoRA B scale in LoRA+ |
| RSLoRA | rslora | Enable RSLoRA |

### 2.1.5 Distributed Parameters

| WebUI Param | Variable | Description |
|--------|-------|-------|
| Tensor Parallel | tensor_parallel_degree | Tensor parallelism degree |
| Pipeline Parallel | pipeline_parallel_degree | Pipeline parallelism degree |
| Sharding Parallel | sharding_parallel_degree | Sharding parallelism degree |
| Pipeline Config | pipeline_parallel_config | Recommended: "disable_partial_send_recv enable_clear_every_step_cache enable_delay_scale_loss enable_overlap_p2p_comm best_unbalanced_scheduler" |
| PP Seg Method | pp_seg_method | Pipeline layer segmentation |
| Sharding | sharding | Sharding stage: stage1 (optimizer), stage2 (gradients), stage3 (model) |
| Use SP Callback | use_sp_callback | Skips redundant gradient calculations |
| MoE Group | moe_group | MoE communication group ("mp" or "dummy") |

<div align="center">
<img src="https://github.com/user-attachments/assets/157ef9af-6741-4ce3-8c2e-2f8004dab170">
</div>

## 2.2. Training Module

Default SFT/DPO configurations for ERNIE-4.5-0.3B-Paddle are provided under "Switch SFT/DPO Presets".

After setting dataset paths/probabilities, click "Preview Dataset" for visualization. Click "Preview" to show configurations, "Start" to begin training, and "Stop" to interrupt.

### 2.2.1 Data Parameters

| WebUI Param | Variable | Description |
|--------|-------|-------|
| Max Sequence Length | max_seq_len | Token limit (adjust lower with larger GBS to avoid OOM) |
| Max Prompt Length | max_prompt_len | For DPO (max: max_seq_len-10) |
| Virtual Epoch Size | num_samples_each_epoch | Recommended default |
| Recompute | recompute | Gradient checkpointing to save memory |
| Training Epochs | num_train_epochs | Overridden by max_steps if both set |
| Max Steps | max_steps | Total training steps |
| Batch Size | batch_size | Micro batch size |
| Gradient Accumulation | gradient_accumulation_steps | Steps for gradient accumulation |

### 2.2.2 Training Dataset
Choose built-in (demo/HuggingFace) or custom datasets (mixed by probability):

| WebUI Param | Variable | Description |
|--------|-------|-------|
| Dataset Path | train_dataset_path | Training dataset path |
| Dataset Probability | train_dataset_prob | Sampling probability |
| Data Type | train_dataset_type | Supported: erniekit, alpaca |

### 2.2.3 Evaluation Dataset
Same options as training dataset:

| WebUI Param | Variable | Description |
|--------|-------|-------|
| Dataset Path | eval_dataset_path | Evaluation dataset path |
| Dataset Probability | eval_dataset_prob | Sampling probability |
| Data Type | eval_dataset_type | Supported: erniekit, alpaca |

### 2.2.4 Dataloader

| WebUI Param | Variable | Description |
|--------|-------|-------|
| Workers | dataloader_num_workers | Subprocess count (0 to disable) |
| Distributed | distributed_dataloader | Saves memory for large datasets |

### 2.2.5 Optimizer

| WebUI Param | Variable | Description |
|--------|-------|-------|
| LR Scheduler | lr_scheduler_type | linear/cosine/polynomial/constant/constant_with_warmup |
| Learning Rate | learning_rate | Suggested: 3e-5 (SFT), 1e-6 (DPO), 3e-4 (SFT-LoRA), 1e-5 (DPO-LoRA) |
| Min LR | min_lr | For cosine scheduler only |
| Layerwise Decay | layerwise_lr_decay_bound | (0, 1], 1=no decay |
| Warmup Steps | warmup_steps | Typically 1-10% of max_steps |
| Optimizer | optim | Default: adamw |
| Offload Optim | offload_optim | Offload to CPU |
| Release Grads | release_grads | Reduces peak memory (recommended: True) |
| Loss Scaling | scale_loss | For float16 training |
| Weight Decay | weight_decay | AdamW parameter |
| Adam Epsilon | adam_epsilon | AdamW parameter |
| Adam Beta1 | adam_beta1 | AdamW parameter |
| Adam Beta2 | adam_beta2 | AdamW parameter |

### 2.2.6 Output

| WebUI Param | Variable | Description |
|--------|-------|-------|
| Logging Steps | logging_steps | Log interval |
| Eval Steps | eval_steps | Evaluation interval |
| Eval Strategy | evaluation_strategy | "steps" enables periodic evaluation |
| Save Steps | save_steps | Checkpoint interval (when save_strategy=="steps") |
| Save Strategy | save_strategy | Checkpoint saving method |
| Save Limit | save_total_limit | Max checkpoints to keep |

<div align="center">
<img src="https://github.com/user-attachments/assets/43964682-b5da-46d3-b065-8318ed8a66be">
</div>

## 2.3. Chat Module

Load models from Basic Info section. Click "Verify Model Loading" to check status, and "Unload" to release models.

*Note: Full-parameter checkpoints in `output_dir` take priority for deployment.

After successful loading:
- Enter prompts in the input box
- Set roles/system prompts
- Click "Submit" to start chatting
- View history in "Chat History"
- "Clear" resets conversation
- "Stop" interrupts generation

| WebUI Param | Variable | Description |
|--------|-------|-------|
| Max Length | max_model_len | Input+output token limit |
| Port | port | Service port |
| Max New Tokens | max_new_tokens | Generation limit |
| Top-p | top_p | Nucleus sampling (higher=more diverse) |
| Temperature | temperature | Controls randomness (higher=more creative) |

<div align="center">
<img src="https://github.com/user-attachments/assets/008dd8be-5ac2-4a12-ba1e-16ce6a02f1c7">
</div>

## 2.4. Evaluation Module

Select model in Basic Info (latest checkpoint in export dir used by default).

Choose evaluation dataset (built-in/custom). Click "Preview Eval Dataset" for visualization.

"Preview Command" shows configurations. "Start" begins evaluation, "Stop" interrupts.

| WebUI Param | Variable | Description |
|--------|-------|-------|
| Dataset Path | eval_dataset_path | Evaluation dataset path |
| Dataset Probability | eval_dataset_prob | Sampling probability |
| Data Type | eval_dataset_type | Supported: erniekit, alpaca |

<div align="center">
<img src="https://github.com/user-attachments/assets/e7ae0388-4594-43d3-b2f1-75533a7951c8">
</div>

## 2.5. Export Module

Two functions:
1. LoRA weight merging
2. Model weight splitting (safetensors format only)

**LoRA Merging**
Set export directory to training output dir. Click "Start Merge LoRA Weights" to merge into original model (saved in `export_dir/export`).

**Weight Splitting**
For large safetensors files, click "Start Split Model" to split weights (saved in `export_dir/split_export`).

| WebUI Param | Variable | Description |
|--------|-------|-------|
| Max Shard Size (GB) | max_shard_size | Split file size limit |

<div align="center">
<img src="https://github.com/user-attachments/assets/67353c8c-eca6-4aa8-b912-0064c9d41556">
</div>
