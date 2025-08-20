# ERNIEKit: ERNIE Development Toolkit Based on PaddlePaddle

**ERNIEKit** is an industrial-grade development toolkit for ERNIE 4.5. It provides training and compression capabilities, including Pre-Training, Supervised Fine-Tuning (SFT), Low-Rank Adaptation (LoRA), Direct Preference Optimization (DPO), and Quantization-Aware Training (QAT) and Post-Training Quantization (PTQ) techniques. It includes practical applications and tutorials for leveraging ERNIE models.


## News

**[2025-06] 🔥 Released ERNIEKit v1.0:** We're excited to announce ERNIEKit v1.0, the most powerful and efficient toolkit yet for developing with the latest ERNIE models!

## 1. Features

* 🚀 **Industrial-grade High-Performance Pre-Training**
Optimized ERNIE 4.5 pre-training implementation featuring 3D hybrid parallelism and FP8 mixed precision acceleration. Please refer to [Pre-Training](../examples/pre-training/README.md) for more details.

* 🪙 **Low-bit Quantization-aware Fine-tuning**
 To significantly lower the barriers and costs of fine-tuning and deploying the ERNIE 4.5 model, we introduce a novel FP8 Quantization-Aware Training (QAT) methodology. This solution synergistically integrates low-precision training with optimizer offloading. Consequently, the minimum resources for fine-tuning ERNIE 4.5-300B-A47B has been substantially reduced from 96 GPUs to only 16 GPUs, while maintaining the model's original performance. Crucially, unlike prevalent FP8 mixed-precision schemes that rely on online block-wise and tile-wise quantization, the models produced by ERNIEKit's QAT solution achieve a significant advantage: they support highly efficient offline tensor-wise FP8 quantization for inference. This eliminates the computational overhead associated with dynamic quantization at inference time.
For more information, please refer to the [FP8-QAT](./fp8_qat.md) and [WINT4/8-LoRA](wint8mix_lora.md).

* 👁️ **Visual Training & Debugging Interface**
Gradio-based WebUI for zero-code fine-tuning, alignment, and inference. Please refer to [WebUI & CLI](./cli_webui_usage.md) for more details.

* **🔌 Multiple Hardware Support**
Support NVDIA GPU, [Kunlunxin XPU](./devices/README_XPU.md) and [Ascend NPU](./devices/README_NPU.md) Training.

## 2. Installation

### 2.1 Prerequisites
| Dependency | Recommended Version |
| --- | --- |
| **CUDA** |	≥ 12.3  |
| **CUDA Driver** |	≥ 535.171 |
| **nvcc**	| ≥ 12.3 |
| **gcc**	| ≥ 12.2 |
| **Python** | 	3.10 - 3.12 |
| **GPU Architecture** | Ampere/Hopper (80GB+HBM) |


### 2.2 Installing PaddlePaddle

**Docker-Based Installation (Recommended)**

To ensure environment consistency across different hardware configurations, we recommend using our pre-configured Docker images. These images include CUDA, cuDNN, and NCCL dependencies with PaddlePaddle v3.1 pre-installed:

```bash
# Choose based on your CUDA version requirements:
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.1.0-gpu-cuda12.9-cudnn9.9
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.1.0-gpu-cuda12.6-cudnn9.5
```

**Source Code Installation**

If not using Docker, ensure your environment meets the prerequisites in 2.1. ERNIEKit requires PaddlePaddle v3.1+. See official [PaddlePaddle Installation Guide](https://www.paddlepaddle.org.cn/install/quick) for details.


Verify installation with:

```bash
python -c "import paddle;paddle.utils.run_check()"
```

Successful installation shows:

```
PaddlePaddle works well on 8 GPUs.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

### 2.3 Install ERNIEKit

```bash
git clone https://github.com/PaddlePaddle/ERNIE
cd ERNIE
python -m pip install -r requirements/gpu/requirements.txt
python -m pip install -e . # We recommend install in editable mode
```

You can also build docker image yourself which includes all the dependencies listed in `requirements.txt`. Please refer to [build docker](../docker/docker-cuda/README.md) for more details.

### 2.4 Install FastDeploy

Please refer to [FastDeploy installation](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/get_started/installation/).

## 3. Model Training

## 3.1 Training Resources

ERNIEKit supports training for the following models. Before initiating training please ensure:
1. Environment setup is completed
2. Your hardware meets the minimum resource requirements

| Model                          | Post-Training Method | Seq Length | Min Resources       | Recommended Config |
|--------------------------------|----------------------|------------|---------------------|---------------------|
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | SFT         | 8K       | 96x80G A/H GPUs     | [run_sft_8k.sh](../examples/scripts/ERNIE-4.5-300B-A47B/sft/run_sft_8k.sh)|
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | SFT         | 32K      | 112x80G A/H GPUs    | [run_sft_32k.sh](../examples/scripts/ERNIE-4.5-300B-A47B/sft/run_sft_32k.sh)|
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | SFT(FP8)    |    8K   | 16x80G H GPUs + 2TB CPU RAM     | [run_sft_fp8_8k.sh](../examples/scripts/ERNIE-4.5-300B-A47B/sft/run_sft_fp8_8k.sh)|
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | SFT(FP8)    | 32K      | 16x80G H GPUs + 2TB CPU RAM      | [run_sft_fp8_32k.sh](../examples/scripts/ERNIE-4.5-300B-A47B/sft/run_sft_fp8_32k.sh)|
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | SFT-LoRA(wint4/8) | 8K       | 4x80G A/H GPUs     |[run_sft_wint8mix_lora_8k.sh](../examples/scripts/ERNIE-4.5-300B-A47B/sft/run_sft_wint8mix_lora_8k.sh) |
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | SFT-LoRA(wint4/8) | 32K      | 8x80G A/H GPUs     |[run_sft_wint8mix_lora_32k.sh](../examples/scripts/ERNIE-4.5-300B-A47B/sft/run_sft_wint8mix_lora_32k.sh) |
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | DPO         | 8K       | 112x80G A/H GPUs   | [run_dpo_8k.sh](../examples/scripts/ERNIE-4.5-300B-A47B/dpo/run_dpo_8k.sh)|
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | DPO         | 32K      | 112x80G A/H GPUs   | [run_dpo_32k.sh](../examples/scripts/ERNIE-4.5-300B-A47B/dpo/run_dpo_32k.sh)|
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | DPO-LoRA    | 8K       | 16x80G A/H GPUs    | [run_dpo_lora_8k.sh](../examples/scripts/ERNIE-4.5-300B-A47B/dpo/run_dpo_lora_8k.sh)|
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | DPO-LoRA    | 32K      | 16x80G A/H GPUs    | [run_dpo_lora_32k.sh](../examples/scripts/ERNIE-4.5-300B-A47B/dpo/run_dpo_lora_32k.sh)|
| ERNIE-4.5-21B-A3B-Base/ERNIE-4.5-21B-A3B | SFT         | 8K       | 8x80G A/H GPUs     | [run_sft_8k.sh](../examples/scripts/ERNIE-4.5-21B-A3B/sft/run_sft_8k.sh)|
| ERNIE-4.5-21B-A3B-Base/ERNIE-4.5-21B-A3B | SFT         | 32K      | 8x80G A/H GPUs     | [run_sft_32k.sh](../examples/scripts/ERNIE-4.5-21B-A3B/sft/run_sft_32k.sh)|
| ERNIE-4.5-21B-A3B-B base/ERNIE-4.5-21B-A3B | SFT         | 128K     | 8x80G A/H GPUs     | [run_sft_128k.sh](../examples/scripts/ERNIE-4.5-21B-A3B/sft/run_sft_128k.sh)|
| ERNIE-4.5-21B-A3B-Base/ERNIE-4.5-21B-A3B | SFT-LoRA(wint4/8) | 8K       | 1x80G A/H GPUs     | [run_sft_wint8mix_lora_8k.sh](../examples/scripts/ERNIE-4.5-21B-A3B/sft/run_sft_wint8mix_lora_8k.sh)|
| ERNIE-4.5-21B-A3B-Base/ERNIE-4.5-21B-A3B | SFT-LoRA(wint4/8) | 32K      | 1x80G A/H GPUs     | [run_sft_wint8mix_lora_32k.sh](../examples/scripts/ERNIE-4.5-21B-A3B/sft/run_sft_wint8mix_lora_32k.sh)|
| ERNIE-4.5-21B-A3B-Base/ERNIE-4.5-21B-A3B | DPO         | 8K       | 8x80G A/H GPUs     | [run_dpo_8k.sh](../examples/scripts/ERNIE-4.5-21B-A3B/dpo/run_dpo_8k.sh)|
| ERNIE-4.5-21B-A3B-Base/ERNIE-4.5-21B-A3B | DPO         | 32K      | 8x80G A/H GPUs     | [run_dpo_32k.sh](../examples/scripts/ERNIE-4.5-21B-A3B/dpo/run_dpo_32k.sh)|
| ERNIE-4.5-21B-A3B-Base/ERNIE-4.5-21B-A3B | DPO         | 128K     | 8x80G A/H GPUs     | [run_dpo_128k.sh](../examples/scripts/ERNIE-4.5-21B-A3B/dpo/run_dpo_128k.sh)|
| ERNIE-4.5-21B-A3B-Base/ERNIE-4.5-21B-A3B | DPO-LoRA    | 8K       | 1x80G A/H GPUs     | [run_dpo_lora_8k.sh](../examples/scripts/ERNIE-4.5-21B-A3B/dpo/run_dpo_lora_8k.sh)|
| ERNIE-4.5-21B-A3B-Base/ERNIE-4.5-21B-A3B | DPO-LoRA    | 32K      | 1x80G A/H GPUs     | [run_dpo_lora_32k.sh](../examples/scripts/ERNIE-4.5-21B-A3B/dpo/run_dpo_lora_32k.sh)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | SFT         | 8K       | 1x80G A/H GPU      | [run_sft_8k.sh](../examples/scripts/ERNIE-4.5-0.3B/sft/run_sft_8k.sh)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | SFT         | 32K      | 1x80G A/H GPU      | [run_sft_32k.sh](../examples/scripts/ERNIE-4.5-0.3B/sft/run_sft_32k.sh)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | SFT         | 128K      | 1x80G A/H GPU      | [run_sft_128k.sh](../examples/scripts/ERNIE-4.5-0.3B/sft/run_sft_128k.sh)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | SFT-LoRA(wint4/8)         | 8K      | 1x80G A/H GPU      | [run_sft_wint8mix_lora_8k.sh](../examples/scripts/ERNIE-4.5-0.3B/sft/run_sft_wint8mix_lora_8k.sh)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | SFT-LoRA(wint4/8)         | 32K      | 1x80G A/H GPU      | [run_sft_wint8mix_lora_32k.sh](../examples/scripts/ERNIE-4.5-0.3B/sft/run_sft_wint8mix_lora_32k.sh)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | DPO         | 8K       | 1x80G A/H GPU      | [run_dpo_8k.sh](../examples/scripts/ERNIE-4.5-0.3B/dpo/run_dpo_8k.sh)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | DPO         | 32K      | 1x80G A/H GPU      | [run_dpo_32k.sh](../examples/scripts/ERNIE-4.5-0.3B/dpo/run_dpo_32k.sh)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | DPO         | 128K      | 1x80G A/H GPU      | [run_dpo_128k.sh](../examples/scripts/ERNIE-4.5-0.3B/dpo/run_dpo_128k.sh)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | DPO-LoRA    | 8K       | 1x80G A/H GPU      | [run_dpo_lora_8k.sh](../examples/scripts/ERNIE-4.5-0.3B/dpo/run_dpo_lora_8k.sh)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | DPO-LoRA    | 32K      | 1x80G A/H GPU      | [run_dpo_lora_32k.sh](../examples/scripts/ERNIE-4.5-0.3B/dpo/run_dpo_lora_32k.sh)|


### 3.2 Data Preparation

ERNIEKit supports both `alpaca` and `erniekit` dataset formats. For detailed format specifications, refer to [Dataset Guide](./datasets.md).

We provide sample datasets in `erniekit` format for quick start, please refer to [Demo Datasets](../examples/data/) .

Subsequent sections will demonstrate workflows using these sample datasets.

### 3.3 Supervised Fine-tuning

Supervised Fine-Tuning (SFT) adapts pre-trained language models using labeled datasets to enhance task-specific performance and instruction-following capabilities. This parameter-updating method:
- Requires high-quality annotated data
- Adjusts all model parameters
- Ideal for precision-critical specialized tasks

For configuration details:
⚙️ [General Training Settings](./training_eval_args.md#1-General-configuration)
⚙️ [SFT Settings](./training_eval_args.md#21-SFT)

**Example 1: Full-Parameter Supervised Fine-tuning**

The following example requires training on a single 80G A/H GPU machine.

```bash
# download model from huggingface
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# 8K Sequence Length, SFT
erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_sft_8k.yaml
```

```bash
# download model from huggingface
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# 32K Sequence Length, SFT
erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_sft_32k.yaml
```

**Example 2: Parameter Efficient Fine-tuning**

LoRA (Low-Rank Adaptation) leverages matrix low-rank decomposition techniques to achieve model fine-tuning by only adjusting a small number of new parameters. LoRA training reduces resource requirements while often delivering comparable or even superior performance to full-parameter fine-tuning on small datasets.

Compared to standard SFT, enabling LoRA training simply requires adding `fine_tuning: LoRA` to the training configuration. For more training parameters, refer to [LoRA configurations](./training_eval_args.md#22-LoRA).

The following example requires training on a single 80GB A/H GPU card.


```bash
# download model from huggingface
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# 8K Sequence Length, SFT-LoRA
erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_sft_lora_8k.yaml
```
**Viewing Training Logs**

If your script specifies the `logging_dir` argument, we save VisualDL visualization results to that directory. Otherwise, results are stored at the path specified by `output_dir`.

Start VisualDL with the following command to view training logs:

```bash
visualdl --logdir ${YOUR_LOG_DIR} --host ${HOST_IP} --port ${PORT}
```

### 3.4 DPO

Alignment Training is a crucial technique for ensuring the behavior of Large Language Models (LLMs) aligns with human intentions, values, or specific objectives. Its core goal is to address the issue of pretrained models being "powerful but uncontrollable," making model outputs safer, more reliable, and better aligned with human expectations.

Direct Preference Optimization (DPO) is a representative method for achieving human preference alignment. It directly fine-tunes model parameters on annotated preference data. Compared to RLHF, DPO offers higher training stability and lower computational overhead, establishing itself as a mainstream preference alignment approach.

For more training configurations, refer to [Training configuration](./training_eval_args.md#1-General-configuration) and [DPO configuration](./training_eval_args.md#23-DPO).


**Example 1: Full-Parameter Direct Preference Optimization**

The following example requires training on a single 80G A/H GPU machine.

```bash
# download model from huggingface
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# 8K Sequence Length, DPO
erniekit train examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_8k.yaml
```

```bash
# download model from huggingface
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# 32K Sequence Length, DPO
erniekit train examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_32k.yaml
```

**Example 2: Direct Preference Optimization with LoRA**

The following example requires training on a single 80G A/H GPU machine.

```bash
# download model from huggingface
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# 8K Sequence Length, DPO-LoRA
erniekit train examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_lora_8k.yaml
```

### 3.5 Weight Merging

After LoRA fine-tuning, merge LoRA weights with the main model weights. In multi-machine training scenarios:
⚠️ Each machine stores partial model parameters (checkpoint)
⚠️ **Must synchronize parameter files** across all machines before merging LoRA weights or deployment

```bash
path_to_checkpoints/
    ├── added_tokens.json
    ├── config.json
    ├── model-00001-of-00xxx.safetensors
    ├── model-00002-of-00xxx.safetensors
    ├── ...
    ├── model-00xxx-of-00xxx.safetensors
    ├── model.safetensors.index.json
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.model
```

To merge LoRA parameters into the base model after training:

```bash
erniekit export examples/configs/ERNIE-4.5-0.3B/run_export.yaml lora=True
```

## 4. Model Deployment

Trained ERNIEKit weights can be directly deployed using FastDeploy through integrated CLI tools. Below is an example for ERNIE-4.5-0.3B:

```bash
# download model from huggingface
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
erniekit server examples/configs/ERNIE-4.5-0.3B/run_chat.yaml
erniekit chat examples/configs/ERNIE-4.5-0.3B/run_chat.yaml
```
