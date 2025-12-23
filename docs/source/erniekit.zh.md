# ERNIEKit：基于飞桨（PaddlePaddle）的ERNIE开发工具包

**ERNIEKit** 是 ERNIE 4.5 的工业级开发工具包。它提供训练和压缩功能，包括预训练（Pre-Training）、监督微调（Supervised Fine-Tuning, SFT）、低秩适应（Low-Rank Adaptation, LoRA）、直接偏好优化（Direct Preference Optimization, DPO）以及量化感知训练（Quantization-Aware Training, QAT）和训练后量化（Post-Training Quantization, PTQ）技术。它包含利用 ERNIE 模型的实际应用和教程。

## 1. 特性

* 🚀 **工业级高性能预训练**
优化的 ERNIE 4.5 预训练实现，具有 3D 混合并行和 FP8 混合精度加速。更多详情请参考[预训练](../examples/pre-training/README.md)。

* 🪙 **低比特量化感知微调**
为了显著降低 ERNIE 4.5 模型微调和部署的门槛和成本，我们引入了一种新的 FP8 量化感知训练（QAT）方法。该解决方案协同整合了低精度训练与优化器卸载。因此，微调 ERNIE 4.5-300B-A47B 的最低资源需求已从 96 个 GPU 大幅降低至仅 16 个 GPU，同时保持模型的原始性能。最重要的是，与依赖在线逐块和逐瓦片量化的主流 FP8 混合精度方案不同，ERNIEKit 的 QAT 解决方案生成的模型实现了显著优势：它们支持高效的离线张量级 FP8 量化推理。这消除了推理时动态量化相关的计算开销。
更多信息请参考 [FP8-QAT](./fp8_qat.md) 和 [WINT4/8-LoRA](wint8mix_lora.md).

* 👁️ **可视化训练与调试界面**
基于 Gradio 的网页界面（WebUI），无需编写代码即可进行微调、对齐和推理。更多详情请参考 [WebUI & CLI](./cli_webui_usage.md) 

* **🔌 多硬件支持**
支持英伟达（NVIDIA）GPU, [Kunlunxin XPU](./devices/README_XPU.md) 和 [Ascend NPU](./devices/README_NPU.md) 训练.

## 2. 下载

### 2.1 前置条件
| 依赖项 | 推荐版本 |
| --- | --- |
| **CUDA** |	≥ 12.3  |
| **CUDA Driver** |	≥ 535.171 |
| **nvcc**	| ≥ 12.3 |
| **gcc**	| ≥ 12.2 |
| **Python** | 	3.10 - 3.12 |
| **GPU Architecture** | Ampere/Hopper (80GB+HBM) |


### 2.2 安装飞桨

**基于Docker的安装 (推荐)**

为了确保不同硬件配置之间的环境一致性，我们建议使用预配置的 Docker 镜像。这些镜像包含 CUDA、cuDNN 和 NCCL 依赖项，并预装了飞桨 v3.2：

```bash
# 根据您的CUDA版本要求选择:
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.2.0-gpu-cuda12.9-cudnn9.9
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.2.0-gpu-cuda12.6-cudnn9.5
```

**源码安装**

如果不使用 Docker，请确保您的环境满足 2.1 中的前置条件。ERNIEKit 需要飞桨 v3.2 或更高版本。详情请参阅官方[飞桨安装指南](https://www.paddlepaddle.org.cn/install/quick)。


使用以下命令验证安装：

```bash
python -c "import paddle;paddle.utils.run_check()"
```

安装成功显示:

```
PaddlePaddle works well on 8 GPUs.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

### 2.3 安装ERNIEKit

```bash
git clone https://github.com/PaddlePaddle/ERNIE
cd ERNIE
python -m pip install -r requirements/gpu/requirements.txt
python -m pip install -e . # 我们推荐在可编辑模式安装
```

您也可以自己构建 Docker 镜像，其中包含 `requirements.txt` 中列出的所有依赖项。更多详情请参考 [build docker](../docker/docker-cuda/README.md)。

### 2.4 安装 FastDeploy

请参考 [FastDeploy installation](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/get_started/installation/).

## 3. 模型训练

## 3.1 训练资源

ERNIEKit 支持以下模型的训练。在开始训练之前，请确保：
1. 已完成环境设置
2. 您的硬件满足最低资源要求

| 模型                          |多模态模型 | Post-Training 方法 | 序列长度 | 最低资源       | 推荐配置 |
|--------------------------------|-----------------|----------------------|------------|---------------------|---------------------|
| ERNIE-4.5-VL-424B-A47B-Base/ERNIE-4.5-VL-424B-A47B | ✅ | SFT-LORA         | 8K       | 16x80G A/H GPUs     | [run_sft_lora_8k.yaml](../examples/configs/ERNIE-4.5-VL-424B-A47B/sft/run_sft_lora_8k.yaml)|
| ERNIE-4.5-VL-424B-A47B-Base/ERNIE-4.5-VL-424B-A47B | ✅ | SFT-LORA         | 32K       | 16x80G A/H GPUs     | [run_sft_lora_32k.yaml](../examples/configs/ERNIE-4.5-VL-424B-A47B/sft/run_sft_lora_32k.yaml)|
| ERNIE-4.5-VL-424B-A47B-Base/ERNIE-4.5-VL-424B-A47B | ✅ | SFT-LORA(wint4/8)         | 8K       | 8x80G A/H GPUs     | [run_sft_wint8mix_lora_8k.yaml](../examples/configs/ERNIE-4.5-VL-424B-A47B/sft/run_sft_wint8mix_lora_8k.yaml)|
| ERNIE-4.5-VL-424B-A47B-Base/ERNIE-4.5-VL-424B-A47B | ✅ | SFT-LORA(wint4/8)         | 32K       | 8x80G A/H GPUs     | [run_sft_wint8mix_lora_32k.yaml](../examples/configs/ERNIE-4.5-VL-424B-A47B/sft/run_sft_wint8mix_lora_32k.yaml)|
| ERNIE-4.5-VL-424B-A47B-Base/ERNIE-4.5-VL-424B-A47B | ✅ | SFT-LORA(wint4/8)         | 128K       | 16x80G A/H GPUs     | [run_sft_wint8mix_lora_128k.yaml](../examples/configs/ERNIE-4.5-VL-424B-A47B/sft/run_sft_wint8mix_lora_128k.yaml)|
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | ❌ | SFT         | 8K       | 96x80G A/H GPUs     | [run_sft_8k.yaml](../examples/configs/ERNIE-4.5-300B-A47B/sft/run_sft_8k.yaml)|
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | ❌ | SFT         | 32K      | 112x80G A/H GPUs    | [run_sft_32k.yaml](../examples/configs/ERNIE-4.5-300B-A47B/sft/run_sft_32k.yaml)|
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | ❌ | SFT(FP8)    |    8K   | 16x80G H GPUs + 2TB CPU RAM     | [run_sft_fp8_8k.yaml](../examples/configs/ERNIE-4.5-300B-A47B/sft/run_sft_fp8_8k.yaml)|
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | ❌ | SFT(FP8)    | 32K      | 16x80G H GPUs + 2TB CPU RAM      | [run_sft_fp8_32k.yaml](../examples/configs/ERNIE-4.5-300B-A47B/sft/run_sft_fp8_32k.yaml)|
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | ❌ | SFT-LoRA(wint4/8) | 8K       | 4x80G A/H GPUs     |[run_sft_wint8mix_lora_8k.yaml](../examples/configs/ERNIE-4.5-300B-A47B/sft/run_sft_wint8mix_lora_8k.yaml) |
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | ❌ | SFT-LoRA(wint4/8) | 32K      | 8x80G A/H GPUs     |[run_sft_wint8mix_lora_32k.yaml](../examples/configs/ERNIE-4.5-300B-A47B/sft/run_sft_wint8mix_lora_32k.yaml) |
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | ❌ | DPO         | 8K       | 112x80G A/H GPUs   | [run_dpo_8k.yaml](../examples/configs/ERNIE-4.5-300B-A47B/dpo/run_dpo_8k.yaml)|
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | ❌ | DPO         | 32K      | 112x80G A/H GPUs   | [run_dpo_32k.yaml](../examples/configs/ERNIE-4.5-300B-A47B/dpo/run_dpo_32k.yaml)|
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | ❌ | DPO-LoRA    | 8K       | 16x80G A/H GPUs    | [run_dpo_lora_8k.yaml](../examples/configs/ERNIE-4.5-300B-A47B/dpo/run_dpo_lora_8k.yaml)|
| ERNIE-4.5-300B-A47B-Base/ERNIE-4.5-300B-A47B | ❌ | DPO-LoRA    | 32K      | 16x80G A/H GPUs    | [run_dpo_lora_32k.yaml](../examples/configs/ERNIE-4.5-300B-A47B/dpo/run_dpo_lora_32k.yaml)|
| ERNIE-4.5-VL-28B-A3B-Base/ERNIE-4.5-VL-28B-A3B | ✅ | SFT         | 8K       | 8x80G A/H GPUs     | [run_sft_8k.yaml](../examples/configs/ERNIE-4.5-VL-28B-A3B/sft/run_sft_8k.yaml)|
| ERNIE-4.5-VL-28B-A3B-Base/ERNIE-4.5-VL-28B-A3B | ✅ | SFT         | 32K       | 8x80G A/H GPUs     | [run_sft_32k.yaml](../examples/configs/ERNIE-4.5-VL-28B-A3B/sft/run_sft_32k.yaml)|
| ERNIE-4.5-VL-28B-A3B-Base/ERNIE-4.5-VL-28B-A3B | ✅ | SFT         | 128K       | 8x80G A/H GPUs     | [run_sft_128k.yaml](../examples/configs/ERNIE-4.5-VL-28B-A3B/sft/run_sft_128k.yaml)|
| ERNIE-4.5-VL-28B-A3B-Base/ERNIE-4.5-VL-28B-A3B | ✅ | SFT-LoRA         | 8K       | 4x80G A/H GPUs     | [run_sft_lora_8k.yaml](../examples/configs/ERNIE-4.5-VL-28B-A3B/sft/run_sft_lora_8k.yaml)|
| ERNIE-4.5-VL-28B-A3B-Base/ERNIE-4.5-VL-28B-A3B | ✅ | SFT-LoRA         | 32K       | 4x80G A/H GPUs     | [run_sft_lora_32k.yaml](../examples/configs/ERNIE-4.5-VL-28B-A3B/sft/run_sft_lora_32k.yaml)|
| ERNIE-4.5-VL-28B-A3B-Base/ERNIE-4.5-VL-28B-A3B | ✅ | SFT-LoRA         | 128K       | 4x80G A/H GPUs     | [run_sft_lora_128k.yaml](../examples/configs/ERNIE-4.5-VL-28B-A3B/sft/run_sft_lora_128k.yaml)|
| ERNIE-4.5-21B-A3B-Base/ERNIE-4.5-21B-A3B | ❌ | SFT         | 8K       | 8x80G A/H GPUs     | [run_sft_8k.yaml](../examples/configs/ERNIE-4.5-21B-A3B/sft/run_sft_8k.yaml)|
| ERNIE-4.5-21B-A3B-Base/ERNIE-4.5-21B-A3B | ❌ | SFT         | 32K      | 8x80G A/H GPUs     | [run_sft_32k.yaml](../examples/configs/ERNIE-4.5-21B-A3B/sft/run_sft_32k.yaml)|
| ERNIE-4.5-21B-A3B-B base/ERNIE-4.5-21B-A3B | ❌ | SFT         | 128K     | 8x80G A/H GPUs     | [run_sft_128k.yaml](../examples/configs/ERNIE-4.5-21B-A3B/sft/run_sft_128k.yaml)|
| ERNIE-4.5-21B-A3B-Base/ERNIE-4.5-21B-A3B | ❌ | SFT-LoRA(wint4/8) | 8K       | 1x80G A/H GPUs     | [run_sft_wint8mix_lora_8k.yaml](../examples/configs/ERNIE-4.5-21B-A3B/sft/run_sft_wint8mix_lora_8k.yaml)|
| ERNIE-4.5-21B-A3B-Base/ERNIE-4.5-21B-A3B | ❌ | SFT-LoRA(wint4/8) | 32K      | 1x80G A/H GPUs     | [run_sft_wint8mix_lora_32k.yaml](../examples/configs/ERNIE-4.5-21B-A3B/sft/run_sft_wint8mix_lora_32k.yaml)|
| ERNIE-4.5-21B-A3B-Base/ERNIE-4.5-21B-A3B | ❌ | DPO         | 8K       | 8x80G A/H GPUs     | [run_dpo_8k.yaml](../examples/configs/ERNIE-4.5-21B-A3B/dpo/run_dpo_8k.yaml)|
| ERNIE-4.5-21B-A3B-Base/ERNIE-4.5-21B-A3B | ❌ | DPO         | 32K      | 8x80G A/H GPUs     | [run_dpo_32k.yaml](../examples/configs/ERNIE-4.5-21B-A3B/dpo/run_dpo_32k.yaml)|
| ERNIE-4.5-21B-A3B-Base/ERNIE-4.5-21B-A3B | ❌ | DPO         | 128K     | 8x80G A/H GPUs     | [run_dpo_128k.yaml](../examples/configs/ERNIE-4.5-21B-A3B/dpo/run_dpo_128k.yaml)|
| ERNIE-4.5-21B-A3B-Base/ERNIE-4.5-21B-A3B | ❌ | DPO-LoRA    | 8K       | 1x80G A/H GPUs     | [run_dpo_lora_8k.yaml](../examples/configs/ERNIE-4.5-21B-A3B/dpo/run_dpo_lora_8k.yaml)|
| ERNIE-4.5-21B-A3B-Base/ERNIE-4.5-21B-A3B | ❌ | DPO-LoRA    | 32K      | 1x80G A/H GPUs     | [run_dpo_lora_32k.yaml](../examples/configs/ERNIE-4.5-21B-A3B/dpo/run_dpo_lora_32k.yaml)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | ❌ | SFT         | 8K       | 1x80G A/H GPU      | [run_sft_8k.yaml](../examples/configs/ERNIE-4.5-0.3B/sft/run_sft_8k.yaml)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | ❌ | SFT         | 32K      | 1x80G A/H GPU      | [run_sft_32k.yaml](../examples/configs/ERNIE-4.5-0.3B/sft/run_sft_32k.yaml)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | ❌ | SFT         | 128K      | 1x80G A/H GPU      | [run_sft_128k.yaml](../examples/configs/ERNIE-4.5-0.3B/sft/run_sft_128k.yaml)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | ❌ | SFT-LoRA(wint4/8)         | 8K      | 1x80G A/H GPU      | [run_sft_wint8mix_lora_8k.yaml](../examples/configs/ERNIE-4.5-0.3B/sft/run_sft_wint8mix_lora_8k.yaml)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | ❌ | SFT-LoRA(wint4/8)         | 32K      | 1x80G A/H GPU      | [run_sft_wint8mix_lora_32k.yaml](../examples/configs/ERNIE-4.5-0.3B/sft/run_sft_wint8mix_lora_32k.yaml)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | ❌ | DPO         | 8K       | 1x80G A/H GPU      | [run_dpo_8k.yaml](../examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_8k.yaml)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | ❌ | DPO         | 32K      | 1x80G A/H GPU      | [run_dpo_32k.yaml](../examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_32k.yaml)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | ❌ | DPO         | 128K      | 1x80G A/H GPU      | [run_dpo_128k.yaml](../examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_128k.yaml)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | ❌ | DPO-LoRA    | 8K       | 1x80G A/H GPU      | [run_dpo_lora_8k.yaml](../examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_lora_8k.yaml)|
| ERNIE-4.5-0.3B-Base/ERNIE-4.5-0.3B | ❌ | DPO-LoRA    | 32K      | 1x80G A/H GPU      | [run_dpo_lora_32k.yaml](../examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_lora_32k.yaml)|


### 3.2 数据准备

ERNIEKit 支持 `alpaca` 和 `erniekit` 两种数据集格式。有关详细格式规范，请参考 [Dataset Guide](./datasets.md).

我们提供了 `erniekit` 格式的示例数据集以便快速入门，请参考 [Demo Datasets](../examples/data/) .

后续章节将使用这些示例数据集演示工作流程。

### 3.3 监督微调

监督微调（Supervised Fine-Tuning, SFT）使用标注数据集调整预训练语言模型，以增强特定任务性能和指令遵循能力。这种参数更新方法：
- 需要高质量的标注数据
- 调整所有模型参数
- 适用于对精度要求严格的专业任务

配置详情:
⚙️ [General Training Settings](./training_eval_args.md#1-General-configuration)
⚙️ [SFT Settings](./training_eval_args.md#21-SFT)

**示例1: 全参数监督微调**

以下示例需要在单台 80G A/H GPU 机器上进行训练。

```bash
# 从huggingface下载模型
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# 8K 序列长度, SFT
erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_sft_8k.yaml
```

```bash
# 从huggingface下载模型
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# 32K 序列长度, SFT
erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_sft_32k.yaml
```

**示例2: 参数高效微调**

LoRA（低秩适应，Low-Rank Adaptation）利用矩阵低秩分解技术，通过仅调整少量新参数来实现模型微调。LoRA 训练降低了资源需求，同时在小数据集上通常能提供与全参数微调相当甚至更优的性能。

与标准SFT相比, 启用 LoRA 训练只需在训练配置中添加`fine_tuning: LoRA`。更多训练参数请参考[LoRA configurations](./training_eval_args.md#22-LoRA).

以下示例需要在单张 80GB A/H GPU 卡上进行训练。


```bash
# 从huggingface下载模型
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# 8K 序列长度, SFT-LoRA
erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_sft_lora_8k.yaml
```
**查看训练日志**

如果您的脚本指定了`logging_dir` 参数, 我们会将 VisualDL 可视化结果保存到该目录。否则，结果将存储在`output_dir`指定的路径中。

使用以下命令启动 VisualDL 查看训练日志：

```bash
visualdl --logdir ${YOUR_LOG_DIR} --host ${HOST_IP} --port ${PORT}
```

### 3.4 DPO

对齐训练是确保大型语言模型（LLM）的行为与人类意图、价值观或特定目标保持一致的关键技术。其核心目标是解决预训练模型"功能强大但难以控制"的问题，使模型输出更安全、更可靠，并更符合人类期望。

直接偏好优化（Direct Preference Optimization, DPO）是实现人类偏好对齐的代表性方法。它直接在标注的偏好数据上微调模型参数。与 RLHF 相比，DPO 具有更高的训练稳定性和更低的计算开销，已成为主流的偏好对齐方法。

更多训练配置请参考 [Training configuration](./training_eval_args.md#1-General-configuration) and [DPO configuration](./training_eval_args.md#23-DPO).


**示例1: 全参数直接偏好优化**

以下示例需要在单台 80G A/H GPU 机器上进行训练。

```bash
# 从huggingface下载模型
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# 8K 序列长度, DPO
erniekit train examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_8k.yaml
```

```bash
# 从huggingface下载模型
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# 32K 序列长度, DPO
erniekit train examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_32k.yaml
```

**示例2: LoRA 直接偏好优化**

以下示例需要在单台 80G A/H GPU 机器上进行训练。

```bash
# 从huggingface下载模型
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# 8K 序列长度, DPO-LoRA
erniekit train examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_lora_8k.yaml
```

### 3.5 权重合并

在 LoRA 微调后，将 LoRA 权重与主模型权重合并。在多机训练场景中：
⚠️ 每台机器存储部分模型参数（checkpoint）
⚠️ 必须在合并 LoRA 权重或部署之前**同步所有机器的参数文件** 

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

训练后将 LoRA 参数合并到基础模型中：

```bash
erniekit export examples/configs/ERNIE-4.5-0.3B/run_export.yaml lora=True
```

## 4. Model Deployment

训练好的 ERNIEKit 权重可以通过集成的 CLI 工具，使用 FastDeploy 直接部署。下面以 ERNIE-4.5-0.3B 为例进行说明：

```bash
# 从huggingface下载模型
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
erniekit server examples/configs/ERNIE-4.5-0.3B/run_chat.yaml
erniekit chat examples/configs/ERNIE-4.5-0.3B/run_chat.yaml
```
