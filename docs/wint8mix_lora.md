# WINT8MIX LoRA

## Introduction

To enable developers to efficiently train ERNIE 4.5 models, particularly the 300B model, with minimal resource requirements, we introduce **WINT8MIX LoRA**, a parameter-efficient fine-tuning method that combines Weight Only INT8 quantization with LoRA. This approach dramatically reduces LoRA training resource requirements, enabling 8K sequence training of the 300B model on just 4×80G GPUs, and 32K sequence training on 8×80G GPUs.

## Method

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique for large language models. Instead of updating all model parameters, LoRA freezes the original weights and introduces small, trainable low-rank matrices into each transformer layer.

**WINT8MIX LoRA** enhances traditional LoRA by quantizing base model weights to INT4/INT8 precision, significantly reducing memory usage during fine-tuning. Based on the sparse characteristics of MoE models, we quantize the linear weights in expert modules to INT4, while quantizing the linear weights in other components to INT8.

The Weight Only Quantization (WINT) approach uses **channelwise quantization** and specialized **acceleration operators** to boost training speed with no accuracy degradation.

<p align="center">
  <img src="https://github.com/user-attachments/assets/918f0029-48f1-4743-9ec0-e4b542108a84" width="600px"></a>
</p>
