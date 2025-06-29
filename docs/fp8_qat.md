# FP8 QAT

## Introduction

To enable developers to efficiently train ERNIE 4.5 models with minimal resource requirements, we developed an innovative FP8 Quantization-Aware Training method. Our approach delivers two significant advantages:

1. Training Resource Reduction
    * Enables SFT full-parameter tuning of 300B models using just 16 Hopper 80G GPUs — only **17%** of the hardware resources required for traditional BF16 mixed-precision training.
    * Maintains LLM performance with **nearly no accuracy degradation**.
2. Inference Acceleration
    * Support **tensor-wise static W8A8 FP8 inference** without the need for quantiztaion calibration.
    * Achieves **1.17x speedup** compared to block-wise dynamic FP8 quantization inference.

## Method

As shown in the figure below, we introduce a Hadamard matrix to ensure stable convergence in tensor-wise static FP8 quantization-aware training (QAT). To reduce computational overhead and support varying tensor shapes, a block-diagonal Hadamard matrix is used, with standard submatrices placed along the diagonal.

<p align="center">
  <img src="https://github.com/user-attachments/assets/89e2d4fd-7acf-439b-b953-d91ef6f22993" width="600px"></a>
</p>


In LLM training, GPU memory is primarily consumed by model parameters, gradients, optimizer states, and intermediate activations. In our FP8 quantization-aware training (QAT) approach, model parameters are stored in FP8, while optimizer moments and gradients use BF16. Furthermore, all optimizer states are offloaded to pinned memory, significantly reducing GPU memory usage during training.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7121ecb7-399e-443d-9a42-caa77e9f467c" width="600px"></a>
</p>
