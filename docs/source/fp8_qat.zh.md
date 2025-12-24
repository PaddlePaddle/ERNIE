# FP8 QAT

## 介绍

为了使开发者能够高效地训练ERNIE 4.5模型，并最小化资源需求，我们开发了一种创新的FP8量化感知训练方法。我们的方法带来了两个显著的优势：

1. 训练资源减少
    * 使用仅16个Hopper 80G GPU，就能进行300B模型的SFT全参数调优——这仅占传统BF16混合精度训练所需硬件资源的 17%。
    * 保持LLM（大规模语言模型）的性能 **准确度几乎没有下降**.
2. 推理加速
    * 支持 **张量级静态W8A8 FP8推理** 无需量化校准。
    * **1.17倍加速** 相比块级动态FP8量化推理
## 方法

如下面的图所示，我们引入了Hadamard矩阵，以确保在张量级静态FP8量化感知训练（QAT）中实现稳定收敛。为了减少计算开销并支持不同的张量形状，我们使用了块对角的Hadamard矩阵，并将标准子矩阵沿对角线放置。

<p align="center">
  <img src="https://github.com/user-attachments/assets/89e2d4fd-7acf-439b-b953-d91ef6f22993" width="600px"></a>
</p>


在LLM训练中，GPU内存主要被模型参数、梯度、优化器状态和中间激活所占用。在我们的FP8量化感知训练（QAT）方法中，模型参数以FP8格式存储，而优化器的动量和梯度则使用BF16。此外，所有优化器状态都被卸载到固定内存中，这大大减少了训练过程中GPU内存的使用。

<p align="center">
  <img src="https://github.com/user-attachments/assets/7121ecb7-399e-443d-9a42-caa77e9f467c" width="600px"></a>
</p>
