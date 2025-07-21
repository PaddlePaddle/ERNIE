# 权重转换
## 1. 环境要求

```
numpy
torch==2.6.0
safetensors==0.5.3
```
## 2. Paddle 权重转换 Torch 权重

### 2.1 ERNIE 4.5 语言模型转换

```
# ERNIE-4.5-0.3B
sh tools/paddle2torch/scripts/export_dense.sh /paddle_models/ERNIE-4.5-0.3B /torch_models/ERNIE-4.5-0.3B-torch

# ERNIE-4.5-21B-A3B 和 ERNIE-4.5-300B-A47B
sh tools/paddle2torch/scripts/export_moe.sh /paddle_models/ERNIE-4.5-21B-A3B /torch_models/ERNIE-4.5-21B-A3B-torch
```

# ERNIE-4.5-vl-28B-A3B 和 ERNIE-4.5-vl-424B-A47B

```
sh tools/paddle2torch/scripts/export_vl.sh /paddle_models/ERNIE-4.5-vl-28B-A3B /torch_models/ERNIE-4.5-vl-28B-A3B-torch
```
