[English](README.md) | 简体中文

# ERNIE-4.5 自动并行预训练
本文档介绍如何使用自动并行进行 ERNIE-4.5 预训练，运行训练至少需要56卡NVIDIA H800 80G。

## 自动并行
自动并行提供了对分布式计算任务的统一抽象，可以仅用单卡组网+简单的分布式配置就可以实现大规模的大模型分布式训练任务，极大地降低了分布式训练的开发门槛。具体可以参考 [自动并行](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/paddle_v3_features/auto_parallel_cn.html#zidongbingxingxunlian)。

## 数据准备
本repo为您准备了demo数据集以方便您进行测试，demo数据放在了 `examples/pre-training/demo_data` 路径下。如果您想使用其他数据集或使用自定义数据集，
请参考 [Pretrain 数据集](https://paddlenlp.readthedocs.io/zh/latest/llm/dataset.html) 中的内容。

## 镜像准备
您的机器需要安装CUDA驱动（>= 525.60.13），并安装 CUDA toolkit 12.9。您可以使用镜像 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.2.0-gpu-cuda12.9-cudnn9.9` 来进行预训练任务，同时
请确保您的集群中有 `mpi` 环境。

## 环境准备
`mpirun python -m pip install -r requirements.txt --force-reinstall`

注意：paddlepaddle-gpu 需要使用 3.2 版本，安装可使用[参考](https://www.paddlepaddle.org.cn/install/quick?docurl=undefined)

## 开始训练
在准备好环境后。您可以通过执行以下命令来进行56卡预训练：
`mpirun bash train_4p5_300B_A47B.sh`，

- 注意，您需要将 `train_4p5_300B_A47B.sh` 中的 `master_ip` 与 `port` 根据您的环境进行替换。

该工具包提供了使用自动并行完成 ERNIE-4.5 预训练的方法，包括多维混合并行训练策略，更多的优化点和功能会基于此版本持续更新。
