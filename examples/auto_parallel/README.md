English | [简体中文](./README_zh.md)

# ERNIE-4.5-300B-A47B Pre-Training with Auto-Parallel

This document introduce how to using auto-parallel pre-train the ERNIE-4.5-300B-A47B model, the pre-training requires at least 56 NVIDIA H800 80G GPUs.

## Auto Parallel
Auto Parallel provides a unified abstraction for distributed computing tasks. With single-card modeling and simple distributed config, Auto Parallel enables large-scale distributed training for massive models, significantly lowering the development barrier for distributed training. For details, see [Auto Parallel](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/paddle_v3_features/auto_parallel_cn.html#zidongbingxingxunlian).

## Data Preparation
This repository provide a demo dataset on the path `examples/pre-training/demo_data` for quick start. If other dataset or user defined dataset are needed,
please reference this document [Pretrain dataset](https://paddlenlp.readthedocs.io/en/latest/llm/dataset.html).

## Docker Image Preparation
The CUDA driver on your machine should be ‌≥525.60.13, and the CUDA toolkit 12.9 image is needed. You can use `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.2.0-gpu-cuda12.9-cudnn9.9` for training. And `mpi` environment should be deployed on the cluster.

## Runtime Environment Preparation
`mpirun python -m pip install -r requirements.txt --force-reinstall`

Note: paddlepaddle-gpu version requirement: 3.2.0 or later. [install Paddle](https://www.paddlepaddle.org.cn/install/quick?docurl=undefined)

## Start Pre-Training
After the environment is ready, pre-training on 56 GPUs can be launched by:
`mpirun bash train_4p5_300B_A47B.sh`，

- Note that, the `master_ip` and `port` in `mpirun bash train_4p5_300B_A47B.sh`
should be replaced according to the real environment.


The toolkit provides an auto-parallel solution for ERNIE-4.5 pre-training, including the hybrid parallelism training strategy. More advanced optimizations are on the way.
