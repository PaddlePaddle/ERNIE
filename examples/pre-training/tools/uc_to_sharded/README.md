English | [简体中文](./README_zh.md)

# Pretrained Weight Conversion Tool

This tool converts our released pretrained weights into formats compatible with current model.

## Download Pretrained Weights

Download the released weights from our repository. For details, please refer to: [Introduction to ERNIE 4.5](/README.md).

## Generate Model Checkpoint

Start Pre-Training and save checkpoints at any step. More information is available in the [ERNIE-4.5-300B-A47B Pre-Training](/examples/pre-training/README.md).

## Convert Pretrained Weights

`python convert_uc_to_sharded.py --org <path_to_pretrained_weights> --cur <path_to_checkpoint> --dst <path_to_converted_weights>`
