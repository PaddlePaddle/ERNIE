English | [简体中文](./README_zh.md)

# Pretraining Data Conversion Tool

This tool converts plain-text datasets into indexed binary files suitable for model pretraining.

## Prerequisites: Model Weights

Download the released weights from our repository. For details, please refer to: [Introduction to ERNIE 4.5](/README.md).

Note: Tokenizers differ across models, so the converted dataset is model-dependent.

## Prepare the Text Dataset

Download or make your own dataset in `jsonl` format. Each line must be a JSON object containing a `"text"` field with the document content. For example:

```json
{"text": "An Open-Source Deep Learning Platform Originated from Industrial Practice..."}
{"text": "PaddlePaddle is dedicated to facilitating innovations and applications of deep learning..."}
...
```

## Generate the Pretraining Dataset

```bash
python -u create_pretraining_data.py \
    --model_name "/path/to/your/ERNIE-4.5-21B-A3B-Base-Paddle" \
    --data_format "JSON" \
    --input_path "/path/to/your/text/dataset.jsonl" \
    --append_eos \
    --output_prefix "./pretrain_data"  \
    --workers 1 \
    --log_interval 10000 \
    --data_impl "mmap"
```

The output is saved as `./pretrain_data.bin` and `./pretrain_data.idx`.
