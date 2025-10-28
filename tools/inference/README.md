# Dynamic Graph Inference

This document is intended for quickly experiencing dynamic graph inference with the model. For better deployment and performance, please use **FastDeploy**.

## 1. Data Format

### 1.1 Plain Text Data Format

We provide demo data in plain text format, located in the ./data directory. You can use this demo data for quick testing or perform inference with your own data.

- Each line must be a List[Dict], where each dictionary contains two required keys: `role` and `content`.

- Multi-turn conversations must strictly follow the alternating order: user → assistant.

- role: Indicates the role in the conversation. Supported types include:
  - system: system-level setup or instructions
  - user: user input
  - assistant: AI-generated reply

- content: The plain text corresponding to each role.

Example:
```json
[
    {"role": "system", "content": "你是文心一言"},
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好，我是文心一言，很高兴能为你服务。"},
    {"role": "user", "content": "写一个300字的小说大纲，内容是李白穿越到现代，最后成为公司文职人员的故事。"}
]
```

## 1.2 Multimodal Data Format

We provide demo data for multimodal tasks, which can be found at data/multimodal-query-answers-list-small.jsonl.

The prefix field controls whether the model enters "thinking" mode:

- `<think>\n\n</think>\n\n`: disables thinking mode (default)

- `<think>`: enables thinking mode

## 2. Inference

### 2.1 Plain Text Model Inference

To perform WINT8 dynamic graph inference, run:

```bash
bash ./scripts/infer.sh
```

To perform FP8 dynamic graph inference, run:

```bash
bash ./scripts/infer_fp8.sh
```

### 2.2 Multimodal Model Inference

Currently, dynamic graph inference for multimodal models only supports single-GPU inference with the 28B model:
```bash
# Single-GPU dynamic graph inference
bash examples/inference/scripts/infer_vl.sh
```
