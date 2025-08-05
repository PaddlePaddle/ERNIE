# Chat Configuration

This document provides the complete reference for chat model serving and request parameters in ERNIEKit. It covers:

- Model serving deployment configurations (GPU memory, ports, parallel settings)
- Chat request parameters (generation controls, penalties, streaming options)

Each parameter is documented with its type, default value and detailed description to help developers properly configure their chat applications.

## 1. Model Serving Deployment Configuration

## 1. Model Serving Deployment Configuration

| Parameter Name | Type | Default Value | Description |
| --- | --- | --- | --- |
| `model_name_or_path` | str | None | Model name or path |
| `tensor_parallel_degree` | int | 1 | Number of GPUs for tensor parallel configuration |
| `output_dir` | str | Required | Automatically loads the latest checkpoint from this path for model serving (LoRA checkpoints not supported). If no checkpoint exists, loads `model_name_or_path` instead. |
| `host` | str | 127.0.0.1 | IP address for model serving |
| `port` | int | 8188 | Port number for model serving |
| `metrics_port` | int | 8001 | Port number for serving metrics monitoring |
| `engine_worker_queue_port` | int | 8002 | Port for inter-process communication within the engine |
| `max_model_len` | int | 2048 | Maximum sequence length (input + output) during serving |
| `max_num_seqs` | int | 8 | Maximum batch size during decode phase. Requests exceeding this will be queued. |
| `use_warmup` | int | 0 | Whether to perform warmup on startup. Generates maximum-length data for warmup (used by default in KV Cache calculation). |
| `gpu_memory_utilization` | float | 0.9 | GPU memory utilization rate |
| `quantization` | str | None | Model quantization strategy, when loading BF16 CKPT, specifying wint4 or wint8 supports lossless online 4bit/8bit quantization |
| `block_size` | int | 64 | Number of tokens per cache management block |
| `kv_cache_ratio` | float | 0.75 | Ratio of KV Cache allocated to input. Recommended value = average input length/(average input length + average output length) |

* Note: The optimal configuration for model deployment can be referred: https://github.com/PaddlePaddle/FastDeploy/tree/develop/docs/zh/optimal_deployment

## 2. Chat Request Configuration

| Parameter Name | Type | Default Value | Description |
| --- | --- | --- | --- |
| `port` | int | 8188 | Request port number |
| `max_new_tokens` | int | 1024 | Maximum number of tokens to generate |
| `min_tokens` | int | 0 | Minimum number of tokens to generate |
| `temperature` | float | 0.95 | Controls randomness in output: higher values produce more creative/random text, lower values make output more deterministic |
| `top_p` | float | 0.7 | During generation, dynamically selects tokens with cumulative probability ≥ top_p. Higher values increase diversity. |
| `frequency_penalty` | float | 0.0 | `>0.0`: Penalizes new tokens based on their existing frequency in the text, reducing repetition |
| `presence_penalty` | float | 0.0 | `>0.0`: Penalizes tokens already present in the text, increasing likelihood of discussing new topics |
| `repetition_penalty` | float | 1.0 | Controls repetition in generated text. Higher values reduce repetitiveness |
| `stream` | bool | True | Whether to enable streaming output. If `True`, returns tokens incrementally; if `False`, returns complete text at once |
| `stream_options` | StreamOptions | None | Configuration options for customizing streaming behavior |
