[English](./chat_args.md) | 简体中文
# 聊天配置

本文档提供了 ERNIEKit 中聊天模型服务和请求参数的完整参考。它涵盖：

- 模型服务部署配置（GPU 内存、端口、并行设置）
- 聊天请求参数（生成控制、惩罚机制、流式选项）

每个参数都附有其类型、默认值以及详细描述，以帮助开发者正确配置他们的聊天应用。

## 1. 模型服务部署配置

## 1. 模型服务部署配置

| 参数名称 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| `model_name_or_path` | str | None | Model name or path |
| `tensor_parallel_degree` | int | 1 | 用于张量并行配置的 GPU 数量 |
| `output_dir` | str | Required | 自动加载该路径下的最新检查点用于模型服务（不支持 LoRA 检查点）。如果该路径下无检查点，则加载 `model_name_or_path`。 |
| `host` | str | 127.0.0.1 | 模型服务的 IP 地址 |
| `port` | int | 8188 | 模型服务的端口号 |
| `metrics_port` | int | 8001 | 用于服务指标监控的端口号 |
| `engine_worker_queue_port` | int | 8002 | 引擎内部进程间通信的端口 |
| `max_model_len` | int | 2048 | 服务过程中的最大序列长度（输入 + 输出） |
| `max_num_seqs` | int | 8 | 解码阶段的最大批处理大小。超出此数量的请求将被排队 |
| `use_warmup` | int | 0 | 启动时是否执行预热。会生成最大长度的数据用于预热（KV Cache 计算默认使用）。 |
| `gpu_memory_utilization` | float | 0.9 | GPU 内存利用率 |
| `quantization` | str | None | 模型量化策略，加载 BF16 检查点时，指定 wint4 或 wint8 支持无损在线 4bit/8bit 量化 |
| `block_size` | int | 64 | 每个缓存管理块的 token 数量 |
| `kv_cache_ratio` | float | 0.75 | 分配给输入的 KV Cache 比例。推荐值 = 平均输入长度 / (平均输入长度 + 平均输出长度) |

* 注意：模型部署的最优配置可参考： https://github.com/PaddlePaddle/FastDeploy/tree/develop/docs/zh/optimal_deployment

## 2. 聊天请求配置

| 参数名称 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| `port` | int | 8188 | 请求端口号 |
| `max_new_tokens` | int | 1024 | 最大生成令牌数 |
| `min_tokens` | int | 0 | 最小生成令牌数 |
| `temperature` | float | 0.95 | 控制输出的随机性：较高的值产生更具创意/随机性的文本，较低的值使输出更具确定性 |
| `top_p` | float | 0.7 | 在生成过程中，动态选择累积概率 ≥ top_p 的令牌。较高的值增加多样性 |
| `frequency_penalty` | float | 0.0 | `>0.0`: 根据令牌在文本中的出现频率惩罚新令牌，减少重复 |
| `presence_penalty` | float | 0.0 | `>0.0`: 惩罚文本中已经出现的令牌，增加讨论新话题的可能性 |
| `repetition_penalty` | float | 1.0 | 控制生成文本中的重复性。较高的值减少重复 |
| `stream` | bool | True | 是否启用流式输出。如果为True，则逐步返回令牌；如果为False，则一次性返回完整文本 |
| `stream_options` | StreamOptions | None | 自定义流式行为的配置选项 |
