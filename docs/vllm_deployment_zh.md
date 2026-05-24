# 使用 vLLM 部署 ERNIE 4.5 模型

本指南介绍如何使用 [vLLM](https://github.com/vllm-project/vllm) 高性能推理引擎部署 ERNIE 4.5 MoE 系列模型。

## 支持的模型

| 模型 | 总参数量 | 激活参数量 | 上下文窗口 | 推荐 GPU 配置 |
|------|---------|-----------|-----------|--------------|
| ERNIE-4.5-300B-A47B | 300B | 47B | 128K | 8×A100-80G / 8×H100 |
| ERNIE-4.5-21B-A3B | 21B | 3B | 128K | 1×A100-40G / 1×A800 |
| ERNIE-4.5-0.3B | 0.3B | 0.3B | 128K | 1×T4 / CPU |

## 环境准备

- Python 3.9+
- CUDA 12.1+（GPU 推理）
- 至少 16GB 内存（0.3B 模型）

## 安装依赖

```bash
# 安装 vLLM 和 PaddlePaddle
pip install vllm>=0.6.0
pip install paddlepaddle-gpu
```

## 第一步：模型格式转换

ERNIE 4.5 模型以 PaddlePaddle 格式发布，需要转换为 HuggingFace 格式以兼容 vLLM：

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "baidu/ERNIE-4.5-21B-A3B"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 导出为 HuggingFace 格式
output_dir = "./ernie-4.5-21b-hf"
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)
print(f"模型已导出到 {output_dir}")
```

## 第二步：启动推理服务

### 离线批量推理

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="./ernie-4.5-21b-hf",
    tensor_parallel_size=1,       # 21B-A3B 使用 1 块 GPU
    max_model_len=4096,           # 根据显存调整
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
)

# 配置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

# 生成
prompts = [
    "解释 ERNIE 4.5 中的混合专家（MoE）架构。",
    "写一个 Python 函数计算斐波那契数列。",
]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"提示: {output.prompt}")
    print(f"回复: {output.outputs[0].text}\n")
```

### OpenAI 兼容 API 服务

```bash
# 启动服务
python -m vllm.entrypoints.openai.api_server \
    --model ./ernie-4.5-21b-hf \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --trust-remote-code \
    --port 8000
```

使用 OpenAI SDK 调用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="./ernie-4.5-21b-hf",
    messages=[
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "ERNIE 4.5 有哪些关键创新？"},
    ],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

## 第三步：大模型多卡部署

对于 300B-A47B 模型，使用张量并行：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./ernie-4.5-300b-hf \
    --tensor-parallel-size 8 \
    --max-model-len 8192 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --port 8000
```

### 性能调优

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| `--gpu-memory-utilization` | 0.9 - 0.95 | GPU 显存使用比例 |
| `--max-model-len` | 4096 - 32768 | 最大序列长度（显存换长度） |
| `--tensor-parallel-size` | 1 / 4 / 8 | 张量并行 GPU 数量 |
| `--enforce-eager` | 标志 | 禁用 CUDA Graph（调试用） |

## 第四步：Docker 容器部署

```bash
# 运行容器
docker run --gpus all -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model ./ernie-4.5-21b-hf \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --trust-remote-code \
    --host 0.0.0.0
```

## 常见问题

### 显存不足

遇到 CUDA OOM 错误时：

1. 降低 `--max-model-len`（如从 8192 降到 4096）
2. 降低 `--gpu-memory-utilization` 至 0.85
3. 启用量化：`--quantization awq`（如有量化权重）

### 首次请求慢

首次请求包含模型编译时间，可通过预热解决：

```bash
curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "./ernie-4.5-21b-hf", "prompt": "warmup", "max_tokens": 1}'
```

### 模型格式不兼容

确保：
1. 导出时使用 `safe_serialization=True`
2. `config.json` 包含 `model_type` 字段
3. PaddleNLP 版本与导出版本一致

## 参考资料

- [ERNIE 4.5 论文](https://yiyan.baidu.com/blog/publication/)
- [vLLM 文档](https://docs.vllm.ai/)
- [PaddleNLP 模型库](https://paddlenlp.readthedocs.io/)
