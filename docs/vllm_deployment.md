# Deploying ERNIE 4.5 Models with vLLM

This guide walks through deploying ERNIE 4.5 MoE models using [vLLM](https://github.com/vllm-project/vllm), a high-throughput inference engine that supports PaddlePaddle-exported models.

## Supported Models

| Model | Parameters | Active Params | Context Window | Recommended GPUs |
|-------|-----------|---------------|----------------|-----------------|
| ERNIE-4.5-300B-A47B | 300B | 47B | 128K | 8×A100-80G / 8×H100 |
| ERNIE-4.5-21B-A3B | 21B | 3B | 128K | 1×A100-40G / 1×A800 |
| ERNIE-4.5-0.3B | 0.3B | 0.3B | 128K | 1×T4 / CPU |

## Prerequisites

- Python 3.9+
- CUDA 12.1+ (for GPU inference)
- At least 16GB RAM (for 0.3B model)

## Installation

```bash
# Install vLLM with PaddlePaddle support
pip install vllm>=0.6.0
pip install paddlepaddle-gpu
```

## Step 1: Export Model from PaddlePaddle Format

ERNIE 4.5 models are released in PaddlePaddle format. Convert them to HuggingFace format for vLLM compatibility:

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "baidu/ERNIE-4.5-21B-A3B"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Export to HuggingFace format
output_dir = "./ernie-4.5-21b-hf"
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)
print(f"Model exported to {output_dir}")
```

## Step 2: Launch vLLM Server

### Offline Batch Inference

```python
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(
    model="./ernie-4.5-21b-hf",
    tensor_parallel_size=1,       # Use 1 GPU for 21B-A3B
    max_model_len=4096,           # Adjust based on GPU memory
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
)

# Configure sampling
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

# Generate
prompts = [
    "Explain the Mixture of Experts architecture in ERNIE 4.5.",
    "Write a Python function to calculate Fibonacci numbers.",
]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Response: {output.outputs[0].text}\n")
```

### OpenAI-Compatible API Server

```bash
# Start the server
python -m vllm.entrypoints.openai.api_server \
    --model ./ernie-4.5-21b-hf \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --trust-remote-code \
    --port 8000
```

Then use it like an OpenAI API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="./ernie-4.5-21b-hf",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the key innovations in ERNIE 4.5?"},
    ],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

## Step 3: Multi-GPU Deployment for Large Models

For the 300B-A47B model, use tensor parallelism:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./ernie-4.5-300b-hf \
    --tensor-parallel-size 8 \
    --max-model-len 8192 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --port 8000
```

### Performance Tuning

| Parameter | Recommended Value | Description |
|-----------|------------------|-------------|
| `--gpu-memory-utilization` | 0.9 - 0.95 | Fraction of GPU memory to use |
| `--max-model-len` | 4096 - 32768 | Max sequence length (trade memory for length) |
| `--tensor-parallel-size` | 1 / 4 / 8 | Number of GPUs for tensor parallelism |
| `--enforce-eager` | flag | Disable CUDA graphs for debugging |

## Step 4: Docker Deployment

```bash
# Build image
docker build -t ernie-vllm -f - . << 'EOF'
FROM vllm/vllm-openai:latest

RUN pip install paddlepaddle-gpu paddlenlp

COPY --chmod=755 <<'SCRIPT' /serve.sh
#!/bin/bash
python -c "
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('${MODEL_NAME}')
tokenizer = AutoTokenizer.from_pretrained('${MODEL_NAME}')
model.save_pretrained('/models/hf', safe_serialization=True)
tokenizer.save_pretrained('/models/hf')
"
exec python -m vllm.entrypoints.openai.api_server \
    --model /models/hf \
    --tensor-parallel-size ${TP_SIZE:-1} \
    --max-model-len ${MAX_LEN:-4096} \
    --trust-remote-code \
    --host 0.0.0.0
SCRIPT

ENTRYPOINT ["/serve.sh"]
EOF

# Run
docker run --gpus all -p 8000:8000 \
    -e MODEL_NAME=baidu/ERNIE-4.5-21B-A3B \
    -e TP_SIZE=1 \
    -e MAX_LEN=4096 \
    ernie-vllm
```

## Troubleshooting

### Out of Memory

If you encounter CUDA OOM errors:

1. Reduce `--max-model-len` (e.g., from 8192 to 4096)
2. Reduce `--gpu-memory-utilization` to 0.85
3. Enable quantization: `--quantization awq` (if quantized weights available)

### Slow First Request

The first request includes model compilation time. To pre-warm:

```bash
curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "./ernie-4.5-21b-hf", "prompt": "warmup", "max_tokens": 1}'
```

### Model Format Issues

If vLLM reports unsupported model format, ensure:
1. Model was exported with `safe_serialization=True`
2. The `config.json` contains `model_type` field
3. PaddleNLP version matches the export version

## References

- [ERNIE 4.5 Paper](https://yiyan.baidu.com/blog/publication/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [PaddleNLP Model Zoo](https://paddlenlp.readthedocs.io/)
