# Export Configuration

## 1. LoRA Merge Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name_or_path` | str | None | Base model name or path |
| `output_dir` | str | None | Automatically searches for the latest checkpoint in this directory. If found, loads it for LoRA Merge; otherwise raises FileNotFoundError |
| `lora` | bool | False | Set to True to perform LoRA Merge, otherwise raises ValueError |
| `copy_tokenizer` | bool | True | Whether to copy tokenizer files simultaneously |

## 2. Split Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `output_dir` | str | None | Loads the model from this path for splitting |
| `compute_type` | str | bf16 | Computation type, options: "bf16", "fp16", "fp8", "wint8", "wint4/8" |
| `tensor_parallel_degree` | int | 1 | Number of GPUs for tensor parallel configuration |
| `max_shard_size` | int | 5 | Maximum size of each checkpoint file (in GB) |
