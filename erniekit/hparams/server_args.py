# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field


@dataclass
class ServerArguments:
    """Server parameters"""

    # network
    host: str = field(
        default="127.0.0.1",
        metadata={"help": "Hostname of service configuration."},
    )
    port: int = field(
        default=8188,
        metadata={"help": "HTTP port of service configuration."},
    )
    metrics_port: int = field(
        default=8001,
        metadata={"help": "The port of the supervision metric in the model service."},
    )
    engine_worker_queue_port: int = field(
        default=8002,
        metadata={
            "help": "The port used for inter-process communication within the engine."
        },
    )

    # model
    max_model_len: int = field(
        default=2048,
        metadata={"help": "Maximum context length supported by the model."},
    )
    max_num_seqs: int = field(
        default=8, metadata={"help": "Maximum number of sequences per iteration."}
    )
    use_warmup: int = field(
        default=0,
        metadata={"help": "Flag to indicate whether to use warm-up before inference."},
    )
    gpu_memory_utilization: float = field(
        default=0.9, metadata={"help": "The fraction of GPU memory to be utilized."}
    )
    quantization: str = field(
        default=None,
        metadata={
            "help": "Model quantization strategy, when loading BF16 CKPT, specifying wint4 or wint8 supports lossless online 4bit/8bit quantization."
        },
    )

    # cache
    block_size: int = field(
        default=64, metadata={"help": "Number of tokens in one processing block."}
    )
    kv_cache_ratio: float = field(
        default=0.75, metadata={"help": "Ratio of tokens to process in a block."}
    )
