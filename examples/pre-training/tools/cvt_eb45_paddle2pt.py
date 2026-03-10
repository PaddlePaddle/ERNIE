# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import os
import json
import paddle
from safetensors import safe_open
from safetensors.paddle import save_file
from collections import defaultdict

SRC_PATH = "./ERNIE-4.5-300B-A47B-Base-Paddle"
DST_PATH = "./ERNIE-4.5-300B-A47B-Base-PT-out"

os.makedirs(DST_PATH, exist_ok=True)

with open(os.path.join(SRC_PATH, "model.safetensors.index.json")) as f:
    src_map = json.load(f)["weight_map"]
with open(os.path.join(SRC_PATH, "config.json")) as f:
    config = json.load(f)

assert (
    config["hidden_size"] % config["num_attention_heads"] == 0
), "head_dim not divisible"
head_dim = config["hidden_size"] // config["num_attention_heads"]
q_size = head_dim * config["num_attention_heads"]
kv_size = head_dim * config["num_key_value_heads"]

total_size = 0
dst_map = {}

src_rev_map = defaultdict(set)
for k, v in src_map.items():
    src_rev_map[v].add(k)

for src_file, src_keys in src_rev_map.items():
    print("reading:", src_file, "size:", len(src_keys))
    dst_weight = {}

    for src_key in sorted(src_keys):
        with safe_open(
            os.path.join(SRC_PATH, src_file), framework="paddle", device="cpu"
        ) as f:
            tensor = f.get_tensor(src_key)

        base_key = src_key
        if base_key.startswith("ernie."):
            base_key = "model." + base_key[6:]

        if ".up_gate_proj." in src_key:
            # split gate_proj / up_proj (equal halves)
            half = tensor.shape[-1] // 2
            gate_tensor = tensor[:, :half]
            up_tensor = tensor[:, half:]

            # transpose back
            gate_tensor = gate_tensor.T.contiguous()
            up_tensor = up_tensor.T.contiguous()

            gate_key = base_key.replace(".up_gate_proj.", ".gate_proj.")
            up_key = base_key.replace(".up_gate_proj.", ".up_proj.")

            dst_weight[gate_key] = gate_tensor
            dst_weight[up_key] = up_tensor
            dst_map[gate_key] = src_file
            dst_map[up_key] = src_file

        elif ".qkv_proj." in src_key:
            # split q / k / v (unequal: q_size, kv_size, kv_size)
            q_tensor, k_tensor, v_tensor = paddle.split(
                tensor, [q_size, kv_size, kv_size], axis=-1
            )

            # transpose back
            q_tensor = q_tensor.T.contiguous()
            k_tensor = k_tensor.T.contiguous()
            v_tensor = v_tensor.T.contiguous()

            q_key = base_key.replace(".qkv_proj.", ".q_proj.")
            k_key = base_key.replace(".qkv_proj.", ".k_proj.")
            v_key = base_key.replace(".qkv_proj.", ".v_proj.")

            dst_weight[q_key] = q_tensor
            dst_weight[k_key] = k_tensor
            dst_weight[v_key] = v_tensor
            dst_map[q_key] = src_file
            dst_map[k_key] = src_file
            dst_map[v_key] = src_file

        else:
            # no merge, just possibly transpose
            if "_proj." in src_key or ".gate." in src_key or "lm_head" in src_key:
                tensor = tensor.T.contiguous()

            dst_weight[base_key] = tensor
            dst_map[base_key] = src_file

        print(end=".", flush=True)

    save_file(dst_weight, os.path.join(DST_PATH, src_file))
    print()

with open(os.path.join(DST_PATH, "model.safetensors.index.json"), "w") as f:
    data = {
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": dst_map,
    }
    json.dump(data, f, ensure_ascii=False, indent=2)
print("done")
