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

src_path = "./ERNIE-4.5-300B-A47B-Base-PT"
dst_path = "./ERNIE-4.5-300B-A47B-Base-Paddle-out"

os.makedirs(dst_path, exist_ok=True)

with open(os.path.join(src_path, "model.safetensors.index.json")) as f:
    src_map = json.load(f)["weight_map"]

total_size = 0
dst_map = {}

src_rev_map = defaultdict(set)
for k, v in src_map.items():
    src_rev_map[v].add(k)

for src_file, src_keys in src_rev_map.items():
    print("reading:", src_file, "size:", len(src_keys))
    dst_weight = {}

    for src_key in src_keys:
        if ".k_proj." in src_key or ".v_proj." in src_key or ".up_proj." in src_key:
            continue

        dst_key = src_key
        if dst_key.startswith("model."):
            dst_key = "ernie." + dst_key[6:]

        key_decomp = [src_key]
        if ".gate_proj." in src_key:
            dst_key = dst_key.replace(".gate_proj.", ".up_gate_proj.")
            key_decomp = [
                src_key,
                src_key.replace(".gate_proj.", ".up_proj."),
            ]
        elif ".q_proj." in src_key:
            dst_key = dst_key.replace(".q_proj.", ".qkv_proj.")
            key_decomp = [
                src_key,
                src_key.replace(".q_proj.", ".k_proj."),
                src_key.replace(".q_proj.", ".v_proj."),
            ]

        weight_decomp = []
        for key in key_decomp:
            with safe_open(
                os.path.join(src_path, src_file), framework="paddle", device="cpu"
            ) as f:
                tensor = f.get_tensor(key)
            if "_proj." in key or ".gate." in key or "lm_head" in key:
                tensor = tensor.T.contiguous()
            weight_decomp.append(tensor)

        dst_weight[dst_key] = (
            weight_decomp[0]
            if len(weight_decomp) == 1
            else paddle.concat(weight_decomp, axis=-1)
        )
        dst_map[dst_key] = src_file
        print(end=".", flush=True)

    save_file(dst_weight, os.path.join(dst_path, src_file))
    print()

with open(os.path.join(dst_path, "model.safetensors.index.json"), "w") as f:
    data = {
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": dst_map,
    }
    json.dump(data, f, ensure_ascii=False, indent=2)
print("done")
