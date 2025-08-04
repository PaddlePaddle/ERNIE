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

"""
convert model param.
"""

import argparse
import sys
import os
import json
import shutil
import fnmatch
import re
import paddle
from safetensors.numpy import load_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--org", type=str, required=True, help="the path of origin checkpoint."
    )
    parser.add_argument(
        "--cur", type=str, required=True, help="the path of current checkpoint."
    )
    parser.add_argument(
        "--dst", type=str, required=True, help="the path of converted checkpoint."
    )
    args = parser.parse_args()
    return args


def find_files(path, suffixes):
    """
    find all pdparams files or pdopt files.
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, suffixes):
                if "scheduler.pdparams" not in name:
                    result.append(os.path.join(root, name))
    print("find {} {} files".format(len(result), suffixes))
    assert len(result) > 0
    result = sorted(result)
    return result


class Checkpoint:
    def __init__(self, args):
        """
        __init__
        """
        self.args = args
        meta_path = os.path.join(args.cur, "model_meta.json")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        parallel_config = self.meta["parallel_config"]
        self.meta_path = meta_path
        self.mp_degree = parallel_config["mp_degree"]
        self.pp_degree = parallel_config["pp_degree"]
        self.sharding_degree = parallel_config["sharding_degree"]
        print(
            "src ckpt info: tp-degree: {}, pp-degree: {}, sharding-degree: {}".format(
                self.mp_degree, self.pp_degree, self.sharding_degree
            )
        )
        self.sharding_strategy = self.meta["sharding_metas"]["tp00_pp00"][
            "sharding_strategy"
        ]
        self.pdparams_files = find_files(args.cur, "*.pdparams")
        self.pdopt_files = find_files(args.cur, "*.pdopt")
        with open(os.path.join(args.org, "model.safetensors.index.json"), "r") as f:
            self.safetensors_index = json.load(f)
        self.tensor_offset_map = {}
        self.cur_to_org_name_map = {
            "ernie.embed.embed_tokens.weight": "ernie.embed_tokens.weight",
            "self_attn.fused_rms_norm_linear.rms_norm_weight": "input_layernorm.weight",
            "self_attn.fused_rms_norm_linear.linear_weight": "self_attn.qkv_proj.weight",
            "embed_share.embed_tokens.weight": "ernie.embed_tokens.weight",
        }

    def map_to_org_model(self, layer_name):
        for k, v in self.cur_to_org_name_map.items():
            if k in layer_name:
                output_str = layer_name.replace(k, v)
                break
            else:
                output_str = layer_name
        return output_str

    def load_from_org_model(self, layer_name):
        matched_layer_name = self.map_to_org_model(layer_name)
        if matched_layer_name in self.safetensors_index["weight_map"]:
            file_name = self.safetensors_index["weight_map"][matched_layer_name]
            safetensor_file = os.path.join(self.args.org, file_name)
            if not os.path.exists(safetensor_file):
                print("{} not exists".format(safetensor_file))
                return None, None
            ckpt = load_file(safetensor_file)
            if matched_layer_name in ckpt.keys():
                return matched_layer_name, paddle.to_tensor(ckpt[matched_layer_name])
            else:
                print("{} not found in safetensors".format(matched_layer_name))
                return None, None
        else:
            print("{} not found in safetensors index".format(matched_layer_name))
            return None, None

    def process_one_pdparam(self, pdparam_path):
        pdparam = paddle.load(pdparam_path)
        for layer_name, tensor_data in pdparam.items():
            _, loaded_value = self.load_from_org_model(layer_name)
            if loaded_value is None:
                continue
            assert (
                loaded_value.shape == tensor_data.shape
            ), f"Shape mismatch: loaded_value.shape={loaded_value.shape}, expected={tensor_data.shape}"
            pdparam[layer_name].set_value(paddle.cast(loaded_value, tensor_data.dtype))
            print("successfully convert {}: {}".format(layer_name, tensor_data.shape))
        paddle.save(
            pdparam, os.path.join(self.args.dst, os.path.basename(pdparam_path))
        )

    def process_pdparams(self):
        for pdparam_path in self.pdparams_files:
            print("processing {}".format(pdparam_path))
            self.process_one_pdparam(pdparam_path)

    def load_from_org_model_with_tensor_name(
        self, tensor_name, structure_name_mapping, shard_num
    ):
        layer_names = []
        for layer_name, value in structure_name_mapping.items():
            if tensor_name == value:
                layer_names.append(layer_name)
        if len(layer_names) > 1:
            print("find {} layer_names for {}".format(len(layer_names), tensor_name))
            sys.exit(1)
        if len(layer_names) == 0:
            print("{} not found in structure_name_mapping".format(tensor_name))
            return None, None
        matched_layer_name = self.map_to_org_model(layer_names[0])

        if matched_layer_name in self.safetensors_index["weight_map"]:
            file_name = self.safetensors_index["weight_map"][matched_layer_name]
            safetensor_file = os.path.join(self.args.org, file_name)
            if not os.path.exists(safetensor_file):
                print("{} not exists".format(safetensor_file))
                return None, None
            ckpt = load_file(safetensor_file)
            if matched_layer_name in ckpt.keys():
                return matched_layer_name, paddle.to_tensor(ckpt[matched_layer_name])
            else:
                print("{} not found in safetensors".format(matched_layer_name))
                return None, None
        else:
            print("{} not found in safetensors index".format(matched_layer_name))
            return None, None

    def process_one_pdopt(self, pdopt_path):
        match = re.search(r"pp(\d+)_shard(\d+)", pdopt_path)
        assert match is not None
        pp_num = match.group(1)
        shard_num = match.group(2)
        print(f"pp: {pp_num}, shard: {shard_num}")
        sharding_metas_key = "tp00_pp{}".format(pp_num)
        structure_name_mapping = self.meta["sharding_metas"][sharding_metas_key][
            "structure_name_mapping"
        ]

        pdopt = paddle.load(pdopt_path)
        for tensor_name, tensor_data in pdopt["master_weights"].items():
            matched_layer_name, loaded_value = (
                self.load_from_org_model_with_tensor_name(
                    tensor_name, structure_name_mapping, shard_num
                )
            )
            if loaded_value is None:
                continue
            if tensor_name not in self.tensor_offset_map.keys():
                self.tensor_offset_map[tensor_name] = 0
            if "mlp.experts" in matched_layer_name:
                self.tensor_offset_map[tensor_name] = 0
            offset = self.tensor_offset_map[tensor_name]
            tensor_data_num = tensor_data.flatten().shape[0]
            assert loaded_value.flatten().shape[0] >= offset + tensor_data_num, (
                f"Shape mismatch: org_shape={loaded_value.shape}, cur_shape={tensor_data.shape}"
                f", tensor_name={tensor_name}, matched_layer_name={matched_layer_name}, offset={offset}"
            )
            weight_t = paddle.cast(
                loaded_value.flatten()[offset : offset + tensor_data_num],
                tensor_data.dtype,
            )
            pdopt["master_weights"][tensor_name].set_value(weight_t)
            print(
                "successfully convert {}: {}".format(
                    matched_layer_name, tensor_data.shape
                )
            )
            self.tensor_offset_map[tensor_name] += tensor_data_num
        paddle.save(pdopt, os.path.join(self.args.dst, os.path.basename(pdopt_path)))

    def process_pdopts(self):
        for pdopt_path in self.pdopt_files:
            print("processing {}".format(pdopt_path))
            self.process_one_pdopt(pdopt_path)


def convert_ckpt(args):
    print("================= start converting checkpoints ===============")
    ckpt = Checkpoint(args)
    ckpt.process_pdparams()
    ckpt.process_pdopts()


if __name__ == "__main__":
    args = parse_args()
    print("origin checkpoint path: ", args.org)
    print("current checkpoint path: ", args.cur)
    print("converted checkpoint path: ", args.dst)
    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    shutil.copytree(args.cur, args.dst)
    convert_ckpt(args)
