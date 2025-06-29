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
"""PadddlePaddle to PyTorch weight convertor"""

import argparse
import copy
import json
import os
import shutil
from typing import Optional

import numpy as np
import paddle
import torch
from safetensors.numpy import load_file
from safetensors.torch import save_file

NEED_TRANSPOSE_KEYS = {"down_proj.weight", "o_proj.weight", "gate.weight"}
KEEP_FP32_KEYS = ("gate.weight", "moe_statics.e_score_correction_bias")


def check_if_transpose_needed(key: str) -> bool:
    """Check if the given tensor key needs transpose operation.

    Args:
        key: The tensor key to check

    Returns:
        bool: True if transpose is needed, False otherwise
    """
    return any(x in key for x in NEED_TRANSPOSE_KEYS)


def convert_numpy_to_tensor(numpy_arr: np.ndarray, needs_transpose: bool) -> torch.Tensor:
    """Convert PaddlePaddle tensor to PyTorch tensor with optional transpose.

    Args:
        paddle_tensor: Input PaddlePaddle tensor
        needs_transpose: Whether to transpose the tensor

    Returns:
        Converted PyTorch tensor
    """
    if needs_transpose:
        tensor = torch.from_numpy(numpy_arr.T).contiguous()
    else:
        tensor = torch.from_numpy(numpy_arr).contiguous()

    if numpy_arr.dtype == np.uint16:
        tensor = tensor.view(torch.bfloat16)

    return tensor


def process_qkv_proj(
    dest_key: str,
    tensor: torch.Tensor,
    processed_tensors: dict,
    processed_safetensors_index: Optional[set] = None,
    file_name: Optional[str] = None,
    head_dim: int = 128,
    num_heads: int = 64,
    kv_heads: int = 8,
) -> None:
    """
    Processes combined QKV projection for Ernie-4.5 models.
    Splits the input tensor into separate Q, K, and V components and performs
    necessary dimension transformations.

    Args:
        dest_key: Prefix for storing processed tensors in the output dictionary
        tensor: Combined QKV weight matrix (shape typically [dim, dim*3])
        processed_tensors: Dictionary to accumulate all processed tensors
        processed_safetensors_index: Optional set to track processed tensor names
        file_name: Optional source filename for debugging purposes
        num_heads: Total number of attention heads
        kv_heads: Number of K/V heads (for Grouped Query Attention)

    Returns:
        None (results are stored in processed_tensors dictionary)
    """
    D, N = tensor.shape
    hidden_size = head_dim * num_heads
    kv_hidden_size = head_dim * kv_heads
    assert (
        N == hidden_size + 2 * kv_hidden_size
    ), f"{N} != {hidden_size} + {2*kv_hidden_size}, \
        qkv_proj.weight dimensions mismatch the model configuration, \
        please check config.json"

    # split
    q_proj, k_proj, v_proj = np.split(tensor, [hidden_size, hidden_size + kv_hidden_size], axis=-1)

    # Transpose into PyTorch format
    q_proj = convert_numpy_to_tensor(q_proj, needs_transpose=True)
    k_proj = convert_numpy_to_tensor(k_proj, needs_transpose=True)
    v_proj = convert_numpy_to_tensor(v_proj, needs_transpose=True)

    q_dest_key = dest_key.replace("qkv_proj", "q_proj")
    k_dest_key = dest_key.replace("qkv_proj", "k_proj")
    v_dest_key = dest_key.replace("qkv_proj", "v_proj")

    # maybe save index
    if processed_safetensors_index is not None and file_name is not None:
        processed_safetensors_index[q_dest_key] = file_name
        processed_safetensors_index[k_dest_key] = file_name
        processed_safetensors_index[v_dest_key] = file_name

    # write
    processed_tensors[q_dest_key] = q_proj
    processed_tensors[k_dest_key] = k_proj
    processed_tensors[v_dest_key] = v_proj


def process_up_gate_proj(
    dest_key: str,
    tensor: torch.Tensor,
    processed_tensors: dict,
    processed_safetensors_index: set,
    file_name: str,
    intermediate_size: Optional[int] = None,
):
    """
    Processes combined up_gate_proj for MLP layers in Transformer models.
    Splits the input tensor into separate up_proj and gate_proj components.

    Typical usage in models like LLaMA where:
    - The combined tensor contains [gate_proj, up_proj] concatenated together
    - Splitting is done along the first dimension (for FFN implementation)

    Args:
        dest_key: Key prefix for storing processed tensors (e.g., 'model.layers.0.mlp')
        tensor: Combined weight matrix of shape [hidden_size, 2*intermediate_size]
        processed_tensors: Dictionary to store processed tensors (modified in-place)
        processed_safetensors_index: Set to track which tensors have been processed
        file_name: filename of safetensors
        intermediate_size: Expected intermediate size (for validation if provided)

    Returns:
        None (results are stored in processed_tensors and tracked in processed_safetensors_index)
    """
    hidden_size, fused_dim = tensor.shape
    # auto infer intermediate_size
    if intermediate_size is None:
        assert (
            fused_dim % 2 == 0
        ), f"Cannot split tensor with odd dimension {fused_dim}. \
             Specify intermediate_size."
        intermediate_size = fused_dim // 2

    assert (
        fused_dim == 2 * intermediate_size
    ), f"Tensor shape {tensor.shape} does not match 2 * intermediate_size {2 * intermediate_size}"

    gate_proj, up_proj = np.split(tensor, [intermediate_size], axis=-1)

    up_proj = convert_numpy_to_tensor(up_proj, True)
    gate_proj = convert_numpy_to_tensor(gate_proj, True)

    up_dest_key = dest_key.replace("up_gate_proj", "up_proj")  # 若 dest_key 包含 "up_gate_proj"
    gate_dest_key = dest_key.replace("up_gate_proj", "gate_proj")

    processed_tensors.update({up_dest_key: up_proj, gate_dest_key: gate_proj})

    processed_safetensors_index.update({up_dest_key: file_name, gate_dest_key: file_name})


def update_config_files(args, dest_dir):
    """update config.json"""

    use_moe = args.use_moe
    config_file_path = os.path.join(dest_dir, "config.json")

    with open(config_file_path, 'r') as f:
        config_dict = json.load(f)

    pop_keys = [
        "add_tail_layers",
        "compression_ratio",
        "fuse_gate_detach_matmul",
        "fuse_linear",
        "fuse_ln",
        "fuse_rms_norm",
        "fuse_rope",
        "fuse_swiglu",
        "global_aux_loss",
        "moe_aux_loss_lambda",
        "moe_group_orthogonal_loss",
        "moe_orthogonal_loss_lambda",
        "moe_z_loss_lambda",
        "use_fast_ln",
        "use_flash_attention",
        "use_rmsnorm",
        "max_sequence_length",
        "model_name",
        "paddlenlp_version",
    ]

    rename_keys = {"dtype": "torch_dtype"}

    new_keys = {
        "model_type": "ernie4_5_moe" if use_moe else "ernie4_5",
        "architectures": ["Ernie4_5_MoeForCausalLM"] if use_moe else ["Ernie4_5_ForCausalLM"],
        "auto_map": {
            "AutoConfig": (
                "configuration_ernie4_5_moe.Ernie4_5_MoeConfig"
                if use_moe
                else "configuration_ernie4_5.Ernie4_5_Config"
            ),
            "AutoModel": "modeling_ernie4_5_moe.Ernie4_5_Model" if use_moe else "modeling_ernie4_5.Ernie4_5_Model",
            "AutoModelForCausalLM": (
                "modeling_ernie4_5_moe.Ernie4_5_MoeForCausalLM"
                if use_moe
                else "modeling_ernie4_5.Ernie4_5_ForCausalLM"
            ),
        },
        "_attn_implementation": "eager",
        "use_cache": True,
    }

    for key in pop_keys:
        if config_dict.get(key) is not None:
            config_dict.pop(key)

    for old_key, new_key in rename_keys.items():
        if config_dict.get(old_key) is not None:
            print(f"Renaming {old_key} to {new_key}")
            config_dict[new_key] = config_dict.pop(old_key)

    config_dict.update(new_keys)
    config_dict = dict(sorted(config_dict.items()))

    with open(config_file_path, 'w') as f:
        json.dump(config_dict, f, indent=4)


def update_tokenizer_config_files(dest_dir):
    """update tokenizer_config.json"""

    config_file_path = os.path.join(dest_dir, "tokenizer_config.json")

    with open(config_file_path, 'r') as f:
        config_dict = json.load(f)

    new_keys = {
        "tokenizer_class": "Ernie4_5_Tokenizer",
        "auto_map": {
            "AutoTokenizer": ["tokenization_ernie4_5.Ernie4_5_Tokenizer", "tokenization_ernie4_5.Ernie4_5_Tokenizer"]
        },
    }

    config_dict.update(new_keys)
    with open(config_file_path, 'w') as f:
        json.dump(config_dict, f, indent=4)


def process_params_file(args, src_file_path: str, dest_file_path: str, src_prefix: str, dst_prefix: str) -> None:
    """Process a single safetensors file, converting PyTorch tensors to PaddlePaddle format.

    Args:
        src_file_path: Path to source safetensors file
        dest_file_path: Path to destination safetensors file
        src_prefix: Source key prefix to replace
        dst_prefix: Destination key prefix
    """
    processed_tensors = {}
    processed_safentensors_index = {}
    new_total_size = 0

    # load params file
    file_name = os.path.basename(src_file_path)

    if src_file_path.endswith("safetensors"):
        pd_tensors = load_file(src_file_path)
    else:
        pd_tensors = paddle.load(src_file_path)

    # load configuration file
    config_file_path = os.path.join(args.src_dir, "config.json")
    assert os.path.exists(config_file_path), f"Config file not found: {config_file_path}"
    with open(config_file_path, 'r') as f:
        config_dict = json.load(f)

    num_attention_heads = config_dict.get('num_attention_heads', 64)
    num_key_value_heads = config_dict.get('num_key_value_heads', 8)
    tie_word_embeddings = config_dict.get('tie_word_embeddings', False)
    hidden_size = config_dict.get('hidden_size', 8192)
    head_dim = config_dict.get('head_dim', hidden_size // num_attention_heads)

    for key, tensor in pd_tensors.items():
        # skip keys
        if any(x in key for x in ["resampler_model", "weight_1"]):
            continue

        dest_key = key.replace(src_prefix, dst_prefix)
        if not isinstance(tensor, np.ndarray):
            tensor = tensor.numpy()

        weight_size = np.prod(tensor.shape) * tensor.dtype.itemsize
        new_total_size += weight_size

        if 'qkv_proj' in key:
            # split qkv_proj to q_proj, k_proj and v_proj
            process_qkv_proj(
                dest_key,
                tensor,
                processed_tensors,
                processed_safentensors_index,
                file_name,
                head_dim,
                num_attention_heads,
                num_key_value_heads,
            )
        elif "up_gate_proj" in key:
            # split up_gate_proj to up_proj and gate_proj
            process_up_gate_proj(dest_key, tensor, processed_tensors, processed_safentensors_index, file_name)
        else:
            if "lm_head.weight" in key:
                needs_transpose = not tie_word_embeddings
            else:
                needs_transpose = check_if_transpose_needed(key)
            print(f"Processing key: {key}, shape: {tensor.shape}, transpose: {needs_transpose}")

            torch_tensor = convert_numpy_to_tensor(tensor, needs_transpose)
            processed_tensors[dest_key] = torch_tensor

            # force keep fp32
            if any(s in key for s in KEEP_FP32_KEYS):
                if processed_tensors[dest_key].dtype != torch.float32:
                    print(f"Tensor {dest_key} should be float32, force set it to float32!")
                    processed_tensors[dest_key] = torch_tensor.float()

            print(
                f"Converted to key: {dest_key},\
                shape: {processed_tensors[dest_key].shape}, \
                dtype: {processed_tensors[dest_key].dtype}"
            )
            processed_safentensors_index[dest_key] = file_name

    # save tensors
    save_file(processed_tensors, dest_file_path, metadata={"format": "pt"})

    return processed_safentensors_index, new_total_size


def process_model_directory(
    args, src_dir: str, dest_dir: str, src_prefix: str, dst_prefix: str, skip_config_update: bool = False
) -> None:
    """Process the entire model directory.

    Args:
        src_dir: Path to source model directory
        dest_dir: Path to destination model directory
        src_prefix: Source key prefix to replace
        dst_prefix: Destination key prefix
        skip_config_update: Whether to skip config file updates
    """
    use_index_file = False

    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    if os.path.exists(dest_dir):
        print(f"Destination directory already exists: {dest_dir}")

    os.makedirs(dest_dir, exist_ok=True)

    index_file = os.path.join(src_dir, "model.safetensors.index.json")

    if os.path.exists(index_file):
        # Process indexed safetensors
        with open(index_file, "r") as f:
            index = json.load(f)

        # Update dst_index weight map keys
        dst_index = copy.deepcopy(index)
        dst_index["weight_map"] = {}
        use_index_file = True

    new_total_size = 0
    for file_name in sorted(os.listdir(src_dir)):
        if file_name.startswith(".") or file_name.startswith("chat_template."):
            continue

        src_path = os.path.join(src_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)

        if not os.path.isfile(src_path):
            print(f"Skipping non-file: {file_name}")
            continue

        if file_name.endswith((".safetensors", ".pdparams")):
            if dest_path.endswith(".pdparams"):
                basename = os.path.basename(dest_path)
                dest_path = dest_path.replace(basename, "model.safetensors")

            if not args.overwrite and os.path.exists(dest_path):
                print(f"File already exists: {dest_path}. Skip conversion.")
                continue
            print(f"Processing params file: {file_name}")
            processed_index, cur_size = process_params_file(args, src_path, dest_path, src_prefix, dst_prefix)
            new_total_size += cur_size.item()

            if use_index_file:
                dst_index['weight_map'].update(processed_index)
        else:
            print(f"Copying non-tensor file: {file_name}")
            shutil.copy(src_path, dest_path)

        # Save updated index file
        if use_index_file:
            dst_index['metadata']['total_size'] = new_total_size
            with open(os.path.join(dest_dir, "model.safetensors.index.json"), "w") as f:
                json.dump(dst_index, f, indent=4)

    # Update config files unless skipped
    if not skip_config_update:
        update_config_files(args, dest_dir)
        update_tokenizer_config_files(dest_dir)
    else:
        print("Skipping config file updates as requested")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Convert Paddle safetensors model to PyTorch format.")

    parser.add_argument(
        "--src_dir",
        type=str,
        default="paddle_models/ERNIE-4.5-0.3B",
        help="Path to destination PaddlePaddle model directory",
    )

    parser.add_argument(
        "--dest_dir",
        type=str,
        default="torch_models/ERNIE-4.5-0.3B",
        help="Path to source PyTorch model directory",
    )

    parser.add_argument(
        "--use_moe",
        action="store_true",
        help="Enable Mixture of Experts (MoE) in the source model",
    )

    parser.add_argument("--src_prefix", type=str, default="ernie.", help="Source key prefix to replace")

    parser.add_argument("--dst_prefix", type=str, default="model.", help="Destination key prefix")

    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination directory if it exists")

    parser.add_argument("--skip_config_update", action="store_true", help="Skip updating config files")

    return parser.parse_args()


def main() -> None:
    """Main function to execute the conversion process."""
    args = parse_arguments()
    print(args)
    # Set default destination directory if not specified
    dest_dir = args.dest_dir

    print("Starting model conversion")
    print(f"Source path: {args.src_dir}")
    print(f"Destination path: {dest_dir}")
    print(f"Source prefix: {args.src_prefix}")
    print(f"Destination prefix: {args.dst_prefix}")

    try:
        process_model_directory(
            args=args,
            src_dir=args.src_dir,
            dest_dir=dest_dir,
            src_prefix=args.src_prefix,
            dst_prefix=args.dst_prefix,
            skip_config_update=args.skip_config_update,
        )
        print("Model conversion completed successfully")
    except Exception as e:
        print(f"Error during model conversion: {e!s}")
        raise


if __name__ == "__main__":
    main()
