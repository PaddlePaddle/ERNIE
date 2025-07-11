import os
import paddle
from safetensors.numpy import load_file
from safetensors.torch import save_file as save_safetensors
import torch
import numpy as np
import json
import argparse


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Convert Paddle safetensors model to PyTorch format.")

    parser.add_argument(
        "--src_dir",
        type=str,
        default="paddle_models/ERNIE-4.5-vl-28B",
        help="Path to destination PaddlePaddle model directory",
    )

    parser.add_argument(
        "--dest_dir",
        type=str,
        default="torch_models/ERNIE-4.5-vl-28B",
        help="Path to source PyTorch model directory",
    )

    return parser.parse_args()

def convert_pdparams_to_safetensors(pdparams_path, safetensors_path, config):
    """
    Convert a PaddlePaddle .pdparams file to a .safetensors file, transposing .weight tensors.
    Args:
        pdparams_path (str): Path to the input .pdparams file.
        safetensors_path (str): Path to the output .safetensors file.
    """
    print("----------------------------------------------------------------")
    print("pdparams_path:", pdparams_path)
    paddle.set_device('cpu')
    # Load the PaddlePaddle model state dictionary
    torch_state_dict = {}
    weight_map = {}
    pd_tensors = load_file(pdparams_path)
    for key, param in pd_tensors.items():
        if param.dtype != 'float32':
            param = (param.astype(np.uint32) << 16).view(np.float32)
        if key.endswith('.weight') and \
            "embed_tokens" not in key and \
            ".gate." not in key and \
            param.ndim == 2:
            param = param.T  # Transpose the parameter
        # vision model参数为float16
        tensor = torch.from_numpy(param)
        if 'vision' in key:
            tensor = tensor.to(torch.float16)
        elif 'mlp.gate.weight' not in key and 'mlp.moe_statics.e_score_correction_bias' not in key:
            tensor = tensor.to(torch.bfloat16)

        key = key.replace('ernie.', 'model.')
        if 'vision' not in key:
            if 'qkv_proj' in key:
                N, D = tensor.shape
                num_heads = config['num_attention_heads']
                kv_heads = config['num_key_value_heads']
                head_dim = D // num_heads

                hidden_size = head_dim * num_heads
                kv_hidden_size = head_dim * kv_heads

                assert (
                    N == hidden_size + 2 * kv_hidden_size
                ), f"{N} != {hidden_size} + {2*kv_hidden_size}, \
                    qkv_proj.weight dimensions mismatch the model configuration, \
                    please check config.json"
                q_proj, k_proj, v_proj = torch.split(tensor, [hidden_size, kv_hidden_size, kv_hidden_size], dim=0)
                torch_state_dict[key.replace('qkv_proj', 'q_proj')] = q_proj.contiguous()
                torch_state_dict[key.replace('qkv_proj', 'k_proj')] = k_proj.contiguous()
                torch_state_dict[key.replace('qkv_proj', 'v_proj')] = v_proj.contiguous()
            elif 'up_gate_proj' in key:
                fused_dim, hidden_size = tensor.shape
                # auto infer intermediate_size
                assert fused_dim % 2 == 0, f"Cannot split tensor with odd dimension {fused_dim}. \
                    Specify intermediate_size."
                intermediate_size = fused_dim // 2

                assert (
                    fused_dim == 2 * intermediate_size
                ), f"Tensor shape {tensor.shape} does not match 2 * intermediate_size {2 * intermediate_size}"

                gate_proj, up_proj = torch.split(tensor, [intermediate_size, intermediate_size], dim=0)
                torch_state_dict[key.replace('up_gate_proj', 'gate_proj')] = gate_proj.contiguous()
                torch_state_dict[key.replace('up_gate_proj', 'up_proj')] = up_proj.contiguous()
            else:
                torch_state_dict[key] = tensor.contiguous()
        else:
            torch_state_dict[key] = tensor.contiguous()
    # Save the state_dict as a .safetensors file
    save_safetensors(torch_state_dict, safetensors_path)

def split_index(index):
    new_weight_map = {}
    for k, v in index['weight_map'].items():
        k = k.replace('ernie.', 'model.')
        if 'vision' not in k:
            if 'qkv_proj' in k:
                new_weight_map[k.replace('qkv_proj', 'q_proj')] = v
                new_weight_map[k.replace('qkv_proj', 'k_proj')] = v
                new_weight_map[k.replace('qkv_proj', 'v_proj')] = v
            elif 'up_gate_proj' in k:
                new_weight_map[k.replace('up_gate_proj', 'gate_proj')] = v
                new_weight_map[k.replace('up_gate_proj', 'up_proj')] = v
            else:
                new_weight_map[k] = v
        else:
            new_weight_map[k] = v
    index['weight_map'] = new_weight_map

    return index

def convert_multiple_pdparams_to_safetensors(input_dir, output_dir):
    """
    Convert all .pdparams files in a directory to .safetensors files.
    Args:
        input_dir (str): Directory containing .pdparams files.
        output_dir (str): Directory to save .safetensors files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(input_dir, 'config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)

    with open(os.path.join(input_dir, 'model.safetensors.index.json'), 'r', encoding='utf-8') as f:
        index = json.load(f)
    index = split_index(index)
    with open(os.path.join(output_dir, 'model.safetensors.index.json'), 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.safetensors'):
            pdparams_path = os.path.join(input_dir, filename)
            safetensors_path = os.path.join(output_dir, filename)
            convert_pdparams_to_safetensors(pdparams_path, safetensors_path, config)
            print(f"Converted {filename} to {safetensors_path}")

# Example usage
if __name__ == "__main__":
    args = parse_arguments()

    convert_multiple_pdparams_to_safetensors(args.src_dir, args.dest_dir)
