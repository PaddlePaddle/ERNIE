import os
import sys
import paddle
import paddle.distributed as dist


def convert_distcp_to_safetensors(input_dir: str, output_dir: str, config_flag: bool = False):
    """
    将分布式检查点合并为单目录（可选择启用内置 AoA 配置）。

    Args:
        input_dir (str): 需要合并的分布式 ckpt 路径（如 checkpoint-N 目录）。
        output_dir (str): 合并后的输出目录。
        config_flag (bool): True 使用内置 AoA 配置，False 使用 None。
    """

    config = None
    if config_flag:
        # 内置 AoA 配置（与你提供的一致）
        config = {
            "aoa_statements": [
                "ernie.layers.$LAYER_ID.self_attn.qkv_proj.weight -> ernie.layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv_old, num_heads=20, num_key_value_groups=4",
                "ernie.layers.$LAYER_ID.self_attn.qkv_proj.weight.moment1_0 -> ernie.layers.$LAYER_ID.self_attn.qkv_proj.weight.moment1_0, fused_qkv_old, num_heads=20, num_key_value_groups=4",
                "ernie.layers.$LAYER_ID.self_attn.qkv_proj.weight.moment2_0 -> ernie.layers.$LAYER_ID.self_attn.qkv_proj.weight.moment2_0, fused_qkv_old, num_heads=20, num_key_value_groups=4",
                "ernie.layers.$LAYER_ID.self_attn.qkv_proj.weight.w_0 -> ernie.layers.$LAYER_ID.self_attn.qkv_proj.weight.w_0, fused_qkv_old, num_heads=20, num_key_value_groups=4",

                "ernie.mtp_block.$LAYER_ID.self_attn.qkv_proj.weight -> ernie.mtp_block.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv_old, num_heads=20, num_key_value_groups=4",
                "ernie.mtp_block.$LAYER_ID.self_attn.qkv_proj.weight.moment1_0 -> ernie.mtp_block.$LAYER_ID.self_attn.qkv_proj.weight.moment1_0, fused_qkv_old, num_heads=20, num_key_value_groups=4",
                "ernie.mtp_block.$LAYER_ID.self_attn.qkv_proj.weight.moment2_0 -> ernie.mtp_block.$LAYER_ID.self_attn.qkv_proj.weight.moment2_0, fused_qkv_old, num_heads=20, num_key_value_groups=4",
                "ernie.mtp_block.$LAYER_ID.self_attn.qkv_proj.weight.w_0 -> ernie.mtp_block.$LAYER_ID.self_attn.qkv_proj.weight.w_0, fused_qkv_old, num_heads=20, num_key_value_groups=4",

                "ernie.layers.$LAYER_ID.mlp.up_gate_proj.weight -> ernie.layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
                "ernie.layers.$LAYER_ID.mlp.up_gate_proj.weight.moment1_0 -> ernie.layers.$LAYER_ID.mlp.up_gate_proj.weight.moment1_0, fused_ffn",
                "ernie.layers.$LAYER_ID.mlp.up_gate_proj.weight.moment2_0 -> ernie.layers.$LAYER_ID.mlp.up_gate_proj.weight.moment2_0, fused_ffn",
                "ernie.layers.$LAYER_ID.mlp.up_gate_proj.weight.w_0 -> ernie.layers.$LAYER_ID.mlp.up_gate_proj.weight.w_0, fused_ffn",

                "ernie.mtp_block.$LAYER_ID.mlp.up_gate_proj.weight -> ernie.mtp_block.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
                "ernie.mtp_block.$LAYER_ID.mlp.up_gate_proj.weight.moment1_0 -> ernie.mtp_block.$LAYER_ID.mlp.up_gate_proj.weight.moment1_0, fused_ffn",
                "ernie.mtp_block.$LAYER_ID.mlp.up_gate_proj.weight.moment2_0 -> ernie.mtp_block.$LAYER_ID.mlp.up_gate_proj.weight.moment2_0, fused_ffn",
                "ernie.mtp_block.$LAYER_ID.mlp.up_gate_proj.weight.w_0 -> ernie.mtp_block.$LAYER_ID.mlp.up_gate_proj.weight.w_0, fused_ffn",

                "ernie.layers.1.mlp.shared_experts.up_gate_proj.weight -> ernie.layers.1.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
                "ernie.layers.1.mlp.shared_experts.up_gate_proj.weight.moment1_0 -> ernie.layers.1.mlp.shared_experts.up_gate_proj.weight.moment1_0, fused_ffn",
                "ernie.layers.1.mlp.shared_experts.up_gate_proj.weight.moment2_0 -> ernie.layers.1.mlp.shared_experts.up_gate_proj.weight.moment2_0, fused_ffn",
                "ernie.layers.1.mlp.shared_experts.up_gate_proj.weight.w_0 -> ernie.layers.1.mlp.shared_experts.up_gate_proj.weight.w_0, fused_ffn",

                # "ernie.layers.2.mlp.shared_experts.up_gate_proj.weight -> ernie.layers.2.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
                # "ernie.layers.2.mlp.shared_experts.up_gate_proj.weight.moment1_0 -> ernie.layers.2.mlp.shared_experts.up_gate_proj.weight.moment1_0, fused_ffn",
                # "ernie.layers.2.mlp.shared_experts.up_gate_proj.weight.moment2_0 -> ernie.layers.2.mlp.shared_experts.up_gate_proj.weight.moment2_0, fused_ffn",
                # "ernie.layers.2.mlp.shared_experts.up_gate_proj.weight.w_0 -> ernie.layers.2.mlp.shared_experts.up_gate_proj.weight.w_0, fused_ffn",

            ]
        }

    os.makedirs(output_dir, exist_ok=True)


    dist.flex_checkpoint.dcp.load_state_dict.merge_sharded_state_dict(
        input_dir,
        output_dir,
        offload=False,
        aoa_config=config,
        safetensors=False,
    )



if __name__ == "__main__":
    # CLI: python transform_from_distcp_to_safetensors.py <input_dir> <output_dir> [config_flag]
    if len(sys.argv) < 3:
        print("用法: python transform_from_distcp_to_safetensors.py <input_dir> <output_dir> [config_flag]")
        sys.exit(1)
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    cfg_flag = False
    if len(sys.argv) >= 4:
        val = str(sys.argv[3]).strip().lower()
        cfg_flag = val in ["1", "true", "yes", "y"]
    convert_distcp_to_safetensors(in_dir, out_dir, cfg_flag)