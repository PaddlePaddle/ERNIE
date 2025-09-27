#!/usr/bin/env bash

set -euxo pipefail

# 在这里配置参数
DEVICES="0,1,2,3"
INPUT_DIR="/home/ERNIE/examples/pre-training/temp0/TP2(EP2)_PP2_to_Sd2_TP2(EP2)/checkpoint-5"          # 需要合并的分布式 ckpt 目录
OUTPUT_DIR="/home/ERNIE/examples/pre-training/output/temp1/merged_checkpoint"       # 输出 safetensors 分片目录
# AOA_CFG="none"   
# 直接在脚本里写 AOA 配置（JSON 字符串），或设为 none 关闭
AOA_CFG=$(cat << 'EOF'
{
  "aoa_statements": [
        "ernie.layers.$LAYER_ID.self_attn.qkv_proj.weight -> ernie.layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv_old, num_heads=20, num_key_value_groups=5",
        "ernie.layers.$LAYER_ID.self_attn.qkv_proj.weight.moment1_0 -> ernie.layers.$LAYER_ID.self_attn.qkv_proj.weight.moment1_0, fused_qkv_old, num_heads=20, num_key_value_groups=5",
        "ernie.layers.$LAYER_ID.self_attn.qkv_proj.weight.moment2_0 -> ernie.layers.$LAYER_ID.self_attn.qkv_proj.weight.moment2_0, fused_qkv_old, num_heads=20, num_key_value_groups=5",
        "ernie.layers.$LAYER_ID.self_attn.qkv_proj.weight.w_0 -> ernie.layers.$LAYER_ID.self_attn.qkv_proj.weight.w_0, fused_qkv_old, num_heads=20, num_key_value_groups=5",

        "ernie.mtp_block.$LAYER_ID.self_attn.qkv_proj.weight -> ernie.mtp_block.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv_old, num_heads=20, num_key_value_groups=5",
        "ernie.mtp_block.$LAYER_ID.self_attn.qkv_proj.weight.moment1_0 -> ernie.mtp_block.$LAYER_ID.self_attn.qkv_proj.weight.moment1_0, fused_qkv_old, num_heads=20, num_key_value_groups=5",
        "ernie.mtp_block.$LAYER_ID.self_attn.qkv_proj.weight.moment2_0 -> ernie.mtp_block.$LAYER_ID.self_attn.qkv_proj.weight.moment2_0, fused_qkv_old, num_heads=20, num_key_value_groups=5",
        "ernie.mtp_block.$LAYER_ID.self_attn.qkv_proj.weight.w_0 -> ernie.mtp_block.$LAYER_ID.self_attn.qkv_proj.weight.w_0, fused_qkv_old, num_heads=20, num_key_value_groups=5",

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
        "ernie.layers.1.mlp.shared_experts.up_gate_proj.weight.w_0 -> ernie.layers.1.mlp.shared_experts.up_gate_proj.weight.w_0, fused_ffn"
  ]
}
EOF
)

# 如需从 safetensors 源加载，开启此项
EXTRA_ARGS=( )
# EXTRA_ARGS=( --src-safetensors )

export FLAGS_call_stack_level=2
export PYTHONUNBUFFERED=1

echo "[INFO] Python: $(which python)"
python -V || true
python - <<'PY'
import sys
try:
    import paddle
    print('[INFO] paddle version:', paddle.__version__)
except Exception as e:
    print('[WARN] paddle import failed:', e, file=sys.stderr)
PY

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "[ERROR] INPUT_DIR not found: ${INPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs"

python -m paddle.distributed.launch --devices "${DEVICES}" \
  --log_dir "${OUTPUT_DIR}/logs" \
  /home/ERNIE/examples/pre-training/transform_from_distcp_to_safetensors.py \
  "${INPUT_DIR}" "${OUTPUT_DIR}" "${AOA_CFG}" "${EXTRA_ARGS[@]}"

echo "Done. Output saved to: ${OUTPUT_DIR}"

echo "==== tail worker logs ===="
for f in "${OUTPUT_DIR}"/logs/workerlog.*; do
  echo "--- ${f} ---"; tail -n 50 "${f}" || true; echo
done
