#!/bin/bash

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

# ==================== 环境变量设置 ====================
echo "设置环境变量..."

export CUDA_MODULE_LOADING=LAZY
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
unset GLOG_vmodule GLOG_v
export PADDLE_DISABLE_CUDNN_FA=1
export FLAGS_use_auto_growth_pinned_allocator=True
export FLAGS_pipeline_nccl_comm_init_option=1
export FLAGS_sharding_v2_check_zero_padding=1
export FLAGS_use_paddle_recall_error=0
export FLAGS_tcp_max_syn_backlog=16384
export FLAGS_call_stack_level=2
# 限制使用的 GPU 数量
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# 检查 GPU 计算能力
SM=`nvidia-smi --query-gpu=compute_cap --format=csv | tail -n 1 | sed 's/\.//g'`
echo "GPU 计算能力: $SM"
# if [ $SM -eq 90 ]
# then
#     export FLAGS_flash_attn_version=3
# else
#     export FLAGS_flash_attn_version=2
# fi
# 设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:./ernie

# ==================== 创建临时配置文件 ====================
echo "创建临时配置文件..."

TEMP_CONFIG_FILE="/tmp/pretrain_config_$$.yaml"

cat > $TEMP_CONFIG_FILE << 'EOF'
# -----------环境变量----------------------#
env:
    HOME: null

# ---------------------------model args-------------------------------------------------#
model_args:
    model_name_or_path: model_configs/ERNIE-4p5-21B-A3B/
    tokenizer_name: ./ernie/src/tokenizers/tokenizer_model
    output_dir: ./output1
    data_load_process_num: 40
    max_seq_length: 1024
    base_seq_length: 1024
    num_consecutive: 32

    enable_global_training_logs: False
    enable_mtp_magic_send: True
    moe_use_aux_free_update_coef: 0.001
    global_logging_interval: 1

    model_config:
        hidden_size: 320
        intermediate_size: 5504
        use_quant_before_a2a: true
        use_async_a2a: false
        use_rms_qkv_recompute: false
        moe_logging: true
        use_recompute: false
        multi_token_pred_depth: 1
        use_fp8_mlp: false
        num_hidden_layers: 5
        remove_tail_layer: 0
        use_fp8_fuse_node: false
        use_flash_attention: 0
        fuse_attention_qkv: true
        fuse_attention_ffn: true
        use_bias: false
        use_ep_comm_overlap: false
        fp8_mem_configs:
            recompute_fwd_gate_up: false
            dequant_input: true
            shared_expert: false
        fp8_fused_ops_configs:
            stack_quant: true
            swiglu_probs_bwd: true
            split_group_gemm: true
            spaq: true
            transpose_split_quant: true
        use_combine_before_a2a: true

# ---------------------------trainer args-------------------------------------------------#
trainer_args:
    input_dir: "0.4 ./demo_data/data-1-part0 0.6 ./demo_data/data-1-part0"
    split: "998,1,1"
    gc_interval: 100000
    use_ortho_loss_callback: true
    do_train: True
    dataloader_num_workers: 8
    prefetch_factor: 32
    overwrite_output_dir: 1
    disable_tqdm: 1
    report_to: none
    logging_steps: 1
    eval_steps: 1000000
    eval_iters: -1
    save_steps: 1
    max_steps: 1
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_epsilon: 1e-8
    learning_rate: 3.14e-4
    min_lr: 3.14e-6
    gradient_accumulation_steps: 16
    per_device_train_batch_size: 1
    # resume_from_checkpoint: ./output/checkpoint-20

    lr_scheduler: wsd:603000
    decay_function: 1-sqrt
    max_grad_norm: 1.0
    weight_decay: 0.1
    warmup_steps: 2000
    save_total_limit: 5
    fp16: True
    fp16_opt_level: "O2"
    scale_loss: 4096
    seed: 42
    use_train_part_sharding: 1
    pre_alloc_memory: 60
    sharding_comm_buffer_size_MB: 2048
    offload_optim: true
    data_parallel_degree: 1
    tensor_parallel_degree: 1
    expert_parallel_degree: 1
    pipeline_parallel_degree: 4
    sharding: "stage1"
    sharding_parallel_degree: 1
    sharding_parallel_config: split_param
    amp_master_grad: 1

    ignore_data_skip: 0
    same_data: True
    enable_timer: 1
    skip_profile_timer: False
    skip_load_data_seq_cache: 1

    load_sharded_model: True
    save_sharded_model: True
    ignore_load_lr_and_optim: False
    moe_with_send_router_loss: False

    use_moe: true
    moe_group: dummy
    from_scratch: 1
    enable_optimizer_timer: False
    using_flex_checkpoint: true
EOF

echo "配置文件已创建: $TEMP_CONFIG_FILE"

# ==================== 开始训练 ====================
echo "开始训练..."

python -m paddle.distributed.launch \
    --master 127.0.0.1:29500 \
    --nnodes 1 \
    --run_mode=collective \
    /home/ERNIE/examples/pre-training/ernie/pretrain.py  \
    --config $TEMP_CONFIG_FILE

# ==================== 清理临时文件 ====================
echo "清理临时文件..."
rm -f $TEMP_CONFIG_FILE

echo "训练完成！" 