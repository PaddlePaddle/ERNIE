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

unset PADDLE_TRAINERS_NUM
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
export PYTHONPATH=$(dirname "$0")/../../../..:$PYTHONPATH
export FLAGS_set_to_1d=False
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_dataloader_use_file_descriptor=False

master_ip=${1:-}
nnodes=${2:-2}
model_path="ERNIE4.5T_chat"
task="sft_fp_8k"
paddle_log_dir="${model_path}_${task}_log"
vdl_log_dir="${model_path}_${task}_vdl"
output_dir="${model_path}_${task}_checkpoint"

rm -rf ${log_dir}

python -m paddle.distributed.launch \
    --log_dir ${paddle_log_dir} \
    --gpus 0,1,2,3,4,5,6,7 \
    --master ${master_ip}:8080 \
    --nnodes ${nnodes} \
    examples/post-training/sft/train.py \
    --logging_dir ${vdl_log_dir} \
    --model_name_or_path ${model_path} \
    --output_dir ${output_dir} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --train_dataset_path "examples/data/sft-train.jsonl" \
    --train_dataset_prob "1.0" \
    --train_dataset_type "erniekit" \
    --eval_dataset_path "examples/data/sft-eval.jsonl" \
    --eval_dataset_prob "1.0" \
    --eval_dataset_type "erniekit" \
    --max_steps 100 \
    --max_evaluate_steps 10000 \
    --num_train_epochs 1 \
    --save_steps 10000000 \
    --logging_steps 1 \
    --eval_steps 10000 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --tensor_parallel_degree 8 \
    --pipeline_parallel_degree 2 \
    --sharding stage1 \
    --max_seq_len 32768 \
    --seed 23 \
    --gradient_accumulation_steps 8 \
    --warmup_steps 20 \
    --weight_decay 0.1 \
    --learning_rate 1e-5 \
    --min_lr 1e-6 \
    --num_samples_each_epoch 6000000 \
    --bf16 \
    --fp16_opt_level O2 \
    --disable_tqdm True \
    --recompute 1 \
    --recompute_granularity "full" \
    --dataloader_num_workers 1 \
    --distributed_dataloader 0 \
    --amp_custom_white_list "lookup_table" "lookup_table_v2" "flash_attn" "matmul" "matmul_v2" "fused_gemm_epilogue" \
    --amp_custom_black_list "reduce_sum" "softmax_with_cross_entropy" "c_softmax_with_cross_entropy" "elementwise_div" "sin" "cos" \
    --use_flash_attention 1 \
    --use_sparse_head_and_loss_fn 1 \
    --use_attn_mask_start_row_indices 1 \
    --pipeline_parallel_config "enable_delay_scale_loss enable_release_grads disable_partial_send_recv" \
    --greedy_intokens 1 \
	--release_grads 1 \
    --lr_scheduler_type cosine \
    --sequence_parallel 1 \
    --moe_group "mp" \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --fuse_rope 1 \
    --disable_ckpt_quant 1 \
    --recompute_use_reentrant True \
    --weight_quantize_algo "fp8linear" \
    --apply_hadamard True \
    --optim "adamw_custom" \
    --use_lowprecision_moment True \
    --tensorwise_offload_optimizer True \
    --pp_seg_method "[0,29,57]" \
    --optim_shard_num 8 \
    --unified_checkpoint_config "ignore_merge_optimizer" \
    --num_nextn_predict_layers 0 \
    # --ignore_save_lr_and_optim 1 \
    # --ignore_load_lr_and_optim 1 \
