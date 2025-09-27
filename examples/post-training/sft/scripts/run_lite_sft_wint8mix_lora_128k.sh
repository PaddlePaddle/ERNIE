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

export NCCL_DEBUG=WARN
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

model_path="ERNIE-4.5-21B-A3B"
task="sft_wint8mix_lora_128k"
paddle_log_dir="${model_path}_${task}_log"
vdl_log_dir="${model_path}_${task}_vdl"
output_dir="${model_path}_${task}_checkpoint"

rm -rf ${log_dir}

python -m paddle.distributed.launch \
    --log_dir ${paddle_log_dir} \
    --gpus 0,1,2,3 \
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
    --tensor_parallel_degree 4 \
    --pipeline_parallel_degree 1 \
    --sharding_parallel_degree 1 \
    --sharding stage1 \
    --max_seq_len 131072 \
    --seed 23 \
    --gradient_accumulation_steps 8 \
    --warmup_steps 20 \
    --weight_decay 0.1 \
    --learning_rate 3e-4 \
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
    --pipeline_parallel_config "disable_partial_send_recv enable_clear_every_step_cache" \
    --greedy_intokens 1 \
	--release_grads 1 \
    --lr_scheduler_type cosine \
    --sequence_parallel 1 \
    --moe_group "mp" \
    --amp_master_grad 1 \
    --fuse_rope 1 \
    --disable_ckpt_quant 1 \
    --recompute_use_reentrant True \
    --unified_checkpoint_config "async_save" \
    --lora \
    --lora_rank 32 \
    --weight_quantize_algo weight_only_mix
