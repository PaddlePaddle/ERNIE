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
task="dpo_lora_8k"
paddle_log_dir="${model_path}_${task}_log"
vdl_log_dir="${model_path}_${task}_vdl"
output_dir="${model_path}_${task}_checkpoint"

rm -rf ${log_dir}

python -m paddle.distributed.launch \
    --log_dir ${paddle_log_dir} \
    --gpus 0,1,2,3,4,5,6,7 \
    --master ${master_ip}:8080 \
    --nnodes ${nnodes} \
    ./examples/post-training/dpo/dpo_train.py \
    --logging_dir ${vdl_log_dir} \
    --model_name_or_path ${model_path} \
    --output_dir ${output_dir} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --train_dataset_path "examples/data/dpo-train.jsonl" \
    --train_dataset_prob "1.0" \
    --train_dataset_type "erniekit" \
    --eval_dataset_path "examples/data/dpo-eval.jsonl" \
    --eval_dataset_prob "1.0" \
    --eval_dataset_type "erniekit" \
    --max_evaluate_steps 10000 \
    --num_train_epochs 1 \
    --max_steps 800 \
    --save_steps 100 \
    --logging_steps 1 \
    --eval_steps 20000 \
    --weight_decay 0.1 \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --tensor_parallel_degree 8 \
    --tensor_parallel_config "sync_param sync_grad sync_moment" \
    --pipeline_parallel_degree $nnodes \
    --sharding_parallel_degree 1 \
    --gradient_accumulation_steps 36 \
    --sharding stage1 \
    --max_seq_len 32768  \
    --seed 42 \
    --warmup_steps 50 \
    --learning_rate 5e-7 \
    --bf16 \
    --fp16_opt_level O2 \
    --disable_tqdm True \
    --recompute 1 \
    --recompute_granularity "full" \
    --dataloader_num_workers 4 \
    --distributed_dataloader 1 \
    --amp_custom_white_list "lookup_table" "lookup_table_v2" "flash_attn" "matmul" "matmul_v2" "fused_gemm_epilogue" \
    --amp_custom_black_list "reduce_sum" "softmax_with_cross_entropy" "c_softmax_with_cross_entropy" "elementwise_div" "sin" "cos" \
    --pipeline_parallel_config "disable_partial_send_recv enable_clear_every_step_cache enable_delay_scale_loss enable_overlap_p2p_comm best_unbalanced_scheduler" \
    --dpo_benchmark 0 \
    --greedy_intokens 1 \
    --beta 0.1 \
    --loss_type "sigmoid" \
    --label_smoothing 0.0 \
    --pref_loss_ratio 1.0 \
    --sft_loss_ratio 0.0 \
    --ref_model_update_steps -1 \
    --sequence_parallel 1 \
    --use_attn_mask_start_row_indices 1 \
    --tensor_parallel_output 1 \
    --reference_free 0 \
    --simpo_gamma 0.5 \
    --recompute_use_reentrant 1	\
    --moe_group mp \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --hidden_dropout_prob 0 \
    --attention_probs_dropout_prob 0.1 \
    --dropout_warmup_steps 100 \
    --adam_epsilon 1e-8 \
    --layerwise_lr_decay_bound 0.5 \
    --use_sp_callback 1 \
    --save_total_limit 5 \
    --scale_loss 8192 \
	--release_grads 1 \
    --amp_master_grad 1 \
    --lr_scheduler_type "cosine" \
    --min_lr 5e-7 \
    --fuse_rope 1 \
    --offset_alpha 0.0 \
    --unified_checkpoint_config "async_save" \
    --lora 1 \
    --lora_rank 32 \
    --lora_alpha 128 \
    --lora_plus_scale 12 \
    --rslora
