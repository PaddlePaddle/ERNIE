# Ascend NPU && ERNIE-4.5-300B-A47B-Base & ERNIE-4.5-300B-A47B Training Quick Start
##  🚀 Quick Start 🚀
The hardware model required for this training test is: Atalas 200T A2 BOX*16;
Memory: 64G;
Core(s) per socket: 48;
Socket(s): 2
Note: Only x86 Ascend NPU machines have been verified in this test, and ARM machines have not been verified yet.
## Instructions for the Minimum Number of NPU Cards Required for Training
* SFT: At least 128 cards of 64G Ascend cards, with an additional 32 cards recommended. (Since 128 cards use the offload_optim method to the CPU for memory saving, it is suggested to increase to 10×16 cards)
* LoRA: At least 16 cards of 64G Ascend cards
## 01. Environment Preparation
1. Pull the Image
Use uname -a to check if the architecture is x86 or arm.

```
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann80RC1-ubuntu20-x86_64-gcc84-py310 #x86
```
2. Start the Container


```
sudo docker run -it --name paddle_test   \
-v /home/:/home/  \
-v `pwd`:/workspace  \
-w /workspace  \
--privileged --network=host --shm-size=128G  \
-v /ssd1/dataset:/workspace/dataset  \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver  \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi  \
-v /usr/local/dcmi:/usr/local/dcmi  \
-e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"  \
registry.baidubce.com/device/cann80RC1-ubuntu20-x86_64-gcc84-py310 /bin/bash
```
3. Paddle Deep Learning Framework Installation

```
python -m pip install paddlepaddle==3.1.0a0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

```
4. Install PaddleCustomDevice

```
python -m pip install paddle-custom-npu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/npu/
```
* For more basic environment preparation (such as source code compilation and installation of PaddleCustomDevice, updating CANN, etc.), please refer to:
https://bbs.huaweicloud.com/blogs/452538

5. Commonly Used Environment Variables for Training
It is recommended to set the following environment variables in advance before training.

* **NPU online compilation: false for off, it is recommended to turn off (important)**
```
export FLAGS_npu_jit_compile=false
```

* Specify the NPU card number (take 16 cards as an example):
```
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```
* Paddle memory allocation method:
The default is to apply for the maximum value, which may introduce memory problems in individual models. It is recommended to use `auto_growth` to apply as needed.

```
export FLAGS_allocator_strategy_kernel=auto_growth
```
* For more other commonly used environment variables, please refer to:
https://bbs.huaweicloud.com/blogs/452539

6. Install requirements

```
cd ERRNIEKit
pip install -r requirements/gpu/requirements.txt
```
### (2) Start post-traning：(This will take a relatively long time)
**Note: After launching the script, you’ll need to wait for an extended period before the training loss appears normally. Please be patient.**

1. SFT fine-tuning
```
#!/bin/bash
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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
export FLAGS_allocator_strategy_kernel=auto_growth
export FLAGS_npu_jit_compile=0
#export LD_LIBRARY_PATH=/usr/local/cuda-12.3/compat/:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
#export ASCEND_CUSTOM_PATH=source /usr/local/Ascend
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export ASCEND_GLOBAL_LOG_LEVEL=0
#export CUSTOM_DEVICE_BLACK_LIST="scatter"
master_ip=141.61.41.162
nnodes=8
model_path="ERNIE-4.5-300B-A47B"
task="sft_8k"
paddle_log_dir="${model_path}_${task}_log"
vdl_log_dir="${model_path}_${task}_vdl"
output_dir="${model_path}_${task}_checkpoint"
rm -rf ${log_dir}

mkdir -p logs
python -m paddle.distributed.launch \
        --log_dir ${paddle_log_dir} \
        --npus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
        --master ${master_ip}:12123 \
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
        --max_steps 200 \
        --max_evaluate_steps 10000 \
        --num_train_epochs 1 \
        --save_steps 10000000 \
        --logging_steps 1 \
        --eval_steps 500 \
        --weight_decay 0.01 \
        --do_train \
        --do_eval \
        --evaluation_strategy steps \
        --tensor_parallel_degree 8 \
        --pipeline_parallel_degree 16 \
        --sharding_parallel_degree 1 \
        --sharding stage1 \
        --max_seq_len 8192 \
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
        --pipeline_parallel_config "enable_delay_scale_loss enable_release_grads disable_batch_p2p_comm  disable_partial_send_recv" \
        --greedy_intokens 1 \
        --sequence_parallel 1 \
        --release_grads 1 \
        --moe_group "mp" \
        --moe_multimodal_dispatch_use_allgather "v2-alltoall-unpad" \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --adam_epsilon 1e-8 \
        --amp_master_grad 1 \
        --fuse_rope 0 \
        --disable_ckpt_quant 1 \
        --recompute_use_reentrant True \
        --unified_checkpoint_config "async_save" \
        --device "npu" \
        --use_sparse_flash_attn  False \
        --use_fused_head_and_loss_fn \
        --use_sparse_head_and_loss_fn \
        --offload_optim \
        --moe_use_aux_free False \
        --fuse_rms_norm False \
        --moe_multimodal_dispatch_use_allgather ""  2>&1 | tee "logs/sft_$(date +%Y%m%d_%H%M).log"

```
2. SFT-LoRA fine-tuning

```
#!/bin/bash
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#implied.
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
export FLAGS_allocator_strategy_kernel=auto_growth
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_dataloader_use_file_descriptor=False
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export FLAGS_npu_jit_compile=0
#export GLOG_v=3
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
#export CUSTOM_DEVICE_BLACK_LIST="scatter"
master_ip=localhost
nnodes=1
rank=3
model_path="ERNIE-4.5-300B-A47B"
task="sft_lora_8k"
paddle_log_dir="${model_path}_${task}_log"
vdl_log_dir="${model_path}_${task}_vdl"
output_dir="${model_path}_${task}_checkpoint"
rm -rf ${log_dir}

mkdir -p logs
python -m paddle.distributed.launch \
        --log_dir ${paddle_log_dir} \
        --npus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
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
        --max_steps 200 \
        --max_evaluate_steps 10000 \
        --num_train_epochs 1 \
        --save_steps 500 \
        --logging_steps 1 \
        --eval_steps 500 \
        --weight_decay 0.01 \
        --do_train \
        --evaluation_strategy steps \
        --tensor_parallel_degree 8 \
        --pipeline_parallel_degree 2 \
        --sharding_parallel_degree 1 \
        --sharding stage1 \
        --max_seq_len 8192 \
        --seed 23 \
        --gradient_accumulation_steps 2 \
        --warmup_steps 2000 \
        --lr_scheduler_type "linear" \
        --learning_rate 3e-4 \
        --num_samples_each_epoch 6000000 \
        --bf16 \
        --fp16_opt_level O2 \
        --amp_custom_white_list "lookup_table" "lookup_table_v2" "flash_attn" "matmul" "matmul_v2" "fused_gemm_epilogue" \
        --amp_custom_black_list "reduce_sum" "softmax_with_cross_entropy" "c_softmax_with_cross_entropy" "elementwise_div" "sin" "cos" \
        --disable_tqdm True \
        --recompute 1 \
        --offload_optim 1 \
        --recompute_granularity "full" \
        --dataloader_num_workers 1 \
        --distributed_dataloader 1 \
        --use_flash_attention 1 \
        --use_sparse_head_and_loss_fn 0 \
        --use_attn_mask_start_row_indices 1 \
        --pipeline_parallel_config "disable_partial_send_recv disable_batch_p2p_comm enable_clear_every_step_cache" \
        --greedy_intokens 1 \
        --lr_scheduler linear \
        --sequence_parallel 1 \
        --release_grads 1 \
        --recompute_use_reentrant True \
        --fuse_rope 0 \
        --lora \
        --lora_rank 32 \
        --moe_group mp \
        --device "npu" \
        --use_fused_head_and_loss_fn \
        --use_sparse_head_and_loss_fn \
        --offload_optim \
        --use_sparse_flash_attn  False \
        --moe_use_aux_free False \
        --fuse_rms_norm False \
        --moe_multimodal_dispatch_use_allgather ""  2>&1 | tee "logs/lora_$(date +%Y%m%d_%H%M).log"




```
