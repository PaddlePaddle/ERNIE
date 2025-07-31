# Metax GPU ERNIE-4.5-21B-A3B-Base & ERNIE-4.5-21B-A3B Training Quick Start


##  🚀 Quick Start🚀

### （0）Before starting, you need a Metax GPU machine, and the system requirements for this machine are as follows:

 | Chip type | Driver version |
 | --- | --- |
 | MetaX C550 | 2.15.9 |

#### Environment Description
- **Machine：** MetaX C550 64GB 8-card machine
- **Docker image：** ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
- **GCC path：** /usr/bin/gcc (9.4)
- **python version：** 3.10

**Note: This example uses an 8-card machine: To verify if your machine is Metax GPU, simply enter the command in the system environment and see if there is any output:**
```
mx-smi
#example：$ mx-smi
mx-smi  version: 2.2.4

=================== MetaX System Management Interface Log ===================
Timestamp                                         : Tue Jul 29 12:16:08 2025

Attached GPUs                                     : 8
+---------------------------------------------------------------------------------+
| MX-SMI 2.2.4                        Kernel Mode Driver Version: 2.15.9          |
| MACA Version: 2.32.0.6              BIOS Version: 1.25.1.0                      |
|------------------------------------+---------------------+----------------------+
| GPU         NAME                   | Bus-id              | GPU-Util             |
| Temp        Pwr:Usage/Cap          | Memory-Usage        | GPU-State            |
|====================================+=====================+======================|
| 0           MetaX C550             | 0000:0f:00.0        | 0%                   |
| 29C         53W / 450W             | 858/65536 MiB       | Available            |
+------------------------------------+---------------------+----------------------+
| 1           MetaX C550             | 0000:34:00.0        | 0%                   |
| 29C         53W / 450W             | 858/65536 MiB       | Available            |
+------------------------------------+---------------------+----------------------+
| 2           MetaX C550             | 0000:48:00.0        | 0%                   |
| 30C         54W / 450W             | 858/65536 MiB       | Available            |
+------------------------------------+---------------------+----------------------+
| 3           MetaX C550             | 0000:5a:00.0        | 0%                   |
| 29C         54W / 450W             | 858/65536 MiB       | Available            |
+------------------------------------+---------------------+----------------------+
| 4           MetaX C550             | 0000:87:00.0        | 0%                   |
| 30C         53W / 450W             | 858/65536 MiB       | Available            |
+------------------------------------+---------------------+----------------------+
| 5           MetaX C550             | 0000:ae:00.0        | 0%                   |
| 32C         55W / 450W             | 858/65536 MiB       | Available            |
+------------------------------------+---------------------+----------------------+
| 6           MetaX C550             | 0000:c2:00.0        | 0%                   |
| 32C         56W / 450W             | 858/65536 MiB       | Available            |
+------------------------------------+---------------------+----------------------+
| 7           MetaX C550             | 0000:d7:00.0        | 0%                   |
| 31C         56W / 450W             | 858/65536 MiB       | Available            |
+------------------------------------+---------------------+----------------------+

+---------------------------------------------------------------------------------+
| Process:                                                                        |
|  GPU                    PID         Process Name                 GPU Memory     |
|                                                                  Usage(MiB)     |
|=================================================================================|
|  no process found                                                               |
+---------------------------------------------------------------------------------+
```

### (1)  Environment Preparation: (This will take you 5-10 minutes)

1. Pull the Image
```
docker login --username=cr_temp_user --password=eyJpbnN0YW5jZUlkIjoiY3JpLXpxYTIzejI2YTU5M3R3M2QiLCJ0aW1lIjoiMTc1Mzg2OTU5MTAwMCIsInR5cGUiOiJzdWIiLCJ1c2VySWQiOiIyMDcwOTQwMTA1NjYzNDE3OTIifQ:7a622354f5a417b9cfba78f4ab7d8dff2cd9a65d cr.metax-tech.com && docker pull cr.metax-tech.com/public-library/maca-c500:2.33.0.6-ubuntu20.04-amd64
```

2. Install the driver kmd on host
```
配置APT源
curl -fsSL https://repos.metax-tech.com/public.gpg.key | apt-key add -
echo "deb [arch=$(dpkg --print-architecture)] https://repos.metax-tech.com/r/metax-driver-ubuntu/ stable main" | tee /etc/apt/sources.list.d/metax-driver-ubuntu.list
apt-get update

安装Metax Driver驱动
bash apt install metax-driver
```

3. Start the Container
```
docker run -it --device=/dev/dri --device=/dev/mxcd --device=/dev/infiniband  --group-add video --name paddle-metax-dev --network=host --uts=host --ipc=host   --privileged=true   --security-opt seccomp=unconfined --security-opt apparmor=unconfined --shm-size '100gb'  --ulimit memlock=-1 -v /data/:/data/ cr.metax-tech.com/public-library/maca-c500:2.33.0.6-ubuntu20.04-amd64  /bin/bash

docker exec -it paddle-metax-dev /bin/bash
```

4. Install paddlepaddle & paddle-metax-gpu
```
# Install PaddlePaddle CPU package
python -m pip install paddlepaddle==3.1.0a0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Install PaddlePaddle iluvatar-gpu plugin package
python -m pip install paddle-metax-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/maca/

Nightly version link:
https://www.paddlepaddle.org.cn/packages/nightly/maca/
```

4. Install requirements
```
pip install paddleformers
```

### (2) Start post-traning：(This will take a relatively long time)

1. SFT fine-tuning

```
#!/bin/bash

unset PADDLE_TRAINERS_NUM
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
export PYTHONPATH=$(dirname "$0")/../../../..:$PYTHONPATH
export FLAGS_set_to_1d=False
export FLAGS_dataloader_use_file_descriptor=False

export MACA_DIRECT_DISPATCH=1
export MCCL_ENABLE_FC=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MACA_SMALL_PAGESIZE_ENABLE=1
export SET_DEVICE_NUMA_PREFERRED=1

export PADDLE_XCCL_BACKEND=metax_gpu
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1

export FLAGS_embedding_deterministic=1

model_path="ERNIE-4.5-21B-A3B-Paddle"
task="sft_8k"
paddle_log_dir="${model_path}_${task}_log"
vdl_log_dir="${model_path}_${task}_vdl"
output_dir="${model_path}_${task}_checkpoint"

rm -rf ${log_dir}
python -m paddle.distributed.launch \
    --log_dir ${paddle_log_dir} \
    --gpus 0,1,2,3,4,5,6,7 \
    examples/post-training/sft/train.py \
    --logging_dir ${vdl_log_dir} \
    --model_name_or_path ${model_path} \
    --output_dir ${output_dir} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --train_dataset_path "./examples/post-training/data/all_ds/ARC-Challenge/train.json,./examples/post-training/data/all_ds/boolq/train.json,./examples/post-training/data/all_ds/piqa/train.json,./examples/post-training/data/all_ds/winogrande/train.json,./examples/post-training/data/all_ds/hellaswag/train.json,./examples/post-training/data/all_ds/ARC-Easy/train.json" \
    --train_dataset_prob "0.2,0.1,0.1,0.2,0.2,0.2" \
    --train_dataset_type "erniekit,erniekit,erniekit,erniekit,erniekit,erniekit" \
    --eval_dataset_path "./examples/post-training/data/all_ds/ARC-Challenge/dev.json,./examples/post-training/data/all_ds/boolq/dev.json,./examples/post-training/data/all_ds/piqa/dev.json,./examples/post-training/data/all_ds/winogrande/dev.json,./examples/post-training/data/all_ds/hellaswag/dev.json,./examples/post-training/data/all_ds/ARC-Easy/dev.json" \
    --eval_dataset_prob "0.2,0.1,0.1,0.2,0.2,0.2" \
    --eval_dataset_type "erniekit,erniekit,erniekit,erniekit,erniekit,erniekit" \
    --max_steps 500 \
    --max_evaluate_steps 1000 \
    --num_train_epochs 1 \
    --save_steps 500 \
    --logging_steps 1 \
    --eval_steps 500 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --tensor_parallel_degree 4 \
    --pipeline_parallel_degree 2 \
    --sharding_parallel_degree 1 \
    --sharding stage1 \
    --max_seq_len 8192 \
    --seed 23 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --learning_rate 3e-4 \
    --min_lr 1e-6 \
    --num_samples_each_epoch 6000000 \
    --bf16 \
    --fp16_opt_level O2 \
    --disable_tqdm True \
    --recompute 1 \
    --recompute_granularity "full" \
    --tensor_parallel_output 0 \
    --dataloader_num_workers 1 \
    --distributed_dataloader 0 \
    --amp_custom_white_list "lookup_table" "lookup_table_v2" "flash_attn" "matmul" "matmul_v2" "fused_gemm_epilogue" \
    --amp_custom_black_list "reduce_sum" "softmax_with_cross_entropy" "c_softmax_with_cross_entropy" "elementwise_div" "sin" "cos" \
    --use_flash_attention 1 \
    --use_sparse_head_and_loss_fn 0 \
    --use_attn_mask_start_row_indices 0 \
    --pipeline_parallel_config "disable_partial_send_recv enable_clear_every_step_cache disable_batch_p2p_comm" \
    --greedy_intokens 1 \
    --release_grads 1 \
    --lr_scheduler_type cosine \
    --sequence_parallel 1 \
    --moe_group "mp" \
    --amp_master_grad 1 \
    --fuse_rope 1 \
    --device "metax_gpu" \
    --use_sparse_flash_attn False \
    --disable_ckpt_quant 1 \
    --recompute_use_reentrant True \
    --unified_checkpoint_config "async_save" \
    --ignore_save_lr_and_optim 1 \
    --ignore_load_lr_and_optim 1 \
    --lora \
    --lora_rank 32

```

2. SFT-LoRA fine-tuning

```
#!/bin/bash

unset PADDLE_TRAINERS_NUM
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
export PYTHONPATH=$(dirname "$0")/../../../..:$PYTHONPATH
export FLAGS_set_to_1d=False
export FLAGS_dataloader_use_file_descriptor=False

export MACA_DIRECT_DISPATCH=1
export MCCL_ENABLE_FC=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MACA_SMALL_PAGESIZE_ENABLE=1
export SET_DEVICE_NUMA_PREFERRED=1

export PADDLE_XCCL_BACKEND=metax_gpu
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1

export FLAGS_embedding_deterministic=1

model_path="ERNIE-4.5-21B-A3B-Paddle"
task="sft_lora_8k"
paddle_log_dir="${model_path}_${task}_log"
vdl_log_dir="${model_path}_${task}_vdl"
output_dir="${model_path}_${task}_checkpoint"

rm -rf ${log_dir}
python -m paddle.distributed.launch \
    --log_dir ${paddle_log_dir} \
    --gpus 0,1,2,3,4,5,6,7 \
    examples/post-training/sft/train.py \
    --logging_dir ${vdl_log_dir} \
    --model_name_or_path ${model_path} \
    --output_dir ${output_dir} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --train_dataset_path "./examples/post-training/data/all_ds/ARC-Challenge/train.json,./examples/post-training/data/all_ds/boolq/train.json,./examples/post-training/data/all_ds/piqa/train.json,./examples/post-training/data/all_ds/winogrande/train.json,./examples/post-training/data/all_ds/hellaswag/train.json,./examples/post-training/data/all_ds/ARC-Easy/train.json" \
    --train_dataset_prob "0.2,0.1,0.1,0.2,0.2,0.2" \
    --train_dataset_type "erniekit,erniekit,erniekit,erniekit,erniekit,erniekit" \
    --eval_dataset_path "./examples/post-training/data/all_ds/ARC-Challenge/dev.json,./examples/post-training/data/all_ds/boolq/dev.json,./examples/post-training/data/all_ds/piqa/dev.json,./examples/post-training/data/all_ds/winogrande/dev.json,./examples/post-training/data/all_ds/hellaswag/dev.json,./examples/post-training/data/all_ds/ARC-Easy/dev.json" \
    --eval_dataset_prob "0.2,0.1,0.1,0.2,0.2,0.2" \
    --eval_dataset_type "erniekit,erniekit,erniekit,erniekit,erniekit,erniekit" \
    --max_steps 500 \
    --max_evaluate_steps 10000 \
    --num_train_epochs 1 \
    --save_steps 500 \
    --logging_steps 1 \
    --eval_steps 500 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --tensor_parallel_degree 4 \
    --pipeline_parallel_degree 2 \
    --sharding_parallel_degree 1 \
    --sharding stage1 \
    --max_seq_len 8192 \
    --seed 23 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --learning_rate 1e-5 \
    --min_lr 1e-6 \
    --tensor_parallel_output 0 \
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
    --use_attn_mask_start_row_indices 0 \
    --pipeline_parallel_config "enable_delay_scale_loss enable_release_grads disable_partial_send_recv disable_batch_p2p_comm" \
    --greedy_intokens 0 \
    --release_grads 1 \
    --lr_scheduler_type cosine \
    --sequence_parallel True \
    --moe_group "mp" \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --amp_master_grad 1 \
    --fuse_rope 1 \
    --device "metax_gpu" \
    --use_sparse_flash_attn False \
    --disable_ckpt_quant 1 \
    --recompute_use_reentrant True \
    --offload_optim True \
    --ignore_save_lr_and_optim 1 \
    --ignore_load_lr_and_optim 1 \
    --unified_checkpoint_config ""

```
