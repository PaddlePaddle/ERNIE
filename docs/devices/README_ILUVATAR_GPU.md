# Iluvatar GPU ERNIE-4.5-21B-A3B-Base & ERNIE-4.5-21B-A3B Training Quick Start


##  🚀 Quick Start🚀

### （0）Before starting, you need a Iluvatar GPU machine, and the system requirements for this machine are as follows:

 | Chip type | Driver version |
 | --- | --- |
 | BI150 | 4.3.0 |

#### Environment Description
- **Machine：** BI150 64GB 8-card machine
- **Docker image：** ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
- **GCC path：** /usr/bin/gcc (9.4)
- **python version：** 3.10

**Note: This example uses an 8-card machine: To verify if your machine is Iluvatar GPU, simply enter the command in the system environment and see if there is any output:**
```
ixsmi
#example：$ ixsmi
Timestamp    Thu Jul 10 16:59:37 2025
+-----------------------------------------------------------------------------+
|  IX-ML: 4.3.0       Driver Version: 4.3.0       CUDA Version: 10.2          |
|-------------------------------+----------------------+----------------------|
| GPU  Name                     | Bus-Id               | Clock-SM  Clock-Mem  |
| Fan  Temp  Perf  Pwr:Usage/Cap|      Memory-Usage    | GPU-Util  Compute M. |
|===============================+======================+======================|
| 0    Iluvatar BI-V150         | 00000000:13:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 1    Iluvatar BI-V150         | 00000000:16:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    103W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 2    Iluvatar BI-V150         | 00000000:1C:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 3    Iluvatar BI-V150         | 00000000:1F:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    106W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 4    Iluvatar BI-V150         | 00000000:27:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 5    Iluvatar BI-V150         | 00000000:2A:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    105W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 6    Iluvatar BI-V150         | 00000000:34:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 7    Iluvatar BI-V150         | 00000000:37:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    106W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 8    Iluvatar BI-V150         | 00000000:3D:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 9    Iluvatar BI-V150         | 00000000:40:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    107W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 10   Iluvatar BI-V150         | 00000000:48:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 11   Iluvatar BI-V150         | 00000000:4B:00.0     | 1500MHz   1600MHz    |
| N/A  33C   P0    103W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 12   Iluvatar BI-V150         | 00000000:54:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 13   Iluvatar BI-V150         | 00000000:57:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    104W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 14   Iluvatar BI-V150         | 00000000:64:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 15   Iluvatar BI-V150         | 00000000:67:00.0     | 1500MHz   1600MHz    |
| N/A  36C   P0    107W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU        PID      Process name                                Usage(MiB) |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### (1)  Environment Preparation: (This will take you 5-15 minutes)

1. Pull the Image
```
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
```

2. Install the driver kmd on host
```
wget https://ai-rank.bj.bcebos.com/Iluvatar/corex-driver-linux64-4.3.0.rc.9.20250624_x86_64_10.2.run
bash corex-driver-linux64-4.3.0.rc.9.20250624_x86_64_10.2.run
```

3. Start the Container
```
docker run -itd --name paddle-ixuca-dev -v /usr/src:/usr/src -v /lib/modules:/lib/modules \
    -v /dev:/dev -v /home:/home --privileged --cap-add=ALL --pid=host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
docker exec -it paddle-ixuca-dev bash
```

4. Install paddlepaddle & paddle-iluvatar-gpu
```
# Install PaddlePaddle CPU package
python -m pip install paddlepaddle==3.1.0a0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Install PaddlePaddle iluvatar-gpu plugin package
python -m pip install paddle-iluvatar-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/ixuca/

Nightly version link:
https://www.paddlepaddle.org.cn/packages/nightly/ixuca/
```

4. Install requirements
```
pip install paddleformers
```

### (2) Start post-traning：(This will take a relatively long time)

SFT-LoRA fine-tuning

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

export PADDLE_XCCL_BACKEND=iluvatar_gpu
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1

export FLAGS_embedding_deterministic=1

model_path="ERNIE-4.5-21B-A3B-Paddle"
task="sft_lora_8k"
paddle_log_dir="${model_path}_${task}_log"
vdl_log_dir="${model_path}_${task}_vdl"
output_dir="${model_path}_${task}_checkpoint"

rm -rf ${log_dir}

python3 -m paddle.distributed.launch \
    --log_dir ${paddle_log_dir} \
    --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    examples/post-training/sft/train.py \
    --logging_dir ${vdl_log_dir} \
    --model_name_or_path ${model_path} \
    --output_dir ${output_dir} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --train_dataset_path "examples/data/ARC-Challenge/train.json,examples/data/boolq/train.json,examples/data/piqa/train.json,examples/data/winogrande/train.json,examples/data/hellaswag/train.json,examples/data/ARC-Easy/train.json" \
    --train_dataset_prob "0.2,0.1,0.1,0.2,0.2,0.2" \
    --train_dataset_type "erniekit,erniekit,erniekit,erniekit,erniekit,erniekit" \
    --eval_dataset_path "examples/data/ARC-Challenge/dev.json,examples/data/boolq/dev.json,examples/data/piqa/dev.json,examples/data/winogrande/dev.json,examples/data/hellaswag/dev.json,examples/data/ARC-Easy/dev.json" \
    --eval_dataset_prob "0.2,0.1,0.1,0.2,0.2,0.2" \
    --eval_dataset_type "erniekit,erniekit,erniekit,erniekit,erniekit,erniekit" \
    --max_steps 500 \
    --max_evaluate_steps 10000 \
    --eval_accumulation_steps 100 \
    --num_train_epochs 1 \
    --save_steps 500 \
    --logging_steps 1 \
    --eval_steps 500 \
    --weight_decay 0.01 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --device iluvatar_gpu \
    --tensor_parallel_degree 4 \
    --pipeline_parallel_degree 4 \
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
    --offload_optim 0 \
    --recompute_granularity "full" \
    --dataloader_num_workers 1 \
    --distributed_dataloader 1 \
    --use_flash_attention 1 \
    --use_sparse_head_and_loss_fn 0 \
    --use_attn_mask_start_row_indices 0 \
    --use_sparse_flash_attn 0 \
    --tensor_parallel_output 0 \
    --pipeline_parallel_config "disable_partial_send_recv enable_clear_every_step_cache disable_batch_p2p_comm" \
    --greedy_intokens 1 \
    --lr_scheduler linear \
    --sequence_parallel 1 \
    --release_grads 1 \
    --recompute_use_reentrant True \
    --fuse_rope 1 \
    --moe_multimodal_dispatch_use_allgather "v2-alltoall" \
    --lora \
    --lora_rank 32 \
    --fuse_rms_norm False \
    --moe_group mp
```
