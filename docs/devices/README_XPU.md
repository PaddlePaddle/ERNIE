# Kunlunxin XPU ERNIE-4.5-300B-A47B-Base & ERNIE-4.5-300B-A47B Training Quick Start


##  🚀 Quick Start🚀

### （0）Before starting, you need a Kunlun XPU machine, and the system requirements for this machine are as follows:

 | Chip type | Driver version |
 | --- | --- |
 | KunlunxinP800 | 5.0.21.21 |

#### Environment Description
- **Machine：** KunlunxinP800 96GB 8-card machine x 14
- **Docker image：** registry.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310
- **GCC path：** /usr/bin/gcc (8.4)
- **python version：** 3.10
**Note: This example uses an 8-card machine: To verify if your machine is a Kunlunxin, simply enter the command in the system environment and see if there is any output:**
```
xpu_smi
#example：$ xpu_smi
Wed Jun 25 19:45:10 2025
+-----------------------------------------------------------------------------+
| XPU-SMI               Driver Version: 5.0.21.21    XPU-RT Version: 5.0.21   |
|-------------------------------+----------------------+----------------------+
| XPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | XPU-Util  Compute M. |
|                               |             L3-Usage |            SR-IOV M. |
|===============================+======================+======================|
|   0  P800 OAM           N/A   | 00000000:03:00.0 N/A |                    0 |
| N/A   37C  N/A     88W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  P800 OAM           N/A   | 00000000:05:00.0 N/A |                    0 |
| N/A   41C  N/A     90W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  P800 OAM           N/A   | 00000000:63:00.0 N/A |                    0 |
| N/A   36C  N/A     89W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  P800 OAM           N/A   | 00000000:65:00.0 N/A |                    0 |
| N/A   36C  N/A     89W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   4  P800 OAM           N/A   | 00000000:83:00.0 N/A |                    0 |
| N/A   40C  N/A     88W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   5  P800 OAM           N/A   | 00000000:85:00.0 N/A |                    0 |
| N/A   40C  N/A     90W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   6  P800 OAM           N/A   | 00000000:A3:00.0 N/A |                    0 |
| N/A   39C  N/A     90W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   7  P800 OAM           N/A   | 00000000:A5:00.0 N/A |                    0 |
| N/A   40C  N/A     87W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  XPU   XI   CI        PID   Type   Process name                  XPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### (1)  Environment Preparation: (This will take you 5-15 minutes)

1. Pull the Image
```

docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310
```

2. Start the Container
```
docker run -it --privileged=true  --net host --device=/dev/xpu0:/dev/xpu0 --device=/dev/xpu1:/dev/xpu1 --device=/dev/xpu2:/dev/xpu2 --device=/dev/xpu3:/dev/xpu3 --device=/dev/xpu4:/dev/xpu4 --device=/dev/xpu5:/dev/xpu5 --device=/dev/xpu6:/dev/xpu6 --device=/dev/xpu7:/dev/xpu7 --device=/dev/xpuctrl:/dev/xpuctrl --name paddle-xpu-dev -v $(pwd):/work -w=/work -v xxx ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310 /bin/bash
```

3. Install paddlepaddle-xpu
```
# The "PaddlePaddle" deep learning framework provides basic computing capabilities
python -m pip install paddlepaddle-xpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/
# Paddle_xpu contains a small number of XPU custom operators, mainly used to support XPU training acceleration
wget https://klx-sdk-release-public.su.bcebos.com/v1/xpaddle/release/3.1/paddle_xpu-0.0.1-py3-none-any.whl
python -m pip install paddle_xpu-0.0.1-py3-none-any.whl

Nightly version link:
https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/paddlepaddle-xpu/
```

4. Install requirements
```
pip install -r requirements/gpu/requirements.txt
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
export PYTHONPATH=$(dirname "$0")
export FLAGS_set_to_1d=False
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_dataloader_use_file_descriptor=False

export FLAGS_set_to_1d=False
export FLAGS_use_stride_kernel=0

export XPU_PADDLE_L3_SIZE=0
export XPUAPI_DEFAULT_SIZE=2205258752

# The driver can support up to 8 streams
export CUDA_DEVICE_MAX_CONNECTIONS=8

# BKCL
export BKCL_TREE_THRESHOLD=0
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_FORCE_TREE=1
export BKCL_TREE_THRESHOLD=0
export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4
export BKCL_SOCKET_IFNAME=eth0
export BKCL_FORCE_L3_RDMA=0
export BKCL_USE_AR=1
export BKCL_RING_OPT=1
export BKCL_RING_HOSTID_USE_RANK=1

export XPU_PADDLE_FC_LOCAL_INT16=1
export XPU_AUTO_BF16_TF32_RADIO=10
export XPU_AUTO_BF16_TF32=1

master_ip=${1:-}
nnodes=${2:-14}
model_path="/work/baidu/paddle_internal/ERNIE-4.5-300B-A47B"
task="sft_8k"
paddle_log_dir="log_sft"
vdl_log_dir="${model_path}_${task}_vdl"
output_dir="${model_path}_${task}_checkpoint"

rm -rf ${paddle_log_dir}

python -m paddle.distributed.launch \
    --log_dir ${paddle_log_dir} \
    --xpus 0,1,2,3,4,5,6,7 \
    --master ${master_ip}:8090 \
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
    --max_steps 500 \
    --max_evaluate_steps 10000 \
    --num_train_epochs 1 \
    --save_steps 10000000 \
    --logging_steps 1 \
    --eval_steps 50 \
    --weight_decay 0.01 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --tensor_parallel_degree 8 \
    --pipeline_parallel_degree ${nnodes} \
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
    --use_sparse_flash_attn 1 \
    --pipeline_parallel_config "enable_delay_scale_loss enable_release_grads disable_partial_send_recv disable_batch_p2p_comm" \
    --greedy_intokens 1 \
    --release_grads 1 \
    --lr_scheduler_type cosine \
    --sequence_parallel 1 \
    --moe_group "mp" \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --amp_master_grad 1 \
    --fuse_rope 1 \
    --disable_ckpt_quant 1 \
    --offload_optim \
    --recompute_use_reentrant True \
    --unified_checkpoint_config "async_save" \
    --continue_training 1 \
    --device "xpu" \
    --moe_multimodal_dispatch_use_allgather "v2-alltoall-unpad"
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
export PYTHONPATH=$(dirname "$0")
export FLAGS_set_to_1d=False
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_dataloader_use_file_descriptor=False

export FLAGS_set_to_1d=False
export FLAGS_use_stride_kernel=0

export XPU_PADDLE_L3_SIZE=0
export XPUAPI_DEFAULT_SIZE=2205258752

# The driver can support up to 8 streams
export CUDA_DEVICE_MAX_CONNECTIONS=8

# BKCL
export BKCL_TREE_THRESHOLD=0
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_FORCE_TREE=1
export BKCL_TREE_THRESHOLD=0
export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4
export BKCL_SOCKET_IFNAME=eth0
export BKCL_FORCE_L3_RDMA=0
export BKCL_USE_AR=1
export BKCL_RING_OPT=1
export BKCL_RING_HOSTID_USE_RANK=1

export XPU_PADDLE_FC_LOCAL_INT16=1
export XPU_AUTO_BF16_TF32_RADIO=10
export XPU_AUTO_BF16_TF32=1

master_ip=${1:-}
nnodes=${2:-2}
model_path="/work/baidu/paddle_internal/ERNIE-4.5-300B-A47B"
task="lora_8k"
paddle_log_dir="log_lora"
vdl_log_dir="${model_path}_${task}_vdl"
output_dir="${model_path}_${task}_checkpoint"

rm -rf ${paddle_log_dir}

python -m paddle.distributed.launch \
    --log_dir ${paddle_log_dir} \
    --xpus 0,1,2,3,4,5,6,7 \
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
    --max_steps 500 \
    --max_evaluate_steps 10000 \
    --num_train_epochs 1 \
    --save_steps 500000 \
    --logging_steps 1 \
    --eval_steps 50 \
    --weight_decay 0.01 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --tensor_parallel_degree 8 \
    --pipeline_parallel_degree ${nnodes} \
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
    --pipeline_parallel_config "disable_batch_p2p_comm disable_partial_send_recv enable_clear_every_step_cache" \
    --greedy_intokens 1 \
    --lr_scheduler linear \
    --sequence_parallel 1 \
    --release_grads 1 \
    --recompute_use_reentrant True \
    --fuse_rope 1 \
    --lora \
    --lora_rank 32 \
    --moe_group mp \
    --device "xpu" \
    --continue_training 1 \
    --moe_multimodal_dispatch_use_allgather "v2-alltoall-unpad"
```
