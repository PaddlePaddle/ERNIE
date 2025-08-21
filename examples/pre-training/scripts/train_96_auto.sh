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

export NNODES=1
export PADDLE_TRAINERS_NUM=1

mpi_rank=${OMPI_COMM_WORLD_RANK:-0}
node_rank=$((mpi_rank+offset))
mpi_node=${OMPI_COMM_WORLD_SIZE:-1}
echo "MPI status:${mpi_rank}/${mpi_node}"
nnode_train=${nnode_set:-${mpi_node}}
master_train=${master:-localhost}
#
echo "Distributed Training ${node_rank}/${nnode_train} master=${master_train}"
set -x

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


# 屏蔽平台预设的环境变量，因为框架采用兼容升级，检测到这些配置会使用原方式启动
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID


export FLAGS_shard_use_reduce=1
export FLAGS_shard_norm_align_dp=0

#加速pin memory save ckpt时间
export FLAGS_use_auto_growth_pinned_allocator=True

# export FLAGS_flash_attn_version=v1
# 开启FA3
SM=`nvidia-smi --query-gpu=compute_cap --format=csv | tail -n 1 | sed 's/\.//g'`
if [ $SM -eq 90 ]
then
    export FLAGS_flash_attn_version=3
else
    export FLAGS_flash_attn_version=2
fi

# 保证集群稳定性的配置，跟性能无关
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=3
# 开启AR功能
export NCCL_IB_ADAPTIVE_ROUTING=1

# 集群hang检测
export PADDLE_PG_TIMEOUT=150000   # 通信组超时时间，单位是ms，默认2分钟
export FLAGS_enable_async_trace=False # True开启通信debug功能，False或不设置关闭，默认开启
# export CUDA_MODULE_LOADING=LAZY

export FLAGS_pipeline_nccl_comm_init_option=1

# 启动方式
cuda_version=`nvidia-smi |grep "CUDA Version" |awk '{print $9}' |awk -F'.' '{print $1}'`
if [ ${cuda_version} != "12" ];then
    export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
fi

master=`cat /root/paddlejob/workspace/hostfile | head -n 1 | awk '{print $1}'`
port=36677


#自动并行相关
export FLAGS_enable_fused_ffn_qkv_pass=1
export FLAGS_enable_pir_api=1
#export FLAGS_enable_sharding_stage1_tensor_fusion=1
export FLAGS_enable_moe_utils=true

#调试相关
export FLAGS_call_stack_level=2
#export GLOG_v=6
#export FLAGS_print_ir=1
#export FLAGS_benchmark=1
#export CUDA_VISIBLE_DEVICES=0,1

export PYTHONPATH=$PYTHONPATH:./ernie

LOG_DIR=output/paddle_distributed_logs

rm -rf output
rm -rf core.*

python -m paddle.distributed.launch \
    --log_dir $LOG_DIR \
    --run_mode=collective \
    ${script:-ernie/pretrain_auto.py}  \
    --config yamls/pretrain_96_auto.yaml
