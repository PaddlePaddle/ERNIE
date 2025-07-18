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

mpi_rank=${OMPI_COMM_WORLD_RANK:-0}
node_rank=$((mpi_rank+offset))
mpi_node=${OMPI_COMM_WORLD_SIZE:-1}
echo "MPI status:${mpi_rank}/${mpi_node}"
nnode_train=${nnode_set:-${mpi_node}}
master_train=${master:-localhost}
#
echo "Distributed Training ${node_rank}/${nnode_train} master=${master_train}"
set -x

# Block the platform's preset environment variables, 
#since the framework adopts a compatible upgrade approach 
#and will detect these configurations to start in the original manner.
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID




export NCCL_DEBUG=INFO
unset GLOG_vmodule GLOG_v
export PYTHONUNBUFFERED=1

#Optimize pinned memory usage and reduce checkpoint saving time.
export FLAGS_use_auto_growth_pinned_allocator=True

# Settings to guarantee cluster stability, independent of performance tuning.
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=3
export NCCL_NVLS_ENABLE=0
# AR
export NCCL_IB_ADAPTIVE_ROUTING=1


export PADDLE_PG_TIMEOUT=150000  
# True to enable communication debugging, False or unset to disable. Default is True.
export FLAGS_enable_async_trace=False 
enable_nccl_proxy_dump=True # enable hang trace through BCCLe
export CUDA_MODULE_LOADING=LAZY
if [[ $enable_nccl_proxy_dump == "True" ]];then
    export NCCL_PROXY_DUMP_SIGNAL=10
fi

export FLAGS_pipeline_nccl_comm_init_option=1
export FLAGS_sharding_v2_check_zero_padding=1

export FLAGS_use_paddle_recall_error=0

# Turn off the CUDNN FA (Fast Algorithm) feature for H-series GPUs.
export PADDLE_DISABLE_CUDNN_FA=1

find /dev/shm/ -type f -name "paddle_*" -print0 | xargs -0 rm -f

cuda_version=`nvidia-smi |grep "CUDA Version" |awk '{print $9}' |awk -F'.' '{print $1}'`
if [ ${cuda_version} != "12" ];then
    export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
fi


export FLAGS_call_stack_level=2


export FLAGS_eager_communication_connection=0


python -m paddle.distributed.launch \
    --log_dir output/paddle_distributed_logs \
    --nnodes 1 \
    --rank 0 \
    --run_mode=collective \
    ${script:-examples/post-training/vl_sft/train.py}  \
    "$@"
