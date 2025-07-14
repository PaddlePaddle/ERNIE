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

# 屏蔽平台预设的环境变量，因为框架采用兼容升级，检测到这些配置会使用原方式启动
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID




export NCCL_DEBUG=INFO
unset GLOG_vmodule GLOG_v

###
export PYTHONUNBUFFERED=1

#加速pin memory save ckpt时间
export FLAGS_use_auto_growth_pinned_allocator=True

# 保证集群稳定性的配置，跟性能无关
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=3
export NCCL_NVLS_ENABLE=0
# 开启AR功能
export NCCL_IB_ADAPTIVE_ROUTING=1

#开启BCCL流量统计
export BCCL_BUS_BW_CALCULATE_MODE=Agg

# 集群hang检测
export PADDLE_PG_TIMEOUT=150000   # 通信组超时时间，单位是ms，默认2分钟
export FLAGS_enable_async_trace=False # True开启通信debug功能，False或不设置关闭，默认开启
enable_nccl_proxy_dump=True # enable hang trace through BCCLe
export CUDA_MODULE_LOADING=LAZY
if [[ $enable_nccl_proxy_dump == "True" ]];then
    export NCCL_PROXY_DUMP_SIGNAL=10
fi

export FLAGS_pipeline_nccl_comm_init_option=1


# 开启Sharding V2 Padding Zero检查
export FLAGS_sharding_v2_check_zero_padding=1

export FLAGS_use_paddle_recall_error=0

# 关闭 H 卡 CUDNN FA 功能
export PADDLE_DISABLE_CUDNN_FA=1

# 使用BCCL，需要配合镜像版本 >= FleetY10.1.0
export LD_LIBRARY_PATH=/usr/local/bccl/lib:$LD_LIBRARY_PATH

# bce-bns-proxy环境变量，启动bos的流量统计和控制功能
export REDIS_HOST=10.11.74.156
export REDIS_PORT=9001
export REDIS_PASSWORD=redis@NLP_2024
export BOS_BNS_TRAFFIC_CONTROL_ENABLE=true

if [ -f /etc/logrotate.d/bccl ]; then
    service cron start
    service cron status > /dev/null

    if [ $? -eq 0 ]; then
        echo "环境变量NCCL_DEBUG_SUBSYS已设置"
    else
        echo "cron服务未运行"
    fi
else
    echo "File /etc/logrotate.d/bccl does not exist. Skipping logrotate."
fi

# 释放shmem
find /dev/shm/ -type f -name "paddle_*" -print0 | xargs -0 rm -f

# 启动方式
cuda_version=`nvidia-smi |grep "CUDA Version" |awk '{print $9}' |awk -F'.' '{print $1}'`
if [ ${cuda_version} != "12" ];then
    export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
fi

master=`cat /root/paddlejob/workspace/hostfile | head -n 1 | awk '{print $1}'`
port=36677

export FLAGS_call_stack_level=2


export FLAGS_eager_communication_connection=0



source ~/anaconda3/etc/profile.d/conda.sh
conda activate erniekit

python -m paddle.distributed.launch \
    --log_dir output/paddle_distributed_logs \
    --nnodes 1 \
    --rank 0 \
    --run_mode=collective \
    ${script:-examples/post-training/vl_sft/train.py}  \
    "$@"
