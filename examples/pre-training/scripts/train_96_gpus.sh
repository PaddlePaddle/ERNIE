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

export PADDLE_XCCL_BACKEND=iluvatar_gpu
export PADDLE_DISTRI_BACKEND=xccl
#export CUDA_VISIBLE_DEVICES=2,3,4,5
#export FLAGCX_SOCKET_IFNAME=ens22f0,ens11f0np0
export FLAGCX_DEBUG=TRACE
export FLAGCX_DEBUG_SUBSYS=INIT

export NNODES=1
export PADDLE_TRAINERS_NUM=1

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
# unset PADDLE_ELASTIC_JOB_ID
# unset PADDLE_TRAINER_ENDPOINTS
# unset DISTRIBUTED_TRAINER_ENDPOINTS
# unset FLAGS_START_PORT
# unset PADDLE_ELASTIC_TIMEOUT
# nnodes=$PADDLE_TRAINERS_NUM
# rank=$PADDLE_TRAINER_ID

# LAUNCH_CMD=`python scripts/selective_launch.py 36677`
# if [[ -z "$LAUNCH_CMD" ]]; then
#     exit 0
# fi


SM=`nvidia-smi --query-gpu=compute_cap --format=csv | tail -n 1 | sed 's/\.//g'`
if [ $SM -eq 90 ]
then
    export FLAGS_flash_attn_version=3
else
    export FLAGS_flash_attn_version=2
fi

export PYTHONPATH=$PYTHONPATH:./ernie


LOG_DIR=output/paddle_distributed_logs

rm -rf output
rm -rf core.*

python -m paddle.distributed.launch \
    --log_dir $LOG_DIR \
    --run_mode=collective \
    ${script:-ernie/pretrain.py}  \
    --config yamls/pretrain_96_gpus.yaml
    # $LAUNCH_CMD \
