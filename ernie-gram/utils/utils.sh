#!/usr/bin/env bash

set -xu
function check_iplist() {

    if [ ${iplist:-} ]; then
        #paddle envs
        export PADDLE_PSERVER_PORT=9184
        export PADDLE_TRAINER_IPS=${iplist} 
        #export PADDLE_CURRENT_IP=`/sbin/ip a | grep inet | grep global | awk '{print $2}' | sed 's/\/[0-9][0-9].*$//g'`
        export PADDLE_CURRENT_IP=`hostname -i`
        
        iparray=(${iplist//,/ })
        for i in "${!iparray[@]}"; do
        if [ ${iparray[$i]} == ${PADDLE_CURRENT_IP} ]; then
            export PADDLE_TRAINER_ID=$i
        fi
        done
        
        export TRAINING_ROLE=TRAINER
        #export PADDLE_PSERVERS=127.0.0.1
        export PADDLE_INIT_TRAINER_COUNT=${#iparray[@]}
        export PADDLE_PORT=${PADDLE_PSERVER_PORT}
        export PADDLE_TRAINERS=${PADDLE_TRAINER_IPS}
        export POD_IP=${PADDLE_CURRENT_IP}
        export PADDLE_TRAINERS_NUM=${PADDLE_INIT_TRAINER_COUNT}
        export PADDLE_IS_LOCAL=0
        echo "****************************************************"
  
        #paddle debug envs
        export GLOG_v=0
        export GLOG_logtostderr=1
        
        #nccl debug envs
        export NCCL_DEBUG=INFO
        export NCCL_IB_GID_INDEX=3
    fi
}

