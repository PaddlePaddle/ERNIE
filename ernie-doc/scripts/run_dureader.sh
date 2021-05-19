#!/bin/sh
set -eux

export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_sync_nccl_allreduce=1

# task
finetuning_task="dureader"
task_data_path="./data/finetune/task_data/${finetuning_task}/"

# model setup
is_zh="True"
use_vars="False"
use_amp="False"
train_all="Fasle"
eval_all="False"
use_vars="False"
use_recompute="False"
rel_pos_params_sharing="False"
lr_scheduler="linear_warmup_decay"
vocab_path="./configs/base/zh/vocab.txt"
config_path="./configs/base/zh/ernie_config.json"
init_model_checkpoint=""
init_model_pretraining=""

# args setup
epoch=5
warmup=0.1
max_len=512
lr_rate=2.75e-4
batch_size=16
weight_decay=0.01
save_steps=10000
validation_steps=100
layer_decay_ratio=0.8
init_loss_scaling=32768

PADDLE_TRAINERS=`hostname -i`
PADDLE_TRAINER_ID="0"
POD_IP=`hostname -i`
selected_gpus="0,1,2,3"


if [[ $finetuning_task == "cmrc2018" ]];then
    do_test=false
elif [[ $finetuning_task == "drcd" ]];then
    do_test=true
elif [[ $finetuning_task == "dureader" ]];then
    do_test=false
fi

mkdir -p log
distributed_args="--node_ips ${PADDLE_TRAINERS} --node_id ${PADDLE_TRAINER_ID} --current_node_ip ${POD_IP} --nproc_per_node 4 --selected_gpus ${selected_gpus}"
python -u ./lanch.py ${distributed_args} \
    run_mrc.py --use_cuda true\
            --is_distributed true \
            --batch_size ${batch_size} \
            --in_tokens false\
            --use_fast_executor ${e_executor:-"true"} \
            --checkpoints ./output \
            --vocab_path ${vocab_path} \
            --do_train true \
            --do_val true \
            --do_test ${do_test} \
            --save_steps ${save_steps:-"10000"} \
            --validation_steps ${validation_steps:-"100"} \
            --warmup_proportion ${warmup} \
            --weight_decay ${weight_decay} \
            --epoch ${epoch} \
            --max_seq_len ${max_len} \
            --ernie_config_path ${config_path} \
            --do_lower_case true \
            --doc_stride 128 \
            --train_set ${task_data_path}/train.json \
            --dev_set ${task_data_path}/dev.json \
            --test_set ${task_data_path}/test.json \
            --learning_rate ${lr_rate} \
            --num_iteration_per_drop_scope 1 \
            --lr_scheduler linear_warmup_decay \
            --layer_decay_ratio ${layer_decay_ratio:-"0.8"} \
            --is_zh ${is_zh:-"True"} \
            --repeat_input ${repeat_input:-"False"} \
            --train_all ${train_all:-"False"} \
            --eval_all ${eval_all:-"False"} \
            --use_vars ${use_vars:-"False"} \
            --init_checkpoint ${init_model_checkpoint:-""} \
            --init_pretraining_params ${init_model_pretraining:-""} \
            --init_loss_scaling ${init_loss_scaling:-32768} \
            --use_recompute ${use_recompute:-"False"} \
            --skip_steps 10 1>log/0_${finetuning_task}_job.log 2>&1

