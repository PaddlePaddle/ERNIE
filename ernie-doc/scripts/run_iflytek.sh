set -eux

export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_sync_nccl_allreduce=1

# task
task_data_path="./data/finetune/task_data/"
task_name="iflytek"

# model setup
is_zh="True"
repeat_input="False"
train_all="Fasle"
eval_all="False"
use_vars="False"
use_amp="False"
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
lr_rate=1.5e-4
batch_size=16
weight_decay=0.01
num_labels=119
save_steps=10000
validation_steps=100
layer_decay_ratio=0.8
init_loss_scaling=32768


PADDLE_TRAINERS=`hostname -i`
PADDLE_TRAINER_ID="0"
POD_IP=`hostname -i`
selected_gpus="0"

mkdir -p log
distributed_args="--node_ips ${PADDLE_TRAINERS} --node_id ${PADDLE_TRAINER_ID} --current_node_ip ${POD_IP} --nproc_per_node 1 --selected_gpus ${selected_gpus}"
python -u ./lanch.py ${distributed_args} \
    ./run_classifier.py --use_cuda true \
                   --is_distributed true \
                   --use_fast_executor ${e_executor:-"true"} \
                   --tokenizer ${TOKENIZER:-"FullTokenizer"} \
                   --use_amp ${use_amp:-"false"} \
                   --do_train true \
                   --do_val true \
                   --do_test false \
                   --batch_size ${batch_size} \
                   --init_checkpoint ${init_model_checkpoint:-""} \
                   --init_pretraining_params ${init_model_pretraining:-""} \
                   --label_map_config "" \
                   --train_set ${task_data_path}/${task_name}/train/1 \
                   --dev_set ${task_data_path}/${task_name}/dev/1 \
                   --test_set ${task_data_path}/${task_name}/test/1 \
                   --vocab_path ${vocab_path} \
                   --checkpoints ./output \
                   --save_steps ${save_steps} \
                   --weight_decay ${weight_decay} \
                   --warmup_proportion ${warmup} \
                   --validation_steps ${validation_steps} \
                   --epoch ${epoch} \
                   --max_seq_len ${max_len} \
                   --ernie_config_path ${config_path} \
                   --learning_rate ${lr_rate} \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 10 \
                   --layer_decay_ratio ${layer_decay_ratio:-"0.8"} \
                   --num_labels ${num_labels} \
                   --is_zh ${is_zh:-"True"} \
                   --repeat_input ${repeat_input:-"False"} \
                   --train_all ${train_all:-"False"} \
                   --eval_all ${eval_all:-"False"} \
                   --use_vars ${use_vars:-"False"} \
                   --init_loss_scaling ${init_loss_scaling:-32768} \
                   --use_recompute ${use_recompute:-"False"} \
                   --random_seed 1 
