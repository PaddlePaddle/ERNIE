set -eux

export BASE_PATH="$PWD"
export PATH="${BASE_PATH}/py37/bin/:$PATH"
export PYTHONPATH="${BASE_PATH}/py37/"

export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_sync_nccl_allreduce=1

export TRAINER_PORTS_NUM='8'
export PADDLE_CURRENT_ENDPOINT=`hostname -i`
export PADDLE_TRAINERS_NUM='1'
export POD_IP=`hostname -i`
export PADDLE_TRAINER_COUNT='8'

# task
task_data_path="./data/imdb/"
task_name="imdb"

# model setup
is_zh="False"
use_vars="False"
use_amp="False"
use_recompute="False"
rel_pos_params_sharing="False"
lr_scheduler="linear_warmup_decay"
vocab_path="./configs/base/en/vocab.txt"
config_path="./configs/base/en/ernie_config.json"
init_model_checkpoint=""
init_model_pretraining=""

# args setup
max_len=512
lr_rate=7e-5
batch_size=16
weight_decay=0.01
warmup=0.1
epoch=3
num_labels=2
save_steps=100000
validation_steps=700
layer_decay_ratio=1
init_loss_scaling=32768

PADDLE_TRAINERS=`hostname -i`
PADDLE_TRAINER_ID="0"
POD_IP=`hostname -i`
selected_gpus="0,1"

distributed_args="--node_ips ${PADDLE_TRAINERS} --node_id ${PADDLE_TRAINER_ID} --current_node_ip ${POD_IP} --nproc_per_node 2 --selected_gpus ${selected_gpus}"
python -u ./lanch.py ${distributed_args} \
    ./run_classifier.py --use_cuda true \
                   --is_distributed true \
                   --use_fast_executor ${e_executor:-"true"} \
                   --tokenizer ${TOKENIZER:-"BPETokenizer"} \
                   --use_amp ${use_amp:-"false"} \
                   --do_train true \
                   --do_val true \
                   --do_test false \
                   --batch_size ${batch_size} \
                   --init_checkpoint ${init_model_checkpoint:-""} \
                   --init_pretraining_params ${init_model_pretraining:-""} \
                   --train_set ${task_data_path}/train.txt \
                   --dev_set ${task_data_path}/test.txt \
                   --test_set ${task_data_path}/test.txt \
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
                   --use_vars ${use_vars:-"False"} \
                   --init_loss_scaling ${init_loss_scaling:-32768} \
                   --use_recompute ${use_recompute:-"False"} \
                   --random_seed 1 
