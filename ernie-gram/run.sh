
#set -ex

source ./utils/utils.sh
source ./task_conf $1 $2
export FLAGS_eager_delete_tensor_gb=2.0
export FLAGS_sync_nccl_allreduce=1


iplist=`hostname -i`
check_iplist

mkdir -p ./tmpout
mkdir -p ./log
mkdir -p ./data

distributed_args="--node_ips ${PADDLE_TRAINERS} \
                --node_id ${PADDLE_TRAINER_ID} \
                --current_node_ip ${POD_IP} \
                --nproc_per_node ${gpu_card} \
                --grid_lr ${lr} \
                --grid_bsz ${batch_size} \
                --grid_epoch ${epoch}"

python -u ./lanch.py ${distributed_args} \
    ./${scripts:-"run_classifier.py"} --use_cuda true \
                   --is_distributed true \
                   --tokenizer ${TOKENIZER:-"FullTokenizer"} \
                   --do_train true \
                   --do_val true \
                   --do_test ${do_test:="false"} \
                   --verbose true \
                   --in_tokens false \
                   --init_pretraining_params ${init_model:-""} \
                   --train_set ${train_set} \
                   --dev_set  ${dev_set} \
                   --test_set  ${test_set} \
                   --run_file_path ${run_file_path:-""} \
                   --vocab_path ${vocab_path} \
                   --ernie_config_path ${CONFIG_PATH} \
                   --checkpoints ./checkpoints \
                   --save_steps 10000000 \
                   --weight_decay ${weight_decay} \
                   --warmup_proportion ${warmup} \
                   --validation_steps 10000000 \
                   --max_seq_len ${max_seq_len:-128} \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels ${num_labels:-2} \
                   --use_multi_gpu_test true \
                   --metric ${metric:-"simple_accuracy"} \
                   --for_race ${for_race:-"false"} \
                   --has_fc ${has_fc:-"true"} \
                   --is_regression ${is_regression:-"false"} \
                   --is_classify ${is_classify:-"true"} \
                   --eval_span ${eval_span:-"false"} \
                   --version_2 ${version_2:-"false"} \
                   --random_seed 1 > log/lanch.log 2>&1









