set -eux

source $1

source ./env.sh

export FLAGS_eager_delete_tensor_gb=1.0
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# check
check_iplist

mkdir -p ./log
mkdir -p ./checkpoints
mkdir -p ./tmpdir

export TMPDIR=`pwd`/tmpdir
export TASK_DATA_PATH=${data_path}
export EVAL_SCRIPT_LOG=`pwd`"/log/eval.log"

export DEV_PREFIX=`echo ${dev_set:-"dev.tsv"} | sed 's/\.tsv$//'`
export TEST_PREFIX=`echo ${test_set:-"test.tsv"} | sed 's/\.tsv$//'`
export PRED_PREFIX=`echo ${pred_set:-"pred.tsv"} | sed 's/\.tsv$//'`


distributed_args="--node_ips ${PADDLE_TRAINERS} \
                --node_id ${PADDLE_TRAINER_ID} \
                --current_node_ip ${POD_IP}"
python -u ./utils/finetune_launch.py ${distributed_args} \
    ./run_seq2seq.py --use_cuda true \
                   --do_train $do_train \
                   --do_val $do_val \
                   --do_test $do_test \
                   --do_pred ${do_pred:-"false"} \
                   --train_set ${TASK_DATA_PATH}/${train_set:-""} \
                   --dev_set ${TASK_DATA_PATH}/${dev_set:-""} \
                   --test_set ${TASK_DATA_PATH}/${test_set:-""} \
                   --pred_set ${TASK_DATA_PATH}/${pred_set:-""} \
                   --epoch ${epoch} \
                   --tokenizer ${tokenizer:-"FullTokenizer"} \
                   --tokenized_input ${tokenized_input:-"False"} \
                   --task_type ${task_type:-"normal"} \
                   --role_type_size ${role_type_size:-0} \
                   --turn_type_size ${turn_type_size:-0} \
                   --max_src_len $max_src_len \
                   --max_tgt_len $max_tgt_len \
                   --max_dec_len $max_dec_len \
                   --hidden_dropout_prob ${hidden_dropout_prob:--1} \
                   --attention_probs_dropout_prob ${attention_probs_dropout_prob:--1} \
                   --random_noise ${random_noise:-"False"} \
                   --noise_prob ${noise_prob:-0.0} \
                   --continuous_position ${continuous_position:-"false"} \
                   --tgt_type_id ${tgt_type_id:-1}\
                   --batch_size $batch_size \
                   --learning_rate $learning_rate \
                   --lr_scheduler ${lr_scheduler:-"linear_warmup_decay"} \
                   --warmup_proportion ${warmup_proportion:-0.0} \
                   --weight_decay ${weight_decay:-0.0} \
                   --weight_sharing ${weight_sharing:-"True"} \
                   --label_smooth ${label_smooth:-0.0} \
                   --do_decode ${do_decode:-"True"} \
                   --beam_size ${beam_size:-5}  \
                   --length_penalty ${length_penalty:-"0.0"} \
                   --init_pretraining_params ${init_model:-""} \
                   --vocab_path ${vocab_path} \
                   --ernie_config_path ${config_path} \
                   --checkpoints ./checkpoints \
                   --save_and_valid_by_epoch ${save_and_valid_by_epoch:-"True"} \
                   --eval_script ${eval_script:-""} \
                   --eval_mertrics ${eval_mertrics:-"bleu"} \
                   --random_seed ${random_seed:-"-1"} > log/lanch.log 2>&1 &
