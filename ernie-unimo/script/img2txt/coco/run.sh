#!/usr/bin/env bash
set -eux
R_DIR=`dirname $0`; MYDIR=`cd $R_DIR;pwd`
cd ${MYDIR}/../../../
# config env
source ${MYDIR}/model_conf

source ./env.sh
source ./utils.sh

timestamp=`date "+%Y%m%d-%H%M%S"`
echo $timestamp

# check
check_iplist

set -eu
output_dir=../output-coco
log_dir=../log-coco
mkdir -p $output_dir $log_dir

e_executor=$(echo ${use_experimental_executor-'True'} | tr '[A-Z]' '[a-z]')

use_fuse=$(echo ${use_fuse-'False'} | tr '[A-Z]' '[a-z]')
if [[ ${use_fuse} == "true" ]]; then
    #MB
    export FLAGS_fuse_parameter_memory_size=64
fi

export EVAL_SCRIPT_LOG=${MYDIR}/../../../${output_dir}/eval.log
export TASK_DATA_PATH=${data_path}

distributed_args="--node_ips ${PADDLE_TRAINERS} \
                --node_id ${PADDLE_TRAINER_ID} \
                --current_node_ip ${POD_IP} \
                --selected_gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
                --split_log_path $log_dir \
                --nproc_per_node 16"

skip_steps=10
save_steps=10000
validation_steps=10000

for random_seed in "${DD_RAND_SEED[@]}"; do
    echo "random_seed "${random_seed}
    for batch_size in "${BATCH_SIZE[@]}"; do
        echo "batch_size "${batch_size}
        for warmup_proportion in "${WARMUP_PROP[@]}"; do
            echo "warmup_proportion "${warmup_proportion}
            for learning_rate in "${LR_RATE[@]}"; do
                echo "learning rate "${learning_rate}

                python -u ./src/launch.py ${distributed_args} \
                    ./src/run_img2txt.py --use_cuda "True" \
                       --is_distributed "True" \
                       --use_multi_gpu_test ${use_multi_gpu_test:-"True"} \
                       --use_fp16 ${use_fp16:-"False"} \
                       --use_dynamic_loss_scaling ${use_fp16} \
                       --init_loss_scaling ${loss_scaling:-128} \
                       --use_fast_executor ${e_executor:-"True"} \
                       --use_fuse ${use_fuse:-"False"} \
                       --nccl_comm_num ${nccl_comm_num:-1} \
                       --use_hierarchical_allreduce ${use_hierarchical_allreduce:-"False"} \
                       --do_train ${do_train:-"true"} \
                       --do_val ${do_val:-"false"} \
                       --do_test ${do_test:-"true"} \
                       --do_pred ${do_pred:-"false"} \
                       --do_decode ${do_decode:-"True"} \
                       --train_filelist ${data_path}/${train_filelist:-""} \
                       --valid_filelist ${data_path}/${valid_filelist:-""} \
                       --test_filelist ${data_path}/${test_filelist:-""} \
                       --object_file ${data_path}/${object_file_local_path:-""} \
                       --epoch ${epoch} \
                       --task_type ${task_type:-"img2txt"} \
                       --max_seq_len ${max_seq_len} \
                       --max_img_len ${max_img_len} \
                       --max_obj_len ${max_obj_len} \
                       --max_tgt_len ${max_tgt_len} \
                       --max_out_len ${max_out_len} \
                       --min_out_len ${min_out_len} \
                       --block_trigram ${block_trigram:-"True"} \
                       --beam_size ${beam_size:-5}  \
                       --length_penalty ${length_penalty:-0.6} \
                       --hidden_dropout_prob ${hidden_dropout_prob:-0.1} \
                       --attention_probs_dropout_prob ${attention_probs_dropout_prob:-0.1} \
                       --beta1 ${beta1:-0.9} \
                       --beta2 ${beta2:-0.98} \
                       --epsilon ${epsilon:-1e-06} \
                       --tgt_type_id ${tgt_type_id:-1}\
                       --batch_size ${batch_size} \
                       --pred_batch_size ${pred_batch_size} \
                       --learning_rate ${learning_rate} \
                       --lr_scheduler ${lr_scheduler:-"linear_warmup_decay"} \
                       --warmup_proportion ${warmup_proportion:-0.02} \
                       --weight_decay ${weight_decay:-0.01} \
                       --weight_sharing ${weight_sharing:-"True"} \
                       --label_smooth ${label_smooth:-0.1} \
                       --init_pretraining_params ${init_model:-""} \
                       --unimo_vocab_file ${vocab_file} \
                       --encoder_json_file ${bpe_json} \
                       --vocab_bpe_file ${bpe_file} \
                       --unimo_config_path ${config_path} \
                       --checkpoints $output_dir \
                       --adv_step ${adv_step:-2} \
                       --adv_lr ${adv_lr:-0.05} \
                       --adv_type ${adv_type:-"None"} \
                       --norm_type ${norm_type:-"l2"} \
                       --adv_max_norm ${adv_max_norm:-0.4} \
                       --adv_init_mag ${adv_init_mag:-0.4} \
                       --adv_kl_weight ${adv_kl_weight:-1.5} \
                       --save_steps ${save_steps:-10000} \
                       --validation_steps ${validation_steps:-10000} \
                       --skip_steps ${skip_steps:-10} \
                       --save_and_valid_by_epoch ${save_and_valid_by_epoch:-"False"} \
                       --eval_script ${eval_script:-""} \
                       --eval_mertrics ${eval_mertrics:-""} \
                       --random_seed ${random_seed:-"1"} >> $log_dir/lanch.log 2>&1
            done
        done
    done
done

python ./src/utils/extract_eval_res.py --log_dir=$log_dir
exit 0
