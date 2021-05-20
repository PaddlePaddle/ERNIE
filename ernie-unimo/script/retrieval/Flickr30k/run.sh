#!/usr/bin/env bash
set -eux
R_DIR=`dirname $0`; MYDIR=`cd $R_DIR;pwd`
cd ${MYDIR}/../../../
# config env
source ${MYDIR}/model_conf

source ./env.sh
source ./utils.sh

check_iplist
export FLAGS_fuse_parameter_memory_size=64

set -eu
output_dir=./output/${task}
log_dir=${output_dir}/log
save_model_base_dir=$output_dir/save_model
mkdir -p $output_dir $log_dir $save_model_base_dir

log_prefix=$seed"_"$epoch"_"$lr"_"$batch_size"."
eval_dir="${output_dir}/tmp/params.${seed}.${epoch}.${lr}.${batch_size}"
mkdir -p $eval_dir

if [[ ${save_checkpoints} == "True" ]]; then
  save_model_dir="${save_model_base_dir}/params.${seed}.${epoch}.${lr}.${batch_size}"
  mkdir -p $save_model_dir
fi

distributed_args="--node_ips ${PADDLE_TRAINERS} \
                    --node_id ${PADDLE_TRAINER_ID} \
                    --current_node_ip ${POD_IP} \
                    --selected_gpus 0,1,2,3,4,5,6,7 \
                    --split_log_path $log_dir \
                    --log_prefix $log_prefix \
                    --nproc_per_node 8"                    
lanch_start=" -u ./src/launch.py ${distributed_args} "
python $lanch_start ./src/run_retrieval.py \
        --use_cuda "True" \
        --is_distributed ${is_distributed:-"True"} \
        --weight_sharing ${weight_sharing:-"True"} \
        --use_fuse ${use_fuse:-"True"} \
        --use_fast_executor ${e_executor:-"true"} \
        --use_fp16 ${use_fp16:-"false"} \
        --nccl_comm_num ${nccl_comm_num:-2} \
        --use_hierarchical_allreduce ${use_hierarchical_allreduce:-"False"} \
        --use_dynamic_loss_scaling ${use_fp16:-"False"} \
        --use_sigmoid ${use_sigmoid:-"False"} \
        --init_loss_scaling ${loss_scaling:-12800} \
        --beta1 ${beta1:-0.9} \
        --beta2 ${beta2:-0.98} \
        --epsilon ${epsilon:-1e-06} \
        --scale_circle ${scale_circle:-1.0} \
        --margin ${margin:-0.2} \
        --verbose true \
        --samples_num ${samples_num:-20} \
        --run_random ${run_random:-"False"} \
        --do_train ${do_train:-"True"} \
        --do_val ${do_val:-"True"} \
        --do_test ${do_test:-"True"} \
        --batch_size ${batch_size:-16} \
        --test_batch_size ${test_batch_size:-96} \
        --init_pretraining_params ${init_model:-""} \
        --train_image_caption ./data/Flickr30k/flickr30k-textids/train.ids \
        --train_image_feature_dir ./data/Flickr30k/flickr30k-features/$bbox/train \
        --dev_image_caption ./data/Flickr30k/flickr30k-textids/val.all.ids \
        --dev_image_feature_dir ./data/Flickr30k/flickr30k-features/$bbox/dev \
        --test_image_caption ./data/Flickr30k/flickr30k-textids/test.all.ids \
        --test_image_feature_dir ./data/Flickr30k/flickr30k-features/$bbox/test \
        --img_id_path ./data/Flickr30k/flickr30k-textids/dataset_flickr30k_name_id.txt \
        --checkpoints ${save_model_dir:-""} \
        --save_checkpoints ${save_checkpoints:-"True"} \
        --save_steps ${save_steps:-1000} \
        --weight_decay ${weight_decay:-"0.1"} \
        --warmup_step ${warmup_step:-"1"} \
        --validation_steps ${validation_steps:-"100"} \
        --epoch $epoch \
        --max_seq_len ${max_len:-512} \
        --max_img_len ${max_img_len:-37} \
        --learning_rate ${lr:-"5e-6"} \
        --learning_rate_scale ${learning_rate_scale:-0.1} \
        --learning_rate_decay_epoch1 ${learning_rate_decay_epoch1:-24} \
        --learning_rate_decay_epoch2 ${learning_rate_decay_epoch2:-32} \
        --lr_scheduler ${lr_scheduler:-"scale_by_epoch_decay"} \
        --skip_steps ${skip_steps:-"50"} \
        --num_iteration_per_drop_scope 10 \
        --unimo_vocab_file ${vocab_file} \
        --encoder_json_file ${bpe_json} \
        --vocab_bpe_file ${bpe_file} \
        --unimo_config_path ${config_path} \
        --eval_mertrics ${eval_mertrics:-"recall@k"} \
        --eval_dir $eval_dir \
        --random_seed ${seed:-1} \
        >> $log_dir/${log_prefix}lanch.log 2>&1

if [[ $? -ne 0 ]]; then
    echo "run failed"
    exit 1
fi
exit 0
