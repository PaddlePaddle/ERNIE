#!/usr/bin/env bash
set -eux
R_DIR=`dirname $0`; MYDIR=`cd $R_DIR;pwd`
cd ${MYDIR}/../../../
source ${MYDIR}/model_conf

source ./env.sh
source ./utils.sh

check_iplist
export FLAGS_fuse_parameter_memory_size=64

output_dir=./output/${task}
log_dir=${output_dir}/log
eval_dir=${output_dir}/tmp
save_model_base_dir=$output_dir/save_model
mkdir -p $output_dir $log_dir $eval_dir $save_model_base_dir

for seed in "${DD_RAND_SEED[@]}"; do
  echo "seed "$seed
  for epoch in "${EPOCH[@]}"; do
    echo "epoch "$epoch
    for lr in "${LR_RATE[@]}"; do
      echo "learning rate "$lr
      for bs in "${BATCH_SIZE[@]}"; do
        echo "batch_size "$bs

        log_prefix=$seed"_"$epoch"_"$lr"_"$bs"."
        eval_dir="${output_dir}/tmp/params.${seed}.${epoch}.${lr}.${bs}"
        mkdir -p $eval_dir

        if [[ ${bs} == "32" ]]; then
            validation_steps=2000
        fi

        if [[ ${save_checkpoints} == "True" ]]; then
          save_model_dir="${save_model_base_dir}/params.${seed}.${epoch}.${lr}.${bs}"
          mkdir -p $save_model_dir
        fi

        distributed_args="--node_ips ${PADDLE_TRAINERS} \
                --node_id ${PADDLE_TRAINER_ID} \
                --current_node_ip ${POD_IP} \
                --selected_gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
                --split_log_path $log_dir \
                --log_prefix $log_prefix \
                --nproc_per_node 16"

        python -u ./src/launch.py ${distributed_args} \
            ./src/run_visual_entailment.py --use_cuda "True" \
            --is_distributed ${is_distributed:-"True"} \
            --weight_sharing ${weight_sharing:-"True"} \
            --use_fuse ${use_fuse:-"True"} \
            --use_fast_executor ${e_executor:-"true"} \
            --use_fp16 ${use_fp16:-"false"} \
            --nccl_comm_num ${nccl_comm_num:-1} \
            --use_hierarchical_allreduce ${use_hierarchical_allreduce:-"True"} \
            --in_tokens ${in_tokens:-"false"} \
            --use_dynamic_loss_scaling ${use_fp16:-"false"} \
            --init_loss_scaling ${loss_scaling:-12800} \
            --beta1 ${beta1:-0.9} \
            --beta2 ${beta2:-0.999} \
            --epsilon ${epsilon:-1e-06} \
            --verbose true \
            --do_train ${do_train:-"True"} \
            --do_val ${do_val:-"True"} \
            --do_test ${do_test:-"True"} \
            --do_test_hard ${do_test_hard:-"True"} \
            --num_train_examples ${num_train_examples:-529527} \
            --adv_step ${adv_step:-4} \
            --adv_lr ${adv_lr:-0.05} \
            --norm_type ${norm_type:-"l2"} \
            --adv_max_norm ${adv_max_norm:-0.4} \
            --adv_init_mag ${adv_init_mag:-0.4} \
            --batch_size ${bs:-16} \
            --test_batch_size ${test_batch_size:-16} \
            --init_pretraining_params ${init_model:-""} \
            --train_filelist "./data/SNLI-VE/$bbox/train_filelist" \
            --dev_filelist "./data/SNLI-VE/$bbox/dev_filelist" \
            --test_filelist "./data/SNLI-VE/$bbox/test_filelist" \
            --test_hard_filelist ${test_hard_filelist:-""} \
            --checkpoints ${save_model_dir:-""} \
            --save_checkpoints ${save_checkpoints:-"True"} \
            --save_steps ${save_steps:-1000} \
            --weight_decay ${weight_decay:-"0.1"} \
            --warmup_proportion ${warmup_ratio:-"0.06"} \
            --validation_steps ${validation_steps:-"100"} \
            --epoch $epoch \
            --max_seq_len ${max_len:-512} \
            --max_img_len ${max_img_len:-101} \
            --learning_rate ${lr:-"5e-5"} \
            --lr_scheduler ${lr_scheduler:-"linear_warmup_decay"} \
            --skip_steps ${skip_steps:-"100"} \
            --num_iteration_per_drop_scope 10 \
            --num_labels ${num_labels:-3} \
            --unimo_vocab_file ${vocab_file} \
            --encoder_json_file ${bpe_json} \
            --vocab_bpe_file ${bpe_file} \
            --unimo_config_path ${config_path} \
            --eval_mertrics ${eval_mertrics:-"simple_accuracy"} \
            --eval_dir ${eval_dir:-"./output/tmp"} \
            --random_seed ${seed:-1} >> $log_dir/${log_prefix}lanch.log 2>&1
      done
    done
  done
done
if [[ $? -ne 0 ]]; then
    echo "run failed"
    exit 1
fi

python ./src/utils/stat_res.py --log_dir=$log_dir --key_words=job.log.0

exit 0
