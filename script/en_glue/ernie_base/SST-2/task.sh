#!/bin/bash

R_DIR=`dirname $0`; MYDIR=`cd $R_DIR;pwd`
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_sync_nccl_allreduce=1
export PYTHONPATH=./ernie:${PYTHONPATH:-}

if [[ -f ./model_conf ]];then
    source ./model_conf
else
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

mkdir -p log/

lr=2e-5
batch_size=32
epoch=4

for i in {1..5};do

 python -u ./ernie/run_classifier.py                                                       \
      --for_cn  False                                                              \
      --use_cuda true                                                              \
      --use_fast_executor ${e_executor:-"true"}                                    \
      --tokenizer ${TOKENIZER:-"FullTokenizer"}                                    \
      --use_fp16 ${USE_FP16:-"false"}                                              \
      --do_train true                                                              \
      --do_val true                                                                \
      --do_test true                                                               \
      --batch_size $batch_size                                                     \
      --init_pretraining_params ${MODEL_PATH}/params                               \
      --verbose true                                                               \
      --train_set ${TASK_DATA_PATH}/SST-2/train.tsv                                \
      --dev_set   ${TASK_DATA_PATH}/SST-2/dev.tsv                                  \
      --test_set  ${TASK_DATA_PATH}/SST-2/test.tsv                                 \
      --vocab_path script/en_glue/ernie_base/vocab.txt                             \
      --checkpoints ./checkpoints                                                  \
      --save_steps 10000                                                           \
      --weight_decay  0.0                                                          \
      --warmup_proportion 0.1                                                      \
      --validation_steps 800000000000                                              \
      --epoch $epoch                                                               \
      --max_seq_len 128                                                            \
      --ernie_config_path script/en_glue/ernie_base/ernie_config.json              \
      --learning_rate $lr                                                          \
      --skip_steps 10                                                              \
      --num_iteration_per_drop_scope 1                                             \
      --num_labels 2                                                               \
      --test_save output/test_out.$i.$lr.$batch_size.$epoch.tsv                    \
      --random_seed 1 2>&1 | tee  log/job.$i.$lr.$batch_size.$epoch.$timestamp.log \

done

