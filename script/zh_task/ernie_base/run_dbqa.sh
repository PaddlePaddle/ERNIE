set -eux

export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export PYTHONPATH=./ernie:${PYTHONPATH:-}
python ./ernie/finetune_launch.py  \
    --nproc_per_node 8 \
    --selected_gpus 0,1,2,3,4,5,6,7 \
    --node_ips $(hostname -i) \
    --node_id 0 \
./ernie/run_classifier.py \
                   --use_cuda true \
                   --verbose true \
                   --do_train true \
                   --do_val true \
                   --do_test false \
                   --batch_size 8 \
                   --metric "acc_and_f1_and_mrr" \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --train_set ${TASK_DATA_PATH}/nlpcc-dbqa/train.tsv \
                   --dev_set ${TASK_DATA_PATH}/nlpcc-dbqa/dev.tsv,${TASK_DATA_PATH}/nlpcc-dbqa/test.tsv \
                   --use_multi_gpu_test true \
                   --vocab_path ${MODEL_PATH}/vocab.txt \
                   --ernie_config_path ${MODEL_PATH}/ernie_config.json \
                   --checkpoints "./checkpoints" \
                   --save_steps 1000 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.0 \
                   --validation_steps 1000 \
                   --epoch 3 \
                   --max_seq_len 512 \
                   --learning_rate 2e-5 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 2 \
                   --random_seed 1
