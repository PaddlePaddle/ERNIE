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
   --do_train true \
   --do_val true \
   --do_test false \
   --verbose true \
   --in_tokens true \
    --batch_size 8192 \
   --train_set ${TASK_DATA_PATH}/xnli/train.tsv \
   --dev_set ${TASK_DATA_PATH}/xnli/dev.tsv,${TASK_DATA_PATH}/xnli/test.tsv \
   --label_map ${TASK_DATA_PATH}/xnli/label_map.json \
   --vocab_path ${MODEL_PATH}/vocab.txt \
   --ernie_config_path ${MODEL_PATH}/ernie_config.json \
   --init_pretraining_params ${MODEL_PATH}/params \
   --checkpoints ./checkpoints \
   --save_steps 1000 \
   --weight_decay  0.01 \
   --warmup_proportion 0.0 \
   --use_fp16 false \
   --validation_steps 100 \
   --epoch 3 \
   --max_seq_len 512 \
   --learning_rate 1e-4 \
   --skip_steps 10 \
   --num_iteration_per_drop_scope 1 \
   --num_labels 3 \
   --random_seed 1

