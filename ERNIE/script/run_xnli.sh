set -eux

export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -u run_classifier.py \
                   --use_cuda true \
                   --do_train true \
                   --do_val true \
                   --do_test true \
                   --verbose true \
                   --batch_size 8192 \
                   --in_tokens true \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --train_set ${TASK_DATA_PATH}/xnli/train.tsv \
                   --dev_set ${TASK_DATA_PATH}/xnli/dev.tsv \
                   --test_set ${TASK_DATA_PATH}/xnli/test.tsv \
                   --vocab_path config/vocab.txt \
                   --label_map ${TASK_DATA_PATH}/xnli/label_map.json \
                   --ernie_config_path config/ernie_config.json \
                   --checkpoints ./checkpoints \
                   --save_steps 1000 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.0 \
                   --validation_steps 25 \
                   --epoch 3 \
                   --max_seq_len 512 \
                   --learning_rate 1e-4 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 3 \
                   --random_seed 1
