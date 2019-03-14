set -eux

export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0

python -u run_classifier.py \
                   --use_cuda true \
                   --verbose true \
                   --do_train true \
                   --do_val true \
                   --do_test true \
                   --batch_size 32 \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --train_set ${TASK_DATA_PATH}/lcqmc/train.tsv \
                   --dev_set ${TASK_DATA_PATH}/lcqmc/dev.tsv \
                   --test_set ${TASK_DATA_PATH}/lcqmc/test.tsv \
                   --vocab_path config/vocab.txt \
                   --checkpoints ./checkpoints \
                   --save_steps 1000 \
                   --weight_decay  0.0 \
                   --warmup_proportion 0.0 \
                   --validation_steps 100 \
                   --epoch 3 \
                   --max_seq_len 128 \
                   --ernie_config_path config/ernie_config.json \
                   --learning_rate 2e-5 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 2 \
                   --random_seed 1
