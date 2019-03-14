set -eux

export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0

python -u run_sequence_labeling.py \
                   --use_cuda true \
                   --do_train true \
                   --do_val true \
                   --do_test true \
                   --batch_size 16 \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --num_labels 7 \
                   --label_map_config ${TASK_DATA_PATH}/msra_ner/label_map.json \
                   --train_set ${TASK_DATA_PATH}/msra_ner/train.tsv \
                   --dev_set ${TASK_DATA_PATH}/msra_ner/dev.tsv \
                   --test_set ${TASK_DATA_PATH}/msra_ner/test.tsv \
                   --vocab_path config/vocab.txt \
                   --ernie_config_path config/ernie_config.json \
                   --checkpoints ./checkpoints \
                   --save_steps 100000 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.0 \
                   --validation_steps 100 \
                   --epoch 3 \
                   --max_seq_len 256 \
                   --learning_rate 5e-5 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --random_seed 1
