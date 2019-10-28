set -eux

export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export PYTHONPATH=./ernie:${PYTHONPATH:-}
python ./ernie/finetune_launch.py  \
    --nproc_per_node 8 \
    --selected_gpus 0,1,2,3,4,5,6,7 \
    --node_ips $(hostname -i) \
    --node_id 0 \
./ernie/run_mrc.py --use_cuda true\
                    --batch_size 8 \
                    --in_tokens false\
                    --use_fast_executor true \
                    --checkpoints ./checkpoints \
                    --vocab_path ./config/vocab.txt  \
                    --ernie_config_path ./config/ernie_config.json \
                    --do_train true \
                    --do_val true \
                    --do_test true \
                    --verbose true \
                    --save_steps 1000 \
                    --validation_steps 100 \
                    --warmup_proportion 0.0 \
                    --weight_decay  0.01 \
                    --epoch 2 \
                    --max_seq_len 512 \
                    --do_lower_case true \
                    --doc_stride 128 \
                    --train_set ${TASK_DATA_PATH}/drcd/train.json \
                    --dev_set ${TASK_DATA_PATH}/drcd/dev.json \
                    --test_set ${TASK_DATA_PATH}/drcd/test.json \
                    --learning_rate 3e-5 \
                    --num_iteration_per_drop_scope 1 \
                    --init_pretraining_params ${MODEL_PATH}/params \
                    --skip_steps 10

