export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

distributed_args="--node_ips $(hostname -i) --node_id 0 --current_node_ip $(hostname -i) --log_prefix mlqa_ --nproc_per_node 2 --selected_gpus 0,1"
python -u ./launch.py ${distributed_args} \
   ./run_mrc.py --use_cuda true \
   --is_distributed true \
   --do_train true \
   --do_val true \
   --do_test false \
   --in_tokens false \
   --batch_size 16 \
   --train_set ./data/mlqa/train/train.en.json \
   --dev_set ./data/mlqa/dev/dev.%s.json,./data/mlqa/test/test.%s.json \
   --test_set ./data/mlqa/test/test.%s.json \
   --test_save ./output/mlqa/test/test.%s.json \
   --lang_map_config ./data/mlqa/lang_map.json \
   --vocab_path ./configs/vocab.txt \
   --piece_model_path ./configs/piece.model \
   --ernie_config_path ./configs/base/ernie_config.json \
   --init_pretraining_params "./configs/base/params" \
   --checkpoints ./checkpoints \
   --save_steps 10000 \
   --weight_decay  0.0 \
   --layerwise_lr_decay 0.8 \
   --warmup_proportion 0.1 \
   --use_fp16 false \
   --validation_steps 100 \
   --epoch 2 \
   --doc_stride 128 \
   --max_seq_len 384 \
   --learning_rate 3e-4 \
   --skip_steps 10 \
   --num_iteration_per_drop_scope 1 \
   --random_seed 1
