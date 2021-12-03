export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

distributed_args="--node_ips $(hostname -i) --node_id 0 --current_node_ip $(hostname -i) --log_prefix xnli_"
python -u ./launch.py ${distributed_args} \
   ./run_classifier.py --use_cuda true \
   --is_distributed true \
   --do_train true \
   --do_val true \
   --do_test false \
   --in_tokens false \
   --batch_size 16 \
   --train_set ./data/xnli/train/train.all.tsv \
   --dev_set ./data/xnli/dev/dev.%s.tsv,./data/xnli/test/test.%s.tsv \
   --test_set ./data/xnli/test/test.%s.tsv \
   --test_save ./output/xnli/test/test.%s.tsv \
   --label_map_config ./data/xnli/label_map.json \
   --lang_map_config ./data/xnli/lang_map.json \
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
   --validation_steps 1000 \
   --epoch 2 \
   --max_seq_len 256 \
   --learning_rate 5e-5 \
   --skip_steps 10 \
   --num_iteration_per_drop_scope 1 \
   --num_labels 3 \
   --random_seed 1
