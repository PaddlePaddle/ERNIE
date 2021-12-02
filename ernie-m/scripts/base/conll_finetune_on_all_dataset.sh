export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

distributed_args="--node_ips $(hostname -i) --node_id 0 --current_node_ip $(hostname -i) --log_prefix conll_ --nproc_per_node 1 --selected_gpus 0"
python -u ./launch.py ${distributed_args} \
   ./run_sequence_labeling.py --use_cuda true \
   --is_distributed true \
   --do_train true \
   --do_val true \
   --do_test false \
   --in_tokens false \
   --batch_size 8 \
   --train_set ./data/conll/train/train.all.tsv \
   --dev_set ./data/conll/dev/dev.%s.tsv,./data/conll/test/test.%s.tsv \
   --test_set ./data/conll/test/test.%s.tsv \
   --test_save ./output/conll/test/test.%s.tsv \
   --label_map_config ./data/conll/label_map.json \
   --lang_map_config ./data/conll/lang_map.json \
   --vocab_path ./configs/vocab.txt \
   --piece_model_path ./configs/piece.model \
   --ernie_config_path ./configs/base/ernie_config.json \
   --init_pretraining_params "./configs/base/params" \
   --checkpoints ./checkpoints \
   --save_steps 10000 \
   --weight_decay  0.01 \
   --layerwise_lr_decay 0.8 \
   --warmup_proportion 0.1 \
   --use_fp16 false \
   --validation_steps 100 \
   --epoch 10 \
   --max_seq_len 512 \
   --learning_rate 3e-4 \
   --skip_steps 10 \
   --num_iteration_per_drop_scope 1 \
   --num_labels 9 \
   --random_seed 1
