export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0

python -u ./encode_vector.py --use_cuda true \
   --batch_size 16 \
   --data_set ./data/xnli/test/test.en.tsv \
   --output_dir ./output/xnli/test/test.en.tsv \
   --vocab_path ./configs/vocab.txt \
   --piece_model_path ./configs/piece.model \
   --ernie_config_path ./configs/large/ernie_config.json \
   --init_pretraining_params "./configs/large/params" \
   --max_seq_len 256
