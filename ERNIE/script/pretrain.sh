set -eux

export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -u ./train.py --use_cuda True \
                --is_distributed False\
                --use_fast_executor True \
                --weight_sharing True \
                --in_tokens true \
                --batch_size 8192 \
                --vocab_path ./config/vocab.txt \
                --train_filelist ./data/train_filelist \
                --valid_filelist ./data/valid_filelist \
                --validation_steps 100 \
                --num_train_steps 1000000 \
                --checkpoints ./checkpoints \
                --save_steps 10000 \
                --ernie_config_path ./config/ernie_config.json \
                --learning_rate 1e-4 \
                --weight_decay 0.01 \
                --max_seq_len 512 \
                --skip_steps 10
