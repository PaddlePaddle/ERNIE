source $1
export CUDA_VISIBLE_DEVICES=0
python3 -m paddle.distributed.launch ./finetune_mrc.py \
    --train_file $train_file \
    --dev_file $dev_file \
    --max_steps $max_steps \
    --lr $lr \
    --from_pretrained $from_pretrained \
    --save_dir checkpoints
