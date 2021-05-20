source $1

python3 -m paddle.distributed.launch ./finetune_ner.py \
    --data_dir $data_dir \
    --max_steps $max_steps \
    --epoch $epoch \
    --lr $lr \
    --from_pretrained $from_pretrained \
    --save_dir checkpoints
