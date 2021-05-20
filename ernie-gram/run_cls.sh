source $1

python3 -m paddle.distributed.launch ./finetune_classifier_distributed.py \
    --data_dir $data_dir \
    --max_steps $max_steps \
    --bsz $bsz \
    --lr $lr \
    --label_map ${label_map:-""} \
    --num_labels $num_labels \
    --pair_input $pair_input \
    --valid_steps $valid_steps \
    --from_pretrained $from_pretrained \
    --save_dir checkpoints
