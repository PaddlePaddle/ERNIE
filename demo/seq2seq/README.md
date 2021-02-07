# ERNIE-GEN

[ERNIE-GEN](https://arxiv.org/pdf/2001.11314.pdf) is a multi-flow language generation framework for both pre-training and fine-tuning.
Only finetune strategy is illustrated in this section.

## Finetune

We use Abstractive Summarization task CNN/DailyMail to illustate usage of ERNIE-GEN, you can download preprocessed finetune data from [here](https://ernie-github.cdn.bcebos.com/data-cnndm.tar.gz)

To starts finetuning ERNIE-GEN, run:

```script
python3 -m paddle.distributed.launch \
    --log_dir ./log  \
    ./demo/seq2seq/finetune_seq2seq_dygraph.py \
    --from_pretrained ernie-gen-base-en \
    --data_dir ./data/cnndm \
    --save_dir ./model_cnndm \
    --label_smooth 0.1 \
    --use_random_noice \
    --noise_prob 0.7 \
    --predict_output_dir ./pred \
    --max_steps $((287113*30/64))
```

Note that you need more than 2 GPUs to run the finetuning.
During multi-gpu finetuning, `max_steps` is used as stop criteria rather than `epoch` to prevent dead block.
We simply canculate `max_steps` with: `EPOCH * NUM_TRIAN_EXAMPLE / TOTAL_BATCH`.
This demo script will save a finetuned model at `--save_dir`, and do muti-gpu prediction every `--eval_steps` and save prediction results at `--predict_output_dir`.


### Evalution

While finetuning, a serials of prediction files is generated.
First you need to sort and join all files with:

```shell
sort -t$'\t' -k1n ./pred/pred.step60000.* |awk -F"\t" '{print $2}'> final_prediction
```

then use `./eval_cnndm/cnndm_eval.sh` to calcuate all metrics
(`pyrouge` is required to evalute CNN/Daily Mail.)

```shell
sh cnndm_eval.sh final_prediction ./data/cnndm/dev.summary
```


### Inference

To run beam serach decode after you got a finetuned model. try:

```shell

cat one_column_source_text| python3 demo/seq2seq/decode.py \
    --from_pretrained ./ernie_gen_large \
    --save_dir ./model_cnndm \
    --bsz 8
```
