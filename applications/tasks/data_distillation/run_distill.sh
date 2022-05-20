#!/bin/bash
# set -x
export ALL_TRAINERS_SAVE=1
export CUDA_VISIBLE_DEVICES=0
# step 1
python run_trainer.py --param_path ./examples/cls_ernie_fc_ch.json
(($?!=0)) && echo "Something goes wrong at Step 1, please check" && exit -1

# step 2:
mkdir -p predict_input
cat distill/chnsenticorp/student/unsup_train_aug/part.0 |awk -F"\t" '{print $2}' > predict_input/part.0
python run_infer.py --param_path ./examples/cls_ernie_fc_ch_infer.json > predict_score
(($?!=0)) && echo "Something goes wrong at Step 2, please check" && exit -1
paste distill/chnsenticorp/student/unsup_train_aug/part.0 ./output/predict_result.txt | awk -F"\t" '{split($4,prob,", ");
prob1=strtonum(substr(prob[1],2));prob2=strtonum(substr(prob[2],1,length(prob[2])-1));
if(prob1>prob2)print $1"\t"0; else print $1"\t"1}' > distill/chnsenticorp/student/train/part.1

export ALL_TRAINERS_SAVE=0
# step 3:
python run_trainer.py --param_path ./examples/cls_cnn_ch.json
(($?!=0)) && echo "Something goes wrong at Step 3, please check" && exit -1
exit 0
