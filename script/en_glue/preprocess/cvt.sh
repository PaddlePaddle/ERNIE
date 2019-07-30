#!/bin/bash
set -ex
R_DIR=`dirname $0`; MY_DIR=`cd $R_DIR;pwd`; 

INPUT=$1

if [[ ! -d ./glue_data_processed/ ]];then
    mkdir ./glue_data_processed/
fi


### CoLA
mkdir -p ./glue_data_processed/CoLA
cat $INPUT/CoLA/train.tsv | awk -F"\t"  '{if(NR==1){print "label\ttext_a"} else {print $2"\t"$4}}' > ./glue_data_processed/CoLA/train.tsv
cat $INPUT/CoLA/dev.tsv   | awk -F"\t"  '{if(NR==1){print "label\ttext_a"} else {print $2"\t"$4}}' > ./glue_data_processed/CoLA/dev.tsv
cat $INPUT/CoLA/test.tsv  | awk -F"\t"  '{if(NR==1){print "qid\ttext_a\tlabel"}   else {print $0"\t-1"}}'       > ./glue_data_processed/CoLA/test.tsv

### SST-2
mkdir -p ./glue_data_processed/SST-2
cat $INPUT/SST-2/train.tsv | awk -F"\t"    '{if(NR==1){print "label\ttext_a"}  else if(NF==2) {print $2"\t"$1}}' > ./glue_data_processed/SST-2/train.tsv
cat $INPUT/SST-2/dev.tsv   | awk -F"\t"    '{if(NR==1){print "label\ttext_a"}  else if(NF==2) {print $2"\t"$1}}' > ./glue_data_processed/SST-2/dev.tsv
cat $INPUT/SST-2/test.tsv  | awk -F"\t"    '{if(NR==1){print "qid\ttext_a\tlabel"}    else {print $0"\t-1"}}'    > ./glue_data_processed/SST-2/test.tsv

### MRPC
mkdir -p ./glue_data_processed/MRPC
cat $INPUT/MRPC/train.tsv | awk -F"\t" '{if(NR==1){print "text_a\ttext_b\tlabel"} else{print $4"\t"$5"\t"$1}}' > ./glue_data_processed/MRPC/train.tsv
cat $INPUT/MRPC/dev.tsv   | awk -F"\t" '{if(NR==1){print "text_a\ttext_b\tlabel"} else{print $4"\t"$5"\t"$1}}' > ./glue_data_processed/MRPC/dev.tsv
cat $INPUT/MRPC/test.tsv  | awk -F"\t" '{if(NR==1){print "qid\ttext_a\ttext_b\tlabel"}   else{print $1"\t"$4"\t"$5"\t-1"}}' > ./glue_data_processed/MRPC/test.tsv

### STS-B
mkdir -p ./glue_data_processed/STS-B
cat $INPUT/STS-B/train.tsv | awk -F"\t" '{if(NR==1){print "text_a\ttext_b\tlabel"} else{print $8"\t"$9"\t"$10}}' > ./glue_data_processed/STS-B/train.tsv
cat $INPUT/STS-B/dev.tsv   | awk -F"\t" '{if(NR==1){print "text_a\ttext_b\tlabel"} else{print $8"\t"$9"\t"$10}}' > ./glue_data_processed/STS-B/dev.tsv
cat $INPUT/STS-B/test.tsv  | awk -F"\t" '{if(NR==1){print "qid\ttext_a\ttext_b\tlabel"}   else{print $1"\t"$8"\t"$9"\t-1"}}'  > ./glue_data_processed/STS-B/test.tsv

### QQP
mkdir -p ./glue_data_processed/QQP
cat $INPUT/QQP/train.tsv | awk -F"\t"  '{if(NR==1){print "text_a\ttext_b\tlabel"} else if($6!="") {print $4"\t"$5"\t"$6}}' > ./glue_data_processed/QQP/train.tsv
cat $INPUT/QQP/dev.tsv   | awk -F"\t"  '{if(NR==1){print "text_a\ttext_b\tlabel"} else if($6!="") {print $4"\t"$5"\t"$6}}' > ./glue_data_processed/QQP/dev.tsv
cat $INPUT/QQP/test.tsv  | awk -F"\t"  '{if(NR==1){print "qid\ttext_a\ttext_b\tlabel"}   else {print $0"\t-1"}}'           > ./glue_data_processed/QQP/test.tsv

### MNLI
mkdir -p ./glue_data_processed/MNLI
cat $INPUT/MNLI/train.tsv            | python $MY_DIR/mnli.py > ./glue_data_processed/MNLI/train.tsv

mkdir -p ./glue_data_processed/MNLI/m
cat $INPUT/MNLI/dev_matched.tsv      | python $MY_DIR/mnli.py > ./glue_data_processed/MNLI/m/dev.tsv
cat $INPUT/MNLI/test_matched.tsv     | python $MY_DIR/mnli.py > ./glue_data_processed/MNLI/m/test.tsv

mkdir -p ./glue_data_processed/MNLI/mm
cat $INPUT/MNLI/dev_mismatched.tsv   | python $MY_DIR/mnli.py  > ./glue_data_processed/MNLI/mm/dev.tsv
cat $INPUT/MNLI/test_mismatched.tsv  | python $MY_DIR/mnli.py > ./glue_data_processed/MNLI/mm/test.tsv

### QNLI
mkdir -p ./glue_data_processed/QNLI
cat $INPUT/QNLI/train.tsv | python $MY_DIR/qnli.py > ./glue_data_processed/QNLI/train.tsv
cat $INPUT/QNLI/dev.tsv   | python $MY_DIR/qnli.py > ./glue_data_processed/QNLI/dev.tsv
cat $INPUT/QNLI/test.tsv  | python $MY_DIR/qnli.py > ./glue_data_processed/QNLI/test.tsv

### RTE
mkdir -p ./glue_data_processed/RTE
cat $INPUT/RTE/train.tsv | python $MY_DIR/qnli.py > ./glue_data_processed/RTE/train.tsv
cat $INPUT/RTE/dev.tsv   | python $MY_DIR/qnli.py > ./glue_data_processed/RTE/dev.tsv
cat $INPUT/RTE/test.tsv  | python $MY_DIR/qnli.py > ./glue_data_processed/RTE/test.tsv

### WNLI
mkdir -p ./glue_data_processed/WNLI
cat $INPUT/WNLI/train.tsv | awk -F"\t"  '{if(NR==1){print "text_a\ttext_b\tlabel"} else {print $2"\t"$3"\t"$4}}' > ./glue_data_processed/WNLI/train.tsv
cat $INPUT/WNLI/dev.tsv   | awk -F"\t"  '{if(NR==1){print "text_a\ttext_b\tlabel"} else {print $2"\t"$3"\t"$4}}' > ./glue_data_processed/WNLI/dev.tsv
cat $INPUT/WNLI/test.tsv  | awk -F"\t"  '{if(NR==1){print "qid\ttext_a\ttext_b\tlabel"}   else {print $1"\t"$2"\t"$3"\t-1"}}' > ./glue_data_processed/WNLI/test.tsv

### Diagnostics
cat $INPUT/diagnostic/diagnostic.tsv | awk -F"\t"  '{if(NR==1){print "qid\ttext_a\ttext_b\tlabel"} else {print $0"\t-1"}}'         > ./glue_data_processed/MNLI/diagnostic.tsv


