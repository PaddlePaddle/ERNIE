#!/bin/bash
set -ex
R_DIR=`dirname $0`; MY_DIR=`cd $R_DIR;pwd`; 

INPUT=$1

if [[ ! -d ./data/ ]];then
        mkdir ./data/
        fi


### CoLA
mkdir -p ./data/CoLA
cat $INPUT/CoLA/train.tsv | awk -F"\t"  '{if(NR==1){print "label\ttext_a"} else {print $2"\t"$4}}' > ./data/CoLA/train.tsv
cat $INPUT/CoLA/dev.tsv   | awk -F"\t"  '{if(NR==1){print "label\ttext_a"} else {print $2"\t"$4}}' > ./data/CoLA/dev.tsv
cat $INPUT/CoLA/test.tsv  | awk -F"\t"  '{if(NR==1){print "qid\ttext_a\tlabel"}   else {print $0"\t-1"}}'       > ./data/CoLA/test.tsv

### SST-2
mkdir -p ./data/SST-2
cat $INPUT/SST-2/train.tsv | awk -F"\t"    '{if(NR==1){print "label\ttext_a"}  else if(NF==2) {print $2"\t"$1}}' > ./data/SST-2/train.tsv
cat $INPUT/SST-2/dev.tsv   | awk -F"\t"    '{if(NR==1){print "label\ttext_a"}  else if(NF==2) {print $2"\t"$1}}' > ./data/SST-2/dev.tsv
cat $INPUT/SST-2/test.tsv  | awk -F"\t"    '{if(NR==1){print "qid\ttext_a\tlabel"}    else {print $0"\t-1"}}'    > ./data/SST-2/test.tsv

### MRPC
mkdir -p ./data/MRPC
cat $INPUT/MRPC/train.tsv | awk -F"\t" '{if(NR==1){print "text_a\ttext_b\tlabel"} else{print $4"\t"$5"\t"$1}}' > ./data/MRPC/train.tsv
cat $INPUT/MRPC/dev.tsv   | awk -F"\t" '{if(NR==1){print "text_a\ttext_b\tlabel"} else{print $4"\t"$5"\t"$1}}' > ./data/MRPC/dev.tsv
cat $INPUT/MRPC/test.tsv  | awk -F"\t" '{if(NR==1){print "qid\ttext_a\ttext_b\tlabel"}   else{print $1"\t"$4"\t"$5"\t-1"}}' > ./data/MRPC/test.tsv

### STS-B
mkdir -p ./data/STS-B
cat $INPUT/STS-B/train.tsv | awk -F"\t" '{if(NR==1){print "text_a\ttext_b\tlabel"} else{print $8"\t"$9"\t"$10}}' > ./data/STS-B/train.tsv
cat $INPUT/STS-B/dev.tsv   | awk -F"\t" '{if(NR==1){print "text_a\ttext_b\tlabel"} else{print $8"\t"$9"\t"$10}}' > ./data/STS-B/dev.tsv
cat $INPUT/STS-B/test.tsv  | awk -F"\t" '{if(NR==1){print "qid\ttext_a\ttext_b\tlabel"}   else{print $1"\t"$8"\t"$9"\t-1"}}'  > ./data/STS-B/test.tsv

### QQP
mkdir -p ./data/QQP
cat $INPUT/QQP/train.tsv | awk -F"\t"  '{if(NR==1){print "text_a\ttext_b\tlabel"} else if($6!="") {print $4"\t"$5"\t"$6}}' > ./data/QQP/train.tsv
cat $INPUT/QQP/dev.tsv   | awk -F"\t"  '{if(NR==1){print "text_a\ttext_b\tlabel"} else if($6!="") {print $4"\t"$5"\t"$6}}' > ./data/QQP/dev.tsv
cat $INPUT/QQP/test.tsv  | awk -F"\t"  '{if(NR==1){print "qid\ttext_a\ttext_b\tlabel"}   else {print $0"\t-1"}}'           > ./data/QQP/test.tsv

### MNLI
mkdir -p ./data/MNLI
cat $INPUT/MNLI/train.tsv            | python $MY_DIR/mnli.py > ./data/MNLI/train.tsv

mkdir -p ./data/MNLI/m
cat $INPUT/MNLI/dev_matched.tsv      | python $MY_DIR/mnli.py > ./data/MNLI/m/dev.tsv
cat $INPUT/MNLI/test_matched.tsv     | python $MY_DIR/mnli.py > ./data/MNLI/m/test.tsv

mkdir -p ./data/MNLI/mm
cat $INPUT/MNLI/dev_mismatched.tsv   | python $MY_DIR/mnli.py  > ./data/MNLI/mm/dev.tsv
cat $INPUT/MNLI/test_mismatched.tsv  | python $MY_DIR/mnli.py > ./data/MNLI/mm/test.tsv

### QNLI
mkdir -p ./data/QNLI
cat $INPUT/QNLI/train.tsv | python $MY_DIR/qnli.py > ./data/QNLI/train.tsv
cat $INPUT/QNLI/dev.tsv   | python $MY_DIR/qnli.py > ./data/QNLI/dev.tsv
cat $INPUT/QNLI/test.tsv  | python $MY_DIR/qnli.py > ./data/QNLI/test.tsv

### RTE
mkdir -p ./data/RTE
cat $INPUT/RTE/train.tsv | python $MY_DIR/qnli.py > ./data/RTE/train.tsv
cat $INPUT/RTE/dev.tsv   | python $MY_DIR/qnli.py > ./data/RTE/dev.tsv
cat $INPUT/RTE/test.tsv  | python $MY_DIR/qnli.py > ./data/RTE/test.tsv

### WNLI
mkdir -p ./data/WNLI
cat $INPUT/WNLI/train.tsv | awk -F"\t"  '{if(NR==1){print "text_a\ttext_b\tlabel"} else {print $2"\t"$3"\t"$4}}' > ./data/WNLI/train.tsv
cat $INPUT/WNLI/dev.tsv   | awk -F"\t"  '{if(NR==1){print "text_a\ttext_b\tlabel"} else {print $2"\t"$3"\t"$4}}' > ./data/WNLI/dev.tsv
cat $INPUT/WNLI/test.tsv  | awk -F"\t"  '{if(NR==1){print "qid\ttext_a\ttext_b\tlabel"}   else {print $1"\t"$2"\t"$3"\t-1"}}' > ./data/WNLI/test.tsv

### Diagnostics
cat $INPUT/diagnostic/diagnostic.tsv | awk -F"\t"  '{if(NR==1){print "qid\ttext_a\ttext_b\tlabel"} else {print $0"\t-1"}}'         > ./data/MNLI/diagnostic.tsv



