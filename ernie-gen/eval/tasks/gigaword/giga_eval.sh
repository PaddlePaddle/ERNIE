set -x

PRED=$1
PREFIX=$2

python pyrouge_set_rouge_path.py `pwd`/file2rouge/
python gigaword/eval.py --pred ${PRED} \
  --gold ${PREFIX}.tgt.txt --perl
