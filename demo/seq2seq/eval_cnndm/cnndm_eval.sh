set -x
(($#!=2)) && echo "Usage predict_file label_file" && exit -1

PRED=$1
PREFIX=$2

python pyrouge_set_rouge_path.py `pwd`/file2rouge/
python cnndm/eval.py --pred ${PRED} \
  --gold ${PREFIX} --trunc_len 100 --perl
