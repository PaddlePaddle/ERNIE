set -x

PRED=$1
PREFIX=$2

python qg/eval_on_unilm_tokenized_ref.py \
    --src ${PREFIX}.pa.tok.txt \
    --tgt ${PREFIX}.q.tok.txt \
    --out_file ${PRED}

