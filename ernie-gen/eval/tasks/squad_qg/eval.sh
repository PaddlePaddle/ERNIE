PRED=`pwd`"/"$1

if [[ $2 == "dev" ]];then
    EVAL_PREFIX=$DEV_PREFIX
elif [[ $2 == "test" ]];then
    EVAL_PREFIX=$TEST_PREFIX
elif [[ $2 == "pred" ]];then
    EVAL_PREFIX=$PRED_PREFIX
fi
PREFIX=`pwd`"/"${TASK_DATA_PATH}"/"${EVAL_PREFIX}

cd `dirname $0`
sh qg_eval.sh $PRED $PREFIX 2>${EVAL_SCRIPT_LOG} | grep -E "Bleu_4|METEOR|ROUGE_L" | python join.py
