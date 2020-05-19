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
python gen_format.py ${PRED} ${PREFIX}.info $2.out 2>${EVAL_SCRIPT_LOG}
python eval.py --pred-file $2.out --data-file ${PREFIX}.json 2>>${EVAL_SCRIPT_LOG}
