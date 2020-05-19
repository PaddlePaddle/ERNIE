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
cat $PRED | python clean.py >$2.clean 2>${EVAL_SCRIPT_LOG}
python eval.py $2.clean ${PREFIX}.tgt 2>>${EVAL_SCRIPT_LOG}
