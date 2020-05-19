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
sh cnndm_eval.sh $PRED $PREFIX 2>${EVAL_SCRIPT_LOG} | grep ROUGE-F | awk -F ": " '{print $2}' | awk -F "/" '{print "{\"rouge-1\": "$1", \"rouge-2\": "$2", \"rouge-l\": "$3"}"}'
