set -eu
set -x

#bash -x ./env.sh

TASK_NAME=$1
CONF_FILE=$2
VOCAB_PATH=$3
ERNIE_VIL_CONFIG=$4
PRETRAIN_MODELS=$5

source $CONF_FILE

#configure your cuda and cudnn 
#configure nccl

export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

e_executor=$(echo ${use_experimental_executor-'True'} | tr '[A-Z]' '[a-z]')

use_fuse=$(echo ${use_fuse-'False'} | tr '[A-Z]' '[a-z]')
if [[ ${use_fuse} == "true" ]]; then
    export FLAGS_fuse_parameter_memory_size=131072
    export FLAGS_fuse_parameter_groups_size=10
fi


TASK_GROUP_JSON=./conf/$TASK_NAME/task_${TASK_NAME}.json

gpu_cnt=`echo $CUDA_VISIBLE_DEVICES | awk -F"\t" '{len=split($0,vec,",");print len}'`
echo "gpu_cnt", $gpu_cnt
python finetune.py --use_cuda "True"             \
                --is_distributed "False"                                       \
                --use_fast_executor ${e_executor-"True"}                       \
                --nccl_comm_num ${nccl_comm_num:-"1"}                          \
                --batch_size $((BATCH_SIZE/gpu_cnt))                                   \
                --do_train "True"  \
                --do_test "False"     \
                --task_name ${TASK_NAME}                      \
                --vocab_path ${VOCAB_PATH}                                     \
                --task_group_json ${TASK_GROUP_JSON}                           \
                --lr_scheduler ${lr_scheduler}                                 \
                --decay_steps ${decay_steps-""}                                 \
                --lr_decay_ratio ${lr_decay_ratio-0.1}                                 \
                --num_train_steps ${num_train_steps}                           \
                --checkpoints $output_model_path                                       \
                --save_steps ${SAVE_STEPS}                                     \
                --init_checkpoint ${PRETRAIN_MODELS}                                 \
                --ernie_config_path ${ERNIE_VIL_CONFIG}                             \
                --learning_rate ${LR_RATE}                                     \
                --warmup_steps ${WARMUP_STEPS}                                               \
                --weight_decay ${WEIGHT_DECAY:-0}                              \
                --max_seq_len ${MAX_LEN}                                       \
                --validation_steps ${VALID_STEPS}                              \
                --skip_steps 10 


