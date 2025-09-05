#!/bin/bash
 
unset PADDLE_TRAINERS_NUM
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
 
mpi_rank=${OMPI_COMM_WORLD_RANK:-0}
node_rank=$((mpi_rank+offset))
mpi_node=${OMPI_COMM_WORLD_SIZE:-1}
echo "MPI status:${mpi_rank}/${mpi_node}"
nnode_train=${nnode_set:-${mpi_node}}
master_train=${master:-localhost}
#
echo "Distributed Training ${node_rank}/${nnode_train} master=${master_train}"
set -x
# 屏蔽平台预设的环境变量，因为框架采用兼容升级，检测到这些配置会使用原方式启动
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
nnodes=1
rank=$PADDLE_TRAINER_ID
#nnodes=36
 
 
### 0715 debug
export NCCL_DEBUG=INFO
#export GLOG_vmodule=dygraph_functions=3,nodes=3,tracer=3,process_group_nccl=3
#export FLAGS_benchmark=1
unset GLOG_vmodule GLOG_v
###
export PYTHONUNBUFFERED=1
#加速pin memory save ckpt时间
export FLAGS_use_auto_growth_pinned_allocator=True
# 保证集群稳定性的配置，跟性能无关
export NCCL_IB_QPS_PER_CONNECTION=8 
export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=3
export NCCL_NVLS_ENABLE=0
# 开启AR功能
export NCCL_IB_ADAPTIVE_ROUTING=1
#开启BCCL流量统计
export BCCL_BUS_BW_CALCULATE_MODE=Agg 
# 集群hang检测
export PADDLE_PG_TIMEOUT=150000   # 通信组超时时间，单位是ms，默认2分钟
export FLAGS_enable_async_trace=False # True开启通信debug功能，False或不设置关闭，默认开启
enable_nccl_proxy_dump=True # enable hang trace through BCCLe
export CUDA_MODULE_LOADING=LAZY
if [[ $enable_nccl_proxy_dump == "True" ]];then
    export NCCL_PROXY_DUMP_SIGNAL=10
fi
 
export FLAGS_pipeline_nccl_comm_init_option=1
 
# 开启Sharding V2 Padding Zero检查
export FLAGS_sharding_v2_check_zero_padding=1
 
export FLAGS_use_paddle_recall_error=0
# 关闭 H 卡 CUDNN FA 功能
export PADDLE_DISABLE_CUDNN_FA=1
# 释放shmem
find /dev/shm/ -type f -name "paddle_*" -print0 | xargs -0 rm -f
# 启动方式
cuda_version=`nvidia-smi |grep "CUDA Version" |awk '{print $9}' |awk -F'.' '{print $1}'`
if [ ${cuda_version} != "12" ];then
    export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
fi
 
master=`cat /root/paddlejob/workspace/hostfile | head -n 1 | awk '{print $1}'`
port=36677
 
export FLAGS_call_stack_level=2
#export GLOG_v=10
#export FLAGS_use_cuda_managed_memory=1
 
export FLAGS_eager_communication_connection=0
 
model_path="/root/paddlejob/workspace/env_run/lrl/new/ERNIE/baidu/ERNIE-4.5-VL-28B-A3B-Paddle"
task="sft_vl_demo"
paddle_log_dir="./output"
output_dir="../output/models"
train_log_name="lite_erniekits_test"

python -m paddle.distributed.launch \
    --log_dir output/paddle_distributed_logs \
    --master $master:$port \
    --nnodes $nnodes \
        examples/post-training/vl_sft/train.py \
    --model_name_or_path ${model_path} \
    --output_dir ${output_dir} \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --train_dataset_path "examples/data/sft_vl-train_demo1.jsonl" \
    --train_dataset_prob "1" \
    --max_steps 8000 \
    --save_steps 10000 \
    --from_scratch 0 \
    --vit_lr_ratio 0.9 \
    --freeze_config "freeze_vision" \
    --multimodal true \
    --data_load_process_num 1 \
    --max_seq_length 8192 \
    --base_seq_length 8192 \
    --variable_resolution 1 \
    --global_shuffle_num_examples 1000000 \
    --sequence_parallel 1 \
    --moe_use_aux_free_update_coef 0.0 \
    --visual_ld 0.9 \
    --modality_ratio [1,1] \
    --moe_gate_lr_ratio 0.01 \
    --dataloader_num_workers 1 \
    --dataset_name "KnowledgeBasedSFTReader" \
    --add_sys_token true \
    --number_of_samples_each_epoch 10000000 \
    --trigger_data_prob 1.0 \
    --drop_history_with_k true \
    --prefetch_factor 10 \
    --one_sample_in_one_seq true \
    --serialize_output false \
    --render_timestamp true \
    --pp_need_data  true \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1.0e-08 \
    --bf16 true \
    --do_train true \
    --fp16_opt_level O2 \
    --learning_rate 1e-05 \
    --logging_steps 1 \
    --lr_scheduler_type cosine \
    --min_lr 1e-06 \
    --same_data true \
    --load_sharded_model true \
    --save_sharded_model true \
    --scale_loss 4096 \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --overwrite_output_dir 1 \
    --pp_need_data_degree 4 \
    --pipeline_parallel_degree 4 \
    --tensor_parallel_degree 2 \
    --virtual_pp_degree 7 \
    --sharding "stage1" \
    --amp_master_grad 1 \
    --pipeline_parallel_config "enable_offload_queue enable_delay_scale_loss enable_overlap_p2p_comm best_unbalanced_scheduler" \
    --sharding_parallel_config "split_param enable_fuse_optimizer_states" \
    --tensor_parallel_config "sync_param sync_grad sync_moment" \
    --pre_alloc_memory 60 \
    --sharding_comm_buffer_size_MB 2048 \
    --use_moe true \
    --moe_group mp \
    --gc_interval 100000 \
    --skip_profile_timer 0 \
    --save_sharding_stage1_model_include_freeze_params true \
    --disable_pipeline_warmup False \
    --use_flash_attention 1 \
    --unified_checkpoint true
