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
#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate pzl
        #--train_dataset_path "ebv_data_sft_codemmsft_1003-part-00000_0.jsonl"  \
        #--train_dataset_prob "1"       \
    #--train_dataset_path "/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/contrastive_caption.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/docmatix.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/simchart9k.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/arxivqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/chart_to_text.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/icdar2019_rects.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/inat2018.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/vcr_wiki_train.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/k12_baidu_image.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/clothes_caption.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/synth_ocr_calligraphy_poetry.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/mpdocvqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/sujet_finance_qa_vision.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/docreason25k.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/ctrip_insight.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/synth_ocr_calligraphy_book.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/kvqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/ocrvqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/metamathqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/super_clevr.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/synth_ocr_calligraph_long_random.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/poie.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/chinese_ocr.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/gqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/dvqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/stvqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/pic_math150.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/ctw.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/unichart.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/laion_gpt4v.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/naf.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/infographic_to_markdown.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/mapqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/educhat_math.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/textocr.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/objects365.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/clevr.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/chartqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/edraw_svg2png.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/sbt_chart_data.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/rlaif_v.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/v3det.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/img_diff.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/pmc_vqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/koniq_10k.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/casia.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/geometry3k.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/downstream_grounding.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/birds_to_words.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/iam.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/fintabnet.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/synth_ocr_calligraph_long_poem.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/douban_movie_tv.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/allava_laion.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/scienceqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/mtwi.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/tabmwp.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/newyorker_caption_contest.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/flickr30k.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/wired_table.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/vistext.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/sharegpt4o.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/arxiv_texteq_ocr.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/synth_ocr_calligraphy_regular.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/fsc147.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/infovqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/spot_the_diff.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/gaokao.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/icdar2017_rctw_17.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/chinese_culture.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/movienet.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/synth_ocr_fanti_regular.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/viet_sharegpt_4o_text_vqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/viquae.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/geoqa_plus.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/multimodal_arxivqa52k.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/sroie.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/dreamsim.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/cyrillic_handwriting.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/mm_chemistry.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/figureqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/visual_genome.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/multi_vqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/tqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/scitsr.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/hme100k.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/chart2text.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/icdar2019_lsvt.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/ctrip_food.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/orand_car_a.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/synth_ocr_vertical_regular.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/synth_ocr_calligraph_short_idiom.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/hateful_memes.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/nlvr2.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/charts2500.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/icdar2019_art.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/estvqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/latex_qa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/art500k.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/mmc_inst.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/chrome_writing.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/mavis.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/vqa_rad.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/lrv_instruction.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/k12_question_bank.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/ai2diagram.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/drawing_to_html.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/coco.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/diagram_image_to_text.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/ocr_thai.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/visual7w.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/wildvision.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/study_com.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/geomverse.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/mmtab.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/infographic_dychart_canva.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/mtvqa_train.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/ocr_caption_chat.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/docvqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/unigeo.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/arxiv_table_ocr.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/pic_math2k.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/geos.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/plotqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/eaten.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/arxiv_equation_ocr.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/textvqa.jsonl,/root/paddlejob/workspace/output/peiziliang/data/multi_image_data_all/web_collect_data.jsonl" \
    #--train_dataset_prob "0.000508184,0.001022817,0.000337868,0.003542906,0.000981894,0.00070355,0.005525793,0.001977519,0.013199404,0.001623086,0.001771541,0.000912645,0.000347257,0.001832695,1.7718e-05,0.001771541,0.000871563,0.007311968,0.006769655,0.00212585,0.000354308,0.000199298,0.001967757,0.00255598,0.007086167,0.000670386,5.3146e-05,0.00206482,0.044721247,0.000193983,2.6395e-05,0.001694497,0.006628578,0.001743905,0.001516227,0.001214214,0.004960317,0.005353422,0.003300382,0.001972161,0.014726651,0.0,0.003269203,0.044494047,0.001947278,3.8849e-05,0.006997236,0.0,0.00023464,0.0,0.001252303,0.000598561,0.001513725,0.003169717,0.003370181,0.000171502,0.005309027,9.2084e-05,0.000266156,0.008821747,0.000529549,0.005314342,0.00057562,0.001771541,0.000324103,0.001684913,0.000283694,0.008087266,0.000141315,0.00929094,0.000952735,0.001171963,0.0,6.5493e-05,0.002562287,0.00184894,5.5449e-05,0.000847204,0.003841624,0.013086309,0.001771541,0.001183567,8.8453e-05,0.002933496,0.000212142,0.00659917,0.000477536,0.000528132,0.000168579,0.000213541,0.000802402,0.000531462,0.000300984,0.001530133,0.000440405,0.000494525,0.001509619,0.001255491,0.001596832,0.005266085,0.000781692,0.003165656,6.3527e-05,0.001275155,0.009919217,0.002510009,0.000300984,0.004826748,5.226e-05,1.571e-06,0.000254411,0.000935196,0.002605229,0.000823589,0.002659722,0.000268618,0.000118303,0.148398171,0.000996651,0.00185321,0.011419005,0.000353245,5.9878e-05,0.004831437,0.001055839,0.000296047,0.009960494,0.005314625" \
 
 
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
