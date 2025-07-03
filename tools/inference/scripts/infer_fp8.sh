#!/bin/bash
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export PYTHONPATH=$(dirname "$0")/..:$PYTHONPATH
export FLAGS_set_to_1d=False
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_dataloader_use_file_descriptor=False

unset PADDLE_TRAINERS_NUM
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

top_p=0.7
temperature=0.95

rm -rf log
python -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir ./log \
    ./tools/inference/infer.py \
    --model_name_or_path "./new-4p5-model/" \
    --batch_size 1 \
    --input_file ./tools/inference/data/query-demo.jsonl \
    --output_file ./predict_sft_out.txt \
    --max_seq_len 4096 \
    --min_dec_len 1 \
    --max_dec_len 2048 \
    --top_p ${top_p} \
    --temperature ${temperature} \
    --weight_quantize_algo fp8linear \
    --data_format "chat"
