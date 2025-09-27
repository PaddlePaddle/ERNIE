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

dataset_output_path="./sft-data"
model_path="./to-your-model-path"
mkdir -p $dataset_output_path

python examples/post-training/sft/create_sft_data.py \
    --train_dataset_path "examples/data/sft-train.jsonl" \
    --train_dataset_prob "1.0" \
    --train_dataset_type "erniekit" \
    --eval_dataset_path "examples/data/sft-eval.jsonl" \
    --eval_dataset_prob "1.0" \
    --eval_dataset_type "erniekit" \
    --model_name_or_path $model_path \
    --num_of_gpus 1 \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 1 \
    --num_samples_each_epoch 6000000 \
    --dataset_output_dir $dataset_output_path  \
    --seed 23 \
    --max_seq_len 8192 \
    --max_steps 1200 \
    --num_train_epochs 1.0 \
    --do_train \
    --do_eval \
    --tp_degree 1 \
    --sdp_degree 1 \
    --pp_degree 1 \
    --random_shuffle \
    --greedy_intokens \
    --estimation_output_file estimate_training.json
