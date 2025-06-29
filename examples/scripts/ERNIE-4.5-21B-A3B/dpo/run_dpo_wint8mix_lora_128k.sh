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

# download model from huggingface
huggingface-cli download baidu/ERNIE-4.5-21B-A3B-Paddle --local-dir baidu/ERNIE-4.5-21B-A3B-Paddle

# # download model from aistudio
# aistudio download --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle --local_dir baidu/ERNIE-4.5-21B-A3B-Paddle
# # download model from modelscope
# modelscope download --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle --local_dir baidu/ERNIE-4.5-21B-A3B-Paddle

CUDA_VISIBLE_DEVICES=0,1,2,3 erniekit train examples/configs/ERNIE-4.5-21B-A3B/dpo/run_dpo_wint8mix_lora_128k.yaml
