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

export PATH=/usr/local/corex/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib
export LIBRARY_PATH=/usr/local/corex/lib
export LD_PRELOAD="${LD_LIBRARY_PATH}/libcuda.so.1"

LORA_BENCHMARK_PATH="/home/datasets/lora_benchmark"
TEST_FILE="test_iluvatar_gpu_moe_model_lora.py"

set -e

echo "==============================="
echo " LoRA E2E Test Runner Starting "
echo "==============================="

rm -rf ./data ./model
ln -sf "${LORA_BENCHMARK_PATH}/data" ./data
ln -sf "${LORA_BENCHMARK_PATH}/model" ./model

echo "Running test: $TEST_FILE::TestLoRATrainingE2E::test_lora_e2e"

set +e
pytest "$TEST_FILE::TestLoRATrainingE2E::test_lora_e2e"
TEST_EXIT_CODE=$?
set -e

echo "Test execution finished with exit code: $TEST_EXIT_CODE"

rm -rf ./data ./model ./log_lora_8k ./__pycache__
rm -rf "${LORA_BENCHMARK_PATH}/model/ERNIE-4.5-21B-A3B-Paddle_lora_8k_checkpoint"
rm -rf "${LORA_BENCHMARK_PATH}/model/ERNIE-4.5-21B-A3B-Paddle_lora_8k_vdl"

echo "==============================="
echo " LoRA E2E Test Runner Finished "
echo "==============================="

exit $TEST_EXIT_CODE
