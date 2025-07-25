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

import os
import sys
import shutil
import subprocess
import allure
import pytest
from pathlib import Path


@allure.epic("End-to-End Tests")
@allure.feature("SFT Training")
class TestSFTTrainingE2E:
    """
    End-to-end test suite for SFT.
    """

    @allure.story("ERNIE-4.5 8K Dummy Data Training")
    @allure.title("Test E2E SFT Training on a Single Node")
    @allure.description(
        "This test runs the SFT training script with dummy data to verify the "
        "end-to-end training pipeline on 8 XPUs. It checks for successful completion "
        "and the creation of checkpoint files."
    )
    def test_sft_e2e(self):
        # ==============================================================================
        # 1. Configuration Section
        # ==============================================================================
        with allure.step("Setup: Configure environment and paths"):
            ERNIE_PROJECT_ROOT = Path.cwd()
            master_ip = "127.0.0.1"
            nnodes = 1
            model_path = os.path.join(
                ERNIE_PROJECT_ROOT, "ERNIE-4.5-21B-A3B-Paddle-dummy-moe"
            )
            task_name = "sft_8k_dummy"

            log_dir = Path(f"log_{task_name}")
            vdl_log_dir = Path(f"{model_path}_{task_name}_vdl")
            output_dir = Path(f"{model_path}_{task_name}_checkpoint")

            env = os.environ.copy()
            vars_to_unset = [
                "PADDLE_TRAINERS_NUM",
                "PADDLE_ELASTIC_JOB_ID",
                "PADDLE_TRAINER_ENDPOINTS",
                "DISTRIBUTED_TRAINER_ENDPOINTS",
                "FLAGS_START_PORT",
                "PADDLE_ELASTIC_TIMEOUT",
            ]
            for var in vars_to_unset:
                env.pop(var, None)

            python_path_parts = [
                str(ERNIE_PROJECT_ROOT.parent),
                str(ERNIE_PROJECT_ROOT),
                env.get("PYTHONPATH", ""),
            ]
            env["PYTHONPATH"] = os.pathsep.join(filter(None, python_path_parts))

            env.update(
                {
                    "FLAGS_set_to_1d": "False",
                    "NVIDIA_TF32_OVERRIDE": "0",
                    "FLAGS_dataloader_use_file_descriptor": "False",
                    "FLAGS_use_stride_kernel": "0",
                    "XPU_PADDLE_L3_SIZE": "0",
                    "XPUAPI_DEFAULT_SIZE": "2205258752",
                    "CUDA_DEVICE_MAX_CONNECTIONS": "8",
                    "CUDA_DEVICE_ORDER": "OAM_ID",
                    "XPULINK_VISIBLE_DEVICES": "2,3,0,1,4,5,6,7",
                    "BKCL_TREE_THRESHOLD": "0",
                    "BKCL_ENABLE_XDR": "1",
                    "BKCL_RDMA_FORCE_TREE": "1",
                    "BKCL_RDMA_NICS": "eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4",
                    "BKCL_SOCKET_IFNAME": "eth0",
                    "BKCL_FORCE_L3_RDMA": "0",
                    "BKCL_USE_AR": "1",
                    "BKCL_RING_OPT": "1",
                    "BKCL_RING_HOSTID_USE_RANK": "1",
                    "XPU_PADDLE_FC_LOCAL_INT16": "1",
                    "XPU_AUTO_BF16_TF32_RADIO": "10",
                    "XPU_AUTO_BF16_TF32": "1",
                }
            )

            allure.attach(env["PYTHONPATH"], name="PYTHONPATH")

        # ==============================================================================
        # 2. Cleanup
        # ==============================================================================
        with allure.step("Cleanup: Remove old log and output directories"):
            if log_dir.exists():
                shutil.rmtree(log_dir)
            if output_dir.exists():
                shutil.rmtree(output_dir)

        # ==============================================================================
        # 3. Command Construction and Execution
        # ==============================================================================
        with allure.step("Execution: Run PaddlePaddle SFT training"):
            training_script_path = (
                ERNIE_PROJECT_ROOT / "examples/post-training/sft/train.py"
            )
            command = [
                sys.executable,
                "-m",
                "paddle.distributed.launch",
                "--log_dir",
                str(log_dir),
                "--xpus",
                "0,1,2,3,4,5,6,7",
                "--master",
                f"{master_ip}:8080",
                "--nnodes",
                str(nnodes),
                str(training_script_path),
                # all arguments
                "--logging_dir",
                str(vdl_log_dir),
                "--model_name_or_path",
                model_path,
                "--output_dir",
                str(output_dir),
                "--per_device_train_batch_size",
                "1",
                "--per_device_eval_batch_size",
                "1",
                "--train_dataset_path",
                str(ERNIE_PROJECT_ROOT / "examples/data/sft-train.jsonl"),
                "--train_dataset_prob",
                "1.0",
                "--train_dataset_type",
                "erniekit",
                "--eval_dataset_path",
                str(ERNIE_PROJECT_ROOT / "examples/data/sft-eval.jsonl"),
                "--eval_dataset_prob",
                "1.0",
                "--eval_dataset_type",
                "erniekit",
                "--max_steps",
                "3",
                "--save_steps",
                "2",
                "--logging_steps",
                "1",
                "--eval_steps",
                "50",
                "--weight_decay",
                "0.01",
                "--do_train",
                "--do_eval",
                "--evaluation_strategy",
                "steps",
                "--tensor_parallel_degree",
                "4",
                "--pipeline_parallel_degree",
                str(nnodes),
                "--sharding_parallel_degree",
                "1",
                "--sharding",
                "stage1",
                "--max_seq_len",
                "8192",
                "--seed",
                "23",
                "--gradient_accumulation_steps",
                "2",
                "--warmup_steps",
                "2000",
                "--lr_scheduler_type",
                "linear",
                "--learning_rate",
                "3e-4",
                "--num_samples_each_epoch",
                "6000000",
                "--bf16",
                "--fp16_opt_level",
                "O2",
                "--amp_custom_white_list",
                "lookup_table",
                "lookup_table_v2",
                "flash_attn",
                "matmul",
                "matmul_v2",
                "fused_gemm_epilogue",
                "--amp_custom_black_list",
                "reduce_sum",
                "softmax_with_cross_entropy",
                "c_softmax_with_cross_entropy",
                "elementwise_div",
                "sin",
                "cos",
                "--disable_tqdm",
                "True",
                "--recompute",
                "1",
                "--offload_optim",
                "1",
                "--recompute_granularity",
                "full",
                "--dataloader_num_workers",
                "1",
                "--distributed_dataloader",
                "1",
                "--use_flash_attention",
                "1",
                "--use_sparse_head_and_loss_fn",
                "0",
                "--use_attn_mask_start_row_indices",
                "1",
                "--pipeline_parallel_config",
                "disable_batch_p2p_comm disable_partial_send_recv enable_clear_every_step_cache",
                "--greedy_intokens",
                "1",
                "--lr_scheduler",
                "linear",
                "--sequence_parallel",
                "1",
                "--release_grads",
                "1",
                "--recompute_use_reentrant",
                "True",
                "--fuse_rope",
                "1",
                "--moe_group",
                "mp",
                "--fuse_rms_norm",
                "False",
                "--device",
                "xpu",
                "--continue_training",
                "0",
                "--moe_multimodal_dispatch_use_allgather",
                "v2-alltoall-unpad",
            ]
            command_str = " ".join(map(str, command))
            allure.attach(
                command_str,
                name="Full Command",
                attachment_type=allure.attachment_type.TEXT,
            )

            process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            stdout_log, stderr_log = [], []
            for line in process.stdout:
                sys.stdout.write(line)
                stdout_log.append(line)
            for line in process.stderr:
                sys.stderr.write(line)
                stderr_log.append(line)
            return_code = process.wait()

            allure.attach(
                "".join(stdout_log),
                name="stdout",
                attachment_type=allure.attachment_type.TEXT,
            )
            allure.attach(
                "".join(stderr_log),
                name="stderr",
                attachment_type=allure.attachment_type.TEXT,
            )
            worker_log_path = log_dir / "workerlog.0"
            if worker_log_path.exists():
                allure.attach.file(
                    str(worker_log_path),
                    name="Worker 0 Log (workerlog.0)",
                    attachment_type=allure.attachment_type.TEXT,
                )

            if return_code != 0:
                pytest.fail(f"Training script failed with exit code {return_code}")

        # ==============================================================================
        # 4. Verification
        # ==============================================================================
        with allure.step("Verification: Check for created checkpoint"):
            expected_checkpoint_dir = output_dir / "checkpoint-2"

            allure.attach(
                f"Verifying existence of checkpoint: {expected_checkpoint_dir}",
                name="Verification Target",
            )

            assert (
                expected_checkpoint_dir.exists()
            ), f"Checkpoint directory '{expected_checkpoint_dir}' was not created."
            assert (
                expected_checkpoint_dir.is_dir()
            ), f"'{expected_checkpoint_dir}' is not a directory."

            expected_state_file = expected_checkpoint_dir / "trainer_state.json"
            assert (
                expected_state_file.exists()
            ), f"Trainer state file '{expected_state_file}' not found in checkpoint."

            expected_peft_index = (
                expected_checkpoint_dir / "model.safetensors.index.json"
            )
            assert (
                expected_peft_index.exists()
            ), f"PEFT model index file '{expected_peft_index}' not found."

            print(
                f"\nSUCCESS: Test completed and checkpoint found at {expected_checkpoint_dir}"
            )
