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
import shutil
import subprocess
import tempfile

import allure
import yaml

OUTPUT_DIR = "./output/"
LOG_DIR = "./erniekit_dist_log/"
MODEL_PATH = "./ERNIE-4.5-0.3B-Paddle-dummy-dense"
CONFIG_PATH = "./examples/configs/ERNIE-4.5-0.3B/"
SFT_CONFIG_PATH = CONFIG_PATH + "sft/"
DPO_CONFIG_PATH = CONFIG_PATH + "dpo/"


def clean_output_dir():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)


def default_args(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_update_config_training(config, steps="train"):
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".yaml", delete=False
    ) as temp_config:
        yaml.dump(config, temp_config)
        temp_config_path = temp_config.name
    cmd = [
        "erniekit",
        steps,
        temp_config_path,
    ]
    if steps == "export":
        cmd.append("lora=True")
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    os.remove(temp_config_path)
    return result.returncode, result.stdout


def assert_result(ret_code, log_output):
    """assert result"""
    if ret_code != 0:
        print("\n".join(log_output.strip().splitlines()[-30:]))
        raise AssertionError("Training Failed")


def attach_log_file():
    log_path = os.path.join(os.getcwd() + "/erniekit_dist_log", "workerlog.0")
    if os.path.exists(log_path):
        allure.attach.file(
            log_path, name="Trainning Log", attachment_type=allure.attachment_type.TEXT
        )
    else:
        allure.attach(
            f"Log file was not generated: {log_path}",
            name="Log Missing",
            attachment_type=allure.attachment_type.TEXT,
        )


def test_sft_default_args():
    clean_output_dir()
    yaml_path = os.path.join(SFT_CONFIG_PATH + "run_sft_8k.yaml")
    config = default_args(yaml_path).copy()
    config["max_steps"] = 3
    config["save_steps"] = 2
    config["model_name_or_path"] = MODEL_PATH
    config["pipeline_parallel_degree"] = 1

    ret_code, err_log = run_update_config_training(config)
    attach_log_file()
    assert_result(ret_code, err_log)


def test_sft_lora_default_args():
    clean_output_dir()
    yaml_path = os.path.join(SFT_CONFIG_PATH + "run_sft_lora_8k.yaml")
    config = default_args(yaml_path).copy()
    config["max_steps"] = 3
    config["save_steps"] = 2
    config["model_name_or_path"] = MODEL_PATH
    config["pipeline_parallel_degree"] = 1

    ret_code, err_log = run_update_config_training(config)
    attach_log_file()
    assert_result(ret_code, err_log)


def test_sft_lora_merge():
    yaml_path = os.path.join(CONFIG_PATH + "run_export.yaml")
    config = default_args(yaml_path).copy()
    config["model_name_or_path"] = MODEL_PATH
    config["pipeline_parallel_degree"] = 1

    ret_code, err_log = run_update_config_training(config)
    attach_log_file()
    assert_result(ret_code, err_log)


def test_dpo_default_args():
    clean_output_dir()
    yaml_path = os.path.join(DPO_CONFIG_PATH + "run_dpo_8k.yaml")
    config = default_args(yaml_path).copy()
    config["max_steps"] = 3
    config["save_steps"] = 2
    config["model_name_or_path"] = MODEL_PATH
    config["pipeline_parallel_degree"] = 1

    ret_code, err_log = run_update_config_training(config)
    attach_log_file()
    assert_result(ret_code, err_log)


def test_dpo_lora_default_args():
    clean_output_dir()
    yaml_path = os.path.join(DPO_CONFIG_PATH + "run_dpo_lora_8k.yaml")
    config = default_args(yaml_path).copy()
    config["max_steps"] = 3
    config["save_steps"] = 2
    config["model_name_or_path"] = MODEL_PATH
    config["pipeline_parallel_degree"] = 1

    ret_code, err_log = run_update_config_training(config)
    attach_log_file()
    assert_result(ret_code, err_log)


# def test_dpo_lora_merge():
#     yaml_path = os.path.join(CONFIG_PATH + "run_export.yaml")
#     config = default_args(yaml_path).copy()
#     config["model_name_or_path"] = MODEL_PATH
#     config["pipeline_parallel_degree"] = 1

#     ret_code,err_log = run_update_config_training(config,steps="export")
#     attach_log_file()
#     assert_result(ret_code, err_log)
