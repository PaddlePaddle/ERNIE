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
import re
import shutil
import subprocess
import tempfile
import urllib.request


import allure
import yaml

OUTPUT_DIR = "./output/"
LOG_DIR = "./erniekit_dist_log/"
MODEL_PATH = "./PaddleOCR-VL/"
CONFIG_PATH = "./examples/configs/PaddleOCR-VL/"
SFT_CONFIG_PATH = CONFIG_PATH + "sft/"
PORT = 8188

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["NCCL_ALGO"] = "Tree"
os.environ["FLAGS_embedding_deterministic"] = "1"
os.environ["FLAGS_cudnn_deterministic"] = "1"


def clean_output_dir():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)


def prepare_data():
    data_dir = "examples/data"
    os.makedirs(data_dir, exist_ok=True)

    files = {
        "ocr_vl_sft-train_Bengali.jsonl": "https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-train_Bengali.jsonl",
        "ocr_vl_sft-test_Bengali.jsonl": "https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-test_Bengali.jsonl",
    }

    for filename, url in files.items():
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            print(f"Downloading {filename} ...")
            try:
                urllib.request.urlretrieve(url, file_path)
                print(f"Saved to {file_path}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
        else:
            print(f"{filename} already exists, skip downloading.")


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

    if steps == "server":
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        return process
    elif steps == "chat":
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return process
    else:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        return result.returncode, result.stdout


def assert_result(ret_code, log_output):
    """assert result"""
    if ret_code != 0:
        print("\n".join(log_output.strip().splitlines()[-30:]))
        raise AssertionError("Training Failed")


def assert_loss(base_loss):
    """
    Calculate the average loss from the log file, and compare it with the expected value.
    """
    log_path = os.path.join(os.getcwd(), "erniekit_dist_log", "workerlog.0")
    loss_pattern = re.compile(r"- loss:\s*([0-9]+\.[0-9]+)")
    with open(log_path, encoding="utf-8") as f:
        content = f.read()
    losses = [float(m.group(1)) for m in loss_pattern.finditer(content)]

    if losses:
        sum_loss = sum(losses) / len(losses)
        avg_loss = round(sum_loss, 6)
    else:
        avg_loss = 0

    assert (
        abs(avg_loss - base_loss) <= 0.0001
    ), f"loss: {avg_loss}, base_loss: {base_loss}, exist diff!"


def attach_log_file():
    log_path = os.path.join(os.getcwd(), "erniekit_dist_log", "workerlog.0")
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


def test_sft():
    clean_output_dir()
    prepare_data()
    yaml_path = os.path.join(SFT_CONFIG_PATH, "run_ocr_vl_sft_16k.yaml")
    config = default_args(yaml_path).copy()
    config["max_steps"] = 3
    config["save_steps"] = 2
    config["model_name_or_path"] = MODEL_PATH
    config["output_dir"] = OUTPUT_DIR

    ret_code, err_log = run_update_config_training(config)
    attach_log_file()
    assert_result(ret_code, err_log)

    base_loss = 14.339227
    assert_loss(base_loss)
