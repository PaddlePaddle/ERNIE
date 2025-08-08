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
import requests
import time
import signal
import pytest


import allure
import yaml

OUTPUT_DIR = "./output/"
LOG_DIR = "./erniekit_dist_log/"
MODEL_PATH = "./ERNIE-4.5-21B-A3B-Paddle-dummy-moe"
CONFIG_PATH = "./examples/configs/xpu/ERNIE-4.5-21B-A3B/"
SFT_CONFIG_PATH = CONFIG_PATH + "sft/"
DPO_CONFIG_PATH = CONFIG_PATH + "dpo/"
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


def default_args(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def kill_process_on_port():
    """
    Kill processes that are listening on the given port.
    """
    try:
        result = subprocess.check_output(f"ps -ef | grep {PORT}", shell=True).decode()
        for line in result.strip().split("\n"):
            if "grep" in line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                pid = int(parts[1])
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception as e:
                    print(f"Failed to kill PID {pid}: {e}")
    except subprocess.CalledProcessError:
        pass


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


def run_check_fastdeploy_infer(process_server, process_chat):
    """
    Check fastdeploy inference
    """
    try:
        for _ in range(180):
            try:
                resp = requests.get("http://0.0.0.0:8188/health", timeout=3)
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            with open("./log/workerlog.0", "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-30:]:
                    print(line, end="")
            pytest.fail("FD server health check fail")

        try:
            user_input = "hello\n"
            print(f"\nUser: {user_input.strip()}")
            process_chat.stdin.write(user_input)
            process_chat.stdin.flush()
        except Exception as e:
            pytest.fail(f"Failed to send input to erniekit chat: {e}")

        start_time = time.time()
        chat_output_lines = []

        while True:
            try:
                line = process_chat.stdout.readline()
                if not line:
                    if process_chat.poll() is not None:
                        break
                    continue

                line = line.strip()
                chat_output_lines.append(line)
                print("Assistant:", line)

                if "Assistant:" in line:
                    break

                if time.time() - start_time > 30:
                    print("chat response timeout")
                    break

            except Exception as e:
                print(f"Exception while reading chat output: {e}")
                break

    finally:
        try:
            os.killpg(os.getpgid(process_server.pid), signal.SIGTERM)
        except Exception as e:
            print(f"server shutdown failed: {e}")
        process_server.wait()

        try:
            process_chat.terminate()
        except Exception as e:
            print(f"chat shutdown failed: {e}")
        process_chat.wait()


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
        abs(avg_loss - base_loss) <= 0.01
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
    yaml_path = os.path.join(SFT_CONFIG_PATH, "run_sft_8k.yaml")
    config = default_args(yaml_path).copy()
    config["max_steps"] = 3
    config["save_steps"] = 2
    config["model_name_or_path"] = MODEL_PATH

    ret_code, err_log = run_update_config_training(config)
    attach_log_file()
    assert_result(ret_code, err_log)

    base_loss = 14.33841
    assert_loss(base_loss)


def test_sft_eval():
    yaml_path = os.path.join(CONFIG_PATH, "run_eval.yaml")
    config = default_args(yaml_path).copy()
    config["model_name_or_path"] = "./output/checkpoint-2"

    ret_code, err_log = run_update_config_training(config, steps="eval")
    attach_log_file()
    assert_result(ret_code, err_log)


def test_sft_fd_server():
    kill_process_on_port()
    yaml_path = os.path.join(CONFIG_PATH, "run_chat.yaml")
    config = default_args(yaml_path).copy()
    config["model_name_or_path"] = "./output/checkpoint-2"
    config["max_new_tokens"] = 10

    process_server = run_update_config_training(config, steps="server")
    process_chat = run_update_config_training(config, steps="chat")
    run_check_fastdeploy_infer(process_server, process_chat)


def test_sft_lora():
    clean_output_dir()
    yaml_path = os.path.join(SFT_CONFIG_PATH, "run_sft_lora_8k.yaml")
    config = default_args(yaml_path).copy()
    config["max_steps"] = 3
    config["save_steps"] = 2
    config["model_name_or_path"] = MODEL_PATH

    ret_code, err_log = run_update_config_training(config)
    attach_log_file()
    assert_result(ret_code, err_log)

    base_loss = 14.333244
    assert_loss(base_loss)


def test_sft_lora_merge():
    yaml_path = os.path.join(CONFIG_PATH, "run_export.yaml")
    config = default_args(yaml_path).copy()
    config["model_name_or_path"] = MODEL_PATH

    ret_code, err_log = run_update_config_training(config, steps="export")
    attach_log_file()
    assert_result(ret_code, err_log)


def test_sft_lora_eval():
    yaml_path = os.path.join(CONFIG_PATH, "run_eval.yaml")
    config = default_args(yaml_path).copy()
    config["model_name_or_path"] = "./output/export"

    ret_code, err_log = run_update_config_training(config, steps="eval")
    attach_log_file()
    assert_result(ret_code, err_log)


def test_sft_lora_fd_server():
    kill_process_on_port()
    yaml_path = os.path.join(CONFIG_PATH, "run_chat.yaml")
    config = default_args(yaml_path).copy()
    config["model_name_or_path"] = "./output/export"
    config["max_new_tokens"] = 10

    process_server = run_update_config_training(config, steps="server")
    process_chat = run_update_config_training(config, steps="chat")
    run_check_fastdeploy_infer(process_server, process_chat)


# The following annotated configurations are currently not supported.
# def test_sft_wint8mix_lora():
#     clean_output_dir()
#     yaml_path = os.path.join(SFT_CONFIG_PATH, "run_sft_wint8mix_lora_8k.yaml")
#     config = default_args(yaml_path).copy()
#     config["max_steps"] = 3
#     config["save_steps"] = 2
#     config["model_name_or_path"] = MODEL_PATH

#     ret_code, err_log = run_update_config_training(config)
#     attach_log_file()
#     assert_result(ret_code, err_log)

#     base_loss = 14.316673
#     assert_loss(base_loss)


# def test_sft_wint8mix_lora_merge():
#     yaml_path = os.path.join(CONFIG_PATH, "run_export.yaml")
#     config = default_args(yaml_path).copy()
#     config["model_name_or_path"] = MODEL_PATH

#     ret_code, err_log = run_update_config_training(config, steps="export")
#     attach_log_file()
#     assert_result(ret_code, err_log)


# def test_sft_wint8mix_lora_fd_server():
#     kill_process_on_port()
#     yaml_path = os.path.join(CONFIG_PATH, "run_chat.yaml")
#     config = default_args(yaml_path).copy()
#     config["model_name_or_path"] = "./output/export"
#     config["max_new_tokens"] = 10

#     process_server = run_update_config_training(config, steps="server")
#     process_chat = run_update_config_training(config, steps="chat")
#     run_check_fastdeploy_infer(process_server, process_chat)


# def test_dpo():
#     clean_output_dir()
#     yaml_path = os.path.join(DPO_CONFIG_PATH, "run_dpo_8k.yaml")
#     config = default_args(yaml_path).copy()
#     config["max_steps"] = 3
#     config["save_steps"] = 2
#     config["model_name_or_path"] = MODEL_PATH

#     ret_code, err_log = run_update_config_training(config)
#     attach_log_file()
#     assert_result(ret_code, err_log)

#     base_loss = 0.690705
#     assert_loss(base_loss)


# def test_dpo_eval():
#     yaml_path = os.path.join(CONFIG_PATH, "run_eval.yaml")
#     config = default_args(yaml_path).copy()
#     config["model_name_or_path"] = "./output/checkpoint-3"

#     ret_code, err_log = run_update_config_training(config, steps="eval")
#     attach_log_file()
#     assert_result(ret_code, err_log)


# def test_dpo_fd_server():
#     kill_process_on_port()
#     yaml_path = os.path.join(CONFIG_PATH, "run_chat.yaml")
#     config = default_args(yaml_path).copy()
#     config["model_name_or_path"] = "./output/checkpoint-3"
#     config["max_new_tokens"] = 10

#     process_server = run_update_config_training(config, steps="server")
#     process_chat = run_update_config_training(config, steps="chat")
#     run_check_fastdeploy_infer(process_server, process_chat)


# def test_dpo_lora():
#     clean_output_dir()
#     yaml_path = os.path.join(DPO_CONFIG_PATH, "run_dpo_lora_8k.yaml")
#     config = default_args(yaml_path).copy()
#     config["max_steps"] = 3
#     config["save_steps"] = 2
#     config["model_name_or_path"] = MODEL_PATH

#     ret_code, err_log = run_update_config_training(config)
#     attach_log_file()
#     assert_result(ret_code, err_log)

#     base_loss = 0.692727
#     assert_loss(base_loss)


# def test_dpo_lora_merge():
#     yaml_path = os.path.join(CONFIG_PATH, "run_export.yaml")
#     config = default_args(yaml_path).copy()
#     config["model_name_or_path"] = MODEL_PATH

#     ret_code, err_log = run_update_config_training(config, steps="export")
#     attach_log_file()
#     assert_result(ret_code, err_log)


# def test_dpo_lora_fd_server():
#     kill_process_on_port()
#     yaml_path = os.path.join(CONFIG_PATH, "run_chat.yaml")
#     config = default_args(yaml_path).copy()
#     config["model_name_or_path"] = "./output/export"
#     config["max_new_tokens"] = 10

#     process_server = run_update_config_training(config, steps="server")
#     process_chat = run_update_config_training(config, steps="chat")
#     run_check_fastdeploy_infer(process_server, process_chat)


# def test_dpo_wint8mix_lora():
#     clean_output_dir()
#     yaml_path = os.path.join(DPO_CONFIG_PATH, "run_dpo_wint8mix_lora_8k.yaml")
#     config = default_args(yaml_path).copy()
#     config["max_steps"] = 3
#     config["save_steps"] = 2
#     config["model_name_or_path"] = MODEL_PATH

#     ret_code, err_log = run_update_config_training(config)
#     attach_log_file()
#     assert_result(ret_code, err_log)

#     base_loss = 0.693656
#     assert_loss(base_loss)


# def test_dpo_wint8mix_lora_merge():
#     yaml_path = os.path.join(CONFIG_PATH, "run_export.yaml")
#     config = default_args(yaml_path).copy()
#     config["model_name_or_path"] = MODEL_PATH

#     ret_code, err_log = run_update_config_training(config, steps="export")
#     attach_log_file()
#     assert_result(ret_code, err_log)


# def test_dpo_wint8mix_lora_fd_server():
#     kill_process_on_port()
#     yaml_path = os.path.join(CONFIG_PATH, "run_chat.yaml")
#     config = default_args(yaml_path).copy()
#     config["model_name_or_path"] = "./output/export"
#     config["max_new_tokens"] = 10

#     process_server = run_update_config_training(config, steps="server")
#     process_chat = run_update_config_training(config, steps="chat")
#     run_check_fastdeploy_infer(process_server, process_chat)
