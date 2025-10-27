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
import signal
import tarfile
import urllib.request


import allure
import yaml

OUTPUT_DIR = "./output/"
LOG_DIR = "./erniekit_dist_log/"
CONFIG_PATH = "./examples/configs/iluvatar_gpu/ERNIE-4.5-VL-28B-A3B/"
SFT_CONFIG_PATH = CONFIG_PATH + "sft/"
PORT = 8188

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
    tar_path = os.path.join(data_dir, "DoclingMatix.tar.gz")
    url = "https://paddleformers.bj.bcebos.com/datasets/DoclingMatix.tar.gz"
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(tar_path):
        print("Downloading DoclingMatix.tar.gz...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download completed.")
        print("Extracting files...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Extraction completed.")
    else:
        print("DoclingMatix datasets already exists.")


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


def test_sft_lora():
    clean_output_dir()
    yaml_path = os.path.join(SFT_CONFIG_PATH, "run_sft_lora_8k.yaml")
    config = default_args(yaml_path).copy()
    config["max_steps"] = 3
    config["save_steps"] = 2

    ret_code, err_log = run_update_config_training(config)
    attach_log_file()
    assert_result(ret_code, err_log)

    base_loss = 0.318546
    assert_loss(base_loss)


def test_sft_lora_merge():
    yaml_path = os.path.join(CONFIG_PATH, "run_export.yaml")
    config = default_args(yaml_path).copy()

    ret_code, err_log = run_update_config_training(config, steps="export")
    attach_log_file()
    assert_result(ret_code, err_log)


def run_check_fastdeploy_infer(config):
    import io
    from PIL import Image
    from fastdeploy.entrypoints.llm import LLM
    from fastdeploy.engine.sampling_params import SamplingParams
    from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer
    from fastdeploy.utils import set_random_seed

    set_random_seed(123)
    tokenizer = Ernie4_5Tokenizer.from_pretrained(config["model_name_or_path"])

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"
                    },
                },
                {"type": "text", "text": "图中的文物属于哪个年代"},
            ],
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    images, videos = [], []
    for message in messages:
        content = message["content"]
        if not isinstance(content, list):
            continue
        for part in content:
            if part["type"] == "image_url":
                url = part["image_url"]["url"]
                image_bytes = requests.get(url).content
                img = Image.open(io.BytesIO(image_bytes))
                images.append(img)
            elif part["type"] == "video_url":
                url = part["video_url"]["url"]
                video_bytes = requests.get(url).content
                videos.append({"video": video_bytes, "max_frames": 30})

    sampling_params = SamplingParams(
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_new_tokens"],
    )
    llm = LLM(
        model=config["model_name_or_path"],
        tensor_parallel_size=config["tensor_parallel_degree"],
        max_model_len=config["max_model_len"],
        block_size=config["block_size"],
        quantization=config["quantization"],
        limit_mm_per_prompt=config["limit_mm_per_prompt"],
        reasoning_parser=config["reasoning_parser"],
    )
    outputs = llm.generate(
        prompts={
            "prompt": prompt,
            "multimodal_data": {"image": images, "video": videos},
        },
        sampling_params=sampling_params,
    )
    assert outputs[0].outputs.token_ids == [
        23,
        3843,
        94206,
        2075,
        52352,
        95532,
        94467,
        100282,
        23,
        2,
    ], f"{outputs[0].outputs.token_ids}"


def test_sft_lora_fd():
    kill_process_on_port()
    yaml_path = os.path.join(CONFIG_PATH, "run_chat.yaml")
    config = default_args(yaml_path).copy()
    config["model_name_or_path"] = "./output/export"
    config["max_new_tokens"] = 10
    run_check_fastdeploy_infer(config)


def test_sft_lora_e2e():
    print("start running test_sft_lora, check log in erniekit_dist_log/workerlog.0")
    test_sft_lora()
    print(
        "start running test_sft_lora_merge, check log in erniekit_dist_log/workerlog.0"
    )
    test_sft_lora_merge()
    print("start running test_sft_lora_fd, check log in log/workerlog.0")
    test_sft_lora_fd()


if __name__ == "__main__":
    test_sft_lora_e2e()
