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
import json
from paddleformers.utils.download import (
    DownloadSource,
    register_model_group,
    check_repo,
)

register_model_group(
    models={
        "ERNIE-4.5-300B-A47B-Base": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-300B-A47B-Base-Paddle",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-300B-A47B-Base-Paddle",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-300B-A47B-Base-Paddle",
        },
        "ERNIE-4.5-300B-A47B": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-300B-A47B-Paddle",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle",
        },
        "ERNIE-4.5-21B-A3B-Base": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-21B-A3B-Base-Paddle",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-21B-A3B-Base-Paddle",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-21B-A3B-Base-Paddle",
        },
        "ERNIE-4.5-21B-A3B": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-21B-A3B-Paddle",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle",
        },
        "ERNIE-4.5-0.3B-Base": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-0.3B-Base-Paddle",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-0.3B-Base-Paddle",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-0.3B-Base-Paddle",
        },
        "ERNIE-4.5-0.3B": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-0.3B-Paddle",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-0.3B-Paddle",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-0.3B-Paddle",
        },
    }
)


register_model_group(
    models={
        "ERNIE-4.5-VL-424B-A47B-Base": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-VL-424B-A47B-Base-Paddle",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-VL-424B-A47B-Base-Paddle",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-VL-424B-A47B-Base-Paddle",
        },
        "ERNIE-4.5-VL-424B": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-VL-424B-Paddle",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-VL-424B-Paddle",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-VL-424B-Paddle",
        },
        "ERNIE-4.5-VL-28B-A3B-Base": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-VL-28B-A3B-Base-Paddle",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Base-Paddle",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Base-Paddle",
        },
        "ERNIE-4.5-VL-28B-A3B": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-VL-28B-A3B-Paddle",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Paddle",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Paddle",
        },
    }
)


def check_download_repo(
    model_name_or_path, download_hub: DownloadSource = DownloadSource.DEFAULT
):
    # Detect torch model.
    is_local = os.path.isfile(model_name_or_path) or os.path.isdir(model_name_or_path)
    if is_local:
        config_path = os.path.join(model_name_or_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        if "torch_dtype" in config_dict:
            raise ValueError(
                "Unsupported weight format: Torch weights are not compatible with Paddle model currently."
            )
    else:
        # check remote repo
        model_name_or_path = check_repo(model_name_or_path, download_hub)

    return model_name_or_path
