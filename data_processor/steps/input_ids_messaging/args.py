# !/usr/bin/env python3

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

"""
InputIdsMassageArguments
"""

from dataclasses import dataclass, field


@dataclass
class InputIdsMassageArguments:
    """
    args for InputIdsMassageProcessor
    """

    corpus_name: str = field(default=None, metadata={"help": "corpus name"})
    im_prefix_length: int = field(
        default=64, metadata={"help": "number of image placeholder"}
    )

    use_pic_id: bool = field(default=True, metadata={"help": "add Picture Id"})

    prompt_dir: str = field(default="./", metadata={"help": "prompt path"})

    serialize_output: bool = field(default=True, metadata={"help": "serialize output"})
    one_sample_in_one_seq: bool = field(
        default=False, metadata={"help": "one sample in one seq"}
    )
    variable_resolution: bool = field(
        default=False, metadata={"help": "use variable resolution"}
    )
    spatial_conv_size: int = field(
        default=2,
        metadata={"help": "spatial conv size"},
    )
    adaptive_max_imgtoken_option: str = field(
        default=None, metadata={"help": "adaptive max image token"}
    )
    adaptive_max_imgtoken_rate: str = field(
        default=None, metadata={"help": "adaptive max image token rate"}
    )
    max_pixels: int = field(default=None, metadata={"help": "adaptive use max-pixels"})
    min_pixels: int = field(default=None, metadata={"help": "adaptiveuse min-pixels"})
    video_max_pixels: int = field(
        default=None, metadata={"help": "video adaptive use max-pixels"}
    )
    video_min_pixels: int = field(
        default=None, metadata={"help": "video adaptiveuse min-pixels"}
    )
    drop_untrainble_sample: bool = field(
        default=False, metadata={"help": "drop untrainable samples"}
    )
    chat_template: str = field(default="ernie", metadata={"help": "chat template"})

    def __post_init__(self):
        if (
            self.adaptive_max_imgtoken_option is not None
            and self.adaptive_max_imgtoken_rate is not None
        ):
            self.adaptive_max_imgtoken_option = [
                int(op) for op in self.adaptive_max_imgtoken_option.strip().split(",")
            ]
            self.adaptive_max_imgtoken_rate = [
                float(op) for op in self.adaptive_max_imgtoken_rate.strip().split(",")
            ]


@dataclass
class InputIdsMassageInferArguments(InputIdsMassageArguments):
    """
    args for InputIdsMassageProcessor
    """

    data_filelist: str = field(default=None, metadata={"help": "data file list"})
    random_seed: int = field(default=42, metadata={"help": "random seed"})
    vision_model_name_or_path: str = field(
        default=None, metadata={"help": "image preprocess path"}
    )
    rope_3d: bool = field(default=False, metadata={"help": "use 3d rope"})
    max_seq_length: int = field(default=8192, metadata={"help": "max sequence length"})
