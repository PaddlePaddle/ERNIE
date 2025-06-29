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
    data_filelist: str = field(default=None, metadata={"help": "data file list"})
    im_prefix_length: int = field(default=64, metadata={"help": "number of image placeholder"})
    max_seq_length: int = field(default=8192, metadata={"help": "max sequence length"})

    crop_tile_option: str = field(default="9,16", metadata={"help": "crop option"})
    crop_tile_rate: str = field(default="0.8,0.2", metadata={"help": "crop rate"})

    crop_width: int = field(default=448, metadata={"help": "crop width"})
    crop_height: int = field(default=448, metadata={"help": "crop height"})
    video_fix_crop: int = field(default=-1, metadata={"help": "vieo fix crop"})

    use_loc_specialtoken: bool = field(default=False, metadata={"help": "location use special token"})
    use_crop_specialtoken: bool = field(default=False, metadata={"help": "crop use special token"})
    use_pic_id: bool = field(default=True, metadata={"help": "add Picture Id"})

    augment_conf: str = field(default="conf/config_transform.yaml", metadata={"help": "image augment conf path"})
    prompt_dir: str = field(default="./", metadata={"help": "prompt path"})

    random_seed: int = field(default=42, metadata={"help": "random seed"})
    serialize_output: bool = field(default=True, metadata={"help": "serialize output"})
    min_crop_flag: bool = field(default=False, metadata={"help": "set 896 * 896"})
    one_sample_in_one_seq: bool = field(default=False, metadata={"help": "one sample in one seq"})
    video_max_tile: int = field(default=-1, metadata={"help": "video_max_tile"})
    use_multi_image_crop: bool = field(default=False, metadata={"help": "use multi image crop"})
    use_smart_resize: bool = field(default=False, metadata={"help": "use smart_resize"})
    min_tile: int = field(default=4, metadata={"help": "min tile"})
    variable_resolution: bool = field(default=False, metadata={"help": "use variable resolution"})
    spatial_conv_size: int = field(
        default=2,
        metadata={"help": "spatial conv size"},
    )
    vision_model_name_or_path: str = field(default=None, metadata={"help": "image preprocess path"})
    adaptive_max_imgtoken_option: str = field(default=None, metadata={"help": "adaptive max image token"})
    adaptive_max_imgtoken_rate: str = field(default=None, metadata={"help": "adaptive max image token rate"})
    max_pixels: int = field(default=None, metadata={"help": "adaptive use max-pixels"})
    min_pixels: int = field(default=None, metadata={"help": "adaptiveuse min-pixels"})
    video_max_pixels: int = field(default=None, metadata={"help": "video adaptive use max-pixels"})
    video_min_pixels: int = field(default=None, metadata={"help": "video adaptiveuse min-pixels"})
    rope_3d: bool = field(default=False, metadata={"help": "use 3d rope"})
    drop_untrainble_sample: bool = field(default=False, metadata={"help": "drop untrainable samples"})
    chat_template: str = field(default="ernie", metadata={"help": "chat template"})

    def __post_init__(self):
        self.crop_tile_option = [int(op) for op in self.crop_tile_option.strip().split(",")]
        self.crop_tile_rate = [float(op) for op in self.crop_tile_rate.strip().split(",")]

        if self.adaptive_max_imgtoken_option is not None and self.adaptive_max_imgtoken_rate is not None:
            self.adaptive_max_imgtoken_option = [
                int(op) for op in self.adaptive_max_imgtoken_option.strip().split(",")
            ]
            self.adaptive_max_imgtoken_rate = [float(op) for op in self.adaptive_max_imgtoken_rate.strip().split(",")]

        assert self.video_fix_crop in [-1, 2], "video_fix_crop only supported values are -1 and 2"
        if self.video_fix_crop != -1:
            self.video_max_tile = self.video_fix_crop
