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
Ernie4_5_VLTokenizer
"""
from .tokenizer import Ernie4_5_Tokenizer

coor_num = 1001
NOT_FOUND_TOKEN_ID = -101
NUM_IMAGE_SPECIAL_TOKEN = 2048
NUM_AUDIO_SPECIAL_TOKEN = 1024
SFT_IMAGE_START_TOKEN = "<|IMAGE_START|>"
SFT_IMAGE_END_TOKEN = "<|IMAGE_END|>"
SFT_VIDEO_START_TOKEN = "<|VIDEO_START|>"
SFT_VIDEO_END_TOKEN = "<|VIDEO_END|>"
SFT_ASR_START_TOKEN = "<|ASR_START|>"
SFT_ASR_END_TOKEN = "<|ASR_END|>"

special_tokens_info = {
    "image_placeholder": "<|IMAGE_PLACEHOLDER|>",
    "crop": ["<|CROP_COL_SEP|>", "<|CROP_ROW_SEP|>", "<|IMAGE_SEP|>"],
    "loc_coor": [f"<|LOC_{i}|>" for i in range(coor_num)],
    "loc_begin_end": ["<|LOC_BEGIN|>", "<|LOC_END|>", "<|LOC_SEP|>"],
    "image_begin_end": ["<|BOI|>", "<|EOI|>"],
    "video_begin_end": ["<|BOV|>", "<|EOV|>"],
    "sft_video_begin_end": [SFT_VIDEO_START_TOKEN, SFT_VIDEO_END_TOKEN],
}


class Ernie4_5_VLTokenizer(Ernie4_5_Tokenizer):
    """Ernie4_5_VLTokenizer"""

    @property
    def space_token(self):
        """Return the space token"""
        return "<mask:1>"

    @property
    def space_token_id(self):
        """Return the ID of the space token"""
        return self.sp_model.piece_to_id("<mask:1>")

    @property
    def gend_token(self):
        """Return the gender token"""
        return "<mask:7>"

    @property
    def gend_token_id(self):
        """Return the ID of the gender token"""
        return self.sp_model.piece_to_id("<mask:7>")

    @property
    def im_start_id(self):
        """Return the ID of the image start token"""
        return self.sp_model.piece_to_id("<|im_start|>")

    @property
    def im_end_id(self):
        """Return the ID of the image end token"""
        return self.sp_model.piece_to_id("<|im_end|>")
