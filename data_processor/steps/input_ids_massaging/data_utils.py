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
utils for data processor
"""

import base64
import math
from io import BytesIO

import xxhash
from PIL import Image

from data_processor.utils.logger_utils import logger


def get_text_token_num(tokenizer, text: str):
    """text tokenize and count"""
    return len(tokenizer.encode(text)["input_ids"])


def get_uniq_id(text):
    """text hash"""
    return xxhash.xxh32_intdigest(text)


def image_to_json_serializable(image):
    """
    Convert an image into a JSON-serializable format.

    Parameters:
    image (PIL.Image or bytes or str): The input image, which can be a PIL Image object, raw bytes,
    or a file path as a string.
    Returns:
    The image data encoded in Base64, or a string (if the input is already a valid path or string representation).
    """
    if isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        output = base64.b64encode(buffered.getvalue()).decode('utf-8')
    elif isinstance(image, bytes):
        output = base64.b64encode(image).decode('utf-8')
    elif isinstance(image, str):
        output = image
    else:
        raise ValueError(f"Unsupported image type {type(image)}.")
    return output


def add_prompt(sample: dict, use_prompt: bool, prompt: str, data_type: str):
    """
    All images come before the text, prompt:
    <imgs, text> -> <imgs, prompt, text>
    For interleaved image-text data, prompt:
    <imgs, texts, imgs, texts> -> <prompt, imgs, texts, imgs, texts>

    Args:
        sample (dict): one sample
        use_prompt (bool): use prompt or not
        prompt (str): prompt text
        data_type (str): data type

    Returns:
        dict: sample with prompt
    """
    if use_prompt and len(prompt) > 0:
        sample["text_info"].insert(0, {"text": prompt, "tag": "mask"})
        if not all(one["matched_text_index"] == 0 for one in sample["image_info"]):
            for image_info in sample["image_info"]:
                image_info["matched_text_index"] += 1
    return sample


def merge_list(lists):
    """merge multi list to one list

    Args:
        lists (list[list]): [[], [], ...]

    Returns:
        list: one list
    """
    new_list = lists[0]
    for one in lists[1:]:
        new_list.extend(one)
    return new_list


def round_by_factor(number: int, factor: int):
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int):
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int, max_ratio: int):
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > max_ratio:
        logger.info(
            f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}"
        )
        return height, width
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar
