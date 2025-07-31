# -*- coding: utf-8 -*-
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

import io
import logging
from PIL import Image
from PIL.ExifTags import TAGS
from .bos_client import BosClient

logger = logging.getLogger(__name__)

__all__ = (
    "bos",
    "bos_path_to_pil_image",
    "MMSpecialTokensConfig",
    "DATATYPE_2_ID",
    "IDTYPES_2_ID",
    "IMAGETYPES_2_ID",
)

try:
    bos = BosClient()
except Exception as e:
    bos = None
    logger.error(f"bos client init error: {e}")

DATATYPE_2_ID = {"mm": 0, "lm": 1, "audio": 2}
IDTYPES_2_ID = {"text": 0, "image": 1, "video": 2, "audio": 3}
IMAGETYPES_2_ID = {"image": 0, "video": 1, "padded_image": 2}


def bos_path_to_pil_image(
    bos_path, exist_flag=False, need_exif_info=True, retry_max_time=0, retry_interval=1
):
    """bos_path_to_pil_image"""
    if bos is None:
        raise ValueError("bos client not init")

    def get_image_exif(image):
        exif_data = image._getexif()
        exif_info = {}
        if exif_data is not None:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                exif_info[tag_name] = value.strip()
        return exif_info

    def has_transparent_background(img):
        """判断图片是否有背景"""
        if img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        ):
            # Check for any pixel with alpha channel less than 255 (fully opaque)
            alpha = img.convert("RGBA").split()[-1]
            if alpha.getextrema()[0] < 255:
                return True
        return False

    def add_white_background(img):
        """
        给透明背景的图，加个白色背景
        """
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        # 创建一个白色背景的图像，尺寸与原图一致
        img_white_background = Image.new("RGBA", img.size, (255, 255, 255))

        # 将原图粘贴到白色背景上
        img_white_background.paste(img, (0, 0), img)

        return img_white_background

    def change_I16_to_L(img):
        """
        将图片从I;16模式转换为L模式
        """
        # 由于I模式的point函数只支持加减乘，所以下面的* (1 / 256)不能改成除法
        return img.point(lambda i: i * (1 / 256)).convert("L")

    if isinstance(bos_path, str):
        if exist_flag:
            if not bos.exists(bos_path):
                raise OSError(f"bos file: {bos_path} not exist")
        pil_image = Image.open(
            io.BytesIO(
                bos.get_bytes(
                    bos_path, retry=retry_max_time, retry_interval=retry_interval
                )
            )
        )
        if need_exif_info:
            try:
                exif_info = get_image_exif(pil_image)
            except Exception:
                # logger.info(f"exif_info error for {bos_path}")
                exif_info = {}
        else:
            exif_info = {}

        try:
            if pil_image.mode == "I;16":
                pil_image = change_I16_to_L(pil_image)
            if has_transparent_background(pil_image):
                pil_image = add_white_background(pil_image)
        except Exception:
            pass
            # raise ValueError(f"fail to add white background, bos_path={bos_path}, Exception: {e}")

        return pil_image.convert("RGB"), exif_info
    else:
        return bos_path, None


class MMSpecialTokensConfig:
    """_summary_"""

    use_ocr_specialtoken = True
    use_crop_specialtoken = True
    coor_num = 1001
    image_placeholder = "<|IMAGE_PLACEHOLDER|>"
    audio_placeholder = "<|AUDIO_PLACEHOLDER|>"
    crop = ["<|CROP_COL_SEP|>", "<|CROP_ROW_SEP|>", "<|IMAGE_SEP|>"]
    ocr_coor = [f"<|LOC_{i}|>" for i in range(coor_num)]
    ocr_begin_end = ["<|LOC_BEGIN|>", "<|LOC_END|>", "<|LOC_SEP|>"]
    mm_begin_end = ["<|BOI|>", "<|EOI|>", "<|BOA|>", "<|EOA|>", "<|BOV|>", "<|EOV|>"]

    @classmethod
    def get_special_tokens_info(cls):
        """_summary_

        Returns:
            _type_: _description_
        """
        return {
            k: getattr(cls, k)
            for k in [
                "image_placeholder",
                "audio_placeholder",
                "crop",
                "ocr_coor",
                "ocr_begin_end",
                "mm_begin_end",
            ]
        }
