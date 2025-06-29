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

"""crop image"""
import math
from functools import partial

import numpy as np

from data_processor.steps.input_ids_massaging.data_crop.crop import Crop

# sys.path.append("..")
from data_processor.steps.input_ids_massaging.data_utils import get_text_token_num, smart_resize
from data_processor.utils.constant import GIVEN_MAX_TILE, MAX_RATIO, MIN_CROP_SUPPORT_RANGE
from data_processor.utils.logger_utils import logger


class CropImage(Crop):
    """CropImage"""

    def __init__(
        self,
        tokenizer,
        im_prefix_length,
        crop_width: int = 448,
        crop_height: int = 448,
        crop_tile_option=None,
        crop_tile_rate=None,
        use_crop_specialtoken: bool = False,
        special_tokens_info=None,
        min_crop_flag: bool = False,
        is_pretraining: bool = False,
        return_partition_info: bool = False,
        use_smart_resize: bool = False,
        min_tile: int = 4,
        **kwargs,
    ):
        """
        crop info

        Args:
            tokenizer: tokenizer.
            im_prefix_length: #tokens in LLM for each image tile.
            crop_width (int, optional): width of one crop image. Defaults to 448.
            crop_height (int, optional): height of one crop image. Defaults to 448.
            crop_tile_option (list, optional): multi max tiles. Defaults to [9, 16].
            crop_tile_rate (list, optional): selection probability of multi max tiles. Defaults to [0.95, 0.05].
            use_crop_specialtoken (bool, optional): use special token for crop position text or not.
        """
        if crop_tile_option is None:
            crop_tile_option = [9, 16]
        if crop_tile_rate is None:
            crop_tile_rate = [0.95, 0.05]
        if special_tokens_info is None:
            special_tokens_info = {}

        self.im_prefix_length = im_prefix_length
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.crop_tile_option = crop_tile_option
        self.crop_tile_rate = crop_tile_rate
        self.use_crop_specialtoken = use_crop_specialtoken
        self.is_pretraining = is_pretraining
        self.min_crop_flag = min_crop_flag
        self.return_partition_info = return_partition_info
        self.use_smart_resize = use_smart_resize
        self.min_tile = min_tile
        if use_crop_specialtoken:
            self.crop_col_sep, self.crop_row_sep, self.crop_image_sep = special_tokens_info["crop"]
        else:
            self.crop_col_sep, self.crop_row_sep, self.crop_image_sep = (
                ",",
                "\n",
                "\n\n",
            )
        self.crop_col_sep_token_num, self.crop_row_sep_token_num, self.crop_image_sep_token_num = map(
            partial(get_text_token_num, tokenizer), [self.crop_col_sep, self.crop_row_sep, self.crop_image_sep]
        )
        self.crop_tag = "no_mask" if use_crop_specialtoken and self.is_pretraining else "mask"

    def process(self, sample: dict):
        """crop image of sample

        Args:
            sample (dict): one raw sample.

        Returns:
            dict: processed sample
        """

        return self._adaptive_partition_with_position(sample)

    def _get_rounded(self, width: int, height: int, patch_size: int = 448, all_resize: bool = False):
        """
        get rounded size according to patch_size
        """

        def get_rounded_helper(length, patch_size):
            ret = int(round(length / patch_size, 0))
            ret = max(1, ret)
            return ret * patch_size

        if width < height:
            width_rounded = get_rounded_helper(width, patch_size)
            height_rounded = round(height * (width_rounded / width), 0)
            if all_resize:
                height_rounded = get_rounded_helper(height_rounded, patch_size)
        else:
            height_rounded = get_rounded_helper(height, patch_size)
            width_rounded = round(width * (height_rounded / height), 0)
            if all_resize:
                width_rounded = get_rounded_helper(width_rounded, patch_size)

        return width_rounded, height_rounded

    def _adaptive_partition(self, img_width: int, img_height: int, max_tile: int = 15):
        """
        adaptive partition for one image
        """

        select_col = 0
        select_row = 0
        select_score = -float("inf")

        for n in range(1, max_tile + 1):
            for m in range(1, max_tile + 1):
                # range [max_tile-2, max_tile]
                if n * m == max_tile or n * m == max_tile - 1 or n * m == max_tile - 2:
                    score = -abs(
                        math.log((img_width * n) / (img_height * m)) - math.log(self.crop_width / self.crop_height)
                    )

                    if score > select_score:
                        select_score = score
                        select_row = n
                        select_col = m

        return select_row, select_col

    def _get_crop_info(self, width: int, height: int, split_row_col=None):
        """get crop info

        img (2051, 3579)
        split_row_col = [4, 2] for crop 4 row in height & crop 2 col in weight
        """
        if split_row_col is None:
            split_row_col = [5, 3]

        num_height, num_width = split_row_col

        part_width = width / num_width
        part_height = height / num_height

        parts = []
        for i in range(num_height):
            one = []
            for j in range(num_width):
                left = round(j * part_width, 2)
                top = round(i * part_height, 2)
                right = round(left + part_width, 2)
                bottom = round(top + part_height, 2)

                # pillow crop
                one.append([left, top, right, bottom])
            parts.append(one)

        return parts

    def process_by_element(self, img_one):
        """
        process_by_element
        """
        return self._adaptive_partition_by_element(img_one)

    def _adaptive_partition_by_element(self, img_one, min_tile=None, max_tile=None):
        """
        adaptive partition for one image
        """
        return_sequence = []  # this is a return sequence for crop

        # [STEP 1] need to crop
        img_width, img_height = img_one["image_width"], img_one["image_height"]
        if max_tile is not None:
            raw_max_tile = max_tile
        else:
            raw_max_tile = np.random.choice(self.crop_tile_option, p=self.crop_tile_rate)

        if self.use_smart_resize:
            max_tile = raw_max_tile
            if min_tile is None:
                min_tile = self.min_tile if self.min_crop_flag else 1

            if max_tile < min_tile:
                min_tile = max_tile
                logger.info(f"set to min_tile({min_tile}) to than max_tile({max_tile})")
            assert self.crop_width == self.crop_height, "crop_width and crop_height should be the same."
            image_factor = self.crop_width
            min_pixels = min_tile * self.crop_width * self.crop_height
            max_pixels = GIVEN_MAX_TILE * self.crop_width * self.crop_height
            img_width, img_height = smart_resize(
                img_width,
                img_height,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                max_ratio=MAX_RATIO,
            )
            if img_width == 0 or img_height == 0:
                img_width = img_one["image_width"]
                img_height = img_one["img_height"]
        else:
            max_tile = raw_max_tile
            if (img_width < self.crop_width) ^ (img_height < self.crop_height):
                img_width, img_height = self._get_rounded(img_width, img_height, self.crop_width, all_resize=False)

        # number of tile
        original_tile = math.ceil(img_width * img_height / (self.crop_width * self.crop_height))
        max_tile = min(original_tile, max_tile)

        if max_tile > 1:
            row, col = self._adaptive_partition(img_width, img_height, max_tile=max_tile)
        else:
            # no need crop
            row, col = 1, 1
        # min_crop
        min_tile = self.min_tile if min_tile is None else min_tile
        if row * col < min_tile and self.min_crop_flag and not self.use_smart_resize:
            assert min_tile in MIN_CROP_SUPPORT_RANGE, f"We only support min_tile = {MIN_CROP_SUPPORT_RANGE}."
            img_width, img_height = self.crop_width * 2, self.crop_height * 2

            max_tile = raw_max_tile
            original_tile = math.ceil(img_width * img_height / (self.crop_width * self.crop_height))
            max_tile = min(original_tile, max_tile)

            row, col = self._adaptive_partition(img_width, img_height, max_tile=max_tile)

        """[STEP 2]"""
        if row * col > 1:
            # start crop
            parts = self._get_crop_info(img_width, img_height, split_row_col=[row, col])
            # list[list]
            for rid, rows in enumerate(parts):
                for cid, crop_info in enumerate(rows):
                    # crop_info = [left, top, right, bottom] of crop image
                    left, top, right, bottom = crop_info
                    crop_img_width = right - left
                    crop_img_height = bottom - top

                    local_one = {
                        "image_url": None,
                        "image_width": crop_img_width,
                        "image_height": crop_img_height,
                        "img_type": "local",
                        "crop_index": rid * len(parts) + cid,
                        "args_crop_fn": {
                            "image": img_one["image_url"],
                            "upscale_image_size": [int(img_width), int(img_height)],
                            "crop_position": crop_info,
                        },
                        "image_type": img_one["image_type"],
                    }
                    return_sequence.append(("image", local_one))

                    # add position info
                    if cid < len(rows) - 1:
                        return_sequence.append(
                            (
                                "text",
                                {
                                    "text": self.crop_col_sep,
                                    "tag": self.crop_tag,
                                    "crop_pos_flag": True,
                                },
                            )
                        )

                return_sequence.append(
                    (
                        "text",
                        {
                            "text": self.crop_row_sep,
                            "tag": self.crop_tag,
                            "crop_pos_flag": True,
                        },
                    )
                )

        img_one["img_type"] = "global"
        img_one["split_row_col"] = [row, col]  # for display
        return_sequence.append(("image", img_one))

        if self.is_pretraining:
            return_sequence.append(
                ("text", {"text": self.crop_image_sep, "tag": self.crop_tag, "crop_pos_flag": True})
            )
        return return_sequence

    def _adaptive_partition_with_position(self, meta: dict):
        """
        adaptive partition with position
        """

        text_info, image_info = [], []
        partition_info = []
        image_token_count = 0
        pre_idx = 0

        for img_one in meta["image_info"]:
            """[STEP 1] need crop"""
            img_width, img_height = img_one["image_width"], img_one["image_height"]
            raw_max_tile = np.random.choice(self.crop_tile_option, p=self.crop_tile_rate)
            if self.use_smart_resize:
                max_tile = raw_max_tile
                min_tile = self.min_tile if self.min_crop_flag else 1
                if max_tile < min_tile:
                    logger.info(f"set to min_tile({min_tile}) to litte max_tile({max_tile})")
                    min_tile = max_tile

                assert self.crop_width == self.crop_height, "crop_width and crop_height should be the same."
                image_factor = self.crop_width
                min_pixels = min_tile * self.crop_width * self.crop_height
                max_pixels = GIVEN_MAX_TILE * self.crop_width * self.crop_height
                img_width, img_height = smart_resize(
                    img_width,
                    img_height,
                    factor=image_factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    max_ratio=MAX_RATIO,
                )
                if img_width == 0 or img_height == 0:
                    img_width = img_one["image_width"]
                    img_height = img_one["img_height"]
            else:
                max_tile = raw_max_tile
                if (img_width < self.crop_width) ^ (img_height < self.crop_height):
                    img_width, img_height = self._get_rounded(img_width, img_height, self.crop_width, all_resize=False)

            # number of tiles
            original_tile = math.ceil(img_width * img_height / (self.crop_width * self.crop_height))
            max_tile = min(original_tile, max_tile)

            if max_tile > 1:
                row, col = self._adaptive_partition(img_width, img_height, max_tile=max_tile)
            else:
                # no need crop
                row, col = 1, 1
            # min_crop
            if row * col < self.min_tile and self.min_crop_flag and not self.use_smart_resize:
                assert self.min_tile in MIN_CROP_SUPPORT_RANGE, f"We only support min_tile = {MIN_CROP_SUPPORT_RANGE}."
                img_width, img_height = self.crop_width * 2, self.crop_height * 2

                max_tile = raw_max_tile
                original_tile = math.ceil(img_width * img_height / (self.crop_width * self.crop_height))
                max_tile = min(original_tile, max_tile)

                row, col = self._adaptive_partition(img_width, img_height, max_tile=max_tile)

            """[STEP 2]"""
            matched_text_index = img_one["matched_text_index"]
            if matched_text_index > 0:
                text_info += meta["text_info"][pre_idx:matched_text_index]
                pre_idx = matched_text_index

            if row * col > 1:
                # start crop
                parts = self._get_crop_info(img_width, img_height, split_row_col=[row, col])

                # list[list]
                for rid, rows in enumerate(parts):
                    for cid, crop_info in enumerate(rows):
                        # crop_info = [left, top, right, bottom] of crop image
                        left, top, right, bottom = crop_info
                        crop_img_width = right - left
                        crop_img_height = bottom - top

                        local_one = {
                            "image_url": None,
                            "image_width": crop_img_width,
                            "image_height": crop_img_height,
                            "matched_text_index": len(text_info),
                            "img_type": "local",
                            "crop_index": rid * len(parts) + cid,
                            "args_crop_fn": {
                                "image": img_one["image_url"],
                                "upscale_image_size": [int(img_width), int(img_height)],
                                "crop_position": crop_info,
                            },
                            "image_type": img_one["image_type"],
                        }
                        image_info.append(local_one)
                        image_token_count += self.im_prefix_length

                        # add position info
                        if cid < len(rows) - 1:
                            text_info.append(
                                {
                                    "text": self.crop_col_sep,
                                    "tag": self.crop_tag,
                                    "crop_pos_flag": True,
                                }
                            )
                            image_token_count += self.crop_col_sep_token_num

                    text_info.append(
                        {
                            "text": self.crop_row_sep,
                            "tag": self.crop_tag,
                            "crop_pos_flag": True,
                        }
                    )
                    image_token_count += self.crop_row_sep_token_num

            img_one["matched_text_index"] = len(text_info)
            img_one["img_type"] = "global"
            img_one["split_row_col"] = [row, col]  # for display
            if self.is_pretraining:
                text_info.append({"text": self.crop_image_sep, "tag": self.crop_tag, "crop_pos_flag": True})
                image_token_count += self.crop_image_sep_token_num
            image_info.append(img_one)
            image_token_count += self.im_prefix_length

            partition_info.append([row, col])

        text_info += meta["text_info"][pre_idx:]

        meta_partition = {
            "text_info": text_info,
            "image_info": image_info,
            "image_token_count": image_token_count,
        }
        if self.return_partition_info:
            meta_partition["partition_info"] = partition_info
        return meta_partition

    def get_cropped_images_token_num(self, image_info: dict):
        """get_cropped_images_token_num"""
        fake_meta = {"image_info": image_info, "text_info": []}
        meta_partition = self._adaptive_partition_with_position(fake_meta)
        image_token_count = meta_partition["image_token_count"]

        return image_token_count

    def get_num_cropped_images(self, element, min_tile=None, max_tile=None):
        """get_num_cropped_images"""
        seq = self._adaptive_partition_by_element(element, min_tile=min_tile, max_tile=max_tile)
        return len([data_type for (data_type, _) in seq if data_type == "image"])
