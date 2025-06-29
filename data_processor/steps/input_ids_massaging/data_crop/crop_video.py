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

"""crop video"""
import copy
import math

import numpy as np

from data_processor.steps.input_ids_massaging.data_crop.crop_image import CropImage

# sys.path.append("..")


class CropVideo(CropImage):
    """CropVideo"""

    def __init__(
        self,
        tokenizer,
        im_prefix_length,
        temporal_conv_size: int = 1,
        crop_width: int = 448,
        crop_height: int = 448,
        crop_tile_option=None,
        crop_tile_rate=None,
        use_crop_specialtoken: bool = False,
        special_tokens_info=None,
        video_fix_crop: int = -1,
        is_pretraining: bool = False,
        **kwargs,
    ):
        """crop info

        Args:
            tokenizer: tokenizer.
            im_prefix_length: #tokens in LLM for each image tile.
            temporal_conv_size (int, optional): kernel size of temporal convolution. Defaults to 1.
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
        super().__init__(
            tokenizer,
            im_prefix_length,
            crop_width=crop_width,
            crop_height=crop_height,
            crop_tile_option=crop_tile_option,
            crop_tile_rate=crop_tile_rate,
            use_crop_specialtoken=use_crop_specialtoken,
            special_tokens_info=special_tokens_info,
            is_pretraining=is_pretraining,
        )
        self.temporal_conv_size = temporal_conv_size
        self.video_fix_crop = video_fix_crop
        self.is_pretraining = is_pretraining

    def _adaptive_partition_with_position(self, meta: dict):
        """adaptively partitioning the input sequence according to its length"""
        text_info, image_info = [], []
        partition_info = []
        image_token_count = 0
        pre_idx = 0

        # make sure all frames are cropped into the same amount of tiles
        img = meta["image_info"][0]
        img_width, img_height = img["image_width"], img["image_height"]
        if (img_width < self.crop_width) ^ (img_height < self.crop_height):
            img_width, img_height = self._get_rounded(img_width, img_height, self.crop_width, all_resize=False)

        # number of tiles in the original image
        original_tile = math.ceil(img_width * img_height / (self.crop_width * self.crop_height))

        if meta.get("video_crop_param", None) is not None:
            crop_tile_option = meta["video_crop_param"]["crop_tile_option"]
            crop_tile_rate = meta["video_crop_param"]["crop_tile_rate"]
        else:
            crop_tile_option = self.crop_tile_option
            crop_tile_rate = self.crop_tile_rate
        max_tile = np.random.choice(crop_tile_option, p=crop_tile_rate)

        # avoid too many tiles
        max_tile = min(original_tile, max_tile)

        # all images belong to one example, so we can simply group the images by conv size.
        for leading_index in range(0, len(meta["image_info"]), self.temporal_conv_size):
            img_one = meta["image_info"][leading_index]

            matched_text_index = img_one["matched_text_index"]
            if matched_text_index > 0:
                text_info += meta["text_info"][pre_idx:matched_text_index]
                pre_idx = matched_text_index

            partitions_for_chunk = []
            for following_index in range(0, self.temporal_conv_size):
                if leading_index + following_index >= len(meta["image_info"]):
                    # TODO: support this plz
                    # this should not happen, coz image info should be a multiple of conv size
                    assert 0, "image info should be a multiple of conv size"
                    break
                img_one = meta["image_info"][leading_index + following_index]
                img_width, img_height = img_one["image_width"], img_one["image_height"]

                if (img_width < self.crop_width) ^ (img_height < self.crop_height):
                    img_width, img_height = self._get_rounded(img_width, img_height, self.crop_width, all_resize=False)

                # number of tiles in the original image
                original_tile = math.ceil(img_width * img_height / (self.crop_width * self.crop_height))

                if self.video_fix_crop != -1:
                    assert self.video_fix_crop == 2
                    if img_width > img_height:
                        row = 1
                        col = 2
                        img_width = self.crop_width * 2
                        img_height = self.crop_height * 1
                    else:
                        row = 2
                        col = 1
                        img_width = self.crop_width * 1
                        img_height = self.crop_height * 2
                else:
                    row, col = self._adaptive_partition(img_width, img_height, max_tile=max_tile)
                if row * col > 1:
                    # start crop
                    parts = self._get_crop_info(img_width, img_height, split_row_col=[row, col])
                else:
                    parts = None
                partitions_for_chunk.append([row, col, img_one, parts])

            partitions_for_chunk_transposed = [list(x) for x in zip(*partitions_for_chunk)]

            # [row_and_col_crop_info_for_image1, row_and_col_crop_info_for_image2, ...]
            parts_for_chunk = partitions_for_chunk_transposed[3]

            # [image1, image2, ...]
            imgs_for_chunk = partitions_for_chunk_transposed[2]

            if partitions_for_chunk[0][0] * partitions_for_chunk[0][1] > 1:
                for rid, rows_for_chunk in enumerate(parts_for_chunk[0]):
                    # coz every element in parts_for_chunk are the same, iterate the first one to get
                    # the crop info along row
                    for cid, crop_info in enumerate(rows_for_chunk):
                        # iterate along the row to get the crop info for each crop
                        for img_one in imgs_for_chunk:
                            # iterate through imgs to get image_url
                            left, top, right, bottom = crop_info
                            crop_img_width = right - left
                            crop_img_height = bottom - top

                            local_one = {
                                "image_url": None,
                                "image_width": crop_img_width,
                                "image_height": crop_img_height,
                                "matched_text_index": len(text_info),
                                "is_padded_image": img_one["is_padded_image"],
                                "img_type": "local",
                                "args_crop_fn": {
                                    "image": img_one["image_url"],
                                    "upscale_image_size": [
                                        int(img_width),
                                        int(img_height),
                                    ],
                                    "crop_position": crop_info,
                                },
                                "image_type": img_one["image_type"],
                            }

                            image_info.append(local_one)
                            image_token_count += self.im_prefix_length
                        if cid < len(rows_for_chunk) - 1:
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

                for img_one in imgs_for_chunk:
                    global_one = copy.deepcopy(img_one)
                    global_one["matched_text_index"] = len(text_info)
                    global_one["img_type"] = "global"
                    global_one["split_row_col"] = [row, col]  # for display

                    image_info.append(global_one)
                    image_token_count += self.im_prefix_length
                    partition_info.append([row, col])
                if self.is_pretraining:
                    text_info.append(
                        {
                            "text": self.crop_image_sep,
                            "tag": self.crop_tag,
                            "crop_pos_flag": True,
                        }
                    )
                    image_token_count += self.crop_image_sep_token_num
            else:
                for img_one in imgs_for_chunk:
                    global_one = copy.deepcopy(img_one)
                    global_one["matched_text_index"] = len(text_info)
                    global_one["img_type"] = "global"
                    global_one["split_row_col"] = [row, col]  # for display
                    image_info.append(global_one)
                    image_token_count += self.im_prefix_length
                    partition_info.append([row, col])
                if self.is_pretraining:
                    text_info.append({"text": self.crop_image_sep, "tag": self.crop_tag, "crop_pos_flag": True})
                    image_token_count += self.crop_image_sep_token_num

            # partition_info.append([row, col])

        text_info += meta["text_info"][pre_idx:]

        meta_partition = {
            "text_info": text_info,
            "image_info": image_info,
            "partition_info": partition_info,
            "image_token_count": image_token_count,
        }
        return meta_partition

    def process_by_element(self, img_one):
        """
        process by single image
        """
        return self._adaptive_partition_by_element(img_one)

    def _adaptive_partition_by_element(self, frames, min_tile=None, max_tile=None):
        """
        process by single image
        """

        assert len(frames) == self.temporal_conv_size, "len(frames)=%s != conv_size:%s" % (
            len(frames),
            self.temporal_conv_size,
        )
        return_sequence = []  # this is a return sequence for crop

        img = frames[0]
        img_width, img_height = img["image_width"], img["image_height"]
        if (img_width < self.crop_width) ^ (img_height < self.crop_height):
            img_width, img_height = self._get_rounded(img_width, img_height, self.crop_width, all_resize=False)

        # number of tiles in the original image
        original_tile = math.ceil(img_width * img_height / (self.crop_width * self.crop_height))

        crop_tile_option = self.crop_tile_option
        crop_tile_rate = self.crop_tile_rate

        if max_tile is None:
            max_tile = np.random.choice(crop_tile_option, p=crop_tile_rate)

        # avoid too many tiles
        max_tile = min(original_tile, max_tile)

        partitions_for_chunk = []
        for following_index in range(0, self.temporal_conv_size):
            img_one = frames[following_index]
            img_width, img_height = img_one["image_width"], img_one["image_height"]

            if (img_width < self.crop_width) ^ (img_height < self.crop_height):
                img_width, img_height = self._get_rounded(img_width, img_height, self.crop_width, all_resize=False)

            # number of tiles in the original image
            original_tile = math.ceil(img_width * img_height / (self.crop_width * self.crop_height))

            if self.video_fix_crop != -1:
                assert self.video_fix_crop == 2
                if img_width > img_height:
                    row = 1
                    col = 2
                    img_width = self.crop_width * 2
                    img_height = self.crop_height * 1
                else:
                    row = 2
                    col = 1
                    img_width = self.crop_width * 1
                    img_height = self.crop_height * 2
            else:
                row, col = self._adaptive_partition(img_width, img_height, max_tile=max_tile)

            if row * col > 1:
                # start crop
                parts = self._get_crop_info(img_width, img_height, split_row_col=[row, col])
            else:
                parts = None
            partitions_for_chunk.append([row, col, img_one, parts])

        partitions_for_chunk_transposed = [list(x) for x in zip(*partitions_for_chunk)]

        # [row_and_col_crop_info_for_image1, row_and_col_crop_info_for_image2, ...]
        parts_for_chunk = partitions_for_chunk_transposed[3]

        # [image1, image2, ...]
        imgs_for_chunk = partitions_for_chunk_transposed[2]

        if partitions_for_chunk[0][0] * partitions_for_chunk[0][1] > 1:
            for rid, rows_for_chunk in enumerate(parts_for_chunk[0]):
                # coz every element in parts_for_chunk are the same, iterate the first one to get
                # the crop info along row
                for cid, crop_info in enumerate(rows_for_chunk):
                    # iterate along the row to get the crop info for each crop
                    for img_one in imgs_for_chunk:
                        # iterate through imgs
                        left, top, right, bottom = crop_info
                        crop_img_width = right - left
                        crop_img_height = bottom - top

                        local_one = {
                            "image_url": None,
                            "image_width": crop_img_width,
                            "image_height": crop_img_height,
                            "is_padded_image": img_one["is_padded_image"],
                            "img_type": "local",
                            "args_crop_fn": {
                                "image": img_one["image_url"],
                                "upscale_image_size": [
                                    int(img_width),
                                    int(img_height),
                                ],
                                "crop_position": crop_info,
                                "image_type": img_one["image_type"],
                            },
                        }

                        return_sequence.append(("image", local_one))

                    if cid < len(rows_for_chunk) - 1:
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

            for img_one in imgs_for_chunk:
                global_one = copy.deepcopy(img_one)
                global_one["img_type"] = "global"
                global_one["split_row_col"] = [row, col]  # for display
                return_sequence.append(("image", global_one))

            if self.is_pretraining:
                return_sequence.append(
                    (
                        "text",
                        {
                            "text": self.crop_image_sep,
                            "tag": self.crop_tag,
                            "crop_pos_flag": True,
                        },
                    )
                )
        else:
            for img_one in imgs_for_chunk:
                global_one = copy.deepcopy(img_one)
                global_one["img_type"] = "global"
                global_one["split_row_col"] = [row, col]  # for display
                return_sequence.append(("image", global_one))

            if self.is_pretraining:
                return_sequence.append(
                    ("text", {"text": self.crop_image_sep, "tag": self.crop_tag, "crop_pos_flag": True})
                )
        return return_sequence
