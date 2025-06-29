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
crop adaptive
"""

import numpy as np

from data_processor.steps.input_ids_massaging.data_crop.crop import Crop
from data_processor.steps.input_ids_massaging.data_utils import get_text_token_num
from data_processor.utils.logger_utils import logger
from data_processor.utils.video_utils import group_frame_by_video


class CropAdaptive(Crop):
    """
    adaptive cropper
    """

    def __init__(
        self,
        image_processor,
        spatial_conv_size,
        temporal_conv_size,
        tokenizer,
        special_tokens_info=None,
        is_pretraining=False,
        use_crop_specialtoken=False,
        adaptive_max_imgtoken_option=None,
        adaptive_max_imgtoken_rate=None,
        video_min_pixels=None,
        video_max_pixels=None,
        rope_3d=False,
        **kwargs,
    ):
        """
        intialize
        """
        self.image_processor = image_processor
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size
        self.tokenizer = tokenizer
        self.is_pretraining = is_pretraining
        self.rope_3d = rope_3d

        if use_crop_specialtoken:
            self.crop_image_sep = special_tokens_info["crop"][-1]
        else:
            self.crop_image_sep = "\n"

        self.crop_image_sep_token_num = get_text_token_num(self.tokenizer, self.crop_image_sep)
        self.crop_tag = "no_mask" if use_crop_specialtoken and self.is_pretraining else "mask"

        # Random Adaptive Max Image token
        self.set_max_image_token_to_processor(adaptive_max_imgtoken_option, adaptive_max_imgtoken_rate)
        # video
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels

    def set_max_image_token_to_processor(self, adaptive_max_imgtoken_option=None, adaptive_max_imgtoken_rate=None):
        """
        set random adaptive max image token to processor
        """
        if adaptive_max_imgtoken_option is not None and adaptive_max_imgtoken_rate is not None:
            max_image_token = np.random.choice(adaptive_max_imgtoken_option, p=adaptive_max_imgtoken_rate)
            max_pixels = max_image_token * (self.image_processor.patch_size**2) * (self.image_processor.merge_size**2)
            max_pixels = int(max_pixels)
            msg = (
                "CropAdaptive set_max_image_token_to_processor() ",
                f"[USE Random Adaptive Max Image token] {max_image_token} from ",
                f"{adaptive_max_imgtoken_option} with {adaptive_max_imgtoken_rate}",
            )
            self.image_processor.set_pixels(max_pixels=max_pixels, msg=msg)

    def set_video_pixels(self, video_min_pixels=None, video_max_pixels=None, msg=""):
        """
        set video pixels
        """
        if video_min_pixels is not None:
            assert isinstance(video_min_pixels, int) and video_min_pixels >= 0, "video_min_pixels must be positive int"
            logger.info(f"{msg} CropAdaptive set video_min_pixels = {video_min_pixels}")
            self.video_min_pixels = video_min_pixels
        if video_max_pixels is not None:
            assert isinstance(video_max_pixels, int) and video_max_pixels > 0, "video_max_pixels must be positive int"
            logger.info(f"{msg} CropAdaptive set video_max_pixels = {video_max_pixels}")
            self.video_max_pixels = video_max_pixels

    def get_num_of_tokens_for_img_one(self, img_one, min_pixels=None, max_pixels=None):
        """
        return the num of placeholder for the given img_one, note that the given img_one can be image or video
        """
        assert "image_height" in img_one and "image_width" in img_one

        if img_one["image_type"] == "video":
            if min_pixels is None:
                min_pixels = self.video_min_pixels
            if max_pixels is None:
                max_pixels = self.video_max_pixels

        height_patches, weight_patches = self.image_processor.get_smarted_resize(
            img_one["image_height"],
            img_one["image_width"],
            min_pixels,
            max_pixels,
        )[1]

        if img_one["image_type"] == "image":
            return height_patches * weight_patches // (self.spatial_conv_size**2)
        elif img_one["image_type"] == "video":
            # !!! CAUTION: the result may not be a integer !!!
            return height_patches * weight_patches // (self.spatial_conv_size**2) / self.temporal_conv_size
        else:
            raise ValueError(f"encounter unsupported image_type: {img_one['image_type']}")

    def process(self, sample: dict):
        """
        adaptive cropper
        """
        image_info = sample["image_info"]
        text_info = sample["text_info"]
        offset = 0
        grouped_frames = group_frame_by_video(image_info)

        for frames in grouped_frames:
            if image_info[frames[0]]["image_type"] == "video":
                assert (
                    len(frames) % self.temporal_conv_size == 0
                ), f"image_info: {image_info}, len(image_info): {len(image_info)}, \
                frames: {frames}, self.temporal_conv_size: {self.temporal_conv_size}"

                for idx, frame_index in enumerate(frames):
                    img_one = image_info[frame_index]
                    height_patches, weight_patches = self.image_processor.get_smarted_resize(
                        img_one["image_height"], img_one["image_width"], self.video_min_pixels, self.video_max_pixels
                    )[1]
                    img_one["grid_h"] = height_patches
                    img_one["grid_w"] = weight_patches

                    img_one["matched_text_index"] += offset
                    matched_text_index = img_one["matched_text_index"]
                    if (
                        (idx % self.temporal_conv_size) == (self.temporal_conv_size - 1)
                        and self.is_pretraining
                        and not self.rope_3d
                    ):
                        text_info = (
                            text_info[:matched_text_index]
                            + [{"text": self.crop_image_sep, "tag": self.crop_tag, "crop_pos_flag": True}]
                            + text_info[matched_text_index:]
                        )
                        offset += 1
            else:
                assert len(frames) == 1
                img_one = image_info[frames[0]]
                height_patches, weight_patches = self.image_processor.get_smarted_resize(
                    img_one["image_height"], img_one["image_width"]
                )[1]
                img_one["grid_h"] = height_patches
                img_one["grid_w"] = weight_patches

                img_one["matched_text_index"] += offset
                matched_text_index = img_one["matched_text_index"]
                if self.is_pretraining:
                    text_info = (
                        text_info[:matched_text_index]
                        + [{"text": self.crop_image_sep, "tag": self.crop_tag, "crop_pos_flag": True}]
                        + text_info[matched_text_index:]
                    )
                    offset += 1

        sample["image_info"] = image_info
        sample["text_info"] = text_info
        return sample

    def get_cropped_images_token_num(self, image_info, min_pixels=None, max_pixels=None, return_detail=False):
        """
        return the number of vision placeholder given the image_info
        """
        ret = []
        grouped_frames = group_frame_by_video(image_info)
        for frames in grouped_frames:
            if image_info[frames[0]]["image_type"] == "video":
                assert (
                    len(frames) % self.temporal_conv_size == 0
                ), f"image_info: {image_info}, len(image_info): {len(image_info)}, \
                frames: {frames}, self.temporal_conv_size: {self.temporal_conv_size}"
                tmp = [self.get_num_of_tokens_for_img_one(image_info[f], min_pixels, max_pixels) for f in frames]
                ret.append(sum(tmp))
            else:
                assert len(frames) == 1
                ret.append(self.get_num_of_tokens_for_img_one(image_info[frames[0]], min_pixels, max_pixels))

        if return_detail:
            assert len(grouped_frames) == len(ret)
            return list(zip(ret, grouped_frames))

        return sum(ret)
