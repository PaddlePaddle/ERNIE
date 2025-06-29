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

"""Video"""

import copy
import json
import math
import random
from collections import defaultdict
from itertools import groupby

from data_processor.steps.input_ids_massaging.data_process.process import Process
from data_processor.steps.input_ids_massaging.data_utils import get_uniq_id
from data_processor.utils.io_utils import get_hashable, image_info_2_hash
from data_processor.utils.logger_utils import logger
from data_processor.utils.random_context import RandomSeedContext
from data_processor.utils.video_utils import group_frame_by_video
from ernie.tokenizer_vl import (
    SFT_ASR_END_TOKEN,
    SFT_ASR_START_TOKEN,
    SFT_IMAGE_END_TOKEN,
    SFT_IMAGE_START_TOKEN,
)


class VideoProcess(Process):
    """VideoProcess"""

    def __init__(
        self,
        max_seq_len,
        temporal_conv_size,
        one_sample_in_one_seq,
        tokenizer,
        im_prefix_length,
        cropper,
        crop_tile_option,
        crop_tile_rate,
        video_max_tile=-1,
        max_dec_len=0,
        is_training=False,
        is_pretraining=False,
        video_fix_crop=-1,
        variable_resolution=False,
        rope_3d=False,
        **kwargs,
    ):
        self.max_seq_len = max_seq_len
        self.temporal_conv_size = temporal_conv_size
        self.one_sample_in_one_seq = one_sample_in_one_seq
        self.tokenizer = tokenizer
        self.im_prefix_length = im_prefix_length
        self.cropper = cropper
        self.crop_tile_option = crop_tile_option
        self.crop_tile_rate = crop_tile_rate

        self.video_fix_crop = video_fix_crop
        self.video_max_tile = video_max_tile

        self.max_dec_len = max_dec_len
        self.is_training = is_training
        self.is_pretraining = is_pretraining

        self.image_start_token = SFT_IMAGE_START_TOKEN
        self.image_end_token = SFT_IMAGE_END_TOKEN

        self.asr_start_token = SFT_ASR_START_TOKEN
        self.asr_end_token = SFT_ASR_END_TOKEN

        self.variable_resolution = variable_resolution
        self.rope_3d = rope_3d

    def process(self, sample, **kwargs):
        """process"""

        """[STEP 0] padding"""
        sample, num_padded_images = self.video_pad(sample)

        sample["num_padded_images"] = num_padded_images

        """[STEP 1] into one seq"""
        if self.one_sample_in_one_seq:
            sample = self.squeeze_video_into_one_seq(sample)

        """[STEP 2] rearrange based on temp conv"""
        sample = self.rearrange_based_on_temp_conv(sample)

        """[STEP 3] add special token"""
        if self.rope_3d:
            sample = self.split_video(sample)

        # sample = self.concat_adjacent_asr(sample)

        sample = self.add_special_tags(sample)

        return sample

    def split_video(self, sample):
        """dummy"""
        uid2count = defaultdict(int)
        for key, group in groupby(sample["image_info"], key=lambda x: x["matched_text_index"]):
            group = list(group)
            uid_tmp = image_info_2_hash(group[0])
            count = uid2count[uid_tmp]
            if count == 0:
                uid = uid_tmp
            else:
                uid = f"{uid_tmp}_{count}"
            uid2count[uid_tmp] += 1
            for img_one in group:
                img_one["video_uid"] = uid

        return sample

    def add_special_tags(self, sample):
        """
        add special tag: <|IMAGE_START|> <|IMAGE_END|> <|ASR_START|> <|ASR_END|>
        """
        assert len(sample["image_info"]) % self.temporal_conv_size == 0
        text_info = sample["text_info"]
        image_info = sample["image_info"]
        if not self.is_pretraining:
            # add image start end
            if not self.rope_3d:
                for img_one_index in range(len(image_info)):
                    img_one = image_info[img_one_index]
                    matched_text_index = img_one["matched_text_index"]
                    if (img_one_index + 1) % self.temporal_conv_size == 1:
                        text_info = (
                            text_info[:matched_text_index]
                            + [{"text": self.image_start_token, "tag": "mask"}]
                            + text_info[matched_text_index:]
                        )
                        for img_index_helper in range(img_one_index, len(image_info)):
                            image_info[img_index_helper]["matched_text_index"] += 1
                    elif (img_one_index + 1) % self.temporal_conv_size == 0:
                        text_info = (
                            text_info[:matched_text_index]
                            + [{"text": self.image_end_token, "tag": "mask"}]
                            + text_info[matched_text_index:]
                        )
                        for img_index_helper in range(img_one_index + 1, len(image_info)):
                            image_info[img_index_helper]["matched_text_index"] += 1
                assert len([i for i in text_info if i["text"] == self.image_end_token]) == len(
                    [i for i in text_info if i["text"] == self.image_start_token]
                )
                assert len([i for i in text_info if i["text"] == self.image_end_token]) == len(image_info) // 2

            # add special tag
            text_one_index = 0
            while text_one_index < len(text_info):
                if text_info[text_one_index].get("is_asr", False):
                    # add asr start and end
                    text_info = (
                        text_info[:text_one_index]
                        + [{"text": self.asr_start_token, "tag": "mask"}]
                        + [text_info[text_one_index]]
                        + [{"text": self.asr_end_token, "tag": "mask"}]
                        + text_info[text_one_index + 1 :]
                    )

                    for img_index in range(len(image_info)):
                        if image_info[img_index]["matched_text_index"] > text_one_index:
                            image_info[img_index]["matched_text_index"] += 2

                    text_one_index += 2

                text_one_index += 1

        sample["text_info"] = text_info
        sample["image_info"] = image_info

        return sample

    def count_video_cropped_image_tokens(self, image_info, max_tile):
        """count the number of tokens needed to store all images"""

        crop_tile_option_ori = copy.deepcopy(self.cropper.crop_tile_option)
        crop_tile_rate_ori = copy.deepcopy(self.cropper.crop_tile_rate)

        self.cropper.crop_tile_option = [max_tile]
        self.cropper.crop_tile_rate = [1.0]

        padded_image_info, _ = self.video_pad_image_info(image_info)
        image_token_count = self.cropper.get_cropped_images_token_num(padded_image_info)

        # special tags for sft
        if not self.is_pretraining:
            image_token_count += len(padded_image_info) * 2

        self.cropper.crop_tile_option = crop_tile_option_ori
        self.cropper.crop_tile_rate = crop_tile_rate_ori

        return image_token_count

    def remove_video_frames(self, meta, frame_indices_to_remove):
        """remove some frames from a video"""
        image_info = meta["image_info"]
        text_info = meta["text_info"]

        image_info = [i for idx, i in enumerate(image_info) if idx not in frame_indices_to_remove]

        meta["image_info"] = image_info
        meta["text_info"] = text_info

        return meta

    def group_frame_by_video(self, schema):
        """group by video"""
        return group_frame_by_video(schema)

    def get_frame_indices_to_remove_for_one_video(self, meta, frames, num_frames_to_be_deleted):
        """get the indices of frames that need to be removed"""
        num_frames = len(frames)
        max_frames = num_frames - num_frames_to_be_deleted

        frame_interval = num_frames // max_frames if num_frames >= max_frames else 1
        frame_indices_selected = frames[::frame_interval]
        if len(frame_indices_selected) > max_frames:
            random_seed = get_uniq_id(get_hashable(meta["image_info"][frames[0]]["image_url"]))
            with RandomSeedContext(random_seed):
                indices_selected = random.sample(range(1, len(frame_indices_selected) - 1), k=max_frames - 2)
            indices_selected.sort()
            indices_selected = [0] + indices_selected + [len(frame_indices_selected) - 1]
            frame_indices_selected = [frame_indices_selected[i] for i in indices_selected]

        frame_indices_selected = set(frame_indices_selected)
        num_frames_selected = len(frame_indices_selected)
        frame_indices_to_remove = [i for i in frames if i not in frame_indices_selected]

        return frame_indices_to_remove

    def squeeze_video_into_one_seq(self, meta, max_seq_len=None):
        """squeeze video into one sequence"""
        if self.variable_resolution:
            return self.squeeze_video_into_one_seq_adaptive(meta, max_seq_len=None)
        else:
            return self.squeeze_video_into_one_seq_crop(meta, max_seq_len=max_seq_len)

    def squeeze_video_into_one_seq_adaptive(self, meta, max_seq_len=None):
        """squeeze video into one sequence adaptively"""

        def get_vision_tokens(tmp_video_min_pixels, tmp_video_max_pixels):
            """
            get image tokens
            """
            tmp_grouped_frames_details = self.cropper.get_cropped_images_token_num(
                [i for i in meta["image_info"] if i["image_type"] == "video"],
                min_pixels=tmp_video_min_pixels,
                max_pixels=tmp_video_max_pixels,
                return_detail=True,
            )

            tmp_vision_tokens_num = sum([i[0] for i in tmp_grouped_frames_details])
            if self.is_pretraining and not self.rope_3d:
                tmp_vision_tokens_num += round(
                    sum(
                        [
                            len(i[1]) / self.temporal_conv_size * self.cropper.crop_image_sep_token_num
                            for i in tmp_grouped_frames_details
                        ]
                    )
                )

            return tmp_grouped_frames_details, tmp_vision_tokens_num

        def judge_single_adaptive_resolution(tmp_video_min_pixels, tmp_video_max_pixels, quota_num_tokens):
            """judge single resolution"""
            _, tmp_vision_tokens_num = get_vision_tokens(tmp_video_min_pixels, tmp_video_max_pixels)

            if tmp_vision_tokens_num < quota_num_tokens:
                return True, tmp_vision_tokens_num

            return False, tmp_vision_tokens_num

        def judge_adaptive_resolution(permt_video_min_pixels, permt_video_max_pixels, quota_num_tokens):
            """judge adaptive resolution"""
            l = int(permt_video_min_pixels)
            r = int(permt_video_max_pixels)
            flag = False

            try:
                while l < r:
                    mid = (l + r + 1) // 2
                    tmp_flag, permt_vision_num_tokens = judge_single_adaptive_resolution(
                        permt_video_min_pixels, mid, quota_num_tokens
                    )
                    if tmp_flag:
                        l = mid
                        flag = True
                        fi_permt_vision_num_tokens = permt_vision_num_tokens
                    else:
                        r = mid - 1
            except ValueError:
                logger.debug("[BINARY SEARCH] encounter resized shape smaller than min_pixels, early exit!")
                return False, r, permt_vision_num_tokens

            if flag:
                return flag, l, fi_permt_vision_num_tokens
            return flag, l, permt_vision_num_tokens

        def calculate_ratios_with_min_one(numbers):
            if not numbers:
                return []

            total = sum(numbers)
            if total == 0:
                raise ValueError("the sum of numbers cannot be 0")

            base_ratios = [num / total for num in numbers]

            min_ratio = min(base_ratios)

            adjusted_ratios = [round(ratio / min_ratio) for ratio in base_ratios]

            return adjusted_ratios

        logger.debug("*******start one video squeeze_video_into_one_seq_adaptive********")
        video_min_pixels = self.cropper.video_min_pixels
        video_max_pixels = self.cropper.video_max_pixels
        if video_min_pixels is None:
            video_min_pixels = self.cropper.image_processor.min_pixels
        if video_max_pixels is None:
            video_max_pixels = self.cropper.image_processor.max_pixels

        assert video_min_pixels > 0
        assert video_max_pixels > 0

        actual_min_pixels = video_min_pixels
        for i in meta["image_info"]:
            if i["image_type"] == "video":
                actual_min_pixels = min(actual_min_pixels, i["image_width"] * i["image_height"])
        if actual_min_pixels < video_min_pixels:
            self.cropper.set_video_pixels(
                video_min_pixels=actual_min_pixels,
                msg="VideoProcess.squeeze_video_into_one_seq_adaptive() set actual_min_pixels",
            )
        video_min_pixels = actual_min_pixels
        logger.debug(f"image_width: {i['image_width']}, image_height: {i['image_height']}")
        logger.debug(f"video_min_pixels: {video_min_pixels}, video_max_pixels: {video_max_pixels}")

        if max_seq_len is None:
            max_seq_len = self.max_seq_len

        # calculate the ratio of each video
        text_token_count = 0
        for item in meta["text_info"]:
            text_token_count += len(self.tokenizer.encode(item["text"])["input_ids"])
        text_token_count += 1  # for eos token

        # consider asr token
        if not self.is_pretraining:
            text_token_count += sum([2 for i in meta["text_info"] if i.get("is_asr", False)])

        if not self.is_training:
            text_token_count += self.max_dec_len

        image_token_limit = max_seq_len - text_token_count

        # elements in the list: [num_of_placeholder_for_the_video, [frame_indices]]
        _, vision_tokens_num = get_vision_tokens(video_min_pixels, video_max_pixels)

        logger.debug(f"original vision_tokens_num: {vision_tokens_num}, image_token_limit: {image_token_limit}")
        if vision_tokens_num < image_token_limit:
            logger.debug(f"seq len: {vision_tokens_num + text_token_count}")
            logger.debug("no need to squeeze")
            return meta

        (judge_adaptive_flag, judge_video_max_pixels, permt_vision_tokens_num) = judge_adaptive_resolution(
            video_min_pixels, video_max_pixels, image_token_limit
        )

        logger.debug(f"after adjust, video_min_pixels: {video_min_pixels}, video_max_pixels: {judge_video_max_pixels}")
        logger.debug(
            f"after adjust, vision_tokens_num: {permt_vision_tokens_num}, image_token_limit: {image_token_limit}"
        )
        self.cropper.set_video_pixels(
            video_max_pixels=judge_video_max_pixels,
            msg="VideoProcess.squeeze_video_into_one_seq_adaptive() set judge_video_max_pixels",
        )

        if judge_adaptive_flag:
            logger.debug("after adjust, directly return meta")
            grouped_frames_details, vision_tokens_num = get_vision_tokens(
                self.cropper.video_min_pixels, self.cropper.video_max_pixels
            )
            logger.debug(f"seq len: {vision_tokens_num + text_token_count}")
            return meta
        else:
            grouped_frames_details, vision_tokens_num = get_vision_tokens(
                self.cropper.video_min_pixels, self.cropper.video_max_pixels
            )

            grouped_frames = [i[1] for i in grouped_frames_details]
            token_per_frame_per_video = [i[0] // len(i[1]) for i in grouped_frames_details]

            if self.rope_3d:
                num_special_tokens_per_conv_size = 0
            else:
                if not self.is_pretraining:
                    # <image start> <image end>
                    num_special_tokens_per_conv_size = 2
                else:
                    # <img sep>
                    num_special_tokens_per_conv_size = self.cropper.crop_image_sep_token_num

            tokens_to_delete = vision_tokens_num - image_token_limit

            num_frames_to_be_deleted_for_each_video = [0 for _ in grouped_frames]
            special_tokens_removed_per_video = [0 for _ in grouped_frames]
            video_cnt = 0
            break_cond = 0

            ratio = calculate_ratios_with_min_one([i[0] for i in grouped_frames_details])
            ratio = [i * self.temporal_conv_size for i in ratio]

            while tokens_to_delete > 0 and break_cond < len(grouped_frames):
                video_index = video_cnt % len(grouped_frames)
                if (
                    len(grouped_frames[video_index])
                    - num_frames_to_be_deleted_for_each_video[video_index]
                    - ratio[video_index]
                    >= 2
                ):
                    # image tokens
                    num_frames_to_be_deleted_for_each_video[video_index] += ratio[video_index]
                    tokens_to_delete -= ratio[video_index] * token_per_frame_per_video[video_index]

                    # special tokens
                    tokens_to_delete += special_tokens_removed_per_video[video_index]
                    special_tokens_removed_per_video[video_index] = (
                        num_frames_to_be_deleted_for_each_video[video_index] // self.temporal_conv_size
                    ) * num_special_tokens_per_conv_size
                    tokens_to_delete -= special_tokens_removed_per_video[video_index]

                    break_cond = 0
                else:
                    break_cond += 1
                video_cnt += 1

            # drop token
            frame_indices_to_remove = []
            for frames, num_frames_to_be_deleted in zip(grouped_frames, num_frames_to_be_deleted_for_each_video):
                logger.debug(
                    f"original frames {len(frames)}, num_frames_to_be_deleted {num_frames_to_be_deleted}, "
                    + f"final frames {len(frames) - num_frames_to_be_deleted}"
                )
                frame_indices_to_remove.extend(
                    self.get_frame_indices_to_remove_for_one_video(meta, frames, num_frames_to_be_deleted)
                )

            meta = self.remove_video_frames(meta, frame_indices_to_remove)
            # debug
            _, final_vision_tokens_num = get_vision_tokens(
                self.cropper.video_min_pixels, self.cropper.video_max_pixels
            )
            logger.debug(
                f"after frame removing, final vision_tokens_num: {final_vision_tokens_num}, "
                + f"text_token_count: {text_token_count}, "
                + f"final seq len: {final_vision_tokens_num + text_token_count}"
            )

        num_frames_selected = sum([len(i) for i in grouped_frames]) - sum(num_frames_to_be_deleted_for_each_video)
        if self.is_training:
            logger.debug(f"for one_sample_in_one_seq_adaptive, num_frames={num_frames_selected}")
        else:
            logger.info(f"for one_sample_in_one_seq_adaptive, num_frames={num_frames_selected}")

        return meta

    def squeeze_video_into_one_seq_crop(self, meta, max_seq_len=None):
        """
        squeeze all videos into a single sequence with cropping
        """
        if max_seq_len is None:
            max_seq_len = self.max_seq_len

        text_token_count = 0
        for item in meta["text_info"]:
            text_token_count += len(self.tokenizer.encode(item["text"])["input_ids"])
        text_token_count += 1  # for eos token

        image_token_limit = max_seq_len - text_token_count

        if not self.is_pretraining:
            image_token_limit -= sum([2 for i in meta["text_info"] if i.get("is_asr", False)])

        if not self.is_training:
            image_token_limit -= self.max_dec_len

        token_per_frame = self.im_prefix_length + self.cropper.crop_image_sep_token_num
        # speical tags in sft
        if not self.is_pretraining:
            token_per_frame += 2
        if self.video_fix_crop != -1:
            token_per_frame *= self.video_fix_crop + 1
        # calculate the ratio of each video
        max_frames = image_token_limit // token_per_frame
        max_frames -= max_frames % self.temporal_conv_size
        if max_frames < self.temporal_conv_size:
            meta["video_crop_param"] = {
                "crop_tile_option": [1],
                "crop_tile_rate": [1.0],
            }
            return meta

        max_tile = max(self.crop_tile_option)

        while max_tile >= 1:
            image_token_count = self.count_video_cropped_image_tokens(meta["image_info"], max_tile)
            if image_token_count <= image_token_limit:
                break
            max_tile -= 1

        # select frames
        num_frames_selected = len(meta["image_info"])
        if max_tile >= max(self.crop_tile_option):
            pass
        elif max_tile > 0 and self.video_fix_crop == -1:
            new_crop_tile_option = []
            new_crop_tile_rate = []
            for option, rate in zip(self.crop_tile_option, self.crop_tile_rate):
                if option < max_tile:
                    new_crop_tile_option.append(option)
                    new_crop_tile_rate.append(rate)

            max_tile_rate = 1 - sum(new_crop_tile_rate)
            if max_tile_rate > 1e-6:
                new_crop_tile_option.append(max_tile)
                new_crop_tile_rate.append(max_tile_rate)

            meta["video_crop_param"] = {
                "crop_tile_option": new_crop_tile_option,
                "crop_tile_rate": new_crop_tile_rate,
            }
        else:
            num_images = len([i for i in meta["image_info"] if i["image_type"] == "video"])

            grouped_frames = self.group_frame_by_video(meta)
            to_delete = num_images - max_frames

            num_frames_to_be_deleted_for_each_video = [0 for _ in grouped_frames]
            video_cnt = 0
            break_cond = 0
            while to_delete > 0 and break_cond < len(grouped_frames):
                video_index = video_cnt % len(grouped_frames)
                if len(grouped_frames[video_index]) - num_frames_to_be_deleted_for_each_video[video_index] - 1 > 2:
                    num_frames_to_be_deleted_for_each_video[video_index] += 1
                    to_delete -= 1
                    break_cond = 0
                else:
                    break_cond += 1
                video_cnt += 1

            # drop token
            frame_indices_to_remove = []
            for frames, num_frames_to_be_deleted in zip(grouped_frames, num_frames_to_be_deleted_for_each_video):
                frame_indices_to_remove.extend(
                    self.get_frame_indices_to_remove_for_one_video(meta, frames, num_frames_to_be_deleted)
                )

            meta = self.remove_video_frames(meta, frame_indices_to_remove)

            meta["video_crop_param"] = {
                "crop_tile_option": [1] if self.video_fix_crop == -1 else [self.video_fix_crop],
                "crop_tile_rate": [1.0],
            }

        if self.is_training:
            logger.debug(
                f"for one_sample_in_one_seq, video_fix_crop={self.video_fix_crop}, \
                  max_tile={max_tile}, num_frames={num_frames_selected}"
            )
        else:
            logger.info(
                f"for one_sample_in_one_seq, video_fix_crop={self.video_fix_crop}, \
                  max_tile={max_tile}, num_frames={num_frames_selected}"
            )

        return meta

    def video_pad_image_info(self, image_info):
        """
        pad the image info to match the temporal_conv_size
        """
        new_image_info = copy.deepcopy(image_info)
        num_padded_images = 0
        for idx in range(len(new_image_info)):
            new_image_info[idx]["is_padded_image"] = False

        grouped_frames = self.group_frame_by_video(new_image_info)

        index_offset = 0
        for frames in grouped_frames:
            if len(frames) % self.temporal_conv_size != 0:
                roundup = math.ceil(len(frames) / self.temporal_conv_size) * self.temporal_conv_size
                num_padded_images = roundup - len(frames)
                tmp = []
                for _ in range(num_padded_images):
                    padded_image = copy.deepcopy(image_info[frames[-1]])
                    padded_image["is_padded_image"] = True
                    tmp.append(padded_image)
                new_image_info = (
                    new_image_info[: index_offset + len(frames)] + tmp + new_image_info[index_offset + len(frames) :]
                )
                index_offset += len(tmp)

            index_offset += len(frames)

        return new_image_info, num_padded_images

    def video_pad(self, meta):
        """
        pad the video sample to match the temporal_conv_size
        """
        new_image_info, num_padded_images = self.video_pad_image_info(meta["image_info"])
        meta["image_info"] = new_image_info

        return meta, num_padded_images

    def rearrange_based_on_temp_conv(self, meta):
        """rearrange based on temp conv size"""

        conv_size = self.temporal_conv_size

        result = copy.deepcopy(meta)
        meta = copy.deepcopy(meta)

        images = meta["image_info"]
        texts = meta["text_info"]

        resulted_image_info = []
        resulted_text_info = []
        images_sliding_window = []
        last_index = 0
        appended_text_index = -1

        for idx, image in enumerate(images):
            images_sliding_window.append(copy.deepcopy(image))
            if len(images_sliding_window) >= conv_size:
                match_indices = sorted([i["matched_text_index"] for i in images_sliding_window])
                # append the corresponding texts except for the last one,
                # coz the current windows may not contain all the correspoding images

                for match_index in range(appended_text_index + 1, match_indices[-1]):
                    resulted_text_info.append(copy.deepcopy(texts[match_index]))
                    appended_text_index = match_index

                # take special care for the last match index
                last_match_index = match_indices[-1]
                if idx < len(images) - 1:
                    contain_all_cor_image = last_match_index != images[idx + 1]["matched_text_index"]
                else:
                    # there isnt any image left, so the current window contains
                    # all correspoding images for the last match index
                    contain_all_cor_image = True

                if contain_all_cor_image:
                    resulted_text_info.append(copy.deepcopy(texts[last_match_index]))
                    appended_text_index = last_match_index

                for tmp in images_sliding_window:
                    tmp["matched_text_index"] = max(last_index, match_indices[0])
                last_index = len(resulted_text_info)
                resulted_image_info.extend(images_sliding_window)
                images_sliding_window = []

        if len(images_sliding_window) != 0:
            match_indices = sorted([i["matched_text_index"] for i in images_sliding_window])
            # sliding window has complete the swiping, so
            # push all left images and texts to the two resulted list.
            for match_index in range(appended_text_index + 1, match_indices[-1]):
                resulted_text_info.append(copy.deepcopy(texts[match_index]))
                appended_text_index = match_index

            for tmp in images_sliding_window:
                tmp["matched_text_index"] = max(last_index, match_indices[0])
            resulted_image_info.extend(images_sliding_window)
            images_sliding_window = []

        if len(resulted_text_info) != len(meta["text_info"]):
            for index in range(appended_text_index + 1, len(meta["text_info"])):
                resulted_text_info.append(copy.deepcopy(meta["text_info"][index]))

        assert len(resulted_text_info) == len(
            meta["text_info"]
        ), f"""len(resulted_text_info): {len(resulted_text_info)},
        len(meta['text_info']): {len(meta['text_info'])},
        meta_debug: {json.dumps(result, indent=4)}"""
        # resulted_text_info: {json.dumps(resulted_text_info, indent=4)},
        # resulted_image_info: {json.dumps(resulted_image_info, indent=4)}"""

        for i in range(len(resulted_image_info)):
            assert (
                resulted_image_info[i]["matched_text_index"] >= 0
            ), f"i: {i}, resulted_image_info[i]: {resulted_image_info[i]}"
            assert resulted_image_info[i]["matched_text_index"] <= images[i]["matched_text_index"]

        assert len(resulted_image_info) == len(
            meta["image_info"]
        ), f"len(resulted_image_info): {len(resulted_image_info)}, len(meta['image_info']): {len(meta['image_info'])}"
        assert len(resulted_text_info) == len(
            meta["text_info"]
        ), f"len(resulted_text_info): {len(resulted_text_info)}, len(meta['text_info']): {len(meta['text_info'])}"

        result["text_info"] = resulted_text_info
        result["image_info"] = resulted_image_info

        for i, j in zip(result["text_info"], meta["text_info"]):
            assert i["text"] == j["text"]

        for i, j in zip(result["image_info"], meta["image_info"]):
            assert i["image_url"] == j["image_url"]

        return result
