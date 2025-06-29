#!/usr/bin/env python3

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

""" Process Interleave"""
import copy
import math
import random
import re
from itertools import groupby

import numpy as np

from data_processor.steps.input_ids_massaging.data_process.process import Process
from data_processor.steps.input_ids_massaging.data_utils import get_uniq_id
from data_processor.utils.io_utils import get_hashable
from data_processor.utils.random_context import RandomSeedContext
from ernie.tokenizer_vl import (
    SFT_ASR_END_TOKEN,
    SFT_ASR_START_TOKEN,
    SFT_IMAGE_END_TOKEN,
    SFT_IMAGE_START_TOKEN,
)

# sys.path.append("..")


class OminiProcess(Process):
    """OminiProcess"""

    def __init__(
        self,
        max_seq_len,
        image_temporal_conv_size,
        video_temporal_conv_size,
        tokenizer,
        im_prefix_length,
        crop_fn_by_type,
        image_crop_tile_option,
        image_crop_tile_rate,
        video_crop_tile_option,
        video_crop_tile_rate,
        use_multi_image_crop,
        special_tokens_info,
        max_dec_len=0,
        is_training=False,
        is_pretraining=False,
        video_max_tile=-1,
        video_fix_crop=-1,
        **kwargs,
    ):
        self.max_seq_len = max_seq_len
        self.crop_fn_by_type = crop_fn_by_type

        self.image_temporal_conv_size = image_temporal_conv_size
        self.video_temporal_conv_size = video_temporal_conv_size

        self.tokenizer = tokenizer
        self.im_prefix_length = im_prefix_length

        self.image_crop_tile_option = image_crop_tile_option
        self.image_crop_tile_rate = image_crop_tile_rate

        self.video_crop_tile_option = video_crop_tile_option
        self.video_crop_tile_rate = video_crop_tile_rate

        self.use_multi_image_crop = use_multi_image_crop

        self.video_fix_crop = video_fix_crop
        self.video_max_tile = video_max_tile

        self.max_dec_len = max_dec_len
        self.is_training = is_training
        self.is_pretraining = is_pretraining

        self.special_tokens_info = special_tokens_info

        self.image_start_token = SFT_IMAGE_START_TOKEN
        self.image_end_token = SFT_IMAGE_END_TOKEN

        self.asr_start_token = SFT_ASR_START_TOKEN
        self.asr_end_token = SFT_ASR_END_TOKEN

    def process(self, sequence, **kwargs):
        """process"""
        crop_options, sequence = self.smart_tile_and_sample_frame(sequence, self.max_seq_len)

        sequence = self.padded_rearrange_based_on_temp_conv(sequence)

        # add asr token
        sequence = self.add_special_tags(sequence)
        return crop_options, sequence

    def padded_rearrange_based_on_temp_conv(self, sequence):
        """padded_rearrange_based_on_temp_conv"""
        sequence = self.video_pad(sequence)

        reversed_new_sequence = []
        frame_buffer = []

        for data_type, item in sequence[::-1]:
            if data_type == "text":
                reversed_new_sequence.append((data_type, item))
            elif data_type == "image":
                if item.get("image_type", "image") == "video":
                    frame_buffer.append((data_type, item))
                    if len(frame_buffer) >= self.video_temporal_conv_size:
                        reversed_new_sequence.extend(frame_buffer)
                        frame_buffer = []
                else:
                    reversed_new_sequence.append((data_type, item))
        return reversed_new_sequence[::-1]

    def add_special_tags(self, sequence):
        """
        add special tag: <|IMAGE_START|> <|IMAGE_END|> <|ASR_START|> <|ASR_END|>
        """
        if self.is_pretraining:
            return sequence
        new_sequence = []
        frame_count = 0

        for dtype, item in sequence:
            # before <IMAGE_START> after <IMAGE_END>
            if dtype == "image":
                if item.get("image_type", "image") == "video":
                    if frame_count % self.video_temporal_conv_size == 0:
                        # start of the frame
                        new_sequence.append(("text", {"text": self.image_start_token, "tag": "mask"}))
                    new_sequence.append(("image", item))
                    frame_count += 1

                    if frame_count % self.video_temporal_conv_size == 0:
                        # end of the frame
                        new_sequence.append(("text", {"text": self.image_end_token, "tag": "mask"}))
                else:
                    # just image
                    new_sequence.append(("image", item))
            elif dtype == "text":
                if item.get("is_asr", False):
                    new_sequence.append(("text", {"text": self.asr_start_token, "tag": "mask"}))
                    new_sequence.append(("text", item))
                    new_sequence.append(("text", {"text": self.asr_end_token, "tag": "mask"}))
                else:
                    # normal text
                    new_sequence.append(("text", item))
        return new_sequence

    def smart_tile_and_sample_frame(self, sequence, max_seq_len=None):
        """smart tile image and sample frames from video"""
        # calc all text tokens
        # calc all image number
        # calc all video frame
        crop_options = {}

        def is_picture_with_integer(text):
            """regex match"""
            pattern = r"^Picture \d*:$"
            return bool(re.match(pattern, text.strip()))

        def is_video_with_integer(text):
            """regex match"""
            pattern = r"^Video \d*:$"
            return bool(re.match(pattern, text.strip()))

        image_count = 0
        frame_count = 0
        frames = []
        video_count = 0
        video_set = set()
        text_list = []
        for dtype, item in sequence:
            if dtype == "text":
                text_list.append(item)

            elif dtype == "image":
                if item.get("image_type", "image") == "video":
                    frame_count += 1
                    frames.append(item)
                    video_set.add(item["video_uid"])
                else:
                    image_count += 1

        video_count = len(video_set)
        text_token_count = 0

        # make smart resize
        special_token_len = len(
            [text["text"] for text in text_list if text.get("text_type", "text") == "special_token"]
        )
        # collect all text
        all_text = [text["text"] for text in text_list if text.get("text_type", "text") == "text"]

        all_text_len = []
        all_picture_num_len = []
        all_video_num_len = []
        for text in all_text:
            if is_picture_with_integer(text):
                all_picture_num_len.append(len(self.tokenizer.encode(text)["input_ids"]))
            elif is_video_with_integer(text):
                all_video_num_len.append(len(self.tokenizer.encode(text)["input_ids"]))
            elif text not in self.special_tokens_info["crop"]:
                all_text_len.append(len(self.tokenizer.encode(text)["input_ids"]))

        special_token_len += sum([2 for i in text_list if i.get("is_asr", False)])

        avg_picture_num_len = sum(all_picture_num_len) / len(all_picture_num_len)

        text_token_count = sum(all_text_len) + special_token_len

        text_token_count = text_token_count - 2 * image_count - 2 * video_count

        # tile upper
        tile_upper = int(math.sqrt(max(self.image_crop_tile_option))) ** 2

        # 1. consider image
        if image_count > 0:
            _max_tile = (
                (max_seq_len - text_token_count - self.max_dec_len) // image_count
                - self.im_prefix_length
                - (avg_picture_num_len + 2)
            ) // (self.im_prefix_length + 1)
            _max_tile = max(1, _max_tile)
            _max_tile = min(int(math.sqrt(_max_tile)) ** 2, tile_upper)
        else:
            _max_tile = tile_upper

        image_crop_tile = _max_tile
        crop_options["image_crop_tile_option"] = []
        crop_options["image_crop_tile_rate"] = []

        def crop_tile_normalize(d):
            return (np.array(d) / np.sum(d)).tolist()

        for crop_opt, crop_rate in zip(self.image_crop_tile_option, self.image_crop_tile_rate):
            if crop_opt >= image_crop_tile:
                continue
            crop_options["image_crop_tile_option"].append(crop_opt)
            crop_options["image_crop_tile_rate"].append(crop_rate)
        crop_options["image_crop_tile_option"].append(image_crop_tile)
        crop_options["image_crop_tile_rate"].append(1 - np.sum(crop_options["image_crop_tile_rate"]))

        text_token_count_left = (
            self.tokenizer.model_max_length
            - text_token_count
            - self.max_dec_len
            - image_count * (image_crop_tile + 1) * (self.im_prefix_length + int(avg_picture_num_len) + 2)
        )

        # 2. consider video
        video_max_tile = max(self.video_crop_tile_option)

        while video_max_tile >= 1:
            video_token_count = self.count_video_cropped_image_tokens(frames, video_max_tile)
            if video_token_count <= text_token_count_left:
                break
            video_max_tile -= 1

        # 3. reset vedio parameter

        if video_max_tile >= max(self.video_crop_tile_option):
            crop_options["video_crop_tile_option"] = self.video_crop_tile_option
            crop_options["video_crop_tile_rate"] = self.video_crop_tile_rate
            pass
        elif video_max_tile > 0 and self.video_fix_crop == -1:
            crop_options["video_crop_tile_option"] = []
            crop_options["video_crop_tile_rate"] = []

            for option, rate in zip(self.video_crop_tile_option, self.video_crop_tile_rate):
                if option < video_max_tile:
                    crop_options["video_crop_tile_option"].append(option)
                    crop_options["video_crop_tile_rate"].append(rate)
            crop_options["video_crop_tile_option"].append(video_max_tile)
            crop_options["video_crop_tile_rate"].append(1 - np.sum(crop_options["video_crop_tile_rate"]))
        else:
            num_frames = len(frames)

            token_per_frame = (
                self.im_prefix_length // self.video_temporal_conv_size
                + self.crop_fn_by_type["video"].crop_image_sep_token_num
            )

            # speical tags in sft
            if not self.is_pretraining:
                token_per_frame += 2

            if self.video_fix_crop != -1:
                token_per_frame *= self.video_fix_crop + 1

            max_frames = text_token_count_left // token_per_frame
            max_frames -= max_frames % self.video_temporal_conv_size

            grouped_frames = self.group_frame_by_video(frames)
            to_delete = num_frames - max_frames

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

            frame_indices_to_remove = []
            for cur_frames, num_frames_to_be_deleted in zip(grouped_frames, num_frames_to_be_deleted_for_each_video):
                frame_indices_to_remove.extend(
                    self.get_frame_indices_to_remove_for_one_video(frames, cur_frames, num_frames_to_be_deleted)
                )

            sequence = self.remove_frame(sequence, frame_indices_to_remove)

            crop_options["video_crop_tile_option"] = [1] if self.video_fix_crop == -1 else [self.video_fix_crop]
            crop_options["video_crop_tile_rate"] = [1.0]
        return crop_options, sequence

    def remove_frame(self, sequence, frame_indices_to_remove):
        """remove frame"""
        frame_indices_to_remove = set(frame_indices_to_remove)
        new_sequence = []
        frame_count = 0

        for dtype, item in sequence:
            if dtype == "text":
                new_sequence.append((dtype, item))
            elif dtype == "image":
                if item.get("image_type", "image") == "video":
                    if frame_count not in frame_indices_to_remove:
                        new_sequence.append((dtype, item))
                    frame_count += 1
                else:
                    new_sequence.append((dtype, item))
        return new_sequence

    def get_frame_indices_to_remove_for_one_video(self, frames, frame_idx, num_frames_to_be_deleted):
        """get frame index to be removed"""
        num_frames = len(frame_idx)
        max_frames = num_frames - num_frames_to_be_deleted

        frame_interval = num_frames // max_frames if num_frames >= max_frames else 1
        frame_indices_selected = frame_idx[::frame_interval]

        if len(frame_indices_selected) > max_frames:
            # random drop
            random_seed = get_uniq_id(get_hashable(frames[frame_idx[0]]["image_url"]))
            with RandomSeedContext(random_seed):
                indices_selected = random.sample(range(1, len(frame_indices_selected) - 1), k=max_frames - 2)
            indices_selected.sort()
            indices_selected = [0] + indices_selected + [len(frame_indices_selected) - 1]
            frame_indices_selected = [frame_indices_selected[i] for i in indices_selected]

        frame_indices_selected = set(frame_indices_selected)
        num_frames_selected = len(frame_indices_selected)
        frame_indices_to_remove = [i for i in frame_idx if i not in frame_indices_selected]

        return frame_indices_to_remove

    def count_video_cropped_image_tokens(self, frames, max_tile):
        """count the number of tokens needed by a given video"""

        padded_frames, _ = self.video_pad_image_info(frames)
        num_cropped_images = 0
        for frame_idx in range(0, len(padded_frames), self.video_temporal_conv_size):
            num_cropped_images += self.crop_fn_by_type["video"].get_num_cropped_images(
                padded_frames[frame_idx : frame_idx + self.video_temporal_conv_size], max_tile=max_tile, min_tile=None
            )
        video_tokens_need = num_cropped_images * self.im_prefix_length // self.video_temporal_conv_size
        return video_tokens_need

    def video_pad(self, sequence):
        """
        pad the video sample to match the temporal_conv_size
        """
        new_sequence = []
        last_video_frame = None
        frame_count = 0
        for data_type, item in sequence:
            if data_type == "text":
                new_sequence.append((data_type, item))
            elif data_type == "image":
                if item.get("image_type", "image") == "video":
                    # deal with video frame
                    item["is_padded_image"] = False
                    if last_video_frame is not None and last_video_frame["video_uid"] != item["video_uid"]:

                        while frame_count % self.video_temporal_conv_size != 0:
                            padded_frame = copy.deepcopy(last_video_frame)
                            padded_frame["is_padded_image"] = True
                            new_sequence.append((data_type, padded_frame))
                            frame_count += 1
                        frame_count = 0
                    last_video_frame = item
                    new_sequence.append((data_type, item))
                    frame_count += 1
                else:
                    # simple image
                    new_sequence.append((data_type, item))

        # final frame
        if last_video_frame is not None:
            while frame_count % self.video_temporal_conv_size != 0:
                padded_frame = copy.deepcopy(last_video_frame)
                padded_frame["is_padded_image"] = True
                new_sequence.append(("image", padded_frame))
                frame_count += 1
            frame_count = 0
        return new_sequence

    def video_pad_image_info(self, all_frames):
        """
        pad the image info to match the temporal_conv_size
        """
        new_frames = copy.deepcopy(all_frames)
        num_padded_images = 0
        for idx in range(len(new_frames)):
            new_frames[idx]["is_padded_image"] = False

        grouped_frames = self.group_frame_by_video(new_frames)

        index_offset = 0
        for frames in grouped_frames:
            if len(frames) % self.video_temporal_conv_size != 0:
                roundup = math.ceil(len(frames) / self.video_temporal_conv_size) * self.video_temporal_conv_size
                num_padded_images = roundup - len(frames)
                tmp = []
                for _ in range(num_padded_images):
                    padded_image = copy.deepcopy(all_frames[frames[-1]])
                    padded_image["is_padded_image"] = True
                    tmp.append(padded_image)
                new_frames = new_frames[: index_offset + len(frames)] + tmp + new_frames[index_offset + len(frames) :]
                index_offset += len(tmp)

            index_offset += len(frames)

        return new_frames, num_padded_images

    def group_frame_by_video(self, frames):
        """group frames by video"""

        cnt = 0
        ret = []
        keys = []
        for key, group in groupby(frames, key=lambda x: x["video_uid"]):
            keys.append(key)
            group_len = len(list(group))
            ret.append(list(range(cnt, group_len + cnt)))
            cnt += group_len

        assert len(keys) == len(set(keys)), f"found duplicate keys: {keys}"
        return ret
