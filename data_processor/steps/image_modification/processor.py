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
This file contains the definition of the VisionProcessor class
"""

import copy
import json
import logging
import os
import random
import sys
from collections import OrderedDict, namedtuple
from itertools import groupby

import numpy as np

base_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_dir, "../../")))


from data_processor.steps.image_modification.render_timestamp import render_frame_timestamp
from data_processor.utils.constant import DATATYPE_2_ID, IDTYPES_2_ID, IMAGETYPES_2_ID
from data_processor.utils.image_enhance import ImageEnhance
from data_processor.utils.logger_utils import logger
from data_processor.utils.processor_base import ProcessorBase
from ernie.tokenizer_vl import special_tokens_info

try:
    from data_processor.utils.io_utils import get_downloadable_image
except Exception as e:
    logger.warning(f" decord not found: {e}")
    get_downloadable_image = None

from paddleformers.transformers.image_utils import ChannelDimension

VisionExample = namedtuple(
    "Example",
    [
        "meta",
        "ids",
        "sids",
        "task",
        "lossmask",
        "src",
        "part",
        "info",
        "name",
        "data_type",
        "token_type_ids",
        "image_type_ids",
    ],
)

Example = namedtuple("Example", ["src", "ids", "lossmask", "token_type_ids"])


logger = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.WARNING)


def crop_fn(image, upscale_image_size, crop_positions, image_type_id, timestamp, render_timestamp=False):
    """pickle_fn_crop

    image pillow image
    crop_positions [[left, top, right, bottom], ...]
    """

    if upscale_image_size is not None and image.size != upscale_image_size:
        image = image.resize(upscale_image_size)

    # add timestamp
    if render_timestamp and image_type_id == IMAGETYPES_2_ID["video"]:
        assert timestamp >= 0, f"When render timestamp is true，meta need timestamp, timestamp is : {timestamp}"
        image = render_frame_timestamp(image, timestamp)
    crop_imgs = []
    for crop_position in crop_positions:

        if len(crop_position) != 0:
            crop_imgs.append(image.crop(crop_position))
        else:
            crop_imgs.append(image)
    return crop_imgs


class ImageModificationProcessor(ProcessorBase):
    """
    ImageModificationProcessor
    """

    def __init__(self, args, tokenizer, image_preprocess):
        """
        init
        """
        super().__init__(args)
        self.tokenizer = tokenizer
        self.image_token_len = args.image_token_len
        self.image_preprocess = image_preprocess
        self.image_dtype = args.image_dtype
        self.render_timestamp = args.render_timestamp
        self.variable_resolution = True
        self.rope_3d = True

        vocab = self.tokenizer.get_vocab()
        self.im_patch_id = vocab[special_tokens_info["image_placeholder"]]
        self.eos_token = self.tokenizer.special_tokens_map.get("eos_token", "</s>")
        self.cls_token = self.tokenizer.special_tokens_map.get("cls_token", "<mask:0>")
        self.sep_token = self.tokenizer.special_tokens_map.get("sep_token", "<|endofprompt|>")
        self.eos_token_id = vocab[self.eos_token]
        self.cls_token_id = vocab[self.cls_token]
        self.sep_token_id = vocab[self.sep_token]
        self.sft_shift_by_one = args.sft_shift_by_one
        self.chat_template = "deepseek"
        self.should_shift_by_one = self.is_training and (self.is_pretraining or self.sft_shift_by_one)

    def lm_example_to_feature(self, example):
        """
        lm example to feature
        """
        if example.lossmask is not None:
            labels = [self.tokenizer.ignored_index if j == 0 else i for i, j in zip(example.ids, example.lossmask)]
        else:
            labels = example.ids
        input_ids = np.array(example.ids, dtype=np.int64)
        labels = np.array(labels, dtype=np.int64)
        if not self.is_pretraining:
            replace_token_id = self.cls_token_id
            if self.chat_template == "ernie":
                replace_token_id = self.cls_token_id
            elif self.chat_template == "deepseek":
                replace_token_id = self.sep_token_id
            else:
                raise NotImplementedError(f"{self.chat_template} is not supported now.")
            # the label of cls_token is eos_token in sft
            labels[labels == replace_token_id] = self.eos_token_id
        features = OrderedDict(
            src_id=example.src,
            images=None,
            input_ids=input_ids[:-1] if self.should_shift_by_one else input_ids,
            labels=labels[1:] if self.should_shift_by_one else labels,
            data_type=DATATYPE_2_ID["lm"],
            token_type_ids=np.array(np.zeros_like(example.ids) + IDTYPES_2_ID["text"], dtype="int64"),
            data_not_valid=np.array([0], dtype="float32"),
            image_type_ids=None,
        )
        return features

    def image_handling_for_crop(self, example, download_fn):
        """
        image handling for crop
        """
        images = []
        for meta in example.meta:
            if isinstance(meta, np.ndarray):
                meta = json.loads(meta.tobytes().decode())
            # get placeholder for the current meta first
            num_images = 0

            for img_one in meta:
                num_images += len(img_one["args_crop_fn"]["crop_positions"])

            # do the actual cropping
            images_for_current_meta = [None for _ in range(num_images)]
            offset_images = 0
            for img_index, img_one in enumerate(meta):
                img = download_fn(
                    img_one["image_url"],
                    need_exif_info=False,
                )[0]

                if "image_enhance_augs" in img_one:
                    img = ImageEnhance.apply_effect(img, img_one["image_enhance_augs"])

                assert len(img_one["args_crop_fn"]["crop_positions"]) == len(img_one["args_crop_fn"]["location"]), (
                    f'len(img_one["args_crop_fn"]["crop_positions"]): '
                    f'{len(img_one["args_crop_fn"]["crop_positions"])},'
                    f'len(img_one["args_crop_fn"]["location"]): '
                    f'{len(img_one["args_crop_fn"]["location"])}'
                )

                crop_img = crop_fn(
                    img,
                    upscale_image_size=img_one["args_crop_fn"]["upscale_image_size"],
                    crop_positions=img_one["args_crop_fn"]["crop_positions"],
                    image_type_id=example.image_type_ids[offset_images],
                    timestamp=img_one.get("time_stamp", -1),
                    render_timestamp=self.render_timestamp,
                )

                assert len(crop_img) == len(img_one["args_crop_fn"]["location"]), (
                    f"len(crop_img): {len(crop_img)}, "
                    f'len(img_one["args_crop_fn"]["location"]): {len(img_one["args_crop_fn"]["location"])}, '
                    f'img_one["args_crop_fn"]["crop_positions"]: {img_one["args_crop_fn"]["crop_positions"]},'
                    f'upscale_image_size: {img_one["args_crop_fn"]["upscale_image_size"]}'
                )
                for im, loc in zip(crop_img, img_one["args_crop_fn"]["location"]):
                    images_for_current_meta[loc] = im
                offset_images += len(img_one["args_crop_fn"]["crop_positions"])
            images.extend(images_for_current_meta)

        if len(images) > 0:
            # rescale in modeli
            images = self.image_preprocess.preprocess(
                images,
                return_tensors="np",
                do_normalize=False,
                do_rescale=False,
            )["pixel_values"]
            # logger.info(f'images after pre:{[np.array(i).shape for i in images]}')
            # images = [np.array(i).transpose([2,0,1]) for i in images]
            images = images.astype(self.image_dtype)
        else:
            images = None

        return images

    def get_rope_index(
        self,
        spatial_merge_size,
        temporal_merge_size,
        image_token_id,
        video_token_id,
        vision_start_indices,
        input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
    ):
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`np.Array` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by
                default should you provide it.
            image_grid_thw (`np.Array` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`np.Array` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`np.Array` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`np.Array` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`np.Array` of shape `(batch_size)`)
        """
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = np.ones([3, input_ids.shape[0], input_ids.shape[1]], dtype=input_ids.dtype)
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                # TODO: CUDA error in some paddle version
                if attention_mask is not None:
                    input_ids = np.array(input_ids[attention_mask[i] == 1])
                image_nums, video_nums = 0, 0
                vision_start_indices_tmp = vision_start_indices[i]
                vision_tokens = input_ids[vision_start_indices_tmp]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item() if t.item() == 1 else t.item() // temporal_merge_size,
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0

                    llm_pos_ids_list.append(np.arange(text_len).reshape([1, -1]).repeat(3, axis=0) + st_idx)

                    t_index = np.tile(np.arange(llm_grid_t).reshape([-1, 1]), ([1, llm_grid_h * llm_grid_w])).flatten()
                    h_index = np.tile(
                        np.arange(llm_grid_h).reshape([1, -1, 1]), ([llm_grid_t, 1, llm_grid_w])
                    ).flatten()
                    w_index = np.tile(
                        np.arange(llm_grid_w).reshape([1, 1, -1]), ([llm_grid_t, llm_grid_h, 1])
                    ).flatten()

                    llm_pos_ids_list.append(np.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(np.arange(text_len).reshape([1, -1]).repeat(3, axis=0) + st_idx)

                llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape([3, -1])
                if False:  # _IS_NPU:
                    # NOTE: bool + id的混合索引赋值未生效，暂时绕过
                    bool_indices = (attention_mask[i] == 1).unsqueeze(0).tile([position_ids.shape[0], 1])
                    position_ids[:, i] = np.index_put(position_ids[:, i], [bool_indices], llm_positions.reshape([-1]))
                else:
                    position_ids[..., i, attention_mask[i] == 1] = llm_positions
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = np.expand_dims(np.array(mrope_position_deltas), 1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = np.asarray(attention_mask, dtype="int64").cumsum(-1) - 1
                position_ids.masked_fill_(mask=attention_mask == 0, value=1)
                position_ids = position_ids.unsqueeze(0).tile([3, 1, 1])
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = np.arange(input_ids.shape[1]).reshape([1, 1, -1]).tile([3, input_ids.shape[0], 1])
                mrope_position_deltas = np.zeros([input_ids.shape[0], 1], dtype=input_ids.dtype)
            return position_ids, mrope_position_deltas

    def position_ids_for_rope_3d(self, feature):
        """
        get position id for 3d rope
        """
        if feature.get("images", None) is None or len(feature["images"]) == 0:
            position_ids = np.repeat(np.arange(feature["input_ids"].shape[0])[:, np.newaxis], 3, axis=1)
            feature["position_ids"] = position_ids
            return feature

        input_ids = copy.deepcopy(feature["input_ids"])
        grid_thw = feature["grid_thw"]
        # TODO: break if not training
        token_type_ids = feature["token_type_ids"][:-1] if self.should_shift_by_one else feature["token_type_ids"]
        image_type_ids = feature["image_type_ids"]

        fake_image_token_id = -10000
        fake_video_token_id = -20000

        input_ids[np.bitwise_and(token_type_ids == IDTYPES_2_ID["image"], input_ids == self.im_patch_id)] = (
            fake_image_token_id
        )
        input_ids[np.bitwise_and(token_type_ids == IDTYPES_2_ID["video"], input_ids == self.im_patch_id)] = (
            fake_video_token_id
        )

        visual_token_indices = np.nonzero(feature["input_ids"] == self.im_patch_id)  # [xxx, 1] -> [xxx]
        visual_token_indices = np.stack(visual_token_indices, axis=0).flatten()

        vision_start_indices = []
        image_grid_thw = []
        video_grid_thw = []
        index_of_visual_token_indices = 0
        index_of_image_type_ids = 0

        for cur_grid_thw in grid_thw:
            vision_start_indices.append(visual_token_indices[index_of_visual_token_indices])
            index_of_visual_token_indices += (
                cur_grid_thw[0] * cur_grid_thw[1] * cur_grid_thw[2] // (self.image_preprocess.merge_size**2)
            )
            if image_type_ids[index_of_image_type_ids] == IMAGETYPES_2_ID["image"]:
                image_grid_thw.append(cur_grid_thw)
            else:
                video_grid_thw.append(cur_grid_thw)
                index_of_visual_token_indices //= self.image_preprocess.temporal_conv_size

        position_ids, position_ids_delta = self.get_rope_index(
            self.image_preprocess.merge_size,
            self.image_preprocess.temporal_conv_size,
            fake_image_token_id,
            fake_video_token_id,
            np.array([vision_start_indices]),
            np.array([input_ids]),
            image_grid_thw,
            video_grid_thw,
            attention_mask=np.ones([1, input_ids.shape[-1]]),
        )
        position_ids = np.squeeze(position_ids, axis=1).transpose([1, 0])  # [seqlen, 3]

        feature["position_ids"] = position_ids

        return feature

    def image_handling_for_adaptive(self, example, download_fn):
        """
        adaptive  image handling
        """
        pixel_values_list = []
        grid_thw_list = []

        # treat all padded image as normal frames
        image_type_ids = np.array(example.image_type_ids)
        image_type_ids[image_type_ids == IMAGETYPES_2_ID["padded_image"]] = IMAGETYPES_2_ID["video"]
        image_type_ids = image_type_ids.tolist()

        metas = []
        for meta in example.meta:
            if isinstance(meta, np.ndarray):
                meta = json.loads(meta.tobytes().decode())
            metas.extend(meta)

        assert len(example.image_type_ids) == len(
            metas
        ), f"len(image_type_ids) {len(example.image_type_ids)} != len(metas) {len(metas)}\
            , image_type_ids: {example.image_type_ids}, metas: {metas}"

        for key, group in groupby(zip(image_type_ids, metas), key=lambda x: x[0]):
            imgs = []
            predetermined_grid_thw = []
            uids = []
            for img_one in group:
                img_one = img_one[1]

                img = download_fn(
                    img_one["image_url"],
                    need_exif_info=False,
                )[0]

                random_resize_factor = img_one.get("random_resize_factor", 1)
                image_enhance_augs = img_one.get("image_enhance_augs", None)
                img = ImageEnhance.apply_effect(img, image_enhance_augs, random_resize_factor)

                img = crop_fn(
                    img,
                    upscale_image_size=None,
                    crop_positions=[[]],
                    image_type_id=key,
                    timestamp=img_one.get("time_stamp", -1),
                    render_timestamp=self.render_timestamp,
                )
                assert len(img) == 1
                img = img[0]

                imgs.append(img.convert("RGB"))
                predetermined_grid_thw.append([img_one.get("grid_h", -1), img_one.get("grid_w", -1)])
                uids.append(img_one.get("video_uid", random.random()))

            predetermined_grid_thw = np.array(predetermined_grid_thw)
            if predetermined_grid_thw[0][0] < -1:
                predetermined_grid_thw = None
            if key == IMAGETYPES_2_ID["image"]:
                ret = self.image_preprocess.preprocess(
                    images=imgs,
                    videos=None,
                    do_normalize=False,
                    do_rescale=False,
                    predetermined_grid_thw=predetermined_grid_thw,
                    do_convert_rgb=True,
                    input_data_format=ChannelDimension.LAST,
                )
                pixel_values = ret["pixel_values"]
                grid_thw = ret["image_grid_thw"]

                pixel_values_list.append(pixel_values)
                grid_thw_list.append(grid_thw)

            elif key == IMAGETYPES_2_ID["video"]:
                # TODO: liuweixin, use original size of frames to seperate videos for now.
                # will instead use video_uid to distinguish videos in the near future.
                cnt = 0
                for uid, group in groupby(zip(uids, imgs), key=lambda x: x[0]):
                    grouped_imgs = [i[1] for i in group]
                    if predetermined_grid_thw is not None:
                        cur_predetermined_grid_thw = predetermined_grid_thw[cnt : cnt + len(grouped_imgs)]
                    else:
                        cur_predetermined_grid_thw = None
                    cnt += len(grouped_imgs)
                    ret = self.image_preprocess.preprocess(
                        images=None,
                        videos=np.stack([np.array(img.convert("RGB")) for img in grouped_imgs], axis=0),
                        do_normalize=False,
                        do_rescale=False,
                        predetermined_grid_thw=cur_predetermined_grid_thw,
                        do_convert_rgb=True,
                        input_data_format=ChannelDimension.LAST,
                    )
                    pixel_values = ret["pixel_values_videos"]
                    grid_thw = ret["video_grid_thw"]

                    pixel_values_list.append(pixel_values)
                    grid_thw_list.append(grid_thw)
            else:
                raise ValueError(f"encounter unsupported image type! {key}")

        pixel_values_list = np.concatenate(pixel_values_list, axis=0)  # .astype(self.image_dtype)
        grid_thw_list = np.concatenate(grid_thw_list, axis=0)

        return pixel_values_list, grid_thw_list

    def mm_example_to_feature(self, example, download_fn=None):
        """
        convert mm example to feature
        """
        download_fn = download_fn or get_downloadable_image
        try:
            assert isinstance(example, VisionExample), " only support VisionExample"
            if self.variable_resolution:
                images, grid_thw = self.image_handling_for_adaptive(example, download_fn=download_fn)
            else:
                images = self.image_handling_for_crop(example, download_fn=download_fn)
                grid_thw = None
            input_ids = np.array(example.ids, dtype=np.int64)

            token_type_ids = np.array(example.token_type_ids, dtype=np.int64)
            image_type_ids = np.array(example.image_type_ids, dtype=np.int64)
            # TODO: confirm this
            image_type_ids[image_type_ids == IMAGETYPES_2_ID["padded_image"]] = IMAGETYPES_2_ID["video"]

            if example.lossmask is not None:
                labels = np.array(
                    [self.tokenizer.ignored_index if j == 0 else i for i, j in zip(example.ids, example.lossmask)],
                    dtype=np.int64,
                )

            else:
                labels = input_ids
            if not self.is_pretraining:
                replace_token_id = self.cls_token_id
                if self.chat_template == "ernie":
                    replace_token_id = self.cls_token_id
                elif self.chat_template == "deepseek":
                    replace_token_id = self.sep_token_id
                else:
                    raise NotImplementedError(f"{self.chat_template} is not supported now.")
                # the label of cls_token is eos_token in sft
                labels[labels == replace_token_id] = self.eos_token_id

            features = OrderedDict(
                src_id=example.src,
                images=images,
                input_ids=input_ids[:-1] if self.should_shift_by_one else input_ids,
                labels=labels[1:] if self.should_shift_by_one else labels,
                data_type=DATATYPE_2_ID["mm"] if images is not None else DATATYPE_2_ID["lm"],
                token_type_ids=token_type_ids,
                image_type_ids=image_type_ids,
                data_not_valid=0,
                grid_thw=grid_thw,
            )
        except Exception as e:
            logger.exception(e)
            if not self.is_training:
                raise e
            if self.variable_resolution:
                images = np.zeros(
                    [4, 3 * (self.image_preprocess.patch_size**2) * self.image_preprocess.temporal_conv_size],
                    dtype=self.image_dtype,
                )
                grid_thw = np.array([[1, 2, 2]])
                input_ids = np.array([self.im_patch_id] * 1 + [1])
                labels = np.ones_like([self.im_patch_id] * 1 + [1]) * self.tokenizer.ignored_index
                token_type_ids = np.array(1 * [IDTYPES_2_ID["image"]] + 1 * [IDTYPES_2_ID["text"]], dtype="int64")
                image_type_ids = np.array(1 * [IMAGETYPES_2_ID["image"]])
                features = OrderedDict(
                    src_id=example.src,
                    images=images,
                    input_ids=input_ids,
                    labels=labels,
                    data_type=DATATYPE_2_ID["mm"],
                    token_type_ids=token_type_ids,
                    image_type_ids=image_type_ids,
                    data_not_valid=1,
                    grid_thw=grid_thw,
                )
            else:
                images = np.zeros(
                    [
                        1,
                        3,
                        self.image_preprocess.crop_size["height"],
                        self.image_preprocess.crop_size["width"],
                    ],
                    dtype=self.image_dtype,
                )
                input_ids = np.array([self.im_patch_id] * self.image_token_len + [1])
                labels = np.ones_like([self.im_patch_id] * self.image_token_len + [1]) * self.tokenizer.ignored_index
                token_type_ids = np.array(
                    self.image_token_len * [IDTYPES_2_ID["image"]] + 2 * [IDTYPES_2_ID["text"]], dtype="int64"
                )
                image_type_ids = np.array(1 * [IMAGETYPES_2_ID["image"]])
                features = OrderedDict(
                    src_id=example.src,
                    images=images,
                    input_ids=input_ids,
                    labels=labels,
                    data_type=DATATYPE_2_ID["mm"],
                    token_type_ids=token_type_ids,
                    image_type_ids=image_type_ids,
                    data_not_valid=1,
                    grid_thw=None,
                )
        finally:
            pass

        return features

    def merge_consecutive_cls_token(self, ids, labels, token_type_ids):
        """Merge consecutive CLS tokens into one"""
        cls_token = self.tokenizer.special_tokens_map.get("cls_token", "<mask:0>")
        cls_token_id = self.tokenizer.get_vocab()[cls_token]
        sep_token = self.tokenizer.special_tokens_map.get("sep_token", "<|endofprompt|>")
        sep_token_id = self.tokenizer.get_vocab()[sep_token]

        ids = np.array(ids)
        labels = np.array(labels)
        token_type_ids = np.array(token_type_ids)

        if not (len(ids) == len(labels) == len(token_type_ids) - 1):
            raise ValueError("a, b, c must have the same length")

        # 找出哪些位置不是重复的 cls_token_id（只保留第一个）
        mask = np.ones_like(ids, dtype=bool)
        if self.chat_template == "ernie":
            mask[1:] = ~((ids[1:] == cls_token_id) & (ids[:-1] == cls_token_id))
        elif self.chat_template == "deepseek":
            mask[1:] = ~((ids[1:] == cls_token_id) & (ids[:-1] == sep_token_id))
        else:
            raise NotImplementedError(f"{self.chat_template} is not supported now.")

        merged_ids = ids[mask]
        merged_labels = labels[mask]
        merged_token_type_ids = token_type_ids[:-1][mask]
        merged_token_type_ids = np.append(merged_token_type_ids, token_type_ids[-1])  # 补上最后一个

        return merged_ids, merged_labels, merged_token_type_ids

    def fill_empty_field_in_features(self, features):
        """
        防止不同模态处理数据时，忘记加其他模态字段，这里统一补齐。
        """
        new_features = OrderedDict(
            src_id=features["src_id"],  # 必有字段
            input_ids=features["input_ids"],  # 必有字段
            labels=features["labels"],  # 必有字段
            data_type=features["data_type"],  # 必有字段
            token_type_ids=features["token_type_ids"],  # 必有字段
            images=features.get("images", None),
            image_type_ids=features.get("image_type_ids", None),
            data_not_valid=features.get(
                "data_not_valid", np.array([1], dtype="float32")
            ),  # 如果不存在该字段，默认为无效样本，防止一些异常情况。
            grid_thw=features.get("grid_thw", None),
            position_ids=features.get("position_ids", None),
        )
        return new_features

    def json_2_example(self, data):
        """
        从json格式转成example
        """

        def _vision_key_formatting(data):
            ret = {}
            ret["meta"] = data["meta"]
            ret["ids"] = data["ids"] if "ids" in data else data["ds16"]
            ret["sids"] = None
            ret["task"] = "mm"
            ret["src"] = data.get("part", -1)  # dummy
            ret["part"] = data.get("part", -1)  # dummy
            ret["lossmask"] = data["lossmask"] if "lossmask" in data else data["ds16_lossmask"]
            ret["info"] = -1  # dummy
            ret["name"] = "dummy"  # dummy
            ret["data_type"] = DATATYPE_2_ID["mm"]
            ret["token_type_ids"] = (
                data["token_type_ids"] if "token_type_ids" in data else data["ds16_tokenwise_type_id"]
            )
            ret["image_type_ids"] = (
                data["image_type_ids"] if "image_type_ids" in data else data["ds16_imagewise_type_id"]
            )

            return ret

        def _text_key_formatting(data):
            ret = {}
            ret["ids"] = data["ids"] if "ids" in data else data["ds16"]
            ret["src"] = data.get("part", -1)  # dummy
            ret["lossmask"] = data["lossmask"] if "lossmask" in data else data["ds16_lossmask"]
            ret["token_type_ids"] = (
                data["token_type_ids"] if "token_type_ids" in data else data["ds16_tokenwise_type_id"]
            )

            return ret

        # def _lm_key_formatting(data):
        assert isinstance(data, dict)
        if "image_type_ids" in data or "ds16_imagewise_type_id" in data:
            data = _vision_key_formatting(data)
            ExampleClass = VisionExample
        else:
            # TODO: fix this
            # assert 0, f"not support yet"
            data = _text_key_formatting(data)
            ExampleClass = Example

        return ExampleClass(**data)

    def get_data_type(self, data):
        """
        放回这个数据的datatype
        """
        if isinstance(data, dict):
            if "data_type" in data:
                return data["data_type"]
            elif "ds16_imagewise_type_id" in data:
                return DATATYPE_2_ID["mm"]
            else:
                return DATATYPE_2_ID["lm"]
        elif isinstance(data, Example):
            return DATATYPE_2_ID["lm"]
        elif isinstance(data, VisionExample):
            return DATATYPE_2_ID["mm"]
        else:
            return getattr(data, "data_type", DATATYPE_2_ID["lm"])

    def process(self, data, **kwargs):
        """
        process
        """
        # update should shift by one, To-Fix: do not this self
        self.should_shift_by_one = self.is_training and (self.is_pretraining or self.sft_shift_by_one)
        # assert example.labels is None
        data_type = self.get_data_type(data)
        if isinstance(data, dict):
            example = self.json_2_example(data)
        else:
            example = data
        if data_type == DATATYPE_2_ID["lm"]:
            features = self.lm_example_to_feature(example)
        elif data_type == DATATYPE_2_ID["mm"]:
            features = self.mm_example_to_feature(example, kwargs.get("download_fn", None))
        else:
            raise RuntimeError(f"unknown data_type: {data_type}")
        if (
            features["data_not_valid"] == 0
            and self.is_training
            and not self.is_pretraining
            and self.should_shift_by_one
        ):
            # 去掉连续的<mask:0>
            (features["input_ids"], features["labels"], features["token_type_ids"]) = self.merge_consecutive_cls_token(
                features["input_ids"], features["labels"], features["token_type_ids"]
            )

        if self.rope_3d:
            features = self.position_ids_for_rope_3d(features)

        features = self.fill_empty_field_in_features(features)
        return features
