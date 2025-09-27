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


"""
Mapper for input_ids_messaging
"""

import copy
import json
import os
import random
import traceback
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import yaml

from data_processor.steps.input_ids_messaging import (
    data_adaptive,
    data_process,
)
from data_processor.steps.input_ids_messaging.data_utils import add_prompt, merge_list
from data_processor.utils.constant import (
    CUT_FLAG,
    DATASET_TYPE_TO_DATA_TYPE,
    DATASET_TYPE_TO_PROCESS_FN,
    IDS_TYPE_FLAG,
    IMAGE_TYPE_FLAG,
)
from data_processor.utils.io_utils import image_info_2_hash
from data_processor.utils.logger_utils import logger
from data_processor.utils.processor_base import ProcessorBase
from ernie.tokenizer_vl import (
    NOT_FOUND_TOKEN_ID,
    SFT_ASR_END_TOKEN,
    SFT_ASR_START_TOKEN,
    SFT_IMAGE_END_TOKEN,
    SFT_IMAGE_START_TOKEN,
    SFT_VIDEO_END_TOKEN,
    SFT_VIDEO_START_TOKEN,
)


class SlidingWindowsContextManager:
    """
    hack model_max_length in tokenizer to perform no window sliding at eval mode
    """

    def __init__(self, tokenizer, is_training):
        self.hack_max_len = 9999999999999
        self.tokenizer = tokenizer
        self.ori_max_length = -1
        self.is_training = is_training

    def __enter__(self):
        self.ori_max_length = self.tokenizer.model_max_length
        self.tokenizer.model_max_length = self.hack_max_len

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tokenizer.model_max_length = self.ori_max_length


class ExampleToFeature(ProcessorBase):
    """Example To Feature

    Args:
        object (_type_): _description_
    """

    def __init__(
        self,
        tokenizer,
        data_filelist=None,
        corpus_name="default",
        im_prefix_length=64,
        max_seq_length=8192,
        seed=42,
        adaptive_max_imgtoken_option=None,
        adaptive_max_imgtoken_rate=None,
        special_tokens_info=None,
        loc_coordinate_num=1001,
        prompt_dir=None,
        one_sample_in_one_seq=False,
        variable_resolution=False,
        spatial_conv_size=1,
        image_processor=None,
        rope_3d=False,
        video_min_pixels=None,
        video_max_pixels=None,
        drop_untrainble_sample=False,
        chat_template="ernie",
    ):
        super().__init__(None)

        self.data_filelist = data_filelist
        if data_filelist:
            with open(data_filelist, "r", encoding="utf-8") as f:
                datasets_config = yaml.load(f.read(), yaml.FullLoader)["datasets"]

            for dataset_config in datasets_config:
                if dataset_config["name"] == corpus_name:
                    break
            else:
                raise ValueError(f"{corpus_name} not in {data_filelist}")

        else:
            # for utterance, fake a dataset_config
            dataset_config = {
                "name": corpus_name,
                "dataset_type": "default",
                "prompt_file": None,
                "data_setting": "{}",
            }

        # data_info
        self.data_info = {
            "dataset_type": dataset_config["dataset_type"],
            "dataset_name": dataset_config["name"],
        }

        # prompt
        if prompt_dir and dataset_config["prompt_file"]:
            prompt_filepath = os.path.join(prompt_dir, dataset_config["prompt_file"])
            with open(prompt_filepath, encoding="utf-8") as f:
                prompt_list = f.read().strip("\n").split("\n")

        else:
            # for untterance, use empty prompt
            prompt_list = [""]
        self.data_info["prompt_list"] = prompt_list
        self.data_info["prompt_id_list"] = list(range(len(prompt_list)))

        # adaptive
        self.variable_resolution = variable_resolution
        self.adaptive_max_imgtoken_option = adaptive_max_imgtoken_option
        self.adaptive_max_imgtoken_rate = adaptive_max_imgtoken_rate
        self.image_processor = image_processor
        # video
        # adaptive
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels

        data_setting = json.loads(dataset_config.get("data_setting", "{}"))
        self.data_info["augment_type"] = data_setting.get("augment_type", [])
        # interleave fill-image-in-front
        self.data_info["interleave_fiif"] = data_setting.get("interleave_fiif", 0)
        self.one_sample_in_one_seq = data_setting.get(
            "one_sample_in_one_seq", one_sample_in_one_seq
        )
        self.data_info["remove_loc"] = data_setting.get("remove_loc", 0)

        self.data_info["random_resize"] = data_setting.get("random_resize", [1, 1])

        dataset_min_pixels = data_setting.get("min_pixels", None)
        dataset_max_pixels = data_setting.get("max_pixels", None)
        if dataset_min_pixels is not None:
            self.image_processor.set_pixels(
                min_pixels=dataset_min_pixels, msg="ExampleToFeature __init__()"
            )
        if dataset_max_pixels is not None:
            self.image_processor.set_pixels(
                max_pixels=dataset_max_pixels, msg="ExampleToFeature __init__()"
            )
            # adaptive_max_imgtoken
            logger.info(
                "Due to min/max_pixels being specified by data_setting, adaptive_max_imgtoken is set to None."
            )
            self.adaptive_max_imgtoken_option = None
            self.adaptive_max_imgtoken_rate = None
        # video的min_pixels, max_pixels
        dataset_video_min_pixels = data_setting.get("video_min_pixels", None)
        dataset_video_max_pixels = data_setting.get("video_max_pixels", None)
        if dataset_video_min_pixels is not None:
            self.video_min_pixels = dataset_video_min_pixels
        if dataset_video_max_pixels is not None:
            self.video_max_pixels = dataset_video_max_pixels
        self.drop_untrainble_sample = drop_untrainble_sample

        # data_type
        self.data_type = DATASET_TYPE_TO_DATA_TYPE.get(
            self.data_info["dataset_type"], None
        )
        logger.info(f"DATASET_TYPE_TO_DATA_TYPE: {DATASET_TYPE_TO_DATA_TYPE}")
        assert (
            self.data_type is not None
        ), f'dataset_type: {self.data_info["dataset_type"]} not support'

        # tokenizer
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = max_seq_length
        self.max_seq_length = max_seq_length

        self.im_prefix_length = im_prefix_length  # one image 64token

        # placeholder ID
        self.vocab = self.tokenizer.get_vocab()
        self.image_token_id = self.vocab[special_tokens_info["image_placeholder"]]

        # Special Token
        self.token_type_mapping = {
            self.image_token_id: IDS_TYPE_FLAG["image"],
        }

        self.eos_token = self.tokenizer.special_tokens_map.get("eos_token", "</s>")
        self.cls_token = self.tokenizer.special_tokens_map.get("cls_token", "<mask:0>")
        self.sep_token = self.tokenizer.special_tokens_map.get(
            "sep_token", "<|endofprompt|>"
        )

        self.not_found_token_id = NOT_FOUND_TOKEN_ID
        self.image_start_token = SFT_IMAGE_START_TOKEN
        self.image_end_token = SFT_IMAGE_END_TOKEN
        self.video_start_token = SFT_VIDEO_START_TOKEN
        self.video_end_token = SFT_VIDEO_END_TOKEN
        self.asr_start_token = SFT_ASR_START_TOKEN
        self.asr_end_token = SFT_ASR_END_TOKEN
        self.cls_token_id = self.vocab.get(self.cls_token, self.not_found_token_id)
        self.sep_token_id = self.vocab.get(self.sep_token, self.not_found_token_id)
        self.image_start_id = self.vocab.get(
            self.image_start_token, self.not_found_token_id
        )
        self.image_end_id = self.vocab.get(
            self.image_end_token, self.not_found_token_id
        )
        self.video_start_id = self.vocab.get(
            self.video_start_token, self.not_found_token_id
        )
        self.video_end_id = self.vocab.get(
            self.video_end_token, self.not_found_token_id
        )
        self.asr_start_id = self.vocab.get(
            self.asr_start_token, self.not_found_token_id
        )
        self.asr_end_id = self.vocab.get(self.asr_end_token, self.not_found_token_id)
        self.eos_token_id = self.vocab.get(self.eos_token, self.not_found_token_id)
        # system setting start
        self.bosys_token = self.tokenizer.special_tokens_map.get(
            "bosys_token", "<mask:4>"
        )
        # system setting end
        self.eosys_token = self.tokenizer.special_tokens_map.get(
            "eosys_token", "<mask:5>"
        )
        self.bosys_token_id = self.vocab[self.bosys_token]
        self.eosys_token_id = self.vocab[self.eosys_token]

        self.token_type_mapping[self.cls_token_id] = IDS_TYPE_FLAG["text"]
        self.token_type_mapping[self.sep_token_id] = IDS_TYPE_FLAG["text"]
        self.token_type_mapping[self.bosys_token_id] = IDS_TYPE_FLAG["text"]
        self.token_type_mapping[self.eosys_token_id] = IDS_TYPE_FLAG["text"]
        self.token_type_mapping[self.image_start_id] = IDS_TYPE_FLAG["image"]
        self.token_type_mapping[self.image_end_id] = IDS_TYPE_FLAG["image"]
        self.token_type_mapping[self.video_start_id] = IDS_TYPE_FLAG["image"]
        self.token_type_mapping[self.video_end_id] = IDS_TYPE_FLAG["image"]
        self.token_type_mapping[self.asr_start_id] = IDS_TYPE_FLAG["image"]
        self.token_type_mapping[self.asr_end_id] = IDS_TYPE_FLAG["image"]

        if self.not_found_token_id in self.token_type_mapping:
            # TODO How to fix unfound special token.
            logger.warning("Special Token Not Found.")

        self.special_tokens_info = special_tokens_info
        self.loc_coordinate_num = loc_coordinate_num
        self.token_type_mapping = self._get_token_type_mapping()

        # lossmask
        self.tokenizer.ignored_index = -100

        self.prompt_rng = random.Random(seed)

        self.video_temporal_conv_size = 2
        self.image_temporal_conv_size = 1
        self.temporal_conv_size = self.video_temporal_conv_size

        self.spatial_conv_size = spatial_conv_size

        # other
        self.rope_3d = rope_3d
        self.chat_template = chat_template

    def example_to_feature(self, meta, **kwargs):
        """example to feature

        Args:
            meta (dict): one sample
        """

        prompt_id = self.prompt_rng.choice(self.data_info["prompt_id_list"])
        prompt = self.data_info["prompt_list"][prompt_id]
        if not self.is_training or self.data_filelist is None:
            if "image_info" not in meta or len(meta["image_info"]) == 0:
                dataset_type = "default"
                meta["image_info"] = []
            else:
                data_types = [item["image_type"] for item in meta["image_info"]]
                if len(set(data_types)) == 1:
                    dataset_type = (
                        "default" if data_types[0] == "image" else data_types[0]
                    )
        else:
            dataset_type = self.data_info["dataset_type"]
        dataset_name = self.data_info["dataset_name"]
        data_type = DATASET_TYPE_TO_DATA_TYPE.get(dataset_type, None)
        assert data_type is not None, f"Unknow dataset type: {dataset_type}."

        for i, sample_info in enumerate(
            self._example_to_feature(
                deepcopy(meta), prompt, data_type, dataset_type, dataset_name, **kwargs
            )
        ):
            """
            sample_info
            img_num
            global_img_num
            """
            if sample_info is not None:
                feature = {
                    "feature": {
                        "ids": sample_info["ids"],
                        "lossmask": sample_info["lossmask"],
                        "ids_type": sample_info["ids_type"],
                        "image_wise_type": sample_info["image_wise_type"],
                    },
                    "meta": self._meta_format(
                        sample_info["image_info"], sample_info["image_wise_type"]
                    ),
                }
                yield feature

    def _example_to_feature(
        self,
        meta: dict,
        prompt: str,
        data_type: str,
        dataset_type: str,
        dataset_name: str,
        use_prompt: bool = True,
        lazy_image: bool = True,
        max_tile: int = -1,
        max_dec_len: int = 0,
        **kwargs,
    ):
        """

        Args:
            meta (dict): one sample with schema format
            prompt (str): one prompt
            dataset_type (str): see DATA_TYPE_TO_DATASET_TYPE
            dataset_name (str): data name
            use_prompt (bool, optional): use prompt. Defaults to True.
            lazy_image (bool, optional): lazy image. Defaults to True.

        Yields:
            dict: max_seqlen sample
            int: the number of image
        """
        original_im_prefix_length = self.im_prefix_length
        original_image_processor = copy.deepcopy(self.image_processor)

        one_sample_in_one_seq = kwargs.get(
            "one_sample_in_one_seq", self.one_sample_in_one_seq
        )
        try:
            if data_type == "video":
                # self.temporal_conv_size = self.video_temporal_conv_size
                assert self.im_prefix_length % self.temporal_conv_size == 0
                self.im_prefix_length //= self.temporal_conv_size
                self.token_type_mapping[self.image_token_id] = IDS_TYPE_FLAG["video"]

                uid = image_info_2_hash(meta["image_info"][0])
                for idx, img_one in enumerate(meta["image_info"]):
                    img_one["image_type"] = img_one.get("image_type", "video")
                    if img_one["image_type"] != "video":
                        img_one["video_uid"] = img_one.get("video_uid", idx)
                    else:
                        img_one["video_uid"] = img_one.get("video_uid", uid)

            elif data_type == "image":
                # self.temporal_conv_size = self.image_temporal_conv_size
                self.token_type_mapping[self.image_token_id] = IDS_TYPE_FLAG["image"]

                for idx, img_one in enumerate(meta["image_info"]):
                    img_one["image_type"] = img_one.get("image_type", "image")

            """[STEP 0] adaptiver """
            adaptiver = getattr(data_adaptive, "Adaptive")(
                image_processor=self.image_processor,
                spatial_conv_size=self.spatial_conv_size,
                temporal_conv_size=self.temporal_conv_size,
                tokenizer=self.tokenizer,
                special_tokens_info=self.special_tokens_info,
                adaptive_max_imgtoken_option=self.adaptive_max_imgtoken_option,
                adaptive_max_imgtoken_rate=self.adaptive_max_imgtoken_rate,
                video_min_pixels=self.video_min_pixels,
                video_max_pixels=self.video_max_pixels,
                rope_3d=self.rope_3d,
            )

            """[STEP 1] add prompt"""
            if self.is_training:
                meta = add_prompt(deepcopy(meta), use_prompt, prompt, data_type)

            """[STEP 2] process"""
            metas = [meta]
            if dataset_type in DATASET_TYPE_TO_PROCESS_FN:
                obj_process = getattr(
                    data_process, DATASET_TYPE_TO_PROCESS_FN[dataset_type]
                )
                processor = obj_process(
                    temporal_conv_size=self.temporal_conv_size,
                    special_tokens_info=self.special_tokens_info,
                    tokenizer=self.tokenizer,
                    max_seq_len=self.tokenizer.model_max_length,
                    im_prefix_length=self.im_prefix_length,
                    adaptiver=adaptiver,
                    interleave_fiif=self.data_info["interleave_fiif"],
                    remove_loc=self.data_info["remove_loc"],
                    one_sample_in_one_seq=one_sample_in_one_seq,
                    max_dec_len=max_dec_len,
                    is_training=self.is_training,
                    variable_resolution=self.variable_resolution,
                    rope_3d=self.rope_3d,
                )
                meta = processor.process(
                    sample=deepcopy(meta),
                    dataset_name=dataset_name,
                    data_type=data_type,
                    dataset_type=dataset_type,
                    use_prompt=use_prompt,
                    prompt=prompt,
                )
                metas = meta if isinstance(meta, list) else [meta]
            if self.is_training and len(metas) == 0:
                yield None
            if not self.is_training:
                assert len(metas) == 1

            # save original value
            for idx, meta in enumerate(metas):
                if not meta:
                    continue
                """[STEP 3] adaptive"""
                meta = adaptiver.process(sample=deepcopy(meta))

                """[STEP 4] text tokenizer & add placeholder"""
                meta = self._text_tokenization_add_placeholder(
                    deepcopy(meta),
                    dataset_name,
                    data_type,
                    adaptiver,
                    add_eos_token=False,
                )

                """[STEP 5] image_wise type id"""
                meta = self._add_image_wise_type_id(deepcopy(meta), data_type)

                """[STEP 6] add mask"""
                with SlidingWindowsContextManager(self.tokenizer, self.is_training):
                    for one in self._sliding_window(
                        meta, dataset_name, data_type, adaptiver
                    ):
                        if one is not None:
                            assert len(one["ids_type"]) == len(
                                one["ids"]
                            ), "the length of ids_type and ids should be equal."
                            assert (
                                self.not_found_token_id not in one["ids"]
                            ), "unknow special tokens in input_ids."
                            if self.is_training:
                                # Perform SFT Truncatation
                                one = self.sft_sliding_window(one, adaptiver)
                            yield one
        except Exception:
            logger.error(traceback.format_exc())
        finally:
            self.im_prefix_length = original_im_prefix_length
            self.adaptive_max_imgtoken_option = None
            self.image_processor = original_image_processor

    def sft_sliding_window(self, meta, adaptiver):
        """
        Slide SFT Data
        """

        def get_image_num(img_tokens, start_img_idx, image_info):
            if self.variable_resolution:
                cnt = 0
                while img_tokens > 0:
                    img_tokens -= adaptiver.get_num_of_tokens_for_img_one(
                        image_info[start_img_idx + cnt]
                    )
                    cnt += 1
                assert (
                    img_tokens == 0
                ), f"the resulted img_tokens must be 0, but now got {img_tokens}"
                return cnt
            else:
                return img_tokens // self.im_prefix_length

        ids = meta["ids"]
        if len(ids) <= self.max_seq_length:
            return meta
        else:
            truncate_ids = np.array(ids[: self.max_seq_length])
            min_num = 2
            split_token_id = self.cls_token_id
            if self.chat_template == "ernie":
                split_token_id = self.cls_token_id
                min_num = 2
            elif self.chat_template == "ernie_vl":
                split_token_id = self.sep_token_id
                min_num = 1
            else:
                raise NotImplementedError(
                    f"{self.chat_tempelate} is not supported now."
                )

            split_token_num = sum(truncate_ids == split_token_id)
            if split_token_num < min_num:
                raise ValueError(
                    f"The data is too long and cannot be truncated, ids: \
                        {len(ids)}, len images: {len(meta['image_info'])}, \
                        ids == self.image_token_id {sum(np.array(ids) ==self.image_token_id)}"
                )
            indices = np.where(truncate_ids == split_token_id)[0]
            truncate_pos = indices[-1]
            logger.info(f"truncate data from {len(ids)} to {truncate_pos}.")
            meta["ids"] = meta["ids"][: truncate_pos + 1]
            meta["ids_type"] = meta["ids_type"][: truncate_pos + 1]
            meta["lossmask"] = meta["lossmask"][: truncate_pos + 1]

            truncate_image_pos = get_image_num(
                sum(np.array(meta["ids"]) == self.image_token_id),
                0,
                meta["image_info"],
            )
            logger.info(
                f"truncate images from {len(meta['image_info'])} to {truncate_image_pos}."
            )
            meta["image_info"] = meta["image_info"][:truncate_image_pos]
            meta["image_wise_type"] = meta["image_wise_type"][:truncate_image_pos]
            assert len(meta["image_info"]) == truncate_image_pos
            assert len(meta["image_wise_type"]) == truncate_image_pos

        return meta

    def _text_tokenization(self, sample, dataset_name, data_type):
        """text tokenization"""

        no_cut_tag = []
        input_ids = []
        labels = []

        whole_text = "".join([mt["text"] for mt in sample["text_info"]])
        assert whole_text != "", f"[ERROR] {dataset_name} text is empty!"

        for idx, item in enumerate(sample["text_info"]):
            text_type = item.get("text_type", "text")
            if text_type == "special_token":
                cur_tokens = [self.tokenizer.convert_tokens_to_ids(item["text"])]
            else:
                cur_tokens = self.tokenizer.encode(
                    item["text"], add_special_tokens=False, return_attention_mask=False
                )["input_ids"]
            input_ids.append(cur_tokens)

            mask_flag = item.get("tag", "no_mask")
            if mask_flag == "mask":
                labels.append([self.tokenizer.ignored_index] * len(cur_tokens))
            else:
                labels.append(deepcopy(cur_tokens))

            no_cut_tag.append([CUT_FLAG["cut"]] * len(cur_tokens))
        return input_ids, labels, no_cut_tag

    def _image_placeholder(self, sample, input_ids, labels, no_cut_tag, adaptiver):
        """image placeholder"""

        # ensure the order of image and text is the same
        img_order = {}
        reminder = 0

        for item in sample["image_info"]:
            match_id = item["matched_text_index"]

            if self.variable_resolution:
                im_prefix_length = adaptiver.get_num_of_tokens_for_img_one(item)
            else:
                im_prefix_length = self.im_prefix_length

            # bcoz of temporal conv, im_prefix_length of a image may not be a integer.
            # use reminder to accumulate the decimal into the last image
            if im_prefix_length % 1 != 0:
                reminder += im_prefix_length % 1
                if reminder == 1:
                    im_prefix_length += 1
                    reminder = 0
                else:
                    assert reminder < 1

            im_prefix_length = int(im_prefix_length)

            img_placeholder = [self.image_token_id] * im_prefix_length

            ignore_placeholder = [self.tokenizer.ignored_index] * len(img_placeholder)

            not_cut_img_tag = [CUT_FLAG["no_cut"]] * len(img_placeholder)
            if match_id not in img_order:
                img_order[match_id] = {
                    "img_placeholder": [],
                    "ignore_placeholder": [],
                    "not_cut_img_tag": [],
                }

            img_order[match_id]["img_placeholder"] += img_placeholder
            img_order[match_id]["ignore_placeholder"] += ignore_placeholder
            img_order[match_id]["not_cut_img_tag"] += not_cut_img_tag

        for match_id in img_order:
            img_placeholder = img_order[match_id]["img_placeholder"]
            ignore_placeholder = img_order[match_id]["ignore_placeholder"]
            not_cut_img_tag = img_order[match_id]["not_cut_img_tag"]

            input_ids[match_id] = img_placeholder + input_ids[match_id]
            labels[match_id] = ignore_placeholder + labels[match_id]
            no_cut_tag[match_id] = not_cut_img_tag + no_cut_tag[match_id]

        return input_ids, labels, no_cut_tag

    def _text_tokenization_add_placeholder(
        self, sample, dataset_name, data_type, adaptiver, add_eos_token=True
    ):
        """text tokenizer & add placeholder"""

        input_ids = []
        labels = []
        input_ids, labels, no_cut_tag = self._text_tokenization(
            sample, dataset_name, data_type
        )

        input_ids, labels, no_cut_tag = self._image_placeholder(
            sample, input_ids, labels, no_cut_tag, adaptiver
        )

        input_ids = merge_list(input_ids)
        labels = merge_list(labels)
        no_cut_tag = merge_list(no_cut_tag)
        if add_eos_token:
            input_ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)
        no_cut_tag.append(CUT_FLAG["cut"])

        assert len(input_ids) == len(
            labels
        ), f"[ERROR] {dataset_name} tokens:{len(input_ids)} != labels:{len(labels)}"

        sample["input_ids"] = np.array(input_ids, dtype=np.int64)
        sample["labels"] = np.array(labels, dtype=np.int64)
        sample["no_cut_tag"] = np.array(no_cut_tag, dtype=np.int64)
        sample["ids_token_type"] = self._token_type(sample["input_ids"])

        return sample

    def _add_image_wise_type_id(self, sample, data_type):
        """add image wise type id"""
        image_wise_type_id = []
        for img in sample.get("image_info", []):
            if img.get("is_padded_image", False):
                image_wise_type_id.append(IMAGE_TYPE_FLAG["padded_image"])
            else:
                image_wise_type_id.append(IMAGE_TYPE_FLAG[data_type])
        sample["image_wise_type_id"] = np.array(image_wise_type_id)

        return sample

    def _sliding_window(self, sample, dataset_name, data_type, adaptiver):
        """
        sliding_window
        """

        def can_left_cut(is_cut, index):
            """can be cut on the left"""
            if is_cut[index] == CUT_FLAG["cut"]:
                return True
            if index == 0 and is_cut[index] == CUT_FLAG["no_cut"]:
                return True
            if (
                index >= 1
                and is_cut[index] == CUT_FLAG["no_cut"]
                and is_cut[index - 1] == CUT_FLAG["cut"]
            ):
                return True
            return False

        def can_right_cut(is_cut, index):
            """can be cut on the right"""
            if is_cut[index] == CUT_FLAG["no_cut"]:
                return False
            return True

        def iterable_slice(is_cut, max_block_size):
            """iterate slice"""

            left_slice, right_slice = 0, 0
            while left_slice < len(is_cut):
                if can_left_cut(is_cut, left_slice):
                    # find start point on the left
                    right_slice = left_slice

                    cut_right_slice = None
                    while right_slice < min(left_slice + max_block_size, len(is_cut)):
                        if can_right_cut(is_cut, right_slice):
                            cut_right_slice = right_slice

                        # move
                        right_slice += 1

                    if cut_right_slice is not None:
                        yield (left_slice, cut_right_slice + 1)

                    left_slice = right_slice
                    continue
                else:
                    left_slice += 1

        def get_image_num(img_tokens, start_img_idx, image_info):
            if self.variable_resolution:
                cnt = 0
                while img_tokens > 0:
                    img_tokens -= adaptiver.get_num_of_tokens_for_img_one(
                        image_info[start_img_idx + cnt]
                    )
                    cnt += 1
                assert (
                    img_tokens == 0
                ), f"the resulted img_tokens must be 0, but now got {img_tokens}"
                return cnt
            else:
                return img_tokens // self.im_prefix_length

        input_ids, labels = (
            sample["input_ids"],
            sample["labels"],
        )
        no_cut_tag, ids_token_type = sample["no_cut_tag"], sample["ids_token_type"]

        # 0 for mask, 1 for no mask
        lossmask = (labels != self.tokenizer.ignored_index).astype("int8")

        for s_idx, e_idx in iterable_slice(no_cut_tag, self.tokenizer.model_max_length):
            img_all_token = (input_ids[:s_idx] == self.image_token_id).sum()

            start_img_idx = get_image_num(img_all_token, 0, sample["image_info"])

            """check content of sample"""
            sample_ids = input_ids[s_idx:e_idx]
            img_all_token = (sample_ids == self.image_token_id).sum()
            img_num = get_image_num(img_all_token, start_img_idx, sample["image_info"])

            pure_text = img_num == 0
            pure_image = img_all_token == len(sample_ids)
            no_trainable_sample = lossmask[s_idx:e_idx].sum() / (e_idx - s_idx) <= 0.0
            no_trainable_sample = no_trainable_sample and self.is_training

            if pure_text or pure_image or no_trainable_sample:
                if self.is_training and self.drop_untrainble_sample:
                    yield None
                else:
                    sample_info = OrderedDict(
                        ids=sample_ids.tolist(),
                        lossmask=lossmask[s_idx:e_idx].tolist(),
                        ids_type=ids_token_type[s_idx:e_idx].tolist(),
                        image_info=[],
                        image_wise_type=[],
                    )
                    yield sample_info
            else:
                # mm sample
                image_info = sample["image_info"][
                    start_img_idx : start_img_idx + img_num
                ]
                image_wise_type = sample["image_wise_type_id"][
                    start_img_idx : start_img_idx + img_num
                ]

                assert img_num == len(image_info)
                assert img_num == image_wise_type.shape[0]

                assert (
                    len([i for i in image_info if i["image_type"] == "video"])
                    % self.temporal_conv_size
                    == 0
                ), (
                    f"[ERROR] {dataset_name} vedio data have wrong information, len(image_info): {len(image_info)},"
                    + f"self.temporal_conv_size: {self.temporal_conv_size}"
                )

                if self.variable_resolution:
                    assert (
                        sample_ids == self.image_token_id
                    ).sum() == adaptiver.get_images_token_num(image_info), (
                        "(sample_ids == self.image_token_id).sum(): "
                        + f"{(sample_ids == self.image_token_id).sum()}, "
                        + "adaptiver.get_images_token_num(image_info): "
                        + f"{adaptiver.get_images_token_num(image_info)}"
                    )

                assert len(image_info) == len(
                    image_wise_type
                ), f"len(image_info) {len(image_info)} != len(image_wise_type) {len(image_wise_type)}"

                sample_info = OrderedDict(
                    ids=sample_ids.tolist(),
                    lossmask=lossmask[s_idx:e_idx].tolist(),
                    ids_type=ids_token_type[s_idx:e_idx].tolist(),
                    image_info=image_info,
                    image_wise_type=image_wise_type.tolist(),
                )
                yield sample_info

    def _meta_format(self, metas, image_type):
        """
        meta format
        """
        if len(metas) == 0:
            return []
        global_img_info = []
        restart_frame_local = True

        location_idx = 0
        index_now = 0
        while index_now < len(metas):
            if image_type[index_now] == 0:  # is image
                conv_size = self.image_temporal_conv_size
            else:  # is padded image or video
                conv_size = self.video_temporal_conv_size

            if restart_frame_local:
                frame_local = [
                    {"upscale_image_size": None, "position": [], "location": []}
                    for _ in range(conv_size)
                ]
                restart_frame_local = False

            frame_global = []
            for j in range(conv_size):
                one = metas[index_now + j]

                img_type = one.get("img_type", "global")

                if img_type == "local":
                    frame_local[j]["upscale_image_size"] = one["args_fn"][
                        "upscale_image_size"
                    ]
                    frame_local[j]["position"].append(one["args_fn"]["position"])
                    frame_local[j]["location"].append(location_idx)
                    location_idx += 1

                else:
                    # collect image
                    frame_global.append(one)

            index_now += conv_size

            if len(frame_global) > 0:
                if len(frame_local) > 0:
                    for local_imgs, global_img in zip(frame_local, frame_global):
                        global_img["args_fn"] = {
                            "upscale_image_size": local_imgs["upscale_image_size"],
                            "positions": local_imgs["position"] + [[]],
                            "location": local_imgs["location"] + [location_idx],
                        }
                        global_img_info.append(global_img)

                        positions_num = len(global_img["args_fn"]["positions"])
                        location_num = len(global_img["args_fn"]["location"])
                        assert (
                            positions_num == location_num
                        ), f"[ERROR] meta format positions {positions_num} != location {location_num}"

                        location_idx += 1
                else:
                    for global_img in frame_global:
                        global_img["args_fn"] = {
                            "upscale_image_size": None,
                            "positions": [[]],
                            "location": [location_idx],
                        }
                        global_img_info.append(global_img)

                        location_idx += 1

                # reset frame local
                restart_frame_local = True

        image_num = 0
        for img in global_img_info:
            image_num += len(img["args_fn"]["positions"])

        return global_img_info

    def _get_token_type_mapping(self):
        """get token type mapping"""

        return self.token_type_mapping

    def _token_type(self, input_ids):
        """get token type"""

        ids_token_type = np.zeros_like(input_ids)

        for i in range(len(ids_token_type)):
            ids = input_ids[i]
            if ids in self.token_type_mapping:
                ids_token_type[i] = self.token_type_mapping[ids]

        return ids_token_type

    def process(self, *args, **kwargs):
        """
        main process function
        """
        return self.example_to_feature(*args, **kwargs)
