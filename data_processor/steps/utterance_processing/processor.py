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
utterance_processor

"""
import io
import uuid
from collections import defaultdict
from typing import List

from PIL import Image

from data_processor.utils.format_utils import get_format_type
from data_processor.utils.io_utils import RAW_IMAGE_DIR, RAW_VIDEO_DIR, get_downloadable
from data_processor.utils.processor_base import ProcessorBase
from data_processor.utils.video_utils import VideoReaderWrapper
from ernie.tokenizer_vl import SFT_VIDEO_END_TOKEN, SFT_VIDEO_START_TOKEN


class UtteranceProcessor(ProcessorBase):
    """
    utterance_processor
    """

    def __init__(self, args, tokenizer):
        """
            Args:
            args (ArgumentParser): parser object containing arguments for the model.
            tokenizer (Tokenizer): tokenizer object used to encode input text.

        Returns:
            None.

        Initializes a RoleBasedTokenizer object with the given tokenizer and args.
        """
        super().__init__(args)
        self.tokenizer = tokenizer
        self.eos_token = None
        self.cls_token = None
        self.sep_token = None
        self.use_pic_id = True
        self.image_start_token = self.tokenizer.special_tokens_map.get(
            "image_start_id", "<|IMAGE_START|>"
        )
        self.image_end_token = self.tokenizer.special_tokens_map.get(
            "image_end_id", "<|IMAGE_END|>"
        )
        self.video_start_token = SFT_VIDEO_START_TOKEN
        self.video_end_token = SFT_VIDEO_END_TOKEN
        self.eos_token = self.tokenizer.special_tokens_map.get("eos_token", "</s>")
        self.cls_token = self.tokenizer.special_tokens_map.get("cls_token", "<mask:0>")
        self.sep_token = self.tokenizer.special_tokens_map.get(
            "sep_token", "<|endofprompt|>'"
        )
        self.role_2_speical_token = {"user": self.cls_token, "bot": self.sep_token}
        self.bosys_token = self.tokenizer.special_tokens_map.get(
            "bosys_token", "<mask:4>"
        )
        self.eosys_token = self.tokenizer.special_tokens_map.get(
            "eosys_token", "<mask:5>"
        )

    def append_prefix_to_infer_schema(self, data, prefix):
        """append_prefix_to_infer_schema"""
        context = data["context"]
        context.append(
            {"role": "bot", "utterance": [{"type": "text", "text": str(prefix)}]}
        )
        data["context"] = context
        return data

    def process(self, data, **kwargs):
        """
        process

        """
        if get_format_type(data) == "schema" and self.is_pretraining:
            return self.schema_correction(data)
        if self.is_pretraining and self.is_training:
            return self.schema_correction(data)
        else:
            if not self.is_training:

                data = self.utterance_2_schema(data, **kwargs)
            if not self.is_pretraining:
                # Support for SFT-Data
                if (
                    len(data.get("image_info", [])) > 0
                    and len(data.get("video_info", [])) > 0
                ):

                    return self.mix_schema_correction(data)
                else:

                    return self.schema_correction(data)
            else:
                return data

    def schema_correction(self, schema):
        """
        schema_correction
        """
        if schema.get("video_info"):
            video_info = schema["video_info"]
            text_info = schema["text_info"]
            matched_text_index_offset = 0
            vid_id = 1
            for video_index, video_one in enumerate(video_info):
                video_width = video_one.get(
                    "video_width", video_one.get("image_width", -1)
                )
                video_height = video_one.get(
                    "video_height", video_one.get("image_height", -1)
                )

                matched_text_index = (
                    video_one.get("matched_text_index", 0) + matched_text_index_offset
                )
                url = video_one["image_url"]
                if video_width == -1 or video_height == -1:
                    downloaded_path = get_downloadable(url, save_to_disk=False)
                    if isinstance(downloaded_path, bytes):
                        bytes_content = io.BytesIO(downloaded_path)
                        vr = VideoReaderWrapper(bytes_content, num_threads=1)
                    else:
                        vr = VideoReaderWrapper(downloaded_path, num_threads=1)
                    tmp_frame = Image.fromarray(vr[0].asnumpy(), "RGB")
                    video_width = tmp_frame.width
                    video_height = tmp_frame.height
                    ret = {
                        "image_url": vr,
                        "matched_text_index": matched_text_index,
                        "image_width": video_width,
                        "image_height": video_height,
                        "is_valid": True,
                        "image_type": "video",
                    }
                else:
                    ret = video_one
                if "extracted_frame_indices" in video_one:
                    ret["extracted_frame_indices"] = video_one[
                        "extracted_frame_indices"
                    ]
                if "subtitles_auto" in video_one:
                    ret["asr"] = video_one["subtitles_auto"]
                if "subtitles" in video_one:
                    ret["asr"] = video_one["subtitles"]
                if "asr" in ret:
                    for asr_one in ret["asr"]:
                        assert len(asr_one) == 3
                        assert isinstance(asr_one[0], str)
                        assert isinstance(asr_one[1], (float, int))
                        assert isinstance(asr_one[2], (float, int))
                        asr_one[1], asr_one[2] = float(asr_one[1]), float(asr_one[2])

                if not self.is_pretraining:
                    if self.use_pic_id:
                        text_info = (
                            text_info[:matched_text_index]
                            + [
                                {
                                    "text": f"Video {vid_id}:",
                                    "tag": "mask",
                                }
                            ]
                            + text_info[matched_text_index:]
                        )
                        vid_id += 1
                        matched_text_index += 1
                        matched_text_index_offset += 1
                    text_info = (
                        text_info[:matched_text_index]
                        + [
                            {
                                "text": self.video_start_token,
                                "tag": "mask",
                                "text_type": "special_token",
                            }
                        ]
                        + [
                            {
                                "text": self.video_end_token,
                                "tag": "mask",
                                "text_type": "special_token",
                            }
                        ]
                        + text_info[matched_text_index:]
                    )
                    matched_text_index += 1
                    matched_text_index_offset += 2
                    ret["matched_text_index"] = matched_text_index

                video_info[video_index] = ret

            schema["text_info"] = text_info
            schema["video_info"] = video_info
        elif schema.get("is_video", False) and "image_info" in schema:
            text_info = schema["text_info"]

            matched_text_index_offset = 0
            if not self.is_pretraining:
                vid_id = 1
                matched_text_index = schema["image_info"][0]["matched_text_index"]
                if self.use_pic_id:
                    text_info = (
                        text_info[:matched_text_index]
                        + [
                            {
                                "text": f"Video {vid_id}:",
                                "tag": "mask",
                            }
                        ]
                        + text_info[matched_text_index:]
                    )
                    matched_text_index_offset += 1
                    matched_text_index += 1
                text_info = (
                    text_info[:matched_text_index]
                    + [
                        {
                            "text": self.video_start_token,
                            "tag": "mask",
                            "text_type": "special_token",
                        }
                    ]
                    + text_info[matched_text_index:]
                )
                matched_text_index += 1
                matched_text_index_offset += 1

            uid = str(uuid.uuid4())
            ret_image_info = []
            for img_one in schema["image_info"]:
                img_one["image_type"] = "video"
                img_one["video_uid"] = uid
                img_one["matched_text_index"] += matched_text_index_offset
                ret_image_info.append(img_one)

            if not self.is_pretraining:
                matched_text_index = ret_image_info[-1]["matched_text_index"]
                text_info = (
                    text_info[:matched_text_index]
                    + [
                        {
                            "text": self.video_end_token,
                            "tag": "mask",
                            "text_type": "special_token",
                        }
                    ]
                    + text_info[matched_text_index:]
                )
            schema["image_info"] = ret_image_info
            schema["text_info"] = text_info

        if self.is_pretraining and self.is_training:
            # pretraining uses the old version of schema (which uses text_list instead of text_info)
            if "text_list" not in schema:
                schema["text_list"] = schema["text_info"]

        return schema

    def mix_schema_correction(self, schema):
        """
        mix schema with image and video
        """
        if "order" not in schema:
            raise ValueError(
                "when image and video both exist, schema must contain order"
            )

        def add_item_to_order(order_new, element, data_type, index):
            order_new["type"].append(data_type)
            order_new["index"].append(index)
            order_new["mask"].append(int(element.get("tag", "mask") == "mask"))

        schema_new = defaultdict(lambda: [])
        order_new = {"type": [], "index": [], "mask": []}
        for data_type, data_ind in zip(
            schema["order"]["type"], schema["order"]["index"]
        ):
            if data_type in ["text", "image"]:
                element = schema[f"{data_type}_info"][data_ind]
                schema_new[f"{data_type}_info"].append(element)
                add_item_to_order(
                    order_new,
                    element,
                    data_type,
                    len(schema_new[f"{data_type}_info"]) - 1,
                )
            elif data_type == "video":
                # get video info
                video_one = schema[f"{data_type}_info"][data_ind]
                video_width = video_one.get(
                    "video_width", video_one.get("image_width", -1)
                )
                video_height = video_one.get(
                    "video_height", video_one.get("image_height", -1)
                )
                url = video_one["image_url"]
                if video_width == -1 or video_height == -1:
                    downloaded_path = get_downloadable(url, save_to_disk=False)
                    if isinstance(downloaded_path, bytes):
                        bytes_content = io.BytesIO(downloaded_path)
                        vr = VideoReaderWrapper(bytes_content, num_threads=1)
                    else:
                        vr = VideoReaderWrapper(downloaded_path, num_threads=1)
                    tmp_frame = Image.fromarray(vr[0].asnumpy(), "RGB")
                    video_width = tmp_frame.width
                    video_height = tmp_frame.height
                    ret = {
                        "image_url": vr,
                        "image_width": video_width,
                        "image_height": video_height,
                        "is_valid": True,
                        "image_type": "video",
                    }
                else:
                    ret = video_one
                if "extracted_frame_indices" in video_one:
                    ret["extracted_frame_indices"] = video_one[
                        "extracted_frame_indices"
                    ]
                if "subtitles_auto" in video_one:
                    ret["asr"] = video_one["subtitles_auto"]
                if "subtitles" in video_one:
                    ret["asr"] = video_one["subtitles"]
                if "asr" in ret:
                    for asr_one in ret["asr"]:
                        assert len(asr_one) == 3
                        assert isinstance(asr_one[0], str)
                        assert isinstance(asr_one[1], (float, int))
                        assert isinstance(asr_one[2], (float, int))
                        asr_one[1], asr_one[2] = float(asr_one[1]), float(asr_one[2])

                if not self.is_pretraining:
                    if self.use_pic_id:
                        vid_id = len(schema_new["video_info"])
                        element = {"text": f"Video {vid_id}:", "tag": "mask"}
                        schema_new["text_info"].append(element)
                        add_item_to_order(
                            order_new, element, "text", len(schema_new["text_info"]) - 1
                        )

                    # video start
                    element = {
                        "text": self.video_start_token,
                        "tag": "mask",
                        "text_type": "special_token",
                    }
                    schema_new["text_info"].append(element)
                    add_item_to_order(
                        order_new, element, "text", len(schema_new["text_info"]) - 1
                    )

                # video
                ret["matched_text_index"] = len(schema_new["text_info"])
                schema_new[f"{data_type}_info"].append(ret)
                add_item_to_order(
                    order_new,
                    video_one,
                    data_type,
                    len(schema_new[f"{data_type}_info"]) - 1,
                )

                if not self.is_pretraining:
                    # video end
                    element = {
                        "text": self.video_end_token,
                        "tag": "mask",
                        "text_type": "special_token",
                    }
                    schema_new["text_info"].append(element)
                    add_item_to_order(
                        order_new, element, "text", len(schema_new["text_info"]) - 1
                    )

        for data_type in ["text", "image", "video"]:
            schema[f"{data_type}_info"] = schema_new[f"{data_type}_info"]
        schema["order"] = order_new

        return schema

    def openai_2_ernie(self, user_input):
        """openai_2_ernie"""
        if "messages" in user_input:
            for idx, context in enumerate(user_input["messages"]):
                if context["role"] == "assistant":
                    user_input["messages"][idx]["role"] = "bot"
                assert (
                    "content" in context
                ), "openai-messages should contain key: context."
                user_input["messages"][idx]["utterance"] = context["content"]
                del user_input["messages"][idx]["content"]
            user_input["context"] = user_input["messages"]
            del user_input["messages"]
        if "max_tokens" in user_input:
            user_input["max_dec_len"] = user_input["max_tokens"]
            del user_input["max_tokens"]
        return user_input

    def utterance_2_schema(self, user_input, save_to_disk=False, **kwargs):
        """
        Convert data from "utterance" format (the input format for inference APIs)
        to a structured "schema" format (a standard data format).
        Note: It is necessary to download the content inside,
        mainly in order to obtain the dimensions (width/height) of images or videos.
        Args:
        utterance: The input data in utterance format
        save_to_disk: Whether to cache the downloaded data to disk
        """
        image_info = []
        text_list = []
        video_info = []
        assert save_to_disk in [
            True,
            False,
        ], f"save_to_disk needs to be a boolean value, but got {save_to_disk}"

        # support openai-format
        user_input = self.openai_2_ernie(user_input)
        if kwargs.get("prefix"):
            prefix = kwargs["prefix"]
            user_input = self.append_prefix_to_infer_schema(user_input, prefix)

        ret = {}
        for context in user_input["context"]:
            role = context["role"]
            utterance = context["utterance"]
            # handle user/bot input
            mask_tag = "mask" if role in ["user", "system"] else "no_mask"
            if isinstance(utterance, List):
                for one in utterance:
                    if one["type"] == "image_url":
                        assert role in [
                            "user",
                            "system",
                        ], "image only in user/system utterance"
                        url = one["image_url"]["url"]
                        image_width = one["image_url"].get("image_width", -1)
                        image_height = one["image_url"].get("image_height", -1)
                        downloaded_path = get_downloadable(
                            url, download_dir=RAW_IMAGE_DIR, save_to_disk=save_to_disk
                        )
                        if isinstance(downloaded_path, bytes):
                            img = io.BytesIO(downloaded_path)
                            img = Image.open(img)
                            downloaded_path = img
                        else:
                            img = Image.open(downloaded_path)
                        image_width = img.width
                        image_height = img.height
                        img_one = {
                            "image_url": downloaded_path,
                            "matched_text_index": len(text_list),
                            "image_width": image_width,
                            "image_height": image_height,
                            "is_valid": True,
                            "image_type": "image",
                        }
                        image_info.append(img_one)
                        # IMAGE_END
                    elif one["type"] == "video_url":
                        video_width = one["video_url"].get("video_width", -1)
                        video_height = one["video_url"].get("video_height", -1)
                        url = one["video_url"]["url"]
                        # VIDEO_START
                        downloaded_path = get_downloadable(
                            url, download_dir=RAW_VIDEO_DIR, save_to_disk=save_to_disk
                        )
                        if isinstance(downloaded_path, bytes):
                            bytes_content = io.BytesIO(downloaded_path)
                            vr = VideoReaderWrapper(bytes_content, num_threads=1)
                        else:
                            vr = VideoReaderWrapper(downloaded_path, num_threads=1)
                        tmp_frame = Image.fromarray(vr[0].asnumpy(), "RGB")
                        video_width = tmp_frame.width
                        video_height = tmp_frame.height
                        video_one = {
                            "image_url": vr,
                            "matched_text_index": len(text_list),
                            "image_width": video_width,
                            "image_height": video_height,
                            "is_valid": True,
                            "image_type": "video",
                        }
                        if "extracted_frame_indices" in one["video_url"]:
                            video_one["extracted_frame_indices"] = one["video_url"][
                                "extracted_frame_indices"
                            ]
                        if "subtitles_auto" in one["video_url"]:
                            video_one["asr"] = one["video_url"]["subtitles_auto"]
                        if "subtitles" in one["video_url"]:
                            video_one["asr"] = one["video_url"]["subtitles"]
                        if "asr" in video_one:
                            for asr_one in video_one["asr"]:
                                assert len(asr_one) == 3
                                assert isinstance(asr_one[0], str)
                                assert isinstance(asr_one[1], (float, int))
                                assert isinstance(asr_one[2], (float, int))
                                asr_one[1], asr_one[2] = float(asr_one[1]), float(
                                    asr_one[2]
                                )

                        video_info.append(video_one)
                    elif one["type"] == "text":
                        text_list.append({"text": one["text"], "tag": mask_tag})
                    else:
                        raise ValueError("Unsupport utterance item type.")
            elif isinstance(utterance, str):
                # pure text input
                text_list.append({"text": utterance, "tag": mask_tag})
            else:
                raise ValueError("Unsupport utterance data type.")

            if role == "system":
                # default
                text_list.append({"text": "好的", "tag": "no_mask"})
                ret["is_system"] = 1

        ret["text_info"] = text_list
        if len(image_info) > 0:
            ret["image_info"] = image_info
        if len(video_info) > 0:
            ret["video_info"] = video_info
        return ret
