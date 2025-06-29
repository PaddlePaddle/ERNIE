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
chat template utils
"""

import io

from PIL import Image

from data_processor.utils.io_utils import get_downloadable
from ernie.tokenizer_vl import (
    SFT_ASR_END_TOKEN,
    SFT_ASR_START_TOKEN,
    SFT_IMAGE_END_TOKEN,
    SFT_IMAGE_START_TOKEN,
    SFT_VIDEO_END_TOKEN,
    SFT_VIDEO_START_TOKEN,
)


def apply_ernie_chat_training_template(data, tokenizer, is_training, use_pic_id=True, save_to_disk=False, **kwargs):
    """
    apply_ernie_chat_training_tempelate
    """

    # used special tokens
    image_start_token = tokenizer.special_tokens_map.get("image_start_token", SFT_IMAGE_START_TOKEN)
    image_end_token = tokenizer.special_tokens_map.get("image_end_token", SFT_IMAGE_END_TOKEN)
    video_start_token = tokenizer.special_tokens_map.get("video_start_token", SFT_VIDEO_START_TOKEN)
    video_end_token = tokenizer.special_tokens_map.get("video_end_token", SFT_VIDEO_END_TOKEN)
    asr_start_token = tokenizer.special_tokens_map.get("asr_start_token", SFT_ASR_START_TOKEN)
    asr_end_token = tokenizer.special_tokens_map.get("asr_end_token", SFT_ASR_END_TOKEN)
    bosys_token = tokenizer.special_tokens_map.get("bosys_token", "<mask:4>")
    eosys_token = tokenizer.special_tokens_map.get("eosys_token", "<mask:5>")
    eos_token = tokenizer.special_tokens_map.get("eos_token", "</s>")
    cls_token = tokenizer.special_tokens_map.get("cls_token", "<mask:0>")
    sep_token = tokenizer.special_tokens_map.get("sep_token", "<|endofprompt|>")
    all_item_list = data["all_item_list"]
    is_system = data["is_system"]

    new_image_info = []
    new_text_info = []
    pic_id = 1
    vid_id = 1
    # support system-setting
    for item_id, item in enumerate(all_item_list):
        if item_id == 0:
            # support system-setting
            if is_system:
                new_text_info.append({"text": bosys_token, "tag": "mask", "text_type": "special_token"})
            else:
                new_text_info.append({"text": cls_token, "tag": "mask", "text_type": "special_token"})
        # HARD-Code: support system-setting, drop first tgt in system-turn
        # TODO：fix this hard code.
        if item_id == 1 and is_system:
            continue
        label = 1
        for sub_item_idx, sub_item in enumerate(item):
            # is image
            if isinstance(sub_item, list):
                for image_item_idx, image_item in enumerate(sub_item):
                    is_video = False  # indicator of whether the current image is a video frame

                    # check if it is video and insert video end if it is a new video
                    if image_item.get("image_type", "image") == "video":
                        is_video = True

                    # pic id
                    if use_pic_id:
                        if not is_video:
                            new_text_info.append({"text": f"Picture {pic_id}:", "tag": "mask"})
                            pic_id += 1

                    # image start token
                    if not is_video:
                        new_text_info.append({"text": image_start_token, "tag": "mask"})

                    # construct the image
                    image_width = image_item.get("image_width", None)
                    image_height = image_item.get("image_height", None)
                    is_valid = image_item.get("is_valid", None)

                    if image_width and image_height and is_valid:
                        img_one = {
                            "image_url": image_item["image_url"],
                            "matched_text_index": len(new_text_info),
                            "image_width": image_width,
                            "image_height": image_height,
                            "is_valid": is_valid,
                            "image_type": "image" if not is_video else "video",
                        }
                    else:
                        downloaded_path = get_downloadable(image_item["image_url"], save_to_disk=save_to_disk)
                        if isinstance(downloaded_path, bytes):
                            img = io.BytesIO(downloaded_path)
                            img = Image.open(img)
                            downloaded_path = img
                        elif isinstance(downloaded_path, Image.Image):
                            img = downloaded_path
                        else:
                            img = Image.open(downloaded_path)
                        image_width = img.width
                        image_height = img.height
                        img_one = {
                            "image_url": downloaded_path,
                            "matched_text_index": len(new_text_info),
                            "image_width": image_width,
                            "image_height": image_height,
                            "is_valid": True,
                            "image_type": "image" if not is_video else "video",
                        }
                    if "time_stamp" in image_item:
                        img_one["time_stamp"] = image_item["time_stamp"]
                    if "video_uid" in image_item:
                        img_one["video_uid"] = image_item["video_uid"]

                    new_image_info.append(img_one)

                    # image end token
                    if not is_video:
                        new_text_info.append({"text": image_end_token, "tag": "mask"})

            else:
                # support label in no-mask data
                if item_id % 2 == 1:
                    label = sub_item.get("label", 1)
                    if label == 0:
                        sub_item["tag"] = "mask"

                new_text_info.append(sub_item)
        # support system-setting
        if item_id == 0 and is_system:
            new_text_info.append({"text": eosys_token, "tag": "mask", "text_type": "special_token"})
            new_text_info.append({"text": cls_token, "tag": "mask", "text_type": "special_token"})
            continue

        if item_id % 2 == 0:
            new_text_info.append({"text": sep_token, "tag": "mask", "text_type": "special_token"})
        else:
            if not is_training and item_id == len(all_item_list) - 1:
                pass
            else:
                new_text_info.append({"text": cls_token, "tag": "no_mask", "text_type": "special_token"})
                if label == 0:
                    new_text_info[-1]["tag"] = "mask"
    video_start_cnt = len([1 for i in new_text_info if i["text"] == video_end_token])
    video_end_cnt = len([1 for i in new_text_info if i["text"] == video_start_token])
    image_start_cnt = len([1 for i in new_text_info if i["text"] == image_start_token])
    image_end_cnt = len([1 for i in new_text_info if i["text"] == image_end_token])
    assert video_start_cnt == video_end_cnt, f"video_start_cnt: {video_start_cnt}, video_end_cnt: {video_end_cnt}"
    assert image_start_cnt == image_end_cnt, f"image_start_cnt: {image_start_cnt}, image_end_cnt: {image_end_cnt}"
    return {"text_info": new_text_info, "image_info": new_image_info}


def apply_deepseek_chat_training_template(data, tokenizer, is_training, use_pic_id=True, save_to_disk=False, **kwargs):
    """
    apply_deepseek_chat_training_tempelate
    """

    # used special tokens
    image_start_token = tokenizer.special_tokens_map.get("image_start_token", SFT_IMAGE_START_TOKEN)
    image_end_token = tokenizer.special_tokens_map.get("image_end_token", SFT_IMAGE_END_TOKEN)
    video_start_token = tokenizer.special_tokens_map.get("video_start_token", SFT_VIDEO_START_TOKEN)
    video_end_token = tokenizer.special_tokens_map.get("video_end_token", SFT_VIDEO_END_TOKEN)
    asr_start_token = tokenizer.special_tokens_map.get("asr_start_token", SFT_ASR_START_TOKEN)
    asr_end_token = tokenizer.special_tokens_map.get("asr_end_token", SFT_ASR_END_TOKEN)
    bosys_token = tokenizer.special_tokens_map.get("bosys_token", "<mask:4>")
    eosys_token = tokenizer.special_tokens_map.get("eosys_token", "<mask:5>")
    eos_token = tokenizer.special_tokens_map.get("eos_token", "</s>")
    cls_token = tokenizer.special_tokens_map.get("cls_token", "<mask:0>")
    sep_token = tokenizer.special_tokens_map.get("sep_token", "<|endofprompt|>")
    all_item_list = data["all_item_list"]
    is_system = data["is_system"]

    new_image_info = []
    new_text_info = []
    pic_id = 1
    vid_id = 1

    for item_id, item in enumerate(all_item_list):
        # append cls token
        if item_id == 0:
            new_text_info.append({"text": cls_token, "tag": "mask", "text_type": "special_token"})
            if is_system:
                pass
        # append user:
        if item_id % 2 == 0 and not (is_system and item_id == 0):
            new_text_info.append({"text": "User: ", "tag": "mask"})

        # HARD-Code: support system-setting, drop first tgt in system-turn
        # TODO：fix this hard code.
        if item_id == 1 and is_system:
            continue

        label = 1
        for sub_item_idx, sub_item in enumerate(item):
            # is image
            if isinstance(sub_item, list):
                for image_item_idx, image_item in enumerate(sub_item):
                    is_video = False  # indicator of whether the current image is a video frame

                    # check if it is video and insert video end if it is a new video
                    if image_item.get("image_type", "image") == "video":
                        is_video = True

                    # pic id
                    if use_pic_id:
                        if not is_video:
                            new_text_info.append({"text": f"Picture {pic_id}:", "tag": "mask"})
                            pic_id += 1

                    # image start token
                    if not is_video:
                        new_text_info.append({"text": image_start_token, "tag": "mask"})

                    # construct the image
                    image_width = image_item.get("image_width", None)
                    image_height = image_item.get("image_height", None)
                    is_valid = image_item.get("is_valid", None)
                    if image_width and image_height and is_valid:
                        img_one = {
                            "image_url": image_item["image_url"],
                            "matched_text_index": len(new_text_info),
                            "image_width": image_width,
                            "image_height": image_height,
                            "is_valid": is_valid,
                            "image_type": "image" if not is_video else "video",
                        }
                    else:
                        downloaded_path = get_downloadable(image_item["image_url"], save_to_disk=save_to_disk)
                        if isinstance(downloaded_path, bytes):
                            img = io.BytesIO(downloaded_path)
                            img = Image.open(img)
                            downloaded_path = img
                        elif isinstance(downloaded_path, Image.Image):
                            img = downloaded_path
                        else:
                            img = Image.open(downloaded_path)
                        image_width = img.width
                        image_height = img.height
                        img_one = {
                            "image_url": downloaded_path,
                            "matched_text_index": len(new_text_info),
                            "image_width": image_width,
                            "image_height": image_height,
                            "is_valid": True,
                            "image_type": "image" if not is_video else "video",
                        }
                    if "time_stamp" in image_item:
                        img_one["time_stamp"] = image_item["time_stamp"]
                    if "video_uid" in image_item:
                        img_one["video_uid"] = image_item["video_uid"]

                    new_image_info.append(img_one)

                    # image end token
                    if not is_video:
                        new_text_info.append({"text": image_end_token, "tag": "mask"})

            else:
                # support label in no-mask data
                if item_id % 2 == 1:
                    label = sub_item.get("label", 1)
                    if label == 0:
                        sub_item["tag"] = "mask"

                new_text_info.append(sub_item)

        if item_id % 2 == 0:
            if is_system and item_id == 0:
                new_text_info.append({"text": "\n", "tag": "mask"})
            else:
                new_text_info.append({"text": "\nAssistant: ", "tag": "mask"})
        else:
            if not is_training and item_id == len(all_item_list) - 1:
                pass
            else:
                new_text_info.append({"text": sep_token, "tag": "no_mask", "text_type": "special_token"})
                if label == 0:
                    new_text_info[-1]["tag"] = "mask"
    video_start_cnt = len([1 for i in new_text_info if i["text"] == video_end_token])
    video_end_cnt = len([1 for i in new_text_info if i["text"] == video_start_token])
    image_start_cnt = len([1 for i in new_text_info if i["text"] == image_start_token])
    image_end_cnt = len([1 for i in new_text_info if i["text"] == image_end_token])
    assert video_start_cnt == video_end_cnt, f"video_start_cnt: {video_start_cnt}, video_end_cnt: {video_end_cnt}"
    assert image_start_cnt == image_end_cnt, f"image_start_cnt: {image_start_cnt}, image_end_cnt: {image_end_cnt}"
    return {"text_info": new_text_info, "image_info": new_image_info}
