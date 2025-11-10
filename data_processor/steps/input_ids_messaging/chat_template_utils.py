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
import json

from PIL import Image

from data_processor.utils.io_utils import get_downloadable
from ernie.tokenizer_vl import (
    SFT_IMAGE_END_TOKEN,
    SFT_IMAGE_START_TOKEN,
    SFT_VIDEO_END_TOKEN,
    SFT_VIDEO_START_TOKEN,
)


def apply_chat_training_template(
    data,
    tokenizer,
    is_training,
    chat_template,
    use_pic_id=True,
    save_to_disk=False,
    **kwargs,
):
    """
    apply_ernie_vl_chat_training_tempelate
    """

    # used special tokens
    image_start_token = tokenizer.special_tokens_map.get(
        "image_start_token", SFT_IMAGE_START_TOKEN
    )
    image_end_token = tokenizer.special_tokens_map.get(
        "image_end_token", SFT_IMAGE_END_TOKEN
    )
    video_start_token = tokenizer.special_tokens_map.get(
        "video_start_token", SFT_VIDEO_START_TOKEN
    )
    video_end_token = tokenizer.special_tokens_map.get(
        "video_end_token", SFT_VIDEO_END_TOKEN
    )
    cls_token = tokenizer.special_tokens_map.get("cls_token", "<mask:0>")
    sep_token = tokenizer.special_tokens_map.get("sep_token", "<|endofprompt|>")
    all_item_list = data["all_item_list"]
    is_system = data["is_system"]

    new_image_info = []
    new_text_info = []
    pic_id = 1

    for item_id, item in enumerate(all_item_list):
        # append cls token
        if item_id == 0:
            new_text_info.append(
                {"text": cls_token, "tag": "mask", "text_type": "special_token"}
            )
            if "tools" in data and data["tools"]:
                new_text_info.append({"text": "\n<tool_list>\n", "tag": "mask"})
                if isinstance(data["tools"], str):
                    new_text_info.append({"text": data["tools"], "tag": "mask"})
                else:
                    new_text_info.append(
                        {"text": json.dumps(data["tools"]), "tag": "mask"}
                    )
                new_text_info.append({"text": "\n</tool_list>\n", "tag": "mask"})
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
                # If the next one is a tool response
                if (
                    sub_item_idx + 1 < len(item)
                    and "tool_response" in item[sub_item_idx + 1]
                    and item[sub_item_idx + 1]["tool_response"]
                ):
                    new_text_info.append({"text": "\n<tool_output>\n", "tag": "mask"})
                for image_item_idx, image_item in enumerate(sub_item):
                    is_video = (
                        False  # indicator of whether the current image is a video frame
                    )

                    # check if it is video and insert video end if it is a new video
                    if image_item.get("image_type", "image") == "video":
                        is_video = True

                    # pic id
                    if use_pic_id:
                        if not is_video:
                            new_text_info.append(
                                {"text": f"Picture {pic_id}:", "tag": "mask"}
                            )
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
                        downloaded_path = get_downloadable(
                            image_item["image_url"], save_to_disk=save_to_disk
                        )
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

                if chat_template == "ernie_vl_thinking":
                    if item_id % 2 == 1:
                        # assistant
                        ## assistant - content
                        if "</think>" in sub_item["text"]:
                            reasoning_content = (
                                sub_item["text"]
                                .split("</think>")[0]
                                .rstrip("\n")
                                .split("<think>")[-1]
                                .lstrip("\n")
                            )
                            content = (
                                sub_item["text"].split("</think>")[-1].lstrip("\n")
                            )
                        else:
                            reasoning_content = ""
                            content = sub_item["text"]

                        if reasoning_content:
                            new_text_info.append({"text": "\n<think>\n", "tag": "mask"})
                            new_text_info.append(
                                {
                                    "text": reasoning_content.strip("\n"),
                                    "tag": sub_item["tag"],
                                }
                            )
                            new_text_info.append(
                                {"text": "\n</think>\n\n", "tag": sub_item["tag"]}
                            )
                        else:
                            new_text_info.append({"text": "\n<think>\n", "tag": "mask"})
                            new_text_info.append(
                                {"text": "\n</think>\n\n", "tag": "mask"}
                            )

                        if len(content) > 0:
                            new_text_info.append(
                                {"text": content, "tag": sub_item["tag"]}
                            )

                        ## assistant - tool calls
                        tool_calls = None
                        if "tool_calls" in sub_item:
                            tool_calls = sub_item.pop("tool_calls")
                        if tool_calls:
                            if isinstance(tool_calls, str):
                                tool_calls = json.loads(tool_calls)
                            if not isinstance(
                                tool_calls, list
                            ):  # parallel function call
                                tool_calls = [tool_calls]

                            for tool_call in tool_calls:
                                if (
                                    "type" in tool_call
                                    and tool_call["type"] == "function"
                                ):
                                    tool_call = tool_call["function"]
                                new_text_info.append(
                                    {
                                        "text": '<tool_call>\n{"name": "',
                                        "tag": sub_item["tag"],
                                    }
                                )
                                new_text_info.append(
                                    {"text": tool_call["name"], "tag": sub_item["tag"]}
                                )
                                new_text_info.append(
                                    {"text": '", "arguments": ', "tag": sub_item["tag"]}
                                )
                                if isinstance(tool_call["arguments"], str):
                                    new_text_info.append(
                                        {
                                            "text": tool_call["arguments"],
                                            "tag": sub_item["tag"],
                                        }
                                    )
                                else:
                                    new_text_info.append(
                                        {
                                            "text": json.dumps(tool_call["arguments"]),
                                            "tag": sub_item["tag"],
                                        }
                                    )
                                new_text_info.append(
                                    {
                                        "text": "}\n</tool_call>\n",
                                        "tag": sub_item["tag"],
                                    }
                                )
                    else:
                        # user / tool
                        tool_response = None
                        if "tool_response" in sub_item and sub_item["tool_response"]:
                            tool_response = sub_item.pop("tool_response")
                        if tool_response:
                            if isinstance(sub_item["text"], str):
                                pass
                            else:
                                sub_item["text"] = json.dumps(sub_item["text"])
                            # If the previous one is not an image / video
                            if sub_item_idx - 1 > 0 and not isinstance(
                                item[sub_item_idx - 1], list
                            ):
                                new_text_info.append(
                                    {"text": "\n<tool_output>\n", "tag": "mask"}
                                )
                            new_text_info.append(sub_item)
                            new_text_info.append(
                                {"text": "\n</tool_output>\n", "tag": "mask"}
                            )
                        else:
                            new_text_info.append(sub_item)
                else:
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
                new_text_info.append(
                    {"text": sep_token, "tag": "no_mask", "text_type": "special_token"}
                )
                if label == 0:
                    new_text_info[-1]["tag"] = "mask"
    video_start_cnt = len([1 for i in new_text_info if i["text"] == video_start_token])
    video_end_cnt = len([1 for i in new_text_info if i["text"] == video_end_token])
    image_start_cnt = len([1 for i in new_text_info if i["text"] == image_start_token])
    image_end_cnt = len([1 for i in new_text_info if i["text"] == image_end_token])
    assert (
        video_start_cnt == video_end_cnt
    ), f"video_start_cnt: {video_start_cnt}, video_end_cnt: {video_end_cnt}"
    assert (
        image_start_cnt == image_end_cnt
    ), f"image_start_cnt: {image_start_cnt}, image_end_cnt: {image_end_cnt}"
    return {"text_info": new_text_info, "image_info": new_image_info}
