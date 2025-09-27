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
refbox_to_points
"""
import re
import sys
from copy import deepcopy


def merge_list(lists):
    """merge multi list to one list

    Args:
        lists (list[list]): [[], [], ...]

    Returns:
        list: one list
    """
    new_list = lists[0]
    for one in lists[1:]:
        new_list.extend(one)
    return new_list


def refbox_to_points(text_item):
    """convert ref box to points"""

    patterns = {
        "ref_box": re.compile(
            r"<ref>(.*?)</ref>" + r"(<box>\((.*?),(.*?)\),\((.*?),(.*?)\)</box>)",
            re.DOTALL,
        ),
        "ref_quad": re.compile(
            r"<ref>(.*?)</ref>" + r"(<quad>\((.*?),(.*?)\),\((.*?),(.*?)\),\((.*?),(.*?)\),\((.*?),(.*?)\)</quad>)",
            re.DOTALL,
        ),
        "box": re.compile(r"<box>\((.*?),(.*?)\),\((.*?),(.*?)\)</box>", re.DOTALL),
        "quad": re.compile(
            r"<quad>\((.*?),(.*?)\),\((.*?),(.*?)\)," + r"\((.*?),(.*?)\),\((.*?),(.*?)\)</quad>",
            re.DOTALL,
        ),
        "segment": re.compile(
            r"(<ref>.*?</ref><box>.*?</box>|" + r"<ref>.*?</ref><quad>.*?</quad>|<box>.*?</box>|<quad>.*?</quad>)",
            re.DOTALL,
        ),
    }

    segments = re.split(patterns["segment"], text_item["text"])
    segments = [part for part in segments if part]

    text_info_output = []
    for part_idx, part in enumerate(segments):
        if patterns["ref_box"].match(part):
            match = patterns["ref_box"].match(part)
            new_text_item = process_match(text_item, match, "box")
            text_info_output.append(new_text_item)
        elif patterns["ref_quad"].match(part):
            match = patterns["ref_quad"].match(part)
            new_text_item = process_match(text_item, match, "quad")
            text_info_output.append(new_text_item)
        elif patterns["box"].match(part):
            match = patterns["box"].match(part)
            add_points_to_last(text_info_output, match.groups(), "box")
        elif patterns["quad"].match(part):
            match = patterns["quad"].match(part)
            add_points_to_last(text_info_output, match.groups(), "quad")
        else:

            new_text_item = deepcopy(text_item)
            new_text_item["text"] = adjust_spaces(part, part_idx, segments)
            new_text_item["points"] = None
            text_info_output.append(new_text_item)

    return text_info_output


def process_match(text_item, match, shape_type):
    """process_match"""
    shape2num_loc = {"box": 4, "quad": 8}
    new_text_item = deepcopy(text_item)
    new_text_item["text"] = match.group(1)
    if not new_text_item["text"]:
        print(
            f"[WARNING] {match.group(0)} empty referring text in text_list: {text_item}",
            file=sys.stderr,
        )
    locs = list(map(int, match.groups()[2:]))  # Skip the first two groups which are <ref> and <shape> tags
    assert len(locs) == shape2num_loc[shape_type], f"[ERROR] {match.group(0)} has wrong number of coordinates!"
    points = [[locs[i], locs[i + 1]] for i in range(0, len(locs), 2)]
    new_text_item["points"] = [points]

    return new_text_item


def add_points_to_last(text_info_output, groups, shape_type):
    """add points to last text_item"""
    shape2num_loc = {"box": 4, "quad": 8}
    locs = list(map(int, groups))
    assert len(locs) == shape2num_loc[shape_type], f"[ERROR] wrong number of coordinates in {groups}!"
    points = [[locs[i], locs[i + 1]] for i in range(0, len(locs), 2)]
    text_info_output[-1]["points"].append(points)


def adjust_spaces(part, part_idx, segments):
    """adjust spaces"""
    text = part
    if not part.startswith(" ") and part_idx != 0:
        text = f" {text}"
    if not part.endswith(" ") and part_idx != len(segments) - 1:
        text = f"{text} "
    return text


def standardization_points(text_info):
    """ref box quad to points"""

    text_idx_map = dict()
    new_text_info = []

    for i in range(len(text_info)):
        if has_ref_box(text_info[i]["text"]):
            # idx mapping
            text_idx_map[i] = len(new_text_info)

            new_text_info += refbox_to_points(deepcopy(text_info[i]))

        else:
            new_text_info.append(text_info[i])

    return new_text_info, text_idx_map


def has_ref_box(text):
    """has_ref_box"""

    if ("<ref>" in text and "</ref>" in text) and ("</box>" in text or "</quad>" in text):
        return True
    return False


def image_info_reorder(text_idx_map, image_info):
    """mapping"""

    images = []
    for image in image_info:

        if image["matched_text_index"] in text_idx_map:
            image_new = deepcopy(image)
            image["matched_text_index"] = text_idx_map[image["matched_text_index"]]
            images.append(image_new)
        else:
            images.append(image)
    return images


def box_to_quad(box):
    """box_to_quad"""
    x1, y1 = box[0]
    x2, y2 = box[1]
    quad = []
    quad.append([x1, y1])
    quad.append([x2, y1])
    quad.append([x2, y2])
    quad.append([x1, y2])
    return quad


def standardization(sample, data_type):
    """
    standardization
    """

    # ref box quad to points
    whole_text = "".join([t["text"] for t in sample["text_info"]])

    if has_ref_box(whole_text):
        text_info, text_idx_map = standardization_points(sample["text_info"])
        image_info = image_info_reorder(text_idx_map, sample["image_info"])

        # check
        max_idx = max([one["matched_text_index"] for one in image_info])
        assert max_idx < len(text_info), "[ERROR] 非文本结尾！"

        sample = {"text_info": text_info, "image_info": image_info}

    for i in range(len(sample["text_info"])):
        points = sample["text_info"][i].get("points", None)
        if points is not None:
            quad = []
            for box in points:
                if len(box) == 2:
                    quad.append(box_to_quad(box))
                else:
                    quad.append(box)
            sample["text_info"][i]["points"] = quad

    # mark origin image resolution
    image_info = []
    for one in sample["image_info"]:
        one["origin_image_width"] = one["image_width"]
        one["origin_image_height"] = one["image_height"]
        image_info.append(one)
    sample["image_info"] = image_info

    return sample


def clear_text_info(sample, **kwargs):
    """
    clear_text_info
    """
    info_key = "text_info"
    sample_id = sample.get("id", "") or sample.get("data_id", "")

    text_list = sample.get("text_list", None)

    if text_list is None or len(text_list) == 0:
        print(f"[ERROR] sample_id: {sample_id} | empty text_list", file=sys.stderr)
        return info_key, None

    for i in range(len(sample["text_list"])):
        if "text" not in sample["text_list"][i]:
            print(
                f"[WARNING] sample_id: {sample_id} | 'text' not in text_list",
                file=sys.stderr,
            )
            sample["text_list"][i]["text"] = ""

    content = "".join([t["text"] for t in sample["text_list"]])
    if content == "":
        print(f"[ERROR] sample_id: {sample_id} | empty content in this sample", file=sys.stderr)
        return info_key, None

    for i in range(len(sample["text_list"])):
        tag = sample["text_list"][i].get("tag", "no_mask")
        is_valid = sample["text_list"][i].get("is_valid", True)
        if tag != "mask":
            tag = "no_mask"
        sample["text_list"][i]["tag"] = tag
        sample["text_list"][i]["is_valid"] = is_valid

    return info_key, sample["text_list"]


def clear_image_info_base(sample, **kwargs):
    """
    clear_image_info_base
    """
    info_key = "image_info"

    if "image_info" not in sample or sample["image_info"] is None or len(sample["image_info"]) == 0:
        print("[ERROR] no image info in this sample: {sample}", file=sys.stderr)
        return None

    text_list_len = len(sample["text_list"])
    img_idx = 0
    while img_idx < len(sample["image_info"]):

        if sample["image_info"][img_idx]["matched_text_index"] == -1:
            sample["image_info"][img_idx]["matched_text_index"] = 0

        image = sample["image_info"][img_idx]
        matched_text_index = image["matched_text_index"]
        skip_image_flag = False

        if matched_text_index < 0:
            print(
                f"Invalid matched_text_index: {matched_text_index} < 0, skip this image: {image}",
                file=sys.stderr,
            )
            skip_image_flag = True

        if matched_text_index >= text_list_len:
            print(
                f"""Invalid matched_text_index: {matched_text_index} > max text list length: {text_list_len},
                    skip this image: {image}""",
                file=sys.stderr,
            )
            skip_image_flag = True

        is_valid_image = image.get("is_valid", True)
        if not is_valid_image:
            print(
                f"is_valid == {is_valid_image}, skip this image: {image}",
                file=sys.stderr,
            )
            skip_image_flag = True

        if skip_image_flag:
            sample["image_info"].pop(img_idx)
        else:
            img_idx += 1

    if len(sample["image_info"]) == 0:
        print("[ERROR] no valid image in this sample!", file=sys.stderr)
        return None
    else:
        return sample


def clear_image_info(sample, **kwargs):
    """
    clear_image_info
    """
    info_key = "image_info"
    sample = clear_image_info_base(sample, **kwargs)
    if sample is None:
        return info_key, None

    images = []
    for image in sample["image_info"]:

        images.append(
            {
                "image_url": image["image_url"],
                "matched_text_index": image["matched_text_index"],
                "image_width": image["image_width"],
                "image_height": image["image_height"],
                "is_valid": image.get("is_valid", True),
                "image_type": "image",
                "data_type": image.get("data_type", None),
            }
        )

    return info_key, images


def clear_video_info(sample, **kwargs):
    """
    clear_video_info
    """
    info_key = "image_info"
    sample = clear_image_info_base(sample, **kwargs)
    if sample is None:
        return info_key, None

    last_matched_text_index = -1
    for image in sample["image_info"]:
        if image.get("time_stamp", -1) > 0 and image["matched_text_index"] < last_matched_text_index:
            print(
                f"[ERROR] matched_text_index is not monotonically increasing, the sample will be discarded: {sample}",
                file=sys.stderr,
            )
            return info_key, None
        last_matched_text_index = image["matched_text_index"]

    images = []
    image_width_0 = sample["image_info"][0]["image_width"]
    image_height_0 = sample["image_info"][0]["image_height"]
    for image in sample["image_info"]:
        if image["time_stamp"] < 0:
            print(
                f"[ERROR] image_info time_stamp: {image['time_stamp']}error, the sample will be discarded: {sample}",
                file=sys.stderr,
            )
            return info_key, None

        if image["image_width"] != image_width_0 or image["image_height"] != image_height_0:
            print(
                f"[ERROR] image_width or image_height of a single video frame differs \
                  from that of the first frame, sample discarded:{sample}",
                file=sys.stderr,
            )
            return info_key, None

        images.append(
            {
                "image_url": image["image_url"],
                "matched_text_index": image["matched_text_index"],
                "image_width": image["image_width"],
                "image_height": image["image_height"],
                "is_valid": image.get("is_valid", True),
                "time_stamp": image["time_stamp"],
                "image_type": "video",
                "data_type": image.get("data_type", None),
            }
        )

    return info_key, images
