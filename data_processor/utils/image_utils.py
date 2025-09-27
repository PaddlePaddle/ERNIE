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
random resize
"""

import math
import random


def random_equal_probability(min_resize_ratio=0.5, max_resize_ratio=2, middle_number=1):
    """
    random_equal_probability
    """
    interval = random.choice([0, 1])

    if interval == 0:

        random_number = random.uniform(min_resize_ratio, middle_number)
    else:

        random_number = random.uniform(middle_number, max_resize_ratio)

    return random_number


def get_random_scale_by_area_ratio(min_resize_ratio=0.25, max_resize_ratio=1.75):
    """
    get_random_scale_by_area_ratio

    Args:
        min_resize_ratio (float): min_resize_ratio, default value 0.5。
        max_resize_ratio (float): max_resize_ratio, default value 2。

    Returns:
        float:

    """
    return math.sqrt(random.uniform(min_resize_ratio, max_resize_ratio))


def random_resize_img(sample: dict, min_resize_ratio=0.25, max_resize_ratio=1.75, middle_number=1):
    """
    random resize
    """
    if min_resize_ratio == max_resize_ratio == 1:
        return sample

    assert (
        min_resize_ratio < max_resize_ratio
    ), f"[ERROR] min_resize_ratio={min_resize_ratio}, max_resize_ratio={max_resize_ratio}"

    random_resize_factor = -1
    text_info, image_info = [], []

    for one in sample["image_info"]:
        random_resize_factor = get_random_scale_by_area_ratio(
            min_resize_ratio=min_resize_ratio,
            max_resize_ratio=max_resize_ratio,
        )
        resized_width = int(one["image_width"] * random_resize_factor)
        resized_height = int(one["image_height"] * random_resize_factor)

        if min(resized_width, resized_height) < 5:
            resized_width, resized_height = one["image_width"], one["image_height"]
            random_resize_factor = 1

        one["image_width"] = resized_width
        one["image_height"] = resized_height
        one["random_resize_factor"] = random_resize_factor
        image_info.append(one)

    for one in sample["text_info"]:

        if "points" in one.keys():
            assert random_resize_factor != -1, f'[ERROR] len(image_info)={len(sample["image_info"])} resize error'
            assert (
                len(sample["image_info"]) == 1
            ), "[ERROR] The coordinate resizing is not suitable for samples with multiple images!"

            if one["points"] is not None:
                new_p = []
                for p in one["points"]:
                    new_p.append(
                        [
                            [
                                int(xy[0] * random_resize_factor),
                                int(xy[1] * random_resize_factor),
                            ]
                            for xy in p
                        ]
                    )
                one["points"] = new_p

        text_info.append(one)

    sample["image_info"] = image_info
    sample["text_info"] = text_info

    return sample
