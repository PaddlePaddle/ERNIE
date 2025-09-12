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
This module provides some utils functions
"""

import copy
from itertools import groupby


def is_gif(data: bytes) -> bool:
    """
    check if a bytes is a gif based on the magic head
    """
    return data[:6] in (b"GIF87a", b"GIF89a")


def group_frame_by_video(schema):
    """
    group frame by video
    """
    if "image_info" in schema:
        image_info = copy.deepcopy(schema["image_info"])
    else:
        image_info = copy.deepcopy(schema)

    for idx, img in enumerate(image_info):
        if img["image_type"] != "video":
            img["video_uid"] = idx

    cnt = 0
    ret = []
    keys = []
    for key, group in groupby(image_info, key=lambda x: x["video_uid"]):
        keys.append(key)
        group_len = len(list(group))
        ret.append(list(range(cnt, group_len + cnt)))
        cnt += group_len

    assert len(keys) == len(set(keys)), f"found duplicate keys: {keys}"
    return ret
