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
import io
import os
from itertools import groupby
from tempfile import NamedTemporaryFile as ntf

import decord

try:
    # moviepy 1.0
    import moviepy.editor as mp
except:
    # moviepy 2.0
    import moviepy as mp


def is_gif(data: bytes) -> bool:
    """
    check if a bytes is a gif based on the magic head
    """
    return data[:6] in (b"GIF87a", b"GIF89a")


class VideoReaderWrapper(decord.VideoReader):
    """
    Solving memory leak bug

    https://github.com/dmlc/decord/issues/208
    """

    def __init__(self, video_path, *args, **kwargs):
        with ntf(delete=True, suffix=".gif") as gif_file:
            gif_input = None
            self.original_file = None
            if isinstance(video_path, str):
                self.original_file = video_path
                if video_path.lower().endswith(".gif"):
                    gif_input = video_path
            elif isinstance(video_path, bytes):
                if is_gif(video_path):
                    gif_file.write(video_path)
                    gif_input = gif_file.name
            elif isinstance(video_path, io.BytesIO):
                video_path.seek(0)
                tmp_bytes = video_path.read()
                video_path.seek(0)
                if is_gif(tmp_bytes):
                    gif_file.write(tmp_bytes)
                    gif_input = gif_file.name

            if gif_input is not None:
                clip = mp.VideoFileClip(gif_input)
                mp4_file = ntf(delete=False, suffix=".mp4")
                clip.write_videofile(mp4_file.name, verbose=False, logger=None)
                clip.close()
                video_path = mp4_file.name
                self.original_file = video_path

            super().__init__(video_path, *args, **kwargs)
            self.seek(0)

    def __getitem__(self, key):
        frames = super().__getitem__(key)
        self.seek(0)
        return frames

    def __del__(self):
        if self.original_file and os.path.exists(self.original_file):
            os.remove(self.original_file)


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
