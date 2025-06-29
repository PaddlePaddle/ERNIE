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
render_single_image_with_timestamp
"""
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

cur_directory = Path(__file__).parent.absolute()
FONT_PATH = os.path.join(cur_directory, "internlmxc2d5_font.ttf")


def render_single_image_with_timestamp(image: Image, number: str, rate: float, font_path: str = FONT_PATH):
    """
    Render a timestamp on a PIL.Image object
    - The size of the timestamp font is determined by `rate * min(width, height)`
    - The font color is black, with a white outline
    - The outline width is 10% of the font size
    - Returns a new `Image` object with the timestamp rendered on it
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font_size = int(min(width, height) * rate)
    outline_size = int(font_size * 0.1)
    font = ImageFont.truetype(font_path, font_size)
    x = 0
    y = 0

    draw.text((x, y), number, font=font, fill=(0, 0, 0), stroke_width=outline_size, stroke_fill=(255, 255, 255))

    return image


def timestamp_converting(time_stamp_in_seconds):
    """
    convert timestamp format from seconds to hr:min:sec
    """
    # get hours
    hours = 0
    while time_stamp_in_seconds >= 3600:
        hours += 1
        time_stamp_in_seconds -= 3600
    # get minutes
    mins = 0
    while time_stamp_in_seconds >= 60:
        mins += 1
        time_stamp_in_seconds -= 60
    time_hours = f"{int(hours):02d}"
    time_mins = f"{int(mins):02d}"
    time_secs = f"{time_stamp_in_seconds:05.02f}"
    fi_time_stamp = time_hours + ":" + time_mins + ":" + time_secs

    return fi_time_stamp


def get_timestamp_for_uniform_frame_extraction(num_frames, frame_id, duration):
    """
    function: get the timestamp of a frame, which will be used as filename。
    num_frames: the total number of frames
    frame_id: the index of the frame
    duration: the duration of the video
    """
    time_stamp = duration * 1.0 * frame_id / num_frames

    return time_stamp


def render_frame_timestamp(frame, timestamp, font_rate=0.1):
    """
    Render an index (frame number or timestamp) onto a given image frame in sequence
    Logic: Render the index in the top-left corner of the image
    Parameters:
    frame: The input image frame, a PIL.Image object
    timestamp: The timestamp to render, in seconds
    font_rate: The ratio of the font size relative to min(width, height) of the image
    Returns: A new PIL.Image object with the index rendered on it
    """

    time_stamp = "time: " + timestamp_converting(timestamp)
    new_frame = render_single_image_with_timestamp(frame, time_stamp, font_rate)

    return new_frame
