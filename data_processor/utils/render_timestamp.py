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

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

cur_directory = Path(__file__).parent.absolute()
FONT_PATH = os.path.join(cur_directory, "Roboto-Regular.ttf")


def render_single_image_with_timestamp(image: Image, number: str, rate: float, font_path: str = FONT_PATH):
    """
    Function: Renders a timestamp to the image of pil.image
    The timestamp size is the rate of min(width, height)
    The font color is black, the outline is white, and the outline size is 10% of the font
    Returns an Image object
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font_size = int(min(width, height) * rate)
    outline_size = int(font_size * 0.1)
    font = ImageFont.truetype(font_path, font_size)
    x = 0
    y = 0

    # Draw a black timestamp with a white border
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
    function: get the timestamp of a frame, used when evenly extracting frames.

    num_frames: total number of frames
    frameid_list: index of the extracted frame
    duration: total duration of the video
    return: timestamp; xx:xx:xx (str)
    """
    time_stamp = duration * 1.0 * frame_id / num_frames

    return time_stamp


def render_frame_timestamp(frame, timestamp, font_rate=0.1):
    """
    Function, given a frame, render the index in order
    Logic: render the index to the upper left corner of the image
    frame: frame, PIL.Image object
    timestamp: timestamp, in seconds
    font_rate: the ratio of font size to min(wi, hei)
    """
    time_stamp = "time: " + timestamp_converting(timestamp)
    new_frame = render_single_image_with_timestamp(frame, time_stamp, font_rate)

    return new_frame
