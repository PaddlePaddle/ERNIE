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

import io
import os
import random

import numpy as np
from paddleformers.utils.log import logger
from PIL import Image

from data_processor.utils.io_utils import EXTRACTED_FRAME_DIR, get_downloadable, get_filename
from data_processor.utils.video_utils import VideoReaderWrapper


def read_video_decord(video_path, save_to_disk):
    """get reader and meta by decord"""
    video_path = get_downloadable(video_path, save_to_disk=save_to_disk)
    if isinstance(video_path, VideoReaderWrapper):
        video_reader = video_path
    else:
        if isinstance(video_path, bytes):
            video_path = io.BytesIO(video_path)
        video_reader = VideoReaderWrapper(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    video_meta = {"fps": fps, "duration": duration, "num_of_frame": vlen}

    return video_reader, video_meta, video_path


def get_frame_indices(
    vlen,
    target_frames=-1,
    target_fps=-1,
    frames_sample="middle",
    fix_start=None,
    input_fps=-1,
):
    """get_frame_indices"""
    assert frames_sample in ["rand", "middle", "leading"]
    if target_frames > 0:
        assert target_fps <= 0, "target_fps must be negative if target_frames is given."
        if target_frames > vlen:
            acc_samples = vlen
            logger.info(
                f"target_frames={target_frames} is larger than video length {vlen}, "
                f"will sample {acc_samples} frames."
            )
        else:
            acc_samples = target_frames
            logger.debug(f"sampling at target_frames={target_frames}, frames_sample={frames_sample}")

        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if frames_sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except Exception as e:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif frames_sample == "leading":
            frame_indices = [x[0] for x in ranges]
        elif frames_sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

    elif target_fps > 0:
        assert target_frames <= 0, "target_frames must be negative if target_fps is given."
        assert input_fps > 0, "input_fps must be provided if target_fps is given."
        logger.info(f"sampling at fps={target_fps}, frames_sample={frames_sample}")
        duration = float(vlen) / input_fps
        delta = 1 / target_fps  # gap between frames, this is also the clip length each frame represents
        if frames_sample == "middle":
            frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        elif frames_sample == "leading":
            frame_seconds = np.arange(0, duration, delta)
        if frames_sample == "rand":
            frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
            rand_offset = np.random.rand(*(frame_seconds.shape)) - 0.5
            frame_seconds += rand_offset * delta
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]

    else:
        raise ValueError("Must provide either positive target_fps or positive target_frames.")

    return frame_indices


def read_frames_decord(
    video_path,
    video_reader,
    video_meta,
    target_frames=-1,
    target_fps=-1,
    frames_sample="middle",
    fix_start=None,
    save_to_disk=False,
    cache_dir=EXTRACTED_FRAME_DIR,
    frame_indices=None,
    tol=10,
):
    """get frames by decord"""

    if frame_indices is None:
        frame_indices = get_frame_indices(
            video_meta["num_of_frame"],
            target_frames=target_frames,
            target_fps=target_fps,
            frames_sample=frames_sample,
            fix_start=fix_start,
            input_fps=video_meta["fps"],
        )

    frames = []
    for frame_indice_index in range(0, len(frame_indices)):
        frame_indice = frame_indices[frame_indice_index]
        try:
            frames.append(video_reader[frame_indice].asnumpy())  # (T, H, W, C)
        except Exception as e:
            logger.debug(f"encounter error when get frame: {frame_indice}, error: {e}")
            previous_counter = 1
            later_counter = 1
            previous_after_flag = True
            if frame_indice == 0 or frame_indice == len(video_reader) - 1:
                cur_tol = tol * 2
            else:
                cur_tol = tol
            while previous_counter < cur_tol or later_counter < cur_tol:
                if previous_after_flag:
                    if frame_indice - previous_counter < 0:
                        previous_counter += 1
                        previous_after_flag = not previous_after_flag
                        continue
                    try:
                        frames.append(video_reader[frame_indice - previous_counter].asnumpy())
                        logger.info(f"replace {frame_indice}-th frame with {frame_indice-previous_counter}-th frame")
                        frame_indices[frame_indice_index] = frame_indice - previous_counter
                        break
                    except Exception as e:
                        previous_counter += 1
                else:
                    if frame_indice + later_counter >= len(video_reader):
                        later_counter += 1
                        previous_after_flag = not previous_after_flag
                        continue
                    try:
                        frames.append(video_reader[frame_indice + later_counter].asnumpy())
                        logger.info(f"replace {frame_indice}-th frame with {frame_indice+later_counter}-th frame")
                        frame_indices[frame_indice_index] = frame_indice + later_counter
                        break
                    except Exception as e:
                        later_counter += 1
                previous_after_flag = not previous_after_flag

    frames = np.stack(frames, axis=0)
    assert len(frames) == len(frame_indices), f"len(frames): {len(frames)} != len(frame_indices): {len(frame_indices)}"

    ret = []

    url_sha1 = get_filename()
    for idx, frame in enumerate(frames):
        tmp = Image.fromarray(frame, "RGB")
        if save_to_disk:
            save_path = os.path.join(cache_dir, f"{url_sha1}", f"{idx}.png")
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            tmp.save(save_path)
            tmp = save_path
        ret.append(tmp)

    time_stamps = [frame_idx * video_meta["duration"] / video_meta["num_of_frame"] for frame_idx in frame_indices]

    return ret, frame_indices, time_stamps
