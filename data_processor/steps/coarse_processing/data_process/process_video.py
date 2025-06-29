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
VideoCoarseProcessor
"""
import io
import os
import random
import uuid

import numpy as np
from PIL import Image

from data_processor.utils.io_utils import EXTRACTED_FRAME_DIR, get_downloadable, get_filename
from data_processor.utils.logger_utils import logger
from data_processor.utils.processor_base import ProcessorBase
from data_processor.utils.relief import omini_convert_schema_to_sequence, omini_convert_sequence_to_schema
from data_processor.utils.video_utils import VideoReaderWrapper


class VideoCoarseProcessor(ProcessorBase):
    """
    VideoCoarseProcessor
    """

    def __init__(
        self,
        video_fps=-1,
        video_min_frames=-1,
        video_max_frames=-1,
        video_target_frames=-1,
        video_frames_sample="middle",
        video_use_asr=False,
        **kwargs,
    ):
        super().__init__(None)
        self.video_fps = video_fps
        self.video_min_frames = video_min_frames
        self.video_max_frames = video_max_frames
        self.video_target_frames = video_target_frames
        self.video_frames_sample = video_frames_sample
        self.video_use_asr = video_use_asr

    def set_video_frame_args(self, video_frame_args, video_meta):
        """
        set video frame args
        """
        # priority：video_target_frames > (video_min_frames, video_max_frames) > video_fps
        if video_frame_args["target_frames"] > 0:
            if video_frame_args["fps"] >= 0:
                raise ValueError("fps must be negative if target_frames is given")
            if (
                video_frame_args["min_frames"] > 0
                and video_frame_args["target_frames"] < video_frame_args["min_frames"]
            ):
                raise ValueError("target_frames must be larger than min_frames")
            if (
                video_frame_args["max_frames"] > 0
                and video_frame_args["target_frames"] > video_frame_args["max_frames"]
            ):
                raise ValueError("target_frames must be smaller than max_frames")
        else:
            if video_frame_args["fps"] < 0:
                raise ValueError("Must provide either positive target_fps or positive target_frames.")
            frames_to_extract = int(video_meta["duration"] * video_frame_args["fps"])

            if (
                video_frame_args["min_frames"] > 0
                and video_frame_args["max_frames"] > 0
                and video_frame_args["min_frames"] > video_frame_args["max_frames"]
            ):
                raise ValueError("min_frames must be smaller than max_frames")
            if video_frame_args["min_frames"] > 0 and frames_to_extract < video_frame_args["min_frames"]:
                logger.debug(
                    f"fps={video_frame_args['fps']} too low for min_frames={video_frame_args['min_frames']}, "
                    f"set target_frames={video_frame_args['min_frames']}"
                )
                video_frame_args["target_frames"] = video_frame_args["min_frames"]
                video_frame_args["fps"] = -1
            if video_frame_args["max_frames"] > 0 and frames_to_extract > video_frame_args["max_frames"]:
                logger.debug(
                    f"fps={video_frame_args['fps']} too large for max_frames={video_frame_args['max_frames']},"
                    f" set target_frames={video_frame_args['max_frames']}"
                )
                video_frame_args["target_frames"] = video_frame_args["max_frames"]
                video_frame_args["fps"] = -1

        return video_frame_args

    def process(self, schema, save_to_disk=False, clean_up=True, **kwargs):
        """
        video processor
        """
        sequence = omini_convert_schema_to_sequence(schema)
        # sequnce is in format
        video_frame_args = dict()
        video_frame_args["fps"] = kwargs.get("video_fps", self.video_fps)
        video_frame_args["min_frames"] = kwargs.get("video_min_frames", self.video_min_frames)
        video_frame_args["max_frames"] = kwargs.get("video_max_frames", self.video_max_frames)
        video_frame_args["target_frames"] = kwargs.get("video_target_frames", self.video_target_frames)
        video_frame_args["frames_sample"] = kwargs.get("video_frames_sample", self.video_frames_sample)

        new_sequence = []

        for element in sequence:
            if element[0] != "video":
                # only process the element that is video
                new_sequence.append(element)
                continue

            # now it comes into video
            video_one = element[1]
            uid = str(uuid.uuid4())
            # first get video basic info, then set frame args
            video_path = video_one["image_url"]
            video_reader, video_meta, video_path = read_video_decord(video_path, save_to_disk=save_to_disk)
            video_frame_args = self.set_video_frame_args(video_frame_args, video_meta)

            ret, frame_indices, time_stamps = read_frames_decord(
                video_path,
                video_reader,
                video_meta,
                target_frames=video_frame_args["target_frames"],
                target_fps=video_frame_args["fps"],
                frames_sample=video_frame_args["frames_sample"],
                fix_start=None,
                save_to_disk=save_to_disk,
                frame_indices=video_one["extracted_frame_indices"] if "extracted_frame_indices" in video_one else None,
            )

            assert len(time_stamps) == len(ret)
            asr_cnt = 0
            for img_idx, (img, time_stamp) in enumerate(zip(ret, time_stamps)):
                # no need matched text any more
                image_ele = {
                    "image_url": img,
                    "image_width": video_one["image_width"],
                    "image_height": video_one["image_height"],
                    "is_valid": True,
                    "image_type": "video",
                    "time_stamp": time_stamp,
                    "video_uid": uid,
                }
                new_sequence.append(("image", image_ele))  # make video into image frame element

                # add asr right after the frame

                if self.video_use_asr and "asr" in video_one:
                    # asr's format: [subtitle, start_second, end_second]
                    asr = video_one["asr"]

                    while asr_cnt < len(video_one["asr"]) and asr[asr_cnt][-1] < time_stamp:
                        time_start = round(asr[asr_cnt][1], 1)
                        time_end = round(asr[asr_cnt][2], 1)
                        asr_text = asr[asr_cnt][0].strip()

                        text_ele = {
                            "text": f"[{time_start},{time_end}]{asr_text}",
                            "tag": "mask",
                            "is_asr": True,
                        }
                        new_sequence.append(("text", text_ele))  # convert it into asr text ele
                        asr_cnt += 1

            # if there is some asr left, take it into account
            if self.video_use_asr and "asr" in video_one and asr_cnt < len(video_one["asr"]):
                text_to_add = []
                asr = video_one["asr"]
                while asr_cnt < len(video_one["asr"]):
                    time_start = round(asr[asr_cnt][1], 1)
                    time_end = round(asr[asr_cnt][2], 1)
                    asr_text = asr[asr_cnt][0].strip()

                    text_ele = {
                        "text": f"[{time_start},{time_end}]{asr_text}",
                        "tag": "mask",
                        "is_asr": True,
                    }
                    new_sequence.append(("text", text_ele))  # convert it into asr text ele
                    asr_cnt += 1

        schema = omini_convert_sequence_to_schema(new_sequence)
        return schema


def read_video_decord(video_path, save_to_disk):
    """get reader and meta by decord"""
    data_in_mem = False
    video_path = get_downloadable(video_path, save_to_disk=save_to_disk)
    if isinstance(video_path, VideoReaderWrapper):
        data_in_mem = True
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
    """
    return frame index
    """
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
