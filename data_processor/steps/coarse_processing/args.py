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
CoarseProcessorArguments

"""

from dataclasses import dataclass, field


@dataclass
class CoarseProcessorArguments:
    """
    args for CoarseProcessor
    """

    video_fps: int = field(default=-1, metadata={"help": "fps for sampling frames"})
    video_min_frames: int = field(
        default=-1, metadata={"help": "fps for sampling frames with min"}
    )
    video_max_frames: int = field(
        default=-1, metadata={"help": "fps for sampling frames with max"}
    )
    video_target_frames: int = field(
        default=-1, metadata={"help": "fps for sampling frames with target"}
    )
    video_frames_sample: str = field(
        default="middle", metadata={"help": " middle, rand, leading"}
    )
    video_use_asr: bool = field(default=False, metadata={"help": "whether to use asr"})

    def __post_init__(self):
        self.video_frames_sample = self.video_frames_sample.lower()
        assert self.video_frames_sample in ["middle", "rand", "leading"]
