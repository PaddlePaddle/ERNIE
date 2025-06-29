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

"""Image Text-Location Pair Process

schema

text_info: [
    {
        "text": "OCR with grounding:",
        "tag": "mask"
    },
    {
        "text": "xxx",
        "tag": "no_mask",
        "points": [
            [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        ]
    },
    {
        "text": "yyy",
        "tag": "no_mask",
        "points": [
            [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        ]
    },
    ...
]

1. norm coordinate to (0, 1000)
2. coordinate format

text_info: [
    {
        "text": "OCR with grounding:",
        "tag": "mask"
    },
    {
        "text": "<ref>xxx</ref><box>(x1,y1)(x2,y2)(x3,y3)(x4,y4)</box>
        <ref>yyy</ref><box>(x1,y1)(x2,y2)(x3,y3)(x4,y4)</box>",
        "tag": "no_mask",
    },
]
"""
import sys

from data_processor.steps.input_ids_massaging.data_process.process import Process


class LocationProcess(Process):
    """LocationProcess"""

    def __init__(
        self,
        coordinate_max: int = 1000,
        use_loc_specialtoken: bool = False,
        special_tokens_info=None,
        remove_loc: int = 0,
        **kwargs,
    ):
        """
        init method
        """
        self.coordinate_max = coordinate_max
        self.use_loc_specialtoken = use_loc_specialtoken
        self.remove_loc = remove_loc

        self.ref_begin, self.ref_end = "<ref>", "</ref>"
        self.loc_begin, self.loc_end, self.box_sep = special_tokens_info["loc_begin_end"]

    def process(self, sample, dataset_name, **kwargs):
        """process"""
        self.dataset_name = dataset_name

        assert (
            len(sample["image_info"]) == 1
        ), f"[ERROR] image-text_location-pair only support one image, \
            current nums {len(sample['image_info'])}: {sample}"
        self.image_width = sample["image_info"][0]["image_width"]
        self.image_height = sample["image_info"][0]["image_height"]

        text_loc_info = []

        for i, text_one in enumerate(sample["text_info"]):

            points = text_one.get("points", None)
            if points is None:
                # text without points, e.g. prompt or plain text in grounded caption
                text_loc_info.append(text_one)
                continue
            content = text_one.get("text", "")
            if len(content) == 0:
                print(
                    f"""[WARNING] {self.dataset_name} schema location text not exist, skip its points\n
                    text: {text_one}\nsample: {sample}""",
                    file=sys.stderr,
                )
                text_loc_info.append(text_one)
                continue
            if isinstance(points, list) and len(points) == 0:
                print(
                    f"""[WARNING] {self.dataset_name} schema loc text_info has not points!\n
                    text: {text_one}\nsample: {sample}""",
                    file=sys.stderr,
                )
            if self.remove_loc:
                # ocr without box, \n join context
                text_wo_loc = content.strip()
                if i != len(sample["text_info"]) - 1:
                    text_wo_loc += '\n'

                text_loc_info.append({"text": text_wo_loc, "tag": "no_mask"})
            else:
                # norm
                norm_points = self._norm_coordinate(points)

                # format
                if self.use_loc_specialtoken:
                    text_loc = self._point_to_special_box(content, norm_points)
                else:
                    text_loc = self._point_to_refbox(content, norm_points)

                text_loc_info.append({"text": text_loc.strip(), "tag": "no_mask"})

        sample_loc = {"image_info": sample["image_info"], "text_info": text_loc_info}
        return sample_loc

    def _norm_coordinate(self, points):
        """norm coordinate"""
        norm_points = []
        for point in points:
            norm_one = []
            for x, y in point:
                if x < 0:
                    print(
                        f"[WARNING] {self.dataset_name} x={x} must be greater than zero ! reset to 0", file=sys.stderr
                    )
                    x = 0
                if y < 0:
                    print(
                        f"[WARNING] {self.dataset_name} y={y} must be greater than zero ! reset to 0", file=sys.stderr
                    )
                    y = 0

                # norm
                norm_x = int(int(x) / self.image_width * self.coordinate_max)
                norm_y = int(int(y) / self.image_height * self.coordinate_max)
                norm_one.append([norm_x, norm_y])
            norm_points.append(norm_one)
        return norm_points

    def _point_to_refbox(self, content, points):
        """points to refbox"""
        box_str = []
        for point in points:
            box = "".join([f"({x},{y})" for x, y in point])
            box_str.append(f"<box>{box}</box>")
        box_str = "".join(box_str)
        return f"<ref>{content}</ref>{box_str}"

    def _point_to_special_box(self, content, points):
        """
        points to special token box
        """
        box_str = []
        for point in points:
            box_str.append("".join([f"<|LOC_{x}|><|LOC_{y}|>" for x, y in point]))
        box_str = self.box_sep.join(box_str)
        return f"{self.ref_begin}{content}{self.ref_end}{self.loc_begin}{box_str}{self.loc_end}"
