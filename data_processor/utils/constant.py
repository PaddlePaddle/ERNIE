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
Constants for dataset
"""

IDTYPES_2_ID = {"text": 0, "image": 1, "video": 2}
IMAGETYPES_2_ID = {"image": 0, "video": 1, "padded_image": 2}
DATATYPE_2_ID = {"mm": 0, "lm": 1}
MIN_CROP_NUM = 4  # only support 4 now
MIN_CROP_SUPPORT_RANGE = [4]  # only support 4 now
MAX_RATIO = 50
GIVEN_MAX_TILE = 144

# map data_type to dataset_type
DATA_TYPE_TO_DATASET_TYPE = {
    "image": ["image-text-pair", "image-text_location-pair", "interleave", "default"],
    "video": ["video"],
    "omini": ["omini"],
}
DATASET_TYPE_TO_DATA_TYPE = {
    dataset_type: data_type
    for data_type, dataset_types in DATA_TYPE_TO_DATASET_TYPE.items()
    for dataset_type in dataset_types
}

# data_type to augment function
AUGMENT_FN = {"image": "AugmentImage"}

# determined by dataset_type
PROCESS_FN_TO_DATASET_TYPE = {
    "LocationProcess": ["image-text_location-pair"],
    "VideoProcess": ["video"],
    "InterleaveProcess": ["interleave"],
}
DATASET_TYPE_TO_PROCESS_FN = {
    dataset_type: process_fn
    for process_fn, dataset_types in PROCESS_FN_TO_DATASET_TYPE.items()
    for dataset_type in dataset_types
}

# determined by data_type
CROP_FN = {
    "image": "CropImage",
    "video": "CropVideo",
    "adaptive": "CropAdaptive",
}


CUT_FLAG = {"cut": 0, "no_cut": 1}


IDS_TYPE_FLAG = {"text": 0, "image": 1, "video": 2}


IMAGE_TYPE_FLAG = {"image": 0, "video": 1, "padded_image": 2}
