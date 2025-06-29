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
schema_utils
"""

import logging
import os

logger = logging.getLogger(__name__)
validate = None
ValidationError = None


def get_format_type(data):
    """
    return whether the data is schema or utterance
    """
    if "text_list" in data or "text_info" in data:
        return "schema"
    else:
        return "utterance"


def remove_info(schema, info_name):
    """
    remove the info from schema
    """
    assert info_name in schema.keys()
    for item in schema[info_name]:
        if "image_url" in item:
            image_url = item["image_url"]
            if isinstance(image_url, str):
                if os.path.exists(image_url) and os.path.isfile(image_url):
                    os.remove(item["image_url"])

    del schema[info_name]
    return schema


def check_schema_format(sample, data_name, data_type):
    """mapper check for multi modal"""
    required_keys = []
    if data_type in ["image", "video"]:
        required_keys.extend(["text_info", "image_info"])
    MULTI_MODAL_SCHEMA = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
            },
            "text_info": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "tag": {
                            "type": "string",
                        },
                        "points": {
                            "type": ["array", "null"],
                            "minItems": 1,
                            "items": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 4,
                                "items": {
                                    "type": "array",
                                    "minItems": 2,
                                    "maxItems": 2,
                                    "item": {"type": "integer"},
                                },
                            },
                        },
                    },
                    "required": ["text", "tag"],
                },
            },
            "image_info": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "image_url": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "matched_text_index": {"type": "integer"},
                        "image_width": {"type": ["integer", "null"]},
                        "image_height": {"type": ["integer", "null"]},
                        "is_valid": {"type": ["boolean", "null"]},
                    },
                    "if": {"properties": {"is_valid": {"const": True}}},
                    "then": {
                        "properties": {
                            "image_width": {"type": "integer"},
                            "image_height": {"type": "integer"},
                        },
                    },
                    "else": {
                        "properties": {
                            "image_width": {"type": ["integer", "null"]},
                            "image_height": {"type": ["integer", "null"]},
                        }
                    },
                    "required": [
                        "image_url",
                        "matched_text_index",
                        "image_width",
                        "image_height",
                        "is_valid",
                    ],
                },
            },
            "video_info": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "image_url": {
                            "type": "string",
                            "minLength": 1,
                            "oneOf": [
                                {"pattern": "^http://.*"},
                            ],
                        },
                        "image_width": {"type": ["integer", "null"]},
                        "image_height": {"type": ["integer", "null"]},
                        "duration": {"type": ["number", "null"]},
                        "bitrate": {"type": ["number", "null"]},
                        "fps": {"type": ["number", "null"]},
                    },
                    "required": ["image_url"],
                },
            },
        },
        "required": required_keys,
    }

    assert data_type in [
        "image",
        "video",
    ], f"[ERROR] {data_name} schema no support {data_type}!"

    assert validate(sample, MULTI_MODAL_SCHEMA) is None, f"[ERROR] {data_name} schema error!\n{ValidationError}"

    for one in sample.get("text_info", []):
        points = one.get("points", None)
        if points:
            for point in points:
                for p in point:
                    assert (
                        len(p) == 2 or len(p) == 4
                    ), f"[ERROR] {data_name} point={point}, \
                        The coordinates cannot be three-point coordinates!"


def schema_to_sequence(schema, serialize=False):
    """
    dummy
    """
    image_info = schema["image_info"]
    text_info = schema["text_info"]
    image_cnt = 0
    result = []
    for text_index, text_one in enumerate(text_info):
        while image_cnt < len(image_info) and image_info[image_cnt]["matched_text_index"] == text_index:
            if serialize:
                result.append(f"<|IMAGE@? [timestamp: {image_info[image_cnt].get('time_stamp', -1)}]|>")
            else:
                result.append(image_info[image_cnt])
            image_cnt += 1
        if serialize:
            result.append(f"{text_one['text']}")
        else:
            result.append(text_one)

    return result
