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

""" relief """
from collections import defaultdict

from data_processor.utils.logger_utils import logger


def has_valid_order(schema):
    """check whether a schema contains valid order"""
    if "order" not in schema:
        return False

    order = schema["order"]
    if len(order["type"]) != len(order["index"]) or len(order["type"]) != len(order["mask"]):
        return False

    item_keys = ["text", "image", "video"]
    num_items = sum([len(schema.get(f"{key}_info", [])) for key in item_keys])
    if num_items != len(order["type"]):
        return False

    return True


def omini_convert_schema_to_sequence(schema):
    """convert the schema format into sequcne"""
    sequence = []

    if has_valid_order(schema):
        logger.debug("convert schema into sequence according to order")
        # reconstruct sequence according to order
        for data_type, data_ind in zip(schema["order"]["type"], schema["order"]["index"]):
            sequence.append((data_type, schema[f"{data_type}_info"][data_ind]))

    else:
        logger.debug("no valid order, convert schema into sequence with default order")

        for tid, text in enumerate(schema["text_info"]):
            sequence.append((tid, 1, tid, "text", text))

        for data_type in ["image", "video"]:
            for i, item in enumerate(schema.get(f"{data_type}_info", [])):
                sequence.append((item.pop("matched_text_index", 0), 0, i, data_type, item))

        sequence = sorted(sequence, key=lambda x: (x[0], x[1], x[2]))
        sequence = [(x[-2], x[-1]) for x in sequence]

    return sequence


def omini_convert_sequence_to_schema(sequence):
    """convert sequence bad to the schema"""
    ret = defaultdict(lambda: [])
    ret["order"] = {"type": [], "index": [], "mask": []}

    for data_type, element in sequence:
        if data_type == "text":
            ret["text_info"].append(element)
        elif data_type == "image":
            element["matched_text_index"] = len(ret["text_info"])
            ret["image_info"].append(element)
        elif data_type == "video":
            element["matched_text_index"] = len(ret["text_info"])
            ret["video_info"].append(element)
        ret["order"]["type"].append(data_type)
        ret["order"]["index"].append(len(ret[f"{data_type}_info"]) - 1)
        ret["order"]["mask"].append(int(element.get("tag", "mask") == "mask"))

    return ret
