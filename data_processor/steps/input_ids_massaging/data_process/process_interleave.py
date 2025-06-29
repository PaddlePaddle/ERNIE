#!/usr/bin/env python3

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

""" Process Interleave"""
import copy
import hashlib
import json
import random

# sys.path.append("..")
from data_processor.steps.input_ids_massaging.data_utils import get_text_token_num, get_uniq_id
from data_processor.utils.random_context import RandomSeedContext


class InterleaveProcess:
    """Datae Process Interface"""

    def __init__(self, interleave_fiif, tokenizer, max_seq_len, cropper, **kwargs):
        self.interleave_fiif = interleave_fiif
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cropper = cropper

    def process(self, sample, **kwargs):
        """process"""
        if self.interleave_fiif:
            return self.normalize_interleave_fiif(sample)
        else:
            return sample

    def generate_short_hashid(self, url, length=4):
        """
        Generate a unique ID based on the URL using SHA-256 algorithm.
        """
        # use SHA-256
        hash_object = hashlib.sha256(url.encode())
        hex_dig = hash_object.hexdigest()
        # extract first 4 characters to generate id
        random_seed = get_uniq_id(url)
        with RandomSeedContext(random_seed):
            short_hashid = random.sample(hex_dig, min(length, len(hex_dig)))
        return "".join(short_hashid)

    def normalize_interleave_fiif(self, meta):
        """normalize_interleave"""
        tokenizer = self.tokenizer
        max_token_num = self.max_seq_len

        # sorted by matched_text_index
        image_info_clean = sorted(meta["image_info"], key=lambda x: x["matched_text_index"])

        text_info = meta["text_info"]
        image_index2item = {}
        for image in image_info_clean:
            image_index2item.setdefault(image["matched_text_index"], []).append(image)

        image_text_list = []
        for idx, item in enumerate(text_info):
            if idx in image_index2item:
                image_text_list.extend(image_index2item[idx])
            image_text_list.append(item)

        # calculate token_num
        token_counters = []
        image_names = set()
        url2name = {}
        for index, item in enumerate(image_text_list):
            if "text" in item:
                token_counters.append((get_text_token_num(tokenizer, item["text"]), "text"))
            else:
                # generate image hashid
                image_name = self.generate_short_hashid(item["image_url"])
                if image_name in image_names:
                    image_name = f"{index}_{image_name}"
                image_names.add(image_name)
                url2name[item["image_url"]] = image_name

                image_token_count = self.cropper.get_cropped_images_token_num([item])
                token_counters.append(
                    (image_token_count + get_text_token_num(tokenizer, f"[{image_name}]\n![{image_name}]"), "image")
                )

        # find valid windows
        valid_windows = []
        i = 0
        end = 0
        while i < len(token_counters):
            has_image = False
            has_text = False
            current_tokens = 0
            start = i
            images_ids = set()
            image_insert_id = 0
            for j in range(i, len(token_counters)):
                if current_tokens + token_counters[j][0] <= max_token_num:
                    current_tokens += token_counters[j][0]
                    if token_counters[j][1] == "image":
                        has_image = True  # contains image
                        images_ids.add(image_insert_id)
                    if token_counters[j][1] == "text":
                        has_text = True  # contains text
                        image_insert_id += 1
                    if j == len(token_counters) - 1 or current_tokens + token_counters[j + 1][0] > max_token_num:
                        if has_image and has_text:
                            is_end_image = token_counters[j][1] == "image"
                            is_interleave = len(images_ids) > 1 or next(iter(images_ids)) > 0
                            valid_windows.append((start, j, is_end_image, is_interleave))  # record window
                            end = j
                            i = end + 1
                        else:
                            i += 1
                        break
                else:
                    i += 1
                    break
            if end == len(token_counters) - 1:
                break
        windows = valid_windows
        if len(windows) == 0:
            # logging.info("No Windows Found")
            return []

        metas = []
        raw_image_text_list = image_text_list
        for window in windows:
            start, end, is_end_image, is_interleave = window
            text_info_new, image_info_new = [], []
            image_text_list = copy.deepcopy(raw_image_text_list)
            if is_interleave or is_end_image:
                for idx, item in enumerate(image_text_list[start : end + 1]):
                    if "text" in item:
                        text_info_new.append(item)
                    else:
                        image_name = url2name[item["image_url"]]
                        text_info_new.append({"text": f"![{image_name}]", "tag": "no_mask"})
                        image_info_new.append(item)
                # shuffle images
                random_seed = get_uniq_id(json.dumps(image_text_list[0]))
                with RandomSeedContext(random_seed):
                    random.shuffle(image_info_new)
                for index, item in enumerate(image_info_new):
                    image_name = url2name[item["image_url"]]
                    text_info_new.insert(index, {"text": f"[{image_name}]\n", "tag": "mask"})
                    item["matched_text_index"] = index
            else:
                # HARD CODE
                continue
            is_end = False
            if end == len(token_counters) - 1:
                is_end = True
            metas.append({"text_info": text_info_new, "image_info": image_info_new, "is_end": is_end})

        return metas
