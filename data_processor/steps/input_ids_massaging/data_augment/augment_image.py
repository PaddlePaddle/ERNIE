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

"""AugmentImage"""
import json

from data_processor.steps.input_ids_massaging.data_augment.augment import Augment
from data_processor.steps.input_ids_massaging.data_utils import get_uniq_id, image_to_json_serializable
from data_processor.utils.image_enhance import ImageEnhance
from data_processor.utils.image_utils import random_resize_img


class AugmentImage(Augment):
    """AugmentImage"""

    def __init__(self, dataset_type, operator_types, config_path, random_resize, **kwargs):
        self.image_enhance = ImageEnhance(config_path)
        self.operator_types = operator_types
        self.dataset_type = dataset_type
        self.random_resize = random_resize

    def process(self, sample):
        """process"""

        # random resize
        sample = random_resize_img(
            sample,
            min_resize_ratio=self.random_resize[0],
            max_resize_ratio=self.random_resize[1],
        )

        # generate random seed
        text_info = json.dumps([t['text'] for t in sample.get('text_info', [])])
        image_info = json.dumps([image_to_json_serializable(t['image_url']) for t in sample.get('image_info', [])])
        random_seed = get_uniq_id(text_info + image_info)
        sample = self.image_enhance.generate_augment_strategies(
            sample,
            dataset_type=self.dataset_type,
            random_seed=random_seed,
            operator_types=self.operator_types,
        )
        return sample
