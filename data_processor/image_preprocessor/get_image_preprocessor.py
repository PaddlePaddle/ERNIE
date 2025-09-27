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

"""get image preprocessor"""
from paddleformers.utils.log import logger

from .image_preprocessor_adaptive import AdaptiveImageProcessor


def get_image_preprocessor(args):
    """
    get_image_preprocessor from args
    """

    if args.model_name_or_path is None:
        return None

    logger.info("use AdaptiveImageProcessor")
    image_preprocess = AdaptiveImageProcessor.from_pretrained(args.model_name_or_path)
    return image_preprocess
