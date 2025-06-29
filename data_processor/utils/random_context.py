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
RandomSeedContext
"""

import random

import numpy as np


class RandomSeedContext:
    """random context guard for imgaug"""

    def __init__(self, seed):
        self.seed = seed
        self.original_numpy_seed = None
        self.original_random_seed = None
        self.original_imgaug_seed = None

    def __enter__(self):
        self.original_numpy_seed = np.random.get_state()
        self.original_random_seed = random.getstate()

        np.random.seed(self.seed)
        random.seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        np.random.set_state(self.original_numpy_seed)
        random.setstate(self.original_random_seed)
