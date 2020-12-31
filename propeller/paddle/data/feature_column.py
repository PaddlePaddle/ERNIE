#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""FeatureColumns and many Column"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import sys
import struct
from six.moves import zip, map
import itertools
import gzip
from functools import partial
import six
import logging

import numpy as np
from glob import glob

from propeller.data.feature_column import FeatureColumns as FCBase
from propeller.paddle.data.functional import Dataset
import multiprocessing

log = logging.getLogger(__name__)

__all__ = ['FeatureColumns']


class FeatureColumns(FCBase):
    """A Dataset Factory object"""

    def build_dataset(self, *args, **kwargs):
        """
        build `Dataset` from `data_dir` or `data_file`
        if `use_gz`, will try to convert data_files to gz format and save to `gz_dir`, if `gz_dir` not given, will create one.
        """
        ds = super(FeatureColumns, self).build_dataset(*args, **kwargs)
        ds.__class__ = Dataset
        return ds

    def build_dataset_from_stdin(self, *args, **kwargs):
        """doc"""
        ds = super(FeatureColumns, self).build_dataset_from_stdin(*args,
                                                                  **kwargs)
        ds.__class__ = Dataset
        return ds
