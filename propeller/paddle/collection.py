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
"""global collections"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import sys

_global_collection = None


class Key(object):
    """predefine collection keys"""
    SUMMARY_SCALAR = 1
    SUMMARY_HISTOGRAM = 2
    SKIP_OPTIMIZE = 3


class Collections(object):
    """global collections to record everything"""

    def __init__(self):
        self.col = {}

    def __enter__(self):
        global _global_collection
        _global_collection = self
        return self

    def __exit__(self, err_type, err_value, trace):
        global _global_collection
        _global_collection = None

    def add(self, key, val):
        """doc"""
        self.col.setdefault(key, []).append(val)

    def get(self, key):
        """doc"""
        return self.col.get(key, None)


def default_collection():
    """return global collection"""
    global _global_collection
    if _global_collection is None:
        _global_collection = Collections()
    return _global_collection
