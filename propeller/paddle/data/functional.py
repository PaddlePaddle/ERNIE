# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Pyreader based Dataset"""

import sys
import numpy as np
import logging

import paddle.fluid as F
import paddle.fluid.layers as L

from propeller.data.functional import Dataset as DatasetBase

log = logging.getLogger(__name__)


class Dataset(DatasetBase):
    """Pyreader based Dataset"""

    def placeholders(self):
        """doc"""
        if self.name is None:
            raise ValueError('can not get feature from unnamed Dataset')

        ret = []
        for i, (shape,
                types) in enumerate(zip(self.data_shapes, self.data_types)):
            ret.append(
                L.data(
                    '%s_placeholder_%d' % (self.name, i),
                    shape=shape,
                    append_batch_size=False,
                    dtype=types))
        return ret

    def features(self):
        """start point of net building. call this in a program scope"""
        if self.name is None:
            raise ValueError('can not get feature from unnamed Dataset')

        if len(self.data_shapes) != len(self.data_types):
            raise ValueError(
                'Dataset shapes and types not match: shape:%s types%s' %
                (repr(self._data_shapes), repr(self._data_types)))
        return self.placeholders()

    def start(self, places=None):
        """start Pyreader"""
        if places is None:
            places = F.cuda_places() if F.core.is_compiled_with_cuda(
            ) else F.cpu_places()
        #assert self.pyreader is not None, 'use Dataset.features to build net first, then start dataset'
        def _gen():
            try:
                for idx, i in enumerate(self.generator()):
                    yield i
            except Exception as e:
                log.exception(e)
                raise e

        r = F.io.PyReader(
            feed_list=self.placeholders(), capacity=50, iterable=True)
        r.decorate_batch_generator(_gen, places=places)
        return r()
