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
coarse_processor

"""

from data_processor.steps.coarse_processing import data_process
from data_processor.utils.processor_base import ProcessorBase


class CoarseProcessor(ProcessorBase):
    """
    coarse processor
    """

    def _select_processor(self, schema):
        """
        select processor according to the schema
        """
        if len(schema.get("video_info", [])) > 0:
            return data_process.VideoCoarseProcessor
        return data_process.IdentityProcessor

    def process(self, data, **kwargs):
        """
        process the data
        """
        # make sure the data is in schema format
        schema = data
        obj_process = self._select_processor(schema)
        processor = obj_process(**vars(self.args))

        return processor.process(schema, **kwargs)
