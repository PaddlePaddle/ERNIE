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
PseudoMultiRoundProcessor
"""
import json

from data_processor.utils.processor_base import ProcessorBase


class PseudoMultiRoundProcessor(ProcessorBase):
    """
    PseudoMultiRoundProcessor
    """

    def __init__(self, args, tokenizer):
        """
        init
        """
        super().__init__(args)

        self.tokenizer = tokenizer
        self.buffer_seq_len = 0
        self.buffer = []

    def concat_multiround(self, buffer_data):
        """
        concat_multiround
        """
        buffer_ids = []
        buffer_lossmask = []
        buffer_tokenwise_type_ids = []
        ret = {"meta": []}
        buffer_imagewise_type_ids = []

        for item in buffer_data:
            assert len(item["feature"]["ids"]) > 0
            assert len(item["feature"]["ids"]) == len(item["feature"]["lossmask"])
            assert len(item["feature"]["lossmask"]) == len(item["feature"]["ids_type"])
            buffer_ids.extend(item["feature"]["ids"])
            buffer_lossmask.extend(item["feature"]["lossmask"])
            buffer_tokenwise_type_ids.extend(item["feature"]["ids_type"])

            buffer_imagewise_type_ids.extend(item["feature"]["image_wise_type"])

            ret["meta"].append(item["meta"])
        assert len(buffer_ids) > 0
        assert len(buffer_lossmask) > 0
        assert len(buffer_tokenwise_type_ids) > 0

        ret["ds16"] = buffer_ids
        ret["ds16_lossmask"] = buffer_lossmask
        ret["ds16_tokenwise_type_id"] = buffer_tokenwise_type_ids

        if len(buffer_imagewise_type_ids) > 0:
            ret["ds16_imagewise_type_id"] = buffer_imagewise_type_ids
        return ret

    def process(self, datas, **kwargs):
        """
        datas: must be a list of data
        """
        results = []
        if isinstance(datas, str):
            datas = json.loads(datas)
        if datas is None:
            datas = []
        for data in datas:
            if not self.is_training or not self.is_pretraining:
                assert len(self.buffer) == 0 and self.buffer_seq_len == 0
                self.buffer.append(data)
                results.append(self.concat_multiround(self.buffer))
                self.buffer = []
            else:
                if self.buffer_seq_len + len(data["feature"]["ids"]) > self.args.span_length:
                    results.append(self.concat_multiround(self.buffer))

                    self.buffer = []
                    self.buffer_seq_len = 0

                self.buffer.append(data)
                self.buffer_seq_len += len(data["feature"]["ids"])

        if len(self.buffer) > 0:
            results.append(self.concat_multiround(self.buffer))
        return results

    def process_generate(self, datas, **kwargs):
        """
        datas: must be a list of data
        """
        eos_token_id = self.tokenizer.eos_token_id
        is_skip_noimage = kwargs.get("is_skip_noimage", False)

        for data_item in datas:
            data = json.loads(data_item)

            if is_skip_noimage:
                if len(data["feature"]["image_wise_type"]) == 0:
                    continue

            if not self.is_training:
                assert len(self.buffer) == 0 and self.buffer_seq_len == 0
                self.buffer.append(data)
                yield self.concat_multiround(self.buffer)
                self.buffer = []
            else:
                data_end_id = data["feature"]["ids"][-1]
                if self.buffer_seq_len + len(data["feature"]["ids"]) > self.args.span_length:

                    yield self.concat_multiround(self.buffer)

                    self.buffer = []
                    self.buffer_seq_len = 0

                self.buffer.append(data)
                self.buffer_seq_len += len(data["feature"]["ids"])

                if data_end_id != eos_token_id and self.is_pretraining:
                    yield self.concat_multiround(self.buffer)

                    self.buffer = []
                    self.buffer_seq_len = 0

        if len(self.buffer) > 0:
            yield self.concat_multiround(self.buffer)
