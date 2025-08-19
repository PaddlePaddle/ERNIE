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
End2EndProcessor

"""
import copy
import json

from data_processor.steps.coarse_processing import CoarseProcessor
from data_processor.steps.image_modification import ImageModificationProcessor
from data_processor.steps.input_ids_messaging import InputIdsMassageProcessor
from data_processor.steps.utterance_processing import UtteranceProcessor
from data_processor.utils.processor_base import ProcessorBase


class End2EndProcessor(ProcessorBase):
    """
    End2End Processor
    """

    def __init__(self, args, tokenizer, image_preprocess):
        """
        init
        """
        # (Liuting) magic number need fix
        super().__init__(args)
        if isinstance(args, tuple):
            self.utterance_process = UtteranceProcessor(args[0], tokenizer)
            self.coarse_processor = CoarseProcessor(args[1])
            self.input_ids_massage_processor = InputIdsMassageProcessor(
                args[2], tokenizer, image_preprocess
            )
            self.image_modification_processor = ImageModificationProcessor(
                args[3], tokenizer, image_preprocess
            )
            self.batch_size = args[4].batch_size
            self.load_args_from_api = args[4].load_args_from_api
        else:
            self.utterance_process = UtteranceProcessor(args, tokenizer)
            self.coarse_processor = CoarseProcessor(args)
            self.input_ids_massage_processor = InputIdsMassageProcessor(
                args, tokenizer, image_preprocess
            )
            self.image_modification_processor = ImageModificationProcessor(
                args, tokenizer, image_preprocess
            )
            self.batch_size = args.batch_size
            self.load_args_from_api = args.load_args_from_api

    def process(self, data, **kwargs):
        """
        process
        """
        # step1: utterance processing
        if not self.is_training and self.load_args_from_api:
            generation_config = copy.deepcopy(data)
            if "context" in generation_config:
                del generation_config["context"]
            kwargs.update(generation_config)
        schema = self.utterance_process.process(data, **kwargs)

        # step2: coarse processing
        schema = self.coarse_processor.process(schema, **kwargs)

        # step3:  ids massaging
        schemas = self.input_ids_massage_processor.process(schema, **kwargs)

        # step4: schemas to rets
        rets = []
        if isinstance(schemas, str):
            schemas = json.loads(schemas)
        if schemas is None:
            schemas = []
        for schema in schemas:
            rets.append(
                {
                    "meta": [schema["meta"]],
                    "ds16": schema["feature"]["ids"],
                    "ds16_lossmask": schema["feature"]["lossmask"],
                    "ds16_tokenwise_type_id": schema["feature"]["ids_type"],
                    "ds16_imagewise_type_id": schema["feature"]["image_wise_type"],
                }
            )

        # step5: image modification
        tensor = []
        for ret in rets:
            tensor.append(self.image_modification_processor.process(ret, **kwargs))

        return tensor

    def streaming_get_batch(self, iterable):
        """
        streaming get batch
        """
        buffer = []
        for item in iterable:
            tensors = self.process(item)
            buffer.extend(tensors)

            while len(buffer) >= self.batch_size:
                yield self.collate(buffer[: self.batch_size])
                buffer = buffer[self.batch_size :]

        # empty the buffer
        while len(buffer) != 0:
            yield self.collate(buffer[: self.batch_size])
            buffer = buffer[self.batch_size :]
