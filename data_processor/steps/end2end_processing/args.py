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
End2EndProcessorArgumentsHelper

"""
from dataclasses import dataclass, field

from data_processor.steps.coarse_processing import CoarseProcessorArguments
from data_processor.steps.image_modification import ImageModificationProcessorArguments
from data_processor.steps.input_ids_messaging import (
    InputIdsMassageArguments,
    InputIdsMassageInferArguments,
)
from data_processor.steps.utterance_processing import UtteranceProcessorArguments


@dataclass
class End2EndProcessorArgumentsHelper:
    """
    args for End2EndProcessorArgumentsHelper
    """

    batch_size: int = field(default=1, metadata={"help": "batch size"})
    load_args_from_api: bool = field(
        default=False, metadata={"help": "load arguments from api"}
    )


End2EndProcessorArguments = (
    UtteranceProcessorArguments,
    CoarseProcessorArguments,
    InputIdsMassageArguments,
    ImageModificationProcessorArguments,
    End2EndProcessorArgumentsHelper,
)

End2EndProcessorInferArguments = (
    UtteranceProcessorArguments,
    CoarseProcessorArguments,
    InputIdsMassageInferArguments,
    ImageModificationProcessorArguments,
    End2EndProcessorArgumentsHelper,
)
