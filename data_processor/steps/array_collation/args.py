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
ArrayCollationProcessorArguments

"""

from dataclasses import dataclass, field


@dataclass
class ArrayCollationProcessorArguments:
    """
    args for ArrayCollationProcessorArguments
    """

    pad_to_max_seqlen: int = field(default=32784, metadata={"help": "seqlen"})
    shift_label: bool = field(default=False, metadata={"help": "shift label or not"})
    combine_batch: int = field(default=1, metadata={"help": "combine_batch"})
    debug_print: int = field(default=0, metadata={"help": "debug print"})

    def __post_init__(self):

        self.pad_to_max_seqlen = self.pad_to_max_seqlen
