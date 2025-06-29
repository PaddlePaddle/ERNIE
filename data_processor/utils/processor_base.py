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
ProcessorBase

"""


class ProcessorBase:
    """
    ProcessorBase
    """

    def __init__(self, args):
        """
        init
        """
        self.args = args
        self.is_training = True
        self.is_pretraining = True

    def process(self, data):
        """
        process
        """
        return NotImplementedError

    def eval(self):
        """
        eval mode
        """
        for i in self.get_processors():
            i.eval()
        self.is_training = False
        return self

    def train(self):
        """
        train mode
        """
        for i in self.get_processors():
            i.train()
        self.is_training = True
        return self

    def pretrain(self):
        """
        pretrain mode
        """
        for i in self.get_processors():
            i.pretrain()
        self.is_pretraining = True
        return self

    def sft(self):
        """
        sft mode
        """
        for i in self.get_processors():
            i.sft()
        self.is_pretraining = False
        return self

    def get_processors(self):
        """
        get_processors
        """
        return [var for var_name, var in vars(self).items() if isinstance(var, ProcessorBase)]
