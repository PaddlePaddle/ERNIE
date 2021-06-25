#    Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" text preprocess """

import random
import sys
import os
import base64
import numpy as np
reload(sys)
sys.setdefaultencoding("utf-8")

from preprocess import tokenization

class PreprocessorBasic(object):
    """
    parent class for preprocess
    """
    def __init__(self,
                 tokenizer_name,
                 vocab_path,
                 tagger_path="",
                 nltk_data_path="",
                 do_lower_case=True):
        self.do_lower_case = do_lower_case
        self.tokenizer = getattr(tokenization, tokenizer_name)(vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
    
    def convert_sentence_to_ids_without_cls(self, sentence):
        """
        convert sentence to ids without cls
        """
        tokens = self.tokenizer.tokenize(sentence)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids
