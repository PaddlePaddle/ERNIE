#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import paddle
paddle_version = [int(i) for i in paddle.__version__.split('.')]
if paddle_version[1] < 7:
    raise RuntimeError('paddle-ernie requires paddle 1.7+, got %s' %
                       paddle.__version__)

from ernie.modeling_ernie import ErnieModel
from ernie.modeling_ernie import (ErnieModelForSequenceClassification, 
        ErnieModelForTokenClassification, 
        ErnieModelForQuestionAnswering, 
        ErnieModelForPretraining)

from ernie.tokenizing_ernie import ErnieTokenizer, ErnieTinyTokenizer

