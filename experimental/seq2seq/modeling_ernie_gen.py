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

import sys
import logging
import numpy as np

import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D

from ernie.modeling_ernie import ErnieModel
from ernie.modeling_ernie import _build_linear, _build_ln, append_name

class ErnieModelForGeneration(ErnieModel):
    resource_map = {
        'ernie-gen-base-en': ErnieModel.bce + 'model-ernie-gen-base-en.1.tar.gz',
        'ernie-gen-large-en': ErnieModel.bce + 'model-ernie-gen-large-en.1.tar.gz',
    }
    def __init__(self, cfg, name=None):
        cfg['return_additional_info'] = True
        cfg['has_pooler'] = False
        super(ErnieModelForGeneration, self).__init__(cfg, name=name)
        initializer = F.initializer.TruncatedNormal(scale=cfg['initializer_range'])
        d_model = cfg['hidden_size']
        d_vocab = cfg['vocab_size']

        self.mlm = _build_linear(d_model, d_model, append_name(name, 'mask_lm_trans_fc'), initializer, act=cfg['hidden_act'])
        self.mlm_ln = _build_ln(d_model, name = append_name(name, 'mask_lm_trans'))
        self.mlm_bias = L.create_parameter(
                dtype='float32',
                shape=[d_vocab], 
                attr=F.ParamAttr(
                    name=append_name(name, 'mask_lm_out_fc.b_0'), 
                    initializer=F.initializer.Constant(value=0.0)
                    ),
                is_bias=True,
            )

    def forward(self, src_ids, *args, **kwargs):
        tgt_labels = kwargs.pop('tgt_labels', None)
        tgt_pos = kwargs.pop('tgt_pos', None)
        encode_only = kwargs.pop('encode_only', False)
        _, encoded, info = ErnieModel.forward(self, src_ids, *args, **kwargs)
        #log.debug('hidden_-1 %r'% L.reduce_mean(info['hiddens'][0]).numpy())
        #log.debug('hidden_0 %r'% L.reduce_mean(info['hiddens'][1]).numpy())
        if encode_only:
            return None, None, info
        elif tgt_labels is None:
            encoded = self.mlm(encoded)
            encoded = self.mlm_ln(encoded)
            logits = L.matmul(encoded, self.word_emb.weight, transpose_y=True) + self.mlm_bias
            output_ids = L.argmax(logits, -1)
            return output_ids, logits, info
        else:
            encoded_2d = L.gather_nd(encoded, tgt_pos)
            #log.debug('input shape %s' % repr(src_ids.shape))
            #log.debug(L.gather_nd(src_ids, tgt_pos).numpy())
            encoded_2d = self.mlm(encoded_2d)
            encoded_2d = self.mlm_ln(encoded_2d)
            logits_2d = L.matmul(encoded_2d, self.word_emb.weight, transpose_y=True) + self.mlm_bias
            if len(tgt_labels.shape) == 1:
                tgt_labels = L.reshape(tgt_labels, [-1, 1])
            
            loss = L.reduce_mean(
                    L.softmax_with_cross_entropy(logits_2d, tgt_labels, soft_label=(tgt_labels.shape[-1] != 1))
                    )
            return loss, logits_2d, info

