#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Ernie model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import json
import six
import logging
import paddle.fluid as fluid
from io import open
from paddle.fluid.layers import core

from model.transformer_encoder import encoder, pre_process_layer

log = logging.getLogger(__name__)

class ErnieConfig(object):
    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf8') as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing Ernie model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict.get(key, None)

    def print_config(self):
        for arg, value in sorted(six.iteritems(self._config_dict)):
            log.info('%s: %s' % (arg, value))
        log.info('------------------------------------------------')


class ErnieModel(object):
    def __init__(self,
                 src_ids,
                 position_ids,
                 input_mask,
                 config,
                 weight_sharing=True,
                 use_fp16=False):

        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._voc_size = config['vocab_size']
        self._max_position_seq_len = config['max_position_embeddings']
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self._weight_sharing = weight_sharing

        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        
        self._dtype = core.VarDesc.VarType.FP16 if use_fp16 else core.VarDesc.VarType.FP32
        self._emb_dtype = core.VarDesc.VarType.FP32

        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config['initializer_range'])

        self._build_model(src_ids, position_ids, input_mask)

    def _build_model(self, src_ids, position_ids, input_mask):
        # padding id in vocabulary must be set to 0
        emb_out = fluid.layers.embedding(
            input=src_ids,
            size=[self._voc_size, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._word_emb_name, initializer=self._param_initializer),
            is_sparse=False)
        
        position_emb_out = fluid.layers.embedding(
            input=position_ids,
            size=[self._max_position_seq_len, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._pos_emb_name, initializer=self._param_initializer))


        emb_out = emb_out + position_emb_out

        emb_out = pre_process_layer(
            emb_out, 'nd', self._prepostprocess_dropout, name='pre_encoder')

        if self._dtype == core.VarDesc.VarType.FP16:
            emb_out = fluid.layers.cast(x=emb_out, dtype=self._dtype)
            input_mask = fluid.layers.cast(x=input_mask, dtype=self._dtype)
        self_attn_mask = fluid.layers.matmul(
            x=input_mask, y=input_mask, transpose_y=True)

        self_attn_mask = fluid.layers.scale(
            x=self_attn_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = fluid.layers.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        self._enc_out = encoder(
            enc_input=emb_out,
            attn_bias=n_head_self_attn_mask,
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._emb_size * 4,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=0,
            hidden_act=self._hidden_act,
            preprocess_cmd="",
            postprocess_cmd="dan",
            param_initializer=self._param_initializer,
            name='encoder')
        if self._dtype == core.VarDesc.VarType.FP16:
            self._enc_out = fluid.layers.cast(
                x=self._enc_out, dtype=self._emb_dtype)


    def get_sequence_output(self):
        return self._enc_out

    def get_pooled_output(self):
        """Get the first feature of each sequence for classification"""
        next_sent_feat = fluid.layers.slice(
            input=self._enc_out, axes=[1], starts=[0], ends=[1])
        next_sent_feat = fluid.layers.fc(
            input=next_sent_feat,
            size=self._emb_size,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name="pooled_fc.w_0", initializer=self._param_initializer),
            bias_attr="pooled_fc.b_0")
        return next_sent_feat 
