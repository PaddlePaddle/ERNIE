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

import json

import six
import paddle.fluid as fluid

from model.transformer_encoder import rel_pos_encoder, pre_process_layer


class ErnieConfig(object):
    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing Ernie model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict.get(key)

    def print_config(self):
        print('-------  Model Arguments ---------')
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


class ErnieModel(object):
    def __init__(self,
                 src_ids,
                 position_ids,
                 sentence_ids,
                 task_ids,
                 input_mask,
                 config,
                 rel_pos_bin=32,
                 weight_sharing=True,
                 use_fp16=False,
                 has_sent_emb=False,
                 name=""):

        self._hidden_size = config['hidden_size']
        self._emb_size = config['emb_size'] or self._hidden_size
        self._out_emb_size = config['out_emb_size'] or self._emb_size
        self._voc_size = config['vocab_size']
        self._rel_pos_bin = rel_pos_bin
        self._out_voc_size = config['out_vocab_size'] or self._voc_size
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._max_position_seq_len = config['max_position_embeddings']
        self._sent_types = config['sent_type_vocab_size']
        self._task_types = config['task_type_vocab_size']
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self._weight_sharing = weight_sharing
        self.has_sent_emb = has_sent_emb
        self._model_name = name
        self._rel_pos_emb_name = self._model_name + "rel_pos_embedding"
        self._word_emb_name = self._model_name + "word_embedding"
        self._pos_emb_name = self._model_name + "pos_embedding"
        self._sent_emb_name = self._model_name + "sent_embedding"
        self._checkpoints = []
        self._input_mask = input_mask
        self._emb_dtype = "float32"

        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config['initializer_range'])

        self._build_model(src_ids, position_ids, sentence_ids, task_ids, input_mask)

    def _build_model(self, src_ids, position_ids, sentence_ids, task_ids, input_mask):
        # padding id in vocabulary must be set to 0
        emb_out = fluid.layers.embedding(
            input=src_ids,
            size=[self._voc_size, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._word_emb_name, initializer=self._param_initializer),
            is_sparse=False)

        position_emb_out = fluid.layers.embedding(
            input=position_ids[0],
            size=[self._max_position_seq_len, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._pos_emb_name, initializer=self._param_initializer))

        rel_position_scaler_emb_out = fluid.layers.embedding(
            input=position_ids[1],
            size=[self._rel_pos_bin + 1, self._n_head],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._rel_pos_emb_name, initializer=self._param_initializer))

        sent_emb_out = fluid.layers.embedding(
            sentence_ids,
            size=[self._sent_types, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._sent_emb_name, initializer=self._param_initializer))
        
        emb_out = emb_out + position_emb_out
        if self.has_sent_emb:
            emb_out = emb_out + sent_emb_out

        emb_out = pre_process_layer(
            emb_out, 'nd', self._prepostprocess_dropout, name=self._model_name + 'pre_encoder')

        if self._emb_size != self._hidden_size:
            emb_out = fluid.layers.fc(input=emb_out, 
                          num_flatten_dims=2,
                          size=self._hidden_size,
                          param_attr=fluid.ParamAttr(
                              name=self._model_name + 'emb_hidden_mapping',
                              initializer=self._param_initializer),
                          bias_attr=self._model_name + 'emb_hidden_mapping_bias')


        self_attn_mask = fluid.layers.matmul(
            x=input_mask, y=input_mask, transpose_y=True)

        self_attn_mask = fluid.layers.scale(
            x=self_attn_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = fluid.layers.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        self._enc_out, encoder_checkpoints = rel_pos_encoder(
            enc_input=emb_out,
            pos_input=rel_position_scaler_emb_out,
            attn_bias=n_head_self_attn_mask,
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._hidden_size // self._n_head,
            d_value=self._hidden_size // self._n_head,
            d_model=self._hidden_size,
            d_inner_hid=self._hidden_size * 4,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=0,
            hidden_act=self._hidden_act,
            preprocess_cmd="",
            postprocess_cmd="dan",
            param_initializer=self._param_initializer,
            name=self._model_name + 'encoder')
        
        self._checkpoints.extend(encoder_checkpoints)

    def get_sequence_output(self):
        _enc_out = fluid.layers.fc(
            input=self._enc_out,
            size=128,
            num_flatten_dims=2,
            act=self._hidden_act,
            param_attr=fluid.ParamAttr(
                name=self._model_name + 'mask_lm_trans_fc.w_0',
                initializer=self._param_initializer),
            bias_attr=fluid.ParamAttr(name=self._model_name + 'mask_lm_trans_fc.b_0'))

        return _enc_out
    
    def get_checkpoints(self):
        """return checkpoints for recomputing"""
        #recompute checkpoints
        return self._checkpoints

    def get_pooled_output(self, has_fc=True):
        """Get the first feature of each sequence for classification"""
        next_sent_feat = fluid.layers.slice(
            input=self._enc_out, axes=[1], starts=[0], ends=[1])
        if has_fc:
            next_sent_feat = fluid.layers.fc(
                input=next_sent_feat,
                size=self._hidden_size,
                act="tanh",
                param_attr=fluid.ParamAttr(
                    name=self._model_name + "pooled_fc.w_0", initializer=self._param_initializer),
                bias_attr=self._model_name + "pooled_fc.b_0")
        else:
            next_sent_feat = fluid.layers.reshape(next_sent_feat, [-1, self._hidden_size])
        
        return next_sent_feat

