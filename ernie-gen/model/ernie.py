#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from model.transformer_model import infilling_transformer, infilling_decode_transformer, pre_process_layer


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
        return self._config_dict.get(key, None)

    def __setitem__(self, key, value):
        self._config_dict[key] = value

    def print_config(self):
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


class ErnieModel(object):
    def __init__(self,
                 emb_ids,
                 input_mask,
                 config,
                 weight_sharing=True,
                 use_fp16=False,
                 task_type="normal",
                 decoding=False,
                 gather_idx=None):

        self._emb_size = config['emb_size'] or config['hidden_size']
        self._hidden_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._voc_size = config['vocab_size']
        self._max_position_seq_len = config['max_position_embeddings']
        
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self._weight_sharing = weight_sharing
        self._task_type = task_type
        self._emb_vocab_size = {"word_embedding": self._voc_size,
                                "pos_embedding": self._max_position_seq_len}

        self._is_dialogue_task = (task_type == "dialog")
        if self._is_dialogue_task:
            self._role_type_size = config["role_type_size"]
            self._turn_type_size = config["turn_type_size"]
            self._emb_vocab_size["role_embedding"] = self._role_type_size
            self._emb_vocab_size["turn_embedding"] = self._turn_type_size
        else:
            self._sent_types = config['type_vocab_size']
            self._emb_vocab_size["sent_embedding"] = self._sent_types

        self._dtype = "float16" if use_fp16 else "float32"
        self._emb_dtype = "float32"

        self._epsilon = config['epsilon'] or 1e-5
        self._param_share = config['param_share'] or "normal"
        self._n_layer_per_block = config['n_layer_per_block'] or 1
        self._pre_encoder_cmd = config['pre_encoder_cmd'] or 'nd'
        self._preprocess_cmd = config['preprocess_cmd'] or ''
        self._postprocess_cmd= config['postprocess_cmd'] or 'dan'
        if self._hidden_size != self._emb_size:
            self._emb_mapping_in = True
        else:
            self._emb_mapping_in = config['emb_mapping_in'] or False

        self._decoding = decoding
        if decoding:
            self.caches = [{
                "k":
                fluid.layers.fill_constant_batch_size_like(
                    input=emb_ids["word_embedding"],
                    shape=[-1, 0, self._hidden_size],
                    dtype=self._dtype,
                    value=0),
                "v":
                fluid.layers.fill_constant_batch_size_like(
                    input=emb_ids["word_embedding"],
                    shape=[-1, 0, self._hidden_size],
                    dtype=self._dtype,
                    value=0),
            } for i in range(self._n_layer)]
        else:
            self.caches = None

        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config['initializer_range'])

        self._build_model(emb_ids, input_mask, gather_idx)

    def _gen_input(self, emb_ids, input_mask):
        emb_out = None
        for emb_name, emb_id in emb_ids.items():
            emb = fluid.layers.embedding(
                input=emb_id,
                size=[self._emb_vocab_size[emb_name], self._emb_size],
                dtype=self._emb_dtype,
                param_attr=fluid.ParamAttr(
                    name=emb_name, initializer=self._param_initializer))
            emb_out = emb_out + emb if emb_out else emb
        
        emb_out = pre_process_layer(
            emb_out,
            self._pre_encoder_cmd,
            self._prepostprocess_dropout,
            name="pre_encoder",
            epsilon=self._epsilon)
        if self._emb_mapping_in:
            emb_out = fluid.layers.fc(input=emb_out,
                          num_flatten_dims=2,
                          size=self._hidden_size,
                          param_attr=fluid.ParamAttr(
                              name='emb_hidden_mapping',
                              initializer=self._param_initializer),
                          bias_attr='emb_hidden_mapping_bias')
        if self._dtype is "float16":
            emb_out = fluid.layers.cast(x=emb_out, dtype=self._dtype)
            input_mask = fluid.layers.cast(x=input_mask, dtype=self._dtype)

        self_attn_mask = input_mask
        self_attn_mask = fluid.layers.scale(
            x=self_attn_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = fluid.layers.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        return emb_out, n_head_self_attn_mask


    def encode(self, emb_ids, input_mask, gather_idx=None, remove_query=True):
        # padding id in vocabulary must be set to 0
        if not self._decoding:
            emb_out, n_head_self_attn_mask = self._gen_input(emb_ids[0], input_mask[0])
            query_emb_out, n_head_query_attn_mask = self._gen_input(emb_ids[1], input_mask[1])
            enc_out = infilling_transformer(
                enc_input_kv=emb_out,
                enc_input_query=query_emb_out,
                attn_bias_kv=n_head_self_attn_mask,
                attn_bias_query=n_head_query_attn_mask,
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
                preprocess_cmd=self._preprocess_cmd,
                postprocess_cmd=self._postprocess_cmd,
                param_initializer=self._param_initializer,
                epsilon=self._epsilon,
                param_share=self._param_share,
                n_layer_per_block=self._n_layer_per_block,
                name='encoder')
        else:
            emb_out, n_head_self_attn_mask = self._gen_input(emb_ids, input_mask)
            enc_out = infilling_decode_transformer(
                enc_input=emb_out,
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
                preprocess_cmd=self._preprocess_cmd,
                postprocess_cmd=self._postprocess_cmd,
                param_initializer=self._param_initializer,
                epsilon=self._epsilon,
                param_share=self._param_share,
                n_layer_per_block=self._n_layer_per_block,
                name='encoder',
                caches=self.caches,
                gather_idx=gather_idx,
                remove_query=remove_query)

        if self._dtype == "float16":
            enc_out = fluid.layers.cast(
                x=enc_out, dtype=self._emb_dtype)
        return enc_out

    def _build_model(self, emb_ids, input_mask, gather_idx=None):
        self._enc_out = self.encode(emb_ids, input_mask, gather_idx, remove_query=False)

    def get_sequence_output(self):
        return self._enc_out

