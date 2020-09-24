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

"""ERNIE-ViL model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import six
import paddle.fluid as fluid

from model.vl_transformer_encoder import encoder, pre_process_layer


class ErnieVilConfig(object):
    """
    configuration for ernie-vil
    """
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
        return self._config_dict[key]

    def print_config(self):
        """
        print configuration value
        """
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


class ErnieVilModel(object):
    """
    main class for ERNIE-ViL model
    """
    def __init__(self,
                 src_ids,
                 position_ids,
                 sentence_ids,
                 task_ids,
                 input_mask,
                 image_embeddings,
                 image_loc,
                 input_image_mask,
                 config,
                 predict_feature=False,
                 predict_class=True,
                 use_attr=False,
                 use_soft_label=True):
        
        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        
        self._v_head = config['v_num_attention_heads']
        self._v_emb_size = config['v_hidden_size']
        self._v_inter_hid = config['v_intermediate_size']

        self._co_head = config['co_num_attention_heads']
        self._co_emb_size = config['co_hidden_size']
        self._co_inter_hid = config['co_intermediate_size']

        self._voc_size = config['vocab_size']
        self._class_size = config['class_size']
        self._class_attr_size = config['class_attr_size']
        self._max_position_seq_len = config['max_position_embeddings']
        self._sent_types = config['sent_type_vocab_size']
        self._task_types = config['task_type_vocab_size']
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self._v_biattention_id = config['v_biattention_id']
        self._t_biattention_id = config['t_biattention_id']

        self._predict_feature = predict_feature
        self._predict_class = predict_class
        self._use_attr = use_attr
        self._use_soft_label = use_soft_label
        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        self._sent_emb_name = "sent_embedding"
        self._image_emb_name = "image_embedding"
        self._loc_emb_name = "loc_embedding"
        self._dtype = "float32"
        self._emb_dtype = "float32"

        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config['initializer_range'])

        self._build_model(src_ids, position_ids, sentence_ids, task_ids, input_mask, \
                image_embeddings, image_loc, input_image_mask)

    def _build_model(self, src_ids, position_ids, sentence_ids, task_ids, input_mask, \
            image_embeddings, image_loc, input_image_mask):
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

        sent_emb_out = fluid.layers.embedding(
            sentence_ids,
            size=[self._sent_types, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._sent_emb_name, initializer=self._param_initializer))

        emb_out = emb_out + position_emb_out
        emb_out = emb_out + sent_emb_out

        emb_out = pre_process_layer(
            emb_out, 'nd', self._prepostprocess_dropout, name='pre_encoder')

        self_attn_mask = fluid.layers.matmul(
            x=input_mask, y=input_mask, transpose_y=True)

        self_attn_mask = fluid.layers.scale(
            x=self_attn_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = fluid.layers.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        image_embeddings = fluid.layers.fc(image_embeddings,
                                      self._v_emb_size,
                                      param_attr=fluid.ParamAttr(
                                            name="image_emb.w_0",
                                            initializer=self._param_initializer),
                                      bias_attr = "image_emb.b_0",
                                      num_flatten_dims = 2)
        loc_emb_out = fluid.layers.fc(image_loc,
                                      self._v_emb_size,
                                      param_attr=fluid.ParamAttr(
                                            name="image_loc.w_0",
                                            initializer=self._param_initializer),
                                      bias_attr = "image_loc.b_0",
                                      num_flatten_dims = 2)

        emb_vl_out = image_embeddings + loc_emb_out
        emb_vl_out = pre_process_layer(  
            emb_vl_out, 'nd', self._prepostprocess_dropout, name='vl_pre_encoder')

        self_attn_image_mask = fluid.layers.matmul(
            x=input_image_mask, y=input_image_mask, transpose_y=True)

        self_attn_image_mask = fluid.layers.scale(
            x=self_attn_image_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_image_mask = fluid.layers.stack(
            x=[self_attn_image_mask] * self._v_head, axis=1)
        n_head_self_attn_image_mask.stop_gradient = True

        self_attn_vl_mask = fluid.layers.matmul(
            x=input_image_mask, y=input_mask, transpose_y=True)
        self_attn_vl_mask = fluid.layers.scale(
            x=self_attn_vl_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_vl_mask = fluid.layers.stack(
            x=[self_attn_vl_mask] * self._co_head, axis=1)
        n_head_self_attn_vl_mask.stop_gradient = True

        self._enc_out, self._enc_vl_out = encoder(
            enc_input=emb_out,
            enc_vl_input=emb_vl_out,
            attn_bias=n_head_self_attn_mask,
            attn_image_bias=n_head_self_attn_image_mask,
            attn_vl_bias=n_head_self_attn_vl_mask,
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._emb_size * 4,
            v_head=self._v_head,
            v_key=self._v_emb_size // self._v_head,
            v_value=self._v_emb_size // self._v_head,
            v_model=self._v_emb_size,
            v_inner_hid=self._v_inter_hid,
            co_head=self._co_head,
            co_key=self._co_emb_size // self._co_head,
            co_value=self._co_emb_size // self._co_head,
            co_model=self._co_emb_size,
            co_inner_hid=self._co_inter_hid,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=0,
            hidden_act=self._hidden_act,
            preprocess_cmd="",
            postprocess_cmd="dan",
            param_initializer=self._param_initializer,
            v_biattention_id = self._v_biattention_id,
            t_biattention_id = self._t_biattention_id,
            name='encoder')

    def get_sequence_output(self):
        """ 
        Return sequence output of all text and img tokens
        """
        return self._enc_out, self._enc_vl_out

    def get_pooled_output(self):
        """
        Get the first feature of each sequence for classification
        """
        text_cls_feat = fluid.layers.slice(
            input=self._enc_out, axes=[1], starts=[0], ends=[1])

        text_cls_feat = fluid.layers.cast(
            x=text_cls_feat, dtype=self._emb_dtype)

        text_cls_feat = fluid.layers.fc(
            input=text_cls_feat,
            size=self._co_emb_size,
            act="relu",
            param_attr=fluid.ParamAttr(
                name="pooled_fc_text.w_0", initializer=self._param_initializer),
            bias_attr="pooled_fc_text.b_0")

        image_cls_feat = fluid.layers.slice(
            input=self._enc_vl_out, axes=[1], starts=[0], ends=[1])

        image_cls_feat = fluid.layers.cast(
                x=image_cls_feat, dtype=self._emb_dtype)

        image_cls_feat = fluid.layers.fc(
            input=image_cls_feat,
            size=self._co_emb_size,
            act="relu",
            param_attr=fluid.ParamAttr(
                name="pooled_fc_image.w_0", initializer=self._param_initializer),
            bias_attr="pooled_fc_image.b_0")
        return text_cls_feat, image_cls_feat

    def get_match_score(self, text, image, dropout_rate=0.0, mode="mul"):
        """
        match score for text [cls] and image [img] tokens
        """
        if mode == "sum":
            emb_fuse = text + image
        elif mode == "mul":
            emb_fuse = text * image
        else:
            "current mode %s is not supported" % mode
            return
        if dropout_rate > 0.0:

            emb_fuse = fluid.layers.dropout(emb_fuse,
                       self._attention_dropout,
                       dropout_implementation="upscale_in_train")
        return emb_fuse



