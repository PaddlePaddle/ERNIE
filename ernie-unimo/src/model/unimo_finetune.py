#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Unified Visual Language model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import six
import codecs
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from model.transformer_encoder import encoder, pre_process_layer


class UNIMOConfig(object):
    """configuration"""

    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with codecs.open(config_path, 'r', encoding='utf-8') as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing unimo model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict.get(key, None)

    def __setitem__(self, key, value):
        self._config_dict[key] = value

    def print_config(self):
        """print config"""
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


class UNIMOModel(object):
    """UNIMO model for finetuning"""

    def __init__(self,
                 emb_ids=None,
                 emb_obj_ids=None,
                 input_mask=None,
                 config=None,
                 image_input=None,
                 text_adv_delta=None,
                 image_adv_delta=None,
                 weight_sharing=True,
                 task_type="normal",
                 decoding=False,
                 gather_idx=None):

        self.text_adv_delta = text_adv_delta
        self.image_adv_delta = image_adv_delta

        self._emb_size = config['hidden_size']
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

        assert emb_ids is not None or image_input is not None, "emb_ids and image_input cannot be both None"
        self._is_dialogue_task = (task_type == "dialog")
        self._is_img2txt_task = (task_type == "img2txt")
        self._is_multimodal_task = (image_input is not None)

        if emb_ids is not None and image_input is not None and emb_obj_ids is not None:
            self._input_type = 'vol'
        elif emb_ids is not None and image_input is not None:
            self._input_type = 'vl'
        elif emb_ids is not None:
            self._input_type = 'l'
        elif image_input is not None and emb_obj_ids is not None:
            self._input_type = 'vo'
        else:
            raise ValueError('input feature error')

        if self._is_dialogue_task:
            self._role_type_size = config["role_type_size"]
            self._turn_type_size = config["turn_type_size"]
            self._emb_vocab_size["role_embedding"] = self._role_type_size
            self._emb_vocab_size["turn_embedding"] = self._turn_type_size
        else:
            self._sent_types = config['type_vocab_size']
            self._emb_vocab_size["sent_embedding"] = self._sent_types
            if self._is_multimodal_task or self._is_img2txt_task:
                self._image_class_size = config['image_class_size']
                self._class_attr_size = config['class_attr_size']
                self._image_embedding_size = config['image_embedding_size']
                self._image_predict_feature = config['image_predict_feature']
                self._image_predict_class = config['image_predict_class']
                self._image_use_attr = config['image_use_attr']
                self._image_use_soft_label = config['image_use_soft_label']
                self._image_emb_name = "image_embedding"
                self._loc_emb_name = "loc_embedding"

        self._emb_dtype = "float32"

        if decoding:
            self.caches = [{
                "k":
                    fluid.layers.fill_constant_batch_size_like(
                        input=emb_ids["word_embedding"] if emb_ids is not None else image_input["image_embedding"],
                        shape=[-1, 0, self._emb_size],
                        dtype=self._emb_dtype,  # float32,
                        value=0),
                "v":
                    fluid.layers.fill_constant_batch_size_like(
                        input=emb_ids["word_embedding"] if emb_ids is not None else image_input["image_embedding"],
                        shape=[-1, 0, self._emb_size],
                        dtype=self._emb_dtype,  # float32,
                        value=0),
            } for i in range(self._n_layer)]
        else:
            self.caches = None

        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config['initializer_range'])

        self._build_model(emb_ids=emb_ids,
                          input_mask=input_mask,
                          image_input=image_input,
                          emb_obj_ids=emb_obj_ids,
                          gather_idx=gather_idx)

    def _build_model(self, emb_ids=None, input_mask=None, image_input=None, emb_obj_ids=None, gather_idx=None):
        """build unimo model"""

        self._enc_vol_out = None
        self._enc_vl_out = None
        self._enc_v_out = None
        self._enc_l_out = None

        if self._input_type == 'vol':
            self._enc_vol_out, self._enc_v_out, self._enc_l_out = self.encode(emb_ids=emb_ids,
                                                                              input_mask=input_mask,
                                                                              image_input=image_input,
                                                                              emb_obj_ids=emb_obj_ids,
                                                                              gather_idx=gather_idx)
        elif self._input_type == 'vl':
            self._enc_vl_out, self._enc_v_out, self._enc_l_out = self.encode(emb_ids=emb_ids,
                                                                             input_mask=input_mask,
                                                                             image_input=image_input,
                                                                             gather_idx=gather_idx)
        elif self._input_type == 'vo':
            self._enc_v_out = self.encode(input_mask=input_mask,
                                          image_input=image_input,
                                          emb_obj_ids=emb_obj_ids,
                                          gather_idx=gather_idx)
        elif self._input_type == 'l':
            self._enc_l_out = self.encode(emb_ids=emb_ids,
                                          input_mask=input_mask,
                                          gather_idx=gather_idx)
        else:
            raise ValueError("The input type is invalid")

    def encode(self, emb_ids=None, input_mask=None, image_input=None, emb_obj_ids=None, gather_idx=None):
        """unimo encoder"""
        emb_feature, n_head_self_attn_mask, _v_seq_len, _o_seq_len = self._gen_input(emb_ids=emb_ids,
                                                                                     input_mask=input_mask,
                                                                                     image_input=image_input,
                                                                                     emb_obj_ids=emb_obj_ids)
        enc_out = encoder(
            enc_input=emb_feature,
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
            name='encoder',
            caches=self.caches,
            gather_idx=gather_idx)

        if self._input_type == 'vol':
            assert _v_seq_len is not None and _o_seq_len is not None, "the input is invalid"
            _vol_seq_len = layers.shape(enc_out)[1]
            enc_v_out = fluid.layers.slice(
                input=enc_out, axes=[1], starts=[0], ends=[_v_seq_len])
            enc_o_out = fluid.layers.slice(
                input=enc_out, axes=[1], starts=[_v_seq_len], ends=[_v_seq_len + _o_seq_len])
            enc_l_out = fluid.layers.slice(
                input=enc_out, axes=[1], starts=[_v_seq_len + _o_seq_len], ends=[_vol_seq_len])
            enc_vol_out = enc_out
            return enc_vol_out, enc_v_out, enc_l_out
        elif self._input_type == 'vl':
            assert _v_seq_len is not None and _o_seq_len is None, "the input is invalid"
            _vl_seq_len = layers.shape(enc_out)[1]
            enc_v_out = fluid.layers.slice(
                input=enc_out, axes=[1], starts=[0], ends=[_v_seq_len])
            enc_l_out = fluid.layers.slice(
                input=enc_out, axes=[1], starts=[_v_seq_len], ends=[_vl_seq_len])
            enc_vl_out = enc_out
            return enc_vl_out, enc_v_out, enc_l_out
        elif self._input_type == 'vo':
            assert _v_seq_len is not None and _o_seq_len is not None, "the input is invalid"
            enc_v_out = fluid.layers.slice(
                input=enc_out, axes=[1], starts=[0], ends=[_v_seq_len])
            return enc_v_out
        elif self._input_type == 'l':
            assert _v_seq_len is None and _o_seq_len is None, "the input is invalid"
            enc_l_out = enc_out
            return enc_l_out
        else:
            raise ValueError("The input type is invalid")

    def _gen_input(self, emb_ids=None, input_mask=None, image_input=None, emb_obj_ids=None):
        assert input_mask is not None, "input_mask should not be none"
        self_attn_mask = input_mask
        self_attn_mask = fluid.layers.scale(
            x=self_attn_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = fluid.layers.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True
        emb_feature, _v_seq_len, _o_seq_len = None, None, None

        if emb_ids is not None:
            emb_out = None
            # text part
            for emb_name, emb_id in emb_ids.items():
                if emb_name == "sent_embedding":
                    continue  # don't use sentence embedding
                emb = fluid.layers.embedding(
                    input=emb_id,
                    size=[self._emb_vocab_size[emb_name], self._emb_size],
                    dtype=self._emb_dtype,
                    param_attr=fluid.ParamAttr(
                        name=emb_name, initializer=self._param_initializer))
                emb_out = emb_out + emb if emb_out else emb

            if self.text_adv_delta is not None:
                emb_out = emb_out + self.text_adv_delta

            emb_out = pre_process_layer(
                emb_out, 'nd', self._prepostprocess_dropout, name="pre_encoder")

        if image_input is not None:
            # visual part
            if self.image_adv_delta is not None:
                emb_v_in = image_input[self._image_emb_name]
                emb_v_in = emb_v_in + self.image_adv_delta
            else:
                emb_v_in = image_input[self._image_emb_name]

            image_embeddings = fluid.layers.fc(emb_v_in,  # [batch_size, 37, 2048]
                                               self._emb_size,
                                               param_attr=fluid.ParamAttr(
                                                   name="image_emb.w_0",
                                                   initializer=self._param_initializer),
                                               bias_attr="image_emb.b_0",
                                               num_flatten_dims=2)

            loc_emb_out = fluid.layers.fc(image_input[self._loc_emb_name],  # [batch_size, 37, 5]
                                          self._emb_size,
                                          param_attr=fluid.ParamAttr(
                                              name="image_loc.w_0",
                                              initializer=self._param_initializer),
                                          bias_attr="image_loc.b_0",
                                          num_flatten_dims=2)

            emb_v_out = image_embeddings + loc_emb_out
            emb_v_out = pre_process_layer(
                emb_v_out, 'nd', self._prepostprocess_dropout, name='v_pre_encoder')

            _v_seq_len = layers.shape(emb_v_out)[1]

        if emb_obj_ids is not None:
            emb_obj_out = None
            # text part
            for emb_obj_name, emb_obj_id in emb_obj_ids.items():
                if emb_obj_name == "sent_embedding":
                    continue  # don't use sentence embedding in roberta
                emb_obj = fluid.layers.embedding(
                    input=emb_obj_id,
                    size=[self._emb_vocab_size[emb_obj_name], self._emb_size],
                    dtype=self._emb_dtype,
                    param_attr=fluid.ParamAttr(
                        name=emb_obj_name, initializer=self._param_initializer))
                emb_obj_out = emb_obj_out + emb_obj if emb_obj_out else emb_obj

            emb_obj_out = pre_process_layer(
                emb_obj_out, 'nd', self._prepostprocess_dropout, name="pre_encoder")
            _o_seq_len = layers.shape(emb_obj_out)[1]

        if self._input_type == 'vol':
            assert emb_ids is not None and image_input is not None and emb_obj_ids is not None, "the input is invalid"
            emb_feature = fluid.layers.concat([emb_v_out, emb_obj_out, emb_out], axis=1)
        elif self._input_type == 'vl':
            assert emb_ids is not None and image_input is not None and emb_obj_ids is None, "the input is invalid"
            emb_feature = fluid.layers.concat([emb_v_out, emb_out], axis=1)
        elif self._input_type == 'l':
            assert emb_ids is not None and image_input is None and emb_obj_ids is None, "the input is invalid"
            emb_feature = emb_out
        elif self._input_type == 'vo':
            assert emb_ids is None and image_input is not None and emb_obj_ids is not None, "the input is invalid"
            emb_feature = fluid.layers.concat([emb_v_out, emb_obj_out], axis=1)
        else:
            raise ValueError("The input type is invalid")

        return [emb_feature, n_head_self_attn_mask, _v_seq_len, _o_seq_len]

    def get_sequence_output(self):
        """get sequence output"""
        return self._enc_l_out

    def get_pooled_output(self):
        """Get the first feature of each sequence for classification"""
        text_feat = self.get_pooled_text_output()
        visual_feat = self.get_pooled_visual_output()
        return text_feat, visual_feat

    def get_pooled_visual_output(self):
        """Get the first feature of each sequence for classification"""
        if self._enc_v_out is None:
            return None
        visual_feat = fluid.layers.slice(
            input=self._enc_v_out, axes=[1], starts=[0], ends=[1])
        visual_feat = fluid.layers.reshape(
            x=visual_feat, shape=[-1, self._emb_size])
        visual_feat = fluid.layers.fc(
            input=visual_feat,
            size=self._emb_size,
            act="relu",
            param_attr=fluid.ParamAttr(
                name="pooled_fc_image.w_0",
                initializer=self._param_initializer),
            bias_attr="pooled_fc_image.b_0")
        return visual_feat

    def get_pooled_text_output(self):
        """Get the first feature of each sequence for classification"""
        if self._enc_l_out is None:
            return None
        text_feat = fluid.layers.slice(
            input=self._enc_l_out, axes=[1], starts=[0], ends=[1])
        text_feat = fluid.layers.reshape(
            x=text_feat, shape=[-1, self._emb_size])
        text_feat = fluid.layers.fc(
            input=text_feat,
            size=self._emb_size,
            act="relu",
            param_attr=fluid.ParamAttr(
                name="pooled_fc_text.w_0",
                initializer=self._param_initializer),
            bias_attr="pooled_fc_text.b_0"
        )
        return text_feat

    def get_match_output(self, text, image, mode="mul"):
        """get_match_output"""
        if mode == "sum":
            emb_fuse = text + image
        elif mode == "mul":
            emb_fuse = text * image
        else:
            "current mode %s is not supported" % mode
            return
        emb_fuse = fluid.layers.dropout(emb_fuse,
                                        self._attention_dropout,
                                        dropout_implementation="upscale_in_train")
        return emb_fuse
