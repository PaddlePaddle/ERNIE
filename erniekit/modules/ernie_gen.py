# *_*coding:utf-8 *_*
"""
import
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import json
import logging

import paddle.fluid as fluid
import six

from erniekit.modules.transformer_encoder_gen import encoder, two_stream_encoder, pre_process_layer
from erniekit.modules.transformer_encoder_gen import gelu
from paddle import nn

class ErnieGenModel(nn.Layer):
    """
    ErnieGenModel
    """

    def __init__(self,
                 src_ids,
                 position_ids,
                 sentence_ids,
                 input_mask,
                 config,
                 use_fp16=False,
                 role_ids=None,
                 turn_ids=None,
                 weight_sharing=True,
                 task_type="normal",
                 two_stream=False,
                 decoding=False,
                 gather_idx=None,
                 key_tag=None):
        """
        :param src_ids:
        :param position_ids:
        :param sentence_ids:
        :param task_ids:
        :param input_mask:
        :param config:
        :param weight_sharing:
        :param use_fp16:
        """
        self._hidden_size = config.get('hidden_size', 768)
        self._emb_size = config.get('emb_size', self._hidden_size)
        self._n_layer = config.get('num_hidden_layers', 12)
        self._n_head = config.get('num_attention_heads', 12)
        self._voc_size = config.get('vocab_size', 30522)
        self._max_position_seq_len = config.get('max_position_embeddings', 512)
        self._param_share = config.get('param_share', "normal")
        self._pre_encoder_cmd = config.get('pre_encoder_cmd', "nd")
        self._preprocess_cmd = config.get('preprocess_cmd', "")
        self._postprocess_cmd = config.get('postprocess_cmd', "dan")
        self._epsilon = config.get('epsilon', 1e-05)
        self._emb_mapping_in = config.get('emb_mapping_in', False)
        self._n_layer_per_block = config.get('n_layer_per_block', 1)

        if config.has('sent_type_vocab_size'):
            self._sent_types = config['sent_type_vocab_size']
        else:
            self._sent_types = config.get('type_vocab_size', 2)

        self._hidden_act = config.get('hidden_act', 'gelu')
        self._prepostprocess_dropout = config.get('hidden_dropout_prob', 0.1)
        self._attention_dropout = config.get('attention_probs_dropout_prob', 0.1)
        self._weight_sharing = weight_sharing

        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        self._sent_emb_name = "sent_embedding"
        self._dtype = "float16" if use_fp16 else "float32"
        self._emb_dtype = "float32"

        self._task_type = task_type
        self._is_dialogue_task = (task_type == "dialog")
        if self._is_dialogue_task:
            self._role_type_size = config["role_type_size"]
            self._turn_type_size = config["turn_type_size"]
            self._role_emb_name = "role_embedding"
            self._turn_emb_name = "turn_embedding"

        self._two_stream = two_stream
        if decoding:
            self.caches = [{
                "k":
                fluid.layers.fill_constant_batch_size_like(
                    input=src_ids,
                    shape=[-1, 0, self._hidden_size],
                    dtype=self._dtype,
                    value=0),
                "v":
                fluid.layers.fill_constant_batch_size_like(
                    input=src_ids,
                    shape=[-1, 0, self._hidden_size],
                    dtype=self._dtype,
                    value=0),
            } for i in range(self._n_layer)]
        else:
            self.caches = None

        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config.get('initializer_range', 0.02))
        self.key_tag = key_tag

        self._build_model(src_ids, position_ids, sentence_ids,
                          input_mask, gather_idx, role_ids, turn_ids)
    
    def _gen_input(self, src_ids, position_ids, sentence_ids, input_mask,
                   role_ids=None, turn_ids=None):
        """
        :param src_ids:
        :param position_ids:
        :param sentence_ids:
        :param task_ids:
        :param input_mask:
        :return:
        """
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

        if not self._is_dialogue_task:
            sent_emb_out = fluid.layers.embedding(
                sentence_ids,
                size=[self._sent_types, self._emb_size],
                dtype=self._emb_dtype,
                param_attr=fluid.ParamAttr(
                    name=self._sent_emb_name, initializer=self._param_initializer))
            emb_out = emb_out + position_emb_out + sent_emb_out

        else:
            role_emb_out = fluid.layers.embedding(
                input=role_ids,
                size=[self._role_type_size, self._emb_size],
                dtype=self._emb_dtype,
                param_attr=fluid.ParamAttr(
                    name=self._role_emb_name, initializer=self._param_initializer))
            turn_emb_out = fluid.layers.embedding(
                input=turn_ids,
                size=[self._turn_type_size, self._emb_size],
                dtype=self._emb_dtype,
                param_attr=fluid.ParamAttr(
                    name=self._turn_emb_name, initializer=self._param_initializer))
            emb_out = emb_out + position_emb_out + role_emb_out + turn_emb_out

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

    def encode(self, src_ids, position_ids, sentence_ids,
               input_mask, gather_idx=None, remove_mask=False,
               role_ids=None, turn_ids=None):
        """transformer encode"""

        if self._two_stream:
            emb_out, n_head_self_attn_mask = self._gen_input(src_ids[0],
                position_ids[0], sentence_ids[0], input_mask[0],
                role_ids=role_ids[0], turn_ids=turn_ids[0])
            g_emb_out, n_head_query_attn_mask = self._gen_input(src_ids[1],
                position_ids[1], sentence_ids[1], input_mask[1],
                role_ids=role_ids[1], turn_ids=turn_ids[1])

            self._enc_out_context, self._enc_out_query, self._checkpoints = two_stream_encoder(
                enc_input_context=emb_out,
                enc_input_query=g_emb_out,
                attn_bias_context=n_head_self_attn_mask,
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
                name='encoder',
                param_share=self._param_share,
                epsilon=self._epsilon,
                n_layer_per_block=self._n_layer_per_block,
                key_tag=self.key_tag)

            enc_out = self._enc_out_query
        else:
            emb_out, n_head_self_attn_mask = self._gen_input(src_ids,
                position_ids, sentence_ids, input_mask,
                role_ids=role_ids, turn_ids=turn_ids)
            enc_out, self._checkpoints= encoder(
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
                name='encoder',
                param_share=self._param_share,
                epsilon=self._epsilon,
                n_layer_per_block=self._n_layer_per_block,
                caches=self.caches,
                gather_idx=gather_idx,
                remove_mask=remove_mask,
                key_tag=self.key_tag)

        if self._dtype == "float16":
            enc_out = fluid.layers.cast(
                x=enc_out, dtype=self._emb_dtype)
        return enc_out

    def _build_model(self, src_ids, position_ids, sentence_ids,
                     input_mask, gather_idx=None, role_ids=None, turn_ids=None):
        self._enc_out = self.encode(src_ids, position_ids, sentence_ids,
                                    input_mask, gather_idx, remove_mask=False,
                                    role_ids=role_ids, turn_ids=turn_ids)

    def get_sequence_output(self):
        """
        :return:
        """
        return self._enc_out

    def get_pooled_output(self):
        """Get the first feature of each sequence for classification"""
        next_sent_feat = fluid.layers.slice(
            input=self._enc_out, axes=[1], starts=[0], ends=[1])
        next_sent_feat = fluid.layers.fc(
            input=next_sent_feat,
            size=self._hidden_size,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name="pooled_fc.w_0", initializer=self._param_initializer),
            bias_attr="pooled_fc.b_0")
        return next_sent_feat

