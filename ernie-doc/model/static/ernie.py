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
"""Ernie Doc model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import six
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from model.static.transformer_encoder import encoder, pre_process_layer

class ErnieConfig(object):
    """ErnieConfig"""
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
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


class ErnieDocModel(object):
    def __init__(self,
                 src_ids,
                 position_ids,
                 task_ids,
                 input_mask,
                 config,
                 number_instance, 
                 weight_sharing=True,
                 rel_pos_params_sharing=False,
                 use_vars=False):
        """
        Fundamental pretrained Ernie Doc model
        """
        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._voc_size = config['vocab_size']
        self._max_position_seq_len = config['max_position_embeddings']
        self._task_types = config['task_type_vocab_size']
        self._hidden_act = config['hidden_act']
        self._memory_len = config["memory_len"]
        self._epsilon = config["epsilon"]
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        
        self._number_instance = number_instance
        self._weight_sharing = weight_sharing
        self._rel_pos_params_sharing = rel_pos_params_sharing
        
        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        self._task_emb_name = "task_embedding"
        self._emb_dtype = "float32"
        self._encoder_checkpints = []
        
        self._batch_size = layers.slice(layers.shape(src_ids), axes=[0], starts=[0], ends=[1])
        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config['initializer_range'])
        
        self._use_vars = use_vars
        self._init_memories()
        self._build_model(src_ids, position_ids, task_ids, input_mask)
    
    def _init_memories(self):
        """Initialize memories"""
        self.memories = []
        for i in range(self._n_layer):
            if self._memory_len:
                if self._use_vars:
                    self.memories.append(layers.create_global_var(
                            shape=[self._number_instance, self._memory_len, self._emb_size],
                            value=0.0,
                            dtype=self._emb_dtype,
                            persistable=True,
                            force_cpu=False,
                            name="memory_%d" % i))
                else:
                    self.memories.append(layers.data(
                        name="memory_%d" % i,
                        shape=[-1, self._memory_len, self._emb_size],
                        dtype=self._emb_dtype,
                        append_batch_size=False))
            else:
                self.memories.append([None])

    def _build_model(self, src_ids, position_ids, task_ids, input_mask):
        """Build Ernie Doc Model"""
        # padding id in vocabulary must be set to 0
        word_emb = layers.embedding(
            input=src_ids,
            size=[self._voc_size, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._word_emb_name, initializer=self._param_initializer),
            is_sparse=False)
        
        pos_emb = layers.embedding(
            input=position_ids,
            size=[self._max_position_seq_len * 2 + self._memory_len, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._pos_emb_name, initializer=self._param_initializer))  
         
        task_ids = layers.concat([
                layers.zeros(
                    shape=[self._batch_size, self._memory_len, 1], 
                    dtype="int64") + task_ids[0, 0, 0],
                task_ids], axis=1)
        task_ids.stop_gradient = True
        task_emb = layers.embedding(
            task_ids,
            size=[self._task_types, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._task_emb_name, initializer=self._param_initializer))
        
        word_emb = pre_process_layer(
            word_emb, 'nd', self._prepostprocess_dropout, name='pre_encoder_emb')
        pos_emb = pre_process_layer(
            pos_emb, 'nd', self._prepostprocess_dropout, name='pre_encoder_r_pos')
        task_emb = pre_process_layer(
            task_emb, 'nd', self._prepostprocess_dropout, name="pre_encoder_r_task")
 
        data_mask = layers.concat([
                layers.ones(
                    shape=[self._batch_size, self._memory_len, 1], 
                    dtype=input_mask.dtype),
                input_mask], axis=1) 
        data_mask.stop_gradient = True
        self_attn_mask = layers.matmul(
            x=input_mask, y=data_mask, transpose_y=True) 
        self_attn_mask = layers.scale(
            x=self_attn_mask, scale=1000000000.0, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = layers.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True
        
        self._enc_out, self._new_mems, self._checkpoints = encoder(
            enc_input=word_emb,
            memories=self.memories,
            rel_pos=pos_emb,
            rel_task=task_emb,
            attn_bias=n_head_self_attn_mask,
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._emb_size * 4,
            memory_len=self._memory_len,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=0,
            hidden_act=self._hidden_act,
            epsilon=self._epsilon,
            preprocess_cmd="",
            postprocess_cmd="dan",
            param_initializer=self._param_initializer,
            rel_pos_params_sharing=self._rel_pos_params_sharing,
            name='encoder',
            use_vars=self._use_vars)

    def get_sequence_output(self):
        return self._enc_out

    def get_checkpoints(self):
        return self._checkpoints

    def get_mem_output(self):
        return self.memories, self._new_mems

    def get_pooled_output(self):
        """Get the last feature of each sequence for classification"""
        next_sent_feat = layers.slice(
            input=self._enc_out, axes=[1], starts=[-1], ends=[self._max_position_seq_len])
        next_sent_feat = layers.fc(
            input=next_sent_feat,
            size=self._emb_size,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name="pooled_fc.w_0", initializer=self._param_initializer),
            bias_attr="pooled_fc.b_0")
        return next_sent_feat

    def get_pretrained_output(self, 
                              mask_label, 
                              mask_pos, 
                              need_cal_loss=True, 
                              reorder_labels=None, 
                              reorder_chose_idx=None, 
                              reorder_need_cal_loss=False):
        """Get the loss & accuracy for pretraining"""
        reshaped_emb_out = fluid.layers.reshape(
            x=self._enc_out, shape=[-1, self._emb_size])
        # extract masked tokens' feature
        mask_feat = layers.gather(input=reshaped_emb_out, 
            index=layers.cast(mask_pos, dtype="int32"))
    
        # transform: fc
        mask_trans_feat = layers.fc(
            input=mask_feat,
            size=self._emb_size,
            act=self._hidden_act,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_fc.w_0',
                initializer=self._param_initializer),
            bias_attr=fluid.ParamAttr(name='mask_lm_trans_fc.b_0'))
        
        # transform: layer norm 
        mask_trans_feat = layers.layer_norm(
            mask_trans_feat,
            begin_norm_axis=len(mask_trans_feat.shape) - 1,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_layer_norm_scale',
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                name='mask_lm_trans_layer_norm_bias',
                initializer=fluid.initializer.Constant(1.)))

        mask_lm_out_bias_attr = fluid.ParamAttr(
            name="mask_lm_out_fc.b_0",
            initializer=fluid.initializer.Constant(value=0.0))
        if self._weight_sharing:
            fc_out = layers.matmul(
                x=mask_trans_feat,
                y=fluid.default_main_program().global_block().var(
                    self._word_emb_name),
                transpose_y=True)
            fc_out += layers.create_parameter(
                shape=[self._voc_size],
                dtype=self._emb_dtype,
                attr=mask_lm_out_bias_attr,
                is_bias=True)
        else:
            fc_out = layers.fc(
                input=mask_trans_feat,
                size=self._voc_size,
                param_attr=fluid.ParamAttr(
                    name="mask_lm_out_fc.w_0",
                    initializer=self._param_initializer),
                bias_attr=mask_lm_out_bias_attr)

        mlm_loss = layers.softmax_with_cross_entropy(
            logits=fc_out, label=mask_label)
        mean_mlm_loss = layers.mean(mlm_loss) * need_cal_loss
        
        # extract the first token feature in each sentence
        self.next_sent_feat = self.get_pooled_output()
        next_sent_feat_filter = layers.gather(
            input=self.next_sent_feat, 
            index=reorder_chose_idx)
        reorder_fc_out = layers.fc(
            input=next_sent_feat_filter,
            size=33,
            param_attr=fluid.ParamAttr(
                name="multi_sent_sorted" + "_fc.w_0", initializer=self._param_initializer),
            bias_attr="multi_sent_sorted" + "_fc.b_0")
        reorder_loss, reorder_softmax = layers.softmax_with_cross_entropy(
            logits=reorder_fc_out, label=reorder_labels, return_softmax=True)
        reorder_acc = fluid.layers.accuracy(
            input=reorder_softmax, label=reorder_labels)
        mean_reorder_loss = fluid.layers.mean(reorder_loss) * reorder_need_cal_loss

        total_loss = mean_mlm_loss + mean_reorder_loss
        reorder_acc *= reorder_need_cal_loss

        return total_loss, mean_mlm_loss, reorder_acc

