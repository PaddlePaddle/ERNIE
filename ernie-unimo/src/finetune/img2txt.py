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
"""image-to-text generation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import time
import numpy as np
import glob
import json

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from model.unimo_finetune import UNIMOModel
from eval.gen_eval import GenerationEval
from finetune.trigram_blocking import TrigramBlocking
import codecs


class Img2Txt(object):
    """image-to-text"""
    def __init__(self, args, vl_config, tokenizer):
        self.vl_config = vl_config
        self.weight_sharing = args.weight_sharing
        self.max_seq_len = args.max_seq_len
        self.max_img_len = args.max_img_len
        self.max_obj_len = args.max_obj_len
        self.label_smooth = args.label_smooth
        self.tgt_type_id = args.tgt_type_id
        self.tokenizer = tokenizer
        self.vocab_size = vl_config["vocab_size"]
        self.args = args
        self.adv_kl_weight = args.adv_kl_weight
        self.with_pure_model = args.with_pure_model

        self._emb_dtype = "float32"

        # for beam_search decoding
        self.do_decode = args.do_decode
        self.length_penalty = args.length_penalty
        self.max_out_len = args.max_out_len
        self.min_out_len = args.min_out_len
        self.block_trigram = args.block_trigram
        self.beam_size = args.beam_size

        self.bos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.sep_token_id
        self.evaluator = GenerationEval(args)
        self.emb_keys = ["word_embedding", "sent_embedding", "pos_embedding"]
        self.img_keys = ["image_embedding", "loc_embedding"]
        self.task_type = "img2txt"

    def _kl_divergence_with_logits(self, q_logits, p_logits):
        """
        symmetric KL-divergence (See SMART, Sec 3.1)
        q_logits: logits
        p_logits: delta_logits
        """
        q = fluid.layers.softmax(input=q_logits)
        p = fluid.layers.softmax(input=p_logits)
        kl_qp = fluid.layers.reduce_sum(q * (fluid.layers.log(q) - fluid.layers.log(p)), -1)
        kl_pq = fluid.layers.reduce_sum(p * (fluid.layers.log(p) - fluid.layers.log(q)), -1)
        vat_loss = fluid.layers.mean(x=kl_qp + kl_pq)
        return vat_loss

    def cal_logit(self, enc_out, tgt_pos):
        """calculate logit"""
        enc_out = fluid.layers.reshape(x=enc_out,
                                       shape=[-1, self.vl_config["hidden_size"]])
        if tgt_pos:
            tgt_pos = fluid.layers.cast(x=tgt_pos, dtype='int32')
            tgt_feat = fluid.layers.gather(input=enc_out, index=tgt_pos)
        else:
            tgt_feat = enc_out

        tgt_trans_feat = fluid.layers.fc(
            input=tgt_feat,
            size=self.vl_config["emb_size"] or self.vl_config["hidden_size"],
            act=self.vl_config["hidden_act"],
            param_attr=fluid.ParamAttr(
                name="mask_lm_trans_fc.w_0",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="mask_lm_trans_fc.b_0",
                initializer=fluid.initializer.Constant(0.)))

        tgt_trans_feat = fluid.layers.layer_norm(
            tgt_trans_feat,
            begin_norm_axis=len(tgt_trans_feat.shape) - 1,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_layer_norm_scale',
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                name='mask_lm_trans_layer_norm_bias',
                initializer=fluid.initializer.Constant(1.)))

        seq2seq_out_bias_attr = fluid.ParamAttr(
            name="mask_lm_out_fc.b_0",
            initializer=fluid.initializer.Constant(value=0.0))

        if self.weight_sharing:
            fc_out = fluid.layers.matmul(
                x=tgt_trans_feat,
                y=fluid.default_main_program().global_block().var(
                    "word_embedding"),
                transpose_y=True)
            fc_out += fluid.layers.create_parameter(
                shape=[self.vl_config['vocab_size']],
                dtype="float32",
                attr=seq2seq_out_bias_attr,
                is_bias=True)
        else:
            out_size = self.vl_config["tgt_vocab_size"] or self.vl_config['vocab_size']
            fc_out = fluid.layers.fc(input=tgt_trans_feat,
                                     size=out_size,
                                     param_attr=fluid.ParamAttr(
                                         name="mask_lm_out_fc.w_0",
                                         initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
                                     bias_attr=seq2seq_out_bias_attr)

        return fc_out

    def to_tensor(self, shapes, dtypes, lod_levels):
        """to tensor"""
        return [fluid.layers.data(name="placeholder_" + str(i), shape=shapes[i], dtype=dtypes[i],
                                  lod_level=lod_levels[i]) for i in range(len(shapes))]

    def create_model_freelb_text(self):
        """create text freelb model"""
        img_input_shapes = [[-1, self.max_img_len, self.vl_config["image_embedding_size"]],  # image_embedding
                            [-1, self.max_img_len, 5],  # image_loc
                            [-1, self.max_seq_len + self.max_obj_len + self.max_img_len,
                             self.max_seq_len + self.max_obj_len + self.max_img_len],  # input_mask
                            [-1, self.max_img_len, 1],  # v_mask
                            [-1, self.max_seq_len, 1],  # t_mask
                            [-1, self.max_obj_len, 1],  # padded_obj_token_id
                            [-1, self.max_obj_len, 1],  # padded_obj_sent_ids
                            [-1, self.max_obj_len, 1]]  # padded_obj_pos_ids
        img_input_dtypes = ['float32', 'float32', 'float32', 'float32', 'float32', 'int64', 'int64', 'int64']
        img_input_lod_levels = [0, 0, 0, 0, 0, 0, 0, 0]

        emb_num = 3
        text_input_shapes = [[-1, self.max_seq_len, 1]] * emb_num + [[-1, 1], [-1, 1]]
        text_input_dtypes = ['int64'] * emb_num + ['int64', 'int64']
        text_input_lod_levels = [0] * emb_num + [0, 0]

        shapes = img_input_shapes + text_input_shapes
        dtypes = img_input_dtypes + text_input_dtypes
        lod_levels = img_input_lod_levels + text_input_lod_levels

        inputs = self.to_tensor(shapes, dtypes, lod_levels)
        pyreader = fluid.io.DataLoader.from_generator(feed_list=inputs, capacity=70, iterable=False)

        img_embs = {}
        emb_obj_ids = {}
        emb_ids = {}
        img_embs["image_embedding"], img_embs["loc_embedding"], input_mask, v_mask, t_mask, \
        emb_obj_ids["word_embedding"], emb_obj_ids["sent_embedding"], emb_obj_ids["pos_embedding"], \
        emb_ids["word_embedding"], emb_ids["sent_embedding"], emb_ids["pos_embedding"], \
        tgt_labels, tgt_pos = inputs

        tot_loss = None
        adv_step = self.args.adv_step
        adv_lr = self.args.adv_lr
        norm_type = self.args.norm_type
        adv_max_norm = self.args.adv_max_norm
        adv_init_mag = self.args.adv_init_mag

        # shape(embedded) = (batch_size, num_timesteps, embedding_dim)
        _emb_shape = [-1, self.max_seq_len, self.vl_config['hidden_size']]
        _fake = fluid.layers.data(name="_fake", shape=_emb_shape, dtype='float32')
        t_mask_slice = fluid.layers.slice(t_mask, axes=[1], starts=[0], ends=fluid.layers.shape(t_mask)[1])
        t_length = fluid.layers.reduce_sum(t_mask_slice, dim=1, keep_dim=True) * self.vl_config['hidden_size']

        if adv_init_mag > 0:
            if norm_type == 'l2':
                delta = fluid.layers.uniform_random_batch_size_like(t_mask, shape=_fake.shape, min=-1.0, max=1.0)
                delta = fluid.layers.slice(delta, axes=[1], starts=[0],
                                           ends=fluid.layers.shape(emb_ids["word_embedding"])[1])
                delta = delta * t_mask_slice
                mag = adv_init_mag / fluid.layers.sqrt(t_length)
                delta = delta * mag
            elif norm_type == 'inf':
                delta = fluid.layers.uniform_random_batch_size_like(t_mask, shape=_fake.shape, min=-adv_init_mag,
                                                                    max=adv_init_mag)
                delta = fluid.layers.slice(delta, axes=[1], starts=[0],
                                           ends=fluid.layers.shape(emb_ids["word_embedding"])[1])
                delta = delta * t_mask_slice
            else:
                print("Norm type not specified: ", norm_type)
                exit()
        else:
            delta = fluid.layers.uniform_random_batch_size_like(t_mask, shape=_fake.shape, min=0.0, max=0.0)
            delta = fluid.layers.slice(delta, axes=[1], starts=[0],
                                       ends=fluid.layers.shape(emb_ids["word_embedding"])[1])
            delta = delta * t_mask_slice

        for iter in range(adv_step):
            if self.with_pure_model:
                gene_pure = UNIMOModel(
                    emb_ids=emb_ids,
                    emb_obj_ids=emb_obj_ids,
                    input_mask=input_mask,
                    config=self.vl_config,
                    image_input=img_embs,
                    weight_sharing=self.weight_sharing,
                    task_type=self.task_type
                )

                pure_enc_out = gene_pure.get_sequence_output()
                pure_fc_out = self.cal_logit(pure_enc_out, tgt_pos)

                if self.label_smooth:
                    out_size = self.vl_config['vocab_size']
                    labels = fluid.layers.label_smooth(
                        label=fluid.layers.one_hot(
                            input=tgt_labels, depth=out_size),
                        epsilon=self.label_smooth)

                    pure_ce_loss = layers.softmax_with_cross_entropy(
                        logits=pure_fc_out, label=labels, soft_label=True)
                else:
                    pure_ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
                        logits=pure_fc_out, label=tgt_labels, return_softmax=True)

                pure_loss = fluid.layers.mean(x=pure_ce_loss)
                pure_loss = pure_loss / adv_step

            gene = UNIMOModel(
                text_adv_delta=delta,
                emb_ids=emb_ids,
                emb_obj_ids=emb_obj_ids,
                input_mask=input_mask,
                config=self.vl_config,
                image_input=img_embs,
                weight_sharing=self.weight_sharing,
                task_type=self.task_type
            )

            enc_out = gene.get_sequence_output()
            fc_out = self.cal_logit(enc_out, tgt_pos)

            if self.label_smooth:
                out_size = self.vl_config['vocab_size']
                labels = fluid.layers.label_smooth(
                    label=fluid.layers.one_hot(
                        input=tgt_labels, depth=out_size),
                    epsilon=self.label_smooth)

                ce_loss = layers.softmax_with_cross_entropy(
                    logits=fc_out, label=labels, soft_label=True)
            else:
                ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
                    logits=fc_out, label=tgt_labels, return_softmax=True)

            loss = fluid.layers.mean(x=ce_loss)
            loss = loss / adv_step

            delta_grad = fluid.backward.gradients(loss, delta)
            delta_grad = delta_grad[0]
            if norm_type == 'l2':
                # update according to grads
                # whether to use scale_l2
                delta_norm = fluid.layers.sqrt(
                    fluid.layers.reduce_sum(
                        fluid.layers.pow(fluid.layers.reshape(delta_grad, [fluid.layers.shape(delta_grad)[0], -1]),
                                         factor=2), dim=1, keep_dim=True))
                _min = float(1e-8)
                delta_norm = fluid.layers.clamp(delta_norm, min=_min)
                delta = delta + adv_lr * delta_grad / delta_norm
                # projection
                if adv_max_norm > 0:
                    exceed_mask = (delta_norm > adv_max_norm).astype('float32')
                    reweights = (adv_max_norm / delta_norm) * exceed_mask + (1 - exceed_mask)
                    delta = delta * reweights
            elif norm_type == 'inf':
                delta_norm = fluid.layers.reduce_max(
                    fluid.layers.abs(fluid.layers.reshape(delta_grad, [fluid.layers.shape(delta_grad)[0], -1])), \
                    dim=1, keep_dim=True)
                _min = float(1e-8)
                delta_norm = fluid.layers.clamp(delta_norm, min=_min)
                delta = delta + adv_lr * delta_grad / delta_norm
                # projection
                if adv_max_norm > 0:
                    delta = fluid.layers.clamp(delta, min=-adv_max_norm, max=adv_max_norm)
            else:
                print("Norm type not specified: ", norm_type)
                exit()
            delta_grad.stop_gradient = True

            if self.with_pure_model:
                kl_adv_text_loss = self._kl_divergence_with_logits(pure_fc_out, fc_out)
                cur_loss = pure_loss + loss + self.adv_kl_weight * kl_adv_text_loss
            else:
                cur_loss = loss

            tot_loss = cur_loss if tot_loss is None else tot_loss + cur_loss

        graph_vars = {"loss": tot_loss}
        for k, v in graph_vars.items():
            v.persistable = True

        return pyreader, graph_vars

    def create_model_freelb_image(self):
        """create image freelb model"""
        img_input_shapes = [[-1, self.max_img_len, self.vl_config["image_embedding_size"]],  # image_embedding
                            [-1, self.max_img_len, 5],  # image_loc
                            [-1, self.max_seq_len + self.max_obj_len + self.max_img_len,
                             self.max_seq_len + self.max_obj_len + self.max_img_len],  # input_mask
                            [-1, self.max_img_len, 1],  # v_mask
                            [-1, self.max_seq_len, 1],  # t_mask
                            [-1, self.max_obj_len, 1],  # padded_obj_token_id
                            [-1, self.max_obj_len, 1],  # padded_obj_sent_ids
                            [-1, self.max_obj_len, 1]]  # padded_obj_pos_ids
        img_input_dtypes = ['float32', 'float32', 'float32', 'float32', 'float32', 'int64', 'int64', 'int64']
        img_input_lod_levels = [0, 0, 0, 0, 0, 0, 0, 0]

        emb_num = 3
        text_input_shapes = [[-1, self.max_seq_len, 1]] * emb_num + [[-1, 1], [-1, 1]]
        text_input_dtypes = ['int64'] * emb_num + ['int64', 'int64']
        text_input_lod_levels = [0] * emb_num + [0, 0]

        shapes = img_input_shapes + text_input_shapes
        dtypes = img_input_dtypes + text_input_dtypes
        lod_levels = img_input_lod_levels + text_input_lod_levels

        inputs = self.to_tensor(shapes, dtypes, lod_levels)
        pyreader = fluid.io.DataLoader.from_generator(feed_list=inputs, capacity=70, iterable=False)

        img_embs = {}
        emb_obj_ids = {}
        emb_ids = {}
        img_embs["image_embedding"], img_embs["loc_embedding"], input_mask, v_mask, t_mask, \
        emb_obj_ids["word_embedding"], emb_obj_ids["sent_embedding"], emb_obj_ids["pos_embedding"], \
        emb_ids["word_embedding"], emb_ids["sent_embedding"], emb_ids["pos_embedding"], \
        tgt_labels, tgt_pos = inputs

        tot_loss = None
        adv_step = self.args.adv_step
        adv_lr = self.args.adv_lr
        norm_type = self.args.norm_type
        adv_max_norm = self.args.adv_max_norm
        adv_init_mag = self.args.adv_init_mag

        # shape(embedded) = (batch_size, num_timesteps, embedding_dim)
        _emb_shape = [-1, self.max_img_len, self.vl_config["image_embedding_size"]]
        _fake = fluid.layers.data(name="_fake", shape=_emb_shape, dtype='float32')
        v_mask_slice = fluid.layers.slice(v_mask, axes=[1], starts=[0], ends=fluid.layers.shape(v_mask)[1])
        v_length = fluid.layers.reduce_sum(v_mask_slice, dim=1, keep_dim=True) * self.vl_config["image_embedding_size"]

        if adv_init_mag > 0:
            if norm_type == 'l2':
                delta = fluid.layers.uniform_random_batch_size_like(v_mask, shape=_fake.shape, min=-1.0, max=1.0)
                delta = fluid.layers.slice(delta, axes=[1], starts=[0],
                                           ends=fluid.layers.shape(img_embs["image_embedding"])[1])
                delta = delta * v_mask_slice
                mag = adv_init_mag / fluid.layers.sqrt(v_length)
                delta = delta * mag
            elif norm_type == 'inf':
                delta = fluid.layers.uniform_random_batch_size_like(v_mask, shape=_fake.shape, min=-adv_init_mag,
                                                                    max=adv_init_mag)
                delta = fluid.layers.slice(delta, axes=[1], starts=[0],
                                           ends=fluid.layers.shape(img_embs["image_embedding"])[1])
                delta = delta * v_mask_slice
            else:
                print("Norm type not specified: ", norm_type)
                exit()
        else:
            delta = fluid.layers.uniform_random_batch_size_like(v_mask, shape=_fake.shape, min=0.0, max=0.0)
            delta = fluid.layers.slice(delta, axes=[1], starts=[0],
                                       ends=fluid.layers.shape(img_embs["image_embedding"])[1])
            delta = delta * v_mask_slice

        for iter in range(adv_step):
            if self.with_pure_model:
                gene_pure = UNIMOModel(
                    emb_ids=emb_ids,
                    emb_obj_ids=emb_obj_ids,
                    input_mask=input_mask,
                    config=self.vl_config,
                    image_input=img_embs,
                    weight_sharing=self.weight_sharing,
                    task_type=self.task_type
                )

                pure_enc_out = gene_pure.get_sequence_output()
                pure_fc_out = self.cal_logit(pure_enc_out, tgt_pos)

                if self.label_smooth:
                    out_size = self.vl_config['vocab_size']
                    labels = fluid.layers.label_smooth(
                        label=fluid.layers.one_hot(
                            input=tgt_labels, depth=out_size),
                        epsilon=self.label_smooth)

                    pure_ce_loss = layers.softmax_with_cross_entropy(
                        logits=pure_fc_out, label=labels, soft_label=True)
                else:
                    pure_ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
                        logits=pure_fc_out, label=tgt_labels, return_softmax=True)

                pure_loss = fluid.layers.mean(x=pure_ce_loss)
                pure_loss = pure_loss / adv_step

            gene = UNIMOModel(
                image_adv_delta=delta,
                emb_ids=emb_ids,
                emb_obj_ids=emb_obj_ids,
                input_mask=input_mask,
                config=self.vl_config,
                image_input=img_embs,
                weight_sharing=self.weight_sharing,
                task_type=self.task_type
            )

            enc_out = gene.get_sequence_output()
            fc_out = self.cal_logit(enc_out, tgt_pos)

            if self.label_smooth:
                out_size = self.vl_config['vocab_size']
                labels = fluid.layers.label_smooth(
                    label=fluid.layers.one_hot(
                        input=tgt_labels, depth=out_size),
                    epsilon=self.label_smooth)

                ce_loss = layers.softmax_with_cross_entropy(
                    logits=fc_out, label=labels, soft_label=True)
            else:
                ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
                    logits=fc_out, label=tgt_labels, return_softmax=True)

            loss = fluid.layers.mean(x=ce_loss)
            loss = loss / adv_step
            delta_grad = fluid.backward.gradients(loss, delta)
            delta_grad = delta_grad[0]
            if norm_type == 'l2':
                # update according to grads
                # whether to use scale_l2
                delta_norm = fluid.layers.sqrt(
                    fluid.layers.reduce_sum(
                        fluid.layers.pow(fluid.layers.reshape(delta_grad, [fluid.layers.shape(delta_grad)[0], -1]),
                                         factor=2), dim=1, keep_dim=True))
                _min = float(1e-8)
                delta_norm = fluid.layers.clamp(delta_norm, min=_min)
                delta = delta + adv_lr * delta_grad / delta_norm
                # projection
                if adv_max_norm > 0:
                    exceed_mask = (delta_norm > adv_max_norm).astype('float32')
                    reweights = (adv_max_norm / delta_norm) * exceed_mask + (1 - exceed_mask)
                    delta = delta * reweights
            elif norm_type == 'inf':
                delta_norm = fluid.layers.reduce_max(
                    fluid.layers.abs(fluid.layers.reshape(delta_grad, [fluid.layers.shape(delta_grad)[0], -1])), \
                    dim=1, keep_dim=True)
                _min = float(1e-8)
                delta_norm = fluid.layers.clamp(delta_norm, min=_min)
                delta = delta + adv_lr * delta_grad / delta_norm
                # projection
                if adv_max_norm > 0:
                    delta = fluid.layers.clamp(delta, min=-adv_max_norm, max=adv_max_norm)
            else:
                print("Norm type not specified: ", norm_type)
                exit()
            delta_grad.stop_gradient = True

            if self.with_pure_model:
                kl_adv_text_loss = self._kl_divergence_with_logits(pure_fc_out, fc_out)
                cur_loss = pure_loss + loss + self.adv_kl_weight * kl_adv_text_loss
            else:
                cur_loss = loss

            tot_loss = cur_loss if tot_loss is None else tot_loss + cur_loss

        graph_vars = {"loss": tot_loss}
        for k, v in graph_vars.items():
            v.persistable = True

        return pyreader, graph_vars

    def create_model_villa(self):
        """create villa model"""
        img_input_shapes = [[-1, self.max_img_len, self.vl_config["image_embedding_size"]],  # image_embedding
                            [-1, self.max_img_len, 5],  # image_loc
                            [-1, self.max_seq_len + self.max_obj_len + self.max_img_len,
                             self.max_seq_len + self.max_obj_len + self.max_img_len],  # input_mask
                            [-1, self.max_img_len, 1],  # v_mask
                            [-1, self.max_seq_len, 1],  # t_mask
                            [-1, self.max_obj_len, 1],  # padded_obj_token_id
                            [-1, self.max_obj_len, 1],  # padded_obj_sent_ids
                            [-1, self.max_obj_len, 1]]  # padded_obj_pos_ids
        img_input_dtypes = ['float32', 'float32', 'float32', 'float32', 'float32', 'int64', 'int64', 'int64']
        img_input_lod_levels = [0, 0, 0, 0, 0, 0, 0, 0]

        emb_num = 3
        text_input_shapes = [[-1, self.max_seq_len, 1]] * emb_num + [[-1, 1], [-1, 1]]
        text_input_dtypes = ['int64'] * emb_num + ['int64', 'int64']
        text_input_lod_levels = [0] * emb_num + [0, 0]

        shapes = img_input_shapes + text_input_shapes
        dtypes = img_input_dtypes + text_input_dtypes
        lod_levels = img_input_lod_levels + text_input_lod_levels

        inputs = self.to_tensor(shapes, dtypes, lod_levels)
        pyreader = fluid.io.DataLoader.from_generator(feed_list=inputs, capacity=70, iterable=False)

        img_embs = {}
        emb_obj_ids = {}
        emb_ids = {}
        img_embs["image_embedding"], img_embs["loc_embedding"], input_mask, v_mask, t_mask, \
        emb_obj_ids["word_embedding"], emb_obj_ids["sent_embedding"], emb_obj_ids["pos_embedding"], \
        emb_ids["word_embedding"], emb_ids["sent_embedding"], emb_ids["pos_embedding"], \
        tgt_labels, tgt_pos = inputs

        tot_loss = None
        adv_step = self.args.adv_step
        adv_lr = self.args.adv_lr
        norm_type = self.args.norm_type
        adv_max_norm = self.args.adv_max_norm
        adv_init_mag = self.args.adv_init_mag

        # shape(embedded) = (batch_size, num_timesteps, embedding_dim)
        t_emb_shape = [-1, self.max_seq_len, self.vl_config['hidden_size']]
        t_fake = fluid.layers.data(name="t_fake", shape=t_emb_shape, dtype='float32')
        t_mask_slice = fluid.layers.slice(t_mask, axes=[1], starts=[0], ends=fluid.layers.shape(t_mask)[1])
        t_length = fluid.layers.reduce_sum(t_mask_slice, dim=1, keep_dim=True) * self.vl_config['hidden_size']

        # shape(embedded) = (batch_size, num_timesteps, embedding_dim)
        v_emb_shape = [-1, self.max_img_len, self.vl_config["image_embedding_size"]]
        v_fake = fluid.layers.data(name="v_fake", shape=v_emb_shape, dtype='float32')
        v_mask_slice = fluid.layers.slice(v_mask, axes=[1], starts=[0], ends=fluid.layers.shape(v_mask)[1])
        v_length = fluid.layers.reduce_sum(v_mask_slice, dim=1, keep_dim=True) * self.vl_config["image_embedding_size"]

        if adv_init_mag > 0:
            if norm_type == 'l2':
                t_delta = fluid.layers.uniform_random_batch_size_like(t_mask, shape=t_fake.shape, min=-1.0, max=1.0)
                t_delta = fluid.layers.slice(t_delta, axes=[1], starts=[0],
                                             ends=fluid.layers.shape(emb_ids["word_embedding"])[1])
                t_delta = t_delta * t_mask_slice
                t_mag = adv_init_mag / fluid.layers.sqrt(t_length)
                t_delta = t_delta * t_mag

                v_delta = fluid.layers.uniform_random_batch_size_like(v_mask, shape=v_fake.shape, min=-1.0, max=1.0)
                v_delta = fluid.layers.slice(v_delta, axes=[1], starts=[0],
                                             ends=fluid.layers.shape(img_embs["image_embedding"])[1])
                v_delta = v_delta * v_mask_slice
                v_mag = adv_init_mag / fluid.layers.sqrt(v_length)
                v_delta = v_delta * v_mag
            elif norm_type == 'inf':
                t_delta = fluid.layers.uniform_random_batch_size_like(t_mask, shape=t_fake.shape, min=-adv_init_mag,
                                                                      max=adv_init_mag)
                t_delta = fluid.layers.slice(t_delta, axes=[1], starts=[0],
                                             ends=fluid.layers.shape(emb_ids["word_embedding"])[1])
                t_delta = t_delta * t_mask_slice

                v_delta = fluid.layers.uniform_random_batch_size_like(v_mask, shape=v_fake.shape, min=-adv_init_mag,
                                                                      max=adv_init_mag)
                v_delta = fluid.layers.slice(v_delta, axes=[1], starts=[0],
                                             ends=fluid.layers.shape(img_embs["image_embedding"])[1])
                v_delta = v_delta * v_mask_slice
            else:
                print("Norm type not specified: ", norm_type)
                exit()
        else:
            t_delta = fluid.layers.uniform_random_batch_size_like(t_mask, shape=t_fake.shape, min=0.0, max=0.0)
            t_delta = fluid.layers.slice(t_delta, axes=[1], starts=[0],
                                         ends=fluid.layers.shape(emb_ids["word_embedding"])[1])
            t_delta = t_delta * t_mask_slice

            v_delta = fluid.layers.uniform_random_batch_size_like(v_mask, shape=v_fake.shape, min=0.0, max=0.0)
            v_delta = fluid.layers.slice(v_delta, axes=[1], starts=[0],
                                         ends=fluid.layers.shape(img_embs["image_embedding"])[1])
            v_delta = v_delta * v_mask_slice

        for iter in range(adv_step):
            if self.with_pure_model:
                gene_pure = UNIMOModel(
                    emb_ids=emb_ids,
                    emb_obj_ids=emb_obj_ids,
                    input_mask=input_mask,
                    config=self.vl_config,
                    image_input=img_embs,
                    weight_sharing=self.weight_sharing,
                    task_type=self.task_type
                )

                pure_enc_out = gene_pure.get_sequence_output()
                pure_fc_out = self.cal_logit(pure_enc_out, tgt_pos)

                if self.label_smooth:
                    out_size = self.vl_config['vocab_size']
                    labels = fluid.layers.label_smooth(
                        label=fluid.layers.one_hot(
                            input=tgt_labels, depth=out_size),
                        epsilon=self.label_smooth)

                    pure_ce_loss = layers.softmax_with_cross_entropy(
                        logits=pure_fc_out, label=labels, soft_label=True)
                else:
                    pure_ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
                        logits=pure_fc_out, label=tgt_labels, return_softmax=True)

                pure_loss = fluid.layers.mean(x=pure_ce_loss)
                pure_loss = pure_loss / adv_step

            # text adversial learning
            gene_text = UNIMOModel(
                text_adv_delta=t_delta,
                emb_ids=emb_ids,
                emb_obj_ids=emb_obj_ids,
                input_mask=input_mask,
                config=self.vl_config,
                image_input=img_embs,
                weight_sharing=self.weight_sharing,
                task_type=self.task_type
            )

            text_enc_out = gene_text.get_sequence_output()
            text_fc_out = self.cal_logit(text_enc_out, tgt_pos)

            if self.label_smooth:
                out_size = self.vl_config['vocab_size']
                labels = fluid.layers.label_smooth(
                    label=fluid.layers.one_hot(
                        input=tgt_labels, depth=out_size),
                    epsilon=self.label_smooth)

                text_ce_loss = layers.softmax_with_cross_entropy(
                    logits=text_fc_out, label=labels, soft_label=True)
            else:
                text_ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
                    logits=text_fc_out, label=tgt_labels, return_softmax=True)

            text_loss = fluid.layers.mean(x=text_ce_loss)
            text_loss = text_loss / adv_step

            # image adversial learning
            gene_img = UNIMOModel(
                image_adv_delta=v_delta,
                emb_ids=emb_ids,
                emb_obj_ids=emb_obj_ids,
                input_mask=input_mask,
                config=self.vl_config,
                image_input=img_embs,
                weight_sharing=self.weight_sharing,
                task_type=self.task_type
            )

            img_enc_out = gene_img.get_sequence_output()
            img_fc_out = self.cal_logit(img_enc_out, tgt_pos)

            if self.label_smooth:
                out_size = self.vl_config['vocab_size']
                labels = fluid.layers.label_smooth(
                    label=fluid.layers.one_hot(
                        input=tgt_labels, depth=out_size),
                    epsilon=self.label_smooth)

                img_ce_loss = layers.softmax_with_cross_entropy(
                    logits=img_fc_out, label=labels, soft_label=True)
            else:
                img_ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
                    logits=img_fc_out, label=tgt_labels, return_softmax=True)

            img_loss = fluid.layers.mean(x=img_ce_loss)
            img_loss = img_loss / adv_step

            # update delta
            delta_grad_text = fluid.backward.gradients(text_loss, t_delta)
            delta_grad_img = fluid.backward.gradients(img_loss, v_delta)
            delta_grad_text = delta_grad_text[0]
            delta_grad_img = delta_grad_img[0]
            if norm_type == 'l2':
                # update according to grads
                # whether to use scale_l2
                # text part
                delta_norm_text = fluid.layers.sqrt(
                    fluid.layers.reduce_sum(
                        fluid.layers.pow(fluid.layers.reshape(delta_grad_text,
                                                              [fluid.layers.shape(delta_grad_text)[0], -1]), factor=2),
                        dim=1, keep_dim=True))
                _min = float(1e-8)
                delta_norm_text = fluid.layers.clamp(delta_norm_text, min=_min)
                t_delta = t_delta + adv_lr * delta_grad_text / delta_norm_text

                # image part
                delta_norm_img = fluid.layers.sqrt(
                    fluid.layers.reduce_sum(
                        fluid.layers.pow(fluid.layers.reshape(delta_grad_img,
                                                              [fluid.layers.shape(delta_grad_img)[0], -1]), factor=2),
                        dim=1, keep_dim=True))
                _min = float(1e-8)
                delta_norm_img = fluid.layers.clamp(delta_norm_img, min=_min)
                v_delta = v_delta + adv_lr * delta_grad_img / delta_norm_img

                # projection
                if adv_max_norm > 0:
                    # text part
                    exceed_mask_text = (delta_norm_text > adv_max_norm).astype('float32')
                    reweights_text = (adv_max_norm / delta_norm_text) * exceed_mask_text + (1 - exceed_mask_text)
                    t_delta = t_delta * reweights_text

                    # image part
                    exceed_mask_img = (delta_norm_img > adv_max_norm).astype('float32')
                    reweights_img = (adv_max_norm / delta_norm_img) * exceed_mask_img + (1 - exceed_mask_img)
                    v_delta = v_delta * reweights_img

            elif norm_type == 'inf':
                # text part
                delta_norm_text = fluid.layers.reduce_max(
                    fluid.layers.abs(fluid.layers.reshape(delta_grad_text,
                                                          [fluid.layers.shape(delta_grad_text)[0], -1])),
                    dim=1, keep_dim=True)
                _min = float(1e-8)
                delta_norm_text = fluid.layers.clamp(delta_norm_text, min=_min)
                t_delta = t_delta + adv_lr * delta_grad_text / delta_norm_text

                # image part
                delta_norm_image = fluid.layers.reduce_max(
                    fluid.layers.abs(fluid.layers.reshape(delta_grad_img,
                                                          [fluid.layers.shape(delta_grad_img)[0], -1])),
                    dim=1, keep_dim=True)
                _min = float(1e-8)
                delta_norm_image = fluid.layers.clamp(delta_norm_image, min=_min)
                v_delta = v_delta + adv_lr * delta_grad_img / delta_norm_image

                # projection
                if adv_max_norm > 0:
                    t_delta = fluid.layers.clamp(t_delta, min=-adv_max_norm, max=adv_max_norm)
                    v_delta = fluid.layers.clamp(v_delta, min=-adv_max_norm, max=adv_max_norm)
            else:
                print("Norm type not specified: ", norm_type)
                exit()

            # delta.stop_gradient=True
            delta_grad_text.stop_gradient = True
            delta_grad_img.stop_gradient = True

            if self.with_pure_model:
                kl_adv_text_loss = self._kl_divergence_with_logits(pure_fc_out, text_fc_out)
                kl_adv_image_loss = self._kl_divergence_with_logits(pure_fc_out, img_fc_out)
                cur_loss = pure_loss + text_loss + img_loss + self.adv_kl_weight * (kl_adv_text_loss + kl_adv_image_loss)
            else:
                cur_loss = text_loss + img_loss

            tot_loss = cur_loss if tot_loss is None else tot_loss + cur_loss

        graph_vars = {"loss": tot_loss}
        for k, v in graph_vars.items():
            v.persistable = True

        return pyreader, graph_vars

    def create_model(self, decoding=False):
        """create model for training"""
        if decoding:
            return self.fast_decode()

        img_input_shapes = [[-1, self.max_img_len, self.vl_config["image_embedding_size"]],  # image_embedding
                            [-1, self.max_img_len, 5],  # image_loc
                            [-1, self.max_seq_len + self.max_obj_len + self.max_img_len,
                             self.max_seq_len + self.max_obj_len + self.max_img_len],  # input_mask
                            [-1, self.max_img_len, 1],  # v_mask
                            [-1, self.max_seq_len, 1],  # t_mask
                            [-1, self.max_obj_len, 1],  # padded_obj_token_id
                            [-1, self.max_obj_len, 1],  # padded_obj_sent_ids
                            [-1, self.max_obj_len, 1]]  # padded_obj_pos_ids
        img_input_dtypes = ['float32', 'float32', 'float32', 'int64', 'int64', 'int64']
        img_input_lod_levels = [0, 0, 0, 0, 0, 0]

        emb_num = 3
        text_input_shapes = [[-1, self.max_seq_len, 1]] * emb_num + [[-1, 1], [-1, 1]]
        text_input_dtypes = ['int64'] * emb_num + ['int64', 'int64']
        text_input_lod_levels = [0] * emb_num + [0, 0]

        shapes = img_input_shapes + text_input_shapes
        dtypes = img_input_dtypes + text_input_dtypes
        lod_levels = img_input_lod_levels + text_input_lod_levels

        inputs = self.to_tensor(shapes, dtypes, lod_levels)
        pyreader = fluid.io.DataLoader.from_generator(feed_list=inputs, capacity=70, iterable=False)

        img_embs = {}
        emb_obj_ids = {}
        emb_ids = {}
        img_embs["image_embedding"], img_embs["loc_embedding"], input_mask, v_mask, t_mask, \
        emb_obj_ids["word_embedding"], emb_obj_ids["sent_embedding"], emb_obj_ids["pos_embedding"], \
        emb_ids["word_embedding"], emb_ids["sent_embedding"], emb_ids["pos_embedding"], \
        tgt_labels, tgt_pos = inputs

        gene = UNIMOModel(
            emb_ids=emb_ids,
            emb_obj_ids=emb_obj_ids,
            input_mask=input_mask,
            config=self.vl_config,
            image_input=img_embs,
            weight_sharing=self.weight_sharing,
            task_type=self.task_type)

        enc_out = gene.get_sequence_output()
        fc_out = self.cal_logit(enc_out, tgt_pos)

        if self.label_smooth:
            out_size = self.vl_config['vocab_size']
            labels = fluid.layers.label_smooth(
                label=fluid.layers.one_hot(
                    input=tgt_labels, depth=out_size),
                epsilon=self.label_smooth)

            ce_loss = layers.softmax_with_cross_entropy(
                logits=fc_out, label=labels, soft_label=True)
        else:
            ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
                logits=fc_out, label=tgt_labels, return_softmax=True)

        loss = fluid.layers.mean(x=ce_loss)
        graph_vars = {"loss": loss}
        for k, v in graph_vars.items():
            v.persistable = True

        return pyreader, graph_vars

    def fast_decode(self):
        """create model for inference"""
        input_shapes = [[-1, self.max_img_len, self.vl_config["image_embedding_size"]],  # image_embedding
                        [-1, self.max_img_len, 5],  # image_loc
                        [-1, self.max_img_len + self.max_obj_len, self.max_img_len + self.max_obj_len],# img_input_mask
                        [-1, 1],  # image_id
                        [-1, self.max_obj_len, 1],  # padded_obj_token_id
                        [-1, self.max_obj_len, 1],  # padded_obj_sent_ids
                        [-1, self.max_obj_len, 1]]  # padded_obj_pos_ids
        input_dtypes = ['float32', 'float32', 'float32', 'int32', 'int64', 'int64', 'int64']
        input_lod_levels = [0, 0, 0, 0, 0, 0, 0]

        shapes = input_shapes + [[-1, 1, 1], [-1, 1, 1],
                                 [-1, 1], [-1], [-1, 1, self.max_img_len + self.max_obj_len]]
        dtypes = input_dtypes + ['int64', 'int64', 'float32', 'int32', 'float32']
        lod_levels = input_lod_levels + [2, 2, 2, 0, 0]

        inputs = self.to_tensor(shapes, dtypes, lod_levels)
        pyreader = fluid.io.DataLoader.from_generator(feed_list=inputs, capacity=70, iterable=False)

        img_embs = {}
        emb_obj_ids = {}
        img_embs["image_embedding"], img_embs["loc_embedding"], input_mask, image_ids, \
        emb_obj_ids["word_embedding"], emb_obj_ids["sent_embedding"], emb_obj_ids["pos_embedding"], \
        tgt_ids, tgt_pos, init_scores, parent_idx, tgt_input_mask = inputs

        gene = UNIMOModel(
            emb_obj_ids=emb_obj_ids,
            input_mask=input_mask,
            image_input=img_embs,
            config=self.vl_config,
            weight_sharing=self.weight_sharing,
            task_type=self.task_type,
            decoding=True,
            gather_idx=parent_idx)

        max_len = layers.fill_constant(
            shape=[1], dtype=tgt_ids.dtype, value=self.max_out_len, force_cpu=True)
        min_len = layers.fill_constant(
            shape=[1], dtype=tgt_ids.dtype, value=self.min_out_len, force_cpu=True)
        neg_inf = layers.fill_constant(
            shape=[1], dtype='float32', value=-1e18)
        step_idx = layers.fill_constant(
            shape=[1], dtype=tgt_ids.dtype, value=0, force_cpu=True)
        step_next_idx = layers.fill_constant(
            shape=[1], dtype=tgt_ids.dtype, value=1, force_cpu=True)
        cond = layers.less_than(x=step_idx, y=max_len)
        while_op = layers.While(cond)

        ids = layers.array_write(layers.reshape(tgt_ids, (-1, 1)), step_idx)
        pos_biases = layers.array_write(tgt_pos, step_idx)
        scores = layers.array_write(init_scores, step_idx)
        tgt_masks = layers.array_write(tgt_input_mask, step_idx)

        trigram_blocking = TrigramBlocking(tgt_ids, self.tokenizer, beam_size=self.beam_size)

        with while_op.block():
            pre_ids = layers.array_read(array=ids, i=step_idx)
            pre_ids = layers.reshape(pre_ids, (-1, 1, 1), inplace=True)
            pre_scores = layers.array_read(array=scores, i=step_idx)
            pos_bias = layers.array_read(array=pos_biases, i=step_idx)
            pos_bias = layers.gather(input=pos_bias, index=parent_idx)

            def gen_batch_like(value, dtype="int64", shape=[-1, 1, 1], is_scalar=True):
                """generate batch"""
                if is_scalar:
                    return layers.fill_constant_batch_size_like(
                        input=parent_idx, value=value, shape=shape, dtype=dtype)
                else:
                    return layers.elementwise_mul(
                        x=layers.fill_constant_batch_size_like(
                            input=parent_idx, value=1, shape=shape, dtype=dtype),
                        y=value, axis=0)

            tmp_mask = layers.array_read(tgt_masks, i=step_idx)
            tmp_mask = layers.gather(input=tmp_mask, index=parent_idx)
            append_1_mask = gen_batch_like(1.0, dtype=tmp_mask.dtype)
            pre_mask = layers.concat([tmp_mask, append_1_mask], axis=2)

            pre_pos = gen_batch_like(step_idx, is_scalar=False)
            pre_pos = pre_pos + pos_bias  ####################### pos start from 2
            pre_sent = gen_batch_like(self.tgt_type_id, dtype=pre_ids.dtype)

            dec_emb_ids = {"word_embedding": pre_ids, "sent_embedding": pre_sent, "pos_embedding": pre_pos}
            dec_out = gene.encode(emb_ids=dec_emb_ids,
                                  input_mask=pre_mask,
                                  gather_idx=parent_idx)
            fc_out = self.cal_logit(dec_out, None)

            # prevent generating end token if length less than min_out_len
            eos_index = layers.fill_constant(shape=[layers.shape(fc_out)[0]],
                                             dtype='int64',
                                             value=self.eos_id)
            eos_index = fluid.one_hot(eos_index, depth=self.vocab_size)
            less_cond = layers.cast(layers.less_than(x=step_idx, y=min_len), dtype='float32')
            less_val = layers.elementwise_mul(less_cond, neg_inf)
            eos_val = layers.elementwise_mul(eos_index, less_val, axis=0)
            revised_logits = layers.elementwise_add(fc_out, eos_val, axis=0)

            # topK reduction across beams, also contain special handle of
            # end beams and end sentences(batch reduction)
            topk_scores, topk_indices = layers.topk(
                input=layers.softmax(revised_logits), k=self.beam_size)

            # Roll-Back previous-scores for length-penalty
            # previous-scores has been length-penaltied, before this timestep length-penalty, need roll-back
            # because of doing this, we need store the length-penaltied score in `scores`
            # while calculating use the un-penaltied score
            # -> safe for step_idx == 0 (initialization state), because previous-score == 0
            pre_timestep_length_penalty = fluid.layers.pow(
                ((5.0 + fluid.layers.cast(step_idx, pre_scores.dtype)) / 6.0), self.length_penalty)
            pre_scores_wo_len_penalty = fluid.layers.elementwise_mul(pre_scores, pre_timestep_length_penalty)

            # calc trigram-blocking delta scores for current alive sequence
            if self.block_trigram:
                trigram_blocking.update_seq(pre_ids, parent_idx)
                trigram_blocking.expand_cand_seq(topk_indices)
                fluid.layers.py_func(func=trigram_blocking.blocking_forward,
                                     x=[trigram_blocking.cand_seq,
                                        trigram_blocking.id2is_full_token],
                                     out=trigram_blocking.delta_score_out,
                                     backward_func=None)
                pre_scores_wo_len_penalty = fluid.layers.elementwise_add(x=trigram_blocking.delta_score_out,
                                                                         y=pre_scores_wo_len_penalty,
                                                                         axis=0)
            # => [N, topk]
            accu_scores = layers.elementwise_add(
                x=layers.log(topk_scores), y=pre_scores_wo_len_penalty, axis=0)

            cur_timestep_length_penalty = layers.pow(((5.0 + layers.cast(step_next_idx, accu_scores.dtype)) / 6.0),
                                                     self.length_penalty)
            curr_scores = layers.elementwise_div(accu_scores, cur_timestep_length_penalty)

            # beam_search op uses lod to differentiate branches.
            curr_scores = layers.lod_reset(curr_scores, pre_ids)
            topk_indices = layers.lod_reset(topk_indices, pre_ids)
            selected_ids, selected_scores, gather_idx = layers.beam_search(
                pre_ids=pre_ids,
                pre_scores=pre_scores,
                ids=topk_indices,
                scores=curr_scores,
                beam_size=self.beam_size,
                end_id=self.eos_id,
                return_parent_idx=True)

            layers.increment(x=step_idx, value=1.0, in_place=True)
            layers.increment(x=step_next_idx, value=1.0, in_place=True)
            # cell states(caches) have been updated in wrap_decoder,
            # only need to update beam search states here.
            layers.array_write(selected_ids, i=step_idx, array=ids)
            layers.array_write(selected_scores, i=step_idx, array=scores)
            layers.array_write(pre_mask, i=step_idx, array=tgt_masks)
            layers.array_write(pos_bias, i=step_idx, array=pos_biases)
            layers.assign(gather_idx, parent_idx)

            length_cond = layers.less_than(x=step_idx, y=max_len)
            finish_cond = layers.logical_not(layers.is_empty(x=selected_ids))
            layers.logical_and(x=length_cond, y=finish_cond, out=cond)

        finished_ids, finished_scores = layers.beam_search_decode(
            ids, scores, beam_size=self.beam_size, end_id=self.eos_id)

        graph_vars = {
            "finished_ids": finished_ids,
            "finished_scores": finished_scores,
            "image_ids": image_ids
        }

        for k, v in graph_vars.items():
            v.persistable = True

        return pyreader, graph_vars

    def post_process_seq(self, seq):
        """
        Post-process the beam-search decoded sequence. Truncate from the first
        <eos> and remove the <bos> and <eos> tokens currently.
        """
        eos_pos = len(seq)
        for i, idx in enumerate(seq):
            if idx == self.eos_id:
                eos_pos = i
                break
        seq = seq[1:eos_pos]
        return seq

    def remove_special_tokens(self, seq, special_tokens):
        """Remove special tokens from output sequence"""
        seq = [idx for idx in seq if idx not in special_tokens]
        return seq

    def evaluate(self, resource, eval_phase, graph_vars, features=None,
                 output_path=None, dev_count=1, gpu_id=0):
        """evaluate model"""
        exe, program, pyreader = resource["exe"], resource["program"], resource["pyreader"]

        if eval_phase == "train":
            fetch_list = [graph_vars["loss"].name]
            if "learning_rate" in graph_vars:
                fetch_list.append(graph_vars["learning_rate"].name)
            outputs = exe.run(fetch_list=fetch_list)
            np_loss = outputs[0]
            ret = {"loss": np.mean(np_loss), "ppl": np.exp(np.mean(np_loss))}
            if "learning_rate" in graph_vars:
                ret["learning_rate"] = float(outputs[1][0])
            return ret

        if self.do_decode:
            return_numpy = False
            outfile = output_path + "/" + eval_phase
            outfile_part = outfile + ".part" + str(gpu_id)
            writer = codecs.open(outfile_part, 'w', encoding='utf-8')
            fetch_keys = ["finished_ids", "finished_scores", "image_ids"]
            special_tokens = [self.tokenizer.cls_token_id,
                              self.tokenizer.sep_token_id,
                              self.tokenizer.mask_token_id,
                              self.tokenizer.pad_token_id,
                              self.tokenizer.unk_token_id]
        else:
            steps = 0
            cost = 0.0
            return_numpy = True
            fetch_keys = ["loss"]

        fetch_list = [graph_vars[key].name for key in fetch_keys]

        time_begin = time.time()
        pyreader.start()
        while True:
            try:
                outputs = exe.run(program=program,
                                  fetch_list=fetch_list,
                                  return_numpy=return_numpy)
                if not self.do_decode:
                    np_loss = outputs[0]
                    cost += np.mean(np_loss)
                    steps += 1
                else:
                    seq_ids, seq_scores, image_ids = outputs
                    seq_ids_list, seq_scores_list = [seq_ids], [seq_scores] \
                        if isinstance(seq_ids, paddle.fluid.core.LoDTensor) else (seq_ids, seq_scores)

                    image_ids = np.array(image_ids).reshape(-1).tolist()
                    data_idx = 0

                    for seq_ids, seq_scores in zip(seq_ids_list, seq_scores_list):
                        # How to parse the results:
                        #   Suppose the lod of seq_ids is:
                        #     [[0, 3, 6], [0, 12, 24, 40, 54, 67, 82]]
                        #   then from lod[0]:
                        #     there are 2 source sentences, beam width is 3.
                        #   from lod[1]:
                        #     the first source sentence has 3 hyps; the lengths are 12, 12, 16
                        #     the second source sentence has 3 hyps; the lengths are 14, 13, 15
                        # hyps = [[] for i in range(len(seq_ids.lod()[0]) - 1)]
                        # scores = [[] for i in range(len(seq_scores.lod()[0]) - 1)]
                        for i in range(len(seq_ids.lod()[0]) - 1):  # for each source sentence
                            start = seq_ids.lod()[0][i]
                            end = seq_ids.lod()[0][i + 1]
                            max_cand = None
                            for j in range(end - start):  # for each candidate
                                sub_start = seq_ids.lod()[1][start + j]
                                sub_end = seq_ids.lod()[1][start + j + 1]
                                token_ids = [int(idx) for idx in self.post_process_seq(
                                    np.array(seq_ids)[sub_start:sub_end])]

                                hyp_ids = self.remove_special_tokens(token_ids, special_tokens)
                                hyp_tokens = self.tokenizer.convert_ids_to_tokens(hyp_ids)
                                hyp_str = self.tokenizer.gptbpe_tokenizer.decode(hyp_tokens)
                                hyp_str = re.sub('\\s+', ' ', hyp_str)

                                score = np.array(seq_scores)[sub_end - 1]
                                if (not max_cand) or score > max_cand[1]:
                                    max_cand = (hyp_str, score)

                            image_id = image_ids[data_idx]
                            data_idx += 1
                            pred = max_cand[0]
                            writer.write("%d\t%s\n" % (image_id, pred))

            except fluid.core.EOFException:
                pyreader.reset()
                break

        time_end = time.time()
        if not self.do_decode:
            eval_result = "loss: %f, ppl: %f" % (cost / steps, np.exp(cost / steps))
            print("[%s evaluation] %s, elapsed time: %f s"
                  % (eval_phase, eval_result, time_end - time_begin))
        else:
            writer.close()
            # tmp_writer = open("%s/%s_dec_finish.%d" % (output_path, eval_phase, gpu_id), "w")
            tmp_writer = codecs.open("%s/%s_dec_finish.%d" % (output_path, eval_phase, gpu_id),
                                     'w', encoding='utf-8')
            tmp_writer.close()
            if gpu_id != 0:
                return

            while True:
                ret = os.popen('find %s -maxdepth 1 -name "%s_dec_finish.*"' %
                               (output_path, eval_phase)).readlines()
                if len(ret) != dev_count:
                    time.sleep(1)
                    continue
                else:
                    break

            all_outfiles = glob.glob("%s.part*" % outfile)
            img_caption_res = []
            unique_image_ids = []
            for cur_file in all_outfiles:
                for line in codecs.open(cur_file, 'r', encoding='utf-8'):
                    image_id, caption = line.strip().split('\t')
                    if image_id in unique_image_ids:
                        print("Warning: Repeated image_id %s" % str(image_id))
                        continue
                    unique_image_ids.append(image_id)
                    img_caption_res.append({"image_id": int(image_id), "caption": caption})

            fout = codecs.open(outfile, 'w', encoding='utf-8')
            fout.write(json.dumps(img_caption_res))
            fout.close()
            os.system("rm %s.part*" % outfile)
            os.system("rm %s/%s_dec_finish.*" % (output_path, eval_phase))

            eval_result = self.evaluator.eval(outfile,
                                              phase=eval_phase.split("_")[0], features=features)
            print("[%s evaluation] %s, elapsed time: %f s"
                  % (eval_phase, eval_result, time_end - time_begin))
