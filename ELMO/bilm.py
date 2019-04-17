#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

# This file is used to finetune.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import paddle.fluid.layers as layers
import paddle.fluid as fluid
import numpy as np

# if you use our release weight layers,do not use the args.
cell_clip = 3.0
proj_clip = 3.0
hidden_size = 4096
vocab_size = 52445
embed_size = 512
# according to orginal paper, dropout need to be modifyed on finetune
modify_dropout = 1
proj_size = 512
num_layers = 2
random_seed = 0
dropout_rate = 0.5


def dropout(input):
    return layers.dropout(
        input,
        dropout_prob=dropout_rate,
        dropout_implementation="upscale_in_train",
        seed=random_seed,
        is_test=False)

def lstmp_encoder(input_seq, gate_size, h_0, c_0, para_name):
    # A lstm encoder implementation with projection.
    # Linear transformation part for input gate, output gate, forget gate
    # and cell activation vectors need be done outside of dynamic_lstm.
    # So the output size is 4 times of gate_size.

    input_proj = layers.fc(input=input_seq,
                           param_attr=fluid.ParamAttr(
                               name=para_name + '_gate_w', initializer=init),
                           size=gate_size * 4,
                           act=None,
                           bias_attr=False)
    hidden, cell = layers.dynamic_lstmp(
        input=input_proj,
        size=gate_size * 4,
        proj_size=proj_size,
        h_0=h_0,
        c_0=c_0,
        use_peepholes=False,
        proj_clip=proj_clip,
        cell_clip=cell_clip,
        proj_activation="identity",
        param_attr=fluid.ParamAttr(initializer=None),
        bias_attr=fluid.ParamAttr(initializer=None))
    return hidden, cell, input_proj


def encoder(x_emb,
            init_hidden=None,
            init_cell=None,
            para_name=''):
    rnn_input = x_emb
    rnn_outs = []
    rnn_outs_ori = []
    cells = []
    projs = []
    for i in range(num_layers):
        if init_hidden and init_cell:
            h0 = layers.squeeze(
                layers.slice(
                    init_hidden, axes=[0], starts=[i], ends=[i + 1]),
                axes=[0])
            c0 = layers.squeeze(
                layers.slice(
                    init_cell, axes=[0], starts=[i], ends=[i + 1]),
                axes=[0])
        else:
            h0 = c0 = None
        rnn_out, cell, input_proj = lstmp_encoder(
            rnn_input, hidden_size, h0, c0,
            para_name + 'layer{}'.format(i + 1))
        rnn_out_ori = rnn_out
        if i > 0:
            rnn_out = rnn_out + rnn_input
        rnn_out.stop_gradient = True
        rnn_outs.append(rnn_out)
        rnn_outs_ori.append(rnn_out_ori)
    # add weight layers for finetone
    a1 = layers.create_parameter(
        [1], dtype="float32", name="gamma1")
    a2 = layers.create_parameter(
        [1], dtype="float32", name="gamma2")
    rnn_outs[0].stop_gradient = True
    rnn_outs[1].stop_gradient = True
    num_layer1 = rnn_outs[0] * a1
    num_layer2 = rnn_outs[1] * a2
    output_layer = num_layer1 * 0.5 + num_layer2 * 0.5
    return output_layer, rnn_outs_ori


def emb(x):
    x_emb = layers.embedding(
        input=x,
        size=[vocab_size, embed_size],
        dtype='float32',
        is_sparse=False,
        param_attr=fluid.ParamAttr(name='embedding_para'))
    return x_emb


def elmo_encoder(x_emb):
    x_emb_r = fluid.layers.sequence_reverse(x_emb, name=None)
    fw_hiddens, fw_hiddens_ori = encoder(
        x_emb,
        para_name='fw_')
    bw_hiddens, bw_hiddens_ori = encoder(
        x_emb_r,
        para_name='bw_')
    embedding = layers.concat(input=[fw_hiddens, bw_hiddens], axis=1)
    # add dropout on finetune
    embedding = dropout(embedding)
    a = layers.create_parameter(
        [1], dtype="float32", name="gamma")
    embedding = embedding * a
    return embedding
