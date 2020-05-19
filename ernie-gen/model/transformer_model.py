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
"""Transformer encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import numpy as np


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + fluid.layers.tanh((np.sqrt(2.0 / np.pi) * (x + 0.044715 * fluid.layers.pow(x, 3.0)))))
  return x * cdf


def multi_head_attention(queries,
                         keys,
                         values,
                         attn_bias,
                         d_key,
                         d_value,
                         d_model,
                         n_head=1,
                         dropout_rate=0.,
                         cache=None,
                         gather_idx=None,
                         remove_query=True,
                         param_initializer=None,
                         name='multi_head_att'):
    """
    Multi-Head Attention. Note that attn_bias is added to the logit before
    computing softmax activiation to mask certain selected positions so that
    they will not considered in attention weights.
    """
    keys = queries if keys is None else keys
    values = keys if values is None else values
    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs: quries, keys and values should all be 3-D tensors.")

    def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
        """
        Add linear projection to queries, keys, and values.
        """
        q = layers.fc(input=queries,
                      size=d_key * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_query_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_query_fc.b_0')
        k = layers.fc(input=keys,
                      size=d_key * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_key_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_key_fc.b_0')
        v = layers.fc(input=values,
                      size=d_value * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_value_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_value_fc.b_0')
        return q, k, v

    def __split_heads(x, n_head):
        """
        Reshape the last dimension of inpunt tensor x so that it becomes two
        dimensions and then transpose. Specifically, input a tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] then output a tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        hidden_size = x.shape[-1]
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped = layers.reshape(
            x=x, shape=[0, 0, n_head, hidden_size // n_head], inplace=True)

        # permuate the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """
        Transpose and then reshape the last two dimensions of inpunt tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=True)

    def scaled_dot_product_attention(q, k, v, attn_bias, d_key, dropout_rate):
        """
        Scaled Dot-Product Attention
        """
        scaled_q = layers.scale(x=q, scale=d_key ** -0.5)
        product = layers.matmul(x=scaled_q, y=k, transpose_y=True)
        if attn_bias:
            product += attn_bias
        weights = layers.softmax(product)
        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=False)
        out = layers.matmul(weights, v)
        return out

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)

    if cache is not None:  # use cache and concat time steps
        # Since the inplace reshape in __split_heads changes the shape of k and
        # v, which is the cache input for next time step, reshape the cache
        # input from the previous time step first.
        cache_k, cache_v = cache["k"], cache["v"]
        select_k = layers.gather(cache_k, index=gather_idx)
        select_v = layers.gather(cache_v, index=gather_idx)
        select_k = layers.reshape(select_k, shape=[0, 0, d_model])
        select_v = layers.reshape(select_v, shape=[0, 0, d_model])
        if remove_query:
            tmp_k = layers.concat([select_k, k[:, :1]], axis=1)
            tmp_v = layers.concat([select_v, v[:, :1]], axis=1)
            layers.assign(tmp_k, cache["k"])
            layers.assign(tmp_v, cache["v"])
            k = layers.concat([select_k, k], axis=1)
            v = layers.concat([select_v, v], axis=1)
        else:
            k = layers.concat([select_k, k], axis=1)
            v = layers.concat([select_v, v], axis=1)
            layers.assign(k, cache["k"])
            layers.assign(v, cache["v"])


    q = __split_heads(q, n_head)
    k = __split_heads(k, n_head)
    v = __split_heads(v, n_head)

    ctx_multiheads = scaled_dot_product_attention(q, k, v, attn_bias, d_key,
                                                  dropout_rate)

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    proj_out = layers.fc(input=out,
                         size=d_model,
                         num_flatten_dims=2,
                         param_attr=fluid.ParamAttr(
                             name=name + '_output_fc.w_0',
                             initializer=param_initializer),
                         bias_attr=name + '_output_fc.b_0')
    return proj_out


def positionwise_feed_forward(x,
                              d_inner_hid,
                              d_hid,
                              dropout_rate,
                              hidden_act,
                              param_initializer=None,
                              name='ffn'):
    """
    Position-wise Feed-Forward Networks.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    if hidden_act == 'gelu' or hidden_act == 'gelu.precise':
        _hidden_act = 'gelu'
    elif hidden_act == 'gelu.approximate':
        _hidden_act = None
    else:
        _hidden_act =  hidden_act
    hidden = layers.fc(input=x,
                       size=d_inner_hid,
                       num_flatten_dims=2,
                       act=_hidden_act,
                       param_attr=fluid.ParamAttr(
                           name=name + '_fc_0.w_0',
                           initializer=param_initializer),
                       bias_attr=name + '_fc_0.b_0')
    if hidden_act == 'gelu.approximate':
        hidden = gelu(hidden)
    if dropout_rate:
        hidden = layers.dropout(
            hidden,
            dropout_prob=dropout_rate,
            dropout_implementation="upscale_in_train",
            is_test=False)
    out = layers.fc(input=hidden,
                    size=d_hid,
                    num_flatten_dims=2,
                    param_attr=fluid.ParamAttr(
                        name=name + '_fc_1.w_0', initializer=param_initializer),
                    bias_attr=name + '_fc_1.b_0')
    return out


def pre_post_process_layer(prev_out,
                           out,
                           process_cmd,
                           dropout_rate=0.,
                           epsilon=1e-5,
                           name=''):
    """
    Add residual connection, layer normalization and droput to the out tensor
    optionally according to the value of process_cmd.
    This will be used before or after multi-head attention and position-wise
    feed-forward networks.
    """
    for cmd in process_cmd:
        if cmd == "a":  # add residual connection
            out = out + prev_out if prev_out else out
        elif cmd == "n":  # add layer normalization
            out_dtype = out.dtype
            if out_dtype == fluid.core.VarDesc.VarType.FP16:
                out = layers.cast(x=out, dtype="float32")
            out = layers.layer_norm(
                out,
                begin_norm_axis=len(out.shape) - 1,
                param_attr=fluid.ParamAttr(
                    name=name + '_layer_norm_scale',
                    initializer=fluid.initializer.Constant(1.)),
                bias_attr=fluid.ParamAttr(
                    name=name + '_layer_norm_bias',
                    initializer=fluid.initializer.Constant(0.)),
                epsilon=epsilon)
            if out_dtype == fluid.core.VarDesc.VarType.FP16:
                out = layers.cast(x=out, dtype="float16")
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = layers.dropout(
                    out,
                    dropout_prob=dropout_rate,
                    dropout_implementation="upscale_in_train",
                    is_test=False)
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


def encoder_layer(query_input,
                  key_input,
                  attn_bias,
                  n_head,
                  d_key,
                  d_value,
                  d_model,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  hidden_act,
                  preprocess_cmd="n",
                  postprocess_cmd="da",
                  param_initializer=None,
                  name='',
                  epsilon=1e-5,
                  cache=None,
                  gather_idx=None,
                  remove_query=True):
    """The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the post_process_layer to add residual connection, layer normalization
    and droput.
    """
    key_input = pre_process_layer(
            key_input,
            preprocess_cmd,
            prepostprocess_dropout,
            epsilon=epsilon,
            name=name + '_pre_att') if key_input else None
    value_input = key_input if key_input else None

    attn_output = multi_head_attention(
        pre_process_layer(
            query_input,
            preprocess_cmd,
            prepostprocess_dropout,
            epsilon=epsilon,
            name=name + '_pre_att'),
        key_input,
        value_input,
        attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        param_initializer=param_initializer,
        name=name + '_multi_head_att',
        cache=cache,
        gather_idx=gather_idx,
        remove_query=remove_query)
    attn_output = post_process_layer(
        query_input,
        attn_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name=name + '_post_att',
        epsilon=epsilon)

    ffd_output = positionwise_feed_forward(
        pre_process_layer(
            attn_output,
            preprocess_cmd,
            prepostprocess_dropout,
            epsilon=epsilon,
            name=name + '_pre_ffn'),
        d_inner_hid,
        d_model,
        relu_dropout,
        hidden_act,
        param_initializer=param_initializer,
        name=name + '_ffn')

    return post_process_layer(
        attn_output,
        ffd_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name=name + '_post_ffn',
        epsilon=epsilon)


def infilling_decode_transformer(enc_input,
            attn_bias,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd="n",
            postprocess_cmd="da",
            param_initializer=None,
            name='',
            epsilon=1e-5,
            n_layer_per_block=1,
            param_share="normal",
            caches=None,
            gather_idx=None,
            remove_query=True):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """
    names = []
    if param_share == "inner_share":
        for _ in range(n_layer // n_layer_per_block):
            for i in range(n_layer_per_block):
                names.append(name + '_layer_' + str(i))
    else:
        for i in range(n_layer // n_layer_per_block):
            for _ in range(n_layer_per_block):
                names.append(name + '_layer_' + str(i))

    for i in range(n_layer):
        enc_output = encoder_layer(
            enc_input,
            None,
            attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            param_initializer=param_initializer,
            epsilon=epsilon,
            name=names[i],
            cache=caches[i] if caches is not None else None,
            gather_idx=gather_idx,
            remove_query=remove_query)
        enc_input = enc_output
    enc_output = pre_process_layer(
        enc_output,
        preprocess_cmd,
        prepostprocess_dropout,
        name="post_encoder",
        epsilon=epsilon)

    return enc_output

def infilling_transformer(enc_input_kv,
            enc_input_query,
            attn_bias_kv,
            attn_bias_query,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd="n",
            postprocess_cmd="da",
            param_initializer=None,
            name='',
            epsilon=1e-5,
            n_layer_per_block=1,
            param_share="normal"):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """
    names = []
    if param_share == "inner_share":
        for _ in range(n_layer // n_layer_per_block):
            for i in range(n_layer_per_block):
                names.append(name + '_layer_' + str(i))
    else:
        for i in range(n_layer // n_layer_per_block):
            for _ in range(n_layer_per_block):
                names.append(name + '_layer_' + str(i))

    for i in range(n_layer):
        enc_output_kv = encoder_layer(
            enc_input_kv,
            None,
            attn_bias_kv,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            param_initializer=param_initializer,
            epsilon=epsilon,
            name=names[i])

        k_v = layers.concat([enc_input_kv, enc_input_query],axis=1)
        enc_output_query = encoder_layer(
            enc_input_query,
            k_v,
            attn_bias_query,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            param_initializer=param_initializer,
            epsilon=epsilon,
            name=names[i])

        enc_input_kv = enc_output_kv
        enc_input_query = enc_output_query

    enc_output_query = pre_process_layer(
        enc_output_query,
        preprocess_cmd,
        prepostprocess_dropout,
        name="post_encoder",
        epsilon=epsilon)

    return enc_output_query
