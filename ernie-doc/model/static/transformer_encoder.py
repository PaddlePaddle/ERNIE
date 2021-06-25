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
"""Transformer encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import paddle.fluid as fluid
import paddle.fluid.layers as layers

def _cache_mem(curr_out, prev_mem, mem_len=None, use_vars=False):
    """generate new memories for next step"""
    if mem_len is None or mem_len == 0:
        return None
    else:
        if prev_mem is None:
            new_mem = curr_out[:, -mem_len:, :]
        else:
            new_mem = layers.concat([prev_mem, curr_out], 1)[:, -mem_len:, :]
        new_mem.stop_gradient = True
        if use_vars:
            layers.assign(new_mem, prev_mem)
    return new_mem


def multi_head_attention(queries,
                         keys,
                         values,
                         rel_pos,
                         rel_task,
                         memory,
                         attn_bias,
                         d_key,
                         d_value,
                         d_model,
                         n_head=1,
                         r_w_bias=None,
                         r_r_bias=None,
                         r_t_bias=None,
                         dropout_rate=0.,
                         cache=None,
                         param_initializer=None,
                         rel_pos_params_sharing=False,
                         name='multi_head_att'):
    """
    Multi-Head Attention. Note that attn_bias is added to the logit before
    computing softmax activiation to mask certain selected positions so that
    they will not considered in attention weights.
    """
    if memory is not None and len(memory.shape) > 1:
        cat = fluid.layers.concat([memory, queries], 1)
    else:
        cat = queries
    keys, values = cat, cat

    if not (len(queries.shape) == len(keys.shape) == len(values.shape) \
            == len(rel_pos.shape) == len(rel_task.shape)== 3):
        raise ValueError(
            "Inputs: quries, keys, values, rel_pos and rel_task should all be 3-D tensors.")
    
    if rel_pos_params_sharing:
        assert (r_w_bias and r_r_bias and r_t_bias) is not None, \
                    "the rel pos bias can not be None when sharing the relative position params"
    else:
        r_w_bias, r_r_bias, r_t_bias = \
                list(map(lambda x: layers.create_parameter(
                                shape=[n_head * d_key], 
                                dtype="float32", 
                                name=name + "_" + x,
                                default_initializer=param_initializer), 
                ["r_w_bias", "r_r_bias", "r_t_bias"]))
    
    def __compute_qkv(queries, keys, values, rel_pos, rel_task, n_head, d_key, d_value):
        """
        Add linear projection to queries, keys, values, postions and tasks.
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
        r = layers.fc(input=rel_pos,
                      size=d_key * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_pos_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_pos_fc.b_0')
        t = layers.fc(input=rel_task,
                      size=d_key * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_task_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_task_fc.b_0')
        return q, k, v, r, t

    def __split_heads(x, n_head, add_bias=None):
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
        
        if add_bias:
            reshaped = reshaped + add_bias

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

    def __rel_shift(x, klen):
        """return relative shift"""
        x_shape = x.shape
        INT_MAX=10000000
        x = layers.reshape(x, [x_shape[0], x_shape[1], x_shape[3], x_shape[2]])
        x = layers.slice(x, [0, 1, 2, 3], [0, 0, 1, 0], [INT_MAX, INT_MAX, INT_MAX, INT_MAX])
        x = layers.reshape(x, [x_shape[0], x_shape[1], x_shape[2], x_shape[3] - 1])
        x = layers.slice(x, [0, 1, 2, 3], [0, 0, 0, 0], [INT_MAX, INT_MAX, INT_MAX, klen])
        return x

    def __scaled_dot_product_attention(q, k, v, r, t, attn_bias, d_key, dropout_rate):
        """
        Scaled Dot-Product Attention
        """
        q_w, q_r, q_t = list(map(lambda x: layers.scale(x=x, scale=d_key ** -0.5), q))
        score_w = layers.matmul(x=q_w, y=k, transpose_y=True)
        score_r = layers.matmul(x=q_r, y=r, transpose_y=True)
        score_r = __rel_shift(score_r, k.shape[2])
        score_t = layers.matmul(x=q_t, y=t, transpose_y=True)
        score = score_w + score_r + score_t
        if attn_bias is not None:
            score += attn_bias
        weights = layers.softmax(score, use_cudnn=True)
        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=False)
        out = layers.matmul(weights, v)
        return out    

    q, k, v, r, t = __compute_qkv(queries, keys, values, rel_pos, rel_task, n_head, d_key, d_value)
    
    if cache is not None:  # use cache and concat time steps
        # Since the inplace reshape in __split_heads changes the shape of k and
        # v, which is the cache input for next time step, reshape the cache
        # input from the previous time step first.
        k = cache["k"] = layers.concat(
            [layers.reshape(
                cache["k"], shape=[0, 0, d_model]), k], axis=1)
        v = cache["v"] = layers.concat(
            [layers.reshape(
                cache["v"], shape=[0, 0, d_model]), v], axis=1)
     
    q_w, q_r, q_t = list(map(lambda x: layers.elementwise_add(q, x, 2), [r_w_bias, r_r_bias, r_t_bias]))
    q_w, q_r, q_t = list(map(lambda x: __split_heads(x, n_head), [q_w, q_r, q_t]))
    k, v, r, t = list(map(lambda x: __split_heads(x, n_head), [k, v, r, t]))
    
    ctx_multiheads = __scaled_dot_product_attention([q_w, q_r, q_t], \
                                    k, v, r, t, attn_bias, d_key, dropout_rate)

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
    hidden = layers.fc(input=x,
                       size=d_inner_hid,
                       num_flatten_dims=2,
                       act=hidden_act,
                       param_attr=fluid.ParamAttr(
                           name=name + '_fc_0.w_0',
                           initializer=param_initializer),
                       bias_attr=name + '_fc_0.b_0')
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


def encoder_layer(enc_input,
                  rel_pos,
                  rel_task,
                  memory,
                  attn_bias,
                  n_head,
                  d_key,
                  d_value,
                  d_model,
                  d_inner_hid,
                  r_w_bias,
                  r_r_bias,
                  r_t_bias,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  hidden_act,
                  epsilon=1e-5,
                  preprocess_cmd="n",
                  postprocess_cmd="da",
                  param_initializer=None,
                  rel_pos_params_sharing=False,
                  name=''):
    """The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the post_process_layer to add residual connection, layer normalization
    and droput.
    """
    attn_output = multi_head_attention(
        pre_process_layer(
            enc_input,
            preprocess_cmd,
            prepostprocess_dropout,
            epsilon=epsilon,
            name=name + '_pre_att'),
        None,
        None,
        rel_pos,
        rel_task,
        memory,
        attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        r_w_bias,
        r_r_bias,
        r_t_bias,
        attention_dropout,
        param_initializer=param_initializer,
        rel_pos_params_sharing=rel_pos_params_sharing,
        name=name + '_multi_head_att')
    attn_output = post_process_layer(
        enc_input,
        attn_output,
        postprocess_cmd,
        prepostprocess_dropout,
        epsilon=epsilon,
        name=name + '_post_att')
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
        epsilon=epsilon,
        name=name + '_post_ffn'), ffd_output


def encoder(enc_input,
            memories,
            rel_pos,
            rel_task,
            attn_bias,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            memory_len,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            epsilon=1e-5,
            preprocess_cmd="n",
            postprocess_cmd="da",
            param_initializer=None,
            rel_pos_params_sharing=False,
            name='',
            use_vars=False):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """
    r_w_bias, r_r_bias, r_t_bias = None, None, None
    if rel_pos_params_sharing:
        r_w_bias, r_r_bias, r_t_bias = \
                list(map(lambda x: layers.create_parameter(
                                shape=[n_head * d_key], 
                                dtype="float32", 
                                name=name + "_" + x,
                                default_initializer=param_initializer), 
                ["r_w_bias", "r_r_bias", "r_t_bias"]))
     
    checkpoints = []
    _new_mems = []
    for i in range(n_layer):
        enc_input, cp = encoder_layer(
            enc_input,
            rel_pos,
            rel_task,
            memories[i],
            attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            r_w_bias,
            r_r_bias,
            r_t_bias,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            epsilon,
            preprocess_cmd,
            postprocess_cmd,
            param_initializer=param_initializer,
            rel_pos_params_sharing=rel_pos_params_sharing,
            name=name + '_layer_' + str(i))
        checkpoints.append(cp.name)
        new_mem = _cache_mem(enc_input, memories[i], memory_len, use_vars=use_vars)
        if not use_vars:
            _new_mems.append(new_mem)
    enc_output = pre_process_layer(
        enc_input, 
        preprocess_cmd, 
        prepostprocess_dropout, 
        epsilon, 
        name="post_encoder")
    return enc_output, _new_mems, checkpoints[:-1]
