"""
The function lex_net(args) define the lexical analysis network structure
"""
import sys
import os
import math

import paddle.fluid as fluid
from paddle.fluid.initializer import NormalInitializer
import paddle.fluid.layers as layers
from bilm import elmo_encoder
from bilm import emb
#import bilm
import ipdb

def lex_net(args, word_dict_len, label_dict_len):
    """
    define the lexical analysis network structure
    """
    word_emb_dim = args.word_emb_dim
    grnn_hidden_dim = args.grnn_hidden_dim
    emb_lr = args.emb_learning_rate
    crf_lr = args.crf_learning_rate
    bigru_num = args.bigru_num
    init_bound = 0.1
    IS_SPARSE = True

    def _bigru_layer(input_feature):
        """
        define the bidirectional gru layer
        """
        pre_gru = fluid.layers.fc(
            input=input_feature,
            size=grnn_hidden_dim * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        gru = fluid.layers.dynamic_gru(
            input=pre_gru,
            size=grnn_hidden_dim,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        pre_gru_r = fluid.layers.fc(
            input=input_feature,
            size=grnn_hidden_dim * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        gru_r = fluid.layers.dynamic_gru(
            input=pre_gru_r,
            size=grnn_hidden_dim,
            is_reverse=True,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        bi_merge = fluid.layers.concat(input=[gru, gru_r], axis=1)
        return bi_merge

    def _net_conf(word, target):
        """
        Configure the network
        """
        #add elmo
        #ipdb.set_trace()
        #elmo_embedding = emb(word)
        #layers.Print(word, message='input_seq', summarize=10)
        #drnn = layers.DynamicRNN()
        #with drnn.block():
         #   elmo_embedding = drnn.step_input(elmo_embedding)
          #  elmo_enc= elmo_encoder(elmo_embedding)
           # drnn.output(elmo_enc)
       # elmo_enc = drnn()
        
        word_embedding = fluid.layers.embedding(
        input=word,
        size=[word_dict_len, word_emb_dim],
        dtype='float32',
        is_sparse=IS_SPARSE,
        param_attr=fluid.ParamAttr(
             learning_rate=emb_lr,
             name="word_emb",
             initializer=fluid.initializer.Uniform(
                 low=-init_bound, high=init_bound)))	
        #layers.Print(word, message='word', summarize=-1)
        #layers.Print(word_r, message='word_r', summarize=-1)
        #word_r=fluid.layers.sequence_reverse(word, name=None)
        #layers.Print(word_r, message='word_r_1', summarize=-1)
        elmo_embedding = emb(word)
        #elmo_embedding_r=emb(word_r)
        #layers.Print(elmo_embedding, message='elmo_embedding', summarize=10)
        #layers.Print(word, message='input_seq', summarize=10)
        #drnn = layers.DynamicRNN()
        #with drnn.block():
        #elmo_embed = drnn.step_input(elmo_embedding)
        #layers.Print(elmo_embed, message='elmo_enc', summarize=10)
        #elmo_enc = elmo_encoder(elmo_embedding)
        elmo_enc = elmo_encoder(elmo_embedding, args.elmo_l2_coef)
        #input_feature=layers.concat(input=[elmo_enc, word_embedding], axis=1)
        #input_feature=elmo_enc
         #input_feature=layers.concat#drnn.output(input_feature)
        #input_feature = drnn()
       # input_feature = word_embedding
        #layers.Print(elmo_enc, message='elmo_enc', summarize=10)
        input_feature=layers.concat(input=[elmo_enc, word_embedding], axis=1)
        for i in range(bigru_num):
            bigru_output = _bigru_layer(input_feature)
            input_feature = bigru_output

        emission = fluid.layers.fc(
            size=label_dict_len,
            input=bigru_output,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        crf_cost = fluid.layers.linear_chain_crf(
            input=emission,
            label=target,
            param_attr=fluid.ParamAttr(
                name='crfw',
                learning_rate=crf_lr))
        crf_decode = fluid.layers.crf_decoding(
            input=emission, param_attr=fluid.ParamAttr(name='crfw'))
        avg_cost = fluid.layers.mean(x=crf_cost)
        return avg_cost, crf_decode

    word = fluid.layers.data(
        name='word', shape=[1], dtype='int64', lod_level=1)
    #word_r = fluid.layers.data(
    #    name='word_r', shape=[1], dtype='int64', lod_level=1)
    target = fluid.layers.data(
        name="target", shape=[1], dtype='int64', lod_level=1)

    avg_cost, crf_decode= _net_conf(word, target)

    return avg_cost, crf_decode, word,target
