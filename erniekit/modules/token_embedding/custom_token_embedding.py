# -*- coding: utf-8 -*
"""
:py:class:`CustomTokenEmbedding`
"""
import sys
import logging
import os
import numpy as np
import time, datetime

from paddle import fluid
from ...utils import log
from ...common.rule import InstanceName
from .base_token_embedding import BaseTokenEmbedding
from ...common.register import RegisterSet


@RegisterSet.embedding.register
class CustomTokenEmbedding(BaseTokenEmbedding):
    """CustomTokenEmbedding: 使用paddle.fluid 的api实现的embedding，加载外部词向量文件，训练过程中不断finetune
    """
    def __init__(self, emb_params):
        BaseTokenEmbedding.__init__(self, emb_params)
        self.params_name = None
        self.emb_dim = self.emb_params["emb_dim"]
        self.vocab_size = self.emb_params["vocab_size"]
        self.vec_path = self.emb_params["vec_path"]
        self.vocab_path = self.emb_params["vocab_path"]

    def build(self, vec_name):
        """
        添加一些自顶一个初始化信息，如参数名称
        :return:
        """
        word2id = {}
        id2word = []
        for line in open(self.vocab_path, "r"):
            word, idx = line.rstrip().split("\t")
            word2id[word] = int(idx)
            id2word.append(word)
        logging.info("the size of the vocab is %d" % self.vocab_size)
        logging.info("loading word2vec from %s" % self.vec_path)
        logging.info("please wait for a minute.")
        start = time.time()
        vecs = []
        word2vec= {}
        with open(self.vec_path, "r") as f:
            f.readline()
            for line in f:
                info = line.strip("\n").split(" ")
                word = info[0]
                if word not in word2id:
                    continue
                vector = info[1:]
                if len(vector) != self.emb_dim:
                    logging.info(len(vector))
                assert(len(vector) == self.emb_dim)
                word2vec[word] = np.asarray(vector, dtype='float32')
        
        for word in id2word:
            if word in word2vec:
                vecs.append(word2vec[word])
            else:
                vecs.append(np.random.uniform(-0.05, 0.05, size=[self.emb_dim]).astype(np.float32))
        vecs = np.stack(vecs) 
        end = time.time()
        logging.info("Spent %s on loading word2vec." % str(datetime.timedelta(seconds = end - start)))
        
        np.save(os.path.join('./data/custom_embedding/', vec_name), vecs)

    def get_token_embedding(self, tokens_dict):
        """
        :param tokens_dict:
        :return:
        """
        emb_dict = {}
        tokens = tokens_dict[InstanceName.SRC_IDS]
        tokens_length = tokens_dict[InstanceName.SEQ_LENS]
        unpad_data = fluid.layers.sequence_unpad(tokens, length=tokens_length)
        vec_name, vec_extension_name = os.path.splitext(os.path.split(self.vec_path)[1])
        
        if not os.path.exists(os.path.join('./data/custom_embedding/', vec_name + '.npy')):
            CustomTokenEmbedding.build(self, vec_name)
        
        weight_data = np.load(os.path.join('./data/custom_embedding/', vec_name + '.npy'))
        w_param_attrs = fluid.ParamAttr(
                name=self.name,
                learning_rate=1,
                initializer=fluid.initializer.NumpyArrayInitializer(weight_data),
                trainable=True)
        emb = fluid.layers.embedding(input=unpad_data, 
                size=[self.vocab_size, self.emb_dim],
                is_sparse=False,
                param_attr=w_param_attrs,
                dtype='float32')
        emb_dict = {
            InstanceName.SEQUENCE_EMB: emb,
            InstanceName.POOLED_EMB: None
        }
        return emb_dict

    def get_output_dim(self):
        """
        :return:
        """
        return self.emb_dim
