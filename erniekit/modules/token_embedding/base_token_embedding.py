# -*- coding: utf-8 -*
"""
:py:class:`BaseTokenEmbedding` is an abstract class for get token embedding
"""


class BaseTokenEmbedding(object):
    """BaseTokenEmbedding
    """
    def __init__(self, emb_params):
        self.name = "token_emb"
        self.emb_params = emb_params

    def build(self):
        """初始化需要的参数
        :return:
        """
        raise NotImplementedError

    def get_token_embedding(self, tokens_dict):
        """
        :param tokens_dict: dict形式，存储转换embedding中需要的原始id及一些参数
        :return: dict形式，词级别和句子级别的embedding
        """
        raise NotImplementedError

    def get_output_dim(self):
        """
        :return:
        """
        raise NotImplementedError
