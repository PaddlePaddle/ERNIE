# -*- coding: utf-8 -*
"""
BOW 分类网络
"""
import paddle
from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
from model.base_matching import BaseMatching

@RegisterSet.models.register
class BowMatchingPairwise(BaseMatching):
    """BowClassification
    """
    def __init__(self, model_params):
        """
        """
        BaseMatching.__init__(self, model_params)

    def structure(self):
        """网络结构组织，定义需要用到的成员变量即可
        :return: None
        """
        self.dict_dim = self.model_params.get('vocab_size', 52445)
        self.emb_dim = self.model_params.get('emb_dim', 128)
        self.hid_dim = self.model_params.get('hid_dim', 128)

        self.embedding = paddle.nn.Embedding(num_embeddings=self.dict_dim, embedding_dim=self.emb_dim)
        self.softsign = paddle.nn.Softsign()
        self.relu = paddle.nn.ReLU()
        self.fc = paddle.nn.Linear(in_features=self.hid_dim, out_features=self.hid_dim)
        self.cos_sim = paddle.nn.CosineSimilarity(axis=1)
        self.loss = paddle.nn.CrossEntropyLoss(use_softmax=False)

    def forward(self, fields_dict, phase):
        """
        :param fields_dict: 动态图模式下是tensor格式，静态图模式下是python数组
        :param phase:
        :return:
        """

        if phase == InstanceName.TRAINING:
            fields_name = ["text_a", "text_b", "text_c"]
        else:
            fields_name = ["text_a", "text_b"]

        emb_dict = {}
        target_feed_name_list = []
        target_feed_list = []
        for name in fields_name:
            instance_text = fields_dict[name]
            record_id_text = instance_text[InstanceName.RECORD_ID]
            text_src_ids = record_id_text[InstanceName.SRC_IDS]
            text_lens = record_id_text[InstanceName.SEQ_LENS]

            if phase != InstanceName.TRAINING:
                target_feed_list.append(text_src_ids)

            emb = self.embedding(text_src_ids)
            emb_pool = paddle.sum(emb, axis=1)
            emb_soft = self.softsign(emb_pool)
            emb_relu = self.relu(self.fc(emb_soft)) 
            emb_dict[name] = emb_relu

        q_embeddings = emb_dict["text_a"]
        pt_embeddings = emb_dict["text_b"]
        
        # 维度是batch_size的一维向量，所以reshape为[batch_size, 1]
        query_pos_title_score =  paddle.reshape(self.cos_sim(q_embeddings, pt_embeddings), [-1, 1])

        if phase == InstanceName.TRAINING:
            neg_t_embeddings = emb_dict["text_c"]
            query_neg_title_score =  paddle.reshape(self.cos_sim(q_embeddings, neg_t_embeddings), [-1, 1])
            labels = paddle.full(shape=[1], fill_value=1.0, dtype='float32')
            avg_cost = paddle.nn.functional.margin_ranking_loss(query_pos_title_score, 
                    query_neg_title_score, labels, margin=0.1, reduction='mean')
            """PREDICT_RESULT,LABEL,LOSS 是关键字，必须要赋值并返回"""
            forward_return_dict = {
                "query_pos_title_score": query_pos_title_score,
                "query_neg_title_score": query_neg_title_score,
                InstanceName.LOSS: avg_cost
            }
            return forward_return_dict

        elif phase == InstanceName.EVALUATE or phase == InstanceName.TEST:
            instance_label = fields_dict["label"]
            record_id_label = instance_label[InstanceName.RECORD_ID]
            label = record_id_label[InstanceName.SRC_IDS]
            ones = paddle.full(shape=[1], fill_value=1.0, dtype='float32')

            query_pos_title_score = (query_pos_title_score + 1) / 2
            sub_pos_score = paddle.subtract(ones, query_pos_title_score)
            predictions = paddle.reshape(paddle.concat(x=[sub_pos_score, query_pos_title_score],
                                                                   axis=-1), shape=[-1, 2])
            cost = self.loss(predictions, label)
            avg_cost = paddle.mean(x=cost)
            forward_return_dict = {
                InstanceName.PREDICT_RESULT: predictions,
                InstanceName.LABEL: label,
                InstanceName.LOSS: avg_cost
            }
            return forward_return_dict

        else:
            ones = paddle.full(shape=[1], fill_value=1.0, dtype='float32')
            # 与评估保持一致，让其取值范围变为0-1
            query_pos_title_score = (query_pos_title_score + 1) / 2
            sub_pos_score = paddle.subtract(ones, query_pos_title_score)
            predictions = paddle.reshape(paddle.concat(x=[sub_pos_score, query_pos_title_score],
                axis=-1), shape=[-1, 2])
            target_feed_name_list = ['text_a#src_ids', 'text_b#src_ids']
            target_predict_list = [predictions]
            forward_return_dict = {
                InstanceName.TARGET_FEED: target_feed_list,
                InstanceName.TARGET_FEED_NAMES: target_feed_name_list,
                InstanceName.TARGET_PREDICTS: target_predict_list
            }
            return forward_return_dict


