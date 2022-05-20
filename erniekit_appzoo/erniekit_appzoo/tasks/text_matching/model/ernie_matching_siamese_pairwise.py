# -*- coding: utf-8 -*
"""
ErnieMatchingFcPointwise
"""
import paddle
import re
from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
from erniekit.modules.ernie import ErnieModel
from erniekit.modules.ernie_config import ErnieConfig
from model.base_matching import BaseMatching
from erniekit.modules.ernie_lr import LinearWarmupDecay



@RegisterSet.models.register
class ErnieMatchingSiamesePairwise(BaseMatching):
    """ErnieMatchingFcPointwise:使用TextFieldReader组装数据,只返回src_id和length，用户可以使用src_id自己生成embedding
    """

    def __init__(self, model_params):
        BaseMatching.__init__(self, model_params)

    def structure(self):
        """网络结构组织
        :return:
        """

        emb_params = self.model_params.get("embedding")
        config_path = emb_params.get("config_path")
        self.ernie_config = ErnieConfig(config_path)

        self.ernie_model = ErnieModel(self.ernie_config, name='')
        self.dropout = paddle.nn.Dropout(p=0.1, mode="upscale_in_train")
        self.cos_sim = paddle.nn.CosineSimilarity(axis=1)
        self.loss = paddle.nn.CrossEntropyLoss(use_softmax=False)

    def forward(self, fields_dict, phase):
        """前向计算组网部分，必须由子类实现
        :return: loss , fetch_list
        """

        if phase == InstanceName.TRAINING:
            fields_name = ["text_a", "text_b", "text_c"]
        else:
            fields_name = ["text_a", "text_b"]

        emb_dict = {}
        target_feed_list = []
        ernie_feed_list = []
        for name in fields_name:
            instance_text = fields_dict[name]
            record_id_text = instance_text[InstanceName.RECORD_ID]
            text_src_ids = record_id_text[InstanceName.SRC_IDS]
            text_sent_ids = record_id_text[InstanceName.SENTENCE_IDS]
            text_task_ids = record_id_text[InstanceName.TASK_IDS]
            cls_embedding, tokens_embedding = self.ernie_model(src_ids=text_src_ids, sent_ids=text_sent_ids,
                                                               task_ids=text_task_ids)
            if phase == InstanceName.SAVE_INFERENCE:
                target_feed_list.append(text_src_ids)
                target_feed_list.append(text_sent_ids)
                if self.ernie_config.get('use_task_id', False):
                    target_feed_list.append(text_task_ids)

            emb_dict[name] = cls_embedding

        emb_text_q = emb_dict["text_a"]
        emb_text_pt = emb_dict["text_b"]

        q_embeddings = self.dropout(emb_text_q)
        pt_embeddings = self.dropout(emb_text_pt)

        query_pos_title_score = paddle.reshape(self.cos_sim(q_embeddings, pt_embeddings), [-1, 1])

        if phase == InstanceName.TRAINING:
            emb_text_nt = emb_dict["text_c"]
            neg_t_embeddings = self.dropout(emb_text_nt)
            query_neg_title_score = paddle.reshape(self.cos_sim(q_embeddings, neg_t_embeddings), [-1, 1])
            labels = paddle.full(shape=[1], fill_value=1.0, dtype='float32')
            avg_cost = paddle.nn.functional.margin_ranking_loss(query_pos_title_score,
                                                                query_neg_title_score, labels, margin=0.2,
                                                                reduction='mean')

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
            target_feed_list.extend(ernie_feed_list)
            if self.ernie_config.get('use_task_id', False):
                target_feed_name_list = ['text_a#src_ids', 'text_a#sent_ids', 'text_a#task_ids', 'text_b#src_ids',
                                         'text_b#sent_ids', 'text_b#task_ids']
            else:
                target_feed_name_list = ['text_a#src_ids', 'text_a#sent_ids', 'text_b#src_ids', 'text_b#sent_ids']
            target_predict_list = [predictions]
            forward_return_dict = {
                InstanceName.TARGET_FEED: target_feed_list,
                InstanceName.TARGET_FEED_NAMES: target_feed_name_list,
                InstanceName.TARGET_PREDICTS: target_predict_list
            }
            return forward_return_dict

    def set_optimizer(self):
        """
        :return: optimizer
        """
        # 学习率和权重的衰减设置在optimizer中，loss的缩放设置在amp中（各个trainer中进行设置）。
        # TODO:需要考虑学习率衰减、权重衰减设置、 loss的缩放设置
        opt_param = self.model_params.get('optimization', None)
        self.lr = opt_param.get("learning_rate", 2e-5)
        weight_decay = opt_param.get("weight_decay", 0.01)
        use_lr_decay = opt_param.get("use_lr_decay", False)
        epsilon = opt_param.get("epsilon", 1e-6)
        g_clip = paddle.nn.ClipGradByGlobalNorm(1.0)
        param_name_to_exclue_from_weight_decay = re.compile(r'.*layer_norm_scale|.*layer_norm_bias|.*b_0')

        parameters = None
        if self.is_dygraph:
            parameters = self.parameters()

        if use_lr_decay:
            max_train_steps = opt_param.get("max_train_steps", 0)
            warmup_steps = opt_param.get("warmup_steps", 0)
            self.lr_scheduler = LinearWarmupDecay(base_lr=self.lr, end_lr=0.0, warmup_steps=warmup_steps,
                                                  decay_steps=max_train_steps, num_train_steps=max_train_steps)
            self.optimizer = paddle.optimizer.AdamW(learning_rate=self.lr_scheduler,
                                                    parameters=parameters,
                                                    weight_decay=weight_decay,
                                                    apply_decay_param_fun=lambda
                                                        n: not param_name_to_exclue_from_weight_decay.match(n),
                                                    epsilon=epsilon,
                                                    grad_clip=g_clip)
        else:
            self.optimizer = paddle.optimizer.AdamW(self.lr,
                                                    parameters=parameters,
                                                    weight_decay=weight_decay,
                                                    apply_decay_param_fun=lambda
                                                        n: not param_name_to_exclue_from_weight_decay.match(n),
                                                    epsilon=epsilon,
                                                    grad_clip=g_clip)
        return self.optimizer
