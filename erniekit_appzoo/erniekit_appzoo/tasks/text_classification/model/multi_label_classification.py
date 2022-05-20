# -*- coding: utf-8 -*
"""
基于ernie进行finetune的分类网络
"""
import collections
import logging
import re
import paddle
from paddle import nn
from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
import numpy as np
from sklearn.metrics import precision_score, recall_score
from erniekit.model.model import BaseModel
from erniekit.modules.ernie import ErnieModel
from erniekit.modules.ernie_config import ErnieConfig
from erniekit.modules.ernie_lr import LinearWarmupDecay


@RegisterSet.models.register
class MultiLabelClassification(BaseModel):
    """MultiLabelClassification
    """

    def __init__(self, model_params):
        """
        """
        BaseModel.__init__(self, model_params)
        # 解析config配置
        self.num_labels = self.model_params.get('num_labels', 2)
        # self.hid_dim = 768

    def structure(self):
        """网络结构组织
        :return:
        """
        emb_params = self.model_params.get("embedding")
        config_path = emb_params.get("config_path")
        self.cfg_dict = ErnieConfig(config_path)
        self.hid_dim = self.cfg_dict['hidden_size']
        self.ernie_model = ErnieModel(self.cfg_dict, name='')
        initializer = nn.initializer.TruncatedNormal(std=0.02)
        self.dropout = nn.Dropout(p=0.1)
        self.fc_layer = nn.Linear(in_features=self.hid_dim, out_features=self.num_labels,
                                  weight_attr=paddle.ParamAttr(name='cls.w_0', initializer=initializer),
                                  bias_attr='cls.b_0')
        self.prediction = nn.Sigmoid()

    def forward(self, fields_dict, phase):
        """ 前向计算
        :param fields_dict:
        :param phase:
        :return:
        """
        fields_dict = self.fields_process(fields_dict, phase)
        instance_text_a = fields_dict["text_a"]
        record_id_text_a = instance_text_a[InstanceName.RECORD_ID]
        text_a_src = record_id_text_a[InstanceName.SRC_IDS]
        text_a_sent = record_id_text_a[InstanceName.SENTENCE_IDS]
        text_a_mask = record_id_text_a[InstanceName.MASK_IDS]
        text_a_task = record_id_text_a[InstanceName.TASK_IDS]

        cls_embedding, tokens_embedding = self.ernie_model(src_ids=text_a_src, sent_ids=text_a_sent,
                                                           task_ids=text_a_task)
        cls_embedding = self.dropout(cls_embedding)
        fc_output = self.fc_layer(cls_embedding)
        probs = self.prediction(fc_output)
        # probs = nn.functional.softmax(prediction)

        if phase == InstanceName.TRAINING or phase == InstanceName.EVALUATE or phase == InstanceName.TEST:
            "train, evaluate, test"
            instance_label = fields_dict["label"]
            record_id_label = instance_label[InstanceName.RECORD_ID]
            label = record_id_label[InstanceName.SRC_IDS]
            # cost = self.loss(probs, label)
            cost = nn.functional.binary_cross_entropy_with_logits(fc_output, label, weight=None, reduction='mean')
            # tips：训练模式下，一定要返回loss
            forward_return_dict = {
                InstanceName.PREDICT_RESULT: probs,
                InstanceName.LABEL: label,
                InstanceName.LOSS: cost
            }

            return forward_return_dict

        elif phase == InstanceName.INFERENCE:
            "infer data with dynamic graph"
            forward_return_dict = {
                InstanceName.PREDICT_RESULT: probs
            }
            return forward_return_dict

        elif phase == InstanceName.SAVE_INFERENCE:
            "save inference model with jit"
            target_predict_list = [probs]
            target_feed_list = [text_a_src, text_a_sent]
            # 以json的形式存入模型的meta文件中，在离线预测的时候用，field_name#field_tensor_name
            target_feed_name_list = ["text_a#src_ids", "text_a#sent_ids"]
            if self.cfg_dict.get('use_task_id', False):
                target_feed_list.append(text_a_task)
                target_feed_name_list.append("text_a#task_ids")
            forward_return_dict = {
                InstanceName.TARGET_FEED: target_feed_list,
                InstanceName.TARGET_PREDICTS: target_predict_list,
                InstanceName.TARGET_FEED_NAMES: target_feed_name_list
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

    def get_metrics(self, forward_return_dict, meta_info, phase):
        """模型效果评估
        :param forward_return_dict: 前向计算得出的结果
        :param meta_info: 常用的meta信息，如step, used_time, gpu_id等
        :param phase: 当前调用的阶段，包含训练和评估
        :return:
        """
        predictions = forward_return_dict[InstanceName.PREDICT_RESULT]
        labels = forward_return_dict[InstanceName.LABEL]
        predictions = np.concatenate(predictions, axis=0).astype(np.float32)
        scores = np.where(predictions > 0.5, 1.0, 0.0)

        if self.is_dygraph:
            if isinstance(labels, list):
                labels = [item.numpy() for item in labels]
            else:
                labels = labels.numpy()

        if phase == InstanceName.EVALUATE or phase == InstanceName.TEST:
            labels = np.concatenate(labels, axis=0).astype(np.float32)
        scores = np.reshape(scores, labels.shape)

        macro_prec = precision_score(labels, scores, average="macro")
        macro_recall = recall_score(labels, scores, average="macro")
        if macro_prec == 0 and macro_recall == 0:
            macro_f1 = 0
        else:
            macro_f1 = 2 * macro_prec * macro_recall / (macro_prec + macro_recall)

        macro_prec = round(macro_prec, 4)
        macro_recall = round(macro_recall, 4)
        macro_f1 = round(macro_f1, 4)

        if phase == InstanceName.TRAINING:
            step = meta_info[InstanceName.STEP]
            time_cost = meta_info[InstanceName.TIME_COST]
            logging.debug("phase = {0} macro-f1 = {1} macro-prec = {2} macro-recall = {3} step = {4} time_cost = {5}" \
                          .format(phase, macro_f1, macro_prec, macro_recall, step, time_cost))
        if phase == InstanceName.EVALUATE or phase == InstanceName.TEST:
            time_cost = meta_info[InstanceName.TIME_COST]
            logging.debug("phase = {0} macro-f1 = {1} macro-prec = {2} macro-recall = {3} time_cost = {4}".format(
                phase, macro_f1, macro_prec, macro_recall, time_cost))

        metrics_return_dict = collections.OrderedDict()
        metrics_return_dict["macro-f1"] = macro_f1
        metrics_return_dict["macro-prec"] = macro_prec
        metrics_return_dict["macro-recall"] = macro_recall
        return metrics_return_dict

    def fields_process(self, fields_dict, phase):
        """对fields中序列化好的id按需做二次处理
        :return: 处理好的fields
        """
        return fields_dict
