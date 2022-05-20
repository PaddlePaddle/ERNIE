# -*- coding: utf-8 -*
"""
基于ernie进行finetune的分类网络
"""
import re
import paddle
from paddle import nn
from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
from erniekit.modules.ernie import ErnieModel
from erniekit.modules.ernie_config import ErnieConfig
from erniekit.modules.ernie_lr import LinearWarmupDecay
from model.base_cls import BaseClassification


@RegisterSet.models.register
class ErnieClassification(BaseClassification):
    """ErnieClassification
    """
    def __init__(self, model_params):
        """
        """
        BaseClassification.__init__(self, model_params)
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
        self.fc_prediction = nn.Linear(in_features=self.hid_dim, out_features=self.num_labels,
                                       weight_attr=paddle.ParamAttr(name='cls.w_0', initializer=initializer),
                                       bias_attr='cls.b_0')
        self.loss = paddle.nn.CrossEntropyLoss(use_softmax=False)

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

        cls_embedding, tokens_embedding = self.ernie_model(src_ids=text_a_src, 
                                                           sent_ids=text_a_sent, task_ids=text_a_task)
        cls_embedding = self.dropout(cls_embedding)
        prediction = self.fc_prediction(cls_embedding)
        probs = nn.functional.softmax(prediction)

        if phase == InstanceName.TRAINING or phase == InstanceName.EVALUATE or phase == InstanceName.TEST:
            "train, evaluate, test"
            instance_label = fields_dict["label"]
            record_id_label = instance_label[InstanceName.RECORD_ID]
            label = record_id_label[InstanceName.SRC_IDS]
            cost = self.loss(probs, label)
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
            target_feed_list = [text_a_src]
            # 以json的形式存入模型的meta文件中，在离线预测的时候用，field_name#field_tensor_name
            # target_feed_name_list = ["text_a#src_ids", "text_a#sent_ids"]
            target_feed_name_list = ["text_a#src_ids"]
            if self.cfg_dict.get('use_sent_id', True):
                target_feed_list.append(text_a_sent)
                target_feed_name_list.append("text_a#sent_ids")            
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

    def fields_process(self, fields_dict, phase):
        """对fields中序列化好的id按需做二次处理
        :return: 处理好的fields
        """
        return fields_dict
