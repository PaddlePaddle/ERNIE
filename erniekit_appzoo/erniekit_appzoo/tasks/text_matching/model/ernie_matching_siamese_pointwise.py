# -*- coding: utf-8 -*
"""
ErnieMatchingFcPointwise
"""
import paddle
import numpy as np
import re
from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
from erniekit.modules.ernie import ErnieModel
from erniekit.modules.ernie_config import ErnieConfig
from model.base_matching import BaseMatching
from erniekit.modules.ernie_lr import LinearWarmupDecay
from erniekit.metrics import metrics
import logging
import collections


@RegisterSet.models.register
class ErnieMatchingSiamesePointwise(BaseMatching):
    """ErnieMatchingFcPointwise:使用TextFieldReader组装数据,只返回src_id和length，用户可以使用src_id自己生成embedding
    """

    def __init__(self, model_params):
        BaseMatching.__init__(self, model_params)

    def structure(self):
        """网络结构组织
        :return:
        """

        self.num_labels = self.model_params.get('num_labels', 2)
        emb_params = self.model_params.get("embedding")
        config_path = emb_params.get("config_path")
        self.ernie_config = ErnieConfig(config_path)
        self.hidden_size = self.ernie_config['hidden_size']
        self.ernie_model = ErnieModel(self.ernie_config, name='')
        self.fc_hidden_size = 128
        initializer = paddle.nn.initializer.TruncatedNormal(std=0.02)
        self.concat_fc = paddle.nn.Linear(in_features=self.hidden_size * 2, out_features=self.fc_hidden_size,
                                          weight_attr=paddle.ParamAttr(name='concat_fc.w', initializer=initializer),
                                          bias_attr='concat.b')

        self.output_layer = paddle.nn.Linear(self.fc_hidden_size, self.num_labels)

        self.loss = paddle.nn.CrossEntropyLoss(use_softmax=False)

    def forward(self, fields_dict, phase):
        """前向计算组网部分，必须由子类实现
        :return: loss , fetch_list
        """

        instance_label = fields_dict["label"]
        record_id_label = instance_label[InstanceName.RECORD_ID]
        label = record_id_label[InstanceName.SRC_IDS]

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

        # [batch_size, hidden_size]
        emb_text_a = emb_dict["text_a"]
        emb_text_b = emb_dict["text_b"]
        # [batch_size, hidden_size*2]
        contacted = paddle.concat([emb_text_a, emb_text_b], axis=-1)
        # [batch_size, fc_hidden_size]
        fc_out = paddle.tanh(self.concat_fc(contacted))
        # [fc_hidden_size, num_labels]
        logits = self.output_layer(fc_out)
        predictions = paddle.nn.functional.softmax(logits)

        if phase == InstanceName.SAVE_INFERENCE:
            target_predict_list = [predictions]
            target_feed_list.extend(ernie_feed_list)
            if self.ernie_config.get('use_task_id', False):
                target_feed_name_list = ['text_a#src_ids', 'text_a#sent_ids', 'text_a#task_ids', 'text_b#src_ids',
                                         'text_b#sent_ids', 'text_b#task_ids']
            else:
                target_feed_name_list = ['text_a#src_ids', 'text_a#sent_ids', 'text_b#src_ids', 'text_b#sent_ids']
            forward_return_dict = {
                InstanceName.TARGET_FEED: target_feed_list,
                InstanceName.TARGET_FEED_NAMES: target_feed_name_list,
                InstanceName.TARGET_PREDICTS: target_predict_list
            }
            return forward_return_dict

        cost = self.loss(predictions, label)

        forward_return_dict = {
            InstanceName.PREDICT_RESULT: predictions,
            InstanceName.LABEL: label,
            InstanceName.LOSS: cost
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
        """
        :param forward_return_dict: 前向计算得出的结果
        :param meta_info: 常用的meta信息，如step, used_time, gpu_id等
        :param phase: 当前调用的阶段，包含训练和评估
        :return:
        """
        predictions = forward_return_dict[InstanceName.PREDICT_RESULT]
        label = forward_return_dict[InstanceName.LABEL]
        # paddle_acc = forward_return_dict["acc"]
        if self.is_dygraph:
            if isinstance(predictions, list):
                predictions = [item.numpy() for item in predictions]
            else:
                predictions = predictions.numpy()
            if isinstance(label, list):
                label = [item.numpy() for item in label]
            else:
                label = label.numpy()
        metrics_acc = metrics.Acc()
        acc = metrics_acc.eval([predictions, label])
        metrics_pres = metrics.Precision()
        precision = metrics_pres.eval([predictions, label])
        if phase == InstanceName.TRAINING:
            step = meta_info[InstanceName.STEP]
            time_cost = meta_info[InstanceName.TIME_COST]
            loss = forward_return_dict[InstanceName.LOSS]
            if isinstance(loss, paddle.Tensor):
                loss_np = loss.numpy()
                mean_loss = np.mean(loss_np)
            else:
                mean_loss = np.mean(loss)
            logging.info("phase = {0} loss = {1} acc = {2} precision = {3} step = {4} time_cost = {5}".format(
                phase, mean_loss, acc, precision, step, round(time_cost, 4)))
        if phase == InstanceName.EVALUATE or phase == InstanceName.TEST:
            time_cost = meta_info[InstanceName.TIME_COST]
            step = meta_info[InstanceName.STEP]
            logging.info("phase = {0} acc = {1} precision = {2} time_cost = {3} step = {4}".format(
                phase, acc, precision, round(time_cost, 4), step))
        metrics_return_dict = collections.OrderedDict()
        metrics_return_dict["acc"] = acc
        metrics_return_dict["precision"] = precision
        return metrics_return_dict

    def metrics_show(self, result_evaluate):
        """评估指标展示
        :return:
        """
        pass
