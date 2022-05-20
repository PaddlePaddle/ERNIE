# -*- coding: utf-8 -*
"""
ErnieFcSeqLabel
"""
import sys
sys.path.append("../../../")
import paddle
import re
from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
from erniekit.modules.ernie import ErnieModel
from erniekit.model.model import BaseModel
from erniekit.modules.ernie_config import ErnieConfig
from erniekit.modules.ernie_lr import LinearWarmupDecay
from erniekit.metrics import chunk_metrics
import logging
import collections


@RegisterSet.models.register
class ErnieFcSeqLabel(BaseModel):
    """ErnieMatchingFcPointwise:使用TextFieldReader组装数据,只返回src_id和length，用户可以使用src_id自己生成embedding
    """

    def __init__(self, model_params):
        BaseModel.__init__(self, model_params)
        self.config_path = self.model_params["embedding"].get("config_path")
        label_map = {}
        with open(self.model_params.get("vocab_path")) as f:
            for line in f:
                line = line.strip()
                if line:
                    label_str, label_id = line.split('\t')
                    if label_map.get(label_id) is None:
                        label_map[label_id] = label_str
        self.label_list = [label_map[str(i)] for i in range(len(label_map))]
        self.metric = chunk_metrics.ChunkEvaluator(label_list=self.label_list)

    def structure(self):
        """网络结构组织
        :return:
        """
        self.num_labels = self.model_params.get('num_labels', 7)
        self.cfg_dict = ErnieConfig(self.config_path)
        self.hid_dim = self.cfg_dict['hidden_size']

        self.ernie_model = ErnieModel(self.cfg_dict, name='')
        initializer = paddle.nn.initializer.TruncatedNormal(std=0.02)
        self.dropout = paddle.nn.Dropout(p=0.1, mode="upscale_in_train")
        self.fc_prediction = paddle.nn.Linear(in_features=self.hid_dim, out_features=self.num_labels,
                                              weight_attr=paddle.ParamAttr(name='cls_seq_label_out_w',
                                                                           initializer=initializer),
                                              bias_attr='cls_seq_label_out_b')
        self.fc = paddle.nn.Linear(in_features=self.num_labels, out_features=self.num_labels)
        self.loss = paddle.nn.CrossEntropyLoss(use_softmax=False, reduction='none')

    def forward(self, fields_dict, phase):
        """前向计算组网部分，必须由子类实现
        :return: loss , fetch_list
        """

        instance_text = fields_dict["text_a"]
        record_id_text = instance_text[InstanceName.RECORD_ID]
        text_a_src = record_id_text[InstanceName.SRC_IDS]
        text_a_sent = record_id_text[InstanceName.SENTENCE_IDS]
        text_a_mask = record_id_text[InstanceName.MASK_IDS]
        text_a_task = record_id_text[InstanceName.TASK_IDS]

        # [batch_size]
        text_a_lens = record_id_text[InstanceName.SEQ_LENS]

        instance_label = fields_dict["label"]
        record_id_label = instance_label[InstanceName.RECORD_ID]
        # [batch_size, max_seq_len]
        label = record_id_label[InstanceName.SRC_IDS]
        label_lens = record_id_label[InstanceName.SEQ_LENS]
        # cls_embedding = [batch_size, hidden_size], tokens_embedding = [batch_size, max_seq_len, hidden_size]
        cls_embedding, tokens_embedding = self.ernie_model(src_ids=text_a_src, sent_ids=text_a_sent,
                                                           task_ids=text_a_task)

        emb_text_a = self.dropout(tokens_embedding)

        # [batch_size, max_seq_len, num_labels] 
        logits = self.fc_prediction(emb_text_a)
        # [batch_size, max_seq_len, 1]
        infers = paddle.argmax(logits, axis=2)

        # [batch_size * max_seq_len]
        labels = paddle.flatten(label, stop_axis=1)

        # [batch_size * max_seq_len, num_labels]
        logits = paddle.flatten(logits, stop_axis=1)
        probs = paddle.nn.functional.softmax(logits)

        # [batch_size * max_seq_len, 1]
        ce_loss = self.loss(probs, labels)

        # [batch_size * max_seq_len]
        mask_ids = paddle.flatten(text_a_mask, stop_axis=1)
        ce_loss = ce_loss * mask_ids

        loss = paddle.mean(ce_loss)

        if phase == InstanceName.SAVE_INFERENCE:

            target_predict_list = [infers]
            target_feed_list = [text_a_src, text_a_sent]
            target_feed_name_list = ["text_a#src_ids", "text_a#sent_ids"]
            if self.cfg_dict.get('use_task_id', False):
                target_feed_list.append(text_a_task)
                target_feed_name_list.append("text_a#task_ids")

            forward_return_dict = {
                InstanceName.TARGET_FEED: target_feed_list,
                InstanceName.TARGET_FEED_NAMES: target_feed_name_list,
                InstanceName.TARGET_PREDICTS: target_predict_list
            }
            return forward_return_dict

        forward_return_dict = {
            InstanceName.LABEL: label,
            InstanceName.PREDICT_RESULT: infers,
            "length": text_a_lens,
            InstanceName.LOSS: loss
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
        """指标评估部分的动态计算和打印
        :param forward_return_dict: executor.run过程中fetch出来的forward中定义的tensor
        :param meta_info：常用的meta信息，如step, used_time, gpu_id等
        :param phase: 当前调用的阶段，包含训练和评估
        :return:
        """
        predictions = forward_return_dict[InstanceName.PREDICT_RESULT]
        labels = forward_return_dict[InstanceName.LABEL]
        lengths = forward_return_dict["length"]
        self.metric.reset()
        if phase != InstanceName.TRAINING:
            for prediction, label, length in zip(predictions, labels, lengths):
                num_infer_chunks, num_label_chunks, num_correct_chunks = self.metric.compute(length, prediction, label)
        else:
            num_infer_chunks, num_label_chunks, num_correct_chunks = self.metric.compute(lengths, predictions, labels)

        precision, recall, f1_score = self.metric.accumulate()

        if phase == InstanceName.TRAINING:
            step = meta_info[InstanceName.STEP]
            time_cost = meta_info[InstanceName.TIME_COST]
            logging.debug("phase = {0} f1 = {1} precision = {2} recall = {3} step = {4} time_cost = {5}".format(
                phase, round(f1_score, 3), round(precision, 3), round(recall, 3), step, time_cost))
        if phase == InstanceName.EVALUATE or phase == InstanceName.TEST:
            time_cost = meta_info[InstanceName.TIME_COST]
            logging.debug("phase = {0} f1 = {1} precision = {2} recall = {3} time_cost = {4}".format(
                phase, round(f1_score, 3), round(precision, 3), round(recall, 3), time_cost))
        metrics_return_dict = collections.OrderedDict()
        metrics_return_dict["f1"] = f1_score
        metrics_return_dict["precision"] = precision
        metrics_return_dict["recall"] = recall
        return metrics_return_dict

    def metrics_show(self, result_evaluate):
        """评估指标展示
        :return:
        """
        pass
