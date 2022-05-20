# -*- coding: utf-8 -*
"""
ErnieMatchingFcPointwise
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
class ErnieFcIe(BaseModel):
    """ErnieMatchingFcPointwise:使用TextFieldReader组装数据,只返回src_id和length，用户可以使用src_id自己生成embedding
    """

    def __init__(self, model_params):
        BaseModel.__init__(self, model_params)

    def structure(self):
        """网络结构组织
        :return:
        """

        self.num_labels = self.model_params.get('num_labels', 12)
        emb_params = self.model_params.get("embedding")
        # use_fp16 = emb_params.get("use_fp16")

        config_path = emb_params.get("config_path")
        self.cfg_dict = ErnieConfig(config_path)
        self.hid_dim = self.cfg_dict['hidden_size']

        self.ernie_model = ErnieModel(self.cfg_dict, name='')
        initializer = paddle.nn.initializer.TruncatedNormal(std=0.02)
        self.dropout = paddle.nn.Dropout(p=0.1, mode="upscale_in_train")

        self.fc_prediction = paddle.nn.Linear(in_features=self.hid_dim, out_features=self.num_labels,
                                              weight_attr=paddle.ParamAttr(name='cls_ie_out.w_0',
                                                                           initializer=initializer),
                                              bias_attr='cls_ie_out.b_0')

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
        ## 比如bs为2，长度分别为20，40，那么seq_len_text_a = [20,40]  shape:[2]
        text_a_seq_lens = record_id_text[InstanceName.SEQ_LENS] 
        text_a_beg_ids = record_id_text[InstanceName.BEG_IDS]
        text_a_end_ids = record_id_text[InstanceName.END_IDS]
        instance_label = fields_dict["label"]
        record_id_label = instance_label[InstanceName.RECORD_ID]
        # [batch_size, max_seq_len,num_lables]
        labels = record_id_label[InstanceName.SRC_IDS]

        # cls_embedding = [batch_size, hidden_size], tokens_embedding = [batch_size, max_seq_len, hidden_size]
        cls_embedding, tokens_embedding = self.ernie_model(src_ids=text_a_src, sent_ids=text_a_sent,
                                                           task_ids=text_a_task)

        emb_text_a = self.dropout(tokens_embedding)

        logits = self.fc_prediction(emb_text_a)

        predictions = paddle.nn.functional.sigmoid(logits)

        predictions = paddle.flatten(predictions, stop_axis=1)

        labels = paddle.flatten(labels, stop_axis=1)

        logits = paddle.flatten(logits, stop_axis=1)
     
        ce_loss = paddle.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        ce_loss = paddle.mean(ce_loss, axis=-1, keepdim=True)
        mask_ids = paddle.flatten(text_a_mask, stop_axis=1)
        #mask_ids是去除padding部分的loss
        ce_loss = ce_loss * mask_ids
        loss = paddle.mean(ce_loss)
        if phase == InstanceName.SAVE_INFERENCE:
            target_predict_list = [predictions]
            target_feed_list = [text_a_src, text_a_sent]
            target_feed_name_list = ["text_a#src_ids", "text_a#sent_ids"]  ##分隔符后遵循rule中定义
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
            InstanceName.PREDICT_RESULT: predictions,  # [batch_size * max_seq_len,num_labels]
            InstanceName.LABEL: labels, # [batch_size * max_seq_len,num_labels]
            InstanceName.LOSS: loss,
            InstanceName.SEQ_LENS: text_a_seq_lens
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

    def _post_proc(self, sample_probs):
        """
        post proc
        """
        sample_probs = np.where(sample_probs < 0.5, 0, 1)
        length, _ = sample_probs.shape
        
        for i in range(length - 1):
            if sample_probs[i][0] == 1 and np.sum(sample_probs[i]) > 1:
                if sample_probs[i + 1][1] == 1:
                    sample_probs[i][0] = 0
                else:
                    sample_probs[i][2:] = 0

        for i in range(length - 1):
            if np.sum(sample_probs[i]) == 0:
                if sample_probs[i - 1][1] == 1 and sample_probs[i + 1][1] == 1:
                    sample_probs[i][1] = 1
                elif sample_probs[i + 1][1] == 1:
                    sample_probs[i][np.argmax(sample_probs[i, 1:]) + 1] = 1
        return sample_probs


    def get_metrics(self, forward_return_dict, meta_info, phase):
        """
        get metrics
        """
        labels = forward_return_dict[InstanceName.LABEL]
        predictions = forward_return_dict[InstanceName.PREDICT_RESULT]
        seq_lens = forward_return_dict[InstanceName.SEQ_LENS]
        loss = forward_return_dict[InstanceName.LOSS]
       
        # step = meta_info[InstanceName.STEP]
        # time_cost = meta_info[InstanceName.TIME_COST]

        if self.is_dygraph:
            if isinstance(predictions, list):
                predictions = [item.numpy() for item in predictions]
            else:
                predictions = predictions.numpy()
            
            if isinstance(labels, list):
                labels = [item.numpy() for item in labels]
            else:
                labels = labels.numpy()
            
            if isinstance(seq_lens, list):
                seq_lens = [item.numpy() for item in seq_lens]
            else:
                seq_lens = seq_lens.numpy()
            if isinstance(loss, list):
                loss = [item.numpy() for item in loss]
            else:
                loss = loss.numpy()

        if phase == InstanceName.TRAINING:
            tp = 0
            tp_fp = 0
            tp_fn = 0
            max_seq_len = labels.shape[0] // seq_lens.shape[0]

            for i in range(seq_lens.shape[0]):
                sample_labels = labels[max_seq_len * i: max_seq_len * i + seq_lens[i]]
                sample_probs = predictions[max_seq_len * i: max_seq_len * i + seq_lens[i]]
                sample_probs = self._post_proc(sample_probs)
                for token_labels, token_probs in zip(sample_labels, sample_probs):
                    if (token_labels == token_probs).all() and token_labels[0] != 1:
                        tp += 1
                    if token_probs[0] != 1:
                        tp_fp += 1
                    if token_labels[0] != 1:
                        tp_fn += 1
            precision = 1.0 * tp / tp_fp if tp_fp != 0 else 0
            recall = 1.0 * tp / tp_fn if tp_fn != 0 else 0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            step = meta_info[InstanceName.STEP]
            time_cost = meta_info[InstanceName.TIME_COST]
            logging.debug("phase = {0} precision = {1} recall = {2} f1 = {3} step = {4}\
            time_cost = {5} loss = {6}".format(phase, round(precision, 3), round
            (recall, 3), round(f1, 3), step, time_cost, loss))
        if phase == InstanceName.EVALUATE or phase == InstanceName.TEST:
            tp = 0
            tp_fp = 0
            tp_fn = 0
 
            for predictions, labels, seq_lens in zip(predictions, labels, seq_lens):

                max_seq_len = labels.shape[0] // seq_lens.shape[0]

                for i in range(seq_lens.shape[0]):
                    sample_labels = labels[max_seq_len * i: max_seq_len * i + seq_lens[i]]
                    sample_probs = predictions[max_seq_len * i: max_seq_len * i + seq_lens[i]]
                    sample_probs = self._post_proc(sample_probs)
                    for token_labels, token_probs in zip(sample_labels, sample_probs):
                        if (token_labels == token_probs).all() and token_labels[0] != 1:
                            tp += 1
                        if token_probs[0] != 1:
                            tp_fp += 1
                        if token_labels[0] != 1:
                            tp_fn += 1
            precision = 1.0 * tp / tp_fp if tp_fp != 0 else 0
            recall = 1.0 * tp / tp_fn if tp_fn != 0 else 0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            step = meta_info[InstanceName.STEP]
            time_cost = meta_info[InstanceName.TIME_COST]
            logging.debug("phase = {0} precision = {1} recall = {2} f1 = {3} time_cost = {4}".format(
                phase, round(precision, 3), round(recall, 3), round(f1, 3), time_cost))

        metrics_return_dict = collections.OrderedDict()
        metrics_return_dict["precision"] = precision
        metrics_return_dict["recall"] = recall
        metrics_return_dict["f1"] = f1
        return metrics_return_dict

    def metrics_show(self, result_evaluate):
        """
        metrics show
        """
        pass