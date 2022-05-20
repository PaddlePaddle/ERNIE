# -*- coding: utf-8 -*
"""
BOW 分类网络
"""
import paddle
from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
from erniekit.model.model import BaseModel
from erniekit.modules.encoder import BoWEncoder
from model.base_cls import BaseClassification


@RegisterSet.models.register
class BowClassification(BaseClassification):
    """BowClassification
    """
    def __init__(self, model_params):
        """
        """
        BaseModel.__init__(self, model_params)

    def structure(self):
        """网络结构组织，定义需要用到的成员变量即可
        :return: None
        """
        self.dict_dim = self.model_params.get('vocab_size', 33261)
        self.emb_dim = self.model_params.get('emb_dim', 128)
        self.hid_dim = self.model_params.get('hid_dim', 128)
        self.hid_dim2 = self.model_params.get('hid_dim2', 96)
        self.num_labels = self.model_params.get('num_labels', 2)

        self.embedding = paddle.nn.Embedding(num_embeddings=self.dict_dim, embedding_dim=self.emb_dim)
        self.bow_encoder = BoWEncoder(self.emb_dim)
        self.fc_1 = paddle.nn.Linear(in_features=self.hid_dim, out_features=self.hid_dim)
        self.fc_2 = paddle.nn.Linear(in_features=self.hid_dim, out_features=self.hid_dim2)
        self.fc_prediction = paddle.nn.Linear(in_features=self.hid_dim2, out_features=self.num_labels)
        self.loss = paddle.nn.CrossEntropyLoss(use_softmax=False)

    def forward(self, fields_dict, phase):
        """
        :param fields_dict: 动态图模式下是tensor格式，静态图模式下是python数组
        :param phase:
        :return:
        """
        instance_text_a = fields_dict["text_a"]
        record_id_text_a = instance_text_a[InstanceName.RECORD_ID]
        text_src = record_id_text_a[InstanceName.SRC_IDS]

        emb_output = self.embedding(text_src)
        bow_output = self.bow_encoder(emb_output)
        # bow_output = paddle.sum(emb_output, axis=1)

        fc_1_output = paddle.tanh(self.fc_1(bow_output))
        fc_2_output = paddle.tanh(self.fc_2(fc_1_output))
        prediction = self.fc_prediction(fc_2_output)
        probs = paddle.nn.functional.softmax(prediction)

        if phase == InstanceName.TRAINING or phase == InstanceName.EVALUATE or phase == InstanceName.TEST:
            instance_label = fields_dict["label"]
            record_id_label = instance_label[InstanceName.RECORD_ID]
            label = record_id_label[InstanceName.SRC_IDS]
            # label = paddle.to_tensor(label)
            cost = self.loss(probs, label)
            forward_return_dict = {
                InstanceName.PREDICT_RESULT: probs,
                InstanceName.LABEL: label,
                InstanceName.LOSS: cost
            }
        elif phase == InstanceName.SAVE_INFERENCE:
            "save inference model with jit"
            target_predict_list = [probs]
            target_feed_list = [text_src]
            # 以json的形式存入模型的meta文件中，在离线预测的时候用，field_name#field_tensor_name
            target_feed_name_list = ["text_a#src_ids"]
            forward_return_dict = {
                InstanceName.TARGET_FEED: target_feed_list,
                InstanceName.TARGET_PREDICTS: target_predict_list,
                InstanceName.TARGET_FEED_NAMES: target_feed_name_list
            }
        else:
            forward_return_dict = {
                InstanceName.PREDICT_RESULT: probs
            }
        return forward_return_dict
