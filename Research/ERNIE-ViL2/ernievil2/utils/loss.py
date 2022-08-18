import paddle.nn as nn
import paddle.nn.functional as F
import paddle
import paddle.distributed as dist
from paddle.nn.initializer import Uniform, Constant,Normal
import numpy as np
class ContrastiveHead(nn.Layer):
    def __init__(self,args):
        super(ContrastiveHead, self).__init__()
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.temperature_b = self.create_parameter(
            shape=(1,), default_initializer=Constant(value=2.65926),attr=paddle.ParamAttr(name='temperature_b'))
    def clip_logit_scale(self):
        self.temperature_b.clip(-100.0, 100.0)
    def forward(self,feature_img,feature_txt):
        feature_img = F.normalize(feature_img)
        feature_txt = F.normalize(feature_txt)
        if self.args.trainer_count>1:
            feature_list_img = []
            feature_list_txt = []
            dist.all_gather(feature_list_img, feature_img)
            dist.all_gather(feature_list_txt, feature_txt)
            img_feature=paddle.concat(x=feature_list_img, axis=0)
            txt_feature=paddle.concat(x=feature_list_txt, axis=0)
        img_feature = feature_img
        txt_feature = feature_txt
        logit_scale = self.temperature_b.exp()
        logits_per_image = paddle.matmul(logit_scale * img_feature, txt_feature.t())
        logits_per_text = paddle.matmul(logit_scale * txt_feature, img_feature.t())
        self.clip_logit_scale()
        img_labels = paddle.arange(paddle.shape(img_feature)[0]).astype('int64')
        text_labels = paddle.arange(paddle.shape(txt_feature)[0]).astype('int64')
        img_loss = self.criterion(logits_per_image, img_labels)
        text_loss = self.criterion(logits_per_text, text_labels)
        loss = (img_loss + text_loss) / 2
        return loss, loss, logit_scale
