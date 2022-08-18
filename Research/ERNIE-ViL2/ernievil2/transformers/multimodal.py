"""ERNIE-ViL 2.0 main class """
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from ernievil2.utils.loss import ContrastiveHead
from ernievil2.transformers.ERNIE import ErnieModel
from ernievil2.transformers.ViT import ViT_base_patch16_224
from ernievil2.utils.tokenizer import FullTokenizer
import json
class MultiModalModel(nn.Layer):
    """ERNIE-ViL 2.0 backbone """
    def __init__(self,
                 image_model=None,
                 text_model=None,
                 args=None):
        super(MultiModalModel, self).__init__()
        self.visual = image_model
        self.text_model = text_model
        self.contrastive_loss_mm = ContrastiveHead(args)

    def get_loss(self):
        """ get contrastive_loss_mm """
        return self.contrastive_loss_mm

    def forward(self, img_word=None, input_ids=None, pos_ids=None):
        """ ERNIE-ViL 2.0 Forward"""
        img_embedding = self.visual(img_word)
        img_embedding = img_embedding[:, 0]
        pool_out, _ = self.text_model(input_ids, pos_ids=pos_ids)
        return img_embedding, pool_out


def ERNIE_ViL2_base(args, load_weights=True):
    """ ERNIE_ViL2_base: ERNIE +ViT-B-16 """
    image_model = ViT_base_patch16_224()
    config_dict = json.load(open(args.text_model_config))
    text_model = ErnieModel(config_dict)
    model = MultiModalModel(image_model, text_model)
    if args.init_from_params and load_weights:
        sd_param = paddle.load(args.init_from_params)
        model.set_state_dict(sd_param)
    return model
