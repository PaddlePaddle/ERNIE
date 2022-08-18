""" Inference """
import os
import yaml
import numpy as np
from pprint import pprint
from attrdict import AttrDict
import paddle
import paddle.nn.functional as F
import sys
from ernievil2.transformers.multimodal import ERNIE_ViL2_base
from ernievil2.utils import reader

def do_predict(args):
    """
    Inference with a file
    """
    if args.device == "gpu":
        place = "gpu"
    else:
        place = "cpu"
    paddle.set_device(place)
    # Define data loader
    test_loader = reader.create_loader(args)

    # Define model
    model = ERNIE_ViL2_base(args)
    
    model.eval()
    out_file = open(args.output_file, "w", encoding="utf-8")
    with paddle.no_grad():
        for input_data in test_loader:
            
            img_word, input_ids, pos_ids = input_data
            img_word = paddle.concat(x=img_word, axis=0)
            enc_output_img, enc_output_text = model(img_word=img_word, input_ids=input_ids, pos_ids=pos_ids)
            ## normalize
            text_emb = F.normalize(enc_output_text).numpy()
            image_emb = F.normalize(enc_output_img).numpy()
            for i in range(len(enc_output_img)):
                txt_str = ' '.join([str(x) for x in text_emb[i]])
                img_str = ' '.join([str(x) for x in image_emb[i]])
                idx_str = '1'
                out_file.write("\t".join([txt_str, img_str, idx_str])+'\n')
        out_file.close()

if __name__ == "__main__":
    with open(sys.argv[1], 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
    pprint(args)
    do_predict(args)
