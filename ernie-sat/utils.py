# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os 
import argparse
import logging
from pathlib import Path

import jsonlines
import numpy as np
import paddle
import soundfile as sf
import yaml
from timer import timer
from yacs.config import CfgNode
from paddlespeech.s2t.utils.dynamic_import import dynamic_import

from paddlespeech.t2s.exps.syn_utils import get_test_dataset
from paddlespeech.t2s.exps.syn_utils import get_voc_inference
from paddlespeech.t2s.utils import str2bool
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2Inference
from paddlespeech.t2s.modules.normalizer import ZScore
from yacs.config import CfgNode
# new add
import paddle.nn.functional as F
from paddlespeech.t2s.modules.nets_utils import make_pad_mask
from paddlespeech.t2s.exps.syn_utils import get_frontend

from sedit_arg_parser import parse_args

model_alias = {
    # acoustic model
    "speedyspeech":
    "paddlespeech.t2s.models.speedyspeech:SpeedySpeech",
    "speedyspeech_inference":
    "paddlespeech.t2s.models.speedyspeech:SpeedySpeechInference",
    "fastspeech2":
    "paddlespeech.t2s.models.fastspeech2:FastSpeech2",
    "fastspeech2_inference":
    "paddlespeech.t2s.models.fastspeech2:FastSpeech2Inference",
    "tacotron2":
    "paddlespeech.t2s.models.tacotron2:Tacotron2",
    "tacotron2_inference":
    "paddlespeech.t2s.models.tacotron2:Tacotron2Inference",
}





def get_voc_out(mel, target_language="chinese"):
    # vocoder
    args = parse_args()
    

    assert target_language == "chinese" or target_language == "english", "In get_voc_out function, target_language is illegal..."
        
    print("current vocoder: ", args.voc)
    with open(args.voc_config) as f:
        voc_config = CfgNode(yaml.safe_load(f))
    # print(voc_config)

    voc_inference = get_voc_inference(args, voc_config)

    mel = paddle.to_tensor(mel)
    # print("masked_mel: ", mel.shape)
    with paddle.no_grad():
        wav = voc_inference(mel)
    # print("shepe of wav (time x n_channels):%s"%wav.shape)   
    return np.squeeze(wav)

# dygraph
def get_am_inference(args, am_config):
    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    # print("vocab_size:", vocab_size)

    tone_size = None
    if 'tones_dict' in args and args.tones_dict:
        with open(args.tones_dict, "r") as f:
            tone_id = [line.strip().split() for line in f.readlines()]
        tone_size = len(tone_id)
        print("tone_size:", tone_size)

    spk_num = None
    if 'speaker_dict' in args and args.speaker_dict:
        with open(args.speaker_dict, 'rt') as f:
            spk_id = [line.strip().split() for line in f.readlines()]
        spk_num = len(spk_id)
        print("spk_num:", spk_num)

    odim = am_config.n_mels
    # model: {model_name}_{dataset}
    am_name = args.am[:args.am.rindex('_')]
    am_dataset = args.am[args.am.rindex('_') + 1:]

    am_class = dynamic_import(am_name, model_alias)
    am_inference_class = dynamic_import(am_name + '_inference', model_alias)

    if am_name == 'fastspeech2':
        am = am_class(
            idim=vocab_size, odim=odim, spk_num=spk_num, **am_config["model"])
    elif am_name == 'speedyspeech':
        am = am_class(
            vocab_size=vocab_size,
            tone_size=tone_size,
            spk_num=spk_num,
            **am_config["model"])
    elif am_name == 'tacotron2':
        am = am_class(idim=vocab_size, odim=odim, **am_config["model"])

    am.set_state_dict(paddle.load(args.am_ckpt)["main_params"])
    am.eval()
    am_mu, am_std = np.load(args.am_stat)
    am_mu = paddle.to_tensor(am_mu)
    am_std = paddle.to_tensor(am_std)
    am_normalizer = ZScore(am_mu, am_std)
    am_inference = am_inference_class(am_normalizer, am)
    am_inference.eval()
    print("acoustic model done!")
    return am, am_inference, am_name, am_dataset, phn_id


def evaluate_durations(phns, target_language="chinese", fs=24000, hop_length=300):
    args = parse_args()
    # args = parser.parse_args(args=[])
    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")


    
    assert target_language == "chinese" or target_language == "english", "In evaluate_durations function, target_language is illegal..."

    # Init body.
    with open(args.am_config) as f:
        am_config = CfgNode(yaml.safe_load(f))
    # print("========Config========")
    # print(am_config)
    # print("---------------------")
    # acoustic model
    am, am_inference, am_name, am_dataset,phn_id = get_am_inference(args, am_config)
    

    torch_phns = phns
    vocab_phones = {}
    for tone, id in phn_id:
        vocab_phones[tone] = int(id)
    # print("vocab_phones: ", len(vocab_phones))
    vocab_size = len(vocab_phones)
    phonemes = [
        phn if phn in vocab_phones else "sp" for phn in torch_phns
    ]
    phone_ids = [vocab_phones[item] for item in phonemes]
    phone_ids_new = phone_ids
    phone_ids_new.append(vocab_size-1)
    phone_ids_new = paddle.to_tensor(np.array(phone_ids_new, np.int64))
    normalized_mel, d_outs, p_outs, e_outs = am.inference(phone_ids_new, spk_id=None, spk_emb=None)
    pre_d_outs = d_outs
    phoneme_durations_new = pre_d_outs * hop_length / fs
    phoneme_durations_new = phoneme_durations_new.tolist()[:-1]
    return phoneme_durations_new



def sentence2phns(sentence, target_language="en"):
    args = parse_args()
    if target_language == 'en':
        args.lang='en'
        args.phones_dict = "download/fastspeech2_nosil_ljspeech_ckpt_0.5/phone_id_map.txt"
    elif target_language == 'zh':
        args.lang='zh'
        args.phones_dict="download/fastspeech2_conformer_baker_ckpt_0.5/phone_id_map.txt" 
    else:
        print("target_language should in {'zh', 'en'}!")
    
    frontend = get_frontend(args)
    merge_sentences = True
    get_tone_ids = False

    if target_language == 'zh':
        input_ids = frontend.get_input_ids(
            sentence,
            merge_sentences=merge_sentences,
            get_tone_ids=get_tone_ids,
            print_info=False)
        phone_ids = input_ids["phone_ids"]

        phonemes = frontend.get_phonemes(
            sentence, 
            merge_sentences=merge_sentences,
            print_info=False)
            
        return phonemes[0], input_ids["phone_ids"][0]

    elif target_language == 'en':
        phonemes = frontend.phoneticize(sentence)
        input_ids = frontend.get_input_ids(
            sentence, merge_sentences=merge_sentences)
        phone_ids = input_ids["phone_ids"]

        phones_list = []
        vocab_phones = {}
        punc = "：，；。？！“”‘’':,;.?!"
        with open(args.phones_dict, 'rt') as f:
            phn_id = [line.strip().split() for line in f.readlines()]
        for phn, id in phn_id:
            vocab_phones[phn] = int(id)

        phones = phonemes[1:-1]
        phones = [phn for phn in phones if not phn.isspace()]
        # replace unk phone with sp
        phones = [
            phn
            if (phn in vocab_phones and phn not in punc) else "sp"
            for phn in phones
        ]
        phones_list.append(phones)
        return phones_list[0], input_ids["phone_ids"][0] 

    else:
        print("lang should in {'zh', 'en'}!")

    


