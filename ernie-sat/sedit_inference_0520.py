#!/usr/bin/env python3

"""Script to run the inference of text-to-speeech model."""

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from parallel_wavegan.utils import download_pretrained_model
from pathlib import Path
import paddle
import soundfile
import os
import math
import string
import numpy as np

from espnet2.tasks.mlm import MLMTask
from read_text import read_2column_text,load_num_sequence_text
from util import sentence2phns,get_voc_out, evaluate_durations
import librosa
import random
from ipywidgets import widgets
import IPython.display as ipd
import soundfile as sf
import sys 
import pickle
from model_paddle import build_model_from_file

from sedit_arg_parser import parse_args
import argparse
from typing import Collection
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

from paddlespeech.t2s.datasets.get_feats import LogMelFBank
from paddlespeech.t2s.modules.nets_utils import make_non_pad_mask 

duration_path_dict = {
    "ljspeech":"/mnt/home/v_baihe/projects/espnet/egs2/ljspeech/tts1/exp/kan-bayashi/ljspeech_tts_train_conformer_fastspeech2_raw_phn_tacotron_g2p_en_no_space_train.loss.ave/train.loss.ave_5best.pth",
    "vctk": "/mnt/home/v_baihe/projects/espnet/egs2/vctk/tts1/exp/kan-bayashi/vctk_tts_train_gst+xvector_conformer_fastspeech2_transformer_teacher_raw_phn_tacotron_g2p_en_no_space_train.loss.ave/train.loss.ave_5best.pth",
    # "ljspeech":"/home/mnt2/zz/workspace/work/espnet_richard_infer/egs2/ljspeech/tts1/exp/kan-bayashi/ljspeech_tts_train_conformer_fastspeech2_raw_phn_tacotron_g2p_en_no_space_train.loss.ave/train.loss.ave_5best.pth",
    # "vctk": "/home/mnt2/zz/workspace/work/espnet_richard_infer/egs2/vctk/tts1/exp/kan-bayashi/vctk_tts_train_gst+xvector_conformer_fastspeech2_transformer_teacher_raw_phn_tacotron_g2p_en_no_space_train.loss.ave/train.loss.ave_5best.pth",
    "vctk_unseen":"/mnt/home/v_baihe/projects/espnet/egs2/vctk/tts1/exp/tts_train_fs2_raw_phn_tacotron_g2p_en_no_space/train.loss.ave_5best.pth",
    "libritts":"/mnt/home/v_baihe/projects/espnet/egs2/libritts/tts1/exp/kan-bayashi/libritts_tts_train_gst+xvector_conformer_fastspeech2_transformer_teacher_raw_phn_tacotron_g2p_en_no_space_train.loss/train.loss.ave_5best.pth"
}

random.seed(0)
np.random.seed(0)


def plot_mel_and_vocode_wav(uid, prefix, clone_uid, clone_prefix, source_language, target_language, model_name, wav_path,full_origin_str, old_str, new_str, vocoder,duration_preditor_path,sid=None, non_autoreg=True):
    wav_org, input_feat, output_feat, old_span_boundary, new_span_boundary, fs, hop_length = get_mlm_output(
                                                            uid,
                                                            prefix,
                                                            clone_uid,
                                                            clone_prefix,
                                                            source_language,
                                                            target_language,
                                                            model_name,
                                                            wav_path,
                                                            old_str,
                                                            new_str, 
                                                            duration_preditor_path,
                                                            use_teacher_forcing=non_autoreg,
                                                            sid=sid
                                                            )
    
    masked_feat = output_feat[new_span_boundary[0]:new_span_boundary[1]].detach().float().cpu().numpy()
    
    if target_language == 'english':
        output_feat_np = output_feat.detach().float().cpu().numpy()
        replaced_wav_paddle_voc = get_voc_out(output_feat_np, target_language)

    elif target_language == 'chinese':
        output_feat_np = output_feat.detach().float().cpu().numpy()
        replaced_wav_only_mask_fst2_voc = get_voc_out(masked_feat, target_language)
    

    old_time_boundary = [hop_length * x  for x in old_span_boundary]
    new_time_boundary = [hop_length * x  for x in new_span_boundary]
    
    
    if target_language == 'english':
        wav_org_replaced_paddle_voc = np.concatenate([wav_org[:old_time_boundary[0]], replaced_wav_paddle_voc[new_time_boundary[0]:new_time_boundary[1]], wav_org[old_time_boundary[1]:]])

        data_dict = {"origin":wav_org,
                    "output":wav_org_replaced_paddle_voc}

    elif  target_language == 'chinese':
        wav_org_replaced_only_mask_fst2_voc = np.concatenate([wav_org[:old_time_boundary[0]], replaced_wav_only_mask_fst2_voc, wav_org[old_time_boundary[1]:]])
        data_dict = {"origin":wav_org,
                    "output": wav_org_replaced_only_mask_fst2_voc,}
    
    return data_dict, old_span_boundary



def load_vocoder(vocoder_tag="parallel_wavegan/libritts_parallel_wavegan.v1"):
    vocoder_tag = vocoder_tag.replace("parallel_wavegan/", "")
    vocoder_file = download_pretrained_model(vocoder_tag)
    vocoder_config = Path(vocoder_file).parent / "config.yml"

    vocoder = TTSTask.build_vocoder_from_file(
                    vocoder_config, vocoder_file, None, 'cpu'
                )
    return vocoder

def load_model(model_name):
    config_path='./pretrained_model/{}/config.yaml'.format(model_name)
    model_path = './pretrained_model/{}/model.pdparams'.format(model_name)
    
    mlm_model, args = build_model_from_file(config_file=config_path,
                                 model_file=model_path)
    return mlm_model, args


def read_data(uid,prefix):
    mfa_text = read_2column_text(prefix+'/text')[uid]
    mfa_wav_path = read_2column_text(prefix+'/wav.scp')[uid]
    if 'mnt' not in mfa_wav_path:
        mfa_wav_path = prefix.split('dump')[0] + mfa_wav_path
    return mfa_text, mfa_wav_path
 
def get_align_data(uid,prefix):
    mfa_path = prefix+"mfa_"
    mfa_text = read_2column_text(mfa_path+'text')[uid]
    mfa_start = load_num_sequence_text(mfa_path+'start',loader_type='text_float')[uid]
    mfa_end = load_num_sequence_text(mfa_path+'end',loader_type='text_float')[uid]
    mfa_wav_path = read_2column_text(mfa_path+'wav.scp')[uid]
    return mfa_text, mfa_start, mfa_end, mfa_wav_path



def get_fs2_model(model_name):
    model, config = TTSTask.build_model_from_file(model_file=model_name)
    processor = TTSTask.build_preprocess_fn(config, train=False)
    return model, processor

def get_masked_mel_boundary(mfa_start, mfa_end, fs, hop_length, span_tobe_replaced):
    align_start=paddle.to_tensor(mfa_start).unsqueeze(0)
    align_end =paddle.to_tensor(mfa_end).unsqueeze(0)
    align_start = paddle.floor(fs*align_start/hop_length).int()
    align_end = paddle.floor(fs*align_end/hop_length).int()
    if span_tobe_replaced[0]>=len(mfa_start):
        span_boundary = [align_end[0].tolist()[-1],align_end[0].tolist()[-1]]
    else:
        span_boundary=[align_start[0].tolist()[span_tobe_replaced[0]],align_end[0].tolist()[span_tobe_replaced[1]-1]]
    return span_boundary


def get_mapping(phn_mapping="./phn_mapping.txt"):
    zh_mapping = {}
    with open(phn_mapping, "r") as f:
        for line in f:
            pd_phn = line.split(" ")[0]
            if pd_phn not in zh_mapping.keys():
                zh_mapping[pd_phn] = " ".join(line.split()[1:])
    return zh_mapping


def gen_phns(zh_mapping, phns):
    new_phns = []
    for x in phns:
        if x in zh_mapping.keys():
            new_phns.extend(zh_mapping[x].split(" "))
        else:
            new_phns.extend(['<unk>'])
    return new_phns

def get_phns_and_spans_paddle(uid, prefix, old_str, new_str, source_language, target_language):
    zh_mapping = get_mapping()
    old_str = old_str.strip()
    new_str = new_str.strip()
    words = []
    for pun in [',', '.', ':', ';', '!', '?', '"', '(', ')', '--', '---', u'，', u'。', u'：', u'；', u'！', u'？', u'（', u'）']:
        old_str = old_str.replace(pun, ' ')
        new_str = new_str.replace(pun, ' ')


    append_new_str = (old_str == new_str[:len(old_str)])
    print("append_new_str: ", append_new_str)
    old_phns, mfa_start, mfa_end = [], [], []
    mfa_text, mfa_start, mfa_end, mfa_wav_path = get_align_data(uid, prefix)
    old_phns = mfa_text.split(" ")

    if append_new_str:
        if source_language != target_language:
            is_cross_lingual = True 
        else:
            is_cross_lingual = False

        new_str_origin = new_str[:len(old_str)]
        new_str_append = new_str[len(old_str):]
        if is_cross_lingual:
            if source_language == "english" and target_language == "chinese": 
                new_phns_origin = old_phns
                new_phns_append, _ = sentence2phns(new_str_append, "zh")

            elif source_language=="chinese" and target_language == "english":
                new_phns_origin = old_phns
                new_phns_append, _ = sentence2phns(new_str_append, "en")
            else:
                assert target_language == "chinese" or target_language == "english", "cloning is not support for this language, please check it."
            
        else:  
            if source_language == target_language and target_language == "english":
                new_phns_origin = old_phns
                new_phns_append, _ = sentence2phns(new_str_append, "en")

            elif source_language == target_language and target_language == "chinese":
                new_phns_origin = old_phns
                new_phns_append, _ = sentence2phns(new_str_append, "zh")
            else:
                assert source_language == target_language, "source language is not same with target language..."

        if target_language == "chinese":
            new_phns_append = gen_phns(zh_mapping, new_phns_append)

        new_phns = new_phns_origin + new_phns_append

        span_tobe_replaced = [len(old_phns),len(old_phns)]
        span_tobe_added = [len(old_phns),len(new_phns)] 
        
    else:
        if source_language == target_language and target_language == "english":
            new_phns, _ = sentence2phns(new_str, "en")
        # 纯中文
        elif source_language == target_language and target_language == "chinese":
            new_phns, _ = sentence2phns(new_str, "zh")
            new_phns = gen_phns(zh_mapping, new_phns)


        else:
            assert source_language == target_language, "source language is not same with target language..."
    
        while(new_phns[-1] == 'sp'):
            new_phns.pop()

        while(new_phns[0] == 'sp'):
            new_phns.pop(0)

        span_tobe_replaced = [0,len(old_phns)-1]
        span_tobe_added = [0,len(new_phns)-1]
        new_phns_left = []
        left_index = 0
        sp_count = 0

        # find the left different index
        for idx, phn in enumerate(old_phns):
            if phn == "sp":
                sp_count += 1 
                new_phns_left.append('sp')
            else:
                idx = idx - sp_count
                if phn == new_phns[idx]:
                    left_index += 1 
                    new_phns_left.append(phn)
                else:
                    span_tobe_replaced[0] = len(new_phns_left)
                    span_tobe_added[0] = len(new_phns_left)
                    break

        right_index = 0
        new_phns_middle = []
        new_phns_right = []
        sp_count = 0
        word2phns_max_index = len(old_phns)
        new_word2phns_max_index = len(new_phns)

        for idx, phn in enumerate(old_phns[::-1]):
            cur_idx = len(old_phns) - 1 - idx
            if phn == "sp":
                sp_count += 1 
                new_phns_right = ['sp']+new_phns_right
            else:
                cur_idx = new_word2phns_max_index - (word2phns_max_index - cur_idx -sp_count)
                if phn == new_phns[cur_idx]:
                    right_index -= 1
                    new_phns_right = [phn] + new_phns_right
                
                else:
                    span_tobe_replaced[1] = len(old_phns) - len(new_phns_right)
                    new_phns_middle = new_phns[left_index:right_index]
                    span_tobe_added[1] = len(new_phns_left) + len(new_phns_middle)
                    if len(new_phns_middle) == 0:
                        span_tobe_added[1] = min(span_tobe_added[1]+1, len(new_phns))
                        span_tobe_added[0] = max(0, span_tobe_added[0]-1)
                        span_tobe_replaced[0] = max(0, span_tobe_replaced[0]-1)
                        span_tobe_replaced[1] = min(span_tobe_replaced[1]+1, len(old_phns))
                    break        

        new_phns = new_phns_left+new_phns_middle+new_phns_right
    
    return mfa_start, mfa_end, old_phns, new_phns, span_tobe_replaced, span_tobe_added




def duration_adjust_factor(original_dur, pred_dur, phns):
    length = 0
    accumulate = 0
    factor_list = []
    for ori,pred,phn in zip(original_dur, pred_dur,phns):
        if pred==0 or phn=='sp':
            continue
        else:
            factor_list.append(ori/pred)
    factor_list = np.array(factor_list)
    factor_list.sort()
    if len(factor_list)<5:
        return 1
    length = 2
    return np.average(factor_list[length:-length])


def prepare_features_with_duration(uid, prefix, clone_uid, clone_prefix, source_language, target_language, mlm_model, old_str, new_str, wav_path,duration_preditor_path,sid=None, mask_reconstruct=False,duration_adjust=True,start_end_sp=False, train_args=None):
    wav_org, rate = librosa.load(wav_path, sr=train_args.feats_extract_conf['fs'])
    fs = train_args.feats_extract_conf['fs']
    hop_length = train_args.feats_extract_conf['hop_length']
    
    mfa_start, mfa_end, old_phns, new_phns, span_tobe_replaced, span_tobe_added = get_phns_and_spans_paddle(uid, prefix, old_str, new_str, source_language, target_language)
    
    if start_end_sp:
        if new_phns[-1]!='sp':
            new_phns = new_phns+['sp']
   

    if target_language == "english":
        old_durations = evaluate_durations(old_phns, target_language=target_language)

    elif target_language =="chinese":
        if source_language == "english":
            old_durations = evaluate_durations(old_phns, target_language=source_language)
        elif source_language == "chinese":
            old_durations = evaluate_durations(old_phns, target_language=source_language)

    else:
        assert target_language == "chinese" or target_language == "english", "calculate duration_predict is not support for this language..."



    original_old_durations = [e-s for e,s in zip(mfa_end, mfa_start)]
    if '[MASK]' in new_str:
        new_phns = old_phns
        span_tobe_added = span_tobe_replaced
        d_factor_left = duration_adjust_factor(original_old_durations[:span_tobe_replaced[0]],old_durations[:span_tobe_replaced[0]], old_phns[:span_tobe_replaced[0]])
        d_factor_right = duration_adjust_factor(original_old_durations[span_tobe_replaced[1]:],old_durations[span_tobe_replaced[1]:], old_phns[span_tobe_replaced[1]:])
        d_factor = (d_factor_left+d_factor_right)/2
        new_durations_adjusted = [d_factor*i for i in old_durations]
    else:
        if duration_adjust:
            d_factor = duration_adjust_factor(original_old_durations,old_durations, old_phns)
            d_factor_paddle = duration_adjust_factor(original_old_durations,old_durations, old_phns)
            if target_language =="chinese":
                d_factor = d_factor * 1.35  
        else:
            d_factor = 1
        
        if target_language == "english":
            new_durations = evaluate_durations(new_phns, target_language=target_language)


        elif target_language =="chinese":
            new_durations = evaluate_durations(new_phns, target_language=target_language)

        new_durations_adjusted = [d_factor*i for i in new_durations]

        if span_tobe_replaced[0]<len(old_phns) and old_phns[span_tobe_replaced[0]] == new_phns[span_tobe_added[0]]:
            new_durations_adjusted[span_tobe_added[0]] = original_old_durations[span_tobe_replaced[0]]
        if span_tobe_replaced[1]<len(old_phns) and span_tobe_added[1]<len(new_phns):
            if old_phns[span_tobe_replaced[1]] == new_phns[span_tobe_added[1]]:
                new_durations_adjusted[span_tobe_added[1]] = original_old_durations[span_tobe_replaced[1]]
    new_span_duration_sum = sum(new_durations_adjusted[span_tobe_added[0]:span_tobe_added[1]])
    old_span_duration_sum = sum(original_old_durations[span_tobe_replaced[0]:span_tobe_replaced[1]])
    duration_offset =  new_span_duration_sum - old_span_duration_sum
    new_mfa_start = mfa_start[:span_tobe_replaced[0]]
    new_mfa_end = mfa_end[:span_tobe_replaced[0]]
    for i in new_durations_adjusted[span_tobe_added[0]:span_tobe_added[1]]:
        if len(new_mfa_end) ==0:
            new_mfa_start.append(0)
            new_mfa_end.append(i)
        else:
            new_mfa_start.append(new_mfa_end[-1])
            new_mfa_end.append(new_mfa_end[-1]+i)
    new_mfa_start += [i+duration_offset for i in mfa_start[span_tobe_replaced[1]:]]
    new_mfa_end += [i+duration_offset for i in mfa_end[span_tobe_replaced[1]:]]
    
    # 3. get new wav 
    if span_tobe_replaced[0]>=len(mfa_start):
        left_index = len(wav_org)
        right_index = left_index
    else:
        left_index = int(np.floor(mfa_start[span_tobe_replaced[0]]*fs))
        right_index = int(np.ceil(mfa_end[span_tobe_replaced[1]-1]*fs))
    new_blank_wav = np.zeros((int(np.ceil(new_span_duration_sum*fs)),), dtype=wav_org.dtype)
    new_wav_org = np.concatenate([wav_org[:left_index], new_blank_wav, wav_org[right_index:]])


    # 4. get old and new mel span to be mask
    old_span_boundary = get_masked_mel_boundary(mfa_start, mfa_end, fs, hop_length, span_tobe_replaced)   # [92, 92]
    new_span_boundary=get_masked_mel_boundary(new_mfa_start, new_mfa_end, fs, hop_length, span_tobe_added) # [92, 174]

    
    return new_wav_org, new_phns, new_mfa_start, new_mfa_end, old_span_boundary, new_span_boundary

def prepare_features(uid, prefix, clone_uid, clone_prefix, source_language, target_language, mlm_model,processor, wav_path, old_str,new_str,duration_preditor_path, sid=None,duration_adjust=True,start_end_sp=False,
mask_reconstruct=False, train_args=None):
    wav_org, phns_list, mfa_start, mfa_end, old_span_boundary, new_span_boundary = prepare_features_with_duration(uid, prefix, clone_uid, clone_prefix, source_language, target_language, mlm_model, old_str, 
    new_str, wav_path,duration_preditor_path,sid=sid,duration_adjust=duration_adjust,start_end_sp=start_end_sp,mask_reconstruct=mask_reconstruct, train_args = train_args)
    speech = np.array(wav_org,dtype=np.float32)
    align_start=np.array(mfa_start)
    align_end =np.array(mfa_end)
    token_to_id = {item: i for i, item in enumerate(train_args.token_list)}
    text = np.array(list(map(lambda x: token_to_id.get(x, token_to_id['<unk>']), phns_list)))
    print('unk id is', token_to_id['<unk>'])
    # text = np.array(processor(uid='1', data={'text':" ".join(phns_list)})['text'])
    span_boundary = np.array(new_span_boundary)
    batch=[('1', {"speech":speech,"align_start":align_start,"align_end":align_end,"text":text,"span_boundary":span_boundary})]
    
    return batch, old_span_boundary, new_span_boundary

def decode_with_model(uid, prefix, clone_uid, clone_prefix, source_language, target_language, mlm_model, processor, collate_fn, wav_path, old_str, new_str,duration_preditor_path, sid=None, decoder=False,use_teacher_forcing=False,duration_adjust=True,start_end_sp=False, train_args=None):
    # fs, hop_length = mlm_model.feats_extract.fs, mlm_model.feats_extract.hop_length
    fs, hop_length = train_args.feats_extract_conf['fs'], train_args.feats_extract_conf['hop_length']

    batch,old_span_boundary,new_span_boundary = prepare_features(uid,prefix, clone_uid, clone_prefix, source_language, target_language, mlm_model,processor,wav_path,old_str,new_str,duration_preditor_path, sid,duration_adjust=duration_adjust,start_end_sp=start_end_sp, train_args=train_args)
    
    feats = pickle.load(open('tmp/tmp_pkl.'+str(uid), 'rb'))

    # wav_len * 80
    # set_all_random_seed(9999)
    if 'text_masked_position' in feats.keys():
        feats.pop('text_masked_position')
    for k, v in feats.items():
        feats[k] = paddle.to_tensor(v)
    rtn = mlm_model.inference(**feats,span_boundary=new_span_boundary,use_teacher_forcing=use_teacher_forcing)
    output = rtn['feat_gen'] 
    if 0 in output[0].shape and 0 not in output[-1].shape:
        output_feat = paddle.concat(output[1:-1]+[output[-1].squeeze()], axis=0).cpu()
    elif 0 not in output[0].shape and 0 in output[-1].shape:
        output_feat = paddle.concat([output[0].squeeze()]+output[1:-1], axis=0).cpu()
    elif 0 in output[0].shape and 0 in output[-1].shape:
        output_feat = paddle.concat(output[1:-1], axis=0).cpu()
    else:
        output_feat = paddle.concat([output[0].squeeze(0)]+ output[1:-1]+[output[-1].squeeze(0)], axis=0).cpu()


    # wav_org, rate = soundfile.read(
    #             wav_path, always_2d=False)
    wav_org, rate = librosa.load(wav_path, sr=train_args.feats_extract_conf['fs'])
    origin_speech = paddle.to_tensor(np.array(wav_org,dtype=np.float32)).unsqueeze(0)
    speech_lengths = paddle.to_tensor(len(wav_org)).unsqueeze(0)
    # input_feat, feats_lengths = mlm_model.feats_extract(origin_speech, speech_lengths)
    # return wav_org, input_feat.squeeze(), output_feat, old_span_boundary, new_span_boundary, fs, hop_length
    return wav_org, None, output_feat, old_span_boundary, new_span_boundary, fs, hop_length

class MLMCollateFn:
    """Functor class of common_collate_fn()"""

    def __init__(
        self,
        feats_extract,
        float_pad_value: Union[float, int] = 0.0,
        int_pad_value: int = -32768,
        not_sequence: Collection[str] = (),
        mlm_prob: float=0.8,
        mean_phn_span: int=8,
        attention_window: int=0,
        pad_speech: bool=False,
        sega_emb: bool=False,
        duration_collect: bool=False,
        text_masking: bool=False

    ):
        self.mlm_prob=mlm_prob
        self.mean_phn_span=mean_phn_span
        self.feats_extract = feats_extract
        self.float_pad_value = float_pad_value
        self.int_pad_value = int_pad_value
        self.not_sequence = set(not_sequence)
        self.attention_window=attention_window
        self.pad_speech=pad_speech
        self.sega_emb=sega_emb
        self.duration_collect = duration_collect
        self.text_masking = text_masking

    def __repr__(self):
        return (
            f"{self.__class__}(float_pad_value={self.float_pad_value}, "
            f"int_pad_value={self.float_pad_value})"
        )

    def __call__(
        self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
    ) -> Tuple[List[str], Dict[str, paddle.Tensor]]:
        return mlm_collate_fn(
            data,
            float_pad_value=self.float_pad_value,
            int_pad_value=self.int_pad_value,
            not_sequence=self.not_sequence,
            mlm_prob=self.mlm_prob, 
            mean_phn_span=self.mean_phn_span,
            feats_extract=self.feats_extract,
            attention_window=self.attention_window,
            pad_speech=self.pad_speech,
            sega_emb=self.sega_emb,
            duration_collect=self.duration_collect,
            text_masking=self.text_masking
        )

def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [paddle.ones(4), paddle.ones(2), paddle.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(paddle.shape(x)[0] for x in xs)
    pad = paddle.full((n_batch, max_len), pad_value, dtype = xs[0].dtype)

    for i in range(n_batch):
        pad[i, : paddle.shape(xs[i])[0]] = xs[i]
    
    return pad

def pad_to_longformer_att_window(text, max_len, max_tlen,attention_window):
    round = max_len % attention_window
    if round != 0:
        max_tlen += (attention_window - round)
        n_batch = paddle.shape(text)[0]
        text_pad = paddle.zeros((n_batch, max_tlen, *paddle.shape(text[0])[1:]), dtype=text.dtype)
        for i in range(n_batch):
            text_pad[i, : paddle.shape(text[i])[0]] = text[i]
    else:
        text_pad = text[:, : max_tlen]
    return text_pad, max_tlen

def make_pad_mask(lengths, xs=None, length_dim=-1):
    print('inputs are:', lengths, xs, length_dim)
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = paddle.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=paddle.uint8)
        >>> xs = paddle.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=paddle.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = paddle.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=paddle.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=paddle.uint8)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = list(lengths)
    print('lengths', lengths)
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = paddle.shape(xs)[length_dim]

    seq_range = paddle.arange(0, maxlen, dtype=paddle.int64)
    seq_range_expand = paddle.expand(paddle.unsqueeze(seq_range, 0), (bs, maxlen))
    seq_length_expand = paddle.unsqueeze(paddle.to_tensor(lengths), -1)
    print('seq_length_expand', paddle.shape(seq_length_expand))
    print('seq_range_expand', paddle.shape(seq_range_expand))
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert paddle.shape(xs)[0] == bs, (paddle.shape(xs)[0], bs)

        if length_dim < 0:
            length_dim = len(paddle.shape(xs)) + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(len(paddle.shape(xs)))
        )
        print('0:', paddle.shape(mask))
        print('1:', paddle.shape(mask[ind]))
        print('2:', paddle.shape(xs))
        mask = paddle.expand(mask[ind], paddle.shape(xs))
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]

        With the reference tensor.

        >>> xs = paddle.zeros((3, 2, 4))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 1, 0],
                 [1, 1, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]]], dtype=paddle.uint8)
        >>> xs = paddle.zeros((3, 2, 6))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=paddle.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = paddle.zeros((3, 6, 6))
        >>> make_non_pad_mask(lengths, xs, 1)
        tensor([[[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]], dtype=paddle.uint8)
        >>> make_non_pad_mask(lengths, xs, 2)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=paddle.uint8)

    """
    return ~make_pad_mask(lengths, xs, length_dim)

def phones_masking(xs_pad, src_mask, align_start, align_end, align_start_lengths, mlm_prob, mean_phn_span, span_boundary=None):
    bz, sent_len, _ = paddle.shape(xs_pad)
    mask_num_lower = math.ceil(sent_len * mlm_prob)
    masked_position = np.zeros((bz, sent_len))
    y_masks = None
    # y_masks = torch.ones(bz,sent_len,sent_len,device=xs_pad.device,dtype=xs_pad.dtype)
    # tril_masks = torch.tril(y_masks)
    if mlm_prob == 1.0:
        masked_position += 1
        # y_masks = tril_masks
    elif mean_phn_span == 0:
        # only speech 
        length = sent_len
        mean_phn_span = min(length*mlm_prob//3, 50)
        masked_phn_indices = random_spans_noise_mask(length,mlm_prob, mean_phn_span).nonzero()
        masked_position[:,masked_phn_indices]=1
    else:
        for idx in range(bz):
            if span_boundary is not None:
                for s,e in zip(span_boundary[idx][::2], span_boundary[idx][1::2]):
                    masked_position[idx, s:e] = 1

                    # y_masks[idx, :, s:e] = tril_masks[idx, :, s:e]
                    # y_masks[idx, e:, s:e ] = 0
            else:
                length = align_start_lengths[idx].item()
                if length<2:
                    continue
                masked_phn_indices = random_spans_noise_mask(length,mlm_prob, mean_phn_span).nonzero()
                masked_start = align_start[idx][masked_phn_indices].tolist()
                masked_end = align_end[idx][masked_phn_indices].tolist()
                for s,e in zip(masked_start, masked_end):
                    masked_position[idx, s:e] = 1
                    # y_masks[idx, :, s:e] = tril_masks[idx, :, s:e]
                    # y_masks[idx, e:, s:e ] = 0
    non_eos_mask = np.array(paddle.reshape(src_mask, paddle.shape(xs_pad)[:2]).float().cpu())
    masked_position = masked_position * non_eos_mask
    # y_masks = src_mask & y_masks.bool()

    return paddle.cast(paddle.to_tensor(masked_position), paddle.bool), y_masks

def get_segment_pos(speech_pad, text_pad, align_start, align_end, align_start_lengths,sega_emb):
    bz, speech_len, _ = speech_pad.size()
    text_segment_pos = paddle.zeros_like(text_pad)
    speech_segment_pos = paddle.zeros((bz, speech_len),dtype=text_pad.dtype)
    if not sega_emb:
        return speech_segment_pos, text_segment_pos
    for idx in range(bz):
        align_length = align_start_lengths[idx].item()
        for j in range(align_length):
            s,e = align_start[idx][j].item(), align_end[idx][j].item()
            speech_segment_pos[idx][s:e] = j+1
            text_segment_pos[idx][j] = j+1
        
    return speech_segment_pos, text_segment_pos

def mlm_collate_fn(
    data: Collection[Tuple[str, Dict[str, np.ndarray]]],
    float_pad_value: Union[float, int] = 0.0,
    int_pad_value: int = -32768,
    not_sequence: Collection[str] = (),
    mlm_prob: float = 0.8, 
    mean_phn_span: int = 8,
    feats_extract=None,
    attention_window: int = 0,
    pad_speech: bool=False,
    sega_emb: bool=False,
    duration_collect: bool=False,
    text_masking: bool=False
) -> Tuple[List[str], Dict[str, paddle.Tensor]]:
    """Concatenate ndarray-list to an array and convert to paddle.Tensor.

    Examples:
        >>> from espnet2.samplers.constant_batch_sampler import ConstantBatchSampler,
        >>> import espnet2.tasks.abs_task
        >>> from espnet2.train.dataset import ESPnetDataset
        >>> sampler = ConstantBatchSampler(...)
        >>> dataset = ESPnetDataset(...)
        >>> keys = next(iter(sampler)
        >>> batch = [dataset[key] for key in keys]
        >>> batch = common_collate_fn(batch)
        >>> model(**batch)

        Note that the dict-keys of batch are propagated from
        that of the dataset as they are.

    """
    uttids = [u for u, _ in data]
    data = [d for _, d in data]

    assert all(set(data[0]) == set(d) for d in data), "dict-keys mismatching"
    assert all(
        not k.endswith("_lengths") for k in data[0]
    ), f"*_lengths is reserved: {list(data[0])}"

    output = {}
    for key in data[0]:
        # NOTE(kamo):
        # Each models, which accepts these values finally, are responsible
        # to repaint the pad_value to the desired value for each tasks.
        if data[0][key].dtype.kind == "i":
            pad_value = int_pad_value
        else:
            pad_value = float_pad_value

        array_list = [d[key] for d in data]

        # Assume the first axis is length:
        # tensor_list: Batch x (Length, ...)
        tensor_list = [paddle.to_tensor(a) for a in array_list]
        # tensor: (Batch, Length, ...)
        tensor = pad_list(tensor_list, pad_value)
        output[key] = tensor

        # lens: (Batch,)
        if key not in not_sequence:
            lens = paddle.to_tensor([d[key].shape[0] for d in data], dtype=paddle.long)
            output[key + "_lengths"] = lens

    f = open('tmp_var.out', 'w')
    for item in [round(item, 6) for item in output["speech"][0].tolist()]:
        f.write(str(item)+'\n')
    feats = feats_extract.get_log_mel_fbank(np.array(output["speech"][0]))
    feats = paddle.to_tensor(feats)
    print('out shape', paddle.shape(feats))
    feats_lengths = paddle.shape(feats)[0]
    feats = paddle.unsqueeze(feats, 0)
    batch_size = paddle.shape(feats)[0]
    if 'text' not in output:
        text=paddle.zeros_like(feats_lengths.unsqueeze(-1))-2
        text_lengths=paddle.zeros_like(feats_lengths)+1
        max_tlen=1
        align_start=paddle.zeros_like(text)
        align_end=paddle.zeros_like(text)
        align_start_lengths=paddle.zeros_like(feats_lengths)
        align_end_lengths=paddle.zeros_like(feats_lengths)
        sega_emb=False
        mean_phn_span = 0
        mlm_prob = 0.15
    else:
        text, text_lengths = output["text"], output["text_lengths"]
        align_start, align_start_lengths, align_end, align_end_lengths = output["align_start"], output["align_start_lengths"], output["align_end"], output["align_end_lengths"]
        align_start = paddle.floor(feats_extract.sr*align_start/feats_extract.hop_length).int()
        align_end = paddle.floor(feats_extract.sr*align_end/feats_extract.hop_length).int()
        max_tlen = max(text_lengths).item()
    max_slen = max(feats_lengths).item()
    speech_pad = feats[:, : max_slen]
    if attention_window>0 and pad_speech:
        speech_pad,max_slen = pad_to_longformer_att_window(speech_pad, max_slen, max_slen, attention_window)
    max_len = max_slen + max_tlen
    if attention_window>0:
        text_pad, max_tlen = pad_to_longformer_att_window(text, max_len, max_tlen, attention_window)
    else:
        text_pad = text
    text_mask = make_non_pad_mask(text_lengths.tolist(), text_pad, length_dim=1).unsqueeze(-2)
    if attention_window>0:
        text_mask = text_mask*2 
    speech_mask = make_non_pad_mask(feats_lengths.tolist(), speech_pad[:,:,0], length_dim=1).unsqueeze(-2)
    span_boundary = None
    if 'span_boundary' in output.keys():
        span_boundary = output['span_boundary']

    if text_masking:
        masked_position, text_masked_position,_ = phones_text_masking(
            speech_pad,
            speech_mask,
            text_pad, 
            text_mask,
            align_start,
            align_end,
            align_start_lengths,
            mlm_prob,
            mean_phn_span,
            span_boundary)
    else:
        text_masked_position = np.zeros(text_pad.size())
        masked_position, _ = phones_masking(
                speech_pad,
                speech_mask,
                align_start,
                align_end,
                align_start_lengths,
                mlm_prob,
                mean_phn_span,
                span_boundary)

    output_dict = {}
    if duration_collect and 'text' in output:
        reordered_index, speech_segment_pos,text_segment_pos, durations,feats_lengths = get_segment_pos_reduce_duration(speech_pad, text_pad, align_start, align_end, align_start_lengths,sega_emb, masked_position, feats_lengths)
        speech_mask = make_non_pad_mask(feats_lengths.tolist(), speech_pad[:,:reordered_index.shape[1],0], length_dim=1).unsqueeze(-2)
        output_dict['durations'] = durations
        output_dict['reordered_index'] = reordered_index
    else:
        speech_segment_pos, text_segment_pos = get_segment_pos(speech_pad, text_pad, align_start, align_end, align_start_lengths,sega_emb)
    output_dict['speech'] = speech_pad
    output_dict['text'] = text_pad
    output_dict['masked_position'] = masked_position
    output_dict['text_masked_position'] = text_masked_position
    output_dict['speech_mask'] = speech_mask
    output_dict['text_mask'] = text_mask
    output_dict['speech_segment_pos'] = speech_segment_pos
    output_dict['text_segment_pos'] = text_segment_pos
    # output_dict['y_masks'] = y_masks
    output_dict['speech_lengths'] = output["speech_lengths"]
    output_dict['text_lengths'] = text_lengths
    output = (uttids, output_dict)
    # assert check_return_type(output)
    return output

def build_collate_fn(
        args: argparse.Namespace, train: bool, epoch=-1
    ):

    # assert check_argument_types()
    # return CommonCollateFn(float_pad_value=0.0, int_pad_value=0)
    feats_extract_class = LogMelFBank
    args_dic = {}
    print ('type is', type(args.feats_extract_conf))
    for k, v in args.feats_extract_conf.items():
        if k == 'fs':
            args_dic['sr'] = v
        else:
            args_dic[k] = v
    # feats_extract = feats_extract_class(**args.feats_extract_conf)
    feats_extract = feats_extract_class(**args_dic)

    sega_emb = True if args.encoder_conf['input_layer'] == 'sega_mlm' else False
    if args.encoder_conf['selfattention_layer_type'] == 'longformer':
        attention_window = args.encoder_conf['attention_window']
        pad_speech = True if 'pre_speech_layer' in args.encoder_conf and args.encoder_conf['pre_speech_layer'] >0 else False
    else:
        attention_window=0
        pad_speech=False
    if epoch==-1:
        mlm_prob_factor = 1
    else:
        mlm_probs = [1.0, 1.0, 0.7, 0.6, 0.5]
        mlm_prob_factor = 0.8 #mlm_probs[epoch // 100]
    if 'duration_predictor_layers' in args.model_conf.keys() and args.model_conf['duration_predictor_layers']>0:
        duration_collect=True
    else:
        duration_collect=False
    return MLMCollateFn(feats_extract, float_pad_value=0.0, int_pad_value=0,
    mlm_prob=args.model_conf['mlm_prob']*mlm_prob_factor,mean_phn_span=args.model_conf['mean_phn_span'],attention_window=attention_window,pad_speech=pad_speech,sega_emb=sega_emb,duration_collect=duration_collect)


def get_mlm_output(uid, prefix, clone_uid, clone_prefix, source_language, target_language, model_name, wav_path, old_str, new_str,duration_preditor_path, sid=None, decoder=False,use_teacher_forcing=False, dynamic_eval=(0,0),duration_adjust=True,start_end_sp=False):
    mlm_model,train_args = load_model(model_name)
    mlm_model.eval()
    # processor = MLMTask.build_preprocess_fn(train_args, False)
    processor = None
    collate_fn = MLMTask.build_collate_fn(train_args, False)
    # collate_fn = build_collate_fn(train_args, False)

    return decode_with_model(uid,prefix, clone_uid, clone_prefix, source_language, target_language, mlm_model, processor, collate_fn, wav_path, old_str, new_str,duration_preditor_path, sid=sid, decoder=decoder, use_teacher_forcing=use_teacher_forcing,
    duration_adjust=duration_adjust,start_end_sp=start_end_sp, train_args = train_args)

def prompt_decoding_fn(model_name, wav_path,full_origin_str, old_str, new_str, vocoder,duration_preditor_path,sid=None, non_autoreg=True, dynamic_eval=(0,0),duration_adjust=True):
    wav_org, input_feat, output_feat, old_span_boundary, new_span_boundary, fs, hop_length = get_mlm_output(
                                                            model_name,
                                                            wav_path,
                                                            old_str,
                                                            new_str, 
                                                            duration_preditor_path,
                                                            use_teacher_forcing=non_autoreg,
                                                            sid=sid,
                                                            dynamic_eval=dynamic_eval,
                                                            duration_adjust=duration_adjust,
                                                            start_end_sp=False
                                                            )

    replaced_wav = vocoder(output_feat).detach().float().data.cpu().numpy()

    old_time_boundary = [hop_length * x  for x in old_span_boundary]
    new_time_boundary = [hop_length * x  for x in new_span_boundary]
    new_wav = replaced_wav[new_time_boundary[0]:]
    # "origin_vocoder":vocoder_origin_wav, 
    data_dict = {"prompt":wav_org,
                "new_wav":new_wav}
    return data_dict

def test_vctk(uid, clone_uid, clone_prefix, source_language, target_language, vocoder, prefix='dump/raw/dev', model_name="conformer", old_str="",new_str="",prompt_decoding=False,dynamic_eval=(0,0), task_name = None):

    new_str = new_str.strip()
    if clone_uid is not None and clone_prefix is not None:
        if target_language == "english":
            duration_preditor_path = duration_path_dict['ljspeech']
        elif target_language == "chinese":    
            duration_preditor_path = duration_path_dict['ljspeech'] 
        else:
            assert target_language == "chinese" or target_language == "english", "duration_preditor_path is not support for this language..."
    
    else:           
        duration_preditor_path = duration_path_dict['ljspeech']

    spemd = None
    full_origin_str,wav_path = read_data(uid, prefix)

    new_str = new_str if task_name == 'edit' else full_origin_str + new_str 
    print('new_str is ', new_str)
    
    if not old_str:
        old_str = full_origin_str
    if not new_str:
        new_str = input("input the new string:")
    if prompt_decoding:
        print(new_str)
        return prompt_decoding_fn(model_name, wav_path,full_origin_str, old_str, new_str,vocoder,duration_preditor_path,sid=spemd,dynamic_eval=dynamic_eval)
    print(full_origin_str)
    results_dict, old_span = plot_mel_and_vocode_wav(uid, prefix, clone_uid, clone_prefix, source_language, target_language, model_name, wav_path,full_origin_str, old_str, new_str,vocoder,duration_preditor_path,sid=spemd)
    return results_dict

if __name__ == "__main__":
    args = parse_args()
    print(args)
    data_dict = test_vctk(args.uid, 
        args.clone_uid, 
        args.clone_prefix, 
        args.source_language, 
        args.target_language, 
        None,
        args.prefix, 
        args.model_name,
        new_str=args.new_str,
        task_name=args.task_name)
    sf.write('./wavs/%s' % args.output_name, data_dict['output'], samplerate=24000)
    