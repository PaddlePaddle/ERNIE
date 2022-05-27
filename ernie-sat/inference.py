#!/usr/bin/env python3

import os 
from pathlib import Path
import librosa
import random
import soundfile as sf
import sys 
import pickle
import argparse
from typing import Collection
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
import paddle
import torch
import math
import string
import numpy as np
from ParallelWaveGAN.parallel_wavegan.utils.utils import download_pretrained_model
from read_text import read_2column_text,load_num_sequence_text
from utils import sentence2phns,get_voc_out, evaluate_durations, is_chinese, build_vocoder_from_file
from model_paddle import build_model_from_file

from sedit_arg_parser import parse_args
from paddlespeech.t2s.datasets.get_feats import LogMelFBank
from dataset import pad_list, pad_to_longformer_att_window, make_pad_mask, make_non_pad_mask, phones_masking, get_segment_pos
from align_english import alignment
from align_mandarin import alignment_zh
from ParallelWaveGAN.parallel_wavegan.utils.utils import download_pretrained_model
random.seed(0)
np.random.seed(0)


PHONEME = 'tools/aligner/english_envir/english2phoneme/phoneme'
MODEL_DIR_EN = 'tools/aligner/english'
MODEL_DIR_ZH = 'tools/aligner/mandarin'


def plot_mel_and_vocode_wav(uid, prefix, clone_uid, clone_prefix, source_language, target_language, model_name, wav_path,full_origin_str, old_str, new_str, use_pt_vocoder, duration_preditor_path,sid=None, non_autoreg=True):
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
        if use_pt_vocoder:
            output_feat = output_feat.detach().float().cpu().numpy()
            output_feat = torch.tensor(output_feat,dtype=torch.float)
            vocoder = load_vocoder('vctk_parallel_wavegan.v1.long')
            replaced_wav = vocoder(output_feat).detach().float().data.cpu().numpy()
        else:
            output_feat_np = output_feat.detach().float().cpu().numpy()
            replaced_wav = get_voc_out(output_feat_np, target_language)

    elif target_language == 'chinese':
        output_feat_np = output_feat.detach().float().cpu().numpy()
        replaced_wav_only_mask_fst2_voc = get_voc_out(masked_feat, target_language)
   
    old_time_boundary = [hop_length * x  for x in old_span_boundary]
    new_time_boundary = [hop_length * x  for x in new_span_boundary]
    
    if target_language == 'english':
        wav_org_replaced_paddle_voc = np.concatenate([wav_org[:old_time_boundary[0]], replaced_wav[new_time_boundary[0]:new_time_boundary[1]], wav_org[old_time_boundary[1]:]])

        data_dict = {
                    "origin":wav_org,
                    "output":wav_org_replaced_paddle_voc}

    elif  target_language == 'chinese':
        wav_org_replaced_only_mask_fst2_voc = np.concatenate([wav_org[:old_time_boundary[0]], replaced_wav_only_mask_fst2_voc, wav_org[old_time_boundary[1]:]])
        data_dict = {
                    "origin":wav_org,
                    "output": wav_org_replaced_only_mask_fst2_voc,}
    
    return data_dict, old_span_boundary



def get_unk_phns(word_str):
    tmpbase = '/tmp/tp.'
    f = open(tmpbase + 'temp.words', 'w')
    f.write(word_str)
    f.close()
    os.system(PHONEME + ' ' + tmpbase + 'temp.words' + ' ' + tmpbase + 'temp.phons')
    f = open(tmpbase + 'temp.phons', 'r')
    lines2 = f.readline().strip().split()
    f.close()
    phns = []
    for phn in lines2:
        phons = phn.replace('\n', '').replace(' ', '')
        seq = []
        j = 0
        while (j < len(phons)):
            if (phons[j] > 'Z'):
                if (phons[j] == 'j'):
                    seq.append('JH')
                elif (phons[j] == 'h'):
                    seq.append('HH')
                else:
                    seq.append(phons[j].upper())
                j += 1
            else:
                p = phons[j:j+2]
                if (p == 'WH'):
                    seq.append('W')
                elif (p in ['TH', 'SH', 'HH', 'DH', 'CH', 'ZH', 'NG']):
                    seq.append(p)
                elif (p == 'AX'):
                    seq.append('AH0')
                else:
                    seq.append(p + '1')
                j += 2
        phns.extend(seq)
    return phns

def words2phns(line):
    dictfile = MODEL_DIR_EN+'/dict'
    tmpbase = '/tmp/tp.'
    line = line.strip()
    words = []
    for pun in [',', '.', ':', ';', '!', '?', '"', '(', ')', '--', '---']:
        line = line.replace(pun, ' ')
    for wrd in line.split():
        if (wrd[-1] == '-'):
            wrd = wrd[:-1]
        if (wrd[0] == "'"):
            wrd = wrd[1:]
        if wrd:
            words.append(wrd)
    ds = set([])
    word2phns_dict = {}
    with open(dictfile, 'r') as fid:
        for line in fid:
            word = line.split()[0]
            ds.add(word)
            if word not in word2phns_dict.keys():
                word2phns_dict[word] = " ".join(line.split()[1:])
    
    phns = []
    wrd2phns = {}
    for index, wrd in enumerate(words):
        if wrd == '[MASK]':
            wrd2phns[str(index)+"_"+wrd] = [wrd]
            phns.append(wrd)
        elif (wrd.upper() not in ds):
            wrd2phns[str(index)+"_"+wrd.upper()] = get_unk_phns(wrd)
            phns.extend(get_unk_phns(wrd))
        else:
            wrd2phns[str(index)+"_"+wrd.upper()] = word2phns_dict[wrd.upper()].split()
            phns.extend(word2phns_dict[wrd.upper()].split())

    return phns, wrd2phns



def words2phns_zh(line):
    dictfile = MODEL_DIR_ZH+'/dict'
    tmpbase = '/tmp/tp.'
    line = line.strip()
    words = []
    for pun in [',', '.', ':', ';', '!', '?', '"', '(', ')', '--', '---', u'，', u'。', u'：', u'；', u'！', u'？', u'（', u'）']:
        line = line.replace(pun, ' ')
    for wrd in line.split():
        if (wrd[-1] == '-'):
            wrd = wrd[:-1]
        if (wrd[0] == "'"):
            wrd = wrd[1:]
        if wrd:
            words.append(wrd)
    
    ds = set([])
    word2phns_dict = {}
    with open(dictfile, 'r') as fid:
        for line in fid:
            word = line.split()[0]
            ds.add(word)
            if word not in word2phns_dict.keys():
                word2phns_dict[word] = " ".join(line.split()[1:])
    
    phns = []
    wrd2phns = {}
    for index, wrd in enumerate(words):
        if wrd == '[MASK]':
            wrd2phns[str(index)+"_"+wrd] = [wrd]
            phns.append(wrd)
        elif (wrd.upper() not in ds):
            print("出现非法词错误,请输入正确的文本...")
        else:
            wrd2phns[str(index)+"_"+wrd] = word2phns_dict[wrd].split()
            phns.extend(word2phns_dict[wrd].split())

    return phns, wrd2phns


def load_vocoder(vocoder_tag="vctk_parallel_wavegan.v1.long"):
    vocoder_tag = vocoder_tag.replace("parallel_wavegan/", "")
    vocoder_file = download_pretrained_model(vocoder_tag)
    vocoder_config = Path(vocoder_file).parent / "config.yml"
    vocoder = build_vocoder_from_file(
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



def recover_dict(word2phns, tp_word2phns):
    dic = {}
    need_del_key = []
    exist_index = []  
    sp_count = 0  
    add_sp_count = 0 
    for key in word2phns.keys():
        idx, wrd = key.split('_')
        if wrd == 'sp':
            sp_count += 1 
            exist_index.append(int(idx))
        else:
            need_del_key.append(key)
    
    for key in need_del_key:
        del word2phns[key]

    cur_id = 0
    for key in tp_word2phns.keys():
        # print("debug: ",key)
        if cur_id in exist_index:
            dic[str(cur_id)+"_sp"] = 'sp'
            cur_id += 1 
            add_sp_count += 1 
        idx, wrd = key.split('_')
        dic[str(cur_id)+"_"+wrd] = tp_word2phns[key]
        cur_id += 1
    
    if add_sp_count + 1 == sp_count:
        dic[str(cur_id)+"_sp"] = 'sp'
        add_sp_count += 1 
    
    assert add_sp_count == sp_count, "sp are not added in dic"
    return dic


def get_phns_and_spans(wav_path, old_str, new_str, source_language, clone_target_language):
    append_new_str = (old_str == new_str[:len(old_str)])
    old_phns, mfa_start, mfa_end = [], [], []

    if source_language == "english":
        times2,word2phns = alignment(wav_path, old_str)
    elif source_language == "chinese":
        times2,word2phns = alignment_zh(wav_path, old_str)
        _, tp_word2phns = words2phns_zh(old_str)  
    
        for key,value in tp_word2phns.items(): 
            idx, wrd = key.split('_')
            cur_val = " ".join(value)
            tp_word2phns[key] = cur_val             

        word2phns = recover_dict(word2phns, tp_word2phns)

    else:
        assert source_language == "chinese" or source_language == "english", "source_language is wrong..."

    for item in times2:
        mfa_start.append(float(item[1]))
        mfa_end.append(float(item[2]))
        old_phns.append(item[0])


    if append_new_str and (source_language != clone_target_language):
        is_cross_lingual_clone = True 
    else:
        is_cross_lingual_clone = False

    if is_cross_lingual_clone:
        new_str_origin = new_str[:len(old_str)]
        new_str_append = new_str[len(old_str):]

        if clone_target_language == "chinese":
            new_phns_origin,new_origin_word2phns = words2phns(new_str_origin)
            new_phns_append,temp_new_append_word2phns = words2phns_zh(new_str_append) 

        elif clone_target_language == "english":
            new_phns_origin,new_origin_word2phns = words2phns_zh(new_str_origin)    # 原始句子
            new_phns_append,temp_new_append_word2phns = words2phns(new_str_append)  # clone句子
        else:
            assert clone_target_language == "chinese" or clone_target_language == "english", "cloning is not support for this language, please check it."
        
        new_phns = new_phns_origin + new_phns_append

        new_append_word2phns = {}
        length = len(new_origin_word2phns)
        for key,value in temp_new_append_word2phns.items():
            idx, wrd = key.split('_')
            new_append_word2phns[str(int(idx)+length)+'_'+wrd] = value
  
        new_word2phns = dict(list(new_origin_word2phns.items()) + list(new_append_word2phns.items())) 

    else:  
        if source_language == clone_target_language and clone_target_language == "english":
            new_phns, new_word2phns = words2phns(new_str)
        elif source_language == clone_target_language and clone_target_language == "chinese":
            new_phns, new_word2phns = words2phns_zh(new_str)
        else:
            assert source_language == clone_target_language, "source language is not same with target language..."
    
    span_tobe_replaced = [0,len(old_phns)-1]
    span_tobe_added = [0,len(new_phns)-1]
    left_index = 0
    new_phns_left = []
    sp_count = 0
    # find the left different index
    for key in word2phns.keys():
        idx, wrd = key.split('_')
        if wrd=='sp':
            sp_count +=1
            new_phns_left.append('sp')
        else:
            idx = str(int(idx) - sp_count)
            if idx+'_'+wrd in new_word2phns:
                left_index+=len(new_word2phns[idx+'_'+wrd])
                new_phns_left.extend(word2phns[key].split())
            else:
                span_tobe_replaced[0] = len(new_phns_left)
                span_tobe_added[0] = len(new_phns_left)
                break
    
    # reverse word2phns and new_word2phns
    right_index = 0
    new_phns_right = []
    sp_count = 0
    word2phns_max_index = int(list(word2phns.keys())[-1].split('_')[0])
    new_word2phns_max_index = int(list(new_word2phns.keys())[-1].split('_')[0])
    new_phns_middle = []
    if append_new_str:   
        new_phns_right = []
        new_phns_middle = new_phns[left_index:]
        span_tobe_replaced[0] = len(new_phns_left)
        span_tobe_added[0] = len(new_phns_left)
        span_tobe_added[1] = len(new_phns_left) + len(new_phns_middle)
        span_tobe_replaced[1] = len(old_phns) - len(new_phns_right)
    else:
        for key in list(word2phns.keys())[::-1]:
            idx, wrd = key.split('_')
            if wrd=='sp':
                sp_count +=1
                new_phns_right = ['sp']+new_phns_right
            else:
                idx = str(new_word2phns_max_index-(word2phns_max_index-int(idx)-sp_count))
                if idx+'_'+wrd in new_word2phns:
                    right_index-=len(new_word2phns[idx+'_'+wrd])
                    new_phns_right = word2phns[key].split() + new_phns_right
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
    
    mfa_start, mfa_end, old_phns, new_phns, span_tobe_replaced, span_tobe_added = get_phns_and_spans(wav_path, old_str, new_str, source_language, target_language)

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
            d_factor = d_factor * 1.25 
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
    # print('unk id is', token_to_id['<unk>'])
    # text = np.array(processor(uid='1', data={'text':" ".join(phns_list)})['text'])
    span_boundary = np.array(new_span_boundary)
    batch=[('1', {"speech":speech,"align_start":align_start,"align_end":align_end,"text":text,"span_boundary":span_boundary})]
    
    return batch, old_span_boundary, new_span_boundary

def decode_with_model(uid, prefix, clone_uid, clone_prefix, source_language, target_language, mlm_model, processor, collate_fn, wav_path, old_str, new_str,duration_preditor_path, sid=None, decoder=False,use_teacher_forcing=False,duration_adjust=True,start_end_sp=False, train_args=None):
    fs, hop_length = train_args.feats_extract_conf['fs'], train_args.feats_extract_conf['hop_length']

    batch,old_span_boundary,new_span_boundary = prepare_features(uid,prefix, clone_uid, clone_prefix, source_language, target_language, mlm_model,processor,wav_path,old_str,new_str,duration_preditor_path, sid,duration_adjust=duration_adjust,start_end_sp=start_end_sp, train_args=train_args)
    
    feats = collate_fn(batch)[1]

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

    wav_org, rate = librosa.load(wav_path, sr=train_args.feats_extract_conf['fs'])
    origin_speech = paddle.to_tensor(np.array(wav_org,dtype=np.float32)).unsqueeze(0)
    speech_lengths = paddle.to_tensor(len(wav_org)).unsqueeze(0)
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
    """Concatenate ndarray-list to an array and convert to torch.Tensor.

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

    feats = feats_extract.get_log_mel_fbank(np.array(output["speech"][0]))
    feats = paddle.to_tensor(feats)
    # print('out shape', paddle.shape(feats))
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
    output_dict['speech_lengths'] = output["speech_lengths"]
    output_dict['text_lengths'] = text_lengths
    output = (uttids, output_dict)
    return output

def build_collate_fn(
        args: argparse.Namespace, train: bool, epoch=-1
    ):
    # -> Callable[
    #     [Collection[Tuple[str, Dict[str, np.ndarray]]]],
    #     Tuple[List[str], Dict[str, torch.Tensor]],
    # ]:

    # assert check_argument_types()
    # return CommonCollateFn(float_pad_value=0.0, int_pad_value=0)
    feats_extract_class = LogMelFBank
    if args.feats_extract_conf['win_length'] == None:
        args.feats_extract_conf['win_length'] = args.feats_extract_conf['n_fft']

    args_dic = {}
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
    processor = None
    collate_fn = build_collate_fn(train_args, False)

    return decode_with_model(uid,prefix, clone_uid, clone_prefix, source_language, target_language, mlm_model, processor, collate_fn, wav_path, old_str, new_str,duration_preditor_path, sid=sid, decoder=decoder, use_teacher_forcing=use_teacher_forcing,
    duration_adjust=duration_adjust,start_end_sp=start_end_sp, train_args = train_args)

def test_vctk(uid, clone_uid, clone_prefix, source_language, target_language, vocoder, prefix='dump/raw/dev', model_name="conformer", old_str="",new_str="",prompt_decoding=False,dynamic_eval=(0,0), task_name = None):

    duration_preditor_path = None
    spemd = None
    full_origin_str,wav_path = read_data(uid, prefix)
                    
    if task_name == 'edit':
        new_str = new_str
    elif task_name == 'synthesize':
        new_str = full_origin_str + new_str 
    else:
        new_str = full_origin_str + ' '.join([ch for ch in new_str if is_chinese(ch)])
    
    print('new_str is ', new_str)
    
    if not old_str:
        old_str = full_origin_str

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
        args.use_pt_vocoder,
        args.prefix, 
        args.model_name,
        new_str=args.new_str,
        task_name=args.task_name)
    sf.write('./wavs/%s' % args.output_name, data_dict['output'], samplerate=24000)
    print("finished...")
    # exit()

