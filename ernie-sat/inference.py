#!/usr/bin/env python3

import os 
from pathlib import Path
import paddle
import math
import string
import numpy as np

from read_text import read_2column_text,load_num_sequence_text
from utils import sentence2phns,get_voc_out, evaluate_durations
import librosa
import random
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
        replaced_wav = replaced_wav_paddle_voc

    elif target_language == 'chinese':
        assert old_span_boundary[1] == new_span_boundary[0], "old_span_boundary[1] is not same with new_span_boundary[0]." 
        output_feat_np = output_feat.detach().float().cpu().numpy()
        replaced_wav = get_voc_out(output_feat_np)

        replaced_wav_only_mask = get_voc_out(masked_feat)
        replaced_wav_only_mask_fst2_voc = get_voc_out(masked_feat, target_language)
   

    old_time_boundary = [hop_length * x  for x in old_span_boundary]
    new_time_boundary = [hop_length * x  for x in new_span_boundary]
    
    wav_org_replaced = np.concatenate([wav_org[:old_time_boundary[0]], replaced_wav[new_time_boundary[0]:new_time_boundary[1]], wav_org[old_time_boundary[1]:]])
    
    if target_language == 'english':
        # new add to test paddle vocoder
        wav_org_replaced_paddle_voc = np.concatenate([wav_org[:old_time_boundary[0]], replaced_wav_paddle_voc[new_time_boundary[0]:new_time_boundary[1]], wav_org[old_time_boundary[1]:]])

        data_dict = {
                    "origin":wav_org,
                    "output":wav_org_replaced_paddle_voc}

    elif  target_language == 'chinese':
        wav_org_replaced_only_mask = np.concatenate([wav_org[:old_time_boundary[0]], replaced_wav_only_mask, wav_org[old_time_boundary[1]:]])
        wav_org_replaced_only_mask_fst2_voc = np.concatenate([wav_org[:old_time_boundary[0]], replaced_wav_only_mask_fst2_voc, wav_org[old_time_boundary[1]:]])
        data_dict = {
                    "origin":wav_org,
                    "output": wav_org_replaced_only_mask_fst2_voc,}
    
    return data_dict, old_span_boundary


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


def gen_phns(zh_mapping, phns):
    new_phns = []
    for x in phns:
        if x in zh_mapping.keys():
            new_phns.extend(zh_mapping[x].split(" "))
        else:
            new_phns.extend(['<unk>'])
    return new_phns
    

def get_mapping(phn_mapping="./phn_mapping.txt"):
    zh_mapping = {}
    with open(phn_mapping, "r") as f:
        for line in f:
            pd_phn = line.split(" ")[0]
            if pd_phn not in zh_mapping.keys():
                zh_mapping[pd_phn] = " ".join(line.split()[1:])
    return zh_mapping


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
    
    feats = pickle.load(open('tmp/tmp_pkl.'+str(uid), 'rb'))
    tmp = feats['speech'][0]
    
    # print('feats end')
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

    wav_org, rate = librosa.load(wav_path, sr=train_args.feats_extract_conf['fs'])
    origin_speech = paddle.to_tensor(np.array(wav_org,dtype=np.float32)).unsqueeze(0)
    speech_lengths = paddle.to_tensor(len(wav_org)).unsqueeze(0)
    return wav_org, None, output_feat, old_span_boundary, new_span_boundary, fs, hop_length


def get_mlm_output(uid, prefix, clone_uid, clone_prefix, source_language, target_language, model_name, wav_path, old_str, new_str,duration_preditor_path, sid=None, decoder=False,use_teacher_forcing=False, dynamic_eval=(0,0),duration_adjust=True,start_end_sp=False):
    mlm_model,train_args = load_model(model_name)
    mlm_model.eval()
    processor = None
    collate_fn = None

    return decode_with_model(uid,prefix, clone_uid, clone_prefix, source_language, target_language, mlm_model, processor, collate_fn, wav_path, old_str, new_str,duration_preditor_path, sid=sid, decoder=decoder, use_teacher_forcing=use_teacher_forcing,
    duration_adjust=duration_adjust,start_end_sp=start_end_sp, train_args = train_args)

def test_vctk(uid, clone_uid, clone_prefix, source_language, target_language, vocoder, prefix='dump/raw/dev', model_name="conformer", old_str="",new_str="",prompt_decoding=False,dynamic_eval=(0,0), task_name = None):

    duration_preditor_path = None
    spemd = None
    full_origin_str,wav_path = read_data(uid, prefix)

    new_str = new_str if task_name == 'edit' else full_origin_str + new_str 
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
        None,
        args.prefix, 
        args.model_name,
        new_str=args.new_str,
        task_name=args.task_name)
    sf.write('./wavs/%s' % args.output_name, data_dict['output'], samplerate=24000)
    # exit()
