#!/usr/bin/env python3
import argparse
import os
import random
from pathlib import Path
from typing import Collection
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import librosa
import numpy as np
import paddle
import soundfile as sf
import torch
from paddle import nn

from ParallelWaveGAN.parallel_wavegan.utils.utils import download_pretrained_model

from align import alignment
from align import alignment_zh
from dataset import get_seg_pos
from dataset import get_seg_pos_reduce_duration
from dataset import pad_to_longformer_att_window
from dataset import phones_masking
from dataset import phones_text_masking
from model_paddle import build_model_from_file
from read_text import load_num_sequence_text
from read_text import read_2column_text
from sedit_arg_parser import parse_args
from utils import build_vocoder_from_file
from utils import evaluate_durations
from utils import get_voc_out
from utils import is_chinese
from paddlespeech.t2s.datasets.get_feats import LogMelFBank
from paddlespeech.t2s.modules.nets_utils import pad_list
from paddlespeech.t2s.modules.nets_utils import make_non_pad_mask
random.seed(0)
np.random.seed(0)

PHONEME = 'tools/aligner/english_envir/english2phoneme/phoneme'
MODEL_DIR_EN = 'tools/aligner/english'
MODEL_DIR_ZH = 'tools/aligner/mandarin'


def plot_mel_and_vocode_wav(uid: str,
                            wav_path: str,
                            prefix: str="./prompt/dev/",
                            source_lang: str='english',
                            target_lang: str='english',
                            model_name: str="conformer",
                            full_origin_str: str="",
                            old_str: str="",
                            new_str: str="",
                            duration_preditor_path: str=None,
                            use_pt_vocoder: bool=False,
                            sid: str=None,
                            non_autoreg: bool=True):
    wav_org, input_feat, output_feat, old_span_bdy, new_span_bdy, fs, hop_length = get_mlm_output(
        uid=uid,
        prefix=prefix,
        source_lang=source_lang,
        target_lang=target_lang,
        model_name=model_name,
        wav_path=wav_path,
        old_str=old_str,
        new_str=new_str,
        duration_preditor_path=duration_preditor_path,
        use_teacher_forcing=non_autoreg,
        sid=sid)

    masked_feat = output_feat[new_span_bdy[0]:new_span_bdy[1]]

    if target_lang == 'english':
        if use_pt_vocoder:
            output_feat = output_feat.cpu().numpy()
            output_feat = torch.tensor(output_feat, dtype=torch.float)
            vocoder = load_vocoder('vctk_parallel_wavegan.v1.long')
            replaced_wav = vocoder(output_feat).cpu().numpy()
        else:
            replaced_wav = get_voc_out(output_feat, target_lang)

    elif target_lang == 'chinese':
        replaced_wav_only_mask_fst2_voc = get_voc_out(masked_feat, target_lang)

    old_time_bdy = [hop_length * x for x in old_span_bdy]
    new_time_bdy = [hop_length * x for x in new_span_bdy]

    if target_lang == 'english':
        wav_org_replaced_paddle_voc = np.concatenate([
            wav_org[:old_time_bdy[0]],
            replaced_wav[new_time_bdy[0]:new_time_bdy[1]],
            wav_org[old_time_bdy[1]:]
        ])

        data_dict = {"origin": wav_org, "output": wav_org_replaced_paddle_voc}

    elif target_lang == 'chinese':
        wav_org_replaced_only_mask_fst2_voc = np.concatenate([
            wav_org[:old_time_bdy[0]], replaced_wav_only_mask_fst2_voc,
            wav_org[old_time_bdy[1]:]
        ])
        data_dict = {
            "origin": wav_org,
            "output": wav_org_replaced_only_mask_fst2_voc,
        }

    return data_dict, old_span_bdy


def get_unk_phns(word_str: str):
    tmpbase = '/tmp/tp.'
    f = open(tmpbase + 'temp.words', 'w')
    f.write(word_str)
    f.close()
    os.system(PHONEME + ' ' + tmpbase + 'temp.words' + ' ' + tmpbase +
              'temp.phons')
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
                p = phons[j:j + 2]
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


def words2phns(line: str):
    dictfile = MODEL_DIR_EN + '/dict'
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
            wrd2phns[str(index) + "_" + wrd] = [wrd]
            phns.append(wrd)
        elif (wrd.upper() not in ds):
            wrd2phns[str(index) + "_" + wrd.upper()] = get_unk_phns(wrd)
            phns.extend(get_unk_phns(wrd))
        else:
            wrd2phns[str(index) +
                     "_" + wrd.upper()] = word2phns_dict[wrd.upper()].split()
            phns.extend(word2phns_dict[wrd.upper()].split())

    return phns, wrd2phns


def words2phns_zh(line: str):
    dictfile = MODEL_DIR_ZH + '/dict'
    line = line.strip()
    words = []
    for pun in [
            ',', '.', ':', ';', '!', '?', '"', '(', ')', '--', '---', u'，',
            u'。', u'：', u'；', u'！', u'？', u'（', u'）'
    ]:
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
            wrd2phns[str(index) + "_" + wrd] = [wrd]
            phns.append(wrd)
        elif (wrd.upper() not in ds):
            print("出现非法词错误,请输入正确的文本...")
        else:
            wrd2phns[str(index) + "_" + wrd] = word2phns_dict[wrd].split()
            phns.extend(word2phns_dict[wrd].split())

    return phns, wrd2phns


def load_vocoder(vocoder_tag: str="vctk_parallel_wavegan.v1.long"):
    vocoder_tag = vocoder_tag.replace("parallel_wavegan/", "")
    vocoder_file = download_pretrained_model(vocoder_tag)
    vocoder_config = Path(vocoder_file).parent / "config.yml"
    vocoder = build_vocoder_from_file(vocoder_config, vocoder_file, None, 'cpu')
    return vocoder


def load_model(model_name: str):
    config_path = './pretrained_model/{}/config.yaml'.format(model_name)
    model_path = './pretrained_model/{}/model.pdparams'.format(model_name)
    mlm_model, args = build_model_from_file(
        config_file=config_path, model_file=model_path)
    return mlm_model, args


def read_data(uid: str, prefix: str):
    mfa_text = read_2column_text(prefix + '/text')[uid]
    mfa_wav_path = read_2column_text(prefix + '/wav.scp')[uid]
    if 'mnt' not in mfa_wav_path:
        mfa_wav_path = prefix.split('dump')[0] + mfa_wav_path
    return mfa_text, mfa_wav_path


def get_align_data(uid: str, prefix: str):
    mfa_path = prefix + "mfa_"
    mfa_text = read_2column_text(mfa_path + 'text')[uid]
    mfa_start = load_num_sequence_text(
        mfa_path + 'start', loader_type='text_float')[uid]
    mfa_end = load_num_sequence_text(
        mfa_path + 'end', loader_type='text_float')[uid]
    mfa_wav_path = read_2column_text(mfa_path + 'wav.scp')[uid]
    return mfa_text, mfa_start, mfa_end, mfa_wav_path


def get_masked_mel_bdy(mfa_start: List[float],
                       mfa_end: List[float],
                       fs: int,
                       hop_length: int,
                       span_to_repl: List[List[int]]):
    align_start = paddle.to_tensor(mfa_start).unsqueeze(0)
    align_end = paddle.to_tensor(mfa_end).unsqueeze(0)
    align_start = paddle.floor(fs * align_start / hop_length).int()
    align_end = paddle.floor(fs * align_end / hop_length).int()
    if span_to_repl[0] >= len(mfa_start):
        span_bdy = [align_end[0].tolist()[-1], align_end[0].tolist()[-1]]
    else:
        span_bdy = [
            align_start[0].tolist()[span_to_repl[0]],
            align_end[0].tolist()[span_to_repl[1] - 1]
        ]
    return span_bdy


def recover_dict(word2phns: Dict[str, str], tp_word2phns: Dict[str, str]):
    dic = {}
    keys_to_del = []
    exist_idx = []
    sp_count = 0
    add_sp_count = 0
    for key in word2phns.keys():
        idx, wrd = key.split('_')
        if wrd == 'sp':
            sp_count += 1
            exist_idx.append(int(idx))
        else:
            keys_to_del.append(key)

    for key in keys_to_del:
        del word2phns[key]

    cur_id = 0
    for key in tp_word2phns.keys():
        if cur_id in exist_idx:
            dic[str(cur_id) + "_sp"] = 'sp'
            cur_id += 1
            add_sp_count += 1
        idx, wrd = key.split('_')
        dic[str(cur_id) + "_" + wrd] = tp_word2phns[key]
        cur_id += 1

    if add_sp_count + 1 == sp_count:
        dic[str(cur_id) + "_sp"] = 'sp'
        add_sp_count += 1

    assert add_sp_count == sp_count, "sp are not added in dic"
    return dic


def get_phns_and_spans(wav_path: str,
                       old_str: str="",
                       new_str: str="",
                       source_lang: str="english",
                       target_lang: str="english"):
    append_new_str = (old_str == new_str[:len(old_str)])
    old_phns, mfa_start, mfa_end = [], [], []

    if source_lang == "english":
        times2, word2phns = alignment(wav_path, old_str)
    elif source_lang == "chinese":
        times2, word2phns = alignment_zh(wav_path, old_str)
        _, tp_word2phns = words2phns_zh(old_str)

        for key, value in tp_word2phns.items():
            idx, wrd = key.split('_')
            cur_val = " ".join(value)
            tp_word2phns[key] = cur_val

        word2phns = recover_dict(word2phns, tp_word2phns)

    else:
        assert source_lang == "chinese" or source_lang == "english", "source_lang is wrong..."

    for item in times2:
        mfa_start.append(float(item[1]))
        mfa_end.append(float(item[2]))
        old_phns.append(item[0])

    if append_new_str and (source_lang != target_lang):
        is_cross_lingual_clone = True
    else:
        is_cross_lingual_clone = False

    if is_cross_lingual_clone:
        new_str_origin = new_str[:len(old_str)]
        new_str_append = new_str[len(old_str):]

        if target_lang == "chinese":
            new_phns_origin, new_origin_word2phns = words2phns(new_str_origin)
            new_phns_append, temp_new_append_word2phns = words2phns_zh(
                new_str_append)

        elif target_lang == "english":
            # 原始句子
            new_phns_origin, new_origin_word2phns = words2phns_zh(
                new_str_origin)
            # clone句子 
            new_phns_append, temp_new_append_word2phns = words2phns(
                new_str_append)
        else:
            assert target_lang == "chinese" or target_lang == "english", \
                "cloning is not support for this language, please check it."

        new_phns = new_phns_origin + new_phns_append

        new_append_word2phns = {}
        length = len(new_origin_word2phns)
        for key, value in temp_new_append_word2phns.items():
            idx, wrd = key.split('_')
            new_append_word2phns[str(int(idx) + length) + '_' + wrd] = value

        new_word2phns = dict(
            list(new_origin_word2phns.items()) + list(
                new_append_word2phns.items()))

    else:
        if source_lang == target_lang and target_lang == "english":
            new_phns, new_word2phns = words2phns(new_str)
        elif source_lang == target_lang and target_lang == "chinese":
            new_phns, new_word2phns = words2phns_zh(new_str)
        else:
            assert source_lang == target_lang, \
                "source language is not same with target language..."

    span_to_repl = [0, len(old_phns) - 1]
    span_to_add = [0, len(new_phns) - 1]
    left_idx = 0
    new_phns_left = []
    sp_count = 0
    # find the left different index
    for key in word2phns.keys():
        idx, wrd = key.split('_')
        if wrd == 'sp':
            sp_count += 1
            new_phns_left.append('sp')
        else:
            idx = str(int(idx) - sp_count)
            if idx + '_' + wrd in new_word2phns:
                left_idx += len(new_word2phns[idx + '_' + wrd])
                new_phns_left.extend(word2phns[key].split())
            else:
                span_to_repl[0] = len(new_phns_left)
                span_to_add[0] = len(new_phns_left)
                break

    # reverse word2phns and new_word2phns
    right_idx = 0
    new_phns_right = []
    sp_count = 0
    word2phns_max_idx = int(list(word2phns.keys())[-1].split('_')[0])
    new_word2phns_max_idx = int(list(new_word2phns.keys())[-1].split('_')[0])
    new_phns_mid = []
    if append_new_str:
        new_phns_right = []
        new_phns_mid = new_phns[left_idx:]
        span_to_repl[0] = len(new_phns_left)
        span_to_add[0] = len(new_phns_left)
        span_to_add[1] = len(new_phns_left) + len(new_phns_mid)
        span_to_repl[1] = len(old_phns) - len(new_phns_right)
    else:
        for key in list(word2phns.keys())[::-1]:
            idx, wrd = key.split('_')
            if wrd == 'sp':
                sp_count += 1
                new_phns_right = ['sp'] + new_phns_right
            else:
                idx = str(new_word2phns_max_idx - (word2phns_max_idx - int(idx)
                                                   - sp_count))
                if idx + '_' + wrd in new_word2phns:
                    right_idx -= len(new_word2phns[idx + '_' + wrd])
                    new_phns_right = word2phns[key].split() + new_phns_right
                else:
                    span_to_repl[1] = len(old_phns) - len(new_phns_right)
                    new_phns_mid = new_phns[left_idx:right_idx]
                    span_to_add[1] = len(new_phns_left) + len(new_phns_mid)
                    if len(new_phns_mid) == 0:
                        span_to_add[1] = min(span_to_add[1] + 1, len(new_phns))
                        span_to_add[0] = max(0, span_to_add[0] - 1)
                        span_to_repl[0] = max(0, span_to_repl[0] - 1)
                        span_to_repl[1] = min(span_to_repl[1] + 1,
                                              len(old_phns))
                    break
    new_phns = new_phns_left + new_phns_mid + new_phns_right

    return mfa_start, mfa_end, old_phns, new_phns, span_to_repl, span_to_add


def duration_adjust_factor(original_dur: List[int],
                           pred_dur: List[int],
                           phns: List[str]):
    length = 0
    factor_list = []
    for ori, pred, phn in zip(original_dur, pred_dur, phns):
        if pred == 0 or phn == 'sp':
            continue
        else:
            factor_list.append(ori / pred)
    factor_list = np.array(factor_list)
    factor_list.sort()
    if len(factor_list) < 5:
        return 1

    length = 2
    return np.average(factor_list[length:-length])


def prepare_features_with_duration(uid: str,
                                   prefix: str,
                                   wav_path: str,
                                   mlm_model: nn.Layer,
                                   source_lang: str="English",
                                   target_lang: str="English",
                                   old_str: str="",
                                   new_str: str="",
                                   duration_preditor_path: str=None,
                                   sid: str=None,
                                   mask_reconstruct: bool=False,
                                   duration_adjust: bool=True,
                                   start_end_sp: bool=False,
                                   train_args=None):
    wav_org, rate = librosa.load(
        wav_path, sr=train_args.feats_extract_conf['fs'])
    fs = train_args.feats_extract_conf['fs']
    hop_length = train_args.feats_extract_conf['hop_length']

    mfa_start, mfa_end, old_phns, new_phns, span_to_repl, span_to_add = get_phns_and_spans(
        wav_path=wav_path,
        old_str=old_str,
        new_str=new_str,
        source_lang=source_lang,
        target_lang=target_lang)

    if start_end_sp:
        if new_phns[-1] != 'sp':
            new_phns = new_phns + ['sp']

    if target_lang == "english":
        old_durations = evaluate_durations(old_phns, target_lang=target_lang)

    elif target_lang == "chinese":

        if source_lang == "english":
            old_durations = evaluate_durations(
                old_phns, target_lang=source_lang)

        elif source_lang == "chinese":
            old_durations = evaluate_durations(
                old_phns, target_lang=source_lang)

    else:
        assert target_lang == "chinese" or target_lang == "english", "calculate duration_predict is not support for this language..."

    original_old_durations = [e - s for e, s in zip(mfa_end, mfa_start)]
    if '[MASK]' in new_str:
        new_phns = old_phns
        span_to_add = span_to_repl
        d_factor_left = duration_adjust_factor(
            original_old_durations[:span_to_repl[0]],
            old_durations[:span_to_repl[0]], old_phns[:span_to_repl[0]])
        d_factor_right = duration_adjust_factor(
            original_old_durations[span_to_repl[1]:],
            old_durations[span_to_repl[1]:], old_phns[span_to_repl[1]:])
        d_factor = (d_factor_left + d_factor_right) / 2
        new_durations_adjusted = [d_factor * i for i in old_durations]
    else:
        if duration_adjust:
            d_factor = duration_adjust_factor(original_old_durations,
                                              old_durations, old_phns)
            d_factor = d_factor * 1.25
        else:
            d_factor = 1

        if target_lang == "english":
            new_durations = evaluate_durations(
                new_phns, target_lang=target_lang)

        elif target_lang == "chinese":
            new_durations = evaluate_durations(
                new_phns, target_lang=target_lang)

        new_durations_adjusted = [d_factor * i for i in new_durations]

        if span_to_repl[0] < len(old_phns) and old_phns[span_to_repl[
                0]] == new_phns[span_to_add[0]]:
            new_durations_adjusted[span_to_add[0]] = original_old_durations[
                span_to_repl[0]]
        if span_to_repl[1] < len(old_phns) and span_to_add[1] < len(new_phns):
            if old_phns[span_to_repl[1]] == new_phns[span_to_add[1]]:
                new_durations_adjusted[span_to_add[1]] = original_old_durations[
                    span_to_repl[1]]
    new_span_duration_sum = sum(
        new_durations_adjusted[span_to_add[0]:span_to_add[1]])
    old_span_duration_sum = sum(
        original_old_durations[span_to_repl[0]:span_to_repl[1]])
    duration_offset = new_span_duration_sum - old_span_duration_sum
    new_mfa_start = mfa_start[:span_to_repl[0]]
    new_mfa_end = mfa_end[:span_to_repl[0]]
    for i in new_durations_adjusted[span_to_add[0]:span_to_add[1]]:
        if len(new_mfa_end) == 0:
            new_mfa_start.append(0)
            new_mfa_end.append(i)
        else:
            new_mfa_start.append(new_mfa_end[-1])
            new_mfa_end.append(new_mfa_end[-1] + i)
    new_mfa_start += [i + duration_offset for i in mfa_start[span_to_repl[1]:]]
    new_mfa_end += [i + duration_offset for i in mfa_end[span_to_repl[1]:]]

    # 3. get new wav 
    if span_to_repl[0] >= len(mfa_start):
        left_idx = len(wav_org)
        right_idx = left_idx
    else:
        left_idx = int(np.floor(mfa_start[span_to_repl[0]] * fs))
        right_idx = int(np.ceil(mfa_end[span_to_repl[1] - 1] * fs))
    new_blank_wav = np.zeros(
        (int(np.ceil(new_span_duration_sum * fs)), ), dtype=wav_org.dtype)
    new_wav_org = np.concatenate(
        [wav_org[:left_idx], new_blank_wav, wav_org[right_idx:]])

    # 4. get old and new mel span to be mask
    # [92, 92]
    old_span_bdy = get_masked_mel_bdy(mfa_start, mfa_end, fs, hop_length,
                                      span_to_repl)
    # [92, 174]
    new_span_bdy = get_masked_mel_bdy(new_mfa_start, new_mfa_end, fs,
                                      hop_length, span_to_add)

    return new_wav_org, new_phns, new_mfa_start, new_mfa_end, old_span_bdy, new_span_bdy


def prepare_features(uid: str,
                     mlm_model: nn.Layer,
                     processor,
                     wav_path: str,
                     prefix: str="./prompt/dev/",
                     source_lang: str="english",
                     target_lang: str="english",
                     old_str: str="",
                     new_str: str="",
                     duration_preditor_path: str=None,
                     sid: str=None,
                     duration_adjust: bool=True,
                     start_end_sp: bool=False,
                     mask_reconstruct: bool=False,
                     train_args=None):
    wav_org, phns_list, mfa_start, mfa_end, old_span_bdy, new_span_bdy = prepare_features_with_duration(
        uid=uid,
        prefix=prefix,
        source_lang=source_lang,
        target_lang=target_lang,
        mlm_model=mlm_model,
        old_str=old_str,
        new_str=new_str,
        wav_path=wav_path,
        duration_preditor_path=duration_preditor_path,
        sid=sid,
        duration_adjust=duration_adjust,
        start_end_sp=start_end_sp,
        mask_reconstruct=mask_reconstruct,
        train_args=train_args)
    speech = wav_org
    align_start = np.array(mfa_start)
    align_end = np.array(mfa_end)
    token_to_id = {item: i for i, item in enumerate(train_args.token_list)}
    text = np.array(
        list(
            map(lambda x: token_to_id.get(x, token_to_id['<unk>']), phns_list)))

    span_bdy = np.array(new_span_bdy)
    batch = [('1', {
        "speech": speech,
        "align_start": align_start,
        "align_end": align_end,
        "text": text,
        "span_bdy": span_bdy
    })]

    return batch, old_span_bdy, new_span_bdy


def decode_with_model(uid: str,
                      mlm_model: nn.Layer,
                      processor,
                      collate_fn,
                      wav_path: str,
                      prefix: str="./prompt/dev/",
                      source_lang: str="english",
                      target_lang: str="english",
                      old_str: str="",
                      new_str: str="",
                      duration_preditor_path: str=None,
                      sid: str=None,
                      decoder: bool=False,
                      use_teacher_forcing: bool=False,
                      duration_adjust: bool=True,
                      start_end_sp: bool=False,
                      train_args=None):
    fs, hop_length = train_args.feats_extract_conf[
        'fs'], train_args.feats_extract_conf['hop_length']

    batch, old_span_bdy, new_span_bdy = prepare_features(
        uid=uid,
        prefix=prefix,
        source_lang=source_lang,
        target_lang=target_lang,
        mlm_model=mlm_model,
        processor=processor,
        wav_path=wav_path,
        old_str=old_str,
        new_str=new_str,
        duration_preditor_path=duration_preditor_path,
        sid=sid,
        duration_adjust=duration_adjust,
        start_end_sp=start_end_sp,
        train_args=train_args)

    feats = collate_fn(batch)[1]

    if 'text_masked_pos' in feats.keys():
        feats.pop('text_masked_pos')
    for k, v in feats.items():
        feats[k] = paddle.to_tensor(v)
    rtn = mlm_model.inference(
        **feats, span_bdy=new_span_bdy, use_teacher_forcing=use_teacher_forcing)
    output = rtn['feat_gen']
    if 0 in output[0].shape and 0 not in output[-1].shape:
        output_feat = paddle.concat(
            output[1:-1] + [output[-1].squeeze()], axis=0).cpu()
    elif 0 not in output[0].shape and 0 in output[-1].shape:
        output_feat = paddle.concat(
            [output[0].squeeze()] + output[1:-1], axis=0).cpu()
    elif 0 in output[0].shape and 0 in output[-1].shape:
        output_feat = paddle.concat(output[1:-1], axis=0).cpu()
    else:
        output_feat = paddle.concat(
            [output[0].squeeze(0)] + output[1:-1] + [output[-1].squeeze(0)],
            axis=0).cpu()

    wav_org, _ = librosa.load(
        wav_path, sr=train_args.feats_extract_conf['fs'])
    return wav_org, None, output_feat, old_span_bdy, new_span_bdy, fs, hop_length


class MLMCollateFn:
    """Functor class of common_collate_fn()"""

    def __init__(self,
                 feats_extract,
                 float_pad_value: Union[float, int]=0.0,
                 int_pad_value: int=-32768,
                 not_sequence: Collection[str]=(),
                 mlm_prob: float=0.8,
                 mean_phn_span: int=8,
                 attention_window: int=0,
                 pad_speech: bool=False,
                 sega_emb: bool=False,
                 duration_collect: bool=False,
                 text_masking: bool=False):
        self.mlm_prob = mlm_prob
        self.mean_phn_span = mean_phn_span
        self.feats_extract = feats_extract
        self.float_pad_value = float_pad_value
        self.int_pad_value = int_pad_value
        self.not_sequence = set(not_sequence)
        self.attention_window = attention_window
        self.pad_speech = pad_speech
        self.sega_emb = sega_emb
        self.duration_collect = duration_collect
        self.text_masking = text_masking

    def __repr__(self):
        return (f"{self.__class__}(float_pad_value={self.float_pad_value}, "
                f"int_pad_value={self.float_pad_value})")

    def __call__(self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
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
            text_masking=self.text_masking)


def mlm_collate_fn(
        data: Collection[Tuple[str, Dict[str, np.ndarray]]],
        float_pad_value: Union[float, int]=0.0,
        int_pad_value: int=-32768,
        not_sequence: Collection[str]=(),
        mlm_prob: float=0.8,
        mean_phn_span: int=8,
        feats_extract=None,
        attention_window: int=0,
        pad_speech: bool=False,
        sega_emb: bool=False,
        duration_collect: bool=False,
        text_masking: bool=False) -> Tuple[List[str], Dict[str, paddle.Tensor]]:
    uttids = [u for u, _ in data]
    data = [d for _, d in data]

    assert all(set(data[0]) == set(d) for d in data), "dict-keys mismatching"
    assert all(not k.endswith("_lens")
               for k in data[0]), f"*_lens is reserved: {list(data[0])}"

    output = {}
    for key in data[0]:
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
            lens = paddle.to_tensor(
                [d[key].shape[0] for d in data], dtype=paddle.int64)
            output[key + "_lens"] = lens

    feats = feats_extract.get_log_mel_fbank(np.array(output["speech"][0]))
    feats = paddle.to_tensor(feats)
    feats_lens = paddle.shape(feats)[0]
    feats = paddle.unsqueeze(feats, 0)
    if 'text' not in output:
        text = paddle.zeros(paddle.shape(feats_lens.unsqueeze(-1))) - 2
        text_lens = paddle.zeros(paddle.shape(feats_lens)) + 1
        max_tlen = 1
        align_start = paddle.zeros(paddle.shape(text))
        align_end = paddle.zeros(paddle.shape(text))
        align_start_lens = paddle.zeros(paddle.shape(feats_lens))
        sega_emb = False
        mean_phn_span = 0
        mlm_prob = 0.15
    else:
        text = output["text"]
        text_lens = output["text_lens"]
        align_start = output["align_start"]
        align_start_lens = output["align_start_lens"]
        align_end = output["align_end"]
        align_start = paddle.floor(feats_extract.sr * align_start /
                                   feats_extract.hop_length).int()
        align_end = paddle.floor(feats_extract.sr * align_end /
                                 feats_extract.hop_length).int()
        max_tlen = max(text_lens)
    max_slen = max(feats_lens)
    speech_pad = feats[:, :max_slen]
    if attention_window > 0 and pad_speech:
        speech_pad, max_slen = pad_to_longformer_att_window(
            speech_pad, max_slen, max_slen, attention_window)
    max_len = max_slen + max_tlen
    if attention_window > 0:
        text_pad, max_tlen = pad_to_longformer_att_window(
            text, max_len, max_tlen, attention_window)
    else:
        text_pad = text
    text_mask = make_non_pad_mask(
        text_lens, text_pad, length_dim=1).unsqueeze(-2)
    if attention_window > 0:
        text_mask = text_mask * 2
    speech_mask = make_non_pad_mask(
        feats_lens, speech_pad[:, :, 0], length_dim=1).unsqueeze(-2)
    span_bdy = None
    if 'span_bdy' in output.keys():
        span_bdy = output['span_bdy']

    if text_masking:
        masked_pos, text_masked_pos, _ = phones_text_masking(
            speech_pad, speech_mask, text_pad, text_mask, align_start,
            align_end, align_start_lens, mlm_prob, mean_phn_span, span_bdy)
    else:
        text_masked_pos = paddle.zeros(paddle.shape(text_pad))
        masked_pos, _ = phones_masking(speech_pad, speech_mask, align_start,
                                       align_end, align_start_lens, mlm_prob,
                                       mean_phn_span, span_bdy)

    output_dict = {}
    if duration_collect and 'text' in output:
        reordered_idx, speech_seg_pos, text_seg_pos, durations, feats_lens = get_seg_pos_reduce_duration(
            speech_pad, text_pad, align_start, align_end, align_start_lens,
            sega_emb, masked_pos, feats_lens)
        speech_mask = make_non_pad_mask(
            feats_lens, speech_pad[:, :reordered_idx.shape[1], 0],
            length_dim=1).unsqueeze(-2)
        output_dict['durations'] = durations
        output_dict['reordered_idx'] = reordered_idx
    else:
        speech_seg_pos, text_seg_pos = get_seg_pos(speech_pad, text_pad,
                                                   align_start, align_end,
                                                   align_start_lens, sega_emb)
    output_dict['speech'] = speech_pad
    output_dict['text'] = text_pad
    output_dict['masked_pos'] = masked_pos
    output_dict['text_masked_pos'] = text_masked_pos
    output_dict['speech_mask'] = speech_mask
    output_dict['text_mask'] = text_mask
    output_dict['speech_seg_pos'] = speech_seg_pos
    output_dict['text_seg_pos'] = text_seg_pos
    output_dict['speech_lens'] = output["speech_lens"]
    output_dict['text_lens'] = text_lens
    output = (uttids, output_dict)
    return output


def build_collate_fn(args: argparse.Namespace, train: bool, epoch=-1):
    # -> Callable[
    #     [Collection[Tuple[str, Dict[str, np.ndarray]]]],
    #     Tuple[List[str], Dict[str, Tensor]],
    # ]:

    # assert check_argument_types()
    # return CommonCollateFn(float_pad_value=0.0, int_pad_value=0)
    feats_extract_class = LogMelFBank
    if args.feats_extract_conf['win_length'] is None:
        args.feats_extract_conf['win_length'] = args.feats_extract_conf['n_fft']

    args_dic = {}
    for k, v in args.feats_extract_conf.items():
        if k == 'fs':
            args_dic['sr'] = v
        else:
            args_dic[k] = v
    feats_extract = feats_extract_class(**args_dic)

    sega_emb = True if args.encoder_conf['input_layer'] == 'sega_mlm' else False
    if args.encoder_conf['selfattention_layer_type'] == 'longformer':
        attention_window = args.encoder_conf['attention_window']
        pad_speech = True if 'pre_speech_layer' in args.encoder_conf and args.encoder_conf[
            'pre_speech_layer'] > 0 else False
    else:
        attention_window = 0
        pad_speech = False
    if epoch == -1:
        mlm_prob_factor = 1
    else:
        mlm_prob_factor = 0.8
    if 'duration_predictor_layers' in args.model_conf.keys(
    ) and args.model_conf['duration_predictor_layers'] > 0:
        duration_collect = True
    else:
        duration_collect = False

    return MLMCollateFn(
        feats_extract,
        float_pad_value=0.0,
        int_pad_value=0,
        mlm_prob=args.model_conf['mlm_prob'] * mlm_prob_factor,
        mean_phn_span=args.model_conf['mean_phn_span'],
        attention_window=attention_window,
        pad_speech=pad_speech,
        sega_emb=sega_emb,
        duration_collect=duration_collect)


def get_mlm_output(uid: str,
                   wav_path: str,
                   prefix: str="./prompt/dev/",
                   model_name: str="conformer",
                   source_lang: str="english",
                   target_lang: str="english",
                   old_str: str="",
                   new_str: str="",
                   duration_preditor_path: str=None,
                   sid: str=None,
                   decoder: bool=False,
                   use_teacher_forcing: bool=False,
                   duration_adjust: bool=True,
                   start_end_sp: bool=False):
    mlm_model, train_args = load_model(model_name)
    mlm_model.eval()
    processor = None
    collate_fn = build_collate_fn(train_args, False)

    return decode_with_model(
        uid=uid,
        prefix=prefix,
        source_lang=source_lang,
        target_lang=target_lang,
        mlm_model=mlm_model,
        processor=processor,
        collate_fn=collate_fn,
        wav_path=wav_path,
        old_str=old_str,
        new_str=new_str,
        duration_preditor_path=duration_preditor_path,
        sid=sid,
        decoder=decoder,
        use_teacher_forcing=use_teacher_forcing,
        duration_adjust=duration_adjust,
        start_end_sp=start_end_sp,
        train_args=train_args)


def evaluate(uid: str,
             source_lang: str="english",
             target_lang: str="english",
             use_pt_vocoder: bool=False,
             prefix: str="./prompt/dev/",
             model_name: str="conformer",
             old_str: str="",
             new_str: str="",
             prompt_decoding: bool=False,
             task_name: str=None):

    duration_preditor_path = None
    spemd = None
    full_origin_str, wav_path = read_data(uid=uid, prefix=prefix)

    if task_name == 'edit':
        new_str = new_str
    elif task_name == 'synthesize':
        new_str = full_origin_str + new_str
    else:
        new_str = full_origin_str + ' '.join(
            [ch for ch in new_str if is_chinese(ch)])

    print('new_str is ', new_str)

    if not old_str:
        old_str = full_origin_str

    results_dict, old_span = plot_mel_and_vocode_wav(
        uid=uid,
        prefix=prefix,
        source_lang=source_lang,
        target_lang=target_lang,
        model_name=model_name,
        wav_path=wav_path,
        full_origin_str=full_origin_str,
        old_str=old_str,
        new_str=new_str,
        use_pt_vocoder=use_pt_vocoder,
        duration_preditor_path=duration_preditor_path,
        sid=spemd)
    return results_dict


if __name__ == "__main__":
    # parse config and args
    args = parse_args()

    data_dict = evaluate(
        uid=args.uid,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        use_pt_vocoder=args.use_pt_vocoder,
        prefix=args.prefix,
        model_name=args.model_name,
        new_str=args.new_str,
        task_name=args.task_name)
    sf.write(args.output_name, data_dict['output'], samplerate=24000)
    print("finished...")
