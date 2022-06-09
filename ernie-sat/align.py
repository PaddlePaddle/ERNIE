#!/usr/bin/env python
""" Usage:
    align.py wavfile trsfile outwordfile outphonefile
"""
import multiprocessing as mp
import os
import sys

from tqdm import tqdm

PHONEME = 'tools/aligner/english_envir/english2phoneme/phoneme'
MODEL_DIR_EN = 'tools/aligner/english'
MODEL_DIR_ZH = 'tools/aligner/mandarin'
HVITE = 'tools/htk/HTKTools/HVite'
HCOPY = 'tools/htk/HTKTools/HCopy'


def prep_txt_zh(line: str, tmpbase: str, dictfile: str):

    words = []
    line = line.strip()
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
    with open(dictfile, 'r') as fid:
        for line in fid:
            ds.add(line.split()[0])

    unk_words = set([])
    with open(tmpbase + '.txt', 'w') as fwid:
        for wrd in words:
            if (wrd not in ds):
                unk_words.add(wrd)
            fwid.write(wrd + ' ')
        fwid.write('\n')
    return unk_words


def prep_txt_en(line: str, tmpbase, dictfile):

    words = []

    line = line.strip()
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
    with open(dictfile, 'r') as fid:
        for line in fid:
            ds.add(line.split()[0])

    unk_words = set([])
    with open(tmpbase + '.txt', 'w') as fwid:
        for wrd in words:
            if (wrd.upper() not in ds):
                unk_words.add(wrd.upper())
            fwid.write(wrd + ' ')
        fwid.write('\n')

    #generate pronounciations for unknows words using 'letter to sound'
    with open(tmpbase + '_unk.words', 'w') as fwid:
        for unk in unk_words:
            fwid.write(unk + '\n')
    try:
        os.system(PHONEME + ' ' + tmpbase + '_unk.words' + ' ' + tmpbase +
                  '_unk.phons')
    except:
        print('english2phoneme error!')
        sys.exit(1)

    #add unknown words to the standard dictionary, generate a tmp dictionary for alignment 
    fw = open(tmpbase + '.dict', 'w')
    with open(dictfile, 'r') as fid:
        for line in fid:
            fw.write(line)
    f = open(tmpbase + '_unk.words', 'r')
    lines1 = f.readlines()
    f.close()
    f = open(tmpbase + '_unk.phons', 'r')
    lines2 = f.readlines()
    f.close()
    for i in range(len(lines1)):
        wrd = lines1[i].replace('\n', '')
        phons = lines2[i].replace('\n', '').replace(' ', '')
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

        fw.write(wrd + ' ')
        for s in seq:
            fw.write(' ' + s)
        fw.write('\n')
    fw.close()


def prep_mlf(txt: str, tmpbase: str):

    with open(tmpbase + '.mlf', 'w') as fwid:
        fwid.write('#!MLF!#\n')
        fwid.write('"' + tmpbase + '.lab"\n')
        fwid.write('sp\n')
        wrds = txt.split()
        for wrd in wrds:
            fwid.write(wrd.upper() + '\n')
            fwid.write('sp\n')
        fwid.write('.\n')


def _get_user():
    return os.path.expanduser('~').split("/")[-1]


def alignment(wav_path: str, text: str):
    tmpbase = '/tmp/' + _get_user() + '_' + str(os.getpid())

    #prepare wav and trs files
    try:
        os.system('sox ' + wav_path + ' -r 16000 ' + tmpbase + '.wav remix -')
    except:
        print('sox error!')
        return None

    #prepare clean_transcript file
    try:
        prep_txt_en(text, tmpbase, MODEL_DIR_EN + '/dict')
    except:
        print('prep_txt error!')
        return None

    #prepare mlf file
    try:
        with open(tmpbase + '.txt', 'r') as fid:
            txt = fid.readline()
        prep_mlf(txt, tmpbase)
    except:
        print('prep_mlf error!')
        return None

    #prepare scp
    try:
        os.system(HCOPY + ' -C ' + MODEL_DIR_EN + '/16000/config ' + tmpbase +
                  '.wav' + ' ' + tmpbase + '.plp')
    except:
        print('HCopy error!')
        return None

    #run alignment
    try:
        os.system(HVITE + ' -a -m -t 10000.0 10000.0 100000.0 -I ' + tmpbase +
                  '.mlf -H ' + MODEL_DIR_EN + '/16000/macros -H ' + MODEL_DIR_EN
                  + '/16000/hmmdefs -i ' + tmpbase + '.aligned ' + tmpbase +
                  '.dict ' + MODEL_DIR_EN + '/monophones ' + tmpbase +
                  '.plp 2>&1 > /dev/null')
    except:
        print('HVite error!')
        return None

    with open(tmpbase + '.txt', 'r') as fid:
        words = fid.readline().strip().split()
    words = txt.strip().split()
    words.reverse()

    with open(tmpbase + '.aligned', 'r') as fid:
        lines = fid.readlines()
    i = 2
    times2 = []
    word2phns = {}
    current_word = ''
    index = 0
    while (i < len(lines)):
        splited_line = lines[i].strip().split()
        if (len(splited_line) >= 4) and (splited_line[0] != splited_line[1]):
            phn = splited_line[2]
            pst = (int(splited_line[0]) / 1000 + 125) / 10000
            pen = (int(splited_line[1]) / 1000 + 125) / 10000
            times2.append([phn, pst, pen])
            # splited_line[-1]!='sp'
            if len(splited_line) == 5:
                current_word = str(index) + '_' + splited_line[-1]
                word2phns[current_word] = phn
                index += 1
            elif len(splited_line) == 4:
                word2phns[current_word] += ' ' + phn
        i += 1
    return times2, word2phns


def alignment_zh(wav_path, text_string):
    tmpbase = '/tmp/' + _get_user() + '_' + str(os.getpid())

    #prepare wav and trs files
    try:
        os.system('sox ' + wav_path + ' -r 16000 -b 16 ' + tmpbase +
                  '.wav remix -')

    except:
        print('sox error!')
        return None

    #prepare clean_transcript file
    try:
        unk_words = prep_txt_zh(text_string, tmpbase, MODEL_DIR_ZH + '/dict')
        if unk_words:
            print('Error! Please add the following words to dictionary:')
            for unk in unk_words:
                print("非法words: ", unk)
    except:
        print('prep_txt error!')
        return None

    #prepare mlf file
    try:
        with open(tmpbase + '.txt', 'r') as fid:
            txt = fid.readline()
        prep_mlf(txt, tmpbase)
    except:
        print('prep_mlf error!')
        return None

    #prepare scp
    try:
        os.system(HCOPY + ' -C ' + MODEL_DIR_ZH + '/16000/config ' + tmpbase +
                  '.wav' + ' ' + tmpbase + '.plp')
    except:
        print('HCopy error!')
        return None

    #run alignment
    try:
        os.system(HVITE + ' -a -m -t 10000.0 10000.0 100000.0 -I ' + tmpbase +
                  '.mlf -H ' + MODEL_DIR_ZH + '/16000/macros -H ' + MODEL_DIR_ZH
                  + '/16000/hmmdefs -i ' + tmpbase + '.aligned ' + MODEL_DIR_ZH
                  + '/dict ' + MODEL_DIR_ZH + '/monophones ' + tmpbase +
                  '.plp 2>&1 > /dev/null')

    except:
        print('HVite error!')
        return None

    with open(tmpbase + '.txt', 'r') as fid:
        words = fid.readline().strip().split()
    words = txt.strip().split()
    words.reverse()

    with open(tmpbase + '.aligned', 'r') as fid:
        lines = fid.readlines()

    i = 2
    times2 = []
    word2phns = {}
    current_word = ''
    index = 0
    while (i < len(lines)):
        splited_line = lines[i].strip().split()
        if (len(splited_line) >= 4) and (splited_line[0] != splited_line[1]):
            phn = splited_line[2]
            pst = (int(splited_line[0]) / 1000 + 125) / 10000
            pen = (int(splited_line[1]) / 1000 + 125) / 10000
            times2.append([phn, pst, pen])
            # splited_line[-1]!='sp'
            if len(splited_line) == 5:
                current_word = str(index) + '_' + splited_line[-1]
                word2phns[current_word] = phn
                index += 1
            elif len(splited_line) == 4:
                word2phns[current_word] += ' ' + phn
        i += 1
    return times2, word2phns
