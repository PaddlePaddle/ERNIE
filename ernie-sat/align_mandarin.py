#!/usr/bin/env python

""" Usage:
      align_mandarin.py wavfile trsfile outwordfile putphonefile
"""

import os
import sys
from tqdm import tqdm
import multiprocessing as mp


MODEL_DIR = 'tools/aligner/mandarin'
HVITE = 'tools/htk/HTKTools/HVite'
HCOPY = 'tools/htk/HTKTools/HCopy'


def prep_txt(line, tmpbase, dictfile):

    words = []
    line = line.strip()
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

def prep_mlf(txt, tmpbase):

    with open(tmpbase + '.mlf', 'w') as fwid:
        fwid.write('#!MLF!#\n')
        fwid.write('"' + tmpbase + '.lab"\n')
        fwid.write('sp\n')
        wrds = txt.split()
        for wrd in wrds:
            fwid.write(wrd.upper() + '\n')
            fwid.write('sp\n')
        fwid.write('.\n')

def gen_res(tmpbase, outfile1, outfile2):
    with open(tmpbase + '.txt', 'r') as fid:
        words = fid.readline().strip().split()
    words = txt.strip().split()
    words.reverse()

    with open(tmpbase + '.aligned', 'r') as fid:
        lines = fid.readlines()
    i = 2
    times1 = []
    times2 = []
    while (i < len(lines)):
        if (len(lines[i].split()) >= 4) and (lines[i].split()[0] != lines[i].split()[1]):
            phn = lines[i].split()[2]
            pst = (int(lines[i].split()[0])/1000+125)/10000
            pen = (int(lines[i].split()[1])/1000+125)/10000
            times2.append([phn, pst, pen])
        if (len(lines[i].split()) == 5):
            if (lines[i].split()[0] != lines[i].split()[1]):
                wrd = lines[i].split()[-1].strip()
                st = (int(lines[i].split()[0])/1000+125)/10000
                j = i + 1
                while (lines[j] != '.\n') and (len(lines[j].split()) != 5):
                    j += 1
                en = (int(lines[j-1].split()[1])/1000+125)/10000
                times1.append([wrd, st, en])
        i += 1

    with open(outfile1, 'w') as fwid:
        for item in times1:
            if (item[0] == 'sp'):
                fwid.write(str(item[1]) + ' ' + str(item[2]) + ' SIL\n')
            else:
                wrd = words.pop()
                fwid.write(str(item[1]) + ' ' + str(item[2]) + ' ' + wrd + '\n')
    if words:
        print('not matched::' + alignfile)
        sys.exit(1)

    with open(outfile2, 'w') as fwid:
        for item in times2:
            fwid.write(str(item[1]) + ' ' + str(item[2]) + ' ' + item[0] + '\n')



def alignment_zh(wav_path, text_string):
    tmpbase = '/tmp/' + os.environ['USER'] + '_' + str(os.getpid())

    #prepare wav and trs files
    try:
        os.system('sox ' + wav_path + ' -r 16000 -b 16 ' + tmpbase + '.wav remix -')

    except:
        print('sox error!')
        return None
    
    #prepare clean_transcript file
    try:
        unk_words = prep_txt(text_string, tmpbase, MODEL_DIR + '/dict')
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
        os.system(HCOPY + ' -C ' + MODEL_DIR + '/16000/config ' + tmpbase + '.wav' + ' ' + tmpbase + '.plp')
    except:
        print('HCopy error!')
        return None

    #run alignment
    try:
        os.system(HVITE + ' -a -m -t 10000.0 10000.0 100000.0 -I ' + tmpbase + '.mlf -H ' + MODEL_DIR + '/16000/macros -H ' + MODEL_DIR + '/16000/hmmdefs -i ' + tmpbase +  '.aligned '  + MODEL_DIR + '/dict ' + MODEL_DIR + '/monophones ' + tmpbase + '.plp 2>&1 > /dev/null')

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
            pst = (int(splited_line[0])/1000+125)/10000
            pen = (int(splited_line[1])/1000+125)/10000
            times2.append([phn, pst, pen])
            # splited_line[-1]!='sp'
            if len(splited_line)==5:
                current_word = str(index)+'_'+splited_line[-1]
                word2phns[current_word] = phn
                index+=1
            elif len(splited_line)==4:
                word2phns[current_word] += ' '+phn
        i+=1
    return times2,word2phns

