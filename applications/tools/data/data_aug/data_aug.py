# -*- coding: utf-8 -*
"""
:py:data augmentation tools
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import argparse
import json
import logging
import numpy as np
import six
import collections

from functools import reduce
from tqdm import tqdm

log = logging.getLogger(__name__)
stream_hdl = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(fmt='[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:    %(message)s')
stream_hdl.setFormatter(formatter)
log.addHandler(stream_hdl)
log.setLevel(logging.DEBUG)


def build_unk_parser(args):
    """
    build unk parser
    """
    log.info('building unk parser')
    max_span_len = 10
    p = 0.2
    span_lens = range(1, max_span_len + 1)
    span_len_dist = [p * (1 - p) ** (i - 1) for i in span_lens]
    span_len_dist = [x / sum(span_len_dist) for x in span_len_dist]
    log.debug('span len dist:')
    avg_span_len = 0.
    for k, v in zip(span_lens, span_len_dist):
        log.debug('\t%d: %f' % (k, v))
        avg_span_len += k * v
    log.debug('avg span len: %f' % avg_span_len)

    def unk_parser(tokens):
        """
        unk parser
        """
        ret, i = [], 0
        while i < len(tokens):
            span_len = np.random.choice(span_lens, p=span_len_dist)
            span_len = min(span_len, len(tokens) - len(ret))
            if np.random.rand() < 0.15:
                ret += [args.unk_token] * span_len
            else:
                ret += tokens[i: i + span_len]
            i += span_len
        ret = ''.join(ret)
        return ret
    log.info('done')
    return unk_parser


def build_trucate_parser(args):
    """
    build truncate parser
    """
    log.info('building truncate parser')
    max_span_len = 10
    p = 0.2
    span_lens = range(1, max_span_len + 1)
    span_len_dist = [p * (1 - p) ** (i - 1) for i in span_lens]
    span_len_dist = [x / sum(span_len_dist) for x in span_len_dist]
    log.debug('span len dist:')
    avg_span_len = 0.
    for k, v in zip(span_lens, span_len_dist):
        log.debug('\t%d: %f' % (k, v))
        avg_span_len += k * v
    log.debug('avg span len: %f' % avg_span_len)

    def truncate_parser(tokens):
        """
        truncate parser
        """
        ret, i = [], 0
        while i < len(tokens):
            span_len = np.random.choice(span_lens, p=span_len_dist)
            span_len = min(span_len, len(tokens) - len(ret))
            if np.random.rand() < 0.15:
                pass
            else:
                ret += tokens[i: i + span_len]
            i += span_len
        ret = ''.join(ret)
        return ret
    log.info('done')
    return truncate_parser

def build_pos_dict(field_list):
    """
    build pos dict for pos parser
    """
    from LAC import LAC
    lac = LAC(mode='lac')
    pos_dict = {}
    for i in field_list:
        #piece, tag = lac.lexer(i.strip(), return_tag=True)
        piece, tag = lac.run(i.strip())
        for p, t in zip(piece, tag):
            pos_dict.setdefault(t, []).append(p)
    return pos_dict 

def build_pos_replace_parser(args):
    """
    build pos_replace parser
    """
    from LAC import LAC
    lac = LAC(mode='lac')
    log.info('building pos replace parser')
    def pos_replace_parser(tokens, pos_dict):
        """
        pos replace parser
        """
        tokens = tokens.strip()
        #piece, tag = lac.lexer(tokens, return_tag=True)
        # piece, tag = lac.run(tokens, return_tag=True)
        piece, tag = lac.run(tokens)
        ret = []
        for p, t in zip(piece, tag):
            if np.random.rand() < 0.15:
                p = np.random.choice(pos_dict[t])
            ret.append(p)
        ret = ''.join(ret)
        return ret

    log.info('done')
    return pos_replace_parser


def build_w2v_replace_parser(args):
    """
    build w2v_replace parser
    """
    import re
    from gensim.models import KeyedVectors
    import gensim
    from LAC import LAC
    lac = LAC(mode='seg')

    bin_file = "./vec2.bin"
    if os.path.exists(bin_file):
        log.debug('loading word2vec....')
        word2vec = KeyedVectors.load_word2vec_format(bin_file)
        log.debug('done loading word2vec....')
    else:
        log.debug('loading word2vec from txt....')
        tmp_file = './vec2.txt'
        #4.0以上版本用法
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(tmp_file, binary=False)
        #word2vec.save_word2vec_format(bin_file)
        log.debug('done loading word2vec....')

        pat = re.compile('[a-zA-Z0-9]+')

        def w2v_parser(tokens):
            """
            w2v parser
            """
            ret = []
            #for i in lac.lexer(tokens):
            for i in lac.run(tokens):
                if np.random.rand() < 0.15 and i in  word2vec.index_to_key:
                    candidate = word2vec.similar_by_word(i, topn=3)
                    t = np.random.choice([c for c, p in candidate])
                    if six.PY3: 
                        t = t.strip()
                    elif six.PY2:
                        t = t.strip().decode("utf8")

                    if pat.match(t):
                        t = '%s ' % t
                    ret.append(t)
                else:
                    ret.append(i)
            ret = ''.join(ret)
            return ret

        return w2v_parser


builders = {
    "unk": build_unk_parser,
    "truncate": build_trucate_parser, 
    "pos_replace": build_pos_replace_parser, 
    "w2v_replace": build_w2v_replace_parser,
}


def build_parser(args):
    """
    build parser
    """
    selected_funcs, probs, selected_func_names = [], [], []
    for func_name in builders:
        p = args.__dict__[func_name]
        print("args.dict", args.__dict__)
        #return
        if p > 0.:
            log.info('using %s with prob %.2f' % (func_name, p))
            probs.append(p)
            func = builders[func_name](args)
            selected_funcs.append(func)
            selected_func_names.append(func_name)
    probs = np.array(probs) 
    probs /= probs.sum()
    def choose_parser():
        """
        choose parser
        """
        f = np.random.choice(selected_funcs, p=probs)
        #print(f)
        return f
    return choose_parser, selected_func_names 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("-n", "--aug_times", type=int, default=4)
    parser.add_argument("-c", "--column_number", type=str, default='1')
    parser.add_argument("-u", "--unk", type=float, default=0.25)
    parser.add_argument("-t", "--truncate", type=float, default=0.25)
    parser.add_argument("-r", "--pos_replace", type=float, default=0.25)
    parser.add_argument("-w", "--w2v_replace", type=float, default=0.25)

    if six.PY2: 
        parser.add_argument("--unk_token", type=unicode, default='\U0001f604')
    elif six.PY3:
        parser.add_argument("--unk_token", type=str, default='\U0001f604')

    args = parser.parse_args()

    col_nums = args.column_number.split(',')
    col_nums = list(map(int, col_nums))

    choose_parser, selected_func_names = build_parser(args)

    data_files = os.listdir(args.input)
    assert len(data_files) > 0, "%s is an empty directory" % args.input
    mkdirlambda =lambda x: os.makedirs(x) if not os.path.exists(x)  else True
    mkdirlambda(args.output)
    
    counter = collections.defaultdict(lambda:0)

    for data_file in data_files:
        input_file_path = os.path.join(args.input, data_file)
        print("input_file_path", input_file_path)
        input_file_name, suffix = os.path.splitext(data_file)
        print(input_file_name, suffix)
        output_file_name = '{0}_aug{1}' . format(input_file_name, suffix)
        output_file_path = os.path.join(args.output, output_file_name)  


        fields_list = []
        for i in range(len(col_nums)):
            fields_list.append([])
        pos_dict = []
        if "pos_replace" in selected_func_names:
            if six.PY3: 
                with open(input_file_path, encoding='UTF-8') as input_file:
                    for l in input_file.readlines():
                        cols = l.strip().split('\t')
                        for j in col_nums:
                            fields_list[j - 1].append(cols[j - 1])
            elif six.PY2:
                with open(input_file_path) as input_file:
                    for l in input_file.readlines():
                        cols = l.strip().decode("utf8").split('\t')
                        for j in col_nums:
                            fields_list[j - 1].append(cols[j - 1])

            #print(np.array(fields_list).shape)
            for j in col_nums:
                pos_dict.append(build_pos_dict(fields_list[j - 1]))

        
        if six.PY3: 
            with open(input_file_path, 'r', encoding='UTF-8') as input_file:
                with open(output_file_path, 'w', encoding='UTF-8') as output_file:
                    for i, l in enumerate(input_file.readlines()):
                        parser = choose_parser()
                        #print(parser.__name__ == "pos_replace_parser")
                        if i % 1000 == 0:
                            log.debug('parsing line %d' % i)
                        print(l.strip(), file=output_file)

                        for k in range(args.aug_times):
                            cols = l.strip().split('\t')
                            for j in col_nums:
                                if parser.__name__ == "pos_replace_parser":
                                    cols[j - 1] = parser(cols[j - 1], pos_dict[j - 1])
                                    counter[parser.__name__] += 1
                                else:
                                    cols[j - 1] = parser(cols[j - 1])
                                    counter[parser.__name__] += 1
                                new_line = '\t'.join(cols)
                                print(new_line, file=output_file)

        elif six.PY2:
            with open(input_file_path) as input_file:
                with open(output_file_path, 'w') as output_file:
                    for i, l in enumerate(input_file.readlines()):
                        parser = choose_parser()
                        #print(parser.__name__ == "pos_replace_parser")
                        if i % 1000 == 0:
                            log.debug('parsing line %d' % i)
                        print(l.strip(), file=output_file)

                        for k in range(args.aug_times):
                            cols = l.strip().decode("utf8").split('\t')
                            for j in col_nums:
                                if parser.__name__ == "pos_replace_parser":
                                    cols[j - 1] = parser(cols[j - 1], pos_dict[j - 1])
                                    counter[parser.__name__] += 1
                                else:
                                    cols[j - 1] = parser(cols[j - 1])
                                    counter[parser.__name__] += 1
                                new_line = '\t'.join(cols)
                                print(new_line.encode("utf8"), file=output_file)
        print("counter", counter)


