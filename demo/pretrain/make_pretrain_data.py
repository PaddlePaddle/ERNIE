import sys
import argparse
import struct
import random as r
import re
import gzip
import logging
from itertools import accumulate
from functools import reduce, partial, wraps
from propeller import log
from propeller.paddle.data import feature_pb2, example_pb2
#jfrom data_util import RawtextColumn

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def gen_segs(segment_piece):
    if len(segment_piece) == 0:
        return []
    else:
        return [min(segment_piece)] * len(segment_piece)


whit_space_pat = re.compile(r'\S+')


def segment(inputs, inputs_segment):
    ret = [r.span() for r in whit_space_pat.finditer(inputs)]
    ret = [(inputs[s:e], gen_segs(inputs_segment[s:e]))
           for i, (s, e) in enumerate(ret)]
    return ret


def tokenize(sen, seg_info):
    """
    char tokenizer (wordpiece english)
    normed txt(space seperated or not) => list of word-piece
    """
    sen = sen.lower()
    res_word, res_segments = [], []
    for match in pat.finditer(sen):
        words, pos = _wordpiece(
            match.group(0), vocab=vocab_set, unk_token='[UNK]')
        start_of_word = match.span()[0]
        for w, p in zip(words, pos):
            res_word.append(w)
            res_segments.append(
                gen_segs(seg_info[p[0] + start_of_word:p[1] + start_of_word]))
    return res_word, res_segments


def parse_txt(line):
    if len(line) == 0:
        return []
    line = line.decode('utf8')
    ret_line, ret_seginfo = [], []

    for l, i in segment(line, list(range(len(line)))):
        for ll, ii in zip(*tokenize(l, i)):
            ret_line.append(ll)
            ret_seginfo.append(ii)

    if args.check and r.random() < 0.005:
        print('****', file=sys.stderr)
        print(line, file=sys.stderr)
        print('|'.join(ret_line), file=sys.stderr)
        print(ret_seginfo, file=sys.stderr)
        print('****', file=sys.stderr)

    ret_line = [vocab.get(r, vocab['[UNK]']) for r in ret_line]
    ret_seginfo = [[-1] if i == [] else i
                   for i in ret_seginfo]  #for sentence piece only
    ret_seginfo = [min(i) for i in ret_seginfo]
    return ret_line, ret_seginfo


def build_example(slots):
    txt, seginfo = slots
    txt_fe_list = feature_pb2.FeatureList(feature=[
        feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=t))
        for t in txt
    ])
    segsinfo_fe_list = feature_pb2.FeatureList(feature=[
        feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=s))
        for s in seginfo
    ])
    assert len(txt_fe_list.feature) == len(
        segsinfo_fe_list.feature), 'txt[%d] and seginfo[%d] size not match' % (
            len(txt_fe_list.feature), len(segsinfo_fe_list.feature))
    features = {
        'txt': txt_fe_list,
        'segs': segsinfo_fe_list,
    }

    ex = example_pb2.SequenceExample(feature_lists=feature_pb2.FeatureLists(
        feature_list=features))
    return ex


def write_gz(serialized, to_file):
    l = len(serialized)
    packed_data = struct.pack('i%ds' % l, l, serialized)
    to_file.write(packed_data)


def build_bb(from_file, to_file):
    slots = []
    for i, line in enumerate(from_file):
        line = line.strip()
        if args.verbose and i % 10000 == 0:
            log.debug(i)
        if len(line) == 0:
            if len(slots) != 0:
                transposed_slots = list(zip(*slots))
                ex = build_example(transposed_slots)
                write_gz(ex.SerializeToString(), to_file)
                slots = []
            continue
        parsed_line = parse_txt(line)
        slots.append(parsed_line)

    if len(slots) != 0:
        transposed_slots = list(zip(*slots))
        ex = build_example(transposed_slots)
        write_gz(ex.SerializeToString(), to_file)
        slots = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pretrain Data Maker')
    parser.add_argument('src', type=str)
    parser.add_argument('tgt', type=str)
    parser.add_argument('--vocab', type=str, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-c', '--check', action='store_true')

    args = parser.parse_args()
    log.setLevel(logging.DEBUG)

    from ernie.tokenizing_ernie import _wordpiece
    pat = re.compile(r'([a-zA-Z0-9]+|\S)')

    vocab = {
        j.strip().split(b'\t')[0].decode('utf8'): i
        for i, j in enumerate(open(args.vocab, 'rb'))
    }
    vocab_set = set(vocab.keys())

    with open(args.src, 'rb') as from_file, gzip.open(args.tgt,
                                                      'wb') as to_file:
        log.info('making gz from bb %s ==> %s' % (from_file, to_file))
        build_bb(from_file, to_file)
        log.info('done: %s' % to_file)
