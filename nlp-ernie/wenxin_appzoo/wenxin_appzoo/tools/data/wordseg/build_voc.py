#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" build dict interface """
import argparse
import os
from collections import defaultdict

import six

def build_dict(input_path,
               output_path,
               col_nums,
               feq_threshold=5,
               sep=' ',
               extra_words=None,
               stop_words=None):
    """build dict"""
    values = defaultdict(int)
    for file_name in os.listdir(input_path):
        file_path = os.path.join(input_path, file_name)
        if not os.path.isfile(file_path):
            continue

        if six.PY3:
            input_file = open(file_path, 'r', encoding='utf8')
        else:
            input_file = open(file_path, 'r')

        for i, l in enumerate(input_file.readlines()):
            cols = l.strip().split('\t')
            selected_cols = ""
            for j in col_nums:
                selected_cols += cols[j - 1]

            for w in selected_cols.split(sep):
                values[w] = values.get(w, 0) + 1

    output_file_path = os.path.join(output_path, "vocab.txt")
    id_index = 0
    with open(output_file_path, "w", encoding='utf8') as f:
        for v, count in sorted(values.items(), key=lambda x: x[1], reverse=True):
            if count < feq_threshold or v in stop_words:
                break
            # f.write("%s\t%d\n" % (v, count))
            f.write("%s\t%d\n" % (v, id_index))
            id_index += 1

        build_in_vocab = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
        for vocab in build_in_vocab:
            extra_words.insert(0, vocab)
        for w in extra_words:
            if (w in values and values[w] < feq_threshold) or w not in values:
                if six.PY3:
                    f.write((u"%s\t%d\n" % (w, id_index)))
                else:
                    f.write((u"%s\t%d\n" % (w, id_index)).encode('utf-8'))
                id_index += 1


def main():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-sep", "--seperator", type=str, default=' ')
    parser.add_argument("-c", "--column_number", type=str, default='1')
    parser.add_argument("-thr", "--feq_threshold", type=int, default='5')
    parser.add_argument("-ew", "--extra_words", type=str, nargs='+', default=[])
    parser.add_argument("-sw", "--stop_words", type=str, nargs='+', default=[])

    # 停用词

    args = parser.parse_args()

    col_nums = args.column_number.split(',')
    col_nums = list(map(int, col_nums))

    data_files = os.listdir(args.input)
    assert len(data_files) > 0, "%s is an empty directory" % args.input
    mkdirlambda = lambda x: os.makedirs(x) if not os.path.exists(x) else True
    mkdirlambda(args.output)

    build_dict(
        input_path=args.input,
        output_path=args.output,
        feq_threshold=args.feq_threshold,
        sep=' ',
        col_nums=col_nums,
        extra_words=args.extra_words,
        stop_words=args.stop_words)


if __name__ == '__main__':
    main()
