# -*- coding: utf-8 -*
"""
:py:lac
"""
import os
import argparse
import six
from LAC import LAC


def run_lac(text):
    """
    build pos dict for pos parser
    """
    if len(text) == 0:
        return text

    seg = lac.run(text)
    sep = " "

    if six.PY3:
        return sep.join(seg)
    else:
        # seg[i] is a list, and len(seg) = 1
        tmp = [w[0] for w in seg]
        return sep.join(tmp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-c", "--column_number", type=str, default='1')
    args = parser.parse_args()

    lac = LAC(mode="seg")

    col_nums = args.column_number.split(',')
    col_nums = list(map(int, col_nums))

    data_files = os.listdir(args.input)
    assert len(data_files) > 0, "%s is an empty directory" % args.input
    mkdirlambda = lambda x: os.makedirs(x) if not os.path.exists(x) else True
    mkdirlambda(args.output)
    for data_file in data_files:
        input_file_path = os.path.join(args.input, data_file)
        print("input_file_path", input_file_path)
        input_file_name, suffix = os.path.splitext(data_file)
        print(input_file_name, suffix)
        output_file_name = '{0}_seg{1}'.format(input_file_name, suffix)
        output_file_path = os.path.join(args.output, output_file_name)
        if six.PY3:
            input_file = open(input_file_path, 'r', encoding='utf8')
            output_file = open(output_file_path, 'w', encoding='utf8')
        else:
            input_file = open(input_file_path, 'r')
            output_file = open(output_file_path, 'w')

        for i, l in enumerate(input_file.readlines()):
            cols = l.strip().split('\t')
            for j in col_nums:
                seg = run_lac(cols[j - 1])
                cols[j - 1] = seg
            output_file.write('\t'.join(cols) + "\n")

        input_file.close()
        output_file.close()


