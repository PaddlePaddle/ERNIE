#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A tool to convert file between utf8 and gb18030 """
import argparse
import os
import sys
import chardet
import logging
import six
import subprocess

def check_file_encoding(file_path):
    """ check file encoding """
    with open(file_path, 'rb') as fd:
        encode_str = chardet.detect(fd.read())['encoding']
        logging.info("input_file: {0}, encoding: {1}".format(file_path, encode_str))
        return encode_str


def convert_gbk_to_utf8(gbk_file, utf8_file):
    """ convert file from gbk to utf8 """
    if not os.path.exists(gbk_file):
        logging.fatal("original file not exists: {0}".format(gbk_file))
        exit(-1)

    logging.info("begin to convert gbk file: {0} to utf8".format(gbk_file))
    if six.PY2:
        with open(gbk_file, "r") as fd, open(utf8_file, "w") as fd1:
            for line in fd:
                fd1.write(line.decode("gb18030", "ignore").encode("utf8", "ignore"))
    else:
        with open(gbk_file, "r", encoding="gb18030") as fd, open(utf8_file, "w", encoding="utf8") as fd1:
            for line in fd:
                fd1.write(line)


def convert_utf8_to_gb18030(utf8_file, gbk_file):
    """ convert file from utf8 to gbk """
    if not os.path.exists(utf8_file):
        logging.fatal("original file not exists: {0}".format(utf8_file))
        exit(-1)

    logging.info("begin to convert utf8 file: {0} to gbk: {1}".format(utf8_file, gbk_file))
    if six.PY2:
        with open(utf8_file, "r") as fd, open(gbk_file, "w") as fd1:
            for line in fd:
                fd1.write(line.decode("utf8", "ignore").encode("gb18030", "ignore"))
    else:
        with open(utf8_file, "r", encoding="utf8") as fd, open(gbk_file, "w", encoding="gb18030") as fd1:
            for line in fd:
                fd1.write(line)


def main():
    """ main function """
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type=str, default="dest_file")
    parser.add_argument("-g2u", "--gb18030_to_utf8", action='store_true')
    parser.add_argument("-u2g", "--utf8_to_gb18030", action='store_true')
    args = parser.parse_args()

    input_file = args.input
    if not os.path.exists(input_file):
        logging.fatal("input file: {0} not exist".format(input_file))
        exit(-1)

    encode_str = check_file_encoding(input_file)
    if args.gb18030_to_utf8:
        if encode_str != "GB2312" and encode_str != "gb18030" and encode_str != "cp936":
            logging.fatal("input_file: {0} encoding is not gbk，failed to convert utf8!")
            exit(-1)

        convert_gbk_to_utf8(input_file, args.output)
    else:
        if args.utf8_to_gb18030:
            if encode_str != "utf-8" and encode_str != "ascii":
                cmd = "file {0} |grep 'UTF-8'".format(input_file)
                p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                stdoutdata, stderrdata = p.communicate()
                if not stdoutdata:
                    logging.fatal("input_file: {0} encoding is not utf8，failed to convert gb18030!")
                    exit(-1)
            
            convert_utf8_to_gb18030(input_file, args.output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s [line:%(lineno)d] [%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S',
                        stream=sys.stdout,
                        filemode='w')
    main()
