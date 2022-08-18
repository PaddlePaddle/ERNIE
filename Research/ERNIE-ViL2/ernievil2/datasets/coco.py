import collections
import os
import warnings
from paddle.io import Dataset
from paddle.utils.download import get_path_from_url
from ernievil2.utils.env import DATA_HOME
from paddle.io import Dataset
import os
import json
import random

__all__ = ['coco']


class coco(Dataset):
    '''
    This dataset coco.
    '''

    def __init__(self):
        print("coco init")

    def _read(self, filename, shuffle=False, seed=0):
        with open(filename) as f:
            for line in f:
                out = line.strip().split('\t')
                yield {
                    "src_lang": 'zh', "origin_src": out[0],
                    "img": out[-1], "img-type":"base64",
                    "filename": filename}