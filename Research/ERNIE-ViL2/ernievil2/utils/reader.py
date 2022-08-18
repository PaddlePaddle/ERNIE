# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import itertools
from functools import partial
import numpy as np
from ernievil2.utils.preprocess_image import decode_image_base64
from paddle.io import BatchSampler, DataLoader
import paddle.distributed as dist
from ernievil2.data import Pad
from ernievil2.datasets import load_dataset, MapDataset
from ernievil2.utils.tokenizer import FullTokenizer
from ernievil2.data.sampler import SamplerHelper
def _load_map_dataset(args, fileslist):
    if 'coco' in fileslist:
        examples = load_dataset(
            "coco",
            fileslist=fileslist)
    else:
        raise ValueError(">>>[load_map_dataset] not support {}".format(fileslist))
    return MapDataset(examples)

def create_loader(args):
    dataset = _load_map_dataset(args, args.test_file)
    if args.vocab_file is not None:
        tokenizer = FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    else:
        raise ValueError(">>> hello_debug [reader] not support args.vocab_file is None")
    args.vocab_size = len(tokenizer.vocab)

    def convert_samples(sample):
        text_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample["origin_src"]))
        image = sample["img"]
        ret, image = decode_image_base64(image,data_augument=True,mode='test')
        return text_id, image
    dataset = dataset.map(convert_samples, lazy=False)
    batch_sampler = SamplerHelper(dataset).batch(
        batch_size=args.infer_batch_size, drop_last=False)
    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=partial(
            prepare_input,
            args=args),
        num_workers=0,
        return_list=True)
    return data_loader

def prepare_input(insts,args):
    word_pad = Pad(args.pad_token, dtype=args.input_dtype)  
    origin_word = word_pad([
        [args.cls_token] + inst[0][:args.max_text_seqlen-2] + [args.sep_token]+ [args.pad_token] * (args.max_text_seqlen - 2 - len(inst[0]))
        for inst in insts
    ])
    pos_ids = np.array([list(range(len(x))) for x in origin_word])
    img_word=[inst[-1] for inst in insts]
    return  img_word, origin_word, pos_ids
