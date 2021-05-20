#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""utils help and eval functions for text generation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import subprocess
from model.tokenization import GptBpeTokenizer, BasicTokenizer


class GenerationEval(object):
    """GenerationEval"""

    def __init__(self, args):
        self.eval_script = args.eval_script.split(" ")
        self.eval_mertrics = args.eval_mertrics.split(",") if args.eval_mertrics else []
        self.basic_tokenizer = BasicTokenizer(do_lower_case=True)
        self.roberta_tokenizer = GptBpeTokenizer(vocab_file=args.unimo_vocab_file,
                                                 encoder_json_file=args.encoder_json_file,
                                                 vocab_bpe_file=args.vocab_bpe_file,
                                                 do_lower_case=True)

    def eval(self, output_file, phase="", features=None):
        """run eval"""
        eval_res = {}
        if self.eval_script:
            eval_res = subprocess.check_output(self.eval_script + [output_file, phase])
            eval_res = json.loads(eval_res)
        else:
            preds = []
            for line in open(output_file):
                preds.append(self.basic_tokenizer.tokenize(line.strip()))

            refs = []
            for id in sorted(features.keys()):
                ref_str = self.roberta_tokenizer.gptbpe_tokenizer.decode(features[id].tgt.split(" "))
                refs.append([self.basic_tokenizer.tokenize(ref_str)])

            for mertric in self.eval_mertrics:
                eval_func = getattr(self, mertric, None)
                if eval_func:
                    eval_res[mertric] = eval_func(refs, preds)

        ret = []
        for mertric in self.eval_mertrics:
            mertric_res = eval_res.get(mertric, None)
            if mertric_res is None:
                raise Exception("Eval mertric: %s is not supported" % mertric)
            ret.append("%s: %f" % (mertric, mertric_res))

        return ", ".join(ret)

    def bleu(self, refs, preds):
        """bleu mertric"""
        return _compute_bleu(refs, preds, max_order=4)[0]


def _get_ngrams(segment, max_order):
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i: i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def _compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / (ratio + 1e-4))

    bleu = geo_mean * bp
    ret = [bleu, precisions, bp, ratio, translation_length, reference_length]
    return ret