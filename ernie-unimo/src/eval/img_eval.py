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
"""ultis help and eval functions for image/text retrieval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import OrderedDict


def recall_at_k(score_matrix, text2img, img2texts):
    """recall@k"""
    assert score_matrix.shape[0] == len(text2img) * len(img2texts)
    cur_img, cur_cap = score_matrix[:, 1], score_matrix[:, 2]
    img_len, cap_len = len(np.unique(cur_img)), len(np.unique(cur_cap))
    
    cur_img_sort = np.reshape(np.argsort(cur_img), [-1, cap_len])
    cur_cap_sort = np.reshape(np.argsort(cur_cap), [-1, img_len])
    i2c = np.take(score_matrix, cur_img_sort, axis=0) # img_len x cap_len x 3
    c2i = np.take(score_matrix, cur_cap_sort, axis=0) # cap_len x img_len x 3

    def get_recall_k(scores, idx, label_dict):
        """
        scores: sample x len x 5
        idx: 1 means text retrieval(i2c), 2 means image retrieval(c2i)
        """
        cand_idx_dict = {1: 2, 2: 1}
        cand_idx = cand_idx_dict[idx]
        tot = scores.shape[0]
        r1, r5, r10, rank_tot = 0, 0, 0, 0

        for i in range(tot):
            score_mat = scores[i]
            cur_ids = score_mat[0][idx]
            ans_ids = label_dict[cur_ids] # when idx is 1, type is list. idx is 2, type is int

            score = score_mat[:, 0]
            score_sort = np.argsort(score)[::-1]
            cand_ans = np.take(score_mat[:, cand_idx], score_sort, axis=0)
            cand_ans = cand_ans.astype(np.int64)

            if isinstance(ans_ids, list):
                rank = min([np.where(cand_ans == ans)[0] for ans in ans_ids])
            elif isinstance(ans_ids, int):
                rank = np.where(cand_ans == ans_ids)[0]
            else:
                raise ValueError('type error')
            if rank < 1:
                r1 += 1.0
            if rank < 5:
                r5 += 1.0
            if rank < 10:
                r10 += 1.0
            rank_tot += (rank + 1)
        ret = {
                'recall@1': float(r1)/tot,
                'recall@5': float(r5)/tot,
                'recall@10': float(r10)/tot,
                'avg_rank': float(rank_tot)/tot
              }
        return ret

    cap_retrieval_recall = get_recall_k(i2c, 1, img2texts)
    img_retrieval_recall = get_recall_k(c2i, 2, text2img)

    ret = OrderedDict()
    ret['img_avg_rank'] = img_retrieval_recall['avg_rank']
    ret['cap_avg_rank'] = cap_retrieval_recall['avg_rank']

    ret['img_recall@1'] = img_retrieval_recall['recall@1']
    ret['img_recall@5'] = img_retrieval_recall['recall@5']
    ret['img_recall@10'] = img_retrieval_recall['recall@10']

    ret['cap_recall@1'] = cap_retrieval_recall['recall@1']
    ret['cap_recall@5'] = cap_retrieval_recall['recall@5']
    ret['cap_recall@10'] = cap_retrieval_recall['recall@10']

    ret['avg_img_recall'] = (img_retrieval_recall['recall@1'] + \
            img_retrieval_recall['recall@5'] + img_retrieval_recall['recall@10']) /3
    ret['avg_cap_recall'] = (cap_retrieval_recall['recall@1'] + \
            cap_retrieval_recall['recall@5'] + cap_retrieval_recall['recall@10']) /3

    ret['avg_recall@1'] = (img_retrieval_recall['recall@1'] + cap_retrieval_recall['recall@1']) /2
    ret['avg_recall@5'] = (img_retrieval_recall['recall@5'] + cap_retrieval_recall['recall@5']) /2
    ret['avg_recall@10'] = (img_retrieval_recall['recall@10'] + cap_retrieval_recall['recall@10']) /2

    ret['key_eval'] = "avg_recall@1"
    return ret