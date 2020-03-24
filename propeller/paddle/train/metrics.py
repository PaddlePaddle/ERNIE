#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""predefined metrics"""

import sys
import os
import six

import numpy as np
import itertools
import logging

import paddle.fluid as F
import paddle.fluid.layers as L
import sklearn.metrics

log = logging.getLogger(__name__)

__all__ = [
    'Metrics', 'F1', 'Recall', 'Precision', 'Mrr', 'Mean', 'Acc', 'ChunkF1',
    'RecallAtPrecision'
]


class Metrics(object):
    """Metrics base class"""

    def __init__(self):
        """doc"""
        self.saver = []

    @property
    def tensor(self):
        """doc"""
        pass

    def update(self, *args):
        """doc"""
        pass

    def eval(self):
        """doc"""
        pass


class Mean(Metrics):
    """doc"""

    def __init__(self, t):
        """doc"""
        self.t = t
        self.reset()

    def reset(self):
        """doc"""
        self.saver = np.array([])

    @property
    def tensor(self):
        """doc"""
        self.t.persistable = True
        return self.t,

    def update(self, args):
        """doc"""
        t, = args
        t = t.reshape([-1])
        self.saver = np.concatenate([self.saver, t])

    def eval(self):
        """doc"""
        return self.saver.mean()


class Ppl(Mean):
    """doc"""

    def eval(self):
        """doc"""
        return np.exp(self.saver.mean())


class Acc(Mean):
    """doc"""

    def __init__(self, label, pred):
        """doc"""
        self.eq = L.equal(pred, label)
        self.reset()

    @property
    def tensor(self):
        """doc"""
        self.eq.persistable = True
        return self.eq,


class MSE(Mean):
    """doc"""

    def __init__(self, label, pred):
        """doc"""
        diff = pred - label
        self.mse = diff * diff
        self.reset()

    @property
    def tensor(self):
        """doc"""
        self.mse.persistable = True
        return self.mse,


class Cosine(Mean):
    """doc"""

    def __init__(self, label, pred):
        """doc"""
        self.cos = L.cos_sim(label, pred)
        self.reset()

    @property
    def tensor(self):
        """doc"""
        self.cos.persistable = True
        return self.cos,


class Precision(Metrics):
    """doc"""

    def __init__(self, label, pred):
        """doc"""
        self.label = label
        self.pred = pred
        self.reset()

    def reset(self):
        """doc"""
        self.label_saver = np.array([], dtype=np.bool)
        self.pred_saver = np.array([], dtype=np.bool)

    @property
    def tensor(self):
        """doc"""
        self.label.persistable = True
        self.pred.persistable = True
        return self.label, self.pred

    def update(self, args):
        """doc"""
        label, pred = args
        label = label.reshape([-1]).astype(np.bool)
        pred = pred.reshape([-1]).astype(np.bool)
        if label.shape != pred.shape:
            raise ValueError(
                'Metrics precesion: input not match: label:%s pred:%s' %
                (label, pred))
        self.label_saver = np.concatenate([self.label_saver, label])
        self.pred_saver = np.concatenate([self.pred_saver, pred])

    def eval(self):
        """doc"""
        tp = (self.label_saver & self.pred_saver).astype(np.int64).sum()
        t = self.label_saver.astype(np.int64).sum()
        return tp / t


class Recall(Precision):
    """doc"""

    def eval(self):
        """doc"""
        tp = (self.label_saver & self.pred_saver).astype(np.int64).sum()
        p = (self.label_saver).astype(np.int64).sum()
        return tp / p


class F1(Precision):
    """doc"""

    def eval(self):
        """doc"""
        tp = (self.label_saver & self.pred_saver).astype(np.int64).sum()
        t = self.label_saver.astype(np.int64).sum()
        p = self.pred_saver.astype(np.int64).sum()
        precision = tp / (t + 1.e-6)
        recall = tp / (p + 1.e-6)
        return 2 * precision * recall / (precision + recall + 1.e-6)


class Auc(Metrics):
    """doc"""

    def __init__(self, label, pred):
        """doc"""
        self.pred = pred
        self.label = label
        self.reset()

    def reset(self):
        """doc"""
        self.pred_saver = np.array([], dtype=np.float32)
        self.label_saver = np.array([], dtype=np.bool)

    @property
    def tensor(self):
        """doc"""
        self.pred.persistable = True
        self.label.persistable = True
        return [self.pred, self.label]

    def update(self, args):
        """doc"""
        pred, label = args
        pred = pred.reshape([-1]).astype(np.float32)
        label = label.reshape([-1]).astype(np.bool)
        self.pred_saver = np.concatenate([self.pred_saver, pred])
        self.label_saver = np.concatenate([self.label_saver, label])

    def eval(self):
        """doc"""
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(
            self.label_saver.astype(np.int64), self.pred_saver)
        auc = sklearn.metrics.auc(fpr, tpr)
        return auc


class RecallAtPrecision(Auc):
    """doc"""

    def __init__(self, label, pred, precision=0.9):
        """doc"""
        super(RecallAtPrecision, self).__init__(label, pred)
        self.precision = precision

    def eval(self):
        """doc"""
        self.pred_saver = self.pred_saver.reshape(
            [self.label_saver.size, -1])[:, -1]
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
            self.label_saver, self.pred_saver)
        for p, r in zip(precision, recall):
            if p > self.precision:
                return r


class PrecisionAtThreshold(Auc):
    """doc"""

    def __init__(self, label, pred, threshold=0.5):
        """doc"""
        super().__init__(label, pred)
        self.threshold = threshold

    def eval(self):
        """doc"""
        infered = self.pred_saver > self.threshold
        correct_num = np.array(infered & self.label_saver).sum()
        infer_num = infered.sum()
        return correct_num / (infer_num + 1.e-6)


class Mrr(Metrics):
    """doc"""

    def __init__(self, qid, label, pred):
        """doc"""
        self.qid = qid
        self.label = label
        self.pred = pred
        self.reset()

    def reset(self):
        """doc"""
        self.qid_saver = np.array([], dtype=np.int64)
        self.label_saver = np.array([], dtype=np.int64)
        self.pred_saver = np.array([], dtype=np.float32)

    @property
    def tensor(self):
        """doc"""
        self.qid.persistable = True
        self.label.persistable = True
        self.pred.persistable = True
        return [self.qid, self.label, self.pred]

    def update(self, args):
        """doc"""
        qid, label, pred = args
        if not (qid.shape[0] == label.shape[0] == pred.shape[0]):
            raise ValueError(
                'Mrr dimention not match: qid[%s] label[%s], pred[%s]' %
                (qid.shape, label.shape, pred.shape))
        self.qid_saver = np.concatenate(
            [self.qid_saver, qid.reshape([-1]).astype(np.int64)])
        self.label_saver = np.concatenate(
            [self.label_saver, label.reshape([-1]).astype(np.int64)])
        self.pred_saver = np.concatenate(
            [self.pred_saver, pred.reshape([-1]).astype(np.float32)])

    def eval(self):
        """doc"""

        def _key_func(tup):
            return tup[0]

        def _calc_func(tup):
            ranks = [
                1. / (rank + 1.)
                for rank, (_, l, p) in enumerate(
                    sorted(
                        tup, key=lambda t: t[2], reverse=True)) if l != 0
            ]
            if len(ranks):
                return ranks[0]
            else:
                return 0.

        mrr_for_qid = [
            _calc_func(tup)
            for _, tup in itertools.groupby(
                sorted(
                    zip(self.qid_saver, self.label_saver, self.pred_saver),
                    key=_key_func),
                key=_key_func)
        ]
        mrr = np.float32(sum(mrr_for_qid) / len(mrr_for_qid))
        return mrr


class ChunkF1(Metrics):
    """doc"""

    def __init__(self, label, pred, seqlen, num_label):
        """doc"""
        self.label = label
        self.pred = pred
        self.seqlen = seqlen
        self.null_index = num_label - 1
        self.label_cnt = 0
        self.pred_cnt = 0
        self.correct_cnt = 0

    def _extract_bio_chunk(self, seq):
        chunks = []
        cur_chunk = None

        for index in range(len(seq)):
            tag = seq[index]
            tag_type = tag // 2
            tag_pos = tag % 2

            if tag == self.null_index:
                if cur_chunk is not None:
                    chunks.append(cur_chunk)
                    cur_chunk = None
                continue

            if tag_pos == 0:
                if cur_chunk is not None:
                    chunks.append(cur_chunk)
                    cur_chunk = {}
                cur_chunk = {"st": index, "en": index + 1, "type": tag_type}
            else:
                if cur_chunk is None:
                    cur_chunk = {
                        "st": index,
                        "en": index + 1,
                        "type": tag_type
                    }
                    continue

                if cur_chunk["type"] == tag_type:
                    cur_chunk["en"] = index + 1
                else:
                    chunks.append(cur_chunk)
                    cur_chunk = {
                        "st": index,
                        "en": index + 1,
                        "type": tag_type
                    }

        if cur_chunk is not None:
            chunks.append(cur_chunk)
        return chunks

    def reset(self):
        """doc"""
        self.label_cnt = 0
        self.pred_cnt = 0
        self.correct_cnt = 0

    @property
    def tensor(self):
        """doc"""
        self.pred.persistable = True
        self.label.persistable = True
        self.seqlen.persistable = True
        return [self.pred, self.label, self.seqlen]

    def update(self, args):
        """doc"""
        pred, label, seqlen = args
        pred = pred.reshape([-1]).astype(np.int32).tolist()
        label = label.reshape([-1]).astype(np.int32).tolist()
        seqlen = seqlen.reshape([-1]).astype(np.int32).tolist()

        max_len = 0
        for l in seqlen:
            max_len = max(max_len, l)

        for i in range(len(seqlen)):
            seq_st = i * max_len + 1
            seq_en = seq_st + (seqlen[i] - 2)
            pred_chunks = self._extract_bio_chunk(pred[seq_st:seq_en])
            label_chunks = self._extract_bio_chunk(label[seq_st:seq_en])
            self.pred_cnt += len(pred_chunks)
            self.label_cnt += len(label_chunks)

            pred_index = 0
            label_index = 0
            while label_index < len(label_chunks) and pred_index < len(
                    pred_chunks):
                if pred_chunks[pred_index]['st'] < label_chunks[label_index][
                        'st']:
                    pred_index += 1
                elif pred_chunks[pred_index]['st'] > label_chunks[label_index][
                        'st']:
                    label_index += 1
                else:
                    if pred_chunks[pred_index]['en'] == label_chunks[label_index]['en'] \
                            and pred_chunks[pred_index]['type'] == label_chunks[label_index]['type']:
                        self.correct_cnt += 1
                    pred_index += 1
                    label_index += 1

    def eval(self):
        """doc"""
        if self.pred_cnt == 0:
            precision = 0.0
        else:
            precision = 1.0 * self.correct_cnt / self.pred_cnt

        if self.label_cnt == 0:
            recall = 0.0
        else:
            recall = 1.0 * self.correct_cnt / self.label_cnt

        if self.correct_cnt == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return np.float32(f1)


class PNRatio(Metrics):
    """doc"""

    def __init__(self, qid, label, pred):
        """doc"""
        self.qid = qid
        self.label = label
        self.pred = pred
        self.saver = {}

    def reset(self):
        """doc"""
        self.saver = {}

    @property
    def tensor(self):
        """doc"""
        self.qid.persistable = True
        self.label.persistable = True
        self.pred.persistable = True
        return [self.qid, self.label, self.pred]

    def update(self, args):
        """doc"""
        qid, label, pred = args
        if not (qid.shape[0] == label.shape[0] == pred.shape[0]):
            raise ValueError('dimention not match: qid[%s] label[%s], pred[%s]'
                             % (qid.shape, label.shape, pred.shape))
        qid = qid.reshape([-1]).tolist()
        label = label.reshape([-1]).tolist()
        pred = pred.reshape([-1]).tolist()
        assert len(qid) == len(label) == len(pred)
        for q, l, p in zip(qid, label, pred):
            if q not in self.saver:
                self.saver[q] = []
            self.saver[q].append((l, p))

    def eval(self):
        """doc"""
        p = 0
        n = 0
        for qid, outputs in self.saver.items():
            for i in range(0, len(outputs)):
                l1, p1 = outputs[i]
                for j in range(i + 1, len(outputs)):
                    l2, p2 = outputs[j]
                    if l1 > l2:
                        if p1 > p2:
                            p += 1
                        elif p1 < p2:
                            n += 1
                    elif l1 < l2:
                        if p1 < p2:
                            p += 1
                        elif p1 > p2:
                            n += 1
        pn = p / n if n > 0 else 0.0
        return np.float32(pn)


class BinaryPNRatio(PNRatio):
    """doc"""

    def __init__(self, qid, label, pred):
        """doc"""
        super(BinaryPNRatio, self).__init__(qid, label, pred)

    def eval(self):
        """doc"""
        p = 0
        n = 0
        for qid, outputs in self.saver.items():
            pos_set = []
            neg_set = []
            for label, score in outputs:
                if label == 1:
                    pos_set.append(score)
                else:
                    neg_set.append(score)

            for ps in pos_set:
                for ns in neg_set:
                    if ps > ns:
                        p += 1
                    elif ps < ns:
                        n += 1
                    else:
                        continue
        pn = p / n if n > 0 else 0.0
        return np.float32(pn)


class PrecisionAtK(Metrics):
    """doc"""

    def __init__(self, qid, label, pred, k=1):
        """doc"""
        self.qid = qid
        self.label = label
        self.pred = pred
        self.k = k
        self.saver = {}

    def reset(self):
        """doc"""
        self.saver = {}

    @property
    def tensor(self):
        """doc"""
        self.qid.persistable = True
        self.label.persistable = True
        self.pred.persistable = True
        return [self.qid, self.label, self.pred]

    def update(self, args):
        """doc"""
        qid, label, pred = args
        if not (qid.shape[0] == label.shape[0] == pred.shape[0]):
            raise ValueError('dimention not match: qid[%s] label[%s], pred[%s]'
                             % (qid.shape, label.shape, pred.shape))
        qid = qid.reshape([-1]).tolist()
        label = label.reshape([-1]).tolist()
        pred = pred.reshape([-1]).tolist()

        assert len(qid) == len(label) == len(pred)
        for q, l, p in zip(qid, label, pred):
            if q not in self.saver:
                self.saver[q] = []
            self.saver[q].append((l, p))

    def eval(self):
        """doc"""
        right = 0
        total = 0
        for v in self.saver.values():
            v = sorted(v, key=lambda x: x[1], reverse=True)
            k = min(self.k, len(v))
            for i in range(k):
                if v[i][0] == 1:
                    right += 1
                    break
            total += 1

        return np.float32(1.0 * right / total)


#class SemanticRecallMetrics(Metrics):
#    def __init__(self, qid, vec, type_id):
#        self.qid = qid
#        self.vec = vec
#        self.type_id = type_id
#        self.reset()
#
#    def reset(self):
#        self.saver = []
#
#    @property
#    def tensor(self):
#        return [self.qid, self.vec, self.type_id]
#
#    def update(self, args):
#        qid, vec, type_id = args
#        self.saver.append((qid, vec, type_id))
#
#    def eval(self):
#        dic = {}
#        for qid, vec, type_id in self.saver():
#            dic.setdefault(i, {}).setdefault(k, []).append(vec)
#        
#        for qid in dic:
#            assert len(dic[qid]) == 3
#            qvec = np.arrray(dic[qid][0])
#            assert len(qvec) == 1
#            ptvec = np.array(dic[qid][1])
#            ntvec = np.array(dic[qid][2])
#
#            np.matmul(qvec, np.transpose(ptvec))
#            np.matmul(qvec, np.transpose(ntvec))
#            
