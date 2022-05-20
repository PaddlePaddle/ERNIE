# *_*coding:utf-8 *_*
""" MRR evaluation metric """
from paddle.fluid.layers.tensor import reverse
from erniekit.metrics.metrics import Metrics
import numpy as np
import collections


class MRR(Metrics):
    """MRR"""
    def __init__(self):
        super(MRR, self).__init__()

    def eval(self, run_value, is_two_tower=False):
        """eval"""
        print("evaluating", is_two_tower)
        if is_two_tower:
            return self.eval_twotower(run_value)
        else:
            return self.eval_pin(run_value)

    def eval_twotower(self, run_value):
        """eval_twotower"""
        h_cls_list, h_img_list = run_value
        h_cls, h_img = np.concatenate(
            h_cls_list, 0), np.concatenate(
            h_img_list, 0)
        score = np.matmul(h_cls, h_img.transpose())  # n * n
        n = score.shape[0]
        result = {}
        # text to image
        y = np.expand_dims(np.arange(n), 1) + \
            np.expand_dims(np.arange(n), 1) * n
        z = np.take(score, y)
        c = np.expand_dims(score, 2) > np.expand_dims(z, 1)
        ans_idx = c.sum(1).min(1)
        r1 = (ans_idx < 1).sum() / n
        r5 = (ans_idx < 5).sum() / n
        r10 = (ans_idx < 10).sum() / n
        mrr = (1.0 / (1.0 + ans_idx)).sum() / n
        mr = (r1 + r5 + r10) / 3.0
        result["text"] = {"r1": r1, "r5": r5, "r10": r10, "mR": mr, "mRR": mrr}
        # image to text
        score = score.transpose()
        y = np.expand_dims(np.arange(n), 1) + \
            np.expand_dims(np.arange(n), 1) * n
        z = np.take(score, y)
        c = np.expand_dims(score, 2) > np.expand_dims(z, 1)
        ans_idx = c.sum(1).min(1)
        r1 = (ans_idx < 1).sum() / n
        r5 = (ans_idx < 5).sum() / n
        r10 = (ans_idx < 10).sum() / n
        mrr = (1.0 / (1.0 + ans_idx)).sum() / n
        mr = (r1 + r5 + r10) / 3.0
        result["image"] = {
            "r1": r1,
            "r5": r5,
            "r10": r10,
            "mR": mr,
            "mRR": mrr}
        return result

    def eval_pin(self, run_value):
        """
        run_value = [match_score, match_in_score]
        match_score和match_in_score形状是batch * batch
        """
        match_all_score_list, image_index_list, text_index_list = run_value
        text_res_dict = collections.defaultdict(list)
        image_res_dict = collections.defaultdict(list)
        for match_all_score, image_index, text_index in zip(
                match_all_score_list, image_index_list, text_index_list
        ):
            # match_score, match_in_score = match_all_score[:,
            #                                               0], match_all_score[:, 1]
            image_index, text_index = (
                np.array(image_index),
                np.array(text_index),
            )
            scores = np.array(match_all_score)
            for score, img_idx, txt_idx in zip(
                    scores, image_index, text_index):
                text_res_dict[txt_idx].append((score, img_idx))
                image_res_dict[img_idx].append((score, txt_idx))
        number = len(text_res_dict)

        result = {}
        # text retrieval
        r1, r5, r10, idx_all, cnt = 0.0, 0.0, 0.0, 0.0, 0.0
        for idx, res_list in text_res_dict.items():
            if len(res_list) != number:
                return {}
            res_list = sorted(res_list, reverse=True)
            image_id_sort = [x[1] for x in res_list]
            ans_idx = image_id_sort.index(idx)
            if ans_idx < 1:
                r1 += 1
            if ans_idx < 5:
                r5 += 1
            if ans_idx < 10:
                r10 += 1
            idx_all += 1.0 / (ans_idx + 1)
            cnt += 1
        r1, r5, r10, mRR = r1 / cnt, r5 / cnt, r10 / cnt, idx_all / cnt
        mR = (r1 + r5 + r10) / 3.0
        result["text"] = {"r1": r1, "r5": r5, "r10": r10, "mR": mR, "mRR": mRR}
        # image retrieval
        r1, r5, r10, idx_all, cnt = 0.0, 0.0, 0.0, 0.0, 0.0
        for idx, res_list in image_res_dict.items():
            if len(res_list) != number:
                return {}
            res_list = sorted(res_list, reverse=True)
            text_id_sort = [x[1] for x in res_list]
            ans_idx = text_id_sort.index(idx)
            if ans_idx < 1:
                r1 += 1
            if ans_idx < 5:
                r5 += 1
            if ans_idx < 10:
                r10 += 1
            idx_all += 1.0 / (ans_idx + 1)
            cnt += 1
        r1, r5, r10, mRR = r1 / cnt, r5 / cnt, r10 / cnt, idx_all / cnt
        mR = (r1 + r5 + r10) / 3.0
        result["image"] = {
            "r1": r1,
            "r5": r5,
            "r10": r10,
            "mR": mR,
            "mRR": mRR}
        return result
