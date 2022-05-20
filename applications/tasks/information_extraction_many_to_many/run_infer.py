# -*- coding: utf-8 -*
"""import"""
import os

import sys
sys.path.append("../../../")
import numpy as np
from erniekit.common.register import RegisterSet
from erniekit.common import register
from erniekit.data.data_set import DataSet
import logging
from erniekit.utils import args
from erniekit.utils import params
from erniekit.utils import log
import collections
import json



def dataset_reader_from_params(params_dict):
    """
    :param params_dict:
    :return:
    """
    dataset_reader = DataSet(params_dict)
    dataset_reader.build()

    return dataset_reader


def build_inference(params_dict, dataset_reader, parser_handler):
    """build trainer"""
    inference_name = params_dict.get("type", "CustomInference")
    inference_class = RegisterSet.inference.__getitem__(inference_name)
    inference = inference_class(params=params_dict, data_set_reader=dataset_reader, 
    parser_handler=parser_multi_infor_extraction)

    return inference

def parser_multi_infor_extraction(predict_result, sample_list, params_dict):
    """属性抽取的预测结果解析
    :param predict_result: 模型预测出来的结果
    :param sample_list: 被预测的样本原文
    :param params_dict: 一些参数配置
    """
    probs, seq_lens, beg_ids, end_ids = predict_result
    probs = np.array(probs.copy_to_cpu())
    seq_lens = np.array(seq_lens)
    beg_ids = np.array(beg_ids)
    end_ids = np.array(end_ids)

    max_seq_len = probs.shape[0] // seq_lens.shape[0]

    return_list = []

    label_map_config = params_dict.get("label_map_config")
    inv_label_map = _get_inv_label_map(label_map_config)

    for i in range(seq_lens.shape[0]):
        sample = sample_list[i]
        sample_beg_ids = beg_ids[i][: seq_lens[i]]
        sample_end_ids = end_ids[i][: seq_lens[i]]
        sample_probs = probs[max_seq_len * i: max_seq_len * i + seq_lens[i]]
        sample_probs = _post_proc(sample_probs)
        output = _gen_output(sample, sample_probs, sample_beg_ids, sample_end_ids, inv_label_map)
        return_list.append([output])
    return return_list


def _get_inv_label_map(label_map_config):
    """
    get inv lable map
    """
    with open(label_map_config, "r") as fp:
        label_map = json.load(fp, encoding="utf-8")
    inv_label_map = collections.OrderedDict()
    for key, value in label_map.items():
        inv_label_map[str(value)] = key
    return inv_label_map


def _post_proc(sample_probs):
    """
    post proc
    """
    sample_probs = np.where(sample_probs < 0.5, 0, 1)
    length, _ = sample_probs.shape

    for i in range(length - 1):
        if sample_probs[i][0] == 1 and np.sum(sample_probs[i]) > 1:
            if sample_probs[i + 1][1] == 1:
                sample_probs[i][0] = 0
            else:
                sample_probs[i][2:] = 0

    for i in range(length - 1):
        if np.sum(sample_probs[i]) == 0:
            if sample_probs[i - 1][1] == 1 and sample_probs[i + 1][1] == 1:
                sample_probs[i][1] = 1
            elif sample_probs[i + 1][1] == 1:
                sample_probs[i][np.argmax(sample_probs[i, 1:]) + 1] = 1
    return sample_probs

def _gen_output(sample, sample_probs, sample_beg_ids, sample_end_ids, inv_label_map):
    """
    :param sample:
    :param sample_probs:
    :param sample_beg_ids:
    :param sample_end_ids:
    :param inv_label_map:
    :return:
    """
    length, _ = sample_probs.shape
    sample_probs = sample_probs[1: length - 1]
    sample_beg_ids = sample_beg_ids[1: length - 1]
    sample_end_ids = sample_end_ids[1: length - 1]
    label_set = []
    for token_probs in zip(sample_probs):
        label_set.extend(np.argwhere(token_probs).flatten().tolist())
    label_set = list(set(label_set))

    cand_subject_label_set = []
    for label in label_set:
        if label != 0 and label != 1:
            if label % 2 == 0 and (label + 1) in label_set:
                cand_subject_label_set.append(label)

    def find_ent(label):
        """
        find ent
        """
        ents = []
        for i in range(len(sample_probs)):
            if sample_probs[i][label] == 1:
                j = 1
                while i + j < len(sample_probs):
                    if sample_probs[i + j][1] == 1:
                        j += 1
                    else:
                        break
                ents.append([int(sample_beg_ids[i]), int(sample_end_ids[i + j - 1] + 1)])
        return ents

    spo_list = []
    for cand_subject_label in cand_subject_label_set:
        subjects = find_ent(cand_subject_label)
        objects = find_ent(cand_subject_label + 1)
        if len(subjects) == 1 and len(objects) > 1:
            for object in objects:
                spo_list.append({"predicate": inv_label_map[str(cand_subject_label)][2:-2],
                                 "subject": subjects[0], "object": object})
        elif len(subjects) > 1 and len(objects) == 1:
            for subject in subjects:
                spo_list.append({"predicate": inv_label_map[str(cand_subject_label)][2:-2],
                                 "subject": subject, "object": objects[0]})
        else:
            for subject in subjects:
                nearest_object = []
                nearest_dist = 9999
                for object in objects:
                    dist = abs(object[0] - subject[0])
                    if dist < nearest_dist:
                        nearest_object = object
                        nearest_dist = dist
                spo_list.append({"predicate": inv_label_map[str(cand_subject_label)][2:-2],
                                 "subject": subject, "object": nearest_object})
    # TODO:为了能时python3和python2兼容，ensure_ascii设为true
    # return json.dumps({"text": sample, "spo_list": spo_list}, ensure_ascii=False)
    res_dic = {}
    res_dic["text"] = sample
    res_dic["spo_list"] = spo_list
    # return json.dumps({"text": sample, "spo_list": spo_list})
    return res_dic

if __name__ == "__main__":
    args = args.build_common_arguments()
    log.init_log("./log/test", level=logging.DEBUG)
    param_dict = params.from_file(args.param_path)
    # param_dict = params.from_file("./examples/cls_bow_ch_infer.json")
    _params = params.replace_none(param_dict)

    # 记得import一下注册的模块
    register.import_modules()
    register.import_new_module("inference", "custom_inference")
    register.import_new_module("data_set_reader", "ie_data_set_reader")

    dataset_reader_params_dict = _params.get("dataset_reader")
    dataset_reader = dataset_reader_from_params(dataset_reader_params_dict)

    inference_params_dict = _params.get("inference")
    inference = build_inference(inference_params_dict, dataset_reader, parser_multi_infor_extraction)

    inference.inference_batch()

    logging.info("os exit.")
    os._exit(0)