# -*- coding: utf-8 -*
"""import"""
import os
import sys
sys.path.append("../../../")
import numpy as np
from erniekit.common.register import RegisterSet
from erniekit.common import register
from erniekit.data.data_set_ernie3 import DataSet
import logging
from erniekit.utils import args
from erniekit.utils import params
from erniekit.utils import log

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
    inference = inference_class(params=params_dict, data_set_reader=dataset_reader, parser_handler=parser_handler)

    return inference

def model_from_params(params_dict, dataset_reader):
    """
    :param params_dict:
    :param dataset_reader
    :return:
    """
    model_name = params_dict.get("type")
    model_class = RegisterSet.models.__getitem__(model_name)
    model = model_class(dataset_reader.predict_reader, params_dict)
    return model

if __name__ == "__main__":
    args = args.build_common_arguments()
    log.init_log("./log/test", level=logging.DEBUG)
    # 分类任务的预测
    param_dict = params.from_file(args.param_path)
    _params = params.replace_none(param_dict)

    # 记得import一下注册的模块
    register.import_modules()
    register.import_new_module("inference", "custom_inference")

    register.import_new_module("data_set_reader", "ernie_gen_infilling_dataset_reader")
    register.import_new_module("model", "ernie_infilling_generation")

    dataset_reader_params_dict = _params.get("dataset_reader")
    dataset_reader = dataset_reader_from_params(dataset_reader_params_dict)

    model_params_dict = param_dict.get("model")
    model = model_from_params(model_params_dict, dataset_reader)

    inference_params_dict = _params.get("inference")
    inference = build_inference(inference_params_dict, dataset_reader, model.parse_predict_result)

    inference.inference_batch()

    logging.info("os exit.")
    os._exit(0)
