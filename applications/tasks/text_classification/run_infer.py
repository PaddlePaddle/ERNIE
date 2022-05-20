# -*- coding: utf-8 -*
"""import"""
import sys
sys.path.append("../../../")
import os
import numpy as np
from erniekit.common.register import RegisterSet
from erniekit.common import register
from erniekit.data.data_set import DataSet
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


def parse_predict_result(predict_result, sample_list, params_dict):
    """按需解析模型预测出来的结果
    :param predict_result: 模型预测出来的结果
    :param sample_list: 样本明文数据，namedtuple类型
    :param params_dict: 一些参数配置
    :return:list 希望保存到文件中的结果，output/predict_result.txt
    """
    num_labels = params_dict.get("num_labels", 2)
    output_tensor = predict_result[0]
    output_data = output_tensor.copy_to_cpu()

    batch_result = np.array(output_data).reshape((-1, num_labels))
    return_list = []
    for index, item_prob in enumerate(batch_result):
        return_list.append(('\t'.join(sample_list[index]), str(item_prob.tolist())))
        logging.info(return_list[index])


    return return_list


if __name__ == "__main__":
    args = args.build_common_arguments()
    log.init_log("./log/test", level=logging.DEBUG)
    # 分类任务的预测
    param_dict = params.from_file(args.param_path)
    _params = params.replace_none(param_dict)

    # 记得import一下注册的模块
    register.import_modules()
    register.import_new_module("inference", "custom_inference")

    dataset_reader_params_dict = _params.get("dataset_reader")
    dataset_reader = dataset_reader_from_params(dataset_reader_params_dict)

    inference_params_dict = _params.get("inference")
    inference = build_inference(inference_params_dict, dataset_reader, parse_predict_result)

    inference.inference_batch()

    logging.info("os exit.")
    os._exit(0)
