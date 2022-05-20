# -*- coding: utf-8 -*
"""import
运行任务之前的一些预处理工具
"""

import copy
import logging
import os
import shutil
from itertools import product

from sklearn.model_selection import KFold


def get_list_params(params):
    """
    处理value为list的params，将其拆分为dict并放入一个list中
    :param params: 待处理的参数
    :return: combination_list, single_param_dict
    Example
    --------
    params={key1:[values1, value2], key2:[value1, value2, value3]}
    combination_list格式[[{key1:value1}, {key1:value2}],[{key2:value1}, {key2:value2}, {key2:value3}]],
    single_param_dict:除去传入的dict中不为list格式的value
    """
    combination_list = []
    single_param_dict = {}
    for key, value in params.items():
        if isinstance(value, list):
            one_keys = []
            for v in value:
                item = {key: v}
                one_keys.append(item)
            combination_list.append(one_keys)
        else:
            single_param_dict[key] = value
    return combination_list, single_param_dict


def build_grid_search_config(params_dict):
    """
    传入一个json，按网格搜索的方式构造出符合条件的N个json, 目前网格搜索只作用在optimization范围内
    :param params_dict:
    :return: param_config_list
    """
    model_params_dict = params_dict.get("model")
    opt_params = model_params_dict.get("optimization", None)
    if not opt_params:
        raise ValueError("optimization's params can't be none")
    # 获取待网格搜索的dict
    train_data_params = params_dict.get("dataset_reader").get("train_reader").get("config", None)
    if not train_data_params:
        raise ValueError("train_reader config's params can't be none")

    # 在need_operate_params中加入待处理网格搜索的dict
    need_operate_params = [opt_params, train_data_params]
    all_combination_list = []
    all_single_param_dict = []
    dict_list_key_num = []
    for one_operate_param in need_operate_params:
        combination_list, single_param_dict = get_list_params(one_operate_param)
        all_combination_list.extend(combination_list)
        all_single_param_dict.append(single_param_dict)
        dict_list_key_num.append(len(combination_list))

    task_param_list = []
    for params in product(*all_combination_list):
        one_task_param = copy.deepcopy(params_dict)
        # 在need_update_param中加入待更新的网格搜索的dict，注意顺序要和need_operate_params保持一致
        need_update_param = [one_task_param["model"]["optimization"],
                             one_task_param["dataset_reader"]["train_reader"]["config"]]
        i = 0
        for index, one_single_param in enumerate(all_single_param_dict):
            single_param = copy.deepcopy(one_single_param)
            for one_grid in params[i:i + dict_list_key_num[index]]:
                single_param.update(one_grid)
            need_update_param[index].update(single_param)
            i += dict_list_key_num[index]
        task_param_list.append(one_task_param)

    return task_param_list


def process_data_with_kfold(data_path, output_path, num_split, num_use_split):
    """将原始数据平均分成K份，其中K-1份用来做训练集，1份用来做验证集。如此循环K次，得到K倍的原始数据。主要调用的是sk-learn的KFold
    :param data_path: 待处理数据集路径（必须是个目录）
    :param output_path: 处理之后生成的数据集路径（必须是个目录）
    :param num_split: 数据分成多少折，取值必须大于等于2
    :param num_use_split: 拆分好的N折数据，真正使用的数量，比如拆分成10折，但是只使用其中5份数据进行训练，
    那么num_split需要设置成10， num_use_split需要设置成5。num_use_split的取值必须小于等于num_split
    :return: 拆分之后的训练集和验证集数据路径
    """
    assert num_split >= 2, "k-fold cross-validation requires at least one, train/dev split by setting n_splits=2 " \
                           "or more, got num_split=%d" % num_split

    assert os.path.isdir(data_path), "%s must be a directory that stores data files" % data_path
    data_files = os.listdir(data_path)

    assert len(data_files) > 0, "%s is an empty directory" % data_path

    if num_use_split <= 0 or num_use_split > num_split:
        num_use_split = num_split

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    examples = []
    for one_file in data_files:
        input_file = os.path.join(data_path, one_file)
        with open(input_file, "r") as f:
            try:
                lines = f.readlines()
                examples.extend(lines)

            except Exception:
                logging.error("error in read tsv %s" % input_file)
        # examples.extend("\n")

    kf = KFold(n_splits=num_split)
    num = 0
    new_train_paths = []
    new_dev_paths = []
    for train_index, dev_index in kf.split(examples):
        if num >= num_use_split:
            return new_train_paths, new_dev_paths
        save_dir_train = os.path.join(output_path, "train_data_" + str(num))
        save_dir_dev = os.path.join(output_path, "dev_data_" + str(num))
        os.makedirs(save_dir_train)
        os.makedirs(save_dir_dev)
        train_file = os.path.join(save_dir_train, "train.txt")
        dev_file = os.path.join(save_dir_dev, "dev.txt")

        with open(train_file, "w") as f:
            for index in train_index:
                line = examples[index]
                f.write(line.rstrip("\n"))
                f.write("\n")

        with open(dev_file, "w") as f:
            for index in dev_index:
                line = examples[index]
                f.write(line.rstrip("\n"))
                f.write("\n")

        new_train_paths.append(save_dir_train)
        new_dev_paths.append(save_dir_dev)

        num += 1

    return new_train_paths, new_dev_paths


def build_kfold_config(params_dict, train_path, dev_path):
    """按k-fold拆分好的数据，构造新的json配置，用来启动训练任务
    :param params_dict: 原始json配置构造出来的param_dict
    :param train_path: k-fold拆分之后的训练集路径，list类型
    :param dev_path: k-fold拆分之后的评估集路径，list类型
    :return: task_param_list: 生成新的json配置，用来启动run_with_json
    """
    assert isinstance(train_path, list), "train_path must be list"
    assert isinstance(dev_path, list), "dev_path must be list"
    assert len(train_path) == len(dev_path), "len(train_path) must == len(dev_path)"
    if not params_dict.__contains__("dataset_reader"):
        raise ValueError("dataset_reader in json config can't be null")

    if not params_dict["dataset_reader"]["train_reader"]:
        raise ValueError("train_reader in json config can't be null")

    if not params_dict["dataset_reader"]["dev_reader"]:
        raise ValueError("dev_reader json config can't be null")

    task_param_list = []
    for index in range(len(train_path)):
        one_task_param = copy.deepcopy(params_dict)
        one_task_param["dataset_reader"]["train_reader"]["config"]["data_path"] = train_path[index]
        one_task_param["dataset_reader"]["dev_reader"]["config"]["data_path"] = dev_path[index]
        one_task_param["dataset_reader"]["dev_reader"]["config"]["shuffle"] = False
        one_task_param["dataset_reader"]["dev_reader"]["config"]["epoch"] = 1
        one_task_param["dataset_reader"]["dev_reader"]["config"]["sampling_rate"] = 1.0
        # 1.7版本去掉这两行设置，以用户的json配置为准；http://wiki.baidu.com/pages/viewpage.action?pageId=1292167804
        # one_task_param["trainer"]["is_eval_dev"] = 1
        # one_task_param["trainer"]["is_eval_test"] = 0
        task_param_list.append(one_task_param)

    return task_param_list
