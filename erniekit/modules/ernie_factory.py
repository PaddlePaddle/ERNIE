# -*- coding: utf-8 -*
"""
1.构造ERNIE model 实例
2.load ERNIE 参数

维护一个dict去存储所有ernie模型的信息
"""
import logging
import os
import paddle
from paddle.utils.download import get_path_from_url
from ..common import register
from ..common.register import RegisterSet
from .ernie_config import ErnieConfig

DOWNLOAD_BASE_URL = "http://bj.bcebos.com/wenxin-models/"

ernie_resource_dict = {
    "ernie_2.0_base_ch": {
        "dir_name": "ernie_2.0_base_ch",
        "vocab_file": "vocab.txt",
        "tokenizer": "FullTokenizer",
        "net_name": "ErnieModel",
        "config_file": "ernie_config.json",
        "params_file": "params"
    },
    "ernie_2.0_large_ch": {
        "dir_name": "ernie_2.0_large_ch",
        "vocab_file": "vocab.txt",
        "tokenizer": "FullTokenizer",
        "net_name": "ErnieModel",
        "config_file": "ernie_config.json",
        "params_file": "params"
    }
}


def generate_ernie_tokenizer(model_name, root_dir):
    """generate ernie tokenizer
    :param model_name: 待获取（下载）的预训练模型的名字，取值为ernie_resource_dict中包含的名称
    :param root_dir: 获取之后的预训吗模型及配置文件存放的位置
    """
    if not model_name or not root_dir:
        raise RuntimeError("model_name and root_dir can't be None")

    url = DOWNLOAD_BASE_URL + model_name + ".tgz"
    base_path = get_path_from_url(url=url, root_dir=root_dir)

    if not os.path.exists(base_path):
        raise RuntimeError("Download  {} failed. ".format(model_name))

    register.import_modules_plugin()

    vocab_file = os.path.join(base_path, ernie_resource_dict[model_name]["vocab_file"])
    tokenizer_type = ernie_resource_dict[model_name]["tokenizer"]

    tokenizer_class = RegisterSet.tokenizer.__getitem__(tokenizer_type)
    tokenizer = tokenizer_class(vocab_file=vocab_file)
    return tokenizer


def generate_ernie_model(model_name, root_dir, extra_params=None, enable_static=False):
    """generate ernie model
    :param model_name: 待获取（下载）的预训练模型的名字，取值为ernie_resource_dict中包含的名称
    :param root_dir: 获取之后的预训吗模型及配置文件存放的位置
    :enable_static: 是否使用静态图方式，静态图模式下需要在插件外部进行预训练模型的参数初始化
    """
    if not model_name or not root_dir:
        raise RuntimeError("model_name and root_dir can't be None")

    url = DOWNLOAD_BASE_URL + model_name + ".tgz"
    base_path = get_path_from_url(url=url, root_dir=root_dir)
    logging.info("base path is {}".format(base_path))

    if not os.path.exists(base_path):
        raise RuntimeError("Download  {} failed. ".format(model_name))

    register.import_modules_plugin()

    cfg_path = os.path.join(base_path, ernie_resource_dict[model_name]["config_file"])
    cfg_dict = ErnieConfig(cfg_path)

    net_name = ernie_resource_dict[model_name]["net_name"]
    net_class = RegisterSet.models.__getitem__(net_name)

    if extra_params is not None:
        ernie_model = net_class(cfg_dict, extra_params)
    else:
        ernie_model = net_class(cfg_dict)

    model_params_path = os.path.join(base_path, ernie_resource_dict[model_name]["params_file"])
    if not enable_static:
        ernie_model_dict = paddle.load(model_params_path)
        ernie_model.set_state_dict(ernie_model_dict, use_structured_name=False)

    return model_params_path, ernie_model
