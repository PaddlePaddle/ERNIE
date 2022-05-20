# -*- coding: utf-8 -*
"""import"""
import os
import sys
sys.path.append("../../../")
from erniekit.common.register import RegisterSet
from erniekit.common import register
from erniekit.data.data_set import DataSet
import logging
from erniekit.utils import args
from erniekit.utils import params
from erniekit.utils import log
import paddle

logging.getLogger().setLevel(logging.INFO)


def dataset_reader_from_params(params_dict):
    """
    :param params_dict:
    :return:
    """
    dataset_reader = DataSet(params_dict)
    dataset_reader.build()

    return dataset_reader


def model_from_params(params_dict, dataset_reader):
    """
    :param params_dict:
    :param dataset_reader
    :return:
    """
    opt_params = params_dict.get("optimization", None)
    num_train_examples = dataset_reader.train_reader.dataset.get_num_examples()
    # 按配置计算warmup_steps
    if opt_params and opt_params.__contains__("warmup_steps"):
        trainers_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        batch_size_train = dataset_reader.train_reader.dataset.config.batch_size
        epoch_train = dataset_reader.train_reader.dataset.config.epoch
        max_train_steps = epoch_train * num_train_examples // batch_size_train // trainers_num
        # 知识蒸馏TD2需要将TD1的max_train_step算进来
        task_distill_params = params_dict.get("task_distill_step2", None)
        if task_distill_params and task_distill_params.__contains__("td1_epoch"):
            # TD1训练的轮数，需要在TD2的配置文件里设置
            td1_epoch = task_distill_params["td1_epoch"]
            # 默认TD1和TD2的batch_size一致，训练样本数一致
            td1_batch_size = task_distill_params.get("td1_batch_size", batch_size_train)
            max_train_steps += td1_epoch * num_train_examples // td1_batch_size // trainers_num

        warmup_steps = opt_params.get("warmup_steps", 0)

        if warmup_steps == 0:
            warmup_proportion = opt_params.get("warmup_proportion", 0.1)
            warmup_steps = int(max_train_steps * warmup_proportion)

        logging.info("Device count: %d" % trainers_num)
        logging.info("Num train examples: %d" % num_train_examples)
        logging.info("Max train steps: %d" % max_train_steps)
        logging.info("Num warmup steps: %d" % warmup_steps)

        opt_params = {}
        opt_params["warmup_steps"] = warmup_steps
        opt_params["max_train_steps"] = max_train_steps
        opt_params["num_train_examples"] = num_train_examples

        # combine params dict
        params_dict["optimization"].update(opt_params)
    model_name = params_dict.get("type")
    model_class = RegisterSet.models.__getitem__(model_name)
    model = model_class(params_dict)
    return model, num_train_examples


def build_trainer(params_dict, dataset_reader, model, num_train_examples=0):
    """build trainer"""
    trainer_name = params_dict.get("type", "CustomTrainer")
    trainer_class = RegisterSet.trainer.__getitem__(trainer_name)
    params_dict["num_train_examples"] = num_train_examples
    trainer = trainer_class(params=params_dict, data_set_reader=dataset_reader, model=model)
    return trainer


def run_trainer(param_dict):
    """
    :param param_dict:
    :return:
    """
    logging.info("run trainer.... pid = " + str(os.getpid()))
    dataset_reader_params_dict = param_dict.get("dataset_reader")
    dataset_reader = dataset_reader_from_params(dataset_reader_params_dict)

    model_params_dict = param_dict.get("model")
    model, num_train_examples = model_from_params(model_params_dict, dataset_reader)
    model_params_dict["num_train_examples"] = num_train_examples

    trainer_params_dict = param_dict.get("trainer")
    trainer = build_trainer(trainer_params_dict, dataset_reader, model, num_train_examples)

    trainer.do_train()
    logging.info("end of run train and eval .....")


if __name__ == "__main__":
    args = args.build_common_arguments()
    log.init_log("./log/test", level=logging.DEBUG)
    param_dict = params.from_file(args.param_path)
    _params = params.replace_none(param_dict)
    
    # 记得import一下注册的模块
    register.import_modules()
    register.import_new_module("model", "bow_matching_pairwise")
    register.import_new_module("model", "ernie_matching_fc_pointwise")
    register.import_new_module("model", "ernie_matching_siamese_pairwise")
    register.import_new_module("model", "ernie_matching_siamese_pointwise")
    register.import_new_module("trainer", "custom_trainer")
    register.import_new_module("trainer", "custom_dynamic_trainer")
    register.import_new_module("data_set_reader", "ernie_classification_dataset_reader")

    # erniekitDataLoader
    trainer_params = param_dict.get("trainer")
    paddle.set_device(trainer_params.get("PADDLE_PLACE_TYPE", "cpu"))
    run_trainer(_params)
    os._exit(0)