# -*- coding: utf-8 -*
"""BaseDynamicTrainer
动态图的trainer基类
"""
import collections
import json
import logging
import os
import paddle
from paddle.distributed import fleet

from .. import version
from ..common.jit_wenxin import WenxinTracedLayer
from ..common.register import RegisterSet
from ..common.rule import InstanceName
from ..utils.util_helper import get_model_paths, save_meta_data, append_name
import paddle.distributed as dist
from functools import reduce


@RegisterSet.trainer.register
class BaseDynamicTrainer(object):
    """BaseDynamicTrainer
    """

    def __init__(self, params, data_set_reader, model_class):
        """
        :param params:
        :param data_set_reader:
        :param model_class:
        """
        self.framework_version = "1.0.0"
        self.data_set_reader = data_set_reader
        self.params = params
        self.visual_manager = None
        if self.params.get("metrics_visual", False):
            from ..utils.visual_manager import VisualManager
            self.visual_manager = VisualManager(logdir=self.params.get("visual_log_dir", None))
        self.model_class = model_class
        self.multi_devices = False  # 是否是多卡（目前文心的多卡使用fleet来启动多个进程）
        self.use_fleet = False

        self.parser_meta()
        # fleet环境初始化
        if self.params.get("PADDLE_IS_FLEET", 0):
            fleet.init(is_collective=True)
            logging.info("fleet init ...")
            self.use_fleet = True
            self.worker_index = paddle.distributed.get_rank()
        else:
            self.use_fleet = False
            self.worker_index = 0

        self.model_class.structure()

        if self.worker_index == 0:
            logging.info("load_pretrain_model...")
            self.load_pretrain_model()

        # 通过Fleet API获取分布式model，用于支持分布式训练
        if self.use_fleet:
            self.model_class = fleet.distributed_model(self.model_class)
            if dist.get_world_size() > 1:
                self.multi_devices = True
                self.original_model = self.model_class._layers
            else:
                self.original_model = self.model_class
        else:
            self.original_model = self.model_class

        self.use_amp = self.params.get("use_amp", False)

        if 'output_path' in self.params.keys() and self.params["output_path"]:
            self.save_checkpoints_path = os.path.join(self.params["output_path"], "save_checkpoints")
            self.save_inference_model_path = os.path.join(self.params["output_path"], "save_inference_model")
        else:
            self.save_checkpoints_path = "./output/save_checkpoints/"
            self.save_inference_model_path = "./output/save_inference_model/"

        # 文心的调用
        if self.multi_devices:
            self.optimizer = self.model_class._layers.set_optimizer()
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
        else:
            self.optimizer = self.model_class.set_optimizer()

    def do_train(self):
        """
        启动数据集循环，开始训练，子类必须实现具体方法
        """
        raise NotImplementedError

    def do_evaluate(self, reader, phase, step):
        """模型效果评估，子类必须实现具体方法
        :param reader:
        :param phase:
        :param step:
        :return: loss
        """
        raise NotImplementedError

    def parser_meta(self):
        """parser_meta
        """
        logging.info("parser meta ....")
        model_meta_info = {}
        if self.params["load_checkpoint"] or self.params["load_parameters"]:
            model_meta_info = self.load_model_meta_info("net_model")
        elif self.params["pre_train_model"]:
            model_meta_info = self.load_model_meta_info("pre_train_model")
        # 由外部json配置传入
        meta_param = {}
        extra_param = self.params.get("extra_param", None)
        if extra_param:
            meta_param = extra_param.get("meta", None)

        self.meta_dict = {
            "framework_version": version.full_version,
            "model_type": model_meta_info.get("model_type", ""),
            "pretrain_model_version": model_meta_info.get("pretrain_model_version", ""),
            "pretrain_model_type": model_meta_info.get("pretrain_model_type", ""),
            "job_type": meta_param.get("job_type", "custom"),
            "net_type": self.model_class.__class__.__name__,
            "task_type": "train",
            "deploy_type": 4,
            "is_dynamic": 1
        }
        return

    def load_model_meta_info(self, load_model):
        """
        获取模型的meta信息
        :param load_model:
        :return:
        """
        meta_info = {}
        if load_model == "net_model":
            if self.params["load_checkpoint"]:
                original_path = self.params["load_checkpoint"]
                meta_info = load_meta(original_path)
            elif self.params["load_parameters"]:
                original_path = self.params["load_parameters"]
                meta_info = load_meta(original_path)
        elif load_model == "pre_train_model":
            for pre_train_model in self.params["pre_train_model"]:
                logging.info("pre_train_model's name = %s" % pre_train_model["name"])
                # params_path = pre_train_model["params_path"]
                params_path = os.path.dirname(pre_train_model["params_path"])
                meta_info = load_meta(params_path)
        return meta_info

    def save_models(self, step, fields_dict, save_checkpoint=True, save_inference=True):
        """模型保存：checkpoints文件用来热启动，inference文件用来预测推理
        :param step:
        :param fields_dict:
        :param save_checkpoint
        :param save_inference
        :return:
        """
        path_dict = get_model_paths(self.save_checkpoints_path, self.save_inference_model_path, step)
        if save_checkpoint:
            save_path = os.path.join(path_dict["checkpoints_model_path"], "wenxin")
            paddle.save(self.original_model.state_dict(), "{0}.pdparams".format(save_path))
            paddle.save(self.original_model.optimizer.state_dict(), "{0}.pdopt".format(save_path))
            meta_path = path_dict["checkpoints_meta_path"]
            save_meta_data(self.meta_dict, meta_path)

        if save_inference:
            save_path = os.path.join(path_dict["inference_model_path"], "wenxin")
            logging.info("save path: {0}".format(path_dict["inference_model_path"]))
            output, static_wenxin = WenxinTracedLayer.trace(self.original_model,
                                                            inputs=fields_dict,
                                                            phase="save_inference")
            static_wenxin.save_inference_model(save_path)

            meta_path = path_dict["inference_meta_path"]
            save_meta_data(self.meta_dict, meta_path)

            infer_dict = {"fields": output[InstanceName.TARGET_FEED_NAMES]}
            infer_meta_path = path_dict["inference_infer_meta_path"]
            save_meta_data(infer_dict, infer_meta_path)

    def load_pretrain_model(self, add_prefix_name=None, set_model_class=None):
        """加载预训练模型或者热启动模型参数
        """
        sd_param = None
        if "load_checkpoint" in self.params and self.params["load_checkpoint"] and "load_parameters" in self.params and \
                self.params["load_parameters"]:
            raise ValueError(
                "ERROR: config 'load_checkpoint' and 'load_parameters' "
                "both are set! Only one of them should be set. "
                "if you want warmstart checkpoint keep its learning_rate and moments, plese set 'load_checkpoint'. "
                "if you want warmstart checkpoint with only its parameters, and you want reset a new learning_rate "
                "by config, plese set 'load_parameters'")
        if "load_checkpoint" in self.params and self.params["load_checkpoint"]:
            logging.info("load checkpoints path: {0}".format(self.params["load_checkpoint"]))
            load_checkpoint_prefix = self.params["load_checkpoint"]
            sd_param = paddle.load(os.path.join(load_checkpoint_prefix, "wenxin.pdparams"))
            sd_opt = paddle.load(os.path.join(load_checkpoint_prefix, "wenxin.pdopt"))
            sd_param.update(sd_opt)
            if set_model_class is not None:
                set_model_class.set_state_dict(sd_param)
            else:
                self.model_class.set_state_dict(sd_param)
        if "load_parameters" in self.params and self.params["load_parameters"]:
            logging.info("load parameters path: {0}".format(self.params["load_parameters"]))
            load_parameters_prefix = self.params["load_parameters"]
            sd_param = paddle.load(os.path.join(load_parameters_prefix, "wenxin.pdparams"))
            if set_model_class is not None:
                set_model_class.set_state_dict(sd_param)
            else:
                self.model_class.set_state_dict(sd_param)
        elif "pre_train_model" in self.params and self.params["pre_train_model"]:
            for pre_train_model in self.params["pre_train_model"]:
                state_dict_path = pre_train_model["params_path"]
                logging.info("load pre_train_model path: {0}".format(state_dict_path))
                if os.path.exists(state_dict_path):
                    sd_param = paddle.load(state_dict_path)
                    if add_prefix_name is not None:
                        all_keys = list(sd_param.keys())
                        for one_key in all_keys:
                            sd_param[append_name(add_prefix_name, one_key)] = sd_param.pop(one_key)
                    if set_model_class is not None:
                        set_model_class.set_state_dict(sd_param, use_structured_name=False)
                    else:
                        self.model_class.set_state_dict(sd_param, use_structured_name=False)
                    # 用于prompt等场景冻结预训练模型参数
                    if self.params.get("is_freeze", False):
                        print("'using freeze in load model!!!! only prompt will be optimized'")
                        trainable_param_num = 0
                        all_param_num = 0
                        for k, v in self.model_class.state_dict().items():
                            if v.name in sd_param.keys():
                                v.trainable = False
                            else:
                                v.trainable = True
                                print('training:{}, shape:{}'.format(k, v.shape))
                                trainable_param_num += reduce(lambda a, b: a * b, v.shape)
                            all_param_num += reduce(lambda a, b: a * b, v.shape)
                        ratio = trainable_param_num * 100 / all_param_num
                        print('trainable-num:{}, all-num:{}, ratio:{}%'.format(trainable_param_num,
                                                                               all_param_num,
                                                                               ratio))

        return sd_param


def load_meta(model_dir):
    """
    :param model_dir:
    :return: meta_dict
    """
    json_path = None
    meta_dict = {}
    for file in os.listdir(model_dir):
        if file.endswith(".meta"):
            json_path = file
            break
    try:
        if json_path:
            json_file = open(os.path.join(model_dir, json_path), 'r')
            model_info = json_file.read()
            meta_dict = json.loads(model_info)
    except Exception as e:
        logging.error("error in parser model.meta.....")
    return meta_dict
