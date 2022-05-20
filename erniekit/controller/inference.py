# -*- coding: utf-8 -*
"""
模型的预测控制器，核心成员有model，reader；核心方法有：
1.模型加载
2.预测：批量预测与单query预测
3.结果解析与回传
"""
import logging
import os
import paddle.inference as paddle_infer

from paddle.utils.cpp_extension import load

from ..utils import params


class BaseInference(object):
    def __init__(self, params, data_set_reader, parser_handler):
        """
        :param params:前端json中设置的参数
        :param data_set_reader: 预测集reader
        :param parser_handler: 飞桨预测结果通过parser_handler参数回调到具体的任务中，由用户控制具体结果解析
        """
        self.data_set_reader = data_set_reader
        if self.data_set_reader is not None:
            self.data_set_reader.predict_reader.need_generate_examples = True
        self.params = params
        self.parser_handler = parser_handler
        self.input_names = []
        self.input_handles = []
        self.predictor = None
        self.place_type = self.params.get("PADDLE_PLACE_TYPE", os.getenv("PADDLE_PLACE_TYPE", "cpu"))
        self.model_path = self.params["inference_model_path"]
        # eg: "./output/cls_bow_ch/save_inference_model/inference_step_250/"
        self.thread_num = self.params.get("thread_num", 1)
        self.use_memory_optim = self.params.get("use_memory_optim", False)
        self.input_keys = []
        self.parser_input_keys()
        self.init_env()

    def inference_batch(self):
        """
        """
        raise NotImplementedError

    def inference_query(self, query):
        """单条query预测
        :param query
        """
        raise NotImplementedError

    def init_env(self):
        logging.info("init env, build inference....")
        self.predictor = self.load_inference_model(self.model_path, self.thread_num, self.use_memory_optim)

        self.input_names = self.predictor.get_input_names()
        for one_input in self.input_names:
            one_handler = self.predictor.get_input_handle(one_input)
            self.input_handles.append(one_handler)

    def load_inference_model(self, model_path, thread_num=1, use_memory_optim=False):
        """
        :param model_path:
        :param thread_num
        :param use_memory_optim 为了防止有些模型在gpu预测的时候报显存，默认关闭，报显存时打开
        :return: inference
        """
        model_file = os.path.join(model_path, "wenxin.pdmodel")
        params_file = os.path.join(model_path, "wenxin.pdiparams")
        config = paddle_infer.Config(model_file, params_file)
        config.switch_ir_optim(False)

        if self.place_type == "xpu" or self.place_type == "XPU":
            logging.info("xpu inference....")
            # config.enable_lite_engine(PrecisionType.Float32, True)
            # l3_workspace_size - l3 cache 分配的显存大小, 以MB为单位
            # https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_doc/Config/XPUConfig.html
            config.enable_xpu(1024)
            config.set_cpu_math_library_num_threads(thread_num)

        elif self.place_type == "gpu" or self.place_type == "GPU":
            logging.info("gpu inference....")
            # config.enable_lite_engine(PrecisionType.Float32, True)
            # memory_pool_init_size_mb - 初始化分配的gpu显存，以MB为单位
            # paddle.inference.Config.enable_use_gpu(memory_pool_init_size_mb: int, device_id: int)
            # https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_doc/Config/GPUConfig.html
            config.enable_use_gpu(1024)
            config.set_cpu_math_library_num_threads(thread_num)
            if use_memory_optim:
                config.enable_memory_optim()

        else:
            # 默认使用CPU预测
            logging.info("cpu inference....")
            # https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_doc/Config/CPUConfig.html
            config.set_cpu_math_library_num_threads(thread_num)
            config.enable_mkldnn()

        predictor = paddle_infer.create_predictor(config)
        return predictor

    def parser_input_keys(self):
        """从meta文件中解析出模型预测过程中需要feed的变量名称，与model.forward的fields_dict对应起来
        """
        data_params_path = os.path.join(self.model_path, "infer_data_params.json")
        param_dict = params.from_file(data_params_path)
        param_dict = params.replace_none(param_dict)
        self.input_keys = param_dict.get("fields")
