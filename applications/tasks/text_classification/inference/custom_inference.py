# -*- coding: utf-8 -*
"""
对内工具包（major）中最常用的inference，必须继承自文心core中的BaseInference基类，必须实现inference_batch, inference_query方法。
"""
import logging
import os
import time
import numpy as np
from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
from erniekit.controller.inference import BaseInference


@RegisterSet.inference.register
class CustomInference(BaseInference):
    """CustomInference
    """
    def __init__(self, params, data_set_reader, parser_handler):
        """
        :param params:前端json中设置的参数
        :param data_set_reader: 预测集reader
        :param parser_handler: 飞桨预测结果通过parser_handler参数回调到具体的任务中，由用户控制具体结果解析
        """
        BaseInference.__init__(self, params, data_set_reader, parser_handler)

    def inference_batch(self):
        """
        批量预测
        """
        logging.info("start do inference....")
        total_time = 0
        output_path = self.params.get("output_path", None)
        if not output_path or output_path == "":
            if not os.path.exists("./output"):
                os.makedirs("./output")
            output_path = "./output/predict_result.txt"

        output_file = open(output_path, "w+")

        dg = self.data_set_reader.predict_reader

        for batch_id, data_t in enumerate(dg()):
            data = data_t[0]
            samples = data_t[1]
            feed_dict = dg.dataset.convert_fields_to_dict(data)
            predict_results = []

            for index, item in enumerate(self.input_keys):
                kv = item.split("#")
                name = kv[0]
                key = kv[1]
                item_instance = feed_dict[name]
                input_item = item_instance[InstanceName.RECORD_ID][key]
                # input_item是tensor类型，需要改为numpy数组
                self.input_handles[index].copy_from_cpu(input_item.numpy())
            
            begin_time = time.time()
            self.predictor.run()
            end_time = time.time()
            total_time += end_time - begin_time

            output_names = self.predictor.get_output_names()
            for i in range(len(output_names)):
                output_tensor = self.predictor.get_output_handle(output_names[i])
                predict_results.append(output_tensor)

            # 回调给解析函数
            write_result_list = self.parser_handler(predict_results, sample_list=samples, params_dict=self.params)
            for write_item in write_result_list:
                size = len(write_item)
                for index, item in enumerate(write_item):
                    output_file.write(str(item))
                    if index != size - 1:
                        output_file.write("\t")

                output_file.write("\n")

        logging.info("total_time:{}".format(total_time))
        output_file.close()

    def inference_query(self, query):
        """单条query预测
        :param query : list
        """
        total_time = 0
        reader = self.data_set_reader.predict_reader.dataset

        data, sample = reader.api_generator(query)
        feed_dict = reader.convert_fields_to_dict(data)

        predict_results = []
        for index, item in enumerate(self.input_keys):
            kv = item.split("#")
            name = kv[0]
            key = kv[1]
            item_instance = feed_dict[name]
            input_item = item_instance[InstanceName.RECORD_ID][key]
            # input_item 是ndarray
            self.input_handles[index].copy_from_cpu(np.array(input_item))

        begin_time = time.time()
        self.predictor.run()
        end_time = time.time()
        total_time += end_time - begin_time

        output_names = self.predictor.get_output_names()
        for i in range(len(output_names)):
            output_tensor = self.predictor.get_output_handle(output_names[i])
            predict_results.append(output_tensor)

        # 回调给解析函数
        result_list = self.parser_handler(predict_results, sample_list=sample, params_dict=self.params)

        return result_list

