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
import json

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

        dg = self.data_set_reader.predict_reader
        # sample_entity_list = None
        result_list = []
        for batch_id, data in enumerate(dg()):
            data_list = data[0]
            sample_entity_list = data[1]
            feed_dict = dg.dataset.convert_fields_to_dict(data_list)
            predict_results = []
            #get seq_len、beg_ids、end_ids
            instance_text = feed_dict["text_a"]
            record_id_text = instance_text[InstanceName.RECORD_ID]
            text_a_seq_lens = record_id_text[InstanceName.SEQ_LENS] 
            text_a_beg_ids = record_id_text[InstanceName.BEG_IDS]
            text_a_end_ids = record_id_text[InstanceName.END_IDS]            


            for index, item in enumerate(self.input_keys):
                if item == "text_a#seq_lens":
                    continue
                kv = item.split("#")
                name = kv[0]
                key = kv[1]
                item_instance = feed_dict[name]
                input_item = item_instance[InstanceName.RECORD_ID][key]
                # input_item是tensor类型
                self.input_handles[index].copy_from_cpu(np.array(input_item))
            
            begin_time = time.time()
            self.predictor.run()
            end_time = time.time()
            total_time += end_time - begin_time
            output_names = self.predictor.get_output_names()
            for i in range(len(output_names)):
                output_tensor = self.predictor.get_output_handle(output_names[i])
                predict_results.append(output_tensor)
            predict_results.append(text_a_seq_lens)
            predict_results.append(text_a_beg_ids)
            predict_results.append(text_a_end_ids)
            # 回调给解析函数
            result_batch_list = self.parser_handler(predict_results,
                                                    sample_list=sample_entity_list, params_dict=self.params)
            # result_list.extend(result_batch_list)
            for sample in result_batch_list:
                result_list.append(sample)
        with open(output_path, 'w') as f: 
            f.write(json.dumps(result_list, indent=1, ensure_ascii=False))

        logging.info("total_time:{}".format(total_time))
        return result_list

    def inference_query(self, query):
        """单条query预测
        :param query
        """
        total_time = 0
        reader = self.data_set_reader.predict_reader.dataset

        data, sample_entity_list = reader.api_generator(query)
        

        feed_dict = reader.convert_fields_to_dict(data)
        predict_results = []

        #get seq_len、beg_ids、end_ids
        instance_text = feed_dict["text_a"]
        record_id_text = instance_text[InstanceName.RECORD_ID]
        text_a_seq_lens = record_id_text[InstanceName.SEQ_LENS] 
        text_a_beg_ids = record_id_text[InstanceName.BEG_IDS]
        text_a_end_ids = record_id_text[InstanceName.END_IDS]            


        for index, item in enumerate(self.input_keys):
            if item == "text_a#seq_lens":
                continue
            kv = item.split("#")
            name = kv[0]
            key = kv[1]
            item_instance = feed_dict[name]
            input_item = item_instance[InstanceName.RECORD_ID][key]
            # input_item是tensor类型
            self.input_handles[index].copy_from_cpu(np.array(input_item))
        
        begin_time = time.time()
        self.predictor.run()
        end_time = time.time()
        total_time += end_time - begin_time
        output_names = self.predictor.get_output_names()
        for i in range(len(output_names)):
            output_tensor = self.predictor.get_output_handle(output_names[i])
            predict_results.append(output_tensor)
        predict_results.append(text_a_seq_lens)
        predict_results.append(text_a_beg_ids)
        predict_results.append(text_a_end_ids)
        # 回调给解析函数
        result_list = self.parser_handler(predict_results,
                                                sample_list=sample_entity_list, params_dict=self.params)
        return result_list


