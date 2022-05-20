# -*- coding: utf-8 -*
"""
对内工具包（major）中最常用的trainer，必须继承自文心core中的BaseTrainer基类，必须实现do_train, do_evaluate, do_visual方法。
"""
import collections
import logging
import time
import numpy as np
from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
import paddle.distributed.fleet as fleet
from erniekit.controller.static_trainer import BaseStaticTrainer
from erniekit.controller.static_trainer_ernie_gen import BaseStaticTrainerErnieGen
import paddle.fluid as fluid
import traceback

@RegisterSet.trainer.register
class CustomGenerationTrainer(BaseStaticTrainerErnieGen):
    """CustomTrainer
    """
    def __init__(self, params, data_set_reader, model):
        """
        :param params:前端json中设置的参数
        :param data_set_reader: 数据集实例，包括训练集、评估集、测试集、预测集
        :param model:模型组网实例
        """
        BaseStaticTrainerErnieGen.__init__(self, params, data_set_reader, model)
        self.return_numpy = self.params.get("return_numpy", True)

    def do_train(self):
        """ 启动数据集循环，开始训练
        :return:
        """
        if self.use_fleet and fleet.is_server():
            logging.debug("is fleet.server, over")
            return
        if self.use_fleet:
            logging.debug("worker_index%d start train...." % fleet.worker_index())

        num_train_examples = self.params.get("num_train_examples", 0)
        if num_train_examples == 0:
            num_train_examples = self.data_set_reader.train_reader.get_num_examples()

        steps = 1
        if self.params["is_do_train"]:
            self.data_set_reader.train_reader.run()
            time_begin = time.time()
            try:
                while True:
                    try:
                        if steps % self.params["train_log_step"] != 0:
                            if self.use_fleet:
                                self.train_exe.run(program=self.train_program, fetch_list=[], return_numpy=True)
                            else:
                                self.train_exe.run(fetch_list=[], return_numpy=True)
                        else:
                            if self.use_fleet:
                                fetch_output = self.train_exe.run(program=self.train_program,
                                                      fetch_list=self.fetch_list_train,
                                                      return_numpy=True)
                            else:
                                fetch_output = self.train_exe.run(
                                        fetch_list=self.fetch_list_train,
                                        return_numpy=True)
                                fetch_output_dict = collections.OrderedDict()
                                for key, value in zip(self.fetch_list_train_key, fetch_output):
                                    value = np.array(value)
                                    fetch_output_dict[key] = value

                                time_end = time.time()    
                                used_time = time_end - time_begin
                                meta_info = collections.OrderedDict()
                                meta_info[InstanceName.STEP] = steps
                                meta_info[InstanceName.GPU_ID] = self.gpu_id
                                meta_info[InstanceName.TIME_COST] = used_time
                                meta_info["speed"] = self.params["train_log_step"] / used_time

                                current_example, current_epoch = self.data_set_reader.train_reader.get_train_progress()
                                logging.info("epoch {0} progress {1}/{2} pyreader queue size {3} learning_rate {4}".
                                        format(current_epoch, current_example, num_train_examples,
                                            self.data_set_reader.train_reader.paddle_data_loader.queue.size(),
                                            np.mean(fetch_output_dict["lr"])))
                                
                                metrics_output = self.model_class.get_metrics(fetch_output_dict,
                                        meta_info, InstanceName.TRAINING, reader=None)
                                time_begin = time.time()

                        if self.model_class.lr_scheduler:
                            # 这一步一定要有，没有的话lr_scheduler不会生效，学习率一直为0
                            self.model_class.lr_scheduler.step()
                                
                        if steps % self.params["eval_step"] == 0:
                            if self.params["is_eval_dev"]:
                                self.do_evaluate(self.data_set_reader.dev_reader, InstanceName.EVALUATE, steps)
                            if self.params["is_eval_test"]:
                                self.do_evaluate(self.data_set_reader.test_reader, InstanceName.TEST, steps)
                        if self.trainer_id == 0:
                            if steps % self.params["save_model_step"] == 0:
                                self.save_model(steps)
                        steps += 1
                    except fluid.core.EOFException:
                        self.data_set_reader.train_reader.stop()
                        break
                        
                if self.params["is_eval_dev"]:
                    logging.info("Final evaluate result: ")
                    self.do_evaluate(self.data_set_reader.dev_reader, InstanceName.EVALUATE, steps)
                if self.params["is_eval_test"]:
                    logging.info("Final test result: ")
                    self.do_evaluate(self.data_set_reader.test_reader, InstanceName.TEST, steps)
                        
            except Exception as e:
                logging.error('traceback.format_exc():%s' % traceback.format_exc())

            if self.trainer_id == 0:
                self.save_model(steps)


    def do_evaluate(self, reader, phase, step):
        """在当前的训练状态下，对某个测试集进行评估
        :param reader:待评估数据集
        :param phase:当前的运行阶段
        :param step:当前的运行步数
        """

        if not reader:
            raise ValueError("{0} reader is none".format(phase))

        meta_info = collections.OrderedDict()
        meta_info[InstanceName.GPU_ID] = self.gpu_id
        meta_info["stage"] = step
        self.model_class.init_metrics(meta_info, phase)

        reader.run()
        all_metrics_tensor_value = []
        time_begin = time.time()
        fetch_output_dict = collections.OrderedDict()
        while True:
            try:
                metrics_tensor_value = self.executor.run(program=self.evaluate_program,
                        fetch_list=self.fetch_list_evaluate,
                        return_numpy=not reader.do_dec)
                cur_output_dict = collections.OrderedDict()
                for key, value in zip(self.fetch_list_evaluate_key, metrics_tensor_value):
                    if reader.do_dec:
                        cur_output_dict[key] = value
                    else:
                        fetch_output_dict[key] = fetch_output_dict.get(key, []) + [value]
                self.model_class.add_metric_result(cur_output_dict, phase)
            except fluid.core.EOFException:
                reader.stop()
                break

        time_end = time.time()
        used_time = time_end - time_begin
        meta_info[InstanceName.TIME_COST] = used_time
        self.model_class.get_metrics(fetch_output_dict, meta_info, phase, reader)
        
#         # if not reader:
#         #     raise ValueError("{0} reader is none".format(phase))
#         # reader.run()
#         all_metrics_tensor_value = []
#         i = 0
#         time_begin = time.time()
#         for batch_id, data in enumerate(reader()):
#             feed_dict = reader.dataset.convert_input_list_to_dict(data)
#             metrics_tensor_value = self.executor.run(program=self.test_program,
#                                              feed=feed_dict,
#                                              fetch_list=self.fetch_list_evaluate,
#                                              return_numpy=True)
#             if i == 0:
#                 all_metrics_tensor_value = [[tensor] for tensor in metrics_tensor_value]
#             else:
#                 for j in range(len(metrics_tensor_value)):
#                     one_tensor_value = all_metrics_tensor_value[j]
#                     all_metrics_tensor_value[j] = one_tensor_value + [metrics_tensor_value[j]]
#             i += 1

#         fetch_output_dict = collections.OrderedDict()
#         for key, value in zip(self.fetch_list_evaluate_key, all_metrics_tensor_value):
#             if key == InstanceName.LOSS and not self.return_numpy:
#                 value = [np.array(item) for item in value]
#             fetch_output_dict[key] = value
#         time_end = time.time()
#         used_time = time_end - time_begin

#         meta_info = collections.OrderedDict()
#         meta_info[InstanceName.STEP] = step
#         meta_info[InstanceName.GPU_ID] = self.gpu_id
#         meta_info[InstanceName.TIME_COST] = used_time
#         metrics_output = self.model_class.get_metrics(fetch_output_dict, meta_info, phase)

#         # if self.params.get("visualdl_log", False):
#         #     assert isinstance(metrics_output, OrderedDict), "the metrics_output must be OrderedDict"
#         #     eval_loss = np.mean(fetch_output_dict[InstanceName.LOSS])
#         #     self.visualdl_log(metrics_output, eval_loss, step, phase=phase)
#         #
#         # if self.visual_manager and self.params.get("metrics_visual", False):
#         #     self.visual_manager.show_metric(metrics_output, step, tag=phase)

#         # 因在evaluate的时候不一定会返回loss，因此注释该部分代码
#         # loss = np.mean(fetch_output_dict[InstanceName.LOSS])
#         # return loss
#         # xx = paddle.io.DataLoader()

#         return metrics_output

    def do_visual(self):
        """评估指标的可视化展示
        """
        pass
