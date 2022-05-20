# -*- coding: utf-8 -*
"""DyGraphTrainer
"""
import collections
import logging
import time
import paddle
from paddle.distributed import fleet
from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
from erniekit.controller.dynamic_trainer import BaseDynamicTrainer


@RegisterSet.trainer.register
class CustomDynamicTrainer(BaseDynamicTrainer):
    """CustomDynamicTrainer
    """
    def __init__(self, params, data_set_reader, model):
        """
        :param params:
        :param data_set_reader:
        :param model_class:
        """
        BaseDynamicTrainer.__init__(self, params, data_set_reader, model)

    def do_train(self):
        """
        :return:
        """
        dg = self.data_set_reader.train_reader

        steps = 1
        opt_params = self.original_model.model_params.get('optimization', None)
        init_loss_scaling = opt_params.get("init_loss_scaling", 1.0)
        incr_every_n_steps = opt_params.get("incr_every_n_steps", 1000)
        decr_every_n_nan_or_inf = opt_params.get("decr_every_n_nan_or_inf", 2)
        incr_ratio = opt_params.get("incr_ratio", 2.0)
        decr_ratio = opt_params.get("decr_ratio", 0.8)

        if self.use_amp:
            self.scaler = paddle.amp.GradScaler(enable=self.use_amp,
                                                init_loss_scaling=init_loss_scaling,
                                                incr_ratio=incr_ratio,
                                                decr_ratio=decr_ratio,
                                                incr_every_n_steps=incr_every_n_steps,
                                                decr_every_n_nan_or_inf=decr_every_n_nan_or_inf)
            if self.multi_devices:
                self.scaler = fleet.distributed_scaler(self.scaler)

        time_begin = time.time()
        for batch_id, data in enumerate(dg()):
            self.model_class.train()
            with paddle.amp.auto_cast(enable=self.use_amp):
                example = self.data_set_reader.train_reader.dataset.convert_fields_to_dict(data, need_emb=False)
                forward_out = self.model_class(example, phase=InstanceName.TRAINING)
                loss = forward_out[InstanceName.LOSS]

            if self.use_amp:
                loss = self.scaler.scale(loss)
                loss.backward()
                self.scaler.minimize(self.optimizer, loss)
            else:
                loss.backward()
                self.optimizer.minimize(loss)
                self.optimizer.step()

            self.model_class.clear_gradients()

            if self.original_model.lr_scheduler:
                cur_lr = self.original_model.lr_scheduler.get_lr()
                self.original_model.lr_scheduler.step()
            else:
                cur_lr = self.original_model.lr

            self.optimizer.clear_grad()

            if steps % self.params["train_log_step"] == 0:
                time_end = time.time()
                used_time = time_end - time_begin
                meta_info = collections.OrderedDict()
                meta_info[InstanceName.STEP] = steps
                meta_info[InstanceName.GPU_ID] = 0
                meta_info[InstanceName.TIME_COST] = used_time
                metrics_output = self.original_model.get_metrics(forward_out, meta_info, InstanceName.TRAINING)
                logging.info("current learning rate: {0}".format(round(cur_lr, 7)))
                time_begin = time.time()

            if steps % self.params["eval_step"] == 0:
                if self.params["is_eval_dev"]:
                    self.do_evaluate(self.data_set_reader.dev_reader, InstanceName.EVALUATE, steps)
                if self.params["is_eval_test"]:
                    self.do_evaluate(self.data_set_reader.test_reader, InstanceName.TEST, steps)

            if steps % self.params["save_model_step"] == 0 and self.worker_index == 0:
                self.save_models(steps, example)

            steps += 1

        if self.params["is_eval_dev"]:
            logging.info("Final evaluate result: ")
            self.do_evaluate(self.data_set_reader.dev_reader, InstanceName.EVALUATE, steps)
        if self.params["is_eval_test"]:
            logging.info("Final test result: ")
            self.do_evaluate(self.data_set_reader.test_reader, InstanceName.TEST, steps)

        if self.worker_index == 0:
            self.save_models(steps, example)

    def do_evaluate(self, reader, phase, step):
        """
        :param reader:
        :param phase:
        :param step:
        :return: loss
        """
        step = 0
        with paddle.no_grad():
            time_begin = time.time()
            # 先切换到eval模式
            self.model_class.eval()

            fetch_output_dict = collections.OrderedDict()
            for batch_id, data in enumerate(reader()):
                step += 1
                example = reader.dataset.convert_fields_to_dict(data, need_emb=False)
                forward_out = self.model_class(example, phase=phase)
                for key, value in forward_out.items():
                    fetch_output_dict.setdefault(key, []).append(value)

            time_end = time.time()
            used_time = time_end - time_begin
            meta_info = collections.OrderedDict()
            meta_info[InstanceName.STEP] = step
            meta_info[InstanceName.TIME_COST] = used_time
            metrics_output = self.original_model.get_metrics(fetch_output_dict, meta_info, phase)
            self.model_class.train()
        logging.info("eval step = {0}".format(step))
