# -*- coding: utf-8 -*
"""import
通用任务的启动入口，在调用原来的python run_trainer.py 之前，加了一些预处理工作。目前主要是网格搜索和交叉验证的数据处理和json重置
其余的需要预处理功能都可以加在这里。
"""

import json
import os
import shutil
import subprocess
import sys

sys.path.append("../../../")
from applications.tools.pretreatment import build_grid_search_config, process_data_with_kfold, build_kfold_config
import logging
from erniekit.utils import args
from erniekit.utils import params
from erniekit.utils import log

logging.getLogger().setLevel(logging.INFO)


def process_trainer(train_json_file, trainer_id, run_script):
    """
    :param train_json_file:
    :param trainer_id:
    :param run_script:
    :return:
    """
    env = os.environ
    logging.info('process_trainer: ' + train_json_file)
    fn = open("./log/trainer.log.%d" % (trainer_id), "a")
    run_py_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), run_script)
    cmd = sys.executable + " " + run_py_path + ' --param_path=' + train_json_file
    current_process = subprocess.Popen(cmd, close_fds=True, preexec_fn=os.setsid, shell=True, env=env,
                                       stdout=fn, stderr=fn)
    current_process.wait()
    logging.info("process trainer errcode is " + str(current_process.returncode))


if __name__ == "__main__":
    args = args.build_common_arguments()
    log.init_log("./log/test", level=logging.DEBUG)
    param_dict = params.from_file(args.param_path)
    run_script = args.run_script
    _params = params.replace_none(param_dict)
    tasks_param_list = [_params]

    # step1: k-fold
    dataset_reader_params_dict = _params.get("dataset_reader")
    if dataset_reader_params_dict.get("k_fold", None):
        k_fold_param = dataset_reader_params_dict.get("k_fold")
        num_fold = k_fold_param.get("num_fold", 2)
        data_path = k_fold_param.get("data_path")
        data_path_split = k_fold_param.get("data_path_split")
        num_use_split = k_fold_param.get("num_use_split")

        train_path, dev_path = process_data_with_kfold(data_path, data_path_split, num_split=num_fold,
                                                       num_use_split=num_use_split)
        tasks_param_list = build_kfold_config(_params, train_path, dev_path)

    # step2: grid search
    tasks = []
    for task_param in tasks_param_list:
        tasks.extend(build_grid_search_config(task_param))

    index = 0
    tmp_dir = "./json_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    logging.info("len(tasks_list) = " + str(len(tasks)))

    # step3: run model (python run_with_json)
    for index, task_param in enumerate(tasks):
        task_param["trainer"]["output_path"] = task_param["trainer"]["output_path"] + "_" + str(index)
        json_str = json.dumps(task_param, indent=4)
        save_file = os.path.join(tmp_dir, "tmp_" + str(index) + ".json")
        with open(save_file, 'w') as json_file:
            json_file.write(json_str)
        process_trainer(save_file, index, run_script)
        index += 1

    logging.info("end of run all train and eval .....")
    logging.info("os exit.")
    os._exit(0)
