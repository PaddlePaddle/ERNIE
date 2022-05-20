# -*- coding: utf-8 -*
"""
模型的训练控制器，核心成员有：model、reader、evaluate(待定)。核心方法有：
0.运行时环境初始化
1.网络初始化
2.reader初始化
3.模型训练
4.模型评估
5.模型保存:meta信息尽可能完整一些
6.模型指标与模型网络结构可视化
7.模型选择的策略

--------------------------

核心方法的调用顺序为：
1.打印meta及version相关日志，便于问题追查
2.鉴权
3.初始化运行所需要的环境
"""
import json
import logging
import multiprocessing
import os
import shutil
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle
import paddle.static as static
from .. import version
from ..common.rule import InstanceName
from ..utils.util_helper import get_model_paths, save_meta_data, make_targz


class BaseStaticTrainer(object):
    def __init__(self, params, data_set_reader, model):
        """
        :param params
        :param data_set_reader
        :param model
        """
        self.params = params
        self.data_set_reader = data_set_reader
        self.model_class = model

        # 参数解析
        # 动态图or静态图
        self.enable_static = True
        self.is_recompute = self.params.get("is_recompute", 0)
        if 'output_path' in self.params.keys() and self.params["output_path"]:
            self.save_checkpoints_path = os.path.join(self.params["output_path"], "save_checkpoints")
            self.save_inference_model_path = os.path.join(self.params["output_path"], "save_inference_model")
        else:
            self.save_checkpoints_path = "./output/save_checkpoints/"
            self.save_inference_model_path = "./output/save_inference_model/"

        self.forward_train_output = {}
        self.fetch_list_train = []
        self.fetch_list_evaluate = []
        self.fetch_list_train_key = []
        self.fetch_list_evaluate_key = []

        self.parser_meta()
        self.use_fleet = False
        self.init_env_static()

    def do_train(self):
        """
        启动数据集循环，开始训练
        :return:
        """
        raise NotImplementedError

    def do_evaluate(self, reader, phase, step):
        """在当前的训练状态下，对某个测试集进行评估
        :param reader:待评估数据集
        :param phase:当前的运行阶段
        :param step:当前的运行步数
        """
        raise NotImplementedError

    def do_visual(self):
        """评估指标的可视化展示
        """
        raise NotImplementedError

    def parser_meta(self):
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
            "is_dynamic": 0
        }
        return

    def init_env_static(self):
        """
        初始化静态图的运行时环境：包括：program、executor、fleet、cuda、place
        :return:
        """
        logging.info("init environment on static mode......")
        paddle.enable_static()

        # step1: init program
        self.startup_program = static.Program()
        self.train_program = static.Program()
        self.test_program = static.Program()
        self.evaluate_program = static.Program()
        self.save_inference_program = static.Program()

        random_seed = self.params.get("random_seed", 0)
        if random_seed is not None:
            self.startup_program.random_seed = random_seed
            self.train_program.random_seed = random_seed
            self.test_program.random_seed = random_seed
            self.evaluate_program.random_seed = random_seed
            self.save_inference_program.random_seed = random_seed

        # step2: init run place、executor、fleet
        self.num_trainers = 1
        self.trainer_id = 0

        self.place_type = self.params.get("PADDLE_PLACE_TYPE", os.getenv("PADDLE_PLACE_TYPE", "cpu"))
        self.params["PADDLE_PLACE_TYPE"] = self.place_type

        # executor执行器的一些参数设置
        self.use_fast_executor = self.params.get("use_fast_executor", False)
        self.exe_strategy = paddle.static.ExecutionStrategy()
        self.exe_strategy.num_iteration_per_run = self.params.get("num_iteration_per_run", 1)
        self.exe_strategy.num_iteration_per_drop_scope = self.params.get("num_iteration_per_drop_scope", 10)

        self.build_strategy = paddle.static.BuildStrategy()

        if self.place_type == "gpu":
            logging.info("gpu place....")
            gpus = os.getenv('FLAGS_selected_gpus', '0').split(",")
            self.gpu_id = int(gpus[0])
            self.run_place = paddle.CUDAPlace(int(gpus[0]))
            self.dev_count = len(gpus)
            self.exe_strategy.num_threads = self.dev_count
            self.use_cuda = True
            """
            gpu fleet 使用三步骤：
            1.导入依赖包：from paddle.distributed import fleet
            2.初始化fleet环境:包括定义缺省的分布式策略，然后通过将参数is_collective设置为True，使训练架构设定为Collective架构。
            strategy = fleet.DistributedStrategy()
            fleet.init(is_collective=True, strategy=strategy)
            3.使用distributed_optimizer设置分布式训练优化器
            optimizer = fleet.distributed_optimizer(optimizer)
            """
            if self.params.get("PADDLE_IS_FLEET", 0):
                fleet.init(is_collective=True)
                logging.info("fleet init ...")
                self.use_fleet = True
                self.strategy = fleet.DistributedStrategy()
                self.strategy.execution_strategy = self.exe_strategy
                self.strategy.build_strategy = self.build_strategy
                # TODO nccl_comm_num 可以加快GPU之间的通信效率，建议单机设置为1，多机设置为2。
                # TODO 找个判断多机的方法，设置nccl_comm_num参数
                self.strategy.nccl_comm_num = 1
                self.strategy.sync_nccl_allreduce = True
                self.strategy.fuse_all_reduce_ops = True

                # amp设置
                self.use_amp = self.params.get("use_amp", False)
                if self.use_amp:
                    opt_params = self.model_class.model_params.get('optimization', None)
                    init_loss_scaling = opt_params.get("init_loss_scaling", 1.0)
                    incr_every_n_steps = opt_params.get("incr_every_n_steps", 1000)
                    decr_every_n_nan_or_inf = opt_params.get("decr_every_n_nan_or_inf", 2)
                    incr_ratio = opt_params.get("incr_ratio", 2.0)
                    decr_ratio = opt_params.get("decr_ratio", 0.8)

                    self.strategy.amp = True
                    self.strategy.amp_configs = {
                        "init_loss_scaling": init_loss_scaling,
                        "decr_every_n_nan_or_inf": decr_every_n_nan_or_inf,
                        "incr_every_n_steps": incr_every_n_steps,
                        "incr_ratio": incr_ratio,
                        "use_dynamic_loss_scaling": True,
                        "decr_ratio": decr_ratio,
                        "custom_white_list": [],
                        "custom_black_list": [],
                    }

                fleet.init(is_collective=True, strategy=self.strategy)
                # 以下代码是为了打印日志，不影响训练
                trainer_id = fleet.worker_index()
                current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
                worker_endpoints = fleet.worker_endpoints()
                trainers_num = len(worker_endpoints)
                logging.debug("worker_endpoints:{} trainers_num:{} current_endpoint:{} trainer_id:{}".format(
                    worker_endpoints,
                    trainers_num,
                    current_endpoint,
                    trainer_id))
                self.num_trainers = trainers_num
                self.trainer_id = trainer_id
            else:
                self.use_fleet = False
                self.num_trainers = 1
                self.trainer_id = 0

        elif self.place_type == "xpu":
            logging.info("xpu_place, support single device mode only")
            xpus = os.getenv('FLAGS_selected_xpus', '0').split(",")
            # self.run_place = paddle.XPUPlace(int(xpus[0]))
            self.run_place = paddle.set_device("xpu:" + xpus[0])
            self.dev_count = 1
            self.exe_strategy.num_threads = self.dev_count
            self.gpu_id = 0
            self.use_cuda = False
            logging.info("finish prepare xpu single deviece env")
            self.use_fleet = False
            self.num_trainers = 1
            self.trainer_id = 0
        else:
            logging.info("cpu place....")
            self.run_place = paddle.CPUPlace()
            self.dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
            self.use_cuda = False
            self.gpu_id = 0
            self.exe_strategy.num_threads = self.dev_count
            """
            cpu fleet 使用步骤
            https://fleetx.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/ps_quick_start.html
            1.导入依赖:
            import paddle.distributed.fleet as fleet
            import paddle.distributed.fleet.base.role_maker as role_maker
            
            2.定义分布式模式并初始化分布式训练环境，当前参数服务器模式只支持静态图模式
            通过fleet.init()接口，用户可以定义训练相关的环境，注意此环境是用户预先在环境变量中配置好的，
            包括：训练节点个数，服务节点个数，当前节点的序号，服务节点完整的IP:PORT列表等。
            paddle.enable_static()
            role = role_maker.PaddleCloudRoleMaker()
            fleet.init(role)
            
            3.组网，加载reader
            model = init_net()
            reader = init_dataset_reader()
            
            4.定义同步训练 Strategy 及 Optimizer
            optimizer = paddle.optimizer.SGD(learning_rate=0.0001)
            strategy = fleet.DistributedStrategy()
            strategy.a_sync = True
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
            optimizer.minimize(model.cost)
            
            5.训练
        
            """
            if self.params.get("PADDLE_IS_FLEET", 0):
                logging.info("int fleet parameter server mode in multi cpus....")
                role = role_maker.PaddleCloudRoleMaker(is_collective=False)
                fleet.init(role)
                self.use_fleet = True
            else:
                self.use_fleet = False
                self.num_trainers = 1
                self.trainer_id = 0

        # step3: init executor with run place
        self.executor = static.Executor(self.run_place)

        # step4: init model net
        self.init_static_model_net()

        # step5: run executor
        self.executor.run(self.startup_program)

        # step6: load model params: checkpoints or pre_train_model
        if self.params["load_checkpoint"] or self.params["load_parameters"]:
            self.load_static_model_params("net_model")
        elif self.params["pre_train_model"]:
            self.load_static_model_params("pre_train_model")

        # step7: init train_executor
        if self.use_fleet:
            self.train_exe = self.executor
        else:
            if self.place_type == "xpu":
                self.train_exe = self.executor
            else:
                # 单机模式下可以使用ParallelExecutor来提速
                self.train_exe = static.ParallelExecutor(
                    use_cuda=self.use_cuda,
                    loss_name=self.forward_train_output[InstanceName.LOSS].name,
                    exec_strategy=self.exe_strategy,
                    build_strategy=self.build_strategy,
                    main_program=self.train_program,
                    num_trainers=self.num_trainers,
                    trainer_id=self.trainer_id)

    def init_static_model_net(self):
        """init static model net
        """
        logging.info("init_model_net.....")
        self.init_static_train_net()
        if self.params["is_eval_dev"]:
            self.evaluate_program = self.init_static_evaluate_net(self.data_set_reader.dev_reader,
                                                                  self.evaluate_program)
        if self.params["is_eval_test"]:
            self.test_program = self.init_static_evaluate_net(self.data_set_reader.test_reader, self.test_program)
        self.init_static_save_inference_net()

    def init_static_train_net(self):
        """
        训练网络初始化，前向+后向
        :return:
        """
        with static.program_guard(self.train_program, self.startup_program):
            with paddle.fluid.unique_name.guard():
                self.data_set_reader.train_reader.dataset.create_reader()
                fields_dict = self.data_set_reader.train_reader.dataset.instance_fields_dict()
                self.model_class.structure()
                if getattr(self.model_class, 'param_attrs', None):
                    self.model_class.set_param_attrs(self.train_program)
                self.forward_train_output = self.model_class.forward(fields_dict, phase=InstanceName.TRAINING)
                loss = self.forward_train_output[InstanceName.LOSS]
                self.model_class.set_optimizer()
                
                # 加入recompute功能
                if self.is_recompute:
                    self.strategy.recompute = True
                    self.strategy.recompute_configs = {"checkpoints": self.forward_train_output['checkpoints']}
                    del self.forward_train_output["checkpoints"]
                
                if self.use_fleet:
                    self.optimizer = fleet.distributed_optimizer(self.model_class.optimizer, strategy=self.strategy)
                else:
                    self.optimizer = self.model_class.optimizer

                self.optimizer.minimize(loss)

                if self.forward_train_output.__contains__(InstanceName.TARGET_FEED):
                    self.forward_train_output.pop(InstanceName.TARGET_FEED)
                if self.forward_train_output.__contains__(InstanceName.TARGET_PREDICTS):
                    self.forward_train_output.pop(InstanceName.TARGET_PREDICTS)
                # TODO:这里需要注意一下，或许有坑
                # self.forward_train_output.update(self.optimizer_output_dict)
                # 如果想获取学习率，加上下面这一行就能fetch出来
                self.forward_train_output.update({"lr": "learning_rate_0"})
                self.fetch_list_train = list(self.forward_train_output.values())
                self.fetch_list_train_key = list(self.forward_train_output.keys())

    def init_static_evaluate_net(self, reader, program):
        """初始化评估过程的网络，网络只有前向
        :return:
        """
        with static.program_guard(program, self.startup_program):
            with paddle.fluid.unique_name.guard():
                reader.dataset.create_reader()
                fields_dict = reader.dataset.instance_fields_dict()
                self.model_class.structure()
                self.forward_evaluate_output = self.model_class.forward(fields_dict, phase=InstanceName.EVALUATE)
                if "mems" in self.forward_evaluate_output.keys():
                    self.mems_eval = self.forward_evaluate_output["mems"]
                    del self.forward_evaluate_output["mems"]

                if self.forward_evaluate_output.__contains__(InstanceName.TARGET_FEED):
                    self.forward_evaluate_output.pop(InstanceName.TARGET_FEED)

                if self.forward_evaluate_output.__contains__(InstanceName.TARGET_PREDICTS):
                    self.forward_evaluate_output.pop(InstanceName.TARGET_PREDICTS)

                self.fetch_list_evaluate = list(self.forward_evaluate_output.values())
                self.fetch_list_evaluate_key = list(self.forward_evaluate_output.keys())

        program = program.clone(for_test=True)
        return program

    def init_static_save_inference_net(self):
        """初始化用来保存inference model的网络，只有前向，且是裁切过后的网络。
        :return:
        """
        with static.program_guard(self.save_inference_program, self.startup_program):
            with paddle.fluid.unique_name.guard():
                self.data_set_reader.predict_reader.dataset.create_reader()
                fields_dict = self.data_set_reader.predict_reader.dataset.instance_fields_dict()
                self.model_class.structure()
                forward_output_dict = self.model_class.forward(fields_dict, phase=InstanceName.SAVE_INFERENCE)
                feed_tensor = forward_output_dict[InstanceName.TARGET_FEED]
                target_feed_list = []
                for x in feed_tensor:
                    target_feed_list.append(x.name)

                self.infer_dict = get_infer_data_meta(target_feed_list, fields_dict)
                self.feed_target_tensor = feed_tensor
                self.inference_output = forward_output_dict[InstanceName.TARGET_PREDICTS]

        self.save_inference_program = self.save_inference_program.clone(for_test=True)

    def load_static_model_params(self, params_type):
        """
        """
        logging.info("load_model_params on static mode....")
        if params_type == "net_model":
            if self.params["load_checkpoint"] and self.params["load_parameters"]:
                raise ValueError(
                    "ERROR: config 'load_checkpoint' and 'load_parameters' "
                    "both are set! Only one of them should be set. "
                    "if you want warmstart checkpoint keep its learning_rate and moments, plese set 'load_checkpoint'. "
                    "if you want warmstart checkpoint with only its parameters, and you want reset a new learning_rate "
                    "by config, plese set 'load_parameters'")
            if self.params["load_checkpoint"]:
                original_path = self.params["load_checkpoint"]
                init_checkpoint(exe=self.executor, init_checkpoint_path=original_path, main_program=self.train_program)
            elif self.params["load_parameters"]:
                original_path = self.params["load_parameters"]
                init_pretraining_params(exe=self.executor,
                                        pretraining_params_path=original_path, main_program=self.train_program)

        elif params_type == "pre_train_model":
            # pretrain_embedding_path = self.get_pretrain_embedding_path()
            for pre_train_model in self.params["pre_train_model"]:
                logging.info("pre_train_model's name = %s" % pre_train_model["name"])
                params_path = pre_train_model["params_path"]
                init_pretraining_params(exe=self.executor,
                                        pretraining_params_path=params_path,
                                        main_program=self.train_program)
        # self.save_model(0)
        # exit()

    def save_model(self, steps, save_checkpoint=True, save_inference=True):
        if self.enable_static:
            logging.info("save model on static....")
            if save_checkpoint:
                self.save_checkpoint(self.executor, self.train_program, steps)
            if save_inference:
                self.save_inference(self.executor, self.feed_target_tensor, self.inference_output,
                                    self.save_inference_program, steps, self.infer_dict)
        else:
            logging.info("save model on dynamic....")

    def save_checkpoint(self, exe, program, steps):
        """
        :param exe:
        :param program:
        :param steps:
        :return:
        """
        path_dict = get_model_paths(self.save_checkpoints_path, self.save_inference_model_path, steps)
        save_path = path_dict["checkpoints_model_path"]
        # todo: 需要验证一下fleet的save和非fleet有没有区别
        paddle.fluid.io.save_persistables(exe, save_path, program)
        meta_path = path_dict["checkpoints_meta_path"]
        save_meta_data(self.meta_dict, meta_path)
        if self.params.get("need_tar", False):
            # 压缩为tar.gz
            errcode = make_targz(save_path + ".tar.gz", save_path)
            if errcode == 0:
                shutil.rmtree(save_path)

    def save_inference(self, exe, feed_vars, target_vars, program, steps, data_dict):
        """
        :param exe:
        :param feed_vars
        :param target_vars
        :param program:
        :param steps:
        :param data_dict:
        :return:
        """
        path_dict = get_model_paths(self.save_checkpoints_path, self.save_inference_model_path, steps)
        save_path = os.path.join(path_dict["inference_model_path"], "wenxin")
        # paddle.fluid.io.save_inference_model
        # paddle.static.save_inference_model
        paddle.static.save_inference_model(
            save_path,
            feed_vars,
            target_vars,
            exe,
            program=program,
            model_filename="model",
            params_filename="params")

        infer_meta_path = path_dict["inference_infer_meta_path"]
        meta_path = path_dict["inference_meta_path"]
        save_meta_data(data_dict, infer_meta_path)
        save_meta_data(self.meta_dict, meta_path)

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
                meta_info = parse_meta(original_path)
            elif self.params["load_parameters"]:
                original_path = self.params["load_parameters"]
                meta_info = parse_meta(original_path)
        elif load_model == "pre_train_model":
            for pre_train_model in self.params["pre_train_model"]:
                logging.info("pre_train_model's name = %s" % pre_train_model["name"])
                params_path = os.path.dirname(pre_train_model["params_path"])
                # original_path = params_path = os.path.dirname(pre_train_model["params_path"])
                meta_info = parse_meta(params_path)
        return meta_info


def get_infer_data_meta(target_feed_list, fields_dict):
    """
    :param target_feed_list:
    :param fields_dict:
    :return:
    """
    infer_dict = {"fields": []}
    for name in target_feed_list:
        for k1, v1 in fields_dict.items():  # dict_keys(['text_a', 'label'])
            for k2, v2 in v1.items():
                if v2:
                    for k3 in v2:
                        # logging.info(k3)
                        if v2[k3] and v2[k3].name == name:
                            field_ele = "%s#%s" % (k1, k3)
                            infer_dict["fields"].append(field_ele)
    return infer_dict


def parse_meta(model_dir):
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


def init_checkpoint(exe, init_checkpoint_path, main_program):
    """加载checkpoints文件
    :param exe:
    :param init_checkpoint_path:
    :param main_program:
    :return:
    """
    assert os.path.exists(init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path

    def existed_persitables(var):
        """
        existed_presitables
        """
        if not paddle.fluid.io.is_persistable(var):
            return False
        return os.path.exists(os.path.join(init_checkpoint_path, var.name))

    paddle.fluid.io.load_vars(exe, init_checkpoint_path, main_program=main_program, predicate=existed_persitables)
    logging.info("Load model from {}".format(init_checkpoint_path))


def init_pretraining_params(exe, pretraining_params_path, main_program):
    """
    :param exe:
    :param pretraining_params_path:
    :param main_program:
    :return:
    """
    assert os.path.exists(pretraining_params_path), "[%s] cann't be found." % pretraining_params_path

    def existed_params(var):
        """
        existed_params
        """
        if not isinstance(var, paddle.fluid.framework.Parameter):
            return False
        return os.path.exists(os.path.join(pretraining_params_path, var.name))

    paddle.fluid.io.load_vars(exe, pretraining_params_path, main_program=main_program, predicate=existed_params)


