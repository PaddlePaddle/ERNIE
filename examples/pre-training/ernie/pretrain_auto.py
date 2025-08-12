# !/usr/bin/env python3

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
pretrain自动并行版本
"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from src.utils import logger

try:
    from paddleformers.trainer.trainer_utils import log_trainer_start
except ImportError:

    def log_trainer_start():
        """print main process messgae"""
        if "MAIN_PROCESS_STARTED" not in os.environ:
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger.info(
                f"The Training Main Process Started Successfully. time: {start_time}, pid: {os.getpid()}"
            )
            os.environ["MAIN_PROCESS_STARTED"] = "1"


log_trainer_start()

import json
import numpy as np

from functools import partial
import random

import paddle
import paddle.distributed.fleet as fleet

try:
    from paddle.distributed.fleet import monitor_perf as collective_perf
except ImportError:
    from paddle.distributed.fleet import collective_perf

from paddleformers.data import default_data_collator
from paddleformers.datasets import MapDataset
from paddleformers.trainer import (
    PdArgumentParser,
    get_last_checkpoint,
)

from src.tokenizers.tokenization_eb_v2 import ErnieBotTokenizer
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from src.callbacks import (
    ProgreesiveBatchingCallback,
    DataTraceCallbackAuto,
    GlobalRNGCallback,
    PPNeedDataCallback,
)
from models.ernie import (
    ErnieConfig,
    ErnieForCausalLMAutoPP,
)
from models.ernie import ErnieConfig as InceptionModelConfig
from models.ernie_moe.configuration import (
    ErnieMoEConfig,
)
from src.datasets import PretrainTask
from src.datasets.pretrain_task import parse_data_weight
from src.datasets.image_preprocessor import CLIPImageProcessor
from src.trainers import AutoPretrainingTrainer, AutoPreTrainingArguments
from src.utils import (
    load_model,
    setup_logger_output_file,
)
from src.utils.data_utils import fancy_print, merge_fn, merge_fn_group_batch
from src.utils.mm_data_utils import MMSpecialTokensConfig
from src.utils.misc import global_training_logs


# from pretrain import create_pretrained_dataset

from config import get_config

assert paddle.version.mkl() == "OFF", (
    "MKL is not supported"
    " in this version. Please set -DWITH_MKL=OFF when compiling PaddlePaddle."
)


def update_model_config_from_args(config: ErnieConfig, model_args: dict):
    """update model config from args

    Args:
        config (ErnieConfig): _description_
        model_args (dict): _description_

    Returns:
        _type_: _description_
    """
    for k, v in model_args.items():
        if hasattr(config, k):
            logger.info(f"update model config: {k} = {v}")
            setattr(config, k, v)
    return config


def init_parameter(model):
    """
    初始化模型的参数。
    该函数会遍历给定的模型，对每个参数进行初始化操作。

    Args:
        model (torch.nn.Module): Torch神经网络模型，需要初始化其参数。

    Returns:
        None, 不返回任何值。
    """
    for param in model.parameters():
        param.initialize()


def main():
    """main function"""
    config = get_config(verbose=True)
    os.makedirs(config.model_args.output_dir, exist_ok=True)
    parser = PdArgumentParser(AutoPreTrainingArguments)
    if not hasattr(config.trainer_args, "pipeline_parallel_config"):
        config.trainer_args.pipeline_parallel_config = ""

    if "enable_dp_comm_overlap" in config.trainer_args.pipeline_parallel_config:
        logger.warning(
            "Pipeline dp_comm_overlap and FusedLinearWithGradAdd can not be used at "
            "the same time."
        )

    if "enable_timer" in config.trainer_args.pipeline_parallel_config:
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallel,
        )

        # Hack， 原 `PipelineParallel.timer_printer` 方法会在 `forward_backward_pipeline` 方法执行完之后清空timmer状态。
        # 避免这种行为的发生。
        PipelineParallel.timer_printer = lambda _: None

    def formatv(v):
        if isinstance(v, ListConfig):
            return list(v)
        elif isinstance(v, DictConfig):
            return dict(v)
        return v

    model_args = {k: formatv(v) for k, v in dict(config.model_args).items()}
    trainer_args = {k: formatv(v) for k, v in dict(config.trainer_args).items()}
    (args,) = parser.parse_dict(dict(**model_args, **trainer_args))

    if args.strategy.pipeline.enable and args.virtual_pp_degree > 1:
        pipeline = args.strategy.pipeline
        pipeline.vpp_degree = args.virtual_pp_degree
        pipeline.vpp_seg_method = args.virtual_pipeline_seg_method

    if args.modality_ratio is not None:
        args.modality_interleave = (
            sum(args.modality_ratio)
            if args.modality_interleave == "acc"
            else sum(args.modality_ratio) * args.gradient_accumulation_steps
        )
        args.modality_ratio = [
            i / sum(args.modality_ratio) for i in args.modality_ratio
        ]

    # combine_batch = args.combine_batch // config.trainer_args.data_parallel_degree
    # data_processor_args = {k: formatv(v) for k, v in dict(getattr(config, "data_processor_args", {})).items()}
    # (args,) = parser.parse_dict(dict(**model_args, **trainer_args, **data_processor_args))
    args.use_moe = dict(**dict(config.model_args), **dict(config.trainer_args)).get(
        "use_moe", False
    )
    model_config = dict(getattr(config.model_args, "model_config", {}))
    model_config = {k: formatv(v) for k, v in model_config.items()}
    logger.info(f"model_config_from_yaml: {json.dumps(model_config, indent=4)}")
    setup_logger_output_file(config.model_args.output_dir, args.local_rank)
    paddle.set_device(args.device)

    np.random.seed(args.seed)
    random.seed(args.seed)
    paddle.seed(args.seed)
    # set_seed(args.seed)

    prop = paddle.device.cuda.get_device_properties()
    if prop.total_memory < args.pre_alloc_memory * 1024 * 1024 * 1024:
        logger.warning(
            "Invalid value for `pre_alloc_memory`, so pre-allocating just failed."
        )
    elif args.pre_alloc_memory > 0:
        logger.warning(
            f"pre-allocating a tensor whose memory capacity is {args.pre_alloc_memory} GB "
            "and then release it."
        )
        memory_size = int(args.pre_alloc_memory * 1024 * 1024 * 1024)
        x = paddle.empty([memory_size], dtype=paddle.uint8)
        del x

    # add fleet test
    try:
        collective_perf(
            "allgather",
            round=50,
            size_and_time={67108864: 0.00625, 234881024: 0.02, 637534208: 0.057},
        )
        logger.info("======monitor allgather done!=======\n")
        collective_perf(
            "allreduce",
            round=50,
            size_and_time={67108864: 0.02, 134217728: 0.038, 268435456: 0.075},
        )
        logger.info("======monitor allreduce done!=======\n")
    except Exception as e:
        logger.warning("fleet test unexcepted error! skip exception[{}]...".format(e))

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Define the metrics of tasks.
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        output = paddle.to_tensor(preds)
        labels = paddle.to_tensor(p.label_ids)
        output = [t.astype("float32").cuda() for t in output]
        labels = [t[t != tokenizer.ignored_index] for t in labels]
        labels = [t.cuda() for t in labels]
        all_numel = (
            (paddle.concat(labels, 0) != tokenizer.ignored_index).astype("int64").sum()
        )
        ignored = (paddle.concat(labels, 0) == -100).astype("int64").sum()
        labels = all_numel - ignored
        output = sum(output)
        logger.info(f"output : {output.item()}, labels : {labels.item()}")
        nll_loss = output / (labels + 1.0e-6)  # nll_loss is global loss
        ppl = paddle.exp(nll_loss)

        return {
            "nll_loss": nll_loss.item(),
            "ppl": ppl.item(),
            "num_token": labels.item(),
        }

    # model
    dtype = "float32"
    if args.fp16 and args.fp16_opt_level == "O2":
        paddle.set_default_dtype("float16")
        dtype = "float16"
    elif args.bf16:
        paddle.set_default_dtype("bfloat16")
        dtype = "bfloat16"

    if args.use_moe:
        global ErnieConfig, ErnieForCausalLMAuto
        """moe 情况下 Config, Causal组网，CausalPP组网，接口与非moe情况完全一致。"""
        # TODO：随着支持的网络越来越多，需要引入组网和 Config 的注册机制, 参考 Paddlenlp 的 `AutoModel`
        ErnieConfig = ErnieMoEConfig
        # ErnieForCausalLMAuto = ErnieMoEForCausalLMAuto

    if args.moe_group.lower() in {"mp", "tp", "model", "dummy"}:
        logger.info(f"disable moe flag when using moe-group={args.moe_group}")
        args.use_moe = False

    cfg = ErnieConfig.from_pretrained(args.model_name_or_path)
    cfg = update_model_config_from_args(cfg, model_config)  # 根据yaml更新modle_config
    cfg.seqlen = args.max_seq_length
    cfg.fp16_opt_level = args.fp16_opt_level
    cfg.moe_group = (
        args.moe_group
    )  # pp mp 下复用sharding 通信组作为moe通信组，其他情况moe通信组都是全局通信组。
    cfg.dtype = dtype
    cfg.pipeline_parallel_degree = args.pipeline_parallel_degree
    cfg.virtual_pp_degree = args.virtual_pp_degree
    if args.tensor_parallel_degree > 1:
        cfg.sequence_parallel = args.sequence_parallel
        cfg.tensor_parallel_degree = max(
            fleet.get_hybrid_communicate_group().get_model_parallel_world_size(), 1
        )
        cfg.tensor_parallel_rank = max(
            fleet.get_hybrid_communicate_group().get_model_parallel_rank(), 0
        )
    else:
        cfg.sequence_parallel = False
        cfg.tensor_parallel_degree = 1
        cfg.tensor_parallel_rank = 0

    tokenizer = ErnieBotTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.ignored_index = cfg.ignored_index
    logger.info(
        f"using tokenizer={type(tokenizer)}, bos:{tokenizer.bos_token_id} "
        f"eos:{tokenizer.eos_token_id} pad:{tokenizer.pad_token_id} "
    )
    vocab = tokenizer.get_vocab()
    image_preprocess = None  # set if `vision_model_name_or_path is not None`

    

    if args.inception_model_name_or_path:
        inception_config = InceptionModelConfig.from_pretrained(
            args.inception_model_name_or_path,
            tensor_parallel_degree=cfg.tensor_parallel_degree,
            tensor_parallel_rank=cfg.tensor_parallel_rank,
        )
        cfg.inception_config = inception_config
    if args.model_type == "ernie":
        model_class = ErnieForCausalLMAuto
    elif args.model_type == "ernie_pp":
        model_class = ErnieForCausalLMAutoPP
    else:
        raise ValueError(f"not support model_type: {args.model_type}")

    if args.from_scratch:
        with paddle.LazyGuard():
            model = model_class(cfg)
    else:
        with paddle.LazyGuard():
            model = model_class.from_pretrained(
                args.model_name_or_path,
                config=cfg,
            )

    if image_preprocess is not None:
        model.add_image_preprocess(image_preprocess)

    cfg = model.config
    logger.info(f"using model type:{type(model)}")
    paddle.set_default_dtype("float32")

    logger.info(f"using model={type(model)}, cfg={cfg}")
    if args.init_ckpt:
        load_model(model, args.init_ckpt)

    freeze_config = set(args.freeze_config.split(" "))
    if "freeze_vision" in freeze_config and hasattr(model, "freeze_vision"):
        logger.info("Freeze model vision module")
        model.freeze_vision()

    # data
    logger.info("loading data...")
    train_file_list, data_weights = parse_data_weight(
        args.data_weights, args.data_filelist
    )
    # train_dataset, eval_dataset, test_dataset, data_collator = create_pretrained_dataset(args)

    max_seq_length = args.max_seq_length

    if args.do_train:
        if args.use_dummy_dataset:
            from src.datasets.pretrain_task import PretrainDummyDataset

            train_dataset = PretrainDummyDataset(args.base_seq_length)
            train_dataset = MapDataset(train_dataset)
            logger.warning("USING DUMMY DATASET !!!")

        elif args.use_streaming_data:
            from src.datasets.streaming_pretrain_reader.streaming_pretrain_task import (
                KnoverDataset,
                create_pyreader,
            )

            logger.info("loading plain text data...")
            with open(config.model_args.train_task_config) as f:
                train_task_group = json.load(f)

            dp_worldsize = args.reeao_data_world_size
            dataset_random_seed = args.seed + args.dataset_rank
            config_dataset = {
                "max_seq_len": max_seq_length,
                "vocab_path": args.tokenizer_name,
                "is_valid": False,
                "add_ditto": False,
                "training_mode": "AR",
                "task_need_convert": args.task_need_convert,
                "batch_size": max_seq_length,  # means total token's number
                "random_seed": dataset_random_seed,
                "dp_worldsize": dp_worldsize,
                "voc_size": tokenizer.vocab_size,
                "task_group": train_task_group,
                # "ignored_index": args.ignored_index,
                "acc_steps": args.gradient_accumulation_steps,
                "worker_index": args.dataset_rank,  # 暂时没用到，可有可无，写日志可以用一把
                "output_dir": args.output_dir,
            }
            # logger.debug(f"Pad ID: {tokenizer.pad_id}")
            # logger.debug(f"ignored_index: {tokenizer.ignored_index}")
            train_reader = create_pyreader(config_dataset)
            # `KnoverDataset` 处理所有的batchinng 和shuffle
            train_dataset = KnoverDataset(
                train_reader.data_generator(),
                args.per_device_train_batch_size,
                ignored_index=tokenizer.ignored_index,
                pad_id=tokenizer.pad_id,
            )
            DEBUG_PRINT_CNT = 0

            def collate_fn(batch):
                nonlocal DEBUG_PRINT_CNT
                batch = default_data_collator(batch, return_tensors="np")
                if DEBUG_PRINT_CNT < 3:
                    DEBUG_PRINT_CNT += 1
                    for k, v in batch.items():
                        logger.debug(
                            f"Example={DEBUG_PRINT_CNT} key={k}, len={len(v[0])if isinstance(v, np.ndarray) else 0}, "
                            f"value={v[0] if isinstance(v, np.ndarray) else v}"
                        )
                    logger.debug(
                        f"Example={DEBUG_PRINT_CNT} text={fancy_print(batch, tokenizer)}"
                    )
                return batch

        else:
            assert (
                args.max_seq_length // args.base_seq_length >= 1
                and args.max_seq_length % args.base_seq_length == 0
            )
            if args.combine_batch > 1:
                logger.info(
                    f"max seq length is larger than base_seq_length, use combine batch: {args.combine_batch}"
                )
                assert (
                    args.use_train_part_sharding
                ), "not `use_train_part_sharding` is not supported when using `combine_batch`"
                assert (
                    args.num_consecutive // args.combine_batch >= 1
                    and args.num_consecutive % args.combine_batch == 0
                ), "num_consecutive must be a multiple of max_seq_length / base_seq_length"
                assert (
                    args.data_weights
                ), "no `data_weights` is not supported when using `combine_batch`"
            max_seq_length = args.base_seq_length
            if args.need_data:
                if args.multimodal:
                    assert False, "Do not support multimodal!"
                else:
                    pretrain_task = PretrainTask(train_file_list, tokenizer)
                train_dataset = pretrain_task.train_data(
                    max_seq_length + 1,
                    stride=max_seq_length,
                    rng=random.Random(args.seed),
                    weights=data_weights,
                    evaluate=False,
                    seed=args.seed,
                    num_consecutive=args.num_consecutive,
                    shuffle=not args.no_part_shuffle,
                    combine_batch=args.combine_batch,
                    load_process_num=args.data_load_process_num,
                )
                train_dataset.load(
                    use_shard=args.use_train_part_sharding,
                    dp_rank=args.reeao_dataset_rank,
                    dp_size=args.reeao_dataset_world_size,
                )  # TODO: MP/PP时候需要传入dp-rank
                train_dataset = MapDataset(train_dataset)
            else:
                logger.info(
                    f"mp_{args.pipeline_parallel_rank}_pp{args.tensor_parallel_rank} no data needed, \
                              skip init train_dataset"
                )
                train_dataset = None
    else:
        train_dataset = None

    if args.do_eval:
        eval_dataset = PretrainTask(
            [[args.dev_data]],
            tokenizer,
            max_seq_len=max_seq_length,
        ).train_data(
            max_seq_length + 1,
            stride=max_seq_length,
            overlap_len=32,
            rng=random.Random(0),
            evaluate=True,
            shuffle=False,
        )
        eval_dataset.load(False, dp_rank=0, dp_size=1)
        eval_dataset = MapDataset(eval_dataset)
    else:
        eval_dataset = None

    if args.do_train and args.use_streaming_data:
        raise NotImplementedError("Streaming data is not supported for now.")
        data_collator = partial(
            merge_fn, tokenizer, pad_to_max_seqlen=args.max_seq_length
        )
    else:
        data_collator = partial(
            merge_fn_group_batch,
            tokenizer,
            pad_to_max_seqlen=args.max_seq_length,
            combine_batch=args.combine_batch,
            image_dtype="uint8",
        )
    callbacks = []
    callbacks = [DataTraceCallbackAuto()] if not args.use_dummy_dataset else []
    callbacks += [GlobalRNGCallback()]
    # if "freeze_lm" in freeze_config and args.modality_ratio is not None:
    #     callbacks += [MultiModalInterleaveCallback()]
    if args.pp_need_data_degree:
        callbacks += [PPNeedDataCallback()]

    if args.batch_size_warmup_steps:
        progreesive_batcing_callback = ProgreesiveBatchingCallback(
            args.gradient_accumulation_steps,
            args.max_gradient_accumulation_steps,
            args.batch_size_warmup_steps,
            args.batch_size_warmup_increment,
        )
        callbacks.append(progreesive_batcing_callback)

    rng_states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cuda": paddle.get_rng_state(),
        "cpu": paddle.framework.core.default_cpu_generator().get_state(),
    }
    init_parameter(model)
    model.apply(model.init_weights)
    trainer = AutoPretrainingTrainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    global_training_logs.accumulate = args.gradient_accumulation_steps
    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Training
    if args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model(args.output_dir)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate and tests model
    if args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()
