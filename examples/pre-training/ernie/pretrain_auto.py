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

import json
import os
import random
import time
from typing import Dict, Any

import numpy as np
import paddle
from omegaconf import ListConfig, DictConfig
from paddle.distributed.fleet import fleet, monitor_perf as collective_perf

from paddleformers.trainer import PdArgumentParser, get_last_checkpoint

from config import get_config
from models.ernie import (
    ErnieConfig,
    ErnieMoEConfig,
    ErnieForCausalLMAuto,
    ErnieForCausalLMAutoPP,
)
from pretrain import create_pretrained_dataset
from src.callbacks import GlobalRNGCallback
from src.tokenizers.tokenization_eb_v2 import ErnieBotTokenizer
from src.trainers import AutoPretrainingTrainer, AutoPreTrainingArguments
from src.utils import logger, setup_logger_output_file
from src.utils.misc import global_training_logs

def log_trainer_start():
    if "MAIN_PROCESS_STARTED" not in os.environ:
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info(
            f"Training Main Process Started. time: {start_time}, pid: {os.getpid()}"
        )
        os.environ["MAIN_PROCESS_STARTED"] = "1"


def format_config_value(v):
    if isinstance(v, (ListConfig, DictConfig)):
        return list(v) if isinstance(v, ListConfig) else dict(v)
    return v


def update_model_config_from_args(config: ErnieConfig, model_args: Dict[str, Any]) -> ErnieConfig:
    for k, v in model_args.items():
        if hasattr(config, k):
            logger.info(f"Updating model config: {k} = {v}")
            setattr(config, k, v)
        else:
            logger.warning(f"Model config key '{k}' does not exist")
    return config


def init_parameters(model):
    for param in model.parameters():
        param.initialize()
    model.apply(model.init_weights)


def setup_device_and_seed(args):
    paddle.set_device(args.device)
    np.random.seed(args.seed)
    random.seed(args.seed)
    paddle.seed(args.seed)


def check_memory_preallocation(args):
    prop = paddle.device.cuda.get_device_properties()
    if prop.total_memory < args.pre_alloc_memory * (1024**3):
        logger.warning("Invalid value for `pre_alloc_memory`, pre-allocation failed.")
    elif args.pre_alloc_memory > 0:
        logger.warning(f"Pre-allocating a tensor {args.pre_alloc_memory}GB memory and then release it")
        memory_size = int(args.pre_alloc_memory * 1024**3)
        x = paddle.empty([memory_size], dtype=paddle.uint8)
        del x


def run_fleet_tests():
    try:
        tests = [
            ("allgather", {67108864: 0.00625, 234881024: 0.02, 637534208: 0.057}),
            ("allreduce", {67108864: 0.02, 134217728: 0.038, 268435456: 0.075}),
        ]
        for test_name, size_time_map in tests:
            collective_perf(test_name, round=50, size_and_time=size_time_map)
            logger.info(f"======monitor {test_name} done!=======\n")
    except Exception as e:
        logger.warning(f"Fleet test error: {e}, skipping...")

def compute_metrics(p, tokenizer):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    output = paddle.to_tensor(preds)
    labels = paddle.to_tensor(p.label_ids)
    
    output = [t.astype("float32").cuda() for t in output]
    labels = [t[t != tokenizer.ignored_index].cuda() for t in labels]
    
    all_numel = (paddle.concat(labels, 0) != tokenizer.ignored_index).astype("int64").sum()
    ignored = (paddle.concat(labels, 0) == -100).astype("int64").sum()
    valid_tokens = all_numel - ignored
    
    total_output = sum(output)
    nll_loss = total_output / (valid_tokens + 1e-6)
    ppl = paddle.exp(nll_loss)
    
    logger.info(f"Output: {output[0].item()}, Valid tokens: {valid_tokens.item()}")
    
    return {
        "nll_loss": nll_loss.item(),
        "ppl": ppl.item(),
        "num_token": valid_tokens.item(),
    }

def setup_model_config(args, model_config):
    config_cls = ErnieMoEConfig if args.use_moe else ErnieConfig
    if args.moe_group.lower() in {"mp", "tp", "model", "dummy"}:
        logger.info(f"disable moe flag when using moe-group={args.moe_group}")
        args.use_moe = False
    args.multi_token_pred_depth = model_config.get("multi_token_pred_depth", 0)
    cfg = config_cls.from_pretrained(args.model_name_or_path)
    
    update_params = {
        "seqlen": args.max_seq_length,
        "token_balance_seqlen": args.max_seq_length * args.per_device_train_batch_size,
        "fp16_opt_level": args.fp16_opt_level,
        "moe_group": args.moe_group,
        "dtype": get_dtype(args),
        "pipeline_parallel_degree": args.pipeline_parallel_degree,
        "virtual_pp_degree": args.virtual_pp_degree,
        "micro_batch_size": args.per_device_train_batch_size,
    }
    
    for key, value in update_params.items():
        setattr(cfg, key, value)
    
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
    
    return update_model_config_from_args(cfg, model_config)


def get_dtype(args):
    if args.fp16 and args.fp16_opt_level == "O2":
        return "float16"
    if args.bf16:
        return "bfloat16"
    return "float32"

def set_dtype(args):
    if args.fp16 and args.fp16_opt_level == "O2":
        paddle.set_default_dtype("float16")
    if args.bf16:
        paddle.set_default_dtype("bfloat16")
    return


def get_model_class(args):
    if args.model_type == "ernie":
        return ErnieForCausalLMAuto
    if args.model_type == "ernie_pp":
        return ErnieForCausalLMAutoPP
    raise ValueError(f"Unsupported model_type: {args.model_type}")


def setup_tokenizer(args, config):
    tokenizer = ErnieBotTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.ignored_index = config.ignored_index
    logger.info(
        f"Using tokenizer={type(tokenizer)}, bos:{tokenizer.bos_token_id} "
        f"eos:{tokenizer.eos_token_id} pad:{tokenizer.pad_token_id}"
    )
    return tokenizer


def get_checkpoint(args, output_dir):
    if not os.path.isdir(output_dir) or not args.do_train or args.overwrite_output_dir:
        return None
    
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
        raise ValueError(
            f"Output directory ({output_dir}) exists and is not empty. "
            "Use --overwrite_output_dir to train from scratch."
        )
    if last_checkpoint is not None and args.resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. "
            "To avoid this, change --output_dir or add --overwrite_output_dir."
        )
    
    return args.resume_from_checkpoint or last_checkpoint

def setup_pipeline_config(args):
    if "enable_dp_comm_overlap" in args.pipeline_parallel_config:
        logger.warning(
            "Pipeline dp_comm_overlap and FusedLinearWithGradAdd cannot be used together."
        )
    if "enable_timer" in args.pipeline_parallel_config:
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import PipelineParallel
        PipelineParallel.timer_printer = lambda _: None
    if args.strategy.pipeline.enable and args.virtual_pp_degree > 1:
        pipeline = args.strategy.pipeline
        pipeline.vpp_degree = args.virtual_pp_degree
        pipeline.vpp_seg_method = args.virtual_pipeline_seg_method
    return args

def main():
    # 1. init config and parse arg
    config = get_config(verbose=True)
    if not hasattr(config.trainer_args, "pipeline_parallel_config"):
        config.trainer_args.pipeline_parallel_config = ""
    os.makedirs(config.model_args.output_dir, exist_ok=True)

    model_args = {k: format_config_value(v) for k, v in dict(config.model_args).items()}
    trainer_args = {k: format_config_value(v) for k, v in dict(config.trainer_args).items()}
    parser = PdArgumentParser(AutoPreTrainingArguments)
    (args,) = parser.parse_dict(dict(**model_args, **trainer_args))
    
    # 2. check and update
    # setup_pipeline_config(config.trainer_args)
    if "enable_dp_comm_overlap" in config.trainer_args.pipeline_parallel_config:
        logger.warning(
            "Pipeline dp_comm_overlap and FusedLinearWithGradAdd cannot be used together."
        )

    if "enable_timer" in config.trainer_args.pipeline_parallel_config:
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import PipelineParallel
        PipelineParallel.timer_printer = lambda _: None

    if args.strategy.pipeline.enable and args.virtual_pp_degree > 1:
        pipeline = args.strategy.pipeline
        pipeline.vpp_degree = args.virtual_pp_degree
        pipeline.vpp_seg_method = args.virtual_pipeline_seg_method
    
    args.use_moe = dict(**dict(config.model_args), **dict(config.trainer_args)).get("use_moe", False)
    args.eval_iters = 10
    args.test_iters = args.eval_iters * 10
    args.enable_delay_scale_loss = "enable_delay_scale_loss" in config.trainer_args.pipeline_parallel_config
    
    # 3. set log and device
    setup_logger_output_file(config.model_args.output_dir, args.local_rank)
    setup_device_and_seed(args)
    check_memory_preallocation(args)
    run_fleet_tests() # liyamei not need？
    set_dtype(args)
    
    # 4. init model
    model_config = {k: format_config_value(v) for k, v in dict(getattr(config.model_args, "model_config", {})).items()}
    logger.info(f"Model config from YAML: {json.dumps(model_config, indent=4)}")
    cfg = setup_model_config(args, model_config)
    model_class = get_model_class(args)
    tokenizer = setup_tokenizer(args, cfg)
    
    with paddle.LazyGuard():
        if args.from_scratch:
            model = model_class(cfg)
        else:
            model = model_class.from_pretrained(args.model_name_or_path, config=cfg)
    
    logger.info(f"Using model: {type(model)}, config: {model.config}")
    paddle.set_default_dtype("float32")
    
    # freeze
    freeze_config = set(args.freeze_config.split())
    if "freeze_vision" in freeze_config and hasattr(model, "freeze_vision"):
        logger.info("Freezing model vision module")
        model.freeze_vision()
    
    # 5. dataset
    logger.info("Loading datasets...")
    train_dataset, eval_dataset, test_dataset, data_collator = create_pretrained_dataset(args)

    # 6. prepare for train/eval
    callbacks = [GlobalRNGCallback()]
    init_parameters(model)
    
    trainer = AutoPretrainingTrainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
        callbacks=callbacks,
    )
    
    global_training_logs.accumulate = args.gradient_accumulation_steps
    checkpoint = get_checkpoint(args, args.output_dir)
    
    # 7.1 train
    if args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model(args.output_dir)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # 7.2 eval
    if args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)


if __name__ == "__main__":
    log_trainer_start()
    assert paddle.version.mkl() == "OFF", (
        "MKL is not supported in this version. "
        "Please set -DWITH_MKL=OFF when compiling PaddlePaddle."
    )
    
    main()