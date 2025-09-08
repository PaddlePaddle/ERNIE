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
from paddle.distributed.fleet import fleet
from paddle.distributed.auto_parallel import get_mesh
from paddleformers.data import Stack
from paddleformers.data.causal_dataset import (
    build_train_valid_test_datasets,
    check_data_split,
)
from paddle.distributed.fleet.meta_parallel.pipeline_parallel import PipelineParallel
from paddleformers.trainer.trainer_utils import get_last_checkpoint

from data_processor.utils.argparser import PdArgumentParser, get_config

from models.configuration import (
    ErnieConfig,
    ErnieMoEConfig,
)
from trainers import (
    PretrainingTrainer,
    PreTrainingArguments,
    MoECorrectionBiasAdjustCallback,
)

from utils import setup_logger_output_file, logger, mock_offload_optimizer
from utils.misc import global_training_logs

from tokenization import ErnieTokenizer

from paddle.distributed.fleet import collective_perf
from paddle import Tensor
from paddle import _C_ops
from paddle.distributed.fleet.base import topology as tp
from paddle.distributed import collective
from paddle.tensor.manipulation import reshape
from typing import Literal, TypeAlias

_ReduceMode: TypeAlias = Literal["mean", "sum", "none"]


# TODO: this function is rewrote from paddle.nn.functional.cross_entropy,
# but better to merge into only one.
def parallel_cross_entropy(
    input: Tensor,
    label: Tensor,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: _ReduceMode = "mean",
    soft_label: bool = False,
    axis: int = -1,
    use_softmax: bool = True,
    label_smoothing: float = 0.0,
    name: str | None = None,
) -> Tensor:
    if reduction not in ["sum", "mean", "none"]:
        raise ValueError(
            "The value of 'reduction' in softmax_cross_entropy"
            f"should be 'sum', 'mean' or 'none', but received {reduction}, which is not allowed."
        )
    if ignore_index > 0 and soft_label:
        raise ValueError(
            "When soft_label == True, the value of 'ignore_index' in softmax_cross_entropy"
            f"should be '-100', but received {ignore_index}, which is not allowed."
        )

    input_dims = len(list(input.shape))
    if input_dims == 0:
        raise ValueError("The dimension of input should be larger than zero!")

    label_dims = len(list(label.shape))
    if input_dims - 1 == label_dims:
        label = paddle.unsqueeze(label, axis=axis)

    if input_dims - 1 != label_dims and input_dims != label_dims:
        raise ValueError(
            f"Expected nput_dims - 1 = label_dims or input_dims == label_dims\
             (got nput_dims{input_dims}, label_dims{label_dims})"
        )

    if label_smoothing > 0.0:
        soft_label = True
        # converting the label to one-hot encoding
        # for 1d case, converting label's shape from [N] to [N, C]
        # for 2d case, converting label's shape from [N, d_1, ..., d_k] to [N, d_1, ..., d_k, C]
        if input_dims - 1 == label_dims:
            label = paddle.squeeze(label, axis=axis)
            label = paddle.nn.functional.one_hot(label, input.shape[-1])

        label = paddle.nn.functional.label_smooth(label, epsilon=label_smoothing)
        label = label.astype(input.dtype)
        label_dims = len(list(label.shape))

    if not soft_label:
        valid_label = paddle.cast(label != ignore_index, dtype=label.dtype) * label
    if not soft_label and is_tensor_sharded(input):
        group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
        ring_id = group.id
        nranks = group.nranks
        global_rank = collective._get_global_env().rank
        rank = group.get_group_rank(global_rank)
        _, out = _C_ops.c_softmax_with_cross_entropy(
            input, label, ignore_index, ring_id, rank, nranks
        )
    else:
        _, out = _C_ops.cross_entropy_with_softmax(
            input, label, soft_label, use_softmax, True, ignore_index, axis
        )

    if weight is not None:
        # trans weight from class to sample, shape:N or [N,H,W] for 1d and 2d cases.
        if soft_label:
            # chajchaj:
            # weight's shape is C, where C is class num.
            # for 1d case: label's shape is [N,C], weight_gather's shape is N.
            # for 2d case: label's shape is [N,H,W,C], weight_gather's shape is [N,H,W].
            weight_gather = paddle.matmul(
                x=paddle.cast(label, weight.dtype),
                y=weight,
                transpose_x=False,
                transpose_y=True,
            )
            out_shape = list(out.shape)
            weight_gather_reshape = reshape(weight_gather, shape=out_shape)
            out = paddle.cast(out, weight_gather_reshape.dtype)

            out = _C_ops.multiply(out, weight_gather_reshape)
        else:
            if input.shape[axis] != weight.shape[-1]:
                raise ValueError(
                    f"input's class_dimension({input.shape[axis]}) must equal to "
                    f"weight's class_dimension({weight.shape[-1]}) "
                    "when weight is provided"
                )

            ignore_weight_mask = paddle.cast((label != ignore_index), out.dtype)
            if ignore_weight_mask.ndim > 1 and ignore_weight_mask.shape[axis] == 1:
                # TODO: Temporarily use squeeze instead of squeeze_
                ignore_weight_mask = paddle.squeeze(ignore_weight_mask, axis)
            if axis != -1 and axis != valid_label.ndim - 1:
                temp_perm = (
                    list(range(axis % valid_label.ndim))
                    + list(range((axis % valid_label.ndim + 1), valid_label.ndim))
                    + [axis % valid_label.ndim]
                )
                weight_gather = _C_ops.gather_nd(
                    weight, valid_label.transpose(temp_perm)
                )
            else:
                weight_gather = _C_ops.gather_nd(weight, valid_label)
            weight_gather = _C_ops.multiply(weight_gather, ignore_weight_mask)
            input_shape = list(label.shape)
            weight_gather_reshape = reshape(weight_gather, shape=input_shape)
            out = paddle.cast(out, weight_gather_reshape.dtype)
            out = _C_ops.multiply(out, weight_gather_reshape)

    if reduction == "sum":
        #   because of base_softmax_with_cross_entropy op's inner logic,
        #   in the out tensor of this op, the loss of sample with class_index==ignore_index is 0
        #   so, reduce_sum all directly is ok
        return _C_ops.sum(out, [], None, False)
    elif reduction == "mean":
        # 1. if weight==none,
        #     numerator: reduce_sum all loss directly is ok causeof base_softmax_with_cross_entropy's inner logic
        #     denominator: count sample num with class_index!=ignore_index
        # 2. else
        #     numerator: loss's weighted sum
        #     denominator: cal the sum of weight where the sample's class_index!=ignore_index
        if ignore_index >= 0:  # ignore label
            out_sum = _C_ops.sum(out, [], None, False)
            # for each label[i],set 1 or 0, according to ignore_index
            # mask[i]=0, if label[i]==ignore_index
            # mask[i]=1, otherwise
            mask = label != ignore_index
            if weight is None:
                mask = paddle.cast(mask, dtype=out_sum.dtype)
                count = _C_ops.sum(mask, [], None, False)
                ret = out_sum / (count + (count == 0.0).astype(count.dtype))
            else:
                mask = paddle.cast(mask, weight_gather_reshape.dtype)
                weight_ignored = _C_ops.multiply(mask, weight_gather_reshape)
                weight_sum = _C_ops.sum(weight_ignored, [], None, False)
                ret = out_sum / (
                    weight_sum + (weight_sum == 0.0).astype(weight_sum.dtype)
                )
            return ret
        elif weight is not None:
            out_sum = _C_ops.sum(out, [], None, False)
            total_weight = _C_ops.sum(weight_gather_reshape, [], None, False)
            return out_sum / (
                total_weight + (total_weight == 0.0).astype(total_weight.dtype)
            )
        else:
            return _C_ops.mean_all(out)

    else:
        if input_dims - 1 == label_dims:
            out = paddle.squeeze(out, axis=axis)
        return out


# TODO: placement[1] may not be mp axis.
def is_tensor_sharded(tensor):
    if not tensor.is_dist():
        return False

    placement = tensor.placements
    return placement[1].is_shard()


def replace_cross_entropy():
    paddle.nn.functional.cross_entropy = parallel_cross_entropy


def log_trainer_start():
    if "MAIN_PROCESS_STARTED" not in os.environ:
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info(
            f"Training Main Process Started. time: {start_time}, pid: {os.getpid()}"
        )
        os.environ["MAIN_PROCESS_STARTED"] = "1"


def create_pretrained_dataset(args):
    assert args.input_dir is not None and len(args.input_dir.split()) > 1

    check_data_split(
        args.split,
        args.do_train,
        args.do_eval,
        args.do_predict,
    )

    train_val_test_num_samples = [
        args.per_device_train_batch_size
        * args.dataset_world_size
        * args.max_steps
        * args.gradient_accumulation_steps,
        args.per_device_eval_batch_size
        * args.dataset_world_size
        * args.eval_iters
        * (args.max_steps // args.eval_steps + 1),
        args.per_device_eval_batch_size * args.dataset_world_size * args.test_iters,
    ]

    train_dataset, valid_dataset, test_dataset = build_train_valid_test_datasets(
        data_prefix=args.input_dir.split(),
        data_impl="mmap",
        splits_string=args.split,
        train_val_test_num_samples=train_val_test_num_samples,
        seq_length=args.max_seq_length + args.multi_token_pred_depth,
        seed=args.seed,
        skip_warmup=True,
        data_cache_path=None,
    )

    def _collate_data(data, stack_fn=Stack()):
        tokens_ = stack_fn([x["text"] for x in data])

        labels = tokens_[:, 1:]
        tokens = tokens_[:, :-1]

        return {
            "input_ids": tokens,
            "labels": labels,
        }

    return train_dataset, valid_dataset, test_dataset, _collate_data


def format_config_value(v):
    if isinstance(v, (ListConfig, DictConfig)):
        return list(v) if isinstance(v, ListConfig) else dict(v)
    return v


def update_model_config_from_args(
    config: ErnieConfig, model_args: Dict[str, Any]
) -> ErnieConfig:
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
        logger.warning(
            f"Pre-allocating a tensor {args.pre_alloc_memory}GB memory and then release it"
        )
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

    all_numel = (
        (paddle.concat(labels, 0) != tokenizer.ignored_index).astype("int64").sum()
    )
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


def setup_tokenizer(args, config):
    tokenizer = ErnieTokenizer.from_pretrained(args.tokenizer_name)
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


def set_moe_config(config):
    if hasattr(config, "use_moe") and config.use_moe:
        if config.moe_group in {"mp", "model", "tp", "mpdp"}:
            assert config.sequence_parallel
            logger.info(
                f"disable FFN tensor model parallel, moe-group={config.moe_group}"
            )
            config.disable_ffn_model_parallel = True

        config.moe_world_size = 1
        if config.moe_group in get_mesh().dim_names:
            config.moe_world_size = max(
                config.moe_world_size,
                get_mesh().get_dim_size(config.moe_group),
            )


def main():
    # 1. init config and parse arg
    config = get_config(verbose=True)
    if not hasattr(config.trainer_args, "pipeline_parallel_config"):
        config.trainer_args.pipeline_parallel_config = ""
    os.makedirs(config.model_args.output_dir, exist_ok=True)

    model_args = {k: format_config_value(v) for k, v in dict(config.model_args).items()}
    trainer_args = {
        k: format_config_value(v) for k, v in dict(config.trainer_args).items()
    }
    parser = PdArgumentParser(PreTrainingArguments)
    (args,) = parser.parse_dict(dict(**model_args, **trainer_args))

    # 2. check and update
    if "enable_dp_comm_overlap" in config.trainer_args.pipeline_parallel_config:
        logger.warning(
            "Pipeline dp_comm_overlap and FusedLinearWithGradAdd cannot be used together."
        )

    if "enable_timer" in config.trainer_args.pipeline_parallel_config:
        PipelineParallel.timer_printer = lambda _: None

    if args.strategy.pipeline.enable and args.virtual_pp_degree > 1:
        pipeline = args.strategy.pipeline
        pipeline.vpp_degree = args.virtual_pp_degree
        pipeline.vpp_seg_method = args.virtual_pipeline_seg_method

    args.use_moe = dict(**dict(config.model_args), **dict(config.trainer_args)).get(
        "use_moe", False
    )
    args.eval_iters = 10
    args.test_iters = args.eval_iters * 10
    args.enable_delay_scale_loss = (
        "enable_delay_scale_loss" in config.trainer_args.pipeline_parallel_config
    )

    # 3. set log and device
    setup_logger_output_file(config.model_args.output_dir, args.local_rank)
    setup_device_and_seed(args)
    check_memory_preallocation(args)
    run_fleet_tests()
    set_dtype(args)

    # 4. init model
    model_config = {
        k: format_config_value(v)
        for k, v in dict(getattr(config.model_args, "model_config", {})).items()
    }
    logger.info(f"Model config from YAML: {json.dumps(model_config, indent=4)}")
    cfg = setup_model_config(args, model_config)
    if args.offload_optimizer:
        mock_offload_optimizer()
    if (
        "replace_with_parallel_cross_entropy" in args.tensor_parallel_config
        and cfg.tensor_parallel_degree > 1
        and not (args.use_intermediate_api and args.pipeline_schedule_mode == "FThenB")
    ):
        replace_cross_entropy()

    set_moe_config(cfg)

    tokenizer = setup_tokenizer(args, cfg)

    vpp_degree = cfg.virtual_pp_degree
    # vpp_degree==1: Implement parallelism using the intermediate API.
    # vpp_degree>1: Implement parallelism using the basic API; the intermediate API does not support VPP for the time being.
    assert vpp_degree >= 1, "vpp_degree must be greater than or equal to 1."
    if vpp_degree == 1:
        from models.modeling import ErnieForCausalLM, ErnieDecoderLayer

        logger.info("Training with the intermediate API. Do not support VPP.")
        modle_class = ErnieForCausalLM
        aux_free_class = ErnieDecoderLayer
    elif vpp_degree > 1:
        from models.modeling_vpp import ErnieForCausalLMVPP, ErnieDecoderLayerVPP

        logger.info("Training VPP parallelism with the basic API")
        modle_class = ErnieForCausalLMVPP
        aux_free_class = ErnieDecoderLayerVPP

    with paddle.LazyGuard():
        model = modle_class(cfg)

    logger.info(f"Using model: {type(model)}, config: {model.config}")
    paddle.set_default_dtype("float32")

    # 5. dataset
    logger.info("Loading datasets...")
    train_dataset, eval_dataset, test_dataset, data_collator = (
        create_pretrained_dataset(args)
    )

    # 6. prepare for train/eval
    callbacks = []
    if getattr(cfg, "moe_use_aux_free", False):
        logger.info("Adding aux free callback")
        callbacks += [
            MoECorrectionBiasAdjustCallback(
                args.moe_use_aux_free_update_coef,
                args.sequence_parallel,
                aux_free_class,
            )
        ]
    init_parameters(model)

    trainer = PretrainingTrainer(
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
