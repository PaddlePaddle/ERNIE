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
""" Training Ernie VL Model. """

import os
import random
from functools import partial

import numpy as np
import paddle

from paddle.distributed import fleet
from paddleformers.datasets import IterDataset
from paddleformers.trainer import get_last_checkpoint
from paddleformers.utils.log import logger
from paddleformers.utils.tools import get_env_device
from .trainer import SFTTrainer

from ernie.callbacks import (
    GlobalRNGCallback,
    MoECorrectionBiasAdjustCallback,
    MultiModalInterleaveCallback,
    OrthogonalCallback,
    PPNeedDataCallback,
    VitTrainableCallback,
)
from ernie.configuration import Ernie4_5_VLMoeConfig
from ernie.dataset.text_sft_reader.sft_task import KnoverDataset, create_pyreader
from ernie.dataset.vl_sft_reader import (
    MixExampleSetJson,
    SFTMultimodalDatasetJson,
)
from ernie.dataset.vl_sft_reader.data_utils import merge_fn_group_batch

from ernie.modeling_moe_vl import Ernie4_5_VLMoeForConditionalGeneration
from ernie.tokenizer_vl import Ernie4_5_VLTokenizer
from ernie.modeling_moe_vl_pp import Ernie4_5_VLMoeForConditionalGenerationPipe
from ernie.utils.misc import global_training_logs
from ernie.utils.mm_data_utils import MMSpecialTokensConfig
from ernie.utils.seed_utils import set_seed

from data_processor.steps.end2end_processing import (
    End2EndProcessor,
)
from data_processor.image_preprocessor.image_preprocessor_adaptive import (
    AdaptiveImageProcessor,
)


def get_resume_checkpoint_path(config):
    """
    get resume checkpoint path from mpirun env
    """
    pdc_init_step = os.getenv("PDC_INIT_STEP")
    if (
        not hasattr(config, "resume_from_checkpoint")
        or config.resume_from_checkpoint is None
        or config.resume_from_checkpoint == ""
        or config.resume_from_checkpoint == "auto"
    ):
        if pdc_init_step is None:
            logger.info(
                "launching training process from scratch with no resume step defined."
            )
            return None
        elif pdc_init_step == "0":
            # from_scratch train process launched by pdc longjob
            logger.info(
                f"resume training process by pdc longjob with resume step: {pdc_init_step}"
            )
            return None
        else:
            # injected with mpirun by pdc longjob
            logger.info(
                f"resume training process by pdc longjob with resume step: {pdc_init_step}"
            )
            return os.path.join(config.output_dir, f"checkpoint-{pdc_init_step}")
    else:
        assert pdc_init_step is None, (
            "setting resume_from_checkpoint by yaml is deprecated in longjob, "
            + "please remove resume_from_checkpoint from yaml "
            + "and use script/restart.sh or mpirun -x PDC_INIT_STEP=<value> bash script/train.sh ..."
        )
        # user defined resume_from_checkpoint
        user_defined_resume_from_checkpoint = getattr(
            config, "resume_from_checkpoint", None
        )
        logger.info(
            f"user has defined resume_from_checkpoint: {user_defined_resume_from_checkpoint}"
        )
        return user_defined_resume_from_checkpoint


def run_vl_sft(
    model_args,
    data_args,
    preprocess_args,
    generating_args,
    finetuning_args,
):
    """
    main function
    """

    preprocess_args.batch_size = finetuning_args.batch_size
    finetuning_args.max_seq_len = data_args.max_seq_len
    finetuning_args.max_seq_length = data_args.max_seq_len

    # create output dir
    os.makedirs(finetuning_args.output_dir, exist_ok=True)

    if not hasattr(finetuning_args, "pipeline_parallel_config"):
        finetuning_args.pipeline_parallel_config = ""

    if getattr(finetuning_args, "sequence_parallel", 0):
        logger.warning(
            "disabling `disable_partial_send_recv` when using sequence parallel"
        )
        finetuning_args.pipeline_parallel_config += " disable_partial_send_recv"

    if (
        getattr(finetuning_args, "bf16", False)
        and "enable_delay_scale_loss" not in finetuning_args.pipeline_parallel_config
    ):
        logger.warning(
            "It is recommended to enable delay_scale_loss for better performance "
            "of precision when using bf16 in training"
        )
        finetuning_args.pipeline_parallel_config += " enable_delay_scale_loss"

    if "enable_dp_comm_overlap" in finetuning_args.pipeline_parallel_config:
        logger.warning(
            "Pipeline dp_comm_overlap and FusedLinearWithGradAdd can not be used at "
            "the same time."
        )

    if "enable_timer" in finetuning_args.pipeline_parallel_config:
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallel,
        )

        PipelineParallel.timer_printer = lambda _: None

    finetuning_args.resume_from_checkpoint = get_resume_checkpoint_path(finetuning_args)
    if (
        finetuning_args.resume_from_checkpoint is not None
        and finetuning_args.resume_from_checkpoint != ""
    ):
        assert os.path.exists(
            finetuning_args.resume_from_checkpoint
        ), f"{finetuning_args.resume_from_checkpoint} does not exist."
    logger.info(
        f"setting resume_from_checkpoint: {finetuning_args.resume_from_checkpoint}"
    )

    # hacking data processor
    preprocess_args.max_seq_length = data_args.max_seq_len
    preprocess_args.data_filelist = None
    preprocess_args.serialize_output = False
    logger.info(preprocess_args)

    finetuning_args.enable_delay_scale_loss = (
        "enable_delay_scale_loss" in finetuning_args.pipeline_parallel_config
    )

    if finetuning_args.modality_ratio is not None:
        finetuning_args.modality_ratio = (
            "".join(finetuning_args.modality_ratio).replace("[", "").replace("]", "")
        )
        finetuning_args.modality_ratio = finetuning_args.modality_ratio.split(",")
        finetuning_args.modality_ratio = [
            int(i) for i in finetuning_args.modality_ratio
        ]
        finetuning_args.modality_interleave = (
            sum(finetuning_args.modality_ratio)
            if finetuning_args.modality_interleave == "acc"
            else sum(finetuning_args.modality_ratio)
            * finetuning_args.gradient_accumulation_steps
        )
        # finetuning_args.modality_ratio = [
        #     i / sum(finetuning_args.modality_ratio) for i in finetuning_args.modality_ratio
        # ]

    # same_data is set to "" and modifed here by default, but can be set to True/False explicitly
    if (
        not hasattr(finetuning_args, "same_data")
        or finetuning_args.same_data is None
        or finetuning_args.same_data == ""
        or finetuning_args.same_data == "auto"
    ):
        finetuning_args.same_data = True
    logger.info(f"setting same_data: {finetuning_args.same_data}")

    image_preprocess_save = AdaptiveImageProcessor.from_pretrained(
        model_args.model_name_or_path
    )
    print("data_processor_args:\n", preprocess_args)

    tokenizer = Ernie4_5_VLTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        model_max_length=data_args.max_seq_len,
    )
    data_processor = End2EndProcessor(preprocess_args, tokenizer, image_preprocess_save)
    data_processor.train().sft()
    logger.info(f"[DEBUG] data_processor_args: {preprocess_args}")

    paddle.set_device(finetuning_args.device)
    np.random.seed(finetuning_args.seed)
    random.seed(finetuning_args.seed)
    set_seed(finetuning_args.seed)

    if get_env_device() == "gpu":
        prop = paddle.device.cuda.get_device_properties()
        if prop.total_memory < finetuning_args.pre_alloc_memory * 1024 * 1024 * 1024:
            logger.warning(
                "Invalid value for `pre_alloc_memory`, so pre-allocating just failed."
            )
        elif finetuning_args.pre_alloc_memory > 0:
            logger.warning(
                f"pre-allocating a tensor whose memory capacity is {finetuning_args.pre_alloc_memory} GB "
                "and then release it."
            )
            memory_size = int(finetuning_args.pre_alloc_memory * 1024 * 1024 * 1024)
            x = paddle.empty([memory_size], dtype=paddle.uint8)
            del x

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(finetuning_args.output_dir)
        and finetuning_args.do_train
        and not finetuning_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(finetuning_args.output_dir)
        if last_checkpoint is None and len(os.listdir(finetuning_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({finetuning_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None
            and finetuning_args.resume_from_checkpoint is None
        ):
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
    if finetuning_args.fp16 and finetuning_args.fp16_opt_level == "O2":
        paddle.set_default_dtype("float16")
        dtype = "float16"
    elif finetuning_args.bf16:
        paddle.set_default_dtype("bfloat16")
        dtype = "bfloat16"

    if finetuning_args.use_moe:
        if finetuning_args.multimodal:
            if finetuning_args.pipeline_parallel_degree > 1:
                assert finetuning_args.pp_need_data

    if model_args.moe_group.lower() in {"mp", "tp", "model", "dummy"}:
        logger.info(f"disable moe flag when using moe-group={model_args.moe_group}")
        finetuning_args.use_moe = False

    cfg = Ernie4_5_VLMoeConfig.from_pretrained(
        os.path.join(model_args.model_name_or_path)
    )
    cfg.use_cache = False
    cfg.max_sequence_length = data_args.max_seq_len
    cfg.seqlen = data_args.max_seq_len
    cfg.token_balance_seqlen = (
        data_args.max_seq_len * finetuning_args.per_device_train_batch_size
    )
    cfg.fp16_opt_level = finetuning_args.fp16_opt_level
    cfg.moe_group = model_args.moe_group  # pp mp use sharding group as moe group
    cfg.dtype = dtype
    cfg.use_fp8 = finetuning_args.use_fp8
    cfg.disable_pipeline_warmup = finetuning_args.disable_pipeline_warmup
    cfg.enable_mtp_magic_send = finetuning_args.enable_mtp_magic_send
    if finetuning_args.tensor_parallel_degree > 1:
        cfg.sequence_parallel = finetuning_args.sequence_parallel
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

    cfg.vision_config.tensor_parallel_degree = cfg.tensor_parallel_degree
    cfg.vision_config.tensor_parallel_rank = cfg.tensor_parallel_rank
    cfg.pixel_hidden_size = cfg.vision_config.hidden_size
    cfg.im_patch_id = tokenizer.get_vocab()[
        MMSpecialTokensConfig.get_special_tokens_info()["image_placeholder"]
    ]
    cfg.max_text_id = cfg.im_patch_id

    image_preprocess = AdaptiveImageProcessor.from_pretrained(
        model_args.model_name_or_path
    )
    image_preprocess.image_mean_tensor = paddle.to_tensor(
        image_preprocess.image_mean, dtype="float32"
    ).reshape([1, 3, 1, 1])
    image_preprocess.image_std_tensor = paddle.to_tensor(
        image_preprocess.image_std, dtype="float32"
    ).reshape([1, 3, 1, 1])
    image_preprocess.rescale_factor = paddle.to_tensor(
        image_preprocess.rescale_factor, dtype="float32"
    )
    image_preprocess.image_mean_tensor = image_preprocess.image_mean_tensor.squeeze(
        [-2, -1]
    ).repeat_interleave(cfg.vision_config.patch_size**2 * 1, -1)
    image_preprocess.image_std_tensor = image_preprocess.image_std_tensor.squeeze(
        [-2, -1]
    ).repeat_interleave(cfg.vision_config.patch_size**2 * 1, -1)

    cfg.use_flash_attention = model_args.use_flash_attention
    cfg.use_recompute_moe = model_args.use_recompute_moe
    cfg.recompute = finetuning_args.recompute
    cfg.recompute_granularity = model_args.recompute_granularity
    cfg.use_recompute_loss_fn = model_args.use_recompute_loss_fn
    cfg.use_sparse_head_and_loss_fn = model_args.use_sparse_head_and_loss_fn
    cfg.use_fused_head_and_loss_fn = model_args.use_fused_head_and_loss_fn
    cfg.use_mem_eff_attn = model_args.use_mem_eff_attn
    cfg.use_flash_attn_with_mask = model_args.use_flash_attn_with_mask
    cfg.hidden_dropout_prob = finetuning_args.hidden_dropout_prob
    cfg.moe_dropout_prob = finetuning_args.moe_dropout_prob
    cfg.token_balance_loss = finetuning_args.token_balance_loss
    cfg.token_balance_seqlen = (
        data_args.max_seq_len * finetuning_args.per_device_train_batch_size
    )

    if finetuning_args.pipeline_parallel_degree > 1:  # pp
        print(f"[sft-debug]: virtual_pp_degree={model_args.virtual_pp_degree}")
        cfg.virtual_pp_degree = model_args.virtual_pp_degree
        cfg.num_acc_steps = finetuning_args.gradient_accumulation_steps
        cfg.moe_with_send_router_loss = model_args.moe_with_send_router_loss
        cfg.enable_delay_scale_loss = finetuning_args.enable_delay_scale_loss
        cfg.balanced_image_preprocess = finetuning_args.balanced_image_preprocess

        if finetuning_args.pp_need_data and not finetuning_args.pp_need_data_degree:
            finetuning_args.pp_need_data_degree = (
                finetuning_args.pipeline_parallel_degree
            )

        if cfg.balanced_image_preprocess:
            assert (
                finetuning_args.pp_need_data
            ), "balanced image preprocess must use with pp_need_data"

        if finetuning_args.from_scratch:
            model = Ernie4_5_VLMoeForConditionalGenerationPipe(cfg)

        else:
            model = Ernie4_5_VLMoeForConditionalGenerationPipe.from_pretrained(
                model_args.model_name_or_path,
                config=cfg,
            )
        if finetuning_args.pp_need_data_degree:
            model.set_pp_need_data_degree(finetuning_args.pp_need_data_degree)
    else:
        if finetuning_args.from_scratch:
            model = Ernie4_5_VLMoeForConditionalGeneration(cfg)
        else:
            model = Ernie4_5_VLMoeForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                config=cfg,
            )
    logger.info(f"vision_model: {model.vision_model}")

    if image_preprocess is not None and hasattr(model, "add_image_preprocess"):
        model.add_image_preprocess(image_preprocess)

    cfg = model.config
    logger.info(f"using model type:{type(model)}")
    paddle.set_default_dtype("float32")

    logger.info(f"using model={type(model)}, cfg={cfg}")
    ortho_loss_lambda = (
        cfg.moe_orthogonal_loss_lambda
        if hasattr(cfg, "moe_orthogonal_loss_lambda")
        else 0.0
    )
    if finetuning_args.use_ortho_loss_callback:
        logger.info("using orthogonal loss callback")
        cfg.moe_orthogonal_loss_lambda = 0.0

    freeze_config = set(finetuning_args.freeze_config.split(" "))
    if "freeze_vision" in freeze_config and hasattr(model, "freeze_vision"):
        logger.info("Freeze model vision module")
        model.freeze_vision()

    # data
    logger.info("loading data...")
    logger.info(f"args.need_data: {finetuning_args.need_data}")

    if finetuning_args.do_train:
        hcg = fleet.get_hybrid_communicate_group()
        dp_rank = hcg.get_data_parallel_rank()
        dp_size = hcg.get_data_parallel_world_size()
        sharding_rank = hcg.get_sharding_parallel_rank()
        sharding_size = hcg.get_sharding_parallel_world_size()
        logger.info(
            f"""[main] hcg: dp_rank: {dp_rank},
            dp_size: {dp_size},
            sharding_rank: {sharding_rank},
            sharding_size: {sharding_size}"""
        )

        finetuning_args.is_train_mm = getattr(data_args, "train_dataset_path", False)
        finetuning_args.is_train_text = getattr(data_args, "text_dataset_path", False)
        assert (
            finetuning_args.is_train_text or finetuning_args.is_train_mm
        ), "At least one of is_train_text or is_train_mm must be specified"

        logger.info(f"[modality_ratio]: {finetuning_args.modality_ratio}")
        modality_ratio = (
            eval(str(finetuning_args.modality_ratio))
            if finetuning_args.modality_ratio is not None
            else [1, 1]
        )
        assert (
            type(modality_ratio) is list and len(modality_ratio) == 2
        ), "Only two modalities are supported."
        for idx, ratio in enumerate(modality_ratio):
            if int(ratio) <= 0:
                assert False, "Ratio Must be greater than zero."
            else:
                if int(ratio) != ratio:
                    logger.warning(
                        f"convert modality_ratio[{idx}] from {ratio} to {int(ratio)}"
                    )
                modality_ratio[idx] = int(ratio)

        if finetuning_args.need_data:
            if finetuning_args.multimodal and finetuning_args.is_train_mm:
                train_dataset = SFTMultimodalDatasetJson(
                    finetuning_args,
                    train_dataset_path=data_args.train_dataset_path,
                    train_dataset_prob=data_args.train_dataset_prob,
                    tokenizer=tokenizer,
                    image_preprocess=image_preprocess,
                    seed=finetuning_args.seed,
                    image_token_len=64,
                    seqlen=data_args.max_seq_len,
                    use_prompt=True,
                    mix_resolution=False,
                    special_token_loss_mask_ratio=0.5,
                    adaptive_resolution=False,
                    im_patch_id=tokenizer.get_vocab()[
                        MMSpecialTokensConfig.get_special_tokens_info()[
                            "image_placeholder"
                        ]
                    ],
                    batch_size=finetuning_args.per_device_train_batch_size,
                    dp_rank=finetuning_args.reeao_dataset_rank,
                    dp_size=finetuning_args.reeao_dataset_world_size,
                    data_processor=data_processor,
                )
                train_dataset._load(shuffle_json=True)
                train_dataset = IterDataset(
                    MixExampleSetJson(
                        finetuning_args,
                        lm_weights=0.0,
                        mm_weights=1.0,
                        lm_example_set=None,
                        mm_example_set=train_dataset,
                    )
                )
            else:
                train_dataset = None
        else:
            logger.info(
                f"mp_{finetuning_args.pipeline_parallel_rank}_pp{finetuning_args.tensor_parallel_rank} no data needed, \
                            skip init train_dataset"
            )
            train_dataset = None

        text_sft_dataset = None
        if finetuning_args.need_data and finetuning_args.is_train_text:
            # text SFT close multi-thread processing data
            text_dataset_path_list = (
                str(data_args.text_dataset_path).replace(" ", "").split(",")
            )
            text_dataset_prob_list = (
                str(data_args.text_dataset_prob).replace(" ", "").split(",")
            )
            train_task_group_text = []
            for filepath, prob in zip(text_dataset_path_list, text_dataset_prob_list):
                train_task_group_text.append(
                    {"filepath": filepath, "prob": float(prob)}
                )
            logger.info(f"train_task_group_text: {train_task_group_text}")

            dataset_seed = (
                finetuning_args.seed
                * finetuning_args.factor
                * (finetuning_args.reeao_dataset_rank + 1)
                if finetuning_args.text_use_train_part_sharding
                else finetuning_args.seed * finetuning_args.factor
            )
            config_dataset_text = {
                "dataset_name": data_args.dataset_name,
                "is_valid": False,
                "max_seq_len": data_args.max_seq_len,
                "random_seed": dataset_seed,
                "dp_worldrank": finetuning_args.reeao_dataset_rank,
                "dp_worldsize": finetuning_args.reeao_dataset_world_size,
                "worker_index": paddle.distributed.get_rank(),
                "prefetch_factor": finetuning_args.prefetch_factor,
                "task_group": train_task_group_text,
                "in_tokens": True,  # True for Text SFT
                "tokenizer": tokenizer,
                "number_of_samples_each_epoch": data_args.num_samples_each_epoch,
                "pseudo_strategy": finetuning_args.pseudo_strategy,
                "example_from_same_task_prob": finetuning_args.example_from_same_task_prob,
                "pseudo_sampling_prob": finetuning_args.pseudo_sampling_prob,
                "trigger_data_prob": finetuning_args.trigger_data_prob,
                "drop_history_with_k": finetuning_args.drop_history_with_k,
                "add_sys_token": finetuning_args.add_sys_token,
                "ignore_load_lr_and_optim": finetuning_args.ignore_load_lr_and_optim,
                "load_optimizer_and_scheduler": True,
                "resume_from_checkpoint": finetuning_args.resume_from_checkpoint,
                "sampling_wo_replacement_data_resuming": finetuning_args.sampling_wo_replacement_data_resuming,
                "min_shot": finetuning_args.min_shot,
                "max_shot": finetuning_args.max_shot,
                "use_train_part_sharding": finetuning_args.text_use_train_part_sharding,
                "rope_3d": model_args.rope_3d,
            }

            text_sft_train_reader = create_pyreader(config_dataset_text)
            text_sft_generator = text_sft_train_reader.data_generator()
            # update each task's sample number
            for task_in_reader, task in zip(
                text_sft_train_reader.task_group, train_task_group_text
            ):
                if "total_num_examples" in task_in_reader:
                    task["total_num_examples"] = task_in_reader["total_num_examples"]

            text_sft_dataset = KnoverDataset(
                text_sft_generator,
                finetuning_args.per_device_train_batch_size,
                ignored_index=tokenizer.ignored_index,
                task_group=train_task_group_text,
                use_mem_eff_attn=True,
            )
        else:
            logger.info("[TEXT SFT] not training pure text sft.")

    else:
        train_dataset = None
        text_sft_dataset = None
        modality_ratio = None

    eval_dataset = None

    data_collator = partial(
        merge_fn_group_batch,
        tokenizer=tokenizer,
        pad_to_max_seqlen=data_args.max_seq_len,
        im_prefix_length=256,
        rng=random.Random(2024),
        combine_batch=1,
    )

    callbacks = []
    callbacks += [GlobalRNGCallback()]
    if "freeze_lm" in freeze_config:
        if finetuning_args.modality_ratio is not None:
            callbacks += [MultiModalInterleaveCallback()]
        elif hasattr(model, "update_params_stat"):
            logger.info("Freeze model lm module")
            model.update_params_stat("lm", stop_gradient=True)
    if finetuning_args.pp_need_data:
        callbacks += [PPNeedDataCallback()]

    callbacks += (
        [OrthogonalCallback(ortho_loss_lambda)]
        if finetuning_args.use_ortho_loss_callback
        else []
    )
    if finetuning_args.pp_need_data_degree:
        callbacks += [PPNeedDataCallback()]

    if getattr(cfg, "moe_use_aux_free", 0.0) > 0.0:
        logger.info("adding aux free callback")
        callbacks += [
            MoECorrectionBiasAdjustCallback(
                finetuning_args.moe_use_aux_free_update_coef,
                finetuning_args.sequence_parallel,
            )
        ]

    vit_trainable_callback = None
    if (
        finetuning_args.pipeline_parallel_degree > 1
        and "freeze_vision" not in freeze_config
        and finetuning_args.multimodal
    ):
        # train VIT
        vit_trainable_callback = VitTrainableCallback(finetuning_args, model)

    trainer = SFTTrainer(
        model=model,
        args=finetuning_args,
        data_collator=data_collator,
        is_train_mm=finetuning_args.is_train_mm,
        train_dataset=train_dataset,
        is_train_text=finetuning_args.is_train_text,
        text_sft_dataset=text_sft_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        modality_ratio=modality_ratio,
        processing_class=image_preprocess_save,
    )
    if vit_trainable_callback is not None:
        vit_trainable_callback.auto_cast_func = trainer.autocast_smart_context_manager
        trainer.add_callback(vit_trainable_callback)

    global_training_logs.accumulate = finetuning_args.gradient_accumulation_steps
    checkpoint = None
    if finetuning_args.resume_from_checkpoint is not None:
        checkpoint = finetuning_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Training
    if finetuning_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model(finetuning_args.output_dir)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate and tests model
    if finetuning_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
