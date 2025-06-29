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

"""predict_vl_generation under dynamic graph"""

import argparse
import json
import logging
import os
import threading

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet.base.topology as tp
from paddle.distributed import fleet
from paddleformers.trainer import RuntimeTimer
from paddleformers.transformers.model_utils import load_tp_checkpoint
from tqdm import tqdm

from data_processor.image_preprocessor.image_preprocessor_adaptive import AdaptiveImageProcessor
from data_processor.steps.end2end_processing import (
    End2EndProcessor,
    End2EndProcessorArguments,
)
from data_processor.utils.argparser import PdArgumentParser, get_config
from ernie.configuration import Ernie4_5_VLMoeConfig
from ernie.modeling_moe_vl import Ernie4_5_VLMoeForConditionalGeneration
from ernie.tokenizer_vl import Ernie4_5_VLTokenizer
from ernie.utils.mm_data_utils import MMSpecialTokensConfig
from ernie.utils.seed_utils import set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
lock = threading.Lock()


def get_parser():
    """
    Create and configure an argument parser for model inference.

    Returns:
        argparse.ArgumentParser: Configured parser with all inference parameters

    The parser includes arguments for:
    - Model configuration
    - Generation parameters
    - Input/output handling
    - Performance optimization
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help="model path")
    parser.add_argument(
        "--input_file",
        type=str,
        default="./examples/inference/data/multimodal-query-answers-list-small.jsonl",
        help="your data input file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predict.json",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--vision_model_name_or_path",
        type=str,
        help="vision model path when use pdparameter",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--rope_3d", type=int, default=1)
    parser.add_argument("--variable_resolution", type=int, default=1, help="variable resolution vit")
    parser.add_argument("--max_seq_length", type=int, default=8182)
    parser.add_argument("--distributed", type=int, default=0, help="distributed mode")
    parser.add_argument("--gpu", type=str, default=None, help="gpu")
    parser.add_argument("--is_moe", type=int, default=1)
    parser.add_argument(
        "--dpconfig",
        type=str,
        default="data_processor/config_processor.yaml",
        help="data_processor config",
    )
    parser.add_argument("--crop_tile_option", type=str, default="36")
    parser.add_argument("--crop_tile_rate", type=str, default="1")
    parser.add_argument("--use_min_crop", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def setup_args():
    """
    Parse and return inference arguments.

    Returns:
        argparse.Namespace: Parsed arguments object

    This is the main entry point for getting configuration arguments.
    Simply combines parser creation with argument parsing.
    """
    parser = get_parser()
    args = parser.parse_args()
    return args


def enforce_stop_tokens(text, stop_sequences) -> str:
    """Code by Langchain"""
    """Cut off the text as soon as any stop words occur."""

    logger.info(f"[stop token] {[text]}")
    for stop in stop_sequences:
        text = text.split(stop)[0]
        logger.info(f"[stop token] {stop} {[text]}")
    return text


def init_dist_env(seed):
    """init tensor parallel env"""
    tensor_parallel_degree = paddle.distributed.get_world_size()
    tensor_parallel_rank = paddle.distributed.get_rank()

    if tensor_parallel_degree > 1:
        hcg = tp._HYBRID_PARALLEL_GROUP
        if hcg is None:
            strategy = fleet.DistributedStrategy()
            strategy.hybrid_configs = {
                "dp_degree": 1,
                "mp_degree": tensor_parallel_degree,
                "pp_degree": 1,
                "sharding_degree": 1,
            }
            fleet.init(is_collective=True, strategy=strategy)
            set_seed(seed)
            hcg = fleet.get_hybrid_communicate_group()

        tensor_parallel_rank = hcg.get_model_parallel_rank()
    return tensor_parallel_rank, tensor_parallel_degree


def infer_save_test_case(case: list[list[dict]], file: str):
    """save test to result file

    Args:
        cases (list[list[dict]]): the content of case
        file (str): the path of saved file
    """
    with open(file, "a+", encoding="utf-8") as f:
        raw = json.dumps(case, indent=4, ensure_ascii=False)
        f.write(raw + "\n")


class Predictor:
    """Predictor"""

    def __init__(self, args):
        """init"""
        self.runtime_timer = RuntimeTimer("Predictor")
        self.args = args

        tokenizer = Ernie4_5_VLTokenizer.from_pretrained(
            args.model_name_or_path,
            model_max_length=args.max_seq_length,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.ignored_index = -100
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        self.tokenizer = tokenizer

        if args.distributed:
            self.tensor_parallel_rank, self.tensor_parallel_degree = init_dist_env(args.seed)
        else:
            self.tensor_parallel_rank, self.tensor_parallel_degree = 0, 1

        self.black = [
            "reduce_sum",
            "c_softmax_with_cross_entropy",
            "elementwise_div",
            "sin",
            "cos",
            "sort",
            "multinomial",
        ]
        self.white = [
            "lookup_table",
            "lookup_table_v2",
            "flash_attn",
            "matmul",
            "matmul_v2",
            "fused_gemm_epilogue",
        ]
        logger.info("load mm model")
        self._init_mm_model(args)

    def _init_mm_model(self, args):
        """
        Initialize the mm model, including the data processor, model, etc.

        Args:
            args (argparse.Namespace):
            Command-line arguments containing information such as the model name and pretrained model path.

        Returns:
            None.
        """

        if not os.path.exists(os.path.join(args.model_name_or_path, "preprocessor_config.json")):
            assert args.vision_model_name_or_path is not None, "vision_model_name_or_path is None"
            vision_model_name_or_path = args.vision_model_name_or_path
        else:
            vision_model_name_or_path = args.model_name_or_path

        image_preprocess = AdaptiveImageProcessor.from_pretrained(vision_model_name_or_path)
        data_processor_config = get_config(
            args.dpconfig,
            tokenizer=args.model_name_or_path,
            vision_model_name_or_path=vision_model_name_or_path,
            save_to_disk=False,
            crop_tile_option=args.crop_tile_option,
            crop_tile_rate=args.crop_tile_rate,
            min_crop_flag=args.use_min_crop,
            variable_resolution=args.variable_resolution,
            rope_3d=args.rope_3d,
        )
        data_processor_parser = PdArgumentParser(End2EndProcessorArguments)
        self.processor = End2EndProcessor(
            data_processor_parser.parse_dict(dict(**dict(data_processor_config.processor_args))),
            tokenizer=self.tokenizer,
            image_preprocess=image_preprocess,
        )

        self.processor.eval()
        self.processor.sft()

        self.image_preprocess = AdaptiveImageProcessor.from_pretrained(vision_model_name_or_path)

        config = Ernie4_5_VLMoeConfig.from_pretrained(
            args.model_name_or_path,
            tensor_parallel_degree=self.tensor_parallel_degree,
            tensor_parallel_rank=self.tensor_parallel_rank,
            moe_group="dummy",
        )
        config.vision_config.attn_sep = False
        config.pixel_hidden_size = config.vision_config.hidden_size
        config.im_patch_id = self.tokenizer.get_vocab()[
            MMSpecialTokensConfig.get_special_tokens_info()["image_placeholder"]
        ]
        config.max_text_id = config.im_patch_id
        logger.info(f"[STAGE] image_placeholder_id: {config.im_patch_id}")

        config.moe_capacity = (
            [config.moe_num_experts[0] * 2] * 3
            if isinstance(config.moe_num_experts, (list, tuple))
            else [config.moe_num_experts] * 3
        )
        config.tensor_parallel_output = False
        config.sequence_parallel = False
        config.use_flash_attn = True
        if config.rope_3d != args.rope_3d:
            logger.warning(f"rope_3d not match, config.rope_3d: {config.rope_3d}, args.rope_3d: {args.rope_3d}")
            config.rope_3d = args.rope_3d
        if (
            isinstance(config.moe_multimodal_dispatch_use_allgather, str)
            and "v2" in config.moe_multimodal_dispatch_use_allgather
        ):
            config.moe_multimodal_dispatch_use_allgather = "v2"
        config.use_flash_attn_with_mask = True
        config.disable_ffn_model_parallel = False

        self.dtype = args.dtype
        logger.info(f"using dtype {args.dtype}")
        paddle.set_default_dtype(args.dtype)

        self.model = Ernie4_5_VLMoeForConditionalGeneration(config)
        self.config = self.model.config
        self.vision_config = config.vision_config

        state_dict = load_tp_checkpoint(
            args.model_name_or_path,
            cls=Ernie4_5_VLMoeForConditionalGeneration,
            config=self.config,
            return_numpy=True,
        )
        if config.tie_word_embeddings and "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["ernie.embed_tokens.weight"]

        has_vision_model = False
        for key in list(state_dict.keys()):
            if key.startswith("vision_model."):
                has_vision_model = True
            if key.startswith("resampler_model."):
                new_key = key.replace("resampler_model.", "ernie.resampler_model.")
                state_dict[new_key] = state_dict.pop(key)

        if not has_vision_model:
            assert args.vision_model_name_or_path is not None, "vision_model_name_or_path is None"
            vision_state_dict = paddle.load(os.path.join(args.vision_model_name_or_path, "model_state.pdparams"))
            for k in list(vision_state_dict.keys()):
                new_k = "vision_model." + k
                vision_state_dict[new_k] = vision_state_dict.pop(k)
            state_dict.update(vision_state_dict)

        logger.info(f"MODEL-CONFIG: {self.config}")
        logger.info("[STAGE] set state_dict")
        self.model.set_state_dict(state_dict)

        if config.dtype != "float32":
            self.model = paddle.amp.decorate(models=self.model, level="O2", dtype=args.dtype)

        self.model.eval()

    def _preprocess(self, inputs):
        """process batch"""
        # get some config for generate
        generation_configs = {
            "max_length": inputs[0]["max_dec_len"],
            "stop_sequences": inputs[0]["stop_sequences"],
            "top_p": inputs[0]["top_p"],
            "temperature": inputs[0]["temperature"],
            "top_k": inputs[0]["top_k"],
            "penalty_score": inputs[0]["penalty_score"],
            "frequency_score": inputs[0]["frequency_score"],
            "presence_score": inputs[0]["presence_score"],
            "eos_token_id": self.tokenizer._convert_token_to_id(self.tokenizer.eos_token),
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        logger.info(f"[gen config] {json.dumps(generation_configs, indent=4, ensure_ascii=False)}")

        try:
            one = self.processor.process(inputs[0])[0]
        except Exception as e:
            raise

        input_ids = one["input_ids"][np.newaxis, :]
        token_type_ids = one["token_type_ids"][np.newaxis, :]
        if one.get("position_ids", None) is not None:
            position_ids = one["position_ids"][np.newaxis, :]
        else:
            position_ids = None
        if one.get("images", None) is not None:
            image_type_ids = one["image_type_ids"][np.newaxis, :]
            images = one["images"]
            grid_thw = one.get("grid_thw", None)

            self.image_preprocess.image_mean_tensor = paddle.to_tensor(
                self.image_preprocess.image_mean, dtype="float32"
            ).reshape([1, 3, 1, 1])
            self.image_preprocess.image_std_tensor = paddle.to_tensor(
                self.image_preprocess.image_std, dtype="float32"
            ).reshape([1, 3, 1, 1])
            self.image_preprocess.rescale_factor = paddle.to_tensor(
                self.image_preprocess.rescale_factor, dtype="float32"
            )
            self.image_preprocess.image_mean_tensor = self.image_preprocess.image_mean_tensor.squeeze(
                [-2, -1]
            ).repeat_interleave(
                self.vision_config.patch_size**2,
                -1,
            )
            self.image_preprocess.image_std_tensor = self.image_preprocess.image_std_tensor.squeeze(
                [-2, -1]
            ).repeat_interleave(self.vision_config.patch_size**2, -1)
            images = self.image_preprocess.rescale_factor * images.astype("float32")
            images = (images - self.image_preprocess.image_mean_tensor) / self.image_preprocess.image_std_tensor

            # to tensor
            input_ids = paddle.to_tensor(input_ids, dtype=paddle.int64)
            image_type_ids = paddle.to_tensor(image_type_ids, dtype=paddle.int64)
            token_type_ids = paddle.to_tensor(token_type_ids, dtype=paddle.int64)
            images = paddle.to_tensor(images, dtype="bfloat16")
            if grid_thw is not None:
                grid_thw = paddle.to_tensor(grid_thw, dtype=paddle.int64)

            logger.info(f"[LOGCHW] input_ids {input_ids.shape} {input_ids.dtype}")
            logger.info(f"[LOGCHW] image_type_ids {image_type_ids.shape} {image_type_ids.dtype}")
            logger.info(f"[LOGCHW] token_type_ids {token_type_ids.shape} {token_type_ids.dtype}")
            logger.info(f"[LOGCHW] images {images.shape} {images.dtype}")
        else:
            image_type_ids, images, grid_thw = None, None, None
            input_ids = paddle.to_tensor(input_ids, dtype=paddle.int64)
            token_type_ids = paddle.to_tensor(token_type_ids, dtype=paddle.int64)
        if position_ids is not None:
            position_ids = paddle.to_tensor(position_ids, dtype=paddle.int64)

        return (
            dict(
                input_ids=input_ids,
                image_type_ids=image_type_ids,
                token_type_ids=token_type_ids,
                images=images,
                grid_thw=grid_thw,
                position_ids=position_ids,
            ),
            generation_configs,
        )

    def _infer(self, inputs, **kwargs):
        """infer"""

        with paddle.no_grad():
            with paddle.amp.auto_cast(
                True,
                custom_black_list=self.black,
                custom_white_list=self.white,
                level="O2",
                dtype=self.dtype,
            ):
                logger.info("[LOGCHW] before generate")
                out = self.model.generate(
                    input_ids=inputs["input_ids"],
                    token_type_ids=inputs["token_type_ids"],
                    image_type_ids=inputs["image_type_ids"],
                    images=inputs["images"],
                    grid_thw=inputs["grid_thw"],
                    position_ids=inputs["position_ids"],
                    **kwargs,
                )
                logger.info("[LOGCHW] after generate")
        return out

    def _postprocess(self, input_data, stop_sequences):
        """postprocess"""
        result = []
        for text_tensor in input_data:
            text_str = self.tokenizer.decode(text_tensor.numpy().tolist(), skip_special_tokens=False)
            text = enforce_stop_tokens(text_str, stop_sequences)
            result.append(text)
        return result

    def predict(self, inputs):
        """predict"""
        try:
            tokenized_source, generation_configs = self._preprocess(inputs)
            predictions = self._infer(tokenized_source, **generation_configs)
            decoded_predictions = self._postprocess(
                predictions[0],
                stop_sequences=generation_configs.get("stop_sequences", []),
            )
        except Exception as e:
            logger.info(e)
            import traceback

            decoded_predictions = [f"<|ERROR|> Please check API! ERROR {e!s}-{traceback.format_exc()}"]
        return decoded_predictions

    def predict_with_timeout(self, inputs, return_dict):
        """predict"""
        try:
            tokenized_source, generation_configs = self._preprocess(inputs)
            predictions = self._infer(tokenized_source, **generation_configs)
            decoded_predictions = self._postprocess(
                predictions[0],
                stop_sequences=generation_configs.get("stop_sequences", []),
            )
        except Exception as e:
            logger.info(e)
            import traceback

            decoded_predictions = [f"<|ERROR|> Please check API! ERROR {e!s}-{traceback.format_exc()}"]

        return_dict["result"] = decoded_predictions


if __name__ == "__main__":
    args = setup_args()
    #  Only support batch_size=1
    if args.batch_size > 1:
        print("`batch_size` is reset to 1. Only support batch_size=1.")
        args.batch_size = 1

    if args.distributed:
        dist.init_parallel_env()
    if args.gpu:
        paddle.set_device(f"gpu:{args.gpu}")
    predictor = Predictor(args)

    # Inference
    infer_dials: list[list[dict]] = []
    if args.input_file is None or not os.path.exists(args.input_file):
        print(f'input file not found: {args.input_file}, run default case')
        infer_dials = [
            {
                "context": [
                    {
                        "role": "user",
                        "utterance": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://ku.baidu-int.com/vk-assets-ltd/space/2025/05/"
                                    "15/7b09e8b14c724cf484fca6c0223a1dd8",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": "帮我check一下虚增营业收入的总额和虚增利润总额能不能和表格里写的对上？",
                            },
                        ],
                    }
                ],
                "max_dec_len": 12288,
                "stop_sequences": [],
                "top_p": 0.8,
                "top_k": 1,
                "temperature": 0.2,
                "penalty_score": 1.0,
                "frequency_score": 0.0,
                "presence_score": 0.0,
                "seed": 0,
                "prefix": "<think>\n\n</think>\n\n",
            }
        ]
    else:
        with open(args.input_file, "r") as fin:
            for i, line in enumerate(fin):
                cur_line = json.loads(line)
                infer_dials.append(cur_line)

    predictor.runtime_timer.start("predict stage running time")
    for idx in tqdm(range(0, len(infer_dials), args.batch_size)):
        batch_dials = infer_dials[idx : idx + args.batch_size]
        print("inputs ->", batch_dials)
        result = predictor.predict(batch_dials)
        print("result ->", result)

        for in_dial, out_resp in zip(batch_dials, result):
            if not isinstance(in_dial, list):
                in_dial = [in_dial]
            conversation_data = in_dial + [{"role": "bot", "content": out_resp}]
            infer_save_test_case(conversation_data, args.output_file)
    logger.info("The task is completed.")

    logger.info(f"{predictor.runtime_timer.log()}")
