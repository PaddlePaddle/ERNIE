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

"""predict_generation under dynamic graph"""

import argparse
import gc
import json
import os
import struct

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddleformers.trainer import RuntimeTimer
from paddleformers.utils.log import logger
from tqdm import tqdm

from ernie.configuration import Ernie4_5_MoeConfig
from ernie.dataset.data_utils import convert_to_input_ids
from ernie.modeling_moe import Ernie4_5_MoeForCausalLM
from ernie.tokenizer import Ernie4_5_Tokenizer
from ernie.utils.common_utils import infer_save_test_case
from ernie.utils.download_utils import check_download_repo


def deserialize_from_file(fp):
    """
    Deserialize data from a file pointer based on the first byte indicating the data type.

    Args:
        fp (file): File pointer to the data file.

    Returns:
        numpy.ndarray: Deserialized data as a NumPy array.

    Raises:
        TypeError: If the first byte of the file does not match any known data type indicator.
    """
    x_type = fp.read(1)
    x_type_out = struct.unpack("c", x_type)[0]
    # data
    data_list = []
    if x_type_out == b"0":
        data = fp.read(4)
        data_out = struct.unpack("f", data)[0]
        while data:
            data_out = struct.unpack("f", data)[0]
            data_list.append(data_out)
            data = fp.read(4)
    elif x_type_out == b"1":
        data = fp.read(8)
        while data:
            data_out = struct.unpack("l", data)[0]
            data_list.append(data_out)
            data = fp.read(8)
    elif x_type_out == b"2":
        data = fp.read(4)
        while data:
            data_out = struct.unpack("i", data)[0]
            data_list.append(data_out)
            data = fp.read(4)
    else:
        print("type error")
    data_arr = np.array(data_list)
    return data_arr


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
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=8192,
        help="The maximum length of input + output.",
    )
    parser.add_argument("--min_dec_len", type=int, default=1)
    parser.add_argument(
        "--max_dec_len",
        type=int,
        default=2048,
        help="The maximum length of output.",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="chat",
        choices=["base", "chat"],
        help="The data format.",
    )
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--weight_quantize_algo",
        type=str,
        default=None,
        help="weight_only_int8",
    )
    parser.add_argument("--from_hf_hub", type=bool, default=False)
    parser.add_argument("--from_aistudio", type=bool, default=False)
    parser.add_argument("--from_modelscope", type=bool, default=False)
    parser.add_argument(
        "--input_file", type=str, default="./examples/inference/data/query-demo.jsonl"
    )
    parser.add_argument("--output_file", type=str, default="predict.json")
    parser.add_argument("--save_output_file_flush", type=int, default=10)
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


class Predictor:
    """
    Predictor
    """

    def __init__(self, args, tokenizer=None, model=None, **kwargs):
        """
        Initialize predictor with distributed setup and model loading.

        Args:
            args (Namespace): Configuration arguments
            tokenizer (Optional): Pre-initialized tokenizer
            model (Optional): Pre-initialized model
            kwargs: Additional model initialization parameters
        """
        args.model_name_or_path = check_download_repo(
            args.model_name_or_path,
            from_hf_hub=args.from_hf_hub,
            from_aistudio=args.from_aistudio,
            from_modelscope=args.from_modelscope,
        )

        if getattr(args, "from_modelscope", False):
            os.environ["from_modelscope"] = "True"

        self.runtime_timer = RuntimeTimer("Predictor")
        self.num_input_tokens = 0
        self.num_output_tokens = 0
        self.args = args

        # init distributed env
        self.tensor_parallel_degree = dist.get_world_size()
        self.tensor_parallel_rank = dist.get_rank()
        if dist.get_world_size() > 1:
            strategy = fleet.DistributedStrategy()
            strategy.hybrid_configs = {
                "dp_degree": 1,
                "mp_degree": self.tensor_parallel_degree,
                "pp_degree": 1,
                "sharding_degree": 1,
            }
            fleet.init(is_collective=True, strategy=strategy)
            hcg = fleet.get_hybrid_communicate_group()
            self.tensor_parallel_rank = hcg.get_model_parallel_rank()

        # init model & tokenizer
        self.tokenizer = Ernie4_5_Tokenizer.from_pretrained(
            args.model_name_or_path,
            from_hf_hub=args.from_hf_hub,
            from_aistudio=args.from_aistudio,
            convert_from_torch=False,
        )
        self.tokenizer.padding_side = "left"
        paddle.set_default_dtype(self.args.dtype)
        self.config = Ernie4_5_MoeConfig.from_pretrained(
            args.model_name_or_path,
            quantization_config=dict(
                weight_quantize_algo=args.weight_quantize_algo,
                ignore_modules=[".*out_linear.*"],
            ),
            dtype=self.args.dtype,
            fused_mt=False,
            tensor_parallel_output=False,
            sequence_parallel=False,
            use_sparse_head_and_loss_fn=False,
            use_fused_head_and_loss_fn=False,
            fuse_linear=False,
            recompute=False,
            tensor_parallel_degree=self.tensor_parallel_degree,
            tensor_parallel_rank=self.tensor_parallel_rank,
            use_flash_attention=True,
            moe_group="dummy",
            num_nextn_predict_layers=0,
            from_hf_hub=args.from_hf_hub,
            from_aistudio=args.from_aistudio,
            convert_from_torch=False,
        )
        self.model = Ernie4_5_MoeForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=self.config,
            from_hf_hub=args.from_hf_hub,
            from_aistudio=args.from_aistudio,
            convert_from_torch=False,
        )
        gc.collect()
        paddle.device.cuda.empty_cache()
        self.model.eval()

    def preprocess(self, dials: list[list[dict]]):
        """
        Preprocess dialogue inputs into model-ready tensors.

        Args:
            dials: List of dialogue sessions

        Returns:
            Dictionary containing:
            - input_ids: Padded token IDs tensor (shape: [batch_size, seq_len])
            - position_ids: Corresponding position IDs tensor
        """
        input_ids, num_input_tokens = convert_to_input_ids(
            dials,
            self.tokenizer,
            self.args.data_format,
            self.args.max_seq_len - self.args.max_dec_len,
        )
        self.num_input_tokens += num_input_tokens

        max_len = max(len(item) for item in input_ids)
        inputs = {}

        inputs["input_ids"] = []
        inputs["position_ids"] = []
        for item in input_ids:
            cur_len = len(item)
            inputs["input_ids"].append(
                [self.tokenizer.pad_token_id] * (max_len - cur_len) + item
            )
            inputs["position_ids"].append(
                [0] * (max_len - cur_len) + list(range(cur_len))
            )
        inputs["input_ids"] = paddle.to_tensor(
            np.array(inputs["input_ids"], dtype="int64")
        )
        inputs["position_ids"] = paddle.to_tensor(
            np.array(inputs["position_ids"], dtype="int64")
        )
        return inputs

    def postprocess(self, infer_data):
        """
        Post-processes inference data by decoding tokens and cleaning results.

        Args:
            infer_data (paddle.Tensor/np.ndarray): Raw inference output containing token IDs.
                Expected shape: (batch_size, sequence_length) or similar tensor/array format.

        Returns:
            dict: Dictionary containing processed results with key "result",
                where value is a list of cleaned decoded strings.
        """
        result = []
        for x in infer_data.tolist():
            res = self.tokenizer.decode(x, skip_special_tokens=True)
            res = res.strip("\n")
            result.append(res)
        out_dict = {"result": result}
        return out_dict

    def infer(self, inputs: dict) -> list[list[int]]:
        """
        Perform the prediction process of the model, where the input is a dictionary-type \
        object containing the input data required by the model.

        Args:
            inputs (dict): Contains the input data required by the model, including both \
                mandatory and optional items. For details, please refer to the model's documentation.

        Returns:
            list[list[int]]: A list-type object where each element is a list representing \
                one or more results generated during each prediction process.
        """

        outputs = self.model.generate(
            **inputs,
            decode_strategy="sampling",
            top_p=self.args.top_p,
            temperature=self.args.temperature,
            max_new_tokens=self.args.max_dec_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=[
                [self.tokenizer.eos_token_id],
                [self.tokenizer.cls_token_id],
            ],
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )[0]

        return outputs

    @paddle.no_grad()
    def predict(self, batch_dials: list[list[dict]]):
        """
        Perform batch inference on dialogue inputs.

        Args:
            batch_dials: Batch of dialogue sessions
        Returns:
            Dictionary containing processed prediction results with structure:
        """
        input_map = self.preprocess(batch_dials)
        infer_result = self.infer(input_map)
        self.num_output_tokens += (
            (
                (infer_result != self.tokenizer.eos_token_id)
                & (infer_result != self.tokenizer.cls_token_id)
            )
            .sum()
            .item()
        )
        output = self.postprocess(infer_result)
        return output


def main():
    """
    Main execution function for model inference pipeline.

    Args:
        None

    Returns:
        None

    """
    args = setup_args()

    #  Only support batch_size=1
    if args.batch_size > 1:
        print("`batch_size` is reset to 1. Only support batch_size=1.")
        args.batch_size = 1

    # Create Predictor
    predictor = Predictor(args)

    # Inference
    infer_dials: list[list[dict]] = []
    if args.input_file is None or not os.path.exists(args.input_file):
        infer_dials = [
            [
                {"role": "user", "content": "北京天安门广场在哪里"},
            ]
        ]
    else:
        with open(args.input_file, "r") as fin:
            for i, line in enumerate(fin):
                cur_line = json.loads(line)
                infer_dials.append(cur_line)

    test_case = []

    predictor.runtime_timer.start("predict stage running time")
    try:
        for idx in tqdm(range(0, len(infer_dials), args.batch_size)):
            batch_dials = infer_dials[idx : idx + args.batch_size]
            print("inputs ->", batch_dials)
            result = predictor.predict(batch_dials)
            print("result ->", result)

            for in_dial, out_resp in zip(batch_dials, result["result"]):
                if not isinstance(in_dial, list):
                    in_dial = []
                conversation_data = in_dial + [{"role": "bot", "content": out_resp}]
                test_case.append(conversation_data)

            if (
                args.save_output_file_flush > 0
                and idx % args.save_output_file_flush == 0
                and idx > 0
            ):
                if paddle.distributed.get_rank() == 0:
                    infer_save_test_case(
                        test_case[idx - args.save_output_file_flush : idx],
                        args.output_file,
                    )
        logger.info(
            f"The task is completed. Total input token: {predictor.num_input_tokens}. \
            Total output token: {predictor.num_output_tokens}"
        )
    except BaseException as e:
        logger.error(e)
        logger.info(
            f"The task is partially successful. Total success input token: \
            {predictor.num_input_tokens}. Total success output token: {predictor.num_output_tokens}"
        )
        logger.info(
            f"The task is stopped/aborted on {len(test_case)} / {len(infer_dials)}. \
            Please download the completion part ({len(test_case)} / {len(infer_dials)}) from \
            {args.output_file} and review it. "
        )

    logger.info(f"{predictor.runtime_timer.log()}")

    if paddle.distributed.get_rank() == 0:
        if args.save_output_file_flush == 0:
            infer_save_test_case(test_case, args.output_file)
        else:
            write_case_idx = (
                len(test_case)
                // args.save_output_file_flush
                * args.save_output_file_flush
            )
            if len(test_case) % args.save_output_file_flush == 0:
                write_case_idx -= args.save_output_file_flush
            infer_save_test_case(test_case[write_case_idx:], args.output_file)


if __name__ == "__main__":
    """
    main
    """
    main()
