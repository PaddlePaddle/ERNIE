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
read dataset json for vl_sft
"""

import gzip
import json
import logging
import random
import re
from collections import OrderedDict, namedtuple
from copy import deepcopy

import numpy as np
import paddle

DATATYPE_2_ID = {"mm": 0, "lm": 1, "audio": 2}

from contextlib import contextmanager

from paddle.io import IterableDataset

log = logging.getLogger(__name__)


def fetch_worker():
    """
    Retrieve information about the current worker processes.
    Args:
        None
    Returns:
        tuple: A tuple containing two elements, representing the
            total number of worker processes and the ID of the current worker process.

    """
    worker_info = paddle.io.get_worker_info()
    if worker_info is None:
        num_workers = 1
        worker_id = 0
    else:
        num_workers = worker_info.num_workers
        worker_id = worker_info.id
    return num_workers, worker_id


def make_seed(*args):
    """
    Generate a hash seed based on multiple parameters.
    Args:
        *args: Any number of parameters, which can be any hashable objects.
    Returns:
        int: A hash seed generated from the provided parameters.
    """
    seed = 0
    for arg in args:
        seed = hash(seed * 1e3 + hash(arg)) & 0x7FFFFFFF
    return seed


@contextmanager
def open_file(filename):
    """Construct a file handler that can read normal or gzip-compressed files.

    The handler automatically detects compression based on file extension.

    Args:
        filename (str): Path to the target file, which may end with .gz for gzip compression.

    Returns:
        Generator[TextIO]: A file object generator that yields lines from the file.
    """
    if filename.endswith(".gz"):
        fp = gzip.open(filename, "rt")
    else:
        fp = open(filename)
    yield fp
    fp.close()


def contains_markdown_table(text):
    """
    Determine whether the given text contains a Markdown-formatted table.

    Args:
        text (str): The text to be checked.

    Returns:
        bool: Returns True if the text contains a Markdown-formatted table; otherwise, returns False.

    """
    lines = text.strip().split('\n')
    for i in range(1, len(lines) - 1):
        separator_line = lines[i].strip()
        # Check if it's a table separator line
        if re.match(r'^(\s*\|?\s*:?-+:?\s*\|)+\s*$', separator_line):
            header_line = lines[i - 1].strip()
            body_line = lines[i + 1].strip()
            # Check if the previous and next lines could be content rows of a table
            if '|' in header_line and '|' in body_line:
                # Check if the number of | characters in each row is consistent
                header_pipes = header_line.count('|')
                body_pipes = body_line.count('|')
                separator_pipes = separator_line.count('|')
                if header_pipes == body_pipes == separator_pipes:
                    return True
    return False


def process_markdown_table(table: str) -> str:
    """
    Process a Markdown table, formatting the cells and alignment line.

    Args:
        table (str): The Markdown table string to be processed.

    Returns:
        str: The processed Markdown table string.

    Note:
        There may be some bugs.
    """
    # may some bug
    if not contains_markdown_table(table):
        return table

    # Remove extra spaces before and after each cell, 
    # and ensure at least one space is present before and after each cell
    def format_row(row: str) -> str:
        return '| ' + ' | '.join(cell.strip() for cell in row.split('|')[1:-1]) + ' |'

    # Limit characters above "---" to at most three '-', 
    # applicable for alignment lines
    def format_alignment_row(row: str) -> str:
        # maybe have some bugs
        line = '| ' + ' | '.join(re.sub(r'-{3,}', '---', cell.strip()) for cell in row.split('|')[1:-1]) + ' |'
        line = line.replace(':---:', ':-:')
        line = line.replace(':--:', ':-:')
        line = line.replace(':---', ':--')
        line = line.replace('---:', '--:')

        return line

    lines = table.splitlines()
    processed_lines = []

    for line in lines:
        if re.match(r'^\s*\|', line):  
            if '-' in line:  
                processed_lines.append(format_alignment_row(line))
            else:  
                processed_lines.append(format_row(line))
        else:
            processed_lines.append(line) 

    return '\n'.join(processed_lines)


Example = namedtuple("Example", ["meta", "task", "prompt", "labels", "src"])


class ExampleSet:
    """use to manage one json file_name data"""

    def __init__(
        self,
        file_name,
        src,
        prompt_list,
        shuffle_json: bool = False,
        process_fn=None,
    ):

        self._file_name = file_name
        self.src = int(src)
        self._process_fn = process_fn
        self._shuffle_json = shuffle_json
        self.prompt_list = prompt_list
        self.process_fn = process_fn
        log.info(f"loading json {self._file_name}")
        with open_file(self._file_name) as fin:
            self.exs = json.load(fin)

    def __len__(self):
        return len(self.exs)

    def __iter__(self):
        if self._shuffle_json:
            np.random.shuffle(self.exs)
        idx = 0
        for meta in self.exs:
            ret = Example(
                meta=meta,
                src=self.src,
                task="lm",
                prompt=None,
                labels=None,
            )

            ret = self.process_fn(ret)
            ret.update(data_id=idx, example_id=idx)
            idx += 1

            yield ret


class SFTMultimodalDatasetJson(IterableDataset):
    """
    SFT Multimodal Dataset Json
    """

    def __init__(
        self,
        train_dataset_path,
        train_dataset_prob,
        tokenizer,
        image_preprocess,
        seed,
        image_token_len,
        seqlen,
        special_token_loss_mask_ratio=None,
        use_prompt=False,
        adaptive_resolution=False,
        im_patch_id=10000000,
        use_crop=False,
        crop_tile_option="16,24",
        crop_tile_rate="0.95,0.05",
        dp_rank=None,
        dp_size=None,
        batch_size=1,
        data_processor=None,
        **kwargs,
    ):
        self.train_dataset_path = train_dataset_path
        self.train_dataset_prob = train_dataset_prob
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.image_token_len = image_token_len
        # self.im_patch_id = len(self.vocab)
        self.im_patch_id = im_patch_id
        # img_token = tokenizer.special_tokens_map.get("img_token", "<mask:1>")
        # self.im_patch_id= self.vocab[img_token]
        self.image_preprocess = image_preprocess
        self.adaptive_resolution = adaptive_resolution
        self.use_crop = use_crop
        if self.use_crop:  # 1
            self.crop_tile_option = [int(op.strip()) for op in crop_tile_option.strip().split(',')]
            self.crop_tile_rate = [float(op.strip()) for op in crop_tile_rate.strip().split(',')]
            assert len(self.crop_tile_option) == len(
                self.crop_tile_rate
            ), 'crop_tile_option and crop_tile_rate len are no match'
            log.info(f'[Use Crop] crop_tile_option={self.crop_tile_option} crop_tile_rate={self.crop_tile_rate}')

        self.wo_vit = False

        self.grouding_loss_mask_rng = random.Random(seed)
        self.special_token_loss_mask_ratio = special_token_loss_mask_ratio

        self.seed = seed
        self.seqlen = seqlen
        self.use_prompt = use_prompt

        self.sys_start_token = self.tokenizer.special_tokens_map.get("sys_start_token", "<mask:4>")
        self.sys_end_token = self.tokenizer.special_tokens_map.get("sys_end_token", "<mask:5>")
        self.eos_token = self.tokenizer.special_tokens_map.get("eos_token", "</s>")
        self.cls_token = self.tokenizer.special_tokens_map.get("cls_token", "<mask:0>")
        self.sep_token = self.tokenizer.special_tokens_map.get("sep_token", "<|endofprompt|>")
        log.info(f"cls token: {self.cls_token}, sep token: {self.sep_token}, eos token: {self.eos_token}")

        # "<|IMAGE_START|>", "<|IMAGE_END|>" "<|CROP_COL_SEP|>", "<|CROP_ROW_SEP|>"
        self.img_start_token = "<|IMAGE_START|>"
        self.img_end_token = "<|IMAGE_END|>"
        self.crop_col_sep_token = "<|CROP_COL_SEP|>"
        self.crop_row_sep_token = "<|CROP_ROW_SEP|>"
        log.info(
            f"""img_start_token: {self.img_start_token},
            img_end_token: {self.img_end_token},
            crop_col_sep_token: {self.crop_col_sep_token},
            crop_row_sep_token: {self.crop_row_sep_token}"""
        )

        self.cls_token_id = self.tokenizer._convert_token_to_id(self.cls_token)
        self.sep_token_id = self.tokenizer._convert_token_to_id(self.sep_token)
        self.eos_token_id = self.tokenizer._convert_token_to_id(self.eos_token)

        log.info(f"cls token: {self.cls_token_id}, sep token: {self.sep_token_id}, eos token: {self.eos_token_id}")

        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(self.img_start_token)
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(self.img_end_token)
        self.crop_col_sep_token_id = self.tokenizer.convert_tokens_to_ids(self.crop_col_sep_token)
        self.crop_row_sep_token_id = self.tokenizer.convert_tokens_to_ids(self.crop_row_sep_token)
        self.sys_start_id = self.tokenizer.convert_tokens_to_ids(self.sys_start_token)
        self.sys_end_id = self.tokenizer.convert_tokens_to_ids(self.sys_end_token)
        log.info(
            f"""img_start_token_id: {self.img_start_token_id},
            img_end_token_id: {self.img_end_token_id},
            crop_col_sep_token_id: {self.crop_col_sep_token_id},
            crop_row_sep_token: {self.crop_row_sep_token_id}"""
        )
        log.info(f"sys_start_id: {self.sys_start_id}, sys_end_id: {self.sys_end_id}")
        self.tokenizer.cls_token_id = self.cls_token_id
        self.tokenizer.sep_token_id = self.sep_token_id
        self.tokenizer.eos_token_id = self.eos_token_id

        self.data_processor = data_processor

        self.task_group = {}
        self.src_id_list = []
        self.weight_list = []
        self.batch_size = batch_size
        self.data_rank, self.data_size = dp_rank, dp_size
        self.num_workers, self.worker_id = fetch_worker()
        self.global_worker_id = self.data_rank * self.num_workers + self.worker_id
        self.local_seed = make_seed(self.seed, self.global_worker_id)
        self.length = 0
        self.epoch = 0

    def load_files_info(self, filelist):
        """
        Load file list information.

        Args:
            filelist (str): Path to the file list.

        Returns:
            list: A list of lines containing file information.

        Raises:
            None

        """
        with open_file(filelist) as f:
            lines = f.read().strip().split("\n")
        if len(lines) <= self.data_size:
            log.warning(
                f"""Expected filelist size >= data_size, but got {len(lines)} vs. {self.data_size},
                different nodes will be assigned the same data"""
            )
        return lines

    def reformat_meta(self, meta):
        """reformat_meta"""
        text_list = "".join([text["text"] for text in meta["text_info"] if text["tag"] == "no_mask"])
        if type(text_list) == str and "<think>" in text_list and "</think>" in text_list:
            # think-data
            pass
        else:
            meta["prefix"] = "<think>\n\n</think>\n\n"
        return meta

    def _load(self, shuffle_json=True):
        """load"""
        process_fn = self.example_to_feature_stage3

        train_dataset_path_list = self.train_dataset_path.split(",")
        train_dataset_prob_list = self.train_dataset_prob.split(",")

        for src_id, path, prob in zip(range(len(train_dataset_path_list)), train_dataset_path_list, train_dataset_prob_list):
            part = ExampleSet(
                file_name=path,
                src=src_id,
                prompt_list=None,
                shuffle_json=shuffle_json,
                process_fn=process_fn,
            )
            self.task_group[part.src] = part
            self.src_id_list.append(part.src)
            self.length += len(part)
            self.weight_list.append(float(prob))
        
        weight_sum = sum(self.weight_list)
        self.weight_list = [item / weight_sum for item in self.weight_list]

    def example_to_feature_stage3(self, example):
        """
        Convert an example to a feature representation (Stage 3).

        Args:
            example (dict): A dictionary containing metadata and raw data.

        Returns:
            OrderedDict: A dictionary containing the processed feature data.

        Raises:
            Exception: Thrown when an exception occurs during processing.

        """
        try:
            meta = example.meta
            raw_meta = deepcopy(meta)

            # Fix Table
            for idx, _ in enumerate(raw_meta["text_info"]):
                raw_meta["text_info"][idx]["text"] = raw_meta["text_info"][idx]["text"].strip()
                if raw_meta["text_info"][idx]["tag"] == "no_mask":
                    raw_text = raw_meta["text_info"][idx]["text"]
                    raw_meta = self.reformat_meta(raw_meta)
                    raw_meta["text_info"][idx]["text"] = process_markdown_table(raw_meta["text_info"][idx]["text"])


            result = self.data_processor.process(raw_meta)
            assert len(result) == 1, f"result: {len(result)}, {result}"
            result = result[0]
            assert len(result["images"]) > 0, "images is None"
            assert result["token_type_ids"] is not None, "token_type_ids is None"
            assert result["image_type_ids"] is not None, "image_type_ids is None"

            return OrderedDict(
                src_id=example.src,
                part_id=0,
                images=result["images"],
                input_ids=result["input_ids"],
                labels=result["labels"],
                token_type_ids=result["token_type_ids"],
                image_type_ids=result["image_type_ids"],
                grid_thw=result["grid_thw"],
                position_ids=result["position_ids"],
                data_not_valid=0,
                data_type=DATATYPE_2_ID["mm"],
            )
        except Exception as e:
            log.info(f"e: {e}")
            features = OrderedDict(
                src_id=example.src,
                part_id=0,
                images=None,
                input_ids=[1],
                labels=[1],
                grid_thw=np.array([[1, 2, 2]]),
                position_ids=np.array([[0, 0, 0]]),
                data_not_valid=1,  # indicate this is a invalid data
                data_type=DATATYPE_2_ID["mm"],
            )
            log.info(meta)
            log.info(f"****** Exception raised in dataset: {e} ******")
            log.exception(e)
            log.info("*********************************************************")
            return features

    def example_to_feature_stage3_video(self, example):
        """
        Deprecated: This function will be removed.
        """
        try:
            meta = json.loads(example.ids.tobytes().decode())
            raw_meta = deepcopy(meta)

            result = self.data_processor.process(raw_meta)
            assert len(result) == 1, f"result: {len(result)}, {result}"
            result = result[0]
            return OrderedDict(
                src_id=example.src,
                part_id=example.part,
                images=result["images"],
                input_ids=result["input_ids"],
                labels=result["labels"],
                token_type_ids=result["token_type_ids"],
                image_type_ids=result["image_type_ids"],
                data_not_valid=0,
                data_type=DATATYPE_2_ID["mm"],
            )
        except Exception as e:
            log.info(f"e: {e}")
            features = OrderedDict(
                src_id=example.src,
                part_id=example.part,
                images=None,
                input_ids=[1],
                labels=[1],
                data_not_valid=1,  # indicate this is a invalid data
                data_type=DATATYPE_2_ID["mm"],
            )
            log.info(meta)
            log.info(f"****** Exception raised in dataset: {e} ******")
            log.exception(e)
            log.info("*********************************************************")
            return features

    def __len__(self):
        return self.length

    def __iter__(self):
        while True:
            np.random.seed(make_seed(self.local_seed, self.epoch))
            sample_list = np.random.choice(self.src_id_list, size=5120, p=self.weight_list)
            self.epoch += 1
            for sample in sample_list:
                data = self.task_group[int(sample)]
                yield from data
