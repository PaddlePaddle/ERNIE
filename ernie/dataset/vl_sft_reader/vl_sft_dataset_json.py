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

import hashlib
import pickle
import gzip
import os
import json
import logging
import random
import re
from collections import OrderedDict, namedtuple
from copy import deepcopy

import numpy as np
import paddle

from contextlib import contextmanager

from paddle.io import IterableDataset

log = logging.getLogger(__name__)

DATATYPE_2_ID = {"mm": 0, "lm": 1, "audio": 2}


@contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class LazyShard:
    def __init__(self, data, worker_id, num_workers):
        self.data = data
        self.worker_id = worker_id
        self.num_workers = num_workers

    def __iter__(self):
        return iter(self.data[self.worker_id :: self.num_workers])


def get_slice_len(full_len, start, interval):
    return (full_len - start - 1) // interval + 1


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
    lines = text.strip().split("\n")
    for i in range(1, len(lines) - 1):
        separator_line = lines[i].strip()
        # Check if it's a table separator line
        if re.match(r"^(\s*\|?\s*:?-+:?\s*\|)+\s*$", separator_line):
            header_line = lines[i - 1].strip()
            body_line = lines[i + 1].strip()
            # Check if the previous and next lines could be content rows of a table
            if "|" in header_line and "|" in body_line:
                # Check if the number of | characters in each row is consistent
                header_pipes = header_line.count("|")
                body_pipes = body_line.count("|")
                separator_pipes = separator_line.count("|")
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
        return "| " + " | ".join(cell.strip() for cell in row.split("|")[1:-1]) + " |"

    # Limit characters above "---" to at most three '-',
    # applicable for alignment lines
    def format_alignment_row(row: str) -> str:
        # maybe have some bugs
        line = (
            "| "
            + " | ".join(
                re.sub(r"-{3,}", "---", cell.strip()) for cell in row.split("|")[1:-1]
            )
            + " |"
        )
        line = line.replace(":---:", ":-:")
        line = line.replace(":--:", ":-:")
        line = line.replace(":---", ":--")
        line = line.replace("---:", "--:")

        return line

    lines = table.splitlines()
    processed_lines = []

    for line in lines:
        if re.match(r"^\s*\|", line):
            if "-" in line:
                processed_lines.append(format_alignment_row(line))
            else:
                processed_lines.append(format_row(line))
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)


Example = namedtuple("Example", ["meta", "task", "prompt", "labels", "src"])


def list_md5(lst):
    """计算列表的MD5哈希值"""
    # 序列化为字节
    bytes_data = pickle.dumps(lst)
    # 计算MD5
    return hashlib.md5(bytes_data).hexdigest()


class ExampleSet:
    """use to manage one json file_name data"""

    def __init__(
        self,
        args,
        file_name,
        src,
        prompt_list,
        shuffle_json: bool = False,
        process_fn=None,
    ):
        self.args = args
        self._file_name = file_name
        self.src = int(src)
        self._process_fn = process_fn
        self._shuffle_json = shuffle_json
        self.prompt_list = prompt_list
        self.process_fn = process_fn
        log.info(f"loading json {self._file_name}")
        if self._file_name.lower().endswith(".json"):
            with open_file(self._file_name) as fin:
                self.exs = json.load(fin)
        elif self._file_name.lower().endswith(".jsonl"):
            with open_file(self._file_name) as fin:
                self.exs = [json.loads(line) for line in fin]
        else:
            raise ValueError(f"Unsupported file type: {self._file_name}")
        new_exs = []
        for ex in self.exs:
            for key in ["image_info", "video_info"]:
                if key not in ex:
                    continue
                new_image_info = []
                for image_info in ex[key]:
                    url = image_info["image_url"]
                    if url.startswith("http") or os.path.isabs(url):
                        pass
                    else:
                        url = os.path.join(os.path.dirname(self._file_name), url)
                    image_info["image_url"] = url
                    new_image_info.append(image_info)
                ex[key] = new_image_info
            new_exs.append(ex)
        self.exs = new_exs

    def __len__(self):
        return len(self.exs)

    def __getitem__(self, idx):
        """
        Basic function of `MapDataset` to get sample from dataset with a given
        index.
        """
        ret = Example(
            meta=self.exs[idx],
            src=self.src,
            task="lm",
            prompt=None,
            labels=None,
        )
        ret = self.process_fn(ret)
        ret.update(data_id=idx, example_id=idx)
        return ret

    def __iter__(self):
        if self._shuffle_json:
            with temp_seed(97):
                if isinstance(self.exs, LazyShard):
                    print(f"{self.src} before shuffle: {list_md5(self.exs.data)}")
                    np.random.shuffle(self.exs.data)
                    print(f"{self.src} after shuffle: {list_md5(self.exs.data)}")
                else:
                    print(f"{self.src} before shuffle: {list_md5(self.exs)}")
                    np.random.shuffle(self.exs)
                    print(f"{self.src} after shuffle: {list_md5(self.exs)}")

        idx, cur = 0, 0
        for meta in self.exs:
            if cur % self.args.pp_need_data_degree == self.args.pipeline_parallel_rank:
                ret = Example(
                    meta=meta,
                    src=self.src,
                    task="lm",
                    prompt=None,
                    labels=None,
                )

                # (LiuTing) todo: can be optimized in pp data shard strategy.
                # import os
                # print(f"Ting: worker shard iter. PID: {os.getpid()}")
                # print(f"Ting: worker shard iter. cur: {cur}, ret: {ret}")
                ret = self.process_fn(ret)
                ret.update(data_id=idx, example_id=idx)
                idx += 1

                yield ret
            cur += 1


class SFTMultimodalDatasetJson(IterableDataset):
    """
    SFT Multimodal Dataset Json
    """

    def __init__(
        self,
        args,
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
        dp_rank=None,
        dp_size=None,
        batch_size=1,
        data_processor=None,
        **kwargs,
    ):
        self.args = args
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

        self.wo_vit = False

        self.grouding_loss_mask_rng = random.Random(seed)
        self.special_token_loss_mask_ratio = special_token_loss_mask_ratio

        self.seed = seed
        self.seqlen = seqlen
        self.use_prompt = use_prompt

        self.sys_start_token = self.tokenizer.special_tokens_map.get(
            "sys_start_token", "<mask:4>"
        )
        self.sys_end_token = self.tokenizer.special_tokens_map.get(
            "sys_end_token", "<mask:5>"
        )
        self.eos_token = self.tokenizer.special_tokens_map.get("eos_token", "</s>")
        self.cls_token = self.tokenizer.special_tokens_map.get("cls_token", "<mask:0>")
        self.sep_token = self.tokenizer.special_tokens_map.get(
            "sep_token", "<|endofprompt|>"
        )
        log.info(
            f"cls token: {self.cls_token}, sep token: {self.sep_token}, eos token: {self.eos_token}"
        )

        # "<|IMAGE_START|>", "<|IMAGE_END|>"
        self.img_start_token = "<|IMAGE_START|>"
        self.img_end_token = "<|IMAGE_END|>"
        log.info(
            f"""img_start_token: {self.img_start_token},
            img_end_token: {self.img_end_token}"""
        )

        self.cls_token_id = self.tokenizer._convert_token_to_id(self.cls_token)
        self.sep_token_id = self.tokenizer._convert_token_to_id(self.sep_token)
        self.eos_token_id = self.tokenizer._convert_token_to_id(self.eos_token)

        log.info(
            f"cls token: {self.cls_token_id}, sep token: {self.sep_token_id}, eos token: {self.eos_token_id}"
        )

        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(
            self.img_start_token
        )
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(self.img_end_token)
        self.sys_start_id = self.tokenizer.convert_tokens_to_ids(self.sys_start_token)
        self.sys_end_id = self.tokenizer.convert_tokens_to_ids(self.sys_end_token)
        log.info(
            f"""img_start_token_id: {self.img_start_token_id},
            img_end_token_id: {self.img_end_token_id}"""
        )
        log.info(f"sys_start_id: {self.sys_start_id}, sys_end_id: {self.sys_end_id}")
        self.tokenizer.cls_token_id = self.cls_token_id
        self.tokenizer.sep_token_id = self.sep_token_id
        self.tokenizer.eos_token_id = self.eos_token_id

        self.data_processor = data_processor

        self.task_group = {}
        self.task_group_iter = {}
        self.lengths = {}
        self.src_id_list = []
        self.weight_list = []
        self.batch_size = batch_size
        self.data_rank, self.data_size = dp_rank, dp_size
        self.num_workers, self.worker_id = fetch_worker()
        # self.global_worker_id = self.data_rank * self.num_workers + self.worker_id
        # self.local_seed = make_seed(self.seed, self.global_worker_id)
        self.local_seed = self.seed
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
        text_list = "".join(
            [text["text"] for text in meta["text_info"] if text["tag"] == "no_mask"]
        )
        if (
            isinstance(text_list, str)
            and "<think>" in text_list
            and "</think>" in text_list
        ):
            # think-data
            pass
        else:
            meta["prefix"] = "<think>\n\n</think>\n\n"
        return meta

    def _load(self, shuffle_json=True):
        """load"""
        process_fn = self.example_to_feature_stage3

        train_dataset_path_list = (
            str(self.train_dataset_path).replace(" ", "").split(",")
        )
        train_dataset_prob_list = (
            str(self.train_dataset_prob).replace(" ", "").split(",")
        )

        for src_id, path, prob in zip(
            range(len(train_dataset_path_list)),
            train_dataset_path_list,
            train_dataset_prob_list,
        ):
            part = ExampleSet(
                args=self.args,
                file_name=path,
                src=src_id,
                prompt_list=None,
                shuffle_json=shuffle_json,
                process_fn=process_fn,
            )

            self.task_group[part.src] = part
            self.src_id_list.append(part.src)
            self.length += len(part)
            self.lengths[part.src] = len(part)
            self.weight_list.append(float(prob))
            # self.weight_list.append(len(part) * float(prob))

        weight_sum = sum(self.weight_list)
        self.weight_list = [item / weight_sum for item in self.weight_list]

    def _worker_shard(self):

        self.num_workers, self.worker_id = fetch_worker()
        # print("Ting: ", self.num_workers)
        if self.num_workers > 1:
            self.length = 0
            self.lengths = {}
            for src in self.task_group:
                # print(f"Ting: {src} worker shard before {len(self.task_group[src])} PID: {os.getpid()}, TID: {threading.current_thread().ident}")
                # self.task_group[src].exs = self.task_group[src].exs[self.worker_id :: self.num_workers]
                self.length += get_slice_len(
                    len(self.task_group[src]), self.worker_id, self.num_workers
                )
                self.lengths[src] = get_slice_len(
                    len(self.task_group[src]), self.worker_id, self.num_workers
                )
                self.task_group[src].exs = LazyShard(
                    self.task_group[src].exs, self.worker_id, self.num_workers
                )
                # print(f"Ting: {src} worker shard after {len(self.task_group[src])} PID: {os.getpid()}, TID: {threading.current_thread().ident}")

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
                raw_meta["text_info"][idx]["text"] = raw_meta["text_info"][idx][
                    "text"
                ].strip()
                if raw_meta["text_info"][idx]["tag"] == "no_mask":
                    raw_meta = self.reformat_meta(raw_meta)
                    raw_meta["text_info"][idx]["text"] = process_markdown_table(
                        raw_meta["text_info"][idx]["text"]
                    )

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
        # return self.length
        return 6000000000

    def __getitem__(self, idx):
        self.num_workers, self.worker_id = fetch_worker()
        self.global_worker_id = (
            self.args.pipeline_parallel_rank * self.num_workers + self.worker_id
        )
        np.random.seed(make_seed(self.local_seed, self.epoch))
        with temp_seed(make_seed(self.seed, self.global_worker_id)):
            sample_list = np.random.choice(self.src_id_list, size=1, p=self.weight_list)
        print(f" {os.getpid()}sample_list: {sample_list}")

        self.epoch += 1
        for sample in sample_list:
            full_len = self.lengths[int(sample)]
            each_len = full_len // self.args.pp_need_data_degree
            print(
                f" {os.getpid()}idx: {idx}-{self.args.pipeline_parallel_rank * each_len + (idx % full_len) % each_len}"
            )
            return self.task_group[int(sample)][(idx % full_len) % each_len]

    def gen_sample_list(self):
        indices = []
        for i, _ in enumerate(self.task_group):
            sample_size = int(self.weight_list[i] * self.length)
            indices.extend([i] * sample_size)
        return indices

    def __iter__(self):

        self._worker_shard()
        self.global_worker_id = (
            self.args.pipeline_parallel_rank * self.num_workers + self.worker_id
        )
        while True:
            with temp_seed(make_seed(87, self.global_worker_id)):
                sample_list = self.gen_sample_list()
                np.random.shuffle(sample_list)

            self.epoch += 1
            for src_id in sample_list:
                src_id = int(src_id)
                if src_id not in self.task_group_iter:
                    self.task_group_iter[src_id] = iter(self.task_group[src_id])
                try:
                    yield next(self.task_group_iter[src_id])
                except StopIteration:
                    print(f"{src_id} source epoch switching.")
                    self.task_group_iter[src_id] = iter(self.task_group[src_id])
                    yield next(self.task_group_iter[src_id])
