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
import unittest

from ernie.tokenizer import Ernie4_5_Tokenizer

from ernie.dataset.dpo import create_dataset as create_dataset_dpo
from ernie.dataset.finetuning import create_dataset as create_dataset_sft
from ernie.dataset.hf.hf_parser import create_dataset_from_file as create_dataset_hf


class SFTDatasetsTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.dataset_path = "./tests/fixtures"
        self.tokenizer_path = "./ERNIE-4.5-21B-A3B-Paddle-dummy-moe"
        self.tokenizer = Ernie4_5_Tokenizer.from_pretrained(self.tokenizer_path)

        self.dataset_config = {
            "tokenizer": self.tokenizer,
            "max_seq_len": 8192,
            "random_seed": 42,
            "num_replicas": 1,
            "rank": 0,
            "num_samples_each_epoch": 6000000,
            "random_shuffle": True,
            "greedy_intokens": True,
        }

    def tearDown(self) -> None:
        super().tearDown()

    def test_erniekit_sft_format(self):

        train_dataset = create_dataset_sft(
            task_group=os.path.join(self.dataset_path, "erniekit.json"),
            task_group_prob="1.0",
            sub_dataset_type="erniekit",
            **self.dataset_config,
        )

        items = next(iter(train_dataset))
        ex = items[0]
        assert ex.num_examples == 1
        assert hasattr(ex, "position_ids")
        assert hasattr(ex, "token_ids")
        assert hasattr(ex, "labels")
        assert hasattr(ex, "loss_mask")

    def test_openai_sft_format(self):

        train_dataset = create_dataset_sft(
            task_group=os.path.join(self.dataset_path, "openai.json"),
            task_group_prob="1.0",
            sub_dataset_type="erniekit",
            **self.dataset_config,
        )

        items = next(iter(train_dataset))
        ex = items[0]
        print(ex)
        assert ex.num_examples == 1
        assert hasattr(ex, "position_ids")
        assert hasattr(ex, "token_ids")
        assert hasattr(ex, "labels")
        assert hasattr(ex, "loss_mask")

    def test_hf_alpaca_formats(self):
        train_dataset = create_dataset_hf(
            file_path=os.path.join(self.dataset_path, "hf_alpaca.json"),
            formatting="alpaca",
            doc_formatting="json",
            process_fn=None,
            shuffle_file=True,
        )

        items = next(iter(train_dataset))
        ex = items
        assert "src" in ex
        assert "tgt" in ex


class DPODatasetsTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.dataset_path = "./tests/fixtures"
        self.tokenizer_path = "./ERNIE-4.5-21B-A3B-Paddle-dummy-moe"
        self.tokenizer = Ernie4_5_Tokenizer.from_pretrained(self.tokenizer_path)

        self.dataset_config = {
            "tokenizer": self.tokenizer,
            "max_seq_len": 16384,
            "max_prompt_len": 1024,
            "buffer_size": 500,
            "random_seed": 42,
            "num_replicas": 1,
            "rank": 0,
            "num_samples_each_epoch": 6000000,
            "random_shuffle": True,
            "greedy_intokens": True,
            "mask_out_eos_token": True,
        }

    def tearDown(self) -> None:
        super().tearDown()

    def test_erniekit_dpo_format(self):
        train_dataset = create_dataset_dpo(
            task_group=os.path.join(self.dataset_path, "erniekit_dpo.json"),
            task_group_prob="1.0",
            sub_dataset_type="erniekit",
            **self.dataset_config,
        )

        items = next(iter(train_dataset))
        ex = items[0]
        assert hasattr(ex, "position_ids")
        assert hasattr(ex, "input_ids")
        assert hasattr(ex, "attn_mask_start_row_indices")
        assert hasattr(ex, "chosen_labels")
        assert hasattr(ex, "chosen_labels")
        assert hasattr(ex, "response_index")

    def test_dataset_max_length(self):
        train_dataset = create_dataset_dpo(
            task_group=os.path.join(self.dataset_path, "erniekit_dpo_max_len.jsonl"),
            task_group_prob="1.0",
            sub_dataset_type="erniekit",
            **self.dataset_config,
        )

        items = next(iter(train_dataset))
        ex = items[0]
        assert hasattr(ex, "position_ids")
        assert hasattr(ex, "input_ids")
        assert hasattr(ex, "attn_mask_start_row_indices")
        assert hasattr(ex, "chosen_labels")
        assert hasattr(ex, "chosen_labels")
        assert hasattr(ex, "response_index")
