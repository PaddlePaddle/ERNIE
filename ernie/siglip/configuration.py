# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is based on https://github.com/Kwai-Keye/Keye/blob/main/keye-vl-8b-preview/configuration_keye.py
# Original header:
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

""" Siglip vision transformer configuration"""

from paddleformers.transformers.configuration_utils import PretrainedConfig

__all__ = [
    "PaddleOCRVisionConfig",
]


class PaddleOCRVisionConfig(PretrainedConfig):

    model_type = "paddleocr_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=14,
        hidden_act="gelu_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=2,
        recompute=False,
        recompute_granularity="full",
        recompute_use_reentrant=True,
        use_flash_attention=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        if hidden_act == "gelu_pytorch_tanh":
            hidden_act = "gelu_new"
        self.hidden_act = hidden_act
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.recompute = recompute
        self.recompute_granularity = recompute_granularity
        self.recompute_use_reentrant = recompute_use_reentrant
        self.use_flash_attention = use_flash_attention

        self.register_unsavable_keys(
            [
                "recompute",
                "recompute_use_reentrant",
                "recompute_granularity",
            ]
        )
