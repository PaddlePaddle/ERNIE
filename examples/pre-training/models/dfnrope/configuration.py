# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
""" Ernie model configuration"""
import logging

from paddleformers.transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

__all__ = [
    "DFNRopeVisionTransformerConfig",
]


class DFNRopeVisionTransformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~ErnieModel`]. It is used to instantiate an Ernie
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Ernie-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    model_type = "DFNRope_vision_transformer"

    def __init__(
        self,
        depth=32,
        embed_dim=1280,
        hidden_size=3584,
        hidden_act="quick_gelu",
        mlp_ratio=4,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        attn_implementation="eager",  # new added
        pp_data_balance=False,
        use_recompute=False,
        attn_sep=False,
        vit_first_fwd_bsz=128,
        vit_num_recompute_layers=10000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.attn_implementation = attn_implementation
        self.pp_data_balance = pp_data_balance
        self.use_recompute = use_recompute
        self.attn_sep = attn_sep
        self.vit_first_fwd_bsz = vit_first_fwd_bsz
        self.vit_num_recompute_layers = vit_num_recompute_layers

        if "vision_config" in kwargs:
            for k in kwargs["vision_config"]:
                if hasattr(self, k):
                    setattr(self, k, kwargs["vision_config"][k])

    # def to_dict(self, saving_file):
    #     """_summary_

    #     Returns:
    #         _type_: _description_
    #     """
    #     output = copy.deepcopy(self.__dict__)
    #     return output
