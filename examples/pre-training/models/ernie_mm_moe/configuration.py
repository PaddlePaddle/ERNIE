# !/usr/bin/env python3
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
import copy
import logging

from models.ernie import ErnieConfig
from models.ernie_moe import ErnieMoEConfig
from models.image_encoder import EVAVisionTransformerConfig
from models.dfnrope.modeling import DFNRopeVisionTransformerConfig
from models.openclip.configuration import CLIPVisionConfig

logger = logging.getLogger(__name__)

__all__ = [
    "ErniemmMoEConfig",
]


class ErniemmMoEConfig(ErnieMoEConfig):
    r"""
    This is the configuration class to store the configuration of a [`~ErnieModel`]. It is used to instantiate an Ernie
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Ernie-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Ernie model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~ErnieModel`] or [`~TFErnieModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        Example:
    ```python
    >>> from paddleformers.transformer import ErnieModel, ErnieConfig

    >>> # Initializing a Ernie ernie-7b style configuration
    >>> configuration = ErnieConfig()

    >>> # Initializing a model from the ernie-7b style configuration
    >>> model = ErnieModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "erniemm"
    attribute_map = {
        "n_positions": "max_position_embeddings",
        "n_embd": "hidden_size",
        "n_layer": "num_hidden_layers",
        "n_head": "num_attention_heads",
        "n_inner": "intermediate_size",
        "activation_function": "hidden_act",
    }

    def __init__(
        self,
        vision_config=None,
        audio_config=None,
        inception_config=None,
        im_patch_id=None,  # [unused999]$
        use_eva_clip=False,
        use_recompute_attn_vision=False,
        pixel_hidden_size=None,  # for fuyu
        freeze="lm",  # 除了 lm-backbone 以外，额外的freeze参数
        use_normed_fc_resampler=False,
        resampler_reduce_token_by_reshape=False,
        pp_first_stage_layers=0,
        pp_recompute_offload_resampler=False,
        resamper_empty_cache=False,
        modality_detach=False,
        temporal_conv_size=1,
        spatial_conv_size=1,
        mm_vocab_size=0,  # vocab for mm specialtokens
        max_text_id=None,
        use_temporal_conv=True,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.use_eva_clip = use_eva_clip
        self.use_recompute_attn_vision = use_recompute_attn_vision
        self.im_patch_id = im_patch_id
        self.vision_config = vision_config
        self.audio_config = audio_config

        if isinstance(inception_config, dict):
            self.inception_config = ErnieConfig(**inception_config)
        else:
            self.inception_config = inception_config
        self.pixel_hidden_size = pixel_hidden_size
        self.freeze = freeze
        self.use_normed_fc_resampler = use_normed_fc_resampler
        self.resampler_reduce_token_by_reshape = resampler_reduce_token_by_reshape
        self.pp_first_stage_layers = pp_first_stage_layers
        self.pp_recompute_offload_resampler = pp_recompute_offload_resampler
        self.resamper_empty_cache = resamper_empty_cache
        self.modality_detach = modality_detach
        self.temporal_conv_size = temporal_conv_size
        self.spatial_conv_size = spatial_conv_size
        self.mm_vocab_size = mm_vocab_size
        self.max_text_id = max_text_id
        self.use_temporal_conv = use_temporal_conv

    def to_dict(self, saving_file=False):
        """to_dict"""
        output = copy.deepcopy(self.__dict__)
        if self.vision_config:
            output["vision_config"] = (
                self.vision_config.to_dict()
                if isinstance(
                    self.vision_config,
                    (
                        EVAVisionTransformerConfig,
                        DFNRopeVisionTransformerConfig,
                        CLIPVisionConfig,
                    ),
                )
                else self.vision_config
            )

        if self.inception_config:
            output["inception_config"] = (
                self.inception_config.to_dict()
                if isinstance(self.inception_config, ErnieConfig)
                else self.inception_config
            )
        output["model_type"] = self.__class__.model_type
        return output
