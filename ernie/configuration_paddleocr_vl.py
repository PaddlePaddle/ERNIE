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

from paddleformers.transformers.configuration_utils import PretrainedConfig
from .siglip.modeling import PaddleOCRVisionConfig


class PaddleOCRVLConfig(PretrainedConfig):
    model_type = "paddleocr_vl"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"vision_config": PaddleOCRVisionConfig}

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=11008,
        max_position_embeddings=32768,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_token_id=101304,
        video_token_id=101305,
        vision_start_token_id=101306,
        rope_scaling=None,
        rms_norm_eps=1e-6,
        use_cache=False,
        use_flash_attention=False,
        recompute=False,
        recompute_granularity="core_attn",
        recompute_use_reentrant=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        head_dim=128,
        hidden_act="silu",
        use_bias=False,
        rope_theta=10000,
        weight_share_add_bias=True,
        ignored_index=-100,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        compression_ratio: float = 1.0,
        num_key_value_heads=None,
        max_sequence_length=None,
        tie_word_embeddings=False,
        vision_config=None,
        **kwargs,
    ):
        # Set default for tied embeddings if not specified.
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.use_flash_attention = use_flash_attention
        self.recompute = recompute
        self.recompute_granularity = recompute_granularity
        self.recompute_use_reentrant = recompute_use_reentrant
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.head_dim = head_dim
        if hidden_act != "silu":
            raise NotImplementedError
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_share_add_bias = weight_share_add_bias
        self.rope_theta = rope_theta
        self.ignored_index = ignored_index
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.compression_ratio = compression_ratio
        self.num_key_value_heads = num_key_value_heads
        self.max_sequence_length = max_sequence_length
        self.rope_scaling = rope_scaling
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        # Currently, these configuration items are hard-coded
        self.fuse_rms_norm = True
        self.use_sparse_flash_attn = True
        self.use_var_len_flash_attn = False
        self.scale_qk_coeff = 1.0
        self.fuse_softmax_mask = False
        self.use_sparse_head_and_loss_fn = False
        self.use_recompute_loss_fn = False
        self.use_fused_head_and_loss_fn = False
        self.fuse_linear = False
        self.token_balance_seqlen = False
        self.use_rmsnorm = True
        self.fuse_ln = False
        self.cachekv_quant = False
        self.fuse_swiglu = False
        self.freq_allocation = 20

        self.register_unsavable_keys(
            [
                "recompute",
                "recompute_use_reentrant",
                "recompute_granularity",
                "use_recompute_loss_fn",
                "use_sparse_flash_attn",
                "use_var_len_flash_attn",
                "use_sparse_head_and_loss_fn",
                "fuse_softmax_mask",
                "cachekv_quant",
                "use_fused_head_and_loss_fn",
                "max_sequence_length",
            ]
        )

        def to_dict(self, saving_file=False):
            """to_dict"""

            # call PretrainedConfig.to_dict method to preprocess the output config, like removing unsavable keys
            output = super().to_dict(saving_file=saving_file)

            if self.vision_config:
                output["vision_config"] = (
                    self.vision_config.to_diff_dict()
                    if isinstance(self.vision_config, (PaddleOCRVisionConfig))
                    else self.vision_config
                )

            output["model_type"] = self.__class__.model_type
            return output
