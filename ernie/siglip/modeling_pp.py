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

import math
import contextlib
from typing import List, Optional, Tuple, Union

import paddle
from paddle.distributed import fleet
from paddleformers.utils.log import logger

from .modeling import (
    SiglipVisionModel,
    BaseModelOutputWithPooling
)
from ..sequence_parallel_utils import (
    mark_as_sequence_parallel_parameter,
    SliceVarlenOp,
    AllGatherVarlenOpV2,
)


class SiglipVisionModelPipe(SiglipVisionModel):
    """
    Pipeline version of SiglipVisionModel with distributed computing support
    """

    def __init__(self, config, use_full_recompute=False):
        self.sorted_thw = None
        self.sorted_idx = None
        self.seq_list = None
        self.new_thw = []
        self.pp_data_balance = getattr(config.vision_config, "pp_data_balance", False)
        self.attn_sep = (
            getattr(config.vision_config, "attn_sep", False)
            and config.tensor_parallel_degree > 1
        )
        self.use_full_recompute = use_full_recompute
        if self.use_full_recompute:
            logger.info("use full recompute, vision model will NOT use recompute inner")
            config.vision_config.recompute = False
        super().__init__(config.vision_config)
        if self.config.tensor_parallel_degree > 1:
            logger.info(
                "use sp extract feature, vit parameter will be marked as sequence parallel"
            )
            for p in self.parameters():
                mark_as_sequence_parallel_parameter(p)

    
    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        attention_mask=None,
        sample_indices=None,
        image_indices=None,
        position_ids=None,
        height_position_ids=None,
        width_position_ids=None,
        cu_seqlens=None,
        padding_mask=None,
        vision_return_embed_list: Optional[bool] = False,
        image_grid_thw: Optional[
            List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]
        ] = None,
        return_pooler_output: Optional[bool] = True,
        use_rope: Optional[bool] = False,
        window_size: Optional[bool] = -1,
    ) -> BaseModelOutputWithPooling:
        ctx = (
            paddle.no_grad
            if getattr(self.config, "freeze_vision", False)
            else contextlib.nullcontext
        )
        with ctx():
            vision_outputs = super().forward(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                vision_return_embed_list=vision_return_embed_list,
                interpolate_pos_encoding=interpolate_pos_encoding,
                sample_indices=sample_indices,
                cu_seqlens=cu_seqlens,
                return_pooler_output=return_pooler_output,
                use_rope=use_rope,
                window_size=window_size,
            )
        return vision_outputs
