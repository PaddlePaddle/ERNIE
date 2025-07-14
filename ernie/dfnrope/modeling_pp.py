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

import math

import paddle
from paddleformers.utils.log import logger

from .modeling import DFNRopeVisionTransformerPretrainedModel
from ..sequence_parallel_utils import (
    mark_as_sequence_parallel_parameter,
    SliceVarlenOp,
    AllGatherVarlenOpV2,
)
from ..comm_utils import mp_slice


class DFNRopeVisionTransformerPipe(DFNRopeVisionTransformerPretrainedModel):
    """
    DFNRopeVisionTransformerPipe
    """

    def __init__(self, config, use_full_recompute=False):
        self.sorted_thw = None
        self.sorted_idx = None
        self.seq_list = None
        self.new_thw = []
        self.pp_data_balance = getattr(config.vision_config, "pp_data_balance", False)
        self.attn_sep = getattr(config.vision_config, "attn_sep", False) and config.tensor_parallel_degree > 1
        self.use_full_recompute = use_full_recompute
        if self.use_full_recompute:
            logger.info("use full recompute, vision model will NOT use recompute inner")
            config.vision_config.recompute = False
        super().__init__(config.vision_config)
        if self.config.tensor_parallel_degree > 1:
            logger.info("use sp extract feature, vit parameter will be marked as sequence parallel")
            for p in self.parameters():
                mark_as_sequence_parallel_parameter(p)

    def extract_feature(self, images, grid_thw, second_fwd=False):
        """extract feature"""
        if self.config.tensor_parallel_degree <= 1:
            return self._extract_feature(images, grid_thw)
        else:
            grid_thw = grid_thw.clone()
            # logger.info("use sp extract feature")
            images_indices = []
            parallelism = self.config.tensor_parallel_degree
            capacity = (grid_thw.prod(-1).sum(-1) + parallelism - 1) // parallelism
            crop_sizes = grid_thw.prod(-1)
            crop_offset = crop_sizes.cumsum(0)
            rank_per_crop = paddle.maximum((crop_offset - 1) // capacity, paddle.to_tensor(0))
            image_size_per_rank = paddle.zeros([parallelism], dtype="int64")
            num_crop_per_rank = paddle.bincount(rank_per_crop, minlength=parallelism)
            image_size_per_rank = paddle.scatter(image_size_per_rank, rank_per_crop, crop_sizes, overwrite=False)

            thw_indices = num_crop_per_rank
            images_indices = image_size_per_rank

            num_pad = 0
            if self.attn_sep:
                seqlen = images.shape[0]
                num_pad = math.ceil(seqlen / parallelism) * parallelism - seqlen
                images = paddle.nn.functional.pad(images, [0, num_pad, 0, 0], value=0)
                images_indices = [images.shape[0] // parallelism for _ in range(parallelism)]
                images = SliceVarlenOp.apply(images, images_indices)
            else:
                images = SliceVarlenOp.apply(images, images_indices)
                images = images.detach()
                grid_thw = mp_slice(grid_thw, thw_indices)

            if len(images):
                image_features = self._extract_feature(images, grid_thw, num_pad=num_pad)
            else:
                image_features = paddle.empty([0, self.config.hidden_size], dtype=self.patch_embed.proj.weight.dtype)
                image_features.stop_gradient = self.patch_embed.proj.weight.stop_gradient
            # sanity check
            if not second_fwd:
                image_features = AllGatherVarlenOpV2.apply(image_features, images_indices)
                if self.attn_sep:
                    image_features = image_features[:seqlen, :]
            # diff = (feas-image_features).abs().mean()
            # logger.info(f'shard vs not shard : {image_features.dtype} {image_features.stop_gradient} {diff}')
            if second_fwd:
                return image_features, images_indices
            return image_features

    def _extract_feature(self, images, grid_thw, num_pad=0):
        """extract feature"""
        ctx = paddle.no_grad if getattr(self.config, "freeze_vision", False) else contextlib.nullcontext
        with ctx():
            image_features = super().forward(images, grid_thw, num_pad)
        return image_features

    def forward(self, args):
        """_summary_

        Args:
            args (_type_): _description_

        Returns:
            _type_: _description_
        """
        raise NotImplementedError