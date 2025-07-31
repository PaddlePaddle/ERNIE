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

import logging

from .modeling import DFNRopeVisionTransformerPretrainedModel

logger = logging.getLogger(__name__)


class DFNRopeVisionTransformerPipe(DFNRopeVisionTransformerPretrainedModel):
    """
    VisionModelPipe 在单个 Pipe 中完成整个 Vit 的计算
    即可以作为 流水线并行的一层，
    也可以作为独立的 feature 抽取模块附加在 PipelineModel 中(通过 `add_vision_model` 方法)
    """

    def __init__(self, config):
        logger.info(f"VISION-CONFIG-{config.vision_config}")
        super().__init__(config.vision_config)
        self.config.freeze_vision = getattr(config, "freeze_vision", False)

    # def extract_feature(self, images):
    #     """
    #     Args: images: Tensor[B,H,W,C]
    #     Returns: image_feautres: Tensor[B,Hh*Hw,D]
    #     """
    #     # logger.info(f"INPUT_IMAGES-{images.shape}")
    #     # logger.info(f"md5 of image: {md5(images)}")
    #     ctx = paddle.no_grad if getattr(self.config, "freeze_vision", False) else contextlib.nullcontext
    #     with ctx():
    #         _, _, image_features, _ = super().forward(
    #             images,
    #             return_all_features=True,
    #         )
    #     # logger.info(f"md5 of imagefea: {md5(image_features)}")
    #     return image_features

    def forward(self, args):
        """Pipeline 模型入口"""
        assert (
            len(args) == 2
        ), f"The number of arguments must be (`hidden_states`、`grid_thw`) but got {len(args)}-{args}"
        hidden_states, grid_thw = args
        image_features = super().forward(hidden_states, grid_thw)
        return image_features

    def extract_feature(self, args):
        return self.forward(args)
