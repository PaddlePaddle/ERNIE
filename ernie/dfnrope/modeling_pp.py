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

from paddleformers.utils.log import logger

from .modeling import DFNRopeVisionTransformerPretrainedModel


class DFNRopeVisionTransformerPipe(DFNRopeVisionTransformerPretrainedModel):
    """
    VisionModelPipe completes the calculation of the entire ViT in a single Pipe.
    It can be used as a parallel layer of the pipeline, or as an independent feature extraction module
    attached to the PipelineModel (via the `add_vision_model` method)
    """

    def __init__(self, config):
        logger.info(f"VISION-CONFIG-{config.vision_config}")
        super().__init__(config.vision_config)
        self.config.freeze_vision = getattr(config, "freeze_vision", False)

    def forward(self, hidden_states, grid_thw):
        """Pipeline entrance"""
        image_features = super().forward(hidden_states, grid_thw)
        return image_features

    def extract_feature(self, hidden_states, grid_thw):
        return self.forward(hidden_states, grid_thw)
