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

""" OffloadCallback """

from paddleformers.trainer.trainer_callback import TrainerCallback

try:
    from paddle.incubate.tensor.manipulation import enable_activation_offload
except ImportError:
    enable_activation_offload = None


class OffloadCallback(TrainerCallback):
    """OffloadCallback"""

    def on_step_begin(self, args, state, control, **kwargs):
        """on_step_begin"""
        model = kwargs.pop("model")
        if enable_activation_offload:
            enable_activation_offload(
                model, retry_times=args.activation_offload_retry_times
            )
