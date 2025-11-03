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


import paddle
import paddle.distributed as dist
from models.moe_layer import MOELayer
from paddle.distributed.fleet import fleet
from paddleformers.trainer.trainer_callback import TrainerCallback


class MoECorrectionBiasAdjustCallback(TrainerCallback):
    def __init__(self, lr, use_sp, model_class):
        super().__init__()
        self.update_lr = float(lr)
        self.use_sp = use_sp
        self.model_class = model_class

    def on_optimizer_end(self, args, state, control, **kwargs):
        model = kwargs["model"]

        usages = {}
        biases = {}

        def get_stat(layer):
            nonlocal usages, biases
            if isinstance(layer, self.model_class):
                if not isinstance(layer.mlp, (MOELayer)):
                    return
                assert hasattr(
                    layer.mlp, "moe_statics"
                ), "make sure update to latest ernie-core, too use AuxFree Balance"
                usages[layer.layer_idx] = layer.mlp.moe_statics.expert_usage
                biases[layer.layer_idx] = layer.mlp.moe_statics.e_score_correction_bias

        model.apply(get_stat)
        if not usages:
            return
        keys, tensor_list = zip(*sorted(usages.items(), key=lambda x: x[0]))
        usages_tensor = paddle.stack(tensor_list, 0)
        if not hasattr(fleet, "_hcg"):
            dist.all_reduce(usages_tensor)
            return

        hcg = fleet.get_hybrid_communicate_group()
        mp_group = hcg.get_model_parallel_group()
        dp_group = hcg.get_data_parallel_group()
        sd_group = hcg.get_sharding_parallel_group()
        if usages_tensor._local_value() is not None:
            if self.use_sp and mp_group.nranks > 1:
                dist.all_reduce(usages_tensor._local_value(), group=mp_group)
            if dp_group.nranks > 1:
                dist.all_reduce(usages_tensor._local_value(), group=dp_group)
            if sd_group.nranks > 1:
                dist.all_reduce(usages_tensor._local_value(), group=sd_group)
            usages_mean = usages_tensor.mean(-1, keepdim=True)
            update = paddle.sign(usages_mean - usages_tensor) * self.update_lr
            update_dict = dict(zip(keys, update))

            def update_bias(layer):
                nonlocal usages, biases
                if isinstance(layer, self.model_class):
                    if not isinstance(layer.mlp, MOELayer):
                        return
                    with paddle.no_grad():
                        if layer.mlp.gate.weight.stop_gradient:
                            update_dict[layer.layer_idx][0, :] = 0
                        biases[layer.layer_idx].add_(
                            update_dict[layer.layer_idx].flatten()
                        )
                        usages[layer.layer_idx].data.zero_()

            model.apply(update_bias)
