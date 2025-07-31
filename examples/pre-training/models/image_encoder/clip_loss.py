# !/usr/bin/env python3

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

"""
from paddlemix
"""

import logging
import paddle
import paddle.nn as nn
from paddle import distributed as dist
from paddle.nn import functional as F

from models.sequence_parallel_utils import AllGatherVarlenOp

logger = logging.getLogger(__name__)


__all__ = ["ClipLoss"]


class AllGather(paddle.autograd.PyLayer):
    """An autograd function that performs allgather on a tensor.
    Performs all_gather operation on the provided tensors.
    *** Warning ***: paddle.distributed.all_gather has no gradient.
    """

    @staticmethod
    def forward(ctx, tensor, group=None):
        """_summary_

        Args:
            ctx (_type_): _description_
            tensor (_type_): _description_
            group (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if group is None:
            world_size = dist.get_world_size()
        else:
            world_size = group.world_size
        tensors_gather = [paddle.empty_like(x=tensor) for _ in range(world_size)]
        paddle.distributed.all_gather(tensors_gather, tensor, group=group)
        ctx.group = group
        ctx.batch_size = tensor.shape[0]
        return paddle.concat(x=tensors_gather, axis=0)

    @staticmethod
    def backward(ctx, grad_output):
        """_summary_

        Args:
            ctx (_type_): _description_
            grad_output (_type_): _description_

        Returns:
            _type_: _description_
        """
        num_or_sections = grad_output.shape[0] // ctx.batch_size
        grad_lst = paddle.split(grad_output, num_or_sections=num_or_sections)
        grad = paddle.zeros_like(x=grad_lst[0])
        dist.reduce_scatter(grad, grad_lst, group=ctx.group)
        return grad


def gather_features_cat_group(
    image_features, text_features, group, gather_with_grad=False
):
    """
    group=None 时，进行全局通信。
    """
    if group is not None and group.world_size <= 1:
        return image_features, text_features
    image_features = AllGatherVarlenOp.apply(image_features, group=group)
    text_features = AllGatherVarlenOp.apply(text_features, group=group)
    return image_features, text_features


def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
):
    """
    收集本地和全局的image_features和text_features

    Args:
        image_features (paddle.Tensor): 存储图像特征的张量，shape为[B, ...], 其中B表示batch size，...表示特征维度。
        text_features (paddle.Tensor): 存储文本特征的张量，shape为[B, ...], 其中B表示batch size，...表示特征维度。
        local_loss (bool, optional): 是否使用本地损失函数，如果为True，则使用本地模型计算特征，否则使用全局模型计算特征。默认为False。
        gather_with_grad (bool, optional): 是否收集具有梯度的数据，如果为True，则将具有梯度的数据从其他进程收集回来。默认为False。
        rank (int, optional): 当前进程的rank编号（从0开始编号）。默认为0。
        world_size (int, optional): 进程总数。默认为1。
        use_horovod (bool, optional): 是否使用Horovod进行分布式训练。默认为False。

    Returns:
        Tuple[paddle.Tensor, paddle.Tensor]
    """
    if hasattr(paddle.distributed.fleet.fleet, "_hcg"):
        hcg = paddle.distributed.fleet.get_hybrid_communicate_group()
        shardinggroup = hcg.get_sharding_parallel_group()
        dpgroup = hcg.get_data_parallel_group()
    else:
        shardinggroup, dpgroup = None, None  # none means world

    # feature_sizes = paddle.to_tensor([len(image_features)], dtype=paddle.int64)
    if shardinggroup is not None and shardinggroup.nranks > 1:
        image_features, text_features = gather_features_cat_group(
            image_features, text_features, shardinggroup, gather_with_grad
        )
        # feature_sizes = all_gather(feature_sizes, shardinggroup)
    if dpgroup is None or dpgroup.nranks > 1:  # dp=none 时一定要执行 world-broadcast
        image_features, text_features = gather_features_cat_group(
            image_features, text_features, dpgroup, gather_with_grad
        )
        # feature_sizes = all_gather(feature_sizes, dpgroup)
    all_image_features = image_features
    all_text_features = text_features

    return all_image_features, all_text_features  # , feature_sizes


def gather_and_cat(feature, group):
    """_summary_

    Args:
        feature (_type_): _description_
        group (_type_): _description_

    Returns:
        _type_: _description_
    """
    if group.nranks <= 1:
        return feature
    gathered_features = []
    dist.all_gather(gathered_features, feature, group=group)
    gathered_features = paddle.concat(gathered_features, axis=0)
    return gathered_features


class ClipLoss(nn.Layer):
    """
    dummy
    """

    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        visual_loss=True,
        text_loss=False,
        rank=0,
        world_size=1,
    ):
        """
        dummy
        """
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels  # useless
        self.rank = rank
        self.world_size = world_size
        self.visual_loss = visual_loss
        self.text_loss = text_loss
        self.temp_gathered_features = None

    def forward(self, preds):
        """
        dummy
        """
        (
            image_features,
            text_features,
            logit_scale,
            images_invalid_accumulated,
            text_invalid_accumulated,
            idx,
            grad_part_size,
        ) = preds
        if self.world_size > 1:
            # logger.info(
            #     f"image_features >>> "
            #     f"image:{image_features.shape},{image_features.dtype} "
            #     f"@ text:{text_features.shape},{text_features.dtype}, "
            # )
            if idx == 0:
                # 第一手 进行features all gather
                self.temp_gathered_features = None
                all_image_features, all_text_features = gather_features(
                    image_features,
                    text_features,
                    self.local_loss,
                    self.gather_with_grad,
                    self.rank,
                    self.world_size,
                )
                self.temp_gathered_features = (all_image_features, all_text_features)
            else:
                # 后续idx 只对当前更新部分 进行all gather
                all_image_features, all_text_features = self.temp_gathered_features
                all_image_features = all_image_features.detach()
                all_text_features = all_text_features.detach()
                grad_image_features, grad_text_features = gather_features(
                    image_features[idx * grad_part_size : (idx + 1) * grad_part_size],
                    text_features[idx * grad_part_size : (idx + 1) * grad_part_size],
                    self.local_loss,
                    self.gather_with_grad,
                    self.rank,
                    self.world_size,
                )

                indices = paddle.concat(
                    [
                        paddle.arange(
                            i * text_features.shape[0] + grad_part_size * idx,
                            i * text_features.shape[0] + grad_part_size * (idx + 1),
                        )
                        for i in range(self.world_size)
                    ]
                )
                all_image_features[indices] = grad_image_features
                all_text_features[indices] = grad_text_features
                self.temp_gathered_features = (all_image_features, all_text_features)

            # logger.info(
            #     f"image_features >>> "
            #     f"all-image:{all_image_features.shape},{all_image_features.dtype} "
            #     f"@ all-text:{all_text_features.shape},{all_text_features.dtype}, "
            #     # f"feature_sizes:{feature_sizes}, "
            #     f"logit_scale:{logit_scale} "
            # )
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
                hcg = paddle.distributed.fleet.get_hybrid_communicate_group()
                shardinggroup = hcg.get_sharding_parallel_group()
                dpgroup = hcg.get_data_parallel_group()

                images_invalid_accumulated = gather_and_cat(
                    images_invalid_accumulated, shardinggroup
                )
                images_invalid_accumulated = gather_and_cat(
                    images_invalid_accumulated, dpgroup
                )
                text_invalid_accumulated = gather_and_cat(
                    text_invalid_accumulated, shardinggroup
                )
                text_invalid_accumulated = gather_and_cat(
                    text_invalid_accumulated, dpgroup
                )
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        total_loss = paddle.to_tensor(0.0)
        visual_loss = paddle.to_tensor(0.0)
        text_loss = paddle.to_tensor(0.0)
        i2t_acc = paddle.to_tensor(-1.0)
        t2i_acc = paddle.to_tensor(-1.0)
        # feature_sizes = F.pad(feature_sizes.cumsum(0), (1, 0)).tolist()

        with paddle.amp.auto_cast(False):
            num_logits = logits_per_image.shape[0]
            offset = logits_per_image.shape[1] // self.world_size
            assert num_logits == offset
            image_labels = paddle.arange(num_logits, dtype=paddle.int64)
            if self.world_size > 1 and self.local_loss:
                image_labels = image_labels + offset * self.rank
                # image_labels = paddle.arange(
                #     feature_sizes[self.rank], feature_sizes[self.rank + 1], dtype=paddle.int64
                # )
            image_labels[images_invalid_accumulated == 1] = -100
            visual_loss = F.cross_entropy(
                logits_per_image.astype("float32"), image_labels
            )
            i2t_acc = (logits_per_image.argmax(-1) == image_labels).sum() / len(
                logits_per_image
            )
            total_loss += visual_loss

            num_logits = logits_per_text.shape[0]
            text_labels = paddle.arange(num_logits, dtype=paddle.int64)
            if self.world_size > 1 and self.local_loss:
                text_labels = text_labels + offset * self.rank
                # text_labels = paddle.arange(
                #     feature_sizes[self.rank], feature_sizes[self.rank + 1], dtype=paddle.int64
                # )
            text_labels[text_invalid_accumulated == 1] = -100
            text_loss = F.cross_entropy(logits_per_text.astype("float32"), text_labels)
            t2i_acc = (logits_per_text.argmax(-1) == text_labels).sum() / len(
                logits_per_text
            )
            total_loss += text_loss

            total_loss /= 2

        info = {
            "logit_scale": logit_scale.detach(),
            "total_loss": total_loss.detach(),
            "visual_loss": visual_loss.detach(),
            "text_loss": text_loss.detach(),
            "i2t_acc": i2t_acc,
            "t2i_acc": t2i_acc,
        }
        return total_loss, info
