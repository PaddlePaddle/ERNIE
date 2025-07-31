# !/usr/bin/env python3

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
2025  # you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# 2025
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Paddle Erniemm model"""
import logging
import re
from typing import List, Optional
from collections import defaultdict
from functools import partial
import contextlib
import math
from copy import deepcopy

import paddle
from paddle import nn
import paddle.distributed as dist
from paddle.distributed import fleet
from models.comm_utils import profile
from models.ernie.modeling_auto import (
    ErniePretrainingCriterion as ErniePretrainingCriterionBase,
)
from models.ernie.modeling_auto import (
    RMSNorm,
    ErnieLMHead,
    parallel_matmul,
    ErnieForCausalLMAuto,
    CausalLMOutputWithCrossAttentionsAuto,
    get_mesh,
)
from models.ernie_mm_moe.modeling import (
    monkey_patch_param_hook,
    create_freeze_hook,
    construct_types_for_video,
)
from models.image_encoder import EVAVisionTransformerConfig
from models.image_encoder import EVAVisionTransformerAuto
from .configuration import ErniemmMoEConfig

try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}

logger = logging.getLogger(__name__)

__all__ = [
    "ErniemmMoEForCausalLMAuto",
]


class TokenType:
    """token type 含义"""

    text = 0
    image = 1
    video = 2
    audio = 3
    # text_in_image = 2


IDTYPES_2_ID = {"text": 0, "image": 1, "video": 2, "audio": 3}
IMAGETYPES_2_ID = {"image": 0, "video": 1, "padded_image": 2}


def get_backbone_lm_param_regex(config):
    """
    返回一个可以匹配所有 *backbone 网络中（不含experts)*  *纯LM* 参数名的 regex
    """
    if config.moe_group in fleet.auto.get_mesh().dim_names:
        moe_rank = fleet.auto.get_mesh().get_rank_by_dim_and_process_id(
            config.moe_group, dist.get_rank()
        )
        if moe_rank < 0:
            moe_rank = 0
    else:
        moe_rank = 0

    moe_world_size = config.moe_world_size

    num_local_experts = (
        sum(config.moe_num_experts) // moe_world_size
        if config.multimodel_experts
        else config.moe_num_experts // moe_world_size
    )
    num_freeze_expert = (
        config.moe_num_experts[0]
        if config.multimodel_experts
        else config.moe_num_experts
    )

    # SA-moe
    num_local_experts_sa = config.moe_num_attn_experts // moe_world_size

    freeze_part = [r"ernie\.norm.*", r"ernie\.layers.*norm.*"]  # freeze all norm
    # we do not include gate weight
    # gate weight 已经进行了模态隔离
    freeze_part += [
        r"ernie\.layers\.(\d+)\.mlp\.(up_gate|gate|up|down)_proj\.*",
        r"ernie\.layers\.(\d+)\.mlp\.shared_experts\.(up_gate|gate|up|down)_proj\.*",
        r"ernie\.layers\.(\d+)\.self_attn.(q|k|v|o|qkv)_proj\.(weight|bias)",
        r"ernie\.layers\.(\d)+\.mlp\.gate\.weight$",
    ]
    logger.info(f"FREEZE_DEBUG: { moe_rank * num_local_experts} {num_freeze_expert}")
    freeze_part += [r"ernie\.embed_tokens\.weight"]
    freeze_part += [r"lm_head\.weight", r"lm_head\.bias"]

    assert freeze_part, f"not freeze any part, moe: {moe_rank}/{moe_world_size}"
    logger.info(f"freeze pattern: {freeze_part}, moe: {moe_rank}/{moe_world_size}")
    freeze_part = re.compile("|".join(freeze_part))
    return freeze_part


class ResamplerModel(nn.Layer):
    """
    ResamplerModel, 负责空间、时间维度缩并。
    """

    def __init__(self, in_dim, out_dim, spatial_conv_size, temporal_conv_size, config):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size
        with paddle.utils.unique_name.guard("mm_resampler_"):
            self.spatial_linear = nn.Linear(
                self.in_dim * self.spatial_conv_size * self.spatial_conv_size,
                self.out_dim,
                bias_attr=True,
            )
            self.act_fn = nn.Silu()
            self.temporal_linear = nn.Linear(
                self.out_dim * self.temporal_conv_size,
                self.out_dim,
                bias_attr=True,
            )
            out_config = deepcopy(config)
            out_config.hidden_size = out_dim
            self.after_norm = RMSNorm(out_config)

        self.spatial_linear.weight = dist.shard_tensor(
            self.spatial_linear.weight,
            get_mesh(),
            [dist.Replicate(), dist.Shard(0)],
        )
        self.spatial_linear.bias = dist.shard_tensor(
            self.spatial_linear.bias,
            get_mesh(),
            [dist.Replicate(), dist.Replicate()],
        )
        self.temporal_linear.weight = dist.shard_tensor(
            self.temporal_linear.weight,
            get_mesh(),
            [dist.Replicate(), dist.Shard(1)],
        )
        self.temporal_linear.bias = dist.shard_tensor(
            self.temporal_linear.bias,
            get_mesh(),
            [dist.Replicate(), dist.Shard(0)],
        )

    def spatial_conv_reshape(self, x, spatial_conv_size):
        """
        Linear 前的 reshape，为了让 Linear 能模仿 conv 的感受野
        """
        B, S, C = x.shape
        H = int(math.sqrt(S))

        # x = x.reshape([B, H, H, C])  # B, H, W, C
        x = x.reshape(
            [B, H, H // spatial_conv_size, int(C * spatial_conv_size)]
        )  # N, H, W/conv, C*conv
        x = x.transpose([0, 2, 1, 3])  # N, W/conv, H, C*conv
        x = x.reshape(
            [
                B,
                H // spatial_conv_size,
                H // spatial_conv_size,
                int(C * spatial_conv_size * spatial_conv_size),
            ]
        )  # N, W/conv, H/conv, C*conv*conv
        x = x.transpose([0, 2, 1, 3])  # N, H/conv, W/conv, C*conv*conv

        x = x.reshape(
            [B, -1, int(C * spatial_conv_size * spatial_conv_size)]
        )  # N, (H * W)/(conv * conv), C*conv*conv

        return x

    def forward(
        self,
        x,
        image_mask,
        token_type_ids,
        image_type_ids,
    ):
        """
        x: image_features
        image_mask: [B]
        token_types_ids: [B]
        image_type_ids:  [B_image]
        """

        assert image_type_ids is not None
        (
            image_is_video,
            compressed_image_indices,
            video_images_with_placeholder,
        ) = construct_types_for_video(image_mask, token_type_ids, image_type_ids)
        assert len(compressed_image_indices) == len(image_mask[image_mask]), (
            f"len(compressed_image_indices): {compressed_image_indices.shape}, "
            f"len(image_mask): {image_mask.astype('int64').sum()} "
        )

        x = self.spatial_conv_reshape(x, self.spatial_conv_size)
        x = self.spatial_linear(x)
        x = self.act_fn(x)

        # separate image and video from features, # B, S, C
        image_features_video = x[image_is_video == 1]
        image_features_nonvideo = x[image_is_video == 0]

        if image_features_video.shape[0] != 0:
            # use video_images_with_placeholder to construct placeholder
            video_placeholder = paddle.zeros(
                video_images_with_placeholder.shape[0:1]
                + image_features_video.shape[1:],
                dtype=image_features_video.dtype,
            )
            paddle.index_put_(
                video_placeholder,
                [video_images_with_placeholder == 1],
                image_features_video,
            )
            B_video_placeholder = video_placeholder.shape[0] // self.temporal_conv_size
        else:
            B_video_placeholder = 0

        if image_features_nonvideo.shape[0] != 0:
            # same padding for image
            # TODO: check repeat_interleave
            image_placeholder = paddle.repeat_interleave(
                image_features_nonvideo, self.temporal_conv_size, 0
            )
            B_image_placeholder = image_placeholder.shape[0] // self.temporal_conv_size
        else:
            B_image_placeholder = 0

        if B_video_placeholder != 0 and B_image_placeholder != 0:
            # merge video and image
            placeholder = paddle.stack([image_placeholder, video_placeholder], axis=0)
        elif B_video_placeholder != 0:
            placeholder = video_placeholder
        elif B_image_placeholder != 0:
            placeholder = image_placeholder
        else:
            raise ValueError(
                f"there is no image and video! image_is_video:{image_is_video}, "
                f"compressed_image_indices:{compressed_image_indices}, "
                f"video_images_with_placeholder:{video_images_with_placeholder}"
            )

        # TODO: check this
        B, S, C = placeholder.shape
        placeholder = placeholder.transpose([1, 0, 2])  # S, B, C
        placeholder = placeholder.reshape(
            [S, B // self.temporal_conv_size, int(C * self.temporal_conv_size)]
        )  # S, B/Sconv, C*Sconv
        placeholder = self.temporal_linear(placeholder)  # S, B/Sconv, C*Sconv
        placeholder = placeholder.transpose([1, 0, 2])  # B/Sconv, S, C*Sconv
        placeholder = self.after_norm(placeholder)  # B/Sconv, S, C*Sconv

        # # separate video and image from placeholder and put them into compressed placeholder
        compressed_placeholder = paddle.zeros(
            compressed_image_indices.shape[0:1] + [placeholder.shape[-1]],
            dtype=placeholder.dtype,
        )
        if B_video_placeholder != 0:
            compressed_video = placeholder[B_image_placeholder:, ...].reshape(
                [-1, placeholder.shape[-1]]
            )
            paddle.index_put_(
                compressed_placeholder,
                [compressed_image_indices == 1],
                compressed_video,
            )
        if B_image_placeholder != 0:
            compressed_image = placeholder[:B_image_placeholder, ...].reshape(
                [-1, placeholder.shape[-1]]
            )
            paddle.index_put_(
                compressed_placeholder,
                [compressed_image_indices == 0],
                compressed_image,
            )

        return compressed_placeholder


class ErniePretrainingCriterion(ErniePretrainingCriterionBase):
    """
    继承顺序：
    ErnieMMMoE -> ErnieMoE -> Ernie
    """

    def __init__(self, config):
        super().__init__(config)
        assert self.enable_parallel_cross_entropy, (
            config.tensor_parallel_degree,
            config.tensor_parallel_output,
        )
        self.im_patch_id = config.im_patch_id
        self.max_text_id = config.max_text_id

    def forward(
        self,
        scores_text,
        scores_image,
        labels,
        token_type_ids_shifted,
        token_type_ids_untouched,
        lm_weight=None,
        lm_bias=None,
        mm_weight=None,
        mm_bias=None,
        router_loss=None,
    ):
        """
        文图分离 Criterion,只返回文本 样本的CEloss，其他 loss 会 update 到 `global_training_logs` 里。
        Args:
            score_text: 文本 logits，只包含纯文数据。
            scores_text_in_image: 只包含文本数据 logits。
            scores_image: 只包含图片 sepecial-token 的 logits。
            labels: 原始 label，可能包含文本、图片 sepecial-token、ignored-index。
            token_type_ids_shifted: `labels` 对应的 token-type。
            router_loss: router_loss
        Returns:
            loss: 纯文样本的 CE loss
            loss_sum. 纯文样本的 CE loss_sum
        """
        image_mask_shifted = token_type_ids_shifted == TokenType.image
        text_pos_shifted = token_type_ids_shifted == TokenType.text

        sample_type_ids = token_type_ids_shifted.max(-1)  # [0:text, 1:lm, 2:video]
        has_video = token_type_ids_untouched.max().item() == TokenType.video
        pure_text = token_type_ids_untouched.max().item() == TokenType.text

        assert (
            scores_text is not None
        ), f"no text or image token provided, text: {scores_text}"

        if scores_text is not None:
            # TODO(zhangyuqin): nonzero问题, 先不使用bool tensor的索引的写法
            # labels_text = labels[text_pos_shifted]
            labels_text = labels
            # labels_text = labels
            # assert labels_text.size > 0, labels
            if self.config.use_recompute_loss_fn:
                assert lm_weight is not None and mm_weight is not None
                loss, loss_sum = super().forward(
                    (scores_text.unsqueeze(0), lm_weight, lm_bias),
                    labels_text.unsqueeze(0),
                )
            else:
                # TODO(zhangyuqin): unsqueeze_grad会报错, 需要排查
                loss, loss_sum = super().forward(scores_text, labels_text)
                # loss, loss_sum = super().forward(scores_text.unsqueeze(0), labels_text.unsqueeze(0))

            if pure_text:
                if loss._is_initialized():
                    global_training_logs.update(lm_loss=loss.detach())
            else:
                if has_video:
                    global_training_logs.update(video_loss=loss.detach())
                else:
                    global_training_logs.update(image_loss=loss.detach())
        else:
            assert 0
            loss = paddle.zeros([], dtype="float32")
            loss.stop_gradient = False

        if scores_image is not None:
            labels_image = labels[image_mask_shifted]
            assert labels_image.size > 0, labels
            labels_image = paddle.where(
                labels_image >= 0, labels_image - self.max_text_id, labels_image
            )  # do not move ignored-index
            if self.config.use_recompute_loss_fn:
                assert mm_weight is not None and mm_bias is not None
                loss_image, _ = super().forward(
                    (scores_image.unsqueeze(0), mm_weight, mm_bias),
                    labels_image.unsqueeze(0),
                )
            else:
                # TODO(zhangyuqin): unsqueeze_grad会报错, 需要排查
                loss_image, _ = super().forward(scores_image, labels_image)
                # loss_image, _ = super().forward(scores_image.unsqueeze(0), labels_image.unsqueeze(0))
            global_training_logs.update(image_special_token_loss=loss_image.detach())
            loss = loss + loss_image - loss_image.detach()

        if router_loss is not None:
            if router_loss._is_initialized():
                global_training_logs.update(router_loss=router_loss.detach())
            loss = loss + router_loss - router_loss.detach()

        return loss, loss_sum


def calc_multimodal_logits(
    last_hidden_state: paddle.Tensor,
    lm_head_weight: paddle.Tensor,
    lm_head_bias: paddle.Tensor,
    mm_head_weight: paddle.Tensor,
    mm_head_bias: paddle.Tensor,
    token_type_ids_shifted: paddle.Tensor,
    config: ErniemmMoEConfig,
):
    """
    分别计算 纯文、图文、图片 位置的 logits
    Args:
        last_hidden_state: 最后一层的 hidden，在 sequence-parallel下，处于切分状态。
        ...
        token_type_ids_shifted: # 非 sp 切分tensor
            label 位置的 token-type-ids，用于选择每个 token 对应的 lm-head。
            注意：图文交替的 id 序列中，最后一个 文本 token 会预测 图id，反之亦然，
            所以需要选择 label type 对应的 lmhead weight。
    """
    # 将 ids 的类型对齐 label的类型，对于最后一个 ids，认为token type不变
    # TODO:从 reader 中传入 token-type-ids
    # token_type_ids_shifted = paddle.concat([token_type_ids[:, 1:], token_type_ids[:, -1:]], 1)  #
    if config.sequence_parallel or config.submatrix_parallel:
        last_hidden_state = dist.reshard(
            last_hidden_state,
            get_mesh(-1),
            [dist.Shard(1), dist.Replicate()],
        )
        # [S, B, H] to [B, S, H]
        last_hidden_state = paddle.transpose(last_hidden_state, [1, 0, 2])

    image_type = paddle.full(
        token_type_ids_shifted.shape,
        TokenType.image,
        dtype=token_type_ids_shifted.dtype,
    )
    text_type = paddle.full(
        token_type_ids_shifted.shape, TokenType.text, dtype=token_type_ids_shifted.dtype
    )

    image_mask_shifted = token_type_ids_shifted == image_type
    text_pos_shifted = token_type_ids_shifted == text_type
    # image_mask_shifted = token_type_ids_shifted == TokenType.image
    # text_pos_shifted = token_type_ids_shifted == TokenType.text

    parallel_matmul_tp = partial(
        parallel_matmul,
        tensor_parallel_degree=config.tensor_parallel_degree,
        tensor_parallel_output=True,
        fuse_linear=config.fuse_linear,
    )
    if config.sequence_parallel:
        # TODO(zhangyuqin): 动手在这里手动执行了GatherOp, 动半直接是全复制。需检查sp场景激活值是否被正确切分。
        last_hidden_state = last_hidden_state.reshape(
            [-1, config.seqlen, last_hidden_state.shape[-1]]
        )

    # assert last_hidden_state.shape[:2] == token_type_ids_shifted.shape, (
    #     last_hidden_state.shape,
    #     token_type_ids_shifted.shape,
    # )

    # TODO(zhangyuqin): 这里pp会有问题, 需要排查
    if paddle.sum(paddle.any(text_pos_shifted)) > 0:
        if config.use_recompute_loss_fn:
            score_text = last_hidden_state[text_pos_shifted]
        else:
            # TODO(zhangyuqin): nonzore的global_shape有问题, 暂时不用last_hidden_state[text_pos_shifted]的写法
            score_text = parallel_matmul_tp(
                last_hidden_state, lm_head_weight, lm_head_bias
            )
            # score_text = parallel_matmul_tp(last_hidden_state[text_pos_shifted], lm_head_weight, lm_head_bias)
    else:
        score_text = None

    if paddle.sum(paddle.any(image_mask_shifted)) > 0:
        if config.use_recompute_loss_fn:
            score_image = last_hidden_state[image_mask_shifted]
        else:
            score_image = parallel_matmul_tp(
                last_hidden_state[image_mask_shifted], mm_head_weight, mm_head_bias
            )
    else:
        score_image = None
    return score_text, score_image


class ErniemmMoEForCausalLMAuto(ErnieForCausalLMAuto):
    """ErniemmForCausalLM"""

    config_class = ErniemmMoEConfig
    main_input_name = "pixel_values"

    def __init__(
        self, config: ErniemmMoEConfig, vision_model=None, resampler_model=None
    ):
        super().__init__(config)
        self.criterion = ErniePretrainingCriterion(config)  # 复写
        self._freeze_vision = False
        self.use_eva_clip = False
        self.wo_vit = False
        self.modality_detach = config.modality_detach
        if vision_model is not None:
            config.vision_config = vision_model.config

        if config.vision_config is None:
            self.vision_model = nn.Identity()
            self.wo_vit = True
        if vision_model is not None:
            logger.info(
                "********** Loading Vision Model from pretrained model **********"
            )
            self.vision_model = vision_model
            if isinstance(self.vision_model.config, EVAVisionTransformerConfig):
                self.use_eva_clip = True
        # ernie-core 中无法 import OpenClip，可以从 __init__中传入OpenClip模型
        # elif isinstance(config.vision_config, CLIPVisionConfig):
        #     logger.info("********** Vision Model <CLIP> is random initialized **********")
        #     self.vision_model = CLIPVisionModel(config.vision_config)
        elif isinstance(config.vision_config, EVAVisionTransformerConfig):
            logger.info(
                "********** Vision Model <EVA-CLIP> is random initialized **********"
            )
            config.vision_config.use_recompute_attn = config.use_recompute_attn_vision
            config.vision_config.tensor_parallel_degree = config.tensor_parallel_degree
            config.vision_config.tensor_parallel_rank = config.tensor_parallel_rank
            self.vision_model = EVAVisionTransformerAuto(config.vision_config)
            self.use_eva_clip = True

        mm_vocab_config = deepcopy(config)
        mm_vocab_config.vocab_size = config.mm_vocab_size
        assert mm_vocab_config.vocab_size > 0, mm_vocab_config
        assert (
            mm_vocab_config.im_patch_id >= mm_vocab_config.max_text_id
        ), mm_vocab_config

        self.mm_head = ErnieLMHead(mm_vocab_config)
        if config.mm_vocab_size > 0:
            self.mm_embed_tokens = nn.Embedding(
                config.mm_vocab_size, config.hidden_size
            )

            if (
                self.config.tensor_parallel_degree > 1
                or self.config.pipeline_parallel_degree > 1
            ):
                self.mm_embed_tokens.weight = dist.shard_tensor(
                    self.mm_embed_tokens.weight,
                    get_mesh(),
                    [dist.Replicate(), dist.Shard(0)],
                )
        else:
            self.mm_embed_tokens = None
        self.resampler_model = ResamplerModel(
            config.pixel_hidden_size,
            config.hidden_size,
            config.spatial_conv_size,
            config.temporal_conv_size,
            config,
        )

        self.adaptive_resolution = (
            not self.wo_vit
            and self.vision_model.config.resolution_transform == "adaptive"
        )
        self._modality_param_mapping = None
        self.image_preprocess = None

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):

        from paddleformers.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}
            if config.fuse_attn_ffn:
                base_actions = {
                    # Column Linear
                    "layers.0.self_attn.qkv_proj.weight": partial(fn, is_column=True),
                    "layers.0.mlp.up_gate_proj.weight": partial(
                        fn, is_column=True, is_naive_2fuse=True
                    ),
                    "lm_head.weight": partial(
                        fn, is_column=not config.tie_word_embeddings
                    ),
                    # Row Linear
                    "embed_tokens.weight": partial(fn, is_column=False),
                    "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                    "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
                }
                if config.use_bias:
                    base_actions.update(
                        {
                            # Column Linear
                            "layers.0.self_attn.qkv_proj.bias": partial(
                                fn, is_column=True
                            ),
                            "layers.0.mlp.up_gate_proj.bias": partial(
                                fn, is_column=True, is_naive_2fuse=True
                            ),
                            "lm_head.bias": partial(fn, is_column=True),
                        }
                    )
            else:
                base_actions = {
                    # Column Linear
                    "layers.0.self_attn.q_proj.weight": partial(fn, is_column=True),
                    "layers.0.self_attn.k_proj.weight": partial(fn, is_column=True),
                    "layers.0.self_attn.v_proj.weight": partial(fn, is_column=True),
                    "layers.0.mlp.gate_proj.weight": partial(fn, is_column=True),
                    "layers.0.mlp.up_proj.weight": partial(fn, is_column=True),
                    "lm_head.weight": partial(
                        fn, is_column=not config.tie_word_embeddings
                    ),
                    # Row Linear
                    "embed_tokens.weight": partial(fn, is_column=False),
                    "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                    "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
                }
                if config.use_bias:
                    base_actions.update(
                        {
                            # Column Linear
                            "layers.0.self_attn.q_proj.bias": partial(
                                fn, is_column=True
                            ),
                            "layers.0.self_attn.k_proj.bias": partial(
                                fn, is_column=True
                            ),
                            "layers.0.self_attn.v_proj.bias": partial(
                                fn, is_column=True
                            ),
                            "layers.0.mlp.gate_proj.bias": partial(fn, is_column=True),
                            "layers.0.mlp.up_proj.bias": partial(fn, is_column=True),
                            "lm_head.bias": partial(fn, is_column=True),
                        }
                    )
            moe_in_mp = config.moe_group in {"mp", "model", "tp", "mpdp"}
            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        if "mlp" in key and i % config.moe_layer_interval == 0:
                            moe_num_experts = (
                                sum(config.moe_num_experts)
                                if config.multimodel_experts
                                else config.moe_num_experts
                            )
                            for expert_id in range(moe_num_experts):
                                _key = key.replace(
                                    "layers.0.mlp",
                                    f"layers.{i}.mlp.experts.{expert_id}",
                                )
                                if moe_in_mp:
                                    final_actions[_key] = lambda x: x
                                else:
                                    final_actions[_key] = action
                        elif "self_attn" in key and (
                            "qkv_proj" in key
                            or "q_proj" in key
                            or "k_proj" in key
                            or "v_proj" in key
                        ):
                            for expert_id in range(config.moe_num_attn_experts):
                                _key = key.replace("layers.0.", f"layers.{i}.").replace(
                                    "_proj.", f"_proj.experts.{expert_id}."
                                )
                                if moe_in_mp:
                                    final_actions[_key] = lambda x: x
                                else:
                                    final_actions[_key] = action
                        else:
                            final_actions[key.replace("layers.0.", f"layers.{i}.")] = (
                                action
                            )
                else:
                    final_actions[key] = action

            if config.vision_config is not None and isinstance(
                config.vision_config, EVAVisionTransformerConfig
            ):
                vfn = split_or_merge_func(
                    is_split=is_split,
                    tensor_parallel_degree=config.tensor_parallel_degree,
                    tensor_parallel_rank=config.tensor_parallel_rank,
                    num_attention_heads=config.vision_config.width
                    // config.vision_config.head_width,
                )
                vision_actions = {
                    # Column Linear
                    "vision_model.blocks.0.mlp.fc1.weight": partial(
                        vfn, is_column=True
                    ),
                    "vision_model.blocks.0.mlp.fc1.bias": partial(vfn, is_column=True),
                    "vision_model.blocks.0.mlp.fc2.weight": partial(
                        vfn, is_column=False
                    ),
                    "vision_model.blocks.0.attn.proj.weight": partial(
                        fn, is_column=False
                    ),
                    # "vision_model.head.weight": partial(vfn, is_column=True),
                    # "vision_model.head.bias": partial(vfn, is_column=True),
                }
                if config.vision_config.fused_qkv_attn:
                    vision_actions.update(
                        {
                            "vision_model.blocks.0.attn.qkv_proj.weight": partial(
                                vfn, is_column=True
                            ),
                        }
                    )
                    if config.vision_config.qkv_bias:
                        vision_actions.update(
                            {
                                "vision_model.blocks.0.attn.qkv_proj.bias": partial(
                                    vfn, is_column=True
                                ),
                            }
                        )
                else:
                    vision_actions.update(
                        {
                            "vision_model.blocks.0.attn.q_proj.weight": partial(
                                vfn, is_column=True
                            ),
                            "vision_model.blocks.0.attn.k_proj.weight": partial(
                                vfn, is_column=True
                            ),
                            "vision_model.blocks.0.attn.v_proj.weight": partial(
                                vfn, is_column=True
                            ),
                        }
                    )
                    if config.vision_config.qkv_bias:
                        vision_actions.update(
                            {
                                "vision_model.blocks.0.attn.q_proj.bias": partial(
                                    vfn, is_column=True
                                ),
                                "vision_model.blocks.0.attn.k_proj.bias": partial(
                                    vfn, is_column=True
                                ),
                                "vision_model.blocks.0.attn.v_proj.bias": partial(
                                    vfn, is_column=True
                                ),
                            }
                        )
                for key, action in vision_actions.items():
                    if "blocks.0." in key:
                        for i in range(config.vision_config.layers):
                            final_actions[key.replace("blocks.0.", f"blocks.{i}.")] = (
                                action
                            )
                    final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        return mappings

    @staticmethod
    def _resolve_prefix_keys(state_keys_base, state_keys_real, ignore_error=False):
        # state_keys_map base to real
        state_keys_map = {}

        state_keys_base = set(state_keys_base)
        state_keys_real = set(state_keys_real)

        for key in state_keys_base:
            for x in state_keys_real:
                # 跳过 vision 和 mapping model 的 mp 参数切分
                if "resampler_model" in x:
                    continue
                if x.endswith(key):
                    state_keys_map[key] = x
                    break
            if key not in state_keys_map:
                if not ignore_error:
                    logger.error(f"could not find name {key} in loaded state dict!")
            else:
                state_keys_real.remove(state_keys_map[key])

        return state_keys_map

    def _set_modality_param_mapping(self):
        """_summary_"""
        lm_pattern = get_backbone_lm_param_regex(self.config)
        self._modality_param_mapping = defaultdict(lambda: [])
        for name, param in self.named_parameters():
            monkey_patch_param_hook(param)
            expert_type = getattr(param, "expert_type", None)
            if "vision_model" in name:
                self._modality_param_mapping["vit"].append(
                    (name, param, create_freeze_hook(name, param))
                )
            elif lm_pattern.match(name) or expert_type == "expert_type_0":
                self._modality_param_mapping["lm"].append(
                    (name, param, create_freeze_hook(name, param))
                )
            else:
                self._modality_param_mapping["mm"].append(
                    (name, param, create_freeze_hook(name, param))
                )
        debug_msg = {
            k: [i[0] for i in v] for k, v in self._modality_param_mapping.items()
        }
        logger.info(f"modality_param_mapping: {debug_msg}")

    def update_params_stat(self, param_group, stop_gradient):
        """freeze mm"""
        assert param_group in (
            "lm",
            "mm",
            "vit",
        ), "param_group must be in ('lm', 'mm', 'vit')"
        if self._modality_param_mapping is None:
            self._set_modality_param_mapping()
        for name, param, _ in self._modality_param_mapping.get(param_group, []):
            logger.info(f"mm: {name} set_stop_gradient to {stop_gradient}")
            param.stop_gradient = stop_gradient

    def freeze_vision(self):
        """freeze_vision"""
        if self._modality_param_mapping is None:
            self._set_modality_param_mapping()
        for name, param, _ in self._modality_param_mapping.get("vit", []):
            logger.info("Freezing vision parameter: {}".format(name))
            param.stop_gradient = True
        self.config.freeze_vision = True

    def vision_forward(
        self,
        images,
        image_position_ids,
        image_attention_mask,
    ):
        """vision_forward"""
        with paddle.no_grad() if self._freeze_vision else contextlib.nullcontext():
            if self.wo_vit:
                return images
            elif self.use_eva_clip:
                cls_fea, image_sizes, image_features, hidden = self.vision_model(
                    images,
                    position_ids=image_position_ids,
                    return_all_features=True,
                )
            else:
                image_forward_out = self.vision_model(
                    images,
                    output_hidden_states=True,
                    position_ids=image_position_ids,
                    attention_mask=image_attention_mask,
                )
                if self.adaptive_resolution:
                    image_features = image_forward_out.hidden_states[-1]
                else:
                    image_features = image_forward_out.hidden_states[-1][:, 1:, :]
        if self._freeze_vision:
            image_features = image_features.detach()
        return image_features

    def add_image_preprocess(self, preprocess):
        """add_image_preprocess"""
        logger.info("image preprocess is set")
        self.image_preprocess = preprocess

    def mapping_forward(
        self,
        token_type_ids,
        input_ids,
        mm_input_ids,
        image_features,
        inputs_embeds,
        image_type_ids,
    ):
        """mapping_forward"""
        # debug = mm_input_ids - self.config.max_text_id

        if self.mm_embed_tokens is not None:
            mm_ids_features = self.mm_embed_tokens(
                mm_input_ids - self.config.max_text_id
            )
            inputs_embeds[token_type_ids == TokenType.image] = mm_ids_features[
                token_type_ids == TokenType.image
            ]
        else:
            assert (mm_input_ids <= self.config.max_text_id).all().item(), (
                f"found vistual token in ids, but `mm_vocab_size` == 0, "
                f"ids:{input_ids}, max_text_id={self.config.max_text_id} "
            )

        image_mask = input_ids == self.config.im_patch_id

        image_features = self.resampler_model(
            image_features,
            image_mask,
            token_type_ids,
            image_type_ids,
        )

        if not self.wo_vit and not self.adaptive_resolution and image_features.dim == 2:
            B, N, C = image_features.shape
            image_features = image_features.reshape([B * N, C]).astype(
                inputs_embeds.dtype
            )
        inputs_embeds[token_type_ids == TokenType.image] = mm_ids_features[
            token_type_ids == TokenType.image
        ]
        # 会覆盖 `mm_ids_features` 中 `ids==im_patch_id` 的部分
        inputs_embeds[image_mask] = image_features
        # # TODO 对部分参数进行normalize 对文部分detach 打印图token
        # text_token_norm = inputs_embeds[input_ids != self.config.im_patch_id].norm(axis=-1).mean()
        # image_token_norm = image_features.norm(axis=-1).mean()
        # image_features /= paddle.sqrt(image_token_norm / text_token_norm)
        return inputs_embeds

    def prepare_inputs_for_generation(
        self,
        input_ids,
        images,
        use_cache=False,
        past_key_values=None,
        inputs_embeds=None,
        image_position_ids=None,
        image_attention_mask=None,
        **kwargs,
    ):
        """prepare_inputs_for_generation"""
        if past_key_values:
            input_ids = input_ids[:, -1:]

        attention_mask = kwargs.get("attention_mask", None)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": True,
                "attention_mask": attention_mask,
                "return_dict": True,
                "images": images,
                "image_position_ids": image_position_ids,
                "image_attention_mask": image_attention_mask,
            }
        )
        return model_inputs

    def _post_init(self, original_init, *args, **kwargs):
        """
        标记模型中所有的多模态参数, 只处理 head 和 Embedding
        experts 参数在`ernie_moe/modeling.py`中已经完成了标记
        """
        super()._post_init(self, original_init, *args, **kwargs)
        self.mm_embed_tokens.weight.expert_type = (
            "expert_type_1"  # 借用 `ernie_moe/modeling.py` 中的标记。
        )
        self.mm_head.weight.expert_type = "expert_type_1"
        if hasattr(self.mm_head, "bias"):
            self.mm_head.bias.expert_type = "expert_type_1"

    def forward(
        self,
        input_ids: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[paddle.Tensor] = None,
        images: Optional[paddle.Tensor] = None,
        ignored_index: Optional[int] = 0,
        return_dict: Optional[bool] = None,
        image_position_ids: Optional[paddle.Tensor] = None,
        image_attention_mask: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        image_type_ids: Optional[paddle.Tensor] = None,
        # cumulative_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        """forward"""

        # from dataloader
        if isinstance(input_ids, list):
            if len(input_ids) == 9:
                (
                    input_ids,
                    labels,
                    data_id,
                    src_id,
                    data_type,
                    images,
                    token_type_ids,
                    image_type_ids,
                    has_images,
                ) = input_ids
            else:
                raise ValueError(
                    f"Unexpected input length, len(inputs) = {len(input_ids)}"
                )

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        image_mask = input_ids == self.config.im_patch_id

        image_rate = image_mask.astype("float32").mean()
        # if labels is not None:
        #     pad_rate = ((labels == self.criterion.ignored_index) & (~image_mask)).astype("float32").mean()
        #     global_training_logs.update(image_rate=image_rate, pad_rate=pad_rate)
        #     assert paddle.any(labels > 0).item(), labels

        if past_key_values is None:
            if has_images.all():
                assert (image_mask).any().item(), (
                    image_mask.numpy().tolist(),
                    input_ids.numpy().tolist(),
                    self.config.im_patch_id,
                    images.shape,
                )

                with profile("extract_image_fea"):
                    if self.image_preprocess is not None:
                        images = self.image_preprocess.rescale_factor * images.astype(
                            "float32"
                        )
                        images = (
                            images - self.image_preprocess.image_mean_tensor
                        ) / self.image_preprocess.image_std_tensor
                        images = images.astype("bfloat16")
                    else:
                        assert images.dtype == paddle.bfloat16, images.dtype

                    image_features = self.vision_forward(
                        images,
                        image_position_ids,
                        image_attention_mask,
                    )
                # image_features = paddle.concat([if1, if2], 0)[:29, ...]
                # logger.info(f'<if2> NonPP image_fea: {if2.reshape([15,-1]).astype("float32").norm(axis=-1)}')
                # mp 切 fea
                image_features = dist.reshard(
                    image_features, get_mesh(), [dist.Shard(0), dist.Shard(2)]
                )
                # image_features = ScatterOp.apply(image_features, axis=-1)
            else:
                image_features = None  # no more faking

        # inputs_embeds.stop_gradient = False
        # 0 == 纯文， 1 == 图片，1 会激活 >1 的 expert
        if token_type_ids is None:
            # assert 0, f"别自己造了，请用数据流给的. token_type_ids: {token_type_ids}, image_type_ids: {image_type_ids}"
            token_type_ids = image_mask.astype("int64")
            token_type_ids_labels = paddle.concat(
                [token_type_ids[:, 1:], token_type_ids[:, -1:]], 1
            )
        else:
            assert (
                token_type_ids.shape[1] == input_ids.shape[1] + 1
            ), f"token_type:{token_type_ids.shape}, ids:{input_ids.shape}"
            token_type_ids_labels = token_type_ids[..., 1:]
            # token_type_ids = token_type_ids[..., :-1]

        lm_input_ids = input_ids.clone()
        mm_input_ids = input_ids.clone()

        lm_input_ids[token_type_ids[..., :-1] == TokenType.image] = 0
        # 在 embedding lookup 的时候会统一减去 `max_text_id`
        # 用 `max_text_id` + 1 来替换文本部分 id，是为了跟 `im_patch_id` 区分开。替换部分不会加到最终的input_embeds上所以无所谓。
        # assert self.config.max_text_id + 1 != self.config.im_patch_id,  \
        #      f'max_text_id:{self.config.max_text_id}, im_pach_id:{self.config.im_patch_id}'
        mm_input_ids[token_type_ids[..., :-1] == TokenType.text] = (
            self.config.max_text_id
        )
        # TODO： audio token

        inputs_embeds = self.ernie.embed_tokens(lm_input_ids)

        if has_images.all():
            inputs_embeds = self.mapping_forward(
                token_type_ids[..., :-1],
                input_ids,
                mm_input_ids,
                image_features,
                inputs_embeds,
                image_type_ids,
            )
        else:
            pass  # do nothing, should not hang under DygraphShardingOptimizerV2

        # ErnieModelAuto
        outputs = self.ernie(
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # if token_type_ids_labels is not None:
        #     token_type_ids_labels = dist.reshard(token_type_ids_labels,
        #           outputs.last_hidden_state.process_mesh,
        #           [dist.Replicate() for _ in range(outputs.last_hidden_state.process_mesh.ndim)])

        logits, logits_image = calc_multimodal_logits(
            outputs.last_hidden_state,
            self.lm_head.weight,
            self.lm_head.bias,
            self.mm_head.weight,
            self.mm_head.bias,
            token_type_ids_labels,
            self.config,
        )
        router_loss = outputs.router_loss

        if return_dict:  # aka Generate Decoding
            if labels is not None:
                loss, _ = self.criterion(
                    logits,
                    None,
                    None,
                    labels,
                    token_type_ids_labels,
                    router_loss=outputs.router_loss,
                )
            else:
                loss = None
            return CausalLMOutputWithCrossAttentionsAuto(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_loss=outputs.router_loss,
            )
        # Pretrain & Eval 必须有labels
        assert labels is not None
        loss = self.criterion(
            logits,
            logits_image,
            labels,
            token_type_ids_labels,
            token_type_ids,
            router_loss=router_loss,
        )
        return loss
