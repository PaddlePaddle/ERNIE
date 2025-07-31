# -*- coding: utf-8 -*-
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
@author: kebo
@contact: kebo01@baidu.com

@version: 1.0
@file: modeling.py
@time: 2024/03/25 11:50:28
@Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved

这一行开始写关于本文件的说明与解释


"""
import logging
import numpy as np
import paddle
import paddle.distributed.fleet as fleet
import paddle.nn as nn
from paddle.nn import functional as F
from paddleformers.trainer.plugins.timer import get_timers
from models.ernie.modeling import ErnieForCausalLM
from models.image_encoder.eva_vit_model import (
    EVAVisionTransformer,
)
from models.image_encoder.configuration import ImageEncoderConfig
from models.ernie.modeling import ErniePretrainingCriterion
from .clip_loss import ClipLoss

FusedLinear = type("Linear", (paddle.incubate.nn.FusedLinear,), {})

logger = logging.getLogger(__name__)

try:
    pass
except Exception:

    logger.warning("Warning, FusedLn module is not available, use LayerNorm instead.")


class FCResampler(paddle.nn.Layer):
    """
    简单可依赖的 Linear 作为 Resampler
    """

    def __init__(self, vision_hidden_size, language_hidden_size):
        super().__init__()
        self.fc = paddle.nn.Linear(vision_hidden_size, language_hidden_size)

    def forward(self, x):
        """doc"""
        x = self.fc(x)
        return x


class LogitScale(nn.Layer):
    """
    防止logit_scale 被amp cast
    """

    def __init__(self, init_exp_logit_scale=1 / 0.07):
        super().__init__()
        init_data = paddle.ones(shape=[1], dtype="float32") * np.log(
            init_exp_logit_scale
        )
        self.scale = self.create_parameter(
            shape=[1],
            default_initializer=paddle.nn.initializer.Assign(init_data),
            dtype="float32",
        )
        self._cast_to_low_precision = False  # 兼容develop分支paddle
        self._cast_to_low_precison = False


class ImageEncoderCriterion(nn.Layer):
    """
    Criterion 同时负责维护 图像特征和文本特征的缓存、text-tower、language-model decoder.
    """

    def __init__(
        self,
        config,
        clipdata_cliploss_weight,
        clipdata_decoderloss_weight,
        decoderdata_decoderloss_weight,
    ):
        super().__init__()
        self.config = config
        # resampler 与 language model 放在criterion中 用于算decoder_loss
        self.train_clip_only = config.train_clip_only
        logger.info("train_clip_only: {}".format(self.train_clip_only))
        self.im_patch_id = config.im_patch_id
        self.pad_token_id = config.pad_token_id
        self.ignore_index = config.ignore_index

        self.text_model = None
        self.language_model = None
        self.resampler = None

        self.logit_scale = LogitScale(config.init_exp_logit_scale)

        # if config.tensor_parallel_rank == 0:
        self.clip_loss = ClipLoss(
            config.clip_local_loss,
            config.clip_gather_with_grad,
            config.clip_cache_labels,
            not config.freeze_vit,
            not config.freeze_text_tower,
            config.clip_data_rank,
            config.clip_data_world_size,
        )

        self.decoder_loss = None
        self.detach_decoderloss = False
        self.accum_image_embs = []
        self.accum_text_embs = []
        self.accum_invalid_ids = []

        self.clipdata_cliploss_weight = clipdata_cliploss_weight
        self.clipdata_decoderloss_weight = clipdata_decoderloss_weight
        self.decoderdata_decoderloss_weight = decoderdata_decoderloss_weight
        self.build_memory_bank = False

        logger.info(
            f"criterion weights: clipdata_cliploss_weight: {self.clipdata_cliploss_weight}, "
            + f"clipdata_decoderloss_weight: {self.clipdata_decoderloss_weight}, "
            + f"decoderdata_decoderloss_weight: {self.decoderdata_decoderloss_weight}"
        )

        if config.tensor_parallel_degree > 1:
            _hcg = fleet.get_hybrid_communicate_group()
            self.mp_group = _hcg.get_model_parallel_group()
            self.mp_src_rank = _hcg.get_model_parallel_group_src_rank()

    def set_language_model(self, model):
        """_summary_

        Args:
            model (_type_): _description_
        """
        self.language_model = model
        self.language_model.config.ignored_index = self.ignore_index
        self.language_model.config.tensor_parallel_output = True
        self.decoder_loss = ErniePretrainingCriterion(model.config)
        self.config.language_model_config = model.config
        if not self.train_clip_only:
            self.resampler = FCResampler(
                self.config.hidden_size, model.config.hidden_size
            )
        logger.info("image_encoder_criterion: language model is set")

    def set_text_model(self, model):
        """_summary_

        Args:
            model (_type_): _description_
        """
        self.text_model = model
        self.config.text_model_config = model.config

    def forward(self, preds, labels):
        """_summary_

        Args:
            preds (_type_): _description_
            labels (_type_): _description_
        """
        (
            image_embs,
            clip_input_ids,
            input_ids,
            invalid_ids,
            idx,
            all_features,
            image_sizes,
            inbatch_pack_offset,
            clip_inbatch_pack_offset,
        ) = preds

        self._timers = get_timers()

        if clip_input_ids is not None:
            clip_data_loss, clip_data_loss_info = self._forward_clip_data(
                image_embs,
                clip_input_ids,
                input_ids,
                labels,
                invalid_ids,
                idx,
                all_features,
                inbatch_pack_offset,
                clip_inbatch_pack_offset,
            )
            return clip_data_loss, clip_data_loss_info
        assert not self.train_clip_only
        decoder_data_loss, decoder_data_loss_info = self._forward_decoder_data(
            input_ids,
            labels,
            all_features,
            inbatch_pack_offset,
        )
        return decoder_data_loss, decoder_data_loss_info

    def _encode_text(self, input_ids, inbatch_pack_offset=None):
        """_summary_"""
        text_features = self.text_model(
            input_ids, inbatch_pack_offset=inbatch_pack_offset
        ).astype("float32")
        with paddle.amp.auto_cast(False):
            text_features = F.normalize(text_features.astype("float32"), axis=-1)
        return text_features

    def _forward_decoder_data(
        self,
        decoder_input_ids,
        decoder_labels,
        all_features,
        inbatch_pack_offset,
    ):
        """calculate decoder lm-loss

        Args:
            decoder_input_ids (_type_): _description_
            decoder_labels (_type_): _description_
            all_features (Tensor[num_images,dim]): image_feature

        Returns:
            _type_: _description_
        """

        self._timers and self._timers("decoder-data-decoder-loss-forward").start()
        if self.detach_decoderloss:
            all_features = all_features.detach()
        decoderdata_decoderloss, decoderdata_decoderloss_info = self._cal_decoder_loss(
            all_features,
            decoder_input_ids,
            decoder_labels,
            inbatch_pack_offset=inbatch_pack_offset,
        )
        self._timers and self._timers("decoder-data-decoder-loss-forward").stop()
        decoderdata_decoderloss = (
            self.decoderdata_decoderloss_weight * decoderdata_decoderloss
        )
        return decoderdata_decoderloss, {
            "decoderdata_decoderloss_info": decoderdata_decoderloss_info
        }

    def _forward_clip_data(
        self,
        image_embs,
        clip_input_ids,
        input_ids,
        labels,
        invalid_ids,
        idx,
        all_features,
        inbatch_pack_offset,
        clip_inbatch_pack_offset,
    ):
        """calculate clip loss & decoder lm-loss

        Args:
            image_embs (_type_): _description_
            clip_input_ids (_type_): _description_
            clip_label_ids (_type_): _description_
            invalid_ids (_type_): _description_
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        if idx is None:
            if not self.build_memory_bank:
                # 重置memory bank
                self.accum_image_embs = []
                self.accum_text_embs = []
                self.accum_invalid_ids = []
            self.build_memory_bank = True
            # 刷库阶段
            if image_embs is not None:
                image_embs = image_embs.astype("float32")
                self.accum_image_embs.append(image_embs.detach())

            self._timers and self._timers(
                "clip-data-shuaku-text-encoder-forward"
            ).start()
            text_embs = self._encode_text(
                clip_input_ids, inbatch_pack_offset=clip_inbatch_pack_offset
            )
            self._timers and self._timers(
                "clip-data-shuaku-text-encoder-forward"
            ).stop()
            self.accum_text_embs.append(text_embs.detach())
            assert len(text_embs) == len(
                image_embs
            ), f"#text:{len(text_embs)} != #image:{len(image_embs)}"
            # logger.info(f"shuaku update: T:{text_embs.shape}, I:{image_embs.shape}")
            self.accum_invalid_ids.append(invalid_ids.detach())
            return None, None

        else:
            self.build_memory_bank = False
            # 从labels里面取idx 用于更新memory bank
            idx = idx[0]  # dirty hack coz idx is a tensor

            # re-forward to obtain computational graph for the current micro batch
            if not self.config.freeze_vit:
                # re-forward has been done in image encoder itself
                image_features = image_embs.astype("float32")

            else:
                assert image_embs is None
                image_features = self.accum_image_embs[idx]

            self._timers and self._timers("clip-data-text-encoder-forward").start()
            if not self.config.freeze_text_tower:
                text_features = self._encode_text(
                    clip_input_ids, inbatch_pack_offset=clip_inbatch_pack_offset
                )
            else:
                text_features = self.accum_text_embs[idx]
            self._timers and self._timers("clip-data-text-encoder-forward").stop()

            images_emb_accumulated = paddle.concat(
                self.accum_image_embs[:idx]
                + [image_features]
                + self.accum_image_embs[idx + 1 :]
            )
            images_invalid_accumulated = paddle.concat(
                self.accum_invalid_ids[:idx]
                + [self.accum_invalid_ids[idx]]
                + self.accum_invalid_ids[idx + 1 :]
            )
            texts_emb_accumulated = paddle.concat(
                self.accum_text_embs[:idx]
                + [text_features]
                + self.accum_text_embs[idx + 1 :]
            )
            text_invalid_accumulated = paddle.concat(
                self.accum_invalid_ids[:idx]
                + [self.accum_invalid_ids[idx]]
                + self.accum_invalid_ids[idx + 1 :]
            )

            logit_scale = self.logit_scale.scale.exp()
            # 非mp 0 不算clip loss
            # if self.config.tensor_parallel_rank == 0:
            self._timers and self._timers("clip-data-clip-loss-forward").start()
            clipdata_cliploss, clipdata_cliploss_info = self.clip_loss(
                (
                    images_emb_accumulated,
                    texts_emb_accumulated,
                    logit_scale,
                    images_invalid_accumulated,
                    text_invalid_accumulated,
                    idx,
                    image_features.shape[0],
                )
            )
            self._timers and self._timers("clip-data-clip-loss-forward").stop()

            # else:
            #     # 防止断grad mp 通信error
            #     clipdata_cliploss = (image_features * text_features).sum() * 0.0
            #     clipdata_cliploss_info = {}

            # _broadcast_clipdata_cliploss = clipdata_cliploss.clone().detach()
            # if self.config.tensor_parallel_degree > 1:
            #     dist.broadcast(_broadcast_clipdata_cliploss, self.mp_src_rank, self.mp_group)

            #     if self.config.tensor_parallel_rank != 0:
            #         clipdata_cliploss += _broadcast_clipdata_cliploss

            if self.train_clip_only:
                return clipdata_cliploss, {
                    "clipdata_cliploss_info": clipdata_cliploss_info
                }

            self._timers and self._timers("clip-data-decoder-loss-forward").start()
            if self.detach_decoderloss:
                all_features = all_features.detach()
            clipdata_decoderloss, clipdata_decoderloss_info = self._cal_decoder_loss(
                all_features,
                input_ids,
                labels,
                inbatch_pack_offset=inbatch_pack_offset,
            )
            self._timers and self._timers("clip-data-decoder-loss-forward").stop()

            clip_data_loss = (
                self.clipdata_cliploss_weight * clipdata_cliploss
                + self.clipdata_decoderloss_weight * clipdata_decoderloss
            )
            return clip_data_loss, {
                "clipdata_decoderloss_info": clipdata_decoderloss_info,
                "clipdata_cliploss_info": clipdata_cliploss_info,
            }

    def _cal_decoder_loss(
        self,
        all_features,
        decoder_input_ids,
        decoder_labels,
        attention_mask=None,
        inbatch_pack_offset=None,
    ):
        """TODO decoder loss

        Args:
            all_features (_type_): _description_
            decoder_input_ids (_type_): _description_
        """

        input_ids_without_image_token_id = decoder_input_ids.clone()
        input_ids_without_image_token_id[decoder_input_ids == self.im_patch_id] = 0
        inputs_embeds = self.language_model.get_input_embeddings()(
            input_ids_without_image_token_id
        )
        image_features = self.resampler(all_features)

        if image_features.ndim == 3:
            B, N, C = image_features.shape
            image_features = image_features.reshape([B * N, C])
        image_features = image_features.astype(inputs_embeds.dtype)
        # logger.info(f'#slot:{len(paddle.where(decoder_input_ids == self.im_patch_id))},'
        # f' #imf_fea:{len(image_features)}')
        num_imgage_in_ids = len(paddle.where(decoder_input_ids == self.im_patch_id)[0])
        assert num_imgage_in_ids == len(image_features), (
            f"num_image_in_ids:{num_imgage_in_ids} != len(images):{len(image_features)}, "
            f"ids={decoder_input_ids}, does your ids truncated?"
        )

        inputs_embeds[decoder_input_ids == self.im_patch_id] = image_features
        # eb decoder model
        if (
            isinstance(self.language_model, ErnieForCausalLM)
            and inbatch_pack_offset is not None
        ):
            attention_mask = None
        # logger.info(f"decoder input-shape: {inputs_embeds.shape}")
        logits = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            inbatch_pack_offset=inbatch_pack_offset,
            return_dict=True,
        ).logits
        decoder_loss, decoder_loss_info = self.decoder_loss(logits, decoder_labels)
        if isinstance(decoder_loss_info, paddle.Tensor):
            decoder_loss_info = {"loss_sum": decoder_loss_info}
        decoder_loss_info["loss"] = decoder_loss.clone().detach()
        return decoder_loss, decoder_loss_info


class ImageEncoder(EVAVisionTransformer):
    """_summary_

    Args:
        EVAVisionTransformer (_type_): _description_
    """

    model_config_file = "config.json"
    config_class = ImageEncoderConfig
    resource_files_names = {"model_state": "model_state.pdparams"}

    def __init__(
        self,
        config,
        criterion=None,
    ):
        super().__init__(config)
        self.criterion = criterion

    def encode_image(
        self, images, image_sizes=None, position_ids=None, return_all_features=False
    ):
        """
        dummy
        """
        # return_all_feature = True , 返回 image_features, image_sizes, all_features, hiddens
        # return_all_feature = False , 返回 image_features, image_sizes
        ret = list(
            super().forward(
                images,
                image_sizes,
                position_ids=position_ids,
                return_all_features=return_all_features,
            )
        )
        with paddle.amp.auto_cast(False):
            image_features = F.normalize(ret[0].astype("float32"), axis=-1)
        ret[0] = image_features
        return ret

    def encode_text(self, input_ids):
        """
        dummy
        """
        return self.criterion._encode_text(input_ids)

    def forward(
        self,
        images,
        image_sizes=None,
        clip_input_ids=None,
        input_ids=None,
        labels=None,
        image_position_ids=None,
        inbatch_pack_offset=None,
        clip_inbatch_pack_offset=None,
        invalid=None,
        idx=None,
        data_id=None,
        src_id=None,
        part_id=None,
        example_id=None,
        data_type=None,
    ):
        """
        Args:
            images :Tensor[B,C,H,W] or [num_tokens, 3 * patch_size * patch_siz], raw-pixels.
               如果输入为固定分辨率，x.shape == [B,C,H,W]
               如果输入为 adaptive 分辨率，需要提前 patchify。x.shape == [num_tokens, 3 * patch_size * patch_size]
            image_sizes :Tensor[num_image, 2], int64, defalts to None,
                list of [patch_size_H, patch_size_W] of all images in `x`, 固定分辨率下为 None。
            position_ids: Optional, Tensor[num_tokens], position ids for each image patches.
            clip_input_ids : Optional, Tensor[B,S], int64, inputs to text tower
            input_ids: Optional, Tensor[B,S] inputs to decoder language model

        Returns:
            x: Tensor[num_sample, C], cls_feature
            all_feature: Tensor[num_tokens,C], optional, all token feature exclude cls.
        """
        self._timers = get_timers()

        if clip_input_ids is not None:
            # clip 数据的时候 进行patch dropout
            self.patch_dropout.prob = self.config.patch_dropout
        else:
            # 否则不做patch dropout
            self.patch_dropout.prob = 0.0
        num_imgage_in_ids = len(
            paddle.where(input_ids == self.criterion.im_patch_id)[0]
        )
        # logger.info(
        #     f"im encoder: image_sizes:{image_sizes}, num_imgage_in_ids:{num_imgage_in_ids} "
        #     f"image_shape={images.shape}, ids_shape={input_ids.shape},  "
        #     f"ids_offset={inbatch_pack_offset.shape if inbatch_pack_offset is not None else None}, "
        #     f"clip_ids_shape={clip_input_ids.shape if clip_input_ids is not None else None}, "
        #     f"clip_offset={clip_inbatch_pack_offset.shape if clip_inbatch_pack_offset is not None else None}, "
        #     f"invalid={invalid}, idx={idx}"
        # )

        if clip_input_ids is None:
            # decoder 数据
            self._timers and self._timers("decoder-data-vit-forward").start()
            image_features, image_sizes, all_features, hiddens = self.encode_image(
                images,
                image_sizes=image_sizes,
                position_ids=image_position_ids,
                return_all_features=True,
            )
            all_features = hiddens[-2]  # 取倒数第二层
            self._timers and self._timers("decoder-data-vit-forward").stop()
        else:
            # clip 数据
            if idx is not None:
                self._timers and self._timers("clip-data-vit-forward").start()
                if not self.config.freeze_vit:
                    # 需要前向拿梯度
                    image_features, image_sizes, all_features, hiddens = (
                        self.encode_image(
                            images,
                            image_sizes=image_sizes,
                            position_ids=image_position_ids,
                            return_all_features=True,
                        )
                    )
                    all_features = hiddens[-2]  # 取倒数第二层
                else:
                    # 直接在criterion里从库里拿
                    assert (
                        self.criterion.train_clip_only
                    ), "不训vit的时候，暂只支持纯clip训练"
                    image_features, image_sizes, all_features = None, None, None
                self._timers and self._timers("clip-data-vit-forward").stop()
            else:
                # 刷库阶段
                self._timers and self._timers("clip-data-shuaku-forward").start()
                image_features, image_sizes = self.encode_image(
                    images,
                    image_sizes,
                    position_ids=image_position_ids,
                    return_all_features=False,
                )
                self._timers and self._timers("clip-data-shuaku-forward").stop()
                all_features = None

        loss = self.criterion(
            (
                image_features,
                clip_input_ids,
                input_ids,
                invalid,
                idx,
                all_features,
                image_sizes,
                inbatch_pack_offset,
                clip_inbatch_pack_offset,
            ),
            labels,
        )
        return loss

    def set_state_dict(self, state_dict, *args, **kwargs):
        """set state dict for log"""
        ret = super().set_state_dict(state_dict, *args, **kwargs)
        logger.info(f"image_encoder load state-dict---- {ret}")
        return ret
