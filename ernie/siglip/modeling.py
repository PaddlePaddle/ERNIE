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

# This file is based on https://github.com/Kwai-Keye/Keye/blob/main/keye-vl-8b-preview/modeling_keye.py
# Original header:
# Copyright 2025 The Keye Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import collections
from typing import List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute
from paddle._typing import ParamAttrLike
from paddle.nn.functional.flash_attention import flashmask_attention
from paddleformers.transformers.model_utils import PretrainedModel
from paddleformers.transformers.model_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
)

from .activation import ACT2FN
from .configuration import PaddleOCRVisionConfig


def rotate_half(x):
    Dh = x.shape[-1]
    x1 = x[..., : Dh // 2]
    x2 = x[..., Dh // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


def _ensure_cos_sin_dim(cos, sin, dim_needed):
    last = cos.shape[-1]
    if last == dim_needed:
        return cos, sin
    elif last * 2 == dim_needed:
        cos = paddle.concat([cos, cos], axis=-1)
        sin = paddle.concat([sin, sin], axis=-1)
        return cos, sin
    else:
        raise ValueError(
            f"Unexpected cos/sin last-dim: {last}, expected {dim_needed} or {dim_needed//2}"
        )


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q = q.astype("float32")
    k = k.astype("float32")

    Dh = q.shape[-1]
    cos = cos.astype("float32")
    sin = sin.astype("float32")
    cos, sin = _ensure_cos_sin_dim(cos, sin, Dh)

    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(orig_q_dtype), k_embed.astype(orig_k_dtype)


def eager_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = paddle.matmul(query, key.transpose((0, 1, 3, 2))) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = paddle.matmul(attn_weights, value)
    attn_output = attn_output.transpose((0, 2, 1, 3)).contiguous()

    return attn_output, attn_weights


class SiglipAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim
        self.scale = self.head_dim**-0.5
        self.dropout = getattr(config, "attention_dropout", 0.0)
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def attn_func(
        self,
        queries: paddle.Tensor,
        keys: paddle.Tensor,
        values: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        cu_seqlens: Optional[List[paddle.Tensor]] = None,
    ):
        if self.config.use_flash_attention:

            # FlashAttentionVarlen cu_seqlens to FlashMask mask
            cu_seqlens_rm_first = cu_seqlens[1:]
            cu_seqlens_rm_last = cu_seqlens[:-1]
            repeats = cu_seqlens_rm_first - cu_seqlens_rm_last

            startend_row_indices_lts = paddle.repeat_interleave(
                cu_seqlens_rm_first, repeats
            ).reshape([1, 1, -1, 1])
            startend_row_indices_ute = paddle.repeat_interleave(
                cu_seqlens_rm_last, repeats
            ).reshape([1, 1, -1, 1])
            startend_row_indices = paddle.concat(
                [startend_row_indices_lts, startend_row_indices_ute], axis=-1
            )

            attn_output = flashmask_attention(
                queries,
                keys,
                values,
                startend_row_indices=startend_row_indices,
                causal=self.is_causal,
                dropout=0.0 if not self.training else self.dropout,
            )
            attn_weights = None
        else:
            attn_output, attn_weights = eager_attention_forward(
                self,
                queries,
                keys,
                values,
                attention_mask,
                is_causal=self.is_causal,
                scaling=self.scale,
                dropout=0.0 if not self.training else self.dropout,
            )

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: paddle.Tensor,  # [B, L, D]
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        cu_seqlens: Optional[List[paddle.Tensor]] = None,
        rope_emb: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,  # (cos, sin)
    ):

        B, L, D = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # [B, L, H, Dh] -> [B, H, L, Dh]

        q = q.reshape([B, L, self.num_heads, self.head_dim])
        k = k.reshape([B, L, self.num_heads, self.head_dim])
        v = v.reshape([B, L, self.num_heads, self.head_dim])
        if rope_emb is not None:
            assert cu_seqlens is not None, "Rope support flash attn only."
            cos, sin = rope_emb
            # if self.config.use_flash_attention:
            #     queries, keys = apply_rotary_pos_emb_flashatt(queries, keys, cos, sin)
            # else:
            #     queries, keys = apply_rotary_pos_emb_vision(queries, keys, cos, sin)
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        if not self.config.use_flash_attention:
            q = q.transpose([0, 2, 1, 3])
            k = k.transpose([0, 2, 1, 3])
            v = v.transpose([0, 2, 1, 3])

        has_gradient = not (q.stop_gradient and k.stop_gradient and v.stop_gradient)

        if (
            self.config.recompute
            and self.config.recompute_granularity == "core_attn"
            and has_gradient
        ):
            attn_output, attn_weights = recompute(
                self.attn_func,
                q,
                k,
                v,
                attention_mask,
                cu_seqlens,
                use_reentrant=self.config.recompute_use_reentrant,
            )

        else:
            attn_output, attn_weights = self.attn_func(
                q,
                k,
                v,
                attention_mask,
                cu_seqlens,
            )

        attn_output = attn_output.reshape([B, L, D]).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class SiglipVisionEmbeddings(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # 1152
        self.image_size = config.image_size  # 384
        self.patch_size = config.patch_size  # 14

        # 注意：Paddle 要用 "VALID" 或 0
        self.patch_embedding = nn.Conv2D(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="VALID",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2  # 729
        self.num_positions = self.num_patches
        self.cache_position_embedding = dict()
        self.cache_position_count = dict()
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.packing_position_embedding = nn.Embedding(32768, self.embed_dim)

        self.register_buffer(
            "position_ids",
            paddle.arange(self.num_positions).unsqueeze(0),
            persistable=False,
        )

    def interpolate_pos_encoding(
        self, embeddings, height: int, width: int, is_after_patchify: bool = False
    ):

        num_positions = self.position_embedding.weight.shape[0]

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        dim = embeddings.shape[-1]

        if is_after_patchify:
            new_height = height
            new_width = width
        else:
            new_height = height // self.patch_size
            new_width = width // self.patch_size

        sqrt_num_positions = paddle.to_tensor(num_positions**0.5, dtype=paddle.int64)
        patch_pos_embed = patch_pos_embed.reshape(
            (1, sqrt_num_positions, sqrt_num_positions, dim)
        )
        patch_pos_embed = patch_pos_embed.transpose((0, 3, 1, 2))

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.transpose((0, 2, 3, 1)).reshape((1, -1, dim))
        return patch_pos_embed

    @staticmethod
    def flatten_list(image_grid_thw):
        tmp_image_grid_thw = list()
        for image_grid in image_grid_thw:
            if isinstance(image_grid, list):
                tmp_image_grid_thw.extend(image_grid)
            else:
                tmp_image_grid_thw.append(image_grid)
        return tmp_image_grid_thw

    def fetch_position_embedding_lfu_cache(self, embeddings, h, w, max_cache=20):
        grid = (h, w)
        if grid in self.cache_position_embedding:
            self.cache_position_count[grid] += 1
            return self.cache_position_embedding[grid]

        if len(self.cache_position_embedding) >= max_cache:
            min_hit_grid = min(
                self.cache_position_count, key=self.cache_position_count.get
            )
            self.cache_position_count.pop(min_hit_grid)
            self.cache_position_embedding.pop(min_hit_grid)

        position_embedding = self.interpolate_pos_encoding(embeddings, h, w, True)
        self.cache_position_count[grid] = 1
        self.cache_position_embedding[grid] = position_embedding
        return position_embedding

    def forward(
        self,
        pixel_values: paddle.Tensor,  # [B, L, C, H, W]
        position_ids: Optional[paddle.Tensor] = None,  # [B or 1, S]
        image_grid_thw: Optional[
            List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]
        ] = None,
        interpolate_pos_encoding: bool = False,
    ) -> paddle.Tensor:
        if pixel_values.dim() == 5:
            assert position_ids is not None
            from einops import rearrange

            batch_size, squence_len, channel, height, width = pixel_values.shape
            target_dtype = self.patch_embedding.weight.dtype
            pixel_values = rearrange(pixel_values, "b l c h w -> (b l) c h w")
            patch_embeds = self.patch_embedding(
                pixel_values.to(dtype=target_dtype)
            )  # shape = [*, width, grid, grid]
            embeddings = patch_embeds.flatten(-2).squeeze(-1)
            embeddings = rearrange(
                embeddings, "(b l) d -> b l d", b=batch_size, l=squence_len
            )

            # todo: not dubug
            if interpolate_pos_encoding and image_grid_thw is not None:
                flatten_image_grid_thw = self.flatten_list(image_grid_thw)
                assert batch_size == 1
                start = 0

                assert (
                    sum([np.prod(x) for x in flatten_image_grid_thw])
                    == embeddings.shape[1]
                ), (flatten_image_grid_thw, embeddings.shape)
                embeddings = embeddings.squeeze(0)
                tmp_embeddings = list()
                for image_grid in image_grid_thw:
                    t, h, w = image_grid
                    end = start + t * h * w
                    image_embeddings = embeddings[int(start) : int(end), :]
                    position_embedding = (
                        self.interpolate_pos_encoding(image_embeddings, h, w, True)
                        .squeeze(0)
                        .tile((t, 1))
                    )
                    image_embeddings = image_embeddings + position_embedding
                    tmp_embeddings.append(image_embeddings)
                    start = end
                embeddings = paddle.concat(tmp_embeddings, axis=0).unsqueeze(0)
            else:
                embeddings = embeddings + self.packing_position_embedding(position_ids)
            return embeddings
        else:
            raise NotImplementedError(str(pixel_values.shape))


class SiglipMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = paddle.nn.LayerNorm(
            self.embed_dim, epsilon=config.layer_norm_eps
        )
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = paddle.nn.LayerNorm(
            self.embed_dim, epsilon=config.layer_norm_eps
        )
        self.mlp = SiglipMLP(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        cu_seqlens=None,
        rope_emb=None,
    ):

        residual = hidden_states
        ############################
        ln1_out = self.layer_norm1(hidden_states)

        x, attn_w = self.self_attn(
            hidden_states=ln1_out,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            cu_seqlens=cu_seqlens,
            rope_emb=rope_emb,
        )

        hs_post_attn = residual + x

        residual = hs_post_attn
        ln2_out = self.layer_norm2(residual)

        mlp_out = self.mlp(ln2_out)

        hidden_states_out = residual + mlp_out

        outputs = (hidden_states_out,)
        if output_attentions:
            outputs += (attn_w,)
        return outputs


class SigLIPRotaryEmbedding(nn.Layer):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.rope_init()

    def rope_init(self):
        arange = paddle.arange(0, self.dim, 2, dtype="float32")
        inv_freq = 1.0 / (self.theta ** (arange / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    def forward(self, seqlen: int) -> paddle.Tensor:
        seq = paddle.arange(seqlen, dtype=self.inv_freq.dtype)
        freqs = paddle.outer(seq, self.inv_freq)
        return freqs


class SiglipEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.layers = nn.LayerList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.rotary_pos_emb = SigLIPRotaryEmbedding(head_dim // 2)
        self.gradient_checkpointing = False

    @staticmethod
    def flatten_list(image_grid_thw):
        tmp_image_grid_thw = list()
        for image_grid in image_grid_thw:
            if isinstance(image_grid, list):
                tmp_image_grid_thw.extend(image_grid)
            else:
                tmp_image_grid_thw.append(image_grid)
        return tmp_image_grid_thw

    def build_window_index(self, image_grid, window_size):
        """
        返回：
          window_indices: int64 [sum(t*h*w_valid)]
          cu_seqlens_within_windows: int32 [num_windows_total*t]，首位补 0 的前缀和
        """
        from einops import rearrange

        window_indices = list()
        pad_values = -100
        start_window_index = 0
        cu_seqlens_within_windows = list()

        for t, h, w in map(int, image_grid):
            window_index = paddle.arange(t * h * w).reshape((t, h, w))
            pad_h = (-h) % window_size
            pad_w = (-w) % window_size
            assert pad_h >= 0 and pad_w >= 0, (pad_h, pad_w)
            window_index = F.pad(window_index, (0, pad_w, 0, pad_h), value=pad_values)
            window_index = rearrange(
                window_index,
                "t (h p1) (w p2) -> t (h w) (p1 p2)",
                p1=window_size,
                p2=window_size,
            )
            window_seqlens = (window_index != pad_values).long().sum(-1).reshape(-1)
            window_index = window_index.reshape(-1)
            window_index = window_index[window_index != pad_values]
            window_indices.append(window_index + start_window_index)
            cu_seqlens_within_windows.append(
                window_seqlens.cumsum(0) + start_window_index
            )
            start_window_index += t * h * w
        window_indices = paddle.concat(window_indices, axis=0)
        cu_seqlens_within_windows = paddle.concat(cu_seqlens_within_windows, axis=0)
        cu_seqlens_within_windows = F.pad(
            cu_seqlens_within_windows, (1, 0), value=0
        ).astype("int32")
        return window_indices, cu_seqlens_within_windows

    def forward(
        self,
        inputs_embeds: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cu_seqlens: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[
            List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]
        ] = None,
        height_position_ids: Optional[paddle.Tensor] = None,
        width_position_ids: Optional[paddle.Tensor] = None,
        use_rope: Optional[bool] = False,
        window_size: Optional[int] = -1,
        vision_or_text: str = "vision",
    ):

        vision_or_text = "vision"
        assert vision_or_text in ["vision", "text"]
        use_window_attn = window_size > 0 and vision_or_text == "vision"
        use_rope = (use_rope is True) and (vision_or_text == "vision")
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds
        attention_mask = (
            attention_mask.to(inputs_embeds.dtype)
            if attention_mask is not None
            else None
        )

        if use_rope is True:
            flatten_image_grid_thw = self.flatten_list(image_grid_thw)
            assert (
                sum([np.prod(x) for x in flatten_image_grid_thw])
                == hidden_states.shape[1]
            ), (flatten_image_grid_thw, hidden_states.shape)

            if width_position_ids is None or height_position_ids is None:
                split_hids = list()
                split_wids = list()
                for t, h, w in flatten_image_grid_thw:
                    t, h, w = map(int, (t, h, w))
                    image_pids = paddle.arange(t * h * w) % (h * w)
                    sample_hids = image_pids // w
                    sample_wids = image_pids % w
                    split_hids.append(sample_hids)
                    split_wids.append(sample_wids)
                width_position_ids = paddle.concat(split_wids, axis=0)
                height_position_ids = paddle.concat(split_hids, axis=0)

            window_indices, cu_seqlens_within_windows = None, None

            if use_window_attn:
                window_indices, cu_seqlens_within_windows = self.build_window_index(
                    flatten_image_grid_thw, window_size
                )
                reversed_window_indices = window_indices.argsort()
                height_position_ids = height_position_ids[window_indices]
                width_position_ids = width_position_ids[window_indices]

            pids = paddle.stack(
                [height_position_ids, width_position_ids], axis=-1
            ).astype(paddle.int64)
            max_grid_size = pids.max() + 1
            rope_emb_max_grid = self.rotary_pos_emb(max_grid_size)

            rope_emb = rope_emb_max_grid[pids].flatten(1)

            rope_emb = rope_emb.tile((1, 2))
            rope_emb = (rope_emb.cos(), rope_emb.sin())

        else:
            rope_emb = None

            window_indices, cu_seqlens_within_windows = None, None

            if use_window_attn:
                flatten_image_grid_thw = self.flatten_list(image_grid_thw)
                assert (
                    sum(
                        [
                            np.prod(x.astype("float32").cpu().numpy())
                            for x in flatten_image_grid_thw
                        ]
                    )
                    == hidden_states.shape[1]
                ), (flatten_image_grid_thw, hidden_states.shape)

                window_indices, cu_seqlens_within_windows = self.build_window_index(
                    flatten_image_grid_thw, window_size
                )
                reversed_window_indices = window_indices.argsort()

        if use_window_attn:
            assert cu_seqlens_within_windows is not None
            attn_cu_seqlens = cu_seqlens_within_windows
            hidden_states = hidden_states[:, window_indices, :]
        else:
            attn_cu_seqlens = cu_seqlens

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (
                    (hidden_states[:, reversed_window_indices, :],)
                    if use_window_attn
                    else (hidden_states,)
                )
            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute
                and self.config.recompute_granularity == "full"
                and has_gradient
            ):
                layer_outputs = recompute(
                    encoder_layer,
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                    cu_seqlens=attn_cu_seqlens,
                    rope_emb=rope_emb,
                    use_reentrant=self.config.recompute_use_reentrant,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                    cu_seqlens=attn_cu_seqlens,
                    rope_emb=rope_emb,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if use_window_attn:
            hidden_states = hidden_states[:, reversed_window_indices, :]
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout (float, optional): The dropout probability used on attention
            weights to drop some attention targets. 0 for no dropout. Default 0
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Default False.
        weight_attr(ParamAttr|None, optional):  To specify the weight parameter property.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :code:`ParamAttr` .
        bias_attr (ParamAttr|bool|None, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            If it is set to False, this layer will not have trainable bias parameter.
            See usage for details in :code:`ParamAttr` .

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> # encoder input: [batch_size, sequence_length, d_model]
            >>> query = paddle.rand((2, 4, 128))
            >>> # self attention mask: [batch_size, num_heads, query_len, query_len]
            >>> attn_mask = paddle.rand((2, 2, 4, 4))
            >>> multi_head_attn = paddle.nn.MultiHeadAttention(128, 2)
            >>> output = multi_head_attn(query, None, None, attn_mask=attn_mask)
            >>> print(output.shape)
            [2, 4, 128]
    """

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    embed_dim: int
    kdim: int
    vdim: int
    num_heads: int
    head_dim: int
    dropout: float
    need_weights: bool

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        kdim: int | None = None,
        vdim: int | None = None,
        need_weights: bool = False,
        weight_attr: ParamAttrLike | None = None,
        bias_attr: ParamAttrLike | None = None,
    ) -> None:
        super().__init__()

        assert (
            embed_dim > 0
        ), f"Expected embed_dim to be greater than 0, but received {embed_dim}"
        assert (
            num_heads > 0
        ), f"Expected num_heads to be greater than 0, but received {num_heads}"

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        # self.q_proj = Linear(
        #     embed_dim, embed_dim, weight_attr, bias_attr=bias_attr
        # )
        # self.k_proj = Linear(
        #     self.kdim, embed_dim, weight_attr, bias_attr=bias_attr
        # )
        # self.v_proj = Linear(
        #     self.vdim, embed_dim, weight_attr, bias_attr=bias_attr
        # )

        # register parameters to keep consistent with torch.nn.MultiHeadAttention
        self.in_proj_weight = self.create_parameter(
            shape=[3 * embed_dim, embed_dim],
            default_initializer=nn.initializer.XavierUniform(),
        )

        self.in_proj_bias = self.create_parameter(
            shape=[3 * embed_dim], default_initializer=nn.initializer.Constant(0.0)
        )

        self.out_proj = nn.Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr
        )

    def forward(
        self,
        query,
        key=None,
        value=None,
        key_padding_mask=None,
        attn_mask=None,
        cache=None,
    ):

        raise NotImplementedError(
            "Please refer to paddle.nn.MultiHeadAttention for more details https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/layer/transformer.py#L132"
        )


class SiglipMultiheadAttentionPoolingHead(nn.Layer):
    """Multihead Attention Pooling."""

    def __init__(self, config):
        super().__init__()

        self.probe = self.create_parameter(
            shape=(1, 1, config.hidden_size),
            default_initializer=paddle.nn.initializer.Normal(),
        )
        self.attention = MultiHeadAttention(
            config.hidden_size, config.num_attention_heads
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_state, key_padding_mask=None):
        batch_size = hidden_state.shape[0]
        probe = self.probe.tile((batch_size, 1, 1))

        hidden_state = self.attention(
            probe, hidden_state, hidden_state, key_padding_mask=key_padding_mask
        )[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SiglipVisionTransformer(nn.Layer):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, epsilon=config.layer_norm_eps)
        self.use_head = (
            True if not hasattr(config, "vision_use_head") else config.vision_use_head
        )
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)

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
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        hidden_states = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
        )

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            image_grid_thw=image_grid_thw,
            use_rope=use_rope,
            height_position_ids=height_position_ids,
            width_position_ids=width_position_ids,
            window_size=window_size,
            vision_or_text="vision",
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        if return_pooler_output is True:
            if sample_indices is not None:
                assert self.use_head is True
                dim = last_hidden_state.shape[-1]
                sample_hidden_state_list = list()

                hidden_state = last_hidden_state.squeeze(0)
                sample_index = sample_indices
                unique_sample_index = (
                    paddle.unique(sample_index).sort().values.unbind(0)
                )
                unique_sample_index = list(unique_sample_index)
                if len(unique_sample_index) > 0 and unique_sample_index[0] == -1:
                    unique_sample_index = unique_sample_index[1:]
                for sample_idx in unique_sample_index:
                    token_indices = (sample_index == sample_idx).nonzero().flatten()
                    sample_hidden_state = hidden_state[token_indices]
                    sample_hidden_state_list.append(sample_hidden_state)

                if not vision_return_embed_list:
                    max_length = max(
                        [_state.shape[0] for _state in sample_hidden_state_list]
                    )
                    tmp_sample_hidden_state_list = list()
                    padding_mask = list()
                    for idx, _state in enumerate(sample_hidden_state_list):
                        padding_length = max_length - _state.shape[0]
                        mask = _state.new_zeros(size=(max_length,), dtype=paddle.int64)
                        mask[-padding_length:] = 1
                        padding_mask.append(mask)
                        padding = _state.new_zeros(size=(padding_length, dim))
                        new_state = paddle.concat([_state, padding], axis=0)
                        tmp_sample_hidden_state_list.append(new_state)
                    sample_hidden_state = paddle.stack(
                        tmp_sample_hidden_state_list, axis=0
                    )
                    padding_mask = (
                        paddle.stack(padding_mask, axis=0)
                        .astype("float32")
                        .to(last_hidden_state.dtype)
                    )
                    pooler_output = self.head(
                        sample_hidden_state, key_padding_mask=padding_mask
                    )
                else:
                    pooler_output = list()
                    for state in sample_hidden_state_list:
                        sample_pooler_output = self.head(state.unsqueeze(0))
                        pooler_output.append(sample_pooler_output)
                    pooler_output = paddle.concat(pooler_output, axis=0)
                    sample_hidden_state = sample_hidden_state_list

                return BaseModelOutputWithPooling(
                    last_hidden_state=sample_hidden_state,
                    pooler_output=pooler_output,
                    hidden_states=encoder_outputs.hidden_states,
                    attentions=encoder_outputs.attentions,
                )
            else:
                pooler_output = self.head(last_hidden_state) if self.use_head else None

            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooler_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

        sample_hidden_state = list()
        assert cu_seqlens is not None
        for i in range(cu_seqlens.shape[0] - 1):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]
            tensor = last_hidden_state[:, start:end, :].squeeze(0)
            sample_hidden_state.append(tensor)

        return BaseModelOutputWithPooling(
            last_hidden_state=sample_hidden_state,
            pooler_output=None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SiglipPreTrainedModel(PretrainedModel):
    config_class = PaddleOCRVisionConfig
    base_model_prefix = "siglip"

    _no_split_modules = [
        "SiglipTextEmbeddings",
        "SiglipEncoderLayer",
        "SiglipVisionEmbeddings",
        "SiglipMultiheadAttentionPoolingHead",
    ]


class SiglipVisionModel(SiglipPreTrainedModel):
    config_class = PaddleOCRVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__(config)

        self.vision_model = SiglipVisionTransformer(config)

    def get_input_embeddings(self) -> nn.Layer:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        sample_indices=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        position_ids=None,
        vision_return_embed_list: Optional[bool] = False,
        image_grid_thw: Optional[
            List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]
        ] = None,
        cu_seqlens=None,
        return_pooler_output: Optional[bool] = True,
        use_rope: Optional[bool] = False,
        window_size: Optional[bool] = -1,
    ) -> BaseModelOutputWithPooling:
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            position_ids=position_ids,
            vision_return_embed_list=vision_return_embed_list,
            image_grid_thw=image_grid_thw,
            sample_indices=sample_indices,
            cu_seqlens=cu_seqlens,
            return_pooler_output=return_pooler_output,
            use_rope=use_rope,
            window_size=window_size,
        )
