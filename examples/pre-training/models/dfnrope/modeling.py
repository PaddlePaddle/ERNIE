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
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute
from paddle.nn.functional.flash_attention import flashmask_attention
from paddleformers.transformers.model_utils import PretrainedModel

from .configuration import DFNRopeVisionTransformerConfig
from .activation import ACT2FN
from models.ernie_moe.modeling_pp import get_hcg

from models.sequence_parallel_utils import (
    _AllToAll,
)

logger = logging.getLogger(__name__)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


def apply_rotary_pos_emb_vision(
    tensor: paddle.Tensor, freqs: paddle.Tensor
) -> paddle.Tensor:
    """_summary_

    Args:
        tensor (paddle.Tensor): _description_
        freqs (paddle.Tensor): _description_

    Returns:
        paddle.Tensor: _description_
    """
    orig_dtype = tensor.dtype

    with paddle.amp.auto_cast(False):
        tensor = tensor.astype(dtype="float32")
        cos = freqs.cos()
        sin = freqs.sin()
        cos = (
            cos.unsqueeze(1)
            .tile(repeat_times=[1, 1, 2])
            .unsqueeze(0)
            .astype(dtype="float32")
        )
        sin = (
            sin.unsqueeze(1)
            .tile(repeat_times=[1, 1, 2])
            .unsqueeze(0)
            .astype(dtype="float32")
        )
        # logger.info(f"[DEBUG] freqs: {freqs.shape} tensor: {tensor.shape}, cos: {cos.shape},
        # sin: {sin.shape}, rotate_half(tensor): {rotate_half(tensor).shape}")
        output = tensor * cos + rotate_half(tensor) * sin
    output = paddle.cast(output, orig_dtype)
    return output


def qkv_reshard_head(tensor, group):

    parallelism = group.nranks
    qkv_seqlen, head_num, head_dim = tensor.shape
    tensor = tensor.transpose(perm=[1, 0, 2]).contiguous()
    out = _AllToAll.apply(tensor, group)
    out = paddle.split(out, parallelism, axis=0)
    output_q = []
    output_k = []
    output_v = []
    for output_i in out:
        outout = output_i.transpose(perm=[1, 0, 2]).contiguous()
        output = paddle.split(outout, 3, axis=0)
        output_q.append(output[0])
        output_k.append(output[1])
        output_v.append(output[2])
    q = paddle.concat(output_q, axis=0)
    k = paddle.concat(output_k, axis=0)
    v = paddle.concat(output_v, axis=0)
    return q, k, v


class VisionFlashAttention2(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=True)
        self.proj = nn.Linear(dim, dim)
        self.head_dim = dim // num_heads  # must added

    def forward(
        self,
        hidden_states: paddle.Tensor,
        startend_row_indices: paddle.Tensor,
        rotary_pos_emb: paddle.Tensor = None,
        attn_sep=False,
    ) -> paddle.Tensor:
        """_summary_

        Args:
            hidden_states (paddle.Tensor): _description_
            cu_seqlens (paddle.Tensor): _description_
            rotary_pos_emb (paddle.Tensor, optional): _description_. Defaults to None.

        Returns:
            paddle.Tensor: _description_
        """
        seq_length = tuple(hidden_states.shape)[0]
        qkv = (
            self.qkv(hidden_states)
            .reshape([seq_length, 3, self.num_heads, -1])
            .transpose(perm=[1, 0, 2, 3])
        )
        q, k, v = qkv.unbind(axis=0)

        if attn_sep:
            hcg = get_hcg()
            assert hcg.get_model_parallel_world_size() > 1
            mp_group = hcg.get_model_parallel_group()
            qkv = paddle.concat([q, k, v], axis=0)
            q, k, v = qkv_reshard_head(qkv, mp_group)
            seq_length = q.shape[0]

        q = apply_rotary_pos_emb_vision(q.unsqueeze(axis=0), rotary_pos_emb).squeeze(
            axis=0
        )
        k = apply_rotary_pos_emb_vision(k.unsqueeze(axis=0), rotary_pos_emb).squeeze(
            axis=0
        )

        attn_output = flashmask_attention(
            q.astype("bfloat16").unsqueeze(0),
            k.astype("bfloat16").unsqueeze(0),
            v.astype("bfloat16").unsqueeze(0),
            startend_row_indices=startend_row_indices,
            causal=False,  # 视觉是双向 Attention
        )
        attn_output = attn_output.reshape([seq_length, -1])

        if attn_sep:
            out = _AllToAll.apply(attn_output, mp_group)
            out = paddle.split(out, mp_group.nranks, axis=0)
            attn_output = paddle.concat(out, axis=1)
        attn_output = attn_output.astype(paddle.float32)
        attn_output = self.proj(attn_output)
        return attn_output


class PatchEmbed(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Linear(
            in_channels * patch_size * patch_size, embed_dim, bias_attr=False
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """_summary_

        Args:
            hidden_states (paddle.Tensor): _description_

        Returns:
            paddle.Tensor: _description_
        """
        target_dtype = self.proj.weight.dtype

        hidden_states = self.proj(paddle.cast(hidden_states, dtype=target_dtype))
        return hidden_states


class VisionMlp(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> paddle.Tensor:
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            paddle.Tensor: _description_
        """
        return self.fc2(self.act(self.fc1(x)))


class VisionRotaryEmbedding(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        """_summary_

        Args:
            dim (int): _description_
            theta (float, optional): _description_. Defaults to 10000.0.
        """
        super().__init__()
        self.inv_freq = 1.0 / theta ** (
            paddle.arange(start=0, end=dim, step=2, dtype="float32") / dim
        )

    def forward(self, seqlen: int) -> paddle.Tensor:
        """_summary_

        Args:
            seqlen (int): _description_

        Returns:
            paddle.Tensor: _description_
        """
        seq = paddle.arange(seqlen).cast(self.inv_freq.dtype)
        freqs = paddle.outer(x=seq, y=self.inv_freq)
        return freqs


class DFNRopeVisionBlock(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        """_summary_

        Args:
            config (_type_): _description_
            attn_implementation (str, optional): _description_. Defaults to "sdpa".
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, epsilon=1e-6)
        self.norm2 = nn.LayerNorm(config.embed_dim, epsilon=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = VisionFlashAttention2(config.embed_dim, num_heads=config.num_heads)
        self.mlp = VisionMlp(
            dim=config.embed_dim,
            hidden_dim=mlp_hidden_dim,
            hidden_act=config.hidden_act,
        )
        self.config = config

    def forward(
        self, hidden_states, startend_row_indices, rotary_pos_emb, attn_sep=False
    ) -> paddle.Tensor:
        """_summary_

        Args:
            hidden_states (_type_): _description_
            cu_seqlens (_type_): _description_
            rotary_pos_emb (_type_): _description_

        Returns:
            paddle.Tensor: _description_
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            startend_row_indices=startend_row_indices,
            rotary_pos_emb=rotary_pos_emb,
            attn_sep=attn_sep,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class PatchMerger(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        """_summary_

        Args:
            dim (int): _description_
            context_dim (int): _description_
            spatial_merge_size (int, optional): _description_. Defaults to 2.
        """
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, epsilon=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """_summary_

        Args:
            x (paddle.Tensor): _description_

        Returns:
            paddle.Tensor: _description_
        """
        x = self.mlp(self.ln_q(x).reshape([-1, self.hidden_size]))
        return x


class DFNRopeVisionTransformerPretrainedModel(PretrainedModel):
    """_summary_

    Args:
        PretrainedModel (_type_): _description_

    Returns:
        _type_: _description_
    """

    base_model_prefix = "vision_model"
    config_class = DFNRopeVisionTransformerConfig

    def __init__(self, config) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.LayerList(
            [DFNRopeVisionBlock(config) for _ in range(config.depth)]
        )

        assert (
            config.hidden_size == config.embed_dim
        ), "in DFNRope, vit's config.hidden must be equal to config.embed_dim"
        # self.merger = PatchMerger(dim=config.hidden_size, context_dim=config.embed_dim)
        self.ln = nn.LayerNorm(config.hidden_size, epsilon=1e-6)

    def get_dtype(self) -> paddle.dtype:
        """_summary_

        Returns:
            paddle.dtype: _description_
        """
        return self.blocks[0].mlp.fc2.weight.dtype

    def rot_pos_emb(self, grid_thw, num_pad=0):
        """_summary_

        Args:
            grid_thw (_type_): _description_

        Returns:
            _type_: _description_
        """
        pos_ids = []
        grid_hw_array = np.array(grid_thw, dtype=np.int64)
        for t, h, w in grid_hw_array:
            hpos_ids = np.arange(h).reshape(-1, 1)
            hpos_ids = np.tile(hpos_ids, (1, w))
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = np.transpose(hpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.flatten()

            wpos_ids = np.arange(w).reshape(1, -1)
            wpos_ids = np.tile(wpos_ids, (h, 1))
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = np.transpose(wpos_ids, (0, 2, 1, 3))
            wpos_ids = wpos_ids.flatten()

            stacked_ids = np.stack([hpos_ids, wpos_ids], axis=-1)
            tiled_ids = np.tile(stacked_ids, (t, 1))
            pos_ids.append(tiled_ids)

        pos_ids = np.concatenate(pos_ids, axis=0)
        if num_pad > 0:
            pos_ids = np.concatenate(
                [pos_ids, np.zeros((num_pad, 2), dtype=pos_ids.dtype)]
            )
        max_grid_size = np.amax(grid_hw_array[:, 1:])
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(start_axis=1)
        return rotary_pos_emb

    def forward(
        self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor, num_pad=0
    ) -> paddle.Tensor:
        """_summary_

        Args:
            hidden_states (paddle.Tensor): _description_
            grid_thw (paddle.Tensor): _description_

        Returns:
            paddle.Tensor: _description_
        """
        # logger.info(f"[DEBUG lwx] just in vit forward: {hidden_states.shape}")
        hidden_states = self.patch_embed(hidden_states)
        # logger.info(f"[DEBUG lwx] in vit after patch_embd: {hidden_states.shape}")

        rotary_pos_emb = self.rot_pos_emb(grid_thw, num_pad=num_pad)

        cu_seqlens = paddle.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(axis=0, dtype="int32")

        if num_pad > 0:
            cu_seqlens = F.pad(cu_seqlens, (1, 1), value=0)
            cu_seqlens[-1] = cu_seqlens[-2] + num_pad
        else:
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

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

        attn_sep = (
            getattr(self.config, "attn_sep", False)
            and self.config.tensor_parallel_degree > 1
        )
        vit_num_recompute_layers = getattr(
            self.config, "vit_num_recompute_layers", self.config.depth
        )

        for idx, blk in enumerate(self.blocks):
            if (
                self.config.use_recompute
                and self.training
                and idx < vit_num_recompute_layers
            ):
                hidden_states = recompute(
                    blk, hidden_states, startend_row_indices, rotary_pos_emb, attn_sep
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    startend_row_indices=startend_row_indices,
                    rotary_pos_emb=rotary_pos_emb,
                    attn_sep=attn_sep,
                )

        # ret = self.merger(hidden_states)
        # ret = hidden_states
        ret = self.ln(hidden_states)  # add norm
        return ret

    def extract_feature(
        self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor
    ) -> paddle.Tensor:
        """_summary_

        Args:
            hidden_states (paddle.Tensor): _description_
            grid_thw (paddle.Tensor): _description_

        Returns:
            paddle.Tensor: _description_
        """
        return self.forward(hidden_states, grid_thw)

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        """
        dummy
        """
        return {}

    def set_state_dict(self, state_dict, *args, **kwargs):
        """_summary_

        Args:
            state_dict (_type_): _description_
        """
        ret = super().set_state_dict(state_dict, *args, **kwargs)
        logger.info(f"dfn rope set_state_dict: {ret}")
