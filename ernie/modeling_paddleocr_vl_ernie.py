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

"""Paddle Ernie model for PaddleOCR-VL"""

import contextlib
import functools
from typing import Optional, Tuple

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import incubate, nn, tensor
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.layers.mpu.mp_layers import (
    VocabParallelEmbedding,
)
from paddle.distributed.fleet.meta_parallel import (
    ParallelCrossEntropy,
    get_rng_state_tracker,
)
from paddle.distributed.fleet.utils import recompute

from paddleformers.utils.log import logger
from paddleformers.transformers.model_utils import PretrainedModel
from paddleformers.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)
from .configuration_paddleocr_vl import PaddleOCRVLConfig
from .distributed import (
    AllGatherVarlenOp,
    GatherOp,
    mark_as_sequence_parallel_parameter,
    parallel_matmul,
    sequence_parallel_sparse_mask_labels,
)
from .fusion_ops import (
    Linear,
    fused_rms_norm_ext,
    fused_swiglu,
    fusion_flash_attention,
)
from .sequence_parallel_utils import ScatterOp


def calc_lm_head_logits(
    config, hidden_states, weight, bias, tensor_parallel_output=None, training=True
):
    """
    Calculate language model head logits with support for various parallelization strategies.

    This is the core function that computes the final output logits for a language model,
    handling sequence parallelism and tensor parallelism configurations.

    Args:
        config (PaddleOCRVLConfig): Model configuration.
        hidden_states (Tensor): Hidden states from the transformer layers
        weight (Tensor): Weight matrix for the language model head
        bias (Tensor): Bias vector for the language model head
        tensor_parallel_output (bool, optional): Override for tensor parallel output behavior.
                                               If None, uses config.tensor_parallel_output.
                                               Defaults to None.
        training (bool, optional): Whether in training mode. Defaults to True.

    Returns:
        Tensor: The computed logits for language modeling.
    """
    if config.sequence_parallel:
        if config.use_sparse_head_and_loss_fn:
            pass  # Nothing needs to be done.
        else:
            hidden_states = GatherOp.apply(hidden_states)
            max_sequence_length = config.max_sequence_length
            hidden_states = hidden_states.reshape(
                [-1, max_sequence_length, hidden_states.shape[-1]]
            )

    if tensor_parallel_output is None:
        tensor_parallel_output = config.tensor_parallel_output
    logits = parallel_matmul(
        hidden_states,
        weight,
        bias=bias,
        transpose_y=config.tie_word_embeddings,
        tensor_parallel_degree=config.tensor_parallel_degree,
        tensor_parallel_output=tensor_parallel_output,
        fuse_linear=config.fuse_linear,
        training=training,
    )

    return logits


def subbatch(f, arg_idx, axis, bs, out_idx, use_recompute=False, same_arg_idx={}):
    """
    Converts a function to one that applies to subbatch of an input dimension.
    This is useful for processing large tensors in smaller chunks to reduce memory usage.

    Args:
        f (Callable): Original function to be converted to subbatch processing.
        arg_idx ([int]): Indices of the inputs to be subbatched.
        axis ([int]): Indices of the dimensions to be subbatched for each input.
        bs (int): Subbatch size (number of elements to process at once).
        out_idx (int): Index of the output dimension that needs stacking.
        use_recompute (bool, optional): Whether to use recomputation for memory savings. Defaults to False.
        same_arg_idx (dict, optional): Mapping of argument indices that share the same tensor.
                                     e.g. {1: 0} means args[1] == args[0], avoiding duplicate slicing.

    Returns:
        Callable: Converted function that processes inputs in subbatches.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):

        assert len(arg_idx) == len(
            axis
        ), "Number of batching args and number of batching dims should match."

        inps = [args[i] for i in arg_idx]
        axis_width = [inp.shape[d] for inp, d in zip(inps, axis)]
        assert len(set(axis_width)) == 1, "Batch sizes should be kept equal."

        inp_axis = {inp: d for inp, d in zip(inps, axis)}

        axis_width = axis_width[0]
        if axis_width < bs:
            return f(*args, **kwargs)

        outs = []
        for slice_at in np.arange(0, axis_width, bs):
            _args = []
            for i, inp in enumerate(args):
                if i in same_arg_idx:
                    assert (
                        i > same_arg_idx[i]
                    ), f"expect i > same_arg_idx[i], but got i: {i} and same_arg_idx[i]: {same_arg_idx[i]}"
                    _args.append(_args[same_arg_idx[i]])
                elif i in arg_idx:
                    inp = inp.slice(
                        [inp_axis[inp]],
                        [slice_at],
                        [min(inp.shape[inp_axis[inp]], slice_at + bs)],
                    )
                    _args.append(inp)
                else:
                    _args.append(inp)
            if use_recompute:
                out = paddle.distributed.fleet.utils.recompute(f, *_args, **kwargs)
            else:
                out = f(*_args, **kwargs)
            outs.append(out)

        return paddle.concat(outs, out_idx)

    return wrapper


def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat((-x2, x1), axis=-1)


def _apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=2):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/)."""
    mrope_section = mrope_section * 2
    cos = paddle.concat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, axis=-1))], axis=-1
    ).unsqueeze(unsqueeze_dim)
    sin = paddle.concat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, axis=-1))], axis=-1
    ).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class FusedDropoutImpl(nn.Layer):
    """
    Fused dropout implementation with residual connection support.

    This layer combines dropout and residual addition in a single operation for better performance,
    particularly on GPU devices. The dropout is conditionally applied based on the probability.

    Args:
        prob (float): Dropout probability (between 0 and 1)
        mode (str): Dropout mode, either 'upscale_in_train' or 'downscale_in_infer'

    Attributes:
        prob (float): Stores the dropout probability
        mode (str): Stores the dropout mode
        dropout (nn.Dropout): The actual dropout layer instance
    """

    def __init__(self, prob, mode):
        """
        Initialize the fused dropout layer.

        Args:
            prob (float): Dropout probability (0 means no dropout)
            mode (str): Dropout mode ('upscale_in_train' or 'downscale_in_infer')
        """
        super().__init__()
        self.prob = prob
        self.mode = mode
        self.dropout = nn.Dropout(p=prob, mode=mode)

    def forward(self, x, y):
        """
        Forward pass of the fused dropout layer.

        Args:
            x (Tensor): Input tensor to potentially apply dropout on
            y (Tensor): Residual tensor to add to the (possibly dropped out) x

        Returns:
            Tensor: Result of x (with optional dropout) + y
        """
        if self.prob > 0:
            x = self.dropout(x)
        output = x + y

        return output


class RMSNorm(nn.Layer):
    """
    Root Mean Square Layer Normalization (RMSNorm) implementation.

    RMSNorm is a simplified version of LayerNorm that focuses on the root mean square of inputs,
    omitting the mean-centering operation. This provides computational efficiency while maintaining
    good performance.

    """

    def __init__(self, config):
        """
        Initialize RMSNorm layer.

        Args:
            config (PaddleOCRVLConfig): Model configuration.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

        if config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.weight)

    def forward(self, hidden_states):
        """
        Apply RMS normalization to input hidden states.

        Args:
            hidden_states (Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Tensor: Normalized output tensor of same shape as input

        Note:
            - Uses fused kernel if config.fuse_rms_norm is True for better performance
            - Otherwise computes RMSNorm manually:
                1. Compute variance of features
                2. Apply reciprocal square root normalization
                3. Scale by learned weight parameter
            - Maintains original dtype for numerical stability during computation
        """
        if self.config.fuse_rms_norm:
            return fused_rms_norm_ext(
                hidden_states, self.weight, self.variance_epsilon
            )[0].astype(self.weight.dtype)
        with paddle.amp.auto_cast(False):
            variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
            hidden_states = (
                paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
            )
        return hidden_states.astype(self.weight.dtype) * self.weight


class LayerNorm(nn.LayerNorm):
    """
    Layer Normalization (LayerNorm) implementation with optional optimizations.

    This extends PaddlePaddle's built-in LayerNorm with:
    1. Sequence parallelism support
    2. Fast fused kernel implementation option
    3. Configurable epsilon value

    """

    def __init__(self, config):
        """
        Initialize LayerNorm with configuration.

        Args:
            config (PaddleOCRVLConfig): Model configuration contains normalization parameters and flags.
        """
        super().__init__(config.hidden_size, epsilon=config.rms_norm_eps)
        self.config = config
        if config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.weight)
            mark_as_sequence_parallel_parameter(self.bias)


class KeyeRotaryEmbedding(nn.Layer):
    def __init__(self, config: PaddleOCRVLConfig):
        super().__init__()
        self.rope_kwargs = {}

        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        if self.rope_type == "default":
            dim = config.head_dim
            inv_freq = 1.0 / (
                config.rope_theta
                ** (paddle.arange(0, dim, 2, dtype="int64").astype("float32") / dim)
            )
            self.attention_scaling = 1.0
        else:
            raise ValueError(f"Unsupported rope type: {self.rope_type}")

        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = self.inv_freq

    @paddle.no_grad()
    def forward(self, x, position_ids):
        # Core RoPE block. In contrast to other models, Keye has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .cast("float32")
            .expand((3, position_ids.shape[1], -1, 1))
        )
        position_ids_expanded = position_ids[:, :, None, :].cast(
            "float32"
        )  # shape (3, bs, 1, positions)

        with paddle.amp.auto_cast(enable=False):
            freqs = (
                inv_freq_expanded.cast("float32")
                @ position_ids_expanded.cast("float32")
            ).transpose((0, 1, 3, 2))
            emb = paddle.concat((freqs, freqs), axis=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.astype(dtype=x.dtype), sin.astype(dtype=x.dtype)


class Ernie4_5MLP(nn.Layer):
    """
    Ernie4_5MLP - Gated Multi-Layer Perceptron module used in Ernie model.
    """

    def __init__(self, config, layer_idx=0):
        """
        Initialize the MLP module with configuration options.

        Args:
            config (PaddleOCRVLConfig): Model configurations.
            layer_idx (int): Index of current layer (default: 0)
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        LinearFN = paddle.incubate.nn.FusedLinear if config.fuse_linear else Linear
        self.gate_proj = LinearFN(
            self.hidden_size, self.intermediate_size, bias_attr=config.use_bias
        )
        self.up_proj = LinearFN(
            self.hidden_size, self.intermediate_size, bias_attr=config.use_bias
        )
        self.down_proj = LinearFN(
            self.intermediate_size, self.hidden_size, bias_attr=config.use_bias
        )

        self.fuse_swiglu = config.fuse_swiglu
        if self.fuse_swiglu:
            assert fused_swiglu is not None, "fused_swiglu operator is not found."

    def forward(self, x):
        """
        Forward pass through the MLP module.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, hidden_size]

        Note:
            Implements SwiGLU activation: swish(Wx) * (Vx) where W and V are
            the first and second halves of up_gate_proj output respectively.
        """
        if self.fuse_swiglu:
            x = self.up_gate_proj(x)
            x = fused_swiglu(x)
        else:
            gate = self.gate_proj(x)
            x = self.up_proj(x)
            x = F.silu(gate) * x
        return self.down_proj(x)


class Ernie4_5Attention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx=0):
        """Initialize the attention layer.

        Args:
            config (PaddleOCRVLConfig): Model configuration.
            layer_idx (int, optional): Index in transformer stack. Defaults to 0.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        if getattr(config, "head_dim", None) is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = config.head_dim
        self.is_gqa = (
            config.num_key_value_heads is not None
            and config.num_key_value_heads != self.num_heads
        )

        self.rope_scaling = config.rope_scaling

        self.freq_allocation = config.get("freq_allocation", 0)

        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), f"num_heads: {self.num_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_heads = self.num_heads // config.tensor_parallel_degree
            if self.is_gqa:
                assert (
                    self.num_key_value_heads % config.tensor_parallel_degree == 0
                ), f"num_heads: {self.num_key_value_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
                self.num_key_value_heads = (
                    self.num_key_value_heads // config.tensor_parallel_degree
                )
        if self.is_gqa:
            logger.info(
                f"use GQA - num_heads: {self.num_heads}- num_key_value_heads: {self.num_key_value_heads}"
            )
            assert (
                self.num_heads % self.num_key_value_heads == 0
            ), f"num_heads: {self.num_heads}, num_key_value_heads: {self.num_key_value_heads}"
            if getattr(config, "head_dim", None) is None:
                kv_hidden_size = (
                    self.hidden_size // self.num_heads * self.num_key_value_heads
                )
            else:
                kv_hidden_size = self.head_dim * config.num_key_value_heads
                q_hidden_size = self.head_dim * config.num_attention_heads
        else:
            q_hidden_size = kv_hidden_size = self.head_dim * config.num_attention_heads

        LinearFN = paddle.incubate.nn.FusedLinear if config.fuse_linear else Linear
        self.q_proj = LinearFN(
            self.hidden_size,
            q_hidden_size,
            bias_attr=config.use_bias,
        )
        self.k_proj = LinearFN(
            self.hidden_size,
            kv_hidden_size,
            bias_attr=config.use_bias,
        )
        self.v_proj = LinearFN(
            self.hidden_size,
            kv_hidden_size,
            bias_attr=config.use_bias,
        )

        self.o_proj = LinearFN(
            (
                self.hidden_size
                if getattr(config, "head_dim", None) is None
                else q_hidden_size
            ),
            self.hidden_size,
            bias_attr=config.use_bias,
        )
        self.config = config

        self._rr_flash_attn = None

        self.set_attn_func()

    def set_attn_func(self):
        """Configure attention function based on settings.

        Selects between flash/core attention.
        """
        config = self.config
        if config.use_flash_attention:
            self.attn_func = self._flash_attention_wrapper
        else:
            self.attn_func = self.core_attn

        if config.cachekv_quant:
            from paddleslim.common.wrapper_function import FuncWrapper

            self.attn_func = FuncWrapper(self.attn_func)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_start_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        token_type_ids: Optional[Tuple[paddle.Tensor]] = None,  # MLLM
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Compute attention outputs.

        Args:
            hidden_states (paddle.Tensor): Input tensor [bsz, seq_len, hidden_size]
            position_embeddings (paddle.Tensor): Position embeddings
            past_key_value (Optional[Tuple[paddle.Tensor, paddle.Tensor]]): Cached key/value states
            attention_mask (Optional[paddle.Tensor]): Attention mask tensor
            attn_mask_start_row_indices (Optional[paddle.Tensor]): Variable length attention indices
            position_ids (Optional[paddle.Tensor]): Position indices for RoPE
            output_attentions (bool): Return attention weights if True
            use_cache (bool): Cache key/value states if True

        Returns:
            Tuple containing:
                - attention_output: [bsz, seq_len, hidden_size]
                - attention_weights: Optional attention probabilities
                - updated_key_value_cache: Optional updated cache
        """
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, :-1]
        if self.config.sequence_parallel:
            if token_type_ids is not None:
                token_type_ids = token_type_ids.reshape([-1])
                token_type_ids = ScatterOp.apply(token_type_ids)
                token_type_ids.stop_gradient = True
            max_sequence_length = self.config.max_sequence_length
            bsz = (
                hidden_states.shape[0]
                * self.config.tensor_parallel_degree
                // max_sequence_length
            )
            q_len = max_sequence_length
        else:
            bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).reshape(
            [bsz, q_len, self.num_heads, self.head_dim]
        )
        if self.is_gqa:
            key_states = self.k_proj(hidden_states).reshape(
                [bsz, q_len, self.num_key_value_heads, self.head_dim]
            )
            value_states = self.v_proj(hidden_states).reshape(
                [bsz, q_len, self.num_key_value_heads, self.head_dim]
            )
        else:
            key_states = self.k_proj(hidden_states).reshape(
                [bsz, q_len, self.num_heads, self.head_dim]
            )
            value_states = self.v_proj(hidden_states).reshape(
                [bsz, q_len, self.num_heads, self.head_dim]
            )

        has_gradient = not (
            query_states.stop_gradient
            and key_states.stop_gradient
            and value_states.stop_gradient
        )
        if (
            self.config.recompute
            and self.config.recompute_granularity == "core_attn"
            and has_gradient
        ):
            assert past_key_value is None, "do not use kv cache in recompute"
            assert not use_cache
            attn_output, attn_weights, past_key_value = recompute(
                self.rope_attn,
                query_states,
                key_states,
                value_states,
                position_embeddings,
                attention_mask,
                position_ids,
                output_attentions,
                past_key_value,
                use_cache,
                attn_mask_start_row_indices,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            attn_output, attn_weights, past_key_value = self.rope_attn(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attn_mask_start_row_indices=attn_mask_start_row_indices,
            )
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    def _flash_attention_wrapper(
        self,
        q,
        k,
        v,
        attention_mask=None,
        attn_mask_start_row_indices=None,
        seq_length=None,
    ):
        """Optimized flash attention implementation.

        Args:
            q (paddle.Tensor): Query tensor
            k (paddle.Tensor): Key tensor
            v (paddle.Tensor): Value tensor
            attention_mask (Optional[paddle.Tensor]): Attention mask
            attn_mask_start_row_indices (Optional[paddle.Tensor]): Variable length indices
            seq_length (Optional[int]): Sequence length

        Returns:
            paddle.Tensor: Attention output tensor
        """
        return fusion_flash_attention(
            q,
            k,
            v,
            self.training,
            self.config.attention_probs_dropout_prob,
            self.config.use_sparse_flash_attn,
            attention_mask,
            attn_mask_start_row_indices,
            seq_length,
            self.config.use_var_len_flash_attn,
            self._rr_flash_attn if self.training else None,
        )

    def core_attn(
        self,
        q,
        k,
        v,
        attention_mask=None,
        attn_mask_start_row_indices=None,
        seq_length=None,
    ):
        """Standard self-attention implementation.

        Args:
            q (paddle.Tensor): Query tensor
            k (paddle.Tensor): Key tensor
            v (paddle.Tensor): Value tensor
            attention_mask (Optional[paddle.Tensor]): Attention mask
            attn_mask_start_row_indices (Optional[paddle.Tensor]): Variable length indices
            seq_length (Optional[int]): Sequence length

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: Attention output and weights
        """
        perm = [
            0,
            2,
            1,
            3,
        ]  # [1, 2, 0, 3] if self.sequence_parallel else [0, 2, 1, 3]
        origin_dtype = q.dtype

        q = tensor.transpose(x=q, perm=perm)
        k = tensor.transpose(x=k, perm=perm)
        v = tensor.transpose(x=v, perm=perm)

        replicate = self.config.num_attention_heads // self.config.num_key_value_heads
        k = paddle.repeat_interleave(k, replicate, axis=1)
        v = paddle.repeat_interleave(v, replicate, axis=1)

        scale_qk_coeff = self.config.scale_qk_coeff * self.head_dim**0.5
        product = paddle.matmul(x=q.scale(1.0 / scale_qk_coeff), y=k, transpose_y=True)

        product = product.cast(paddle.float32)
        if self.config.scale_qk_coeff != 1.0:
            product = product.scale(self.config.scale_qk_coeff)

        if attention_mask is not None:
            attention_mask = attention_mask.cast(paddle.float32)
            if self.config.fuse_softmax_mask:
                weights = incubate.softmax_mask_fuse(product, attention_mask)
            else:
                product = product + attention_mask
                weights = F.softmax(product)
        else:
            weights = incubate.softmax_mask_fuse_upper_triangle(product)

        weights = weights.cast(origin_dtype)

        if self.config.attention_probs_dropout_prob:
            with get_rng_state_tracker().rng_state("local_seed"):
                weights = F.dropout(
                    weights,
                    self.config.attention_probs_dropout_prob,
                    training=self.training,
                    mode="upscale_in_train",
                )

        out = paddle.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        # If sequence_parallel is true, out shape is [s, b, h] after reshape
        # else out shape is [b, s, h]
        out = tensor.reshape(x=out, shape=[0, 0, -1])

        return out, weights

    def rope_attn(
        self,
        query_states,
        key_states,
        value_states,
        position_embeddings,
        attention_mask,
        position_ids,
        output_attentions=False,
        past_key_value=None,
        use_cache=False,
        attn_mask_start_row_indices=None,
    ):
        query_states_dtype = query_states.dtype

        kv_seq_len = position_ids.max() + 1
        offset = 0
        if past_key_value is not None:
            # LLM
            offset = past_key_value[0].shape[-3]
            kv_seq_len += offset

        query_states = query_states.astype(query_states_dtype)
        key_states = key_states.astype(query_states_dtype)

        if position_ids.dim() == 3 and position_ids.shape[0] > 1:
            position_ids = position_ids[0:1]

        cos, sin = position_embeddings
        query_states, key_states = _apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)

        # NOTE(for generation): use list instead of tuple to store the cache
        # tensors, so that we can clear the cache tensors for memory efficiency.
        past_key_value = [key_states, value_states] if use_cache else None
        seq_length = query_states.shape[1]
        attn_output, attn_weights = self.attn_func(
            query_states,
            key_states,
            value_states,
            attention_mask,
            attn_mask_start_row_indices,
            seq_length,
        )
        return attn_output, attn_weights, past_key_value


class FusedHeadParallelCrossEntropy(PyLayer):
    """Fused parallel cross-entropy loss computation for large sequence lengths.

    Combines head projection and loss computation with optimized memory usage for long sequences,
    supporting tensor parallel training.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        weight,
        bias,
        labels,
        tensor_parallel_degree,
        mp_group=None,
        ignore_index=-100,
        seq_chunk_size=8192,
        transpose_y=False,
        fuse_linear=False,
        training=True,
    ):
        """Forward pass for parallel cross-entropy computation.

        Args:
            ctx: Context object for saving tensors between forward/backward
            hidden_states (paddle.Tensor): Input tensor of shape [batch_size*seq_len, hidden_size]
            weight (paddle.Tensor): Weight matrix for projection
            bias (Optional[paddle.Tensor]): Optional bias vector
            labels (paddle.Tensor): Target labels tensor of shape [batch_size*seq_len]
            tensor_parallel_degree (int): Degree of tensor parallelism
            mp_group (Optional[dist.Group]): Model parallel group. Defaults to None (auto-detect)
            ignore_index (int): Index to ignore in loss computation. Defaults to -100
            seq_chunk_size (int): Chunk size for processing long sequences. Defaults to 8192
            transpose_y (bool): Whether to transpose weight matrix. Defaults to False
            fuse_linear (bool): Whether to use fused linear ops. Defaults to False
            training (bool): Whether in training mode. Defaults to True

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]:
                - loss: Computed loss tensor
                - gathered_labels: Concatenated labels from all parallel groups
        """

        ctx.tensor_parallel_degree = tensor_parallel_degree
        ctx.ignore_index = ignore_index
        ctx.seq_chunk_size = seq_chunk_size
        ctx.transpose_y = transpose_y
        ctx.fuse_linear = fuse_linear
        ctx.training = training

        ctx.hidden_states_shape = hidden_states.shape

        ctx.mp_group = (
            fleet.get_hybrid_communicate_group().get_model_parallel_group()
            if mp_group is None
            else mp_group
        )
        ctx.rank = ctx.mp_group.rank
        ctx.world_size = ctx.mp_group.nranks

        loss_all = []
        labels_all = []
        with paddle.no_grad():
            labels = labels.reshape_([-1])
            hidden_states = hidden_states.reshape_([-1, hidden_states.shape[-1]])

            num_tokens_per_rank = []
            dist.stream.all_gather(
                num_tokens_per_rank,
                paddle.to_tensor(hidden_states.shape[0], dtype=paddle.int32),
                group=ctx.mp_group,
            )
            ctx.num_tokens_per_rank = num_tokens_per_rank

            for idx in range(ctx.world_size):
                if idx == ctx.rank:
                    hidden_states_recv = hidden_states
                    labels_recv = labels
                else:
                    hidden_states_recv = paddle.empty(
                        [ctx.num_tokens_per_rank[idx], hidden_states.shape[-1]],
                        dtype=hidden_states.dtype,
                    )
                    labels_recv = paddle.empty(
                        [ctx.num_tokens_per_rank[idx]], dtype=labels.dtype
                    )

                dist.stream.broadcast(
                    hidden_states_recv, src=ctx.mp_group.ranks[idx], group=ctx.mp_group
                )
                dist.stream.broadcast(
                    labels_recv, src=ctx.mp_group.ranks[idx], group=ctx.mp_group
                )

                seq_len = hidden_states_recv.shape[0]
                num_chunk = (seq_len + ctx.seq_chunk_size - 1) // ctx.seq_chunk_size

                loss_chunk = []
                for chunk_idx in range(num_chunk):
                    start = chunk_idx * ctx.seq_chunk_size
                    end = min(start + ctx.seq_chunk_size, seq_len)
                    hidden_states_chunk = hidden_states_recv._slice(start, end)
                    labels_chunk = labels_recv._slice(start, end)

                    logits = parallel_matmul(
                        hidden_states_chunk,
                        weight,
                        bias=bias,
                        transpose_y=ctx.transpose_y,
                        tensor_parallel_degree=ctx.tensor_parallel_degree,
                        tensor_parallel_output=True,
                        fuse_linear=ctx.fuse_linear,
                        training=ctx.training,
                    )

                    with paddle.amp.auto_cast(False):
                        loss = mp_ops._c_softmax_with_cross_entropy(
                            logits.cast("float32"),
                            labels_chunk.unsqueeze(-1),
                            group=ctx.mp_group,
                            ignore_index=ctx.ignore_index,
                        )
                        loss_chunk.append(loss)
                loss_all.append(paddle.concat(loss_chunk, axis=0))
                labels_all.append(labels_recv)

            ctx.loss_concat_sections = [loss.shape[0] for loss in loss_all]
            loss_all = paddle.concat(loss_all, axis=0)
            labels_all = paddle.concat(labels_all, axis=0)

            tensor_inputs = [hidden_states, weight, bias, labels]
            ctx.save_for_backward(*tensor_inputs)

        return loss_all, labels_all

    @staticmethod
    def backward(ctx, loss_all_grad, labels_all_grad):
        """Backward pass for parallel cross-entropy computation.

        Args:
            ctx: Context object with saved tensors from forward
            loss_all_grad (paddle.Tensor): Gradient of loss
            labels_all_grad (paddle.Tensor): Gradient of labels (unused)

        Returns:
            Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[paddle.Tensor], None]:
                - hidden_states_grad: Gradient for input hidden states
                - weight_grad: Gradient for weight matrix (None if not trainable)
                - bias_grad: Gradient for bias vector (None if not trainable or not provided)
                - None: Placeholder for labels gradient
        """

        hidden_states, weight, bias, labels = ctx.saved_tensor()

        loss_all_grad_list = paddle.split(
            loss_all_grad, ctx.loss_concat_sections, axis=0
        )

        def detach_variable(inp):
            if inp is None:
                return None
            x = inp.detach()
            x.stop_gradient = inp.stop_gradient
            return x

        if weight.stop_gradient is False:
            weight_main_grad = paddle.zeros(weight.shape, dtype=paddle.float32)
        else:
            weight_main_grad = None
        if bias is not None and bias.stop_gradient is False:
            bias_main_grad = paddle.zeros(bias.shape, dtype=paddle.float32)
        else:
            bias_main_grad = None

        hidden_states = detach_variable(hidden_states)
        weight = detach_variable(weight)
        bias = detach_variable(bias)
        labels = detach_variable(labels)

        with paddle.base.dygraph.guard():
            tracer = paddle.base.framework._dygraph_tracer()
            tracer._has_grad = True

            for idx in range(ctx.world_size):
                if idx == ctx.rank:
                    hidden_states_recv = hidden_states
                    labels_recv = labels
                else:
                    hidden_states_recv = paddle.empty(
                        [ctx.num_tokens_per_rank[idx], hidden_states.shape[-1]],
                        dtype=hidden_states.dtype,
                    )
                    labels_recv = paddle.empty(
                        [ctx.num_tokens_per_rank[idx]], dtype=labels.dtype
                    )

                dist.stream.broadcast(
                    hidden_states_recv, src=ctx.mp_group.ranks[idx], group=ctx.mp_group
                )
                dist.stream.broadcast(
                    labels_recv, src=ctx.mp_group.ranks[idx], group=ctx.mp_group
                )
                hidden_states_recv.stop_gradient = False

                seq_len = hidden_states_recv.shape[0]
                num_chunk = (seq_len + ctx.seq_chunk_size - 1) // ctx.seq_chunk_size

                for chunk_idx in range(num_chunk):
                    start = chunk_idx * ctx.seq_chunk_size
                    end = min(start + ctx.seq_chunk_size, seq_len)
                    hidden_states_chunk = hidden_states_recv.slice(
                        axes=[0], starts=[start], ends=[end]
                    )
                    labels_chunk = labels_recv._slice(start, end)
                    loss_grad_chunk = loss_all_grad_list[idx]._slice(start, end)

                    logits = parallel_matmul(
                        hidden_states_chunk,
                        weight,
                        bias=bias,
                        transpose_y=ctx.transpose_y,
                        tensor_parallel_degree=ctx.tensor_parallel_degree,
                        tensor_parallel_output=True,
                        fuse_linear=ctx.fuse_linear,
                        training=ctx.training,
                    )

                    with paddle.amp.auto_cast(False):
                        loss_chunk = mp_ops._c_softmax_with_cross_entropy(
                            logits.cast("float32"),
                            labels_chunk.unsqueeze(-1),
                            group=ctx.mp_group,
                            ignore_index=ctx.ignore_index,
                        )

                    with paddle.amp.auto_cast(enable=False):
                        paddle.autograd.backward(loss_chunk, loss_grad_chunk)

                    if weight_main_grad is not None:
                        weight_main_grad.add_(weight.grad.cast(paddle.float32))
                        weight.clear_gradient(True)
                    if bias_main_grad is not None:
                        bias_main_grad.add_(bias.grad.cast(paddle.float32))
                        bias.clear_gradient(True)

                if idx == ctx.rank:
                    hidden_states_grad = hidden_states_recv.grad
                    hidden_states_grad = hidden_states_grad.reshape(
                        ctx.hidden_states_shape
                    )

        if weight_main_grad is not None:
            weight_main_grad = weight_main_grad.astype(weight.dtype)
        if bias_main_grad is not None:
            bias_main_grad = bias_main_grad.astype(bias.dtype)

        return (
            hidden_states_grad,
            weight_main_grad,
            bias_main_grad,
            None,
        )


class ErniePretrainingCriterion(paddle.nn.Layer):
    """Criterion for ERNIE pretraining task."""

    def __init__(self, config, return_tuple=True):
        """Initialize the pretraining criterion.

        Args:
            config (PaddleOCRVLConfig): Model configuration.
            return_tuple (bool): Whether to return loss as tuple (loss, loss_sum). Defaults to True.
        """
        super(ErniePretrainingCriterion, self).__init__()
        self.ignored_index = getattr(config, "ignored_index", -100)
        self.config = config
        self.return_tuple = return_tuple
        self.enable_parallel_cross_entropy = (
            config.tensor_parallel_degree > 1 and config.tensor_parallel_output
        )

        if (
            self.enable_parallel_cross_entropy
        ):  # and False: # and lm_head is distributed
            logger.info("using parallel cross entroy, take care")
            self.loss_func = ParallelCrossEntropy()
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(
                reduction="none",
            )
        self.token_balance_loss = config.token_balance_loss

    def forward(self, prediction_scores, masked_lm_labels, loss_mask=None):
        """Compute the pretraining loss.

        Args:
            prediction_scores (Union[paddle.Tensor, Tuple[paddle.Tensor, ...]]):
                Either:
                - Direct logits tensor [batch_size, seq_len, vocab_size]
                - Tuple of (hidden_states, weight, bias) for sparse head computation
            masked_lm_labels (paddle.Tensor): Target labels tensor [batch_size, seq_len]
            loss_mask (Optional[paddle.Tensor]): Optional mask for valid tokens. Defaults to None.

        Returns:
            Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
                - If return_tuple=False: Single loss tensor
                - If return_tuple=True: Tuple of (normalized_loss, sum_loss)
        """

        if self.config.use_sparse_head_and_loss_fn:
            hidden_states, outlinear_weight, outlinear_bias = prediction_scores[:3]

            if self.config.sequence_parallel:
                masked_lm_labels, sparse_label_idx = (
                    sequence_parallel_sparse_mask_labels(
                        masked_lm_labels, self.ignored_index
                    )
                )
                sparse_label_idx = sparse_label_idx.reshape([-1, 1])
                hidden_states = paddle.gather(hidden_states, sparse_label_idx, axis=0)
                hidden_states = AllGatherVarlenOp.apply(hidden_states)
            else:
                masked_lm_labels = masked_lm_labels.flatten()
                sparse_label_idx = paddle.nonzero(
                    masked_lm_labels != self.ignored_index
                ).flatten()
                masked_lm_labels = paddle.take_along_axis(
                    masked_lm_labels, sparse_label_idx, axis=0
                )

                hidden_states = hidden_states.reshape([-1, hidden_states.shape[-1]])
                hidden_states = paddle.take_along_axis(
                    hidden_states, sparse_label_idx.reshape([-1, 1]), axis=0
                )

            # `loss_mask` must be reset to None and re-calculate it in ErnieBotPretrainingCriterion
            # when use use_sparse_head_and_loss_fn.
            loss_mask = None
            if self.config.use_recompute_loss_fn:
                offload_kwargs = {}
                if self.config.get("offload_lm_head", False):
                    offload_kwargs["offload_indices"] = [1]
                res = recompute(
                    self.forward_impl_with_calc_logits,
                    masked_lm_labels,
                    loss_mask,
                    hidden_states,
                    outlinear_weight,
                    outlinear_bias,
                    **offload_kwargs,
                )
            else:
                logits = calc_lm_head_logits(
                    self.config,
                    hidden_states,
                    outlinear_weight,
                    outlinear_bias,
                    training=self.training,
                )
                res = self.forward_impl(logits, masked_lm_labels, loss_mask)
        elif self.config.use_recompute_loss_fn:
            if self.config.use_fused_head_and_loss_fn:
                res = self.forward_impl_with_fused_head_loss_fn(
                    masked_lm_labels, loss_mask, *prediction_scores
                )
            else:
                assert isinstance(prediction_scores, tuple) and len(
                    prediction_scores
                ) in [3, 4], prediction_scores
                res = recompute(
                    self.forward_impl_with_calc_logits,
                    masked_lm_labels,
                    loss_mask,
                    *prediction_scores,
                )
        else:
            res = self.forward_impl(prediction_scores, masked_lm_labels, loss_mask)

        return res

    def forward_impl_with_fused_head_loss_fn(
        self,
        masked_lm_labels,
        loss_mask,
        hidden_states,
        outlinear_weight,
        outlinear_bias,
    ):
        """Compute loss with fused head and parallel cross-entropy.

        Args:
            masked_lm_labels (paddle.Tensor): Target labels tensor [batch_size, seq_len]
            loss_mask (Optional[paddle.Tensor]): Optional mask for valid tokens
            hidden_states (paddle.Tensor): Hidden states from transformer [batch_size, seq_len, hidden_size]
            outlinear_weight (paddle.Tensor): Weight matrix for output projection
            outlinear_bias (Optional[paddle.Tensor]): Optional bias for output projection

        Returns:
            Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
                Same return format as forward()
        """
        assert (
            self.config.tensor_parallel_degree > 0
        ), "use_fused_head_and_loss_fn require tensor_parallel_degree > 0"
        masked_lm_loss, masked_lm_labels_all = FusedHeadParallelCrossEntropy.apply(
            hidden_states,
            outlinear_weight,
            outlinear_bias,
            masked_lm_labels,
            self.config.tensor_parallel_degree,
            ignore_index=self.ignored_index,
            seq_chunk_size=self.config.get("loss_subbatch_seqlen", 32768),
            transpose_y=self.config.tie_word_embeddings,
            fuse_linear=self.config.fuse_linear,
            training=self.training,
        )
        if loss_mask is None:
            loss_mask = masked_lm_labels_all != self.ignored_index
        if (~loss_mask).all():  # empty span
            logger.warning(
                f"encounter empty span when calculate loss, ignored_index={self.ignored_index}"
            )
            loss = paddle.mean(masked_lm_loss) * 0.0
            loss_sum = masked_lm_loss.sum().detach()
        else:
            loss_mask = loss_mask.reshape([-1]).cast(paddle.float32)
            # 逐位对齐, 全精度聚合
            masked_lm_loss = paddle.sum(
                masked_lm_loss.cast(paddle.float32).reshape([-1]) * loss_mask
            )
            loss = masked_lm_loss / loss_mask.sum()
            if self.token_balance_loss:
                _loss = masked_lm_loss / self.config.token_balance_seqlen
                loss = _loss - _loss.detach() + loss.detach()  # for 对线
            loss_sum = masked_lm_loss.sum().detach()
        if not self.return_tuple:  # only used in pp
            if self.training:
                return loss
            return loss_sum
        return loss, loss_sum

    def forward_impl_with_calc_logits(
        self,
        masked_lm_labels,
        loss_mask,
        hidden_states,
        outlinear_weight,
        outlinear_bias,
    ):
        """Compute logits then calculate loss.

        Args:
            Same as forward_impl_with_fused_head_loss_fn()

        Returns:
            Same return format as forward()
        """

        logits = calc_lm_head_logits(
            self.config,
            hidden_states,
            outlinear_weight,
            outlinear_bias,
            training=self.training,
        )

        return self.forward_impl(logits, masked_lm_labels, loss_mask)

    def loss_impl(self, prediction_scores, masked_lm_labels):
        """Core loss computation without reduction.

        Args:
            prediction_scores (paddle.Tensor): Logits tensor [batch_size, seq_len, vocab_size]
            masked_lm_labels (paddle.Tensor): Target labels tensor [batch_size, seq_len]

        Returns:
            paddle.Tensor: Unreduced loss tensor
        """
        prediction_scores = prediction_scores.cast("float32")
        masked_lm_loss = self.loss_func(
            prediction_scores, masked_lm_labels.unsqueeze(-1)
        )
        return masked_lm_loss

    def forward_impl(self, prediction_scores, masked_lm_labels, loss_mask=None):
        """Standard loss computation with reduction and masking.

        Args:
            prediction_scores (paddle.Tensor): Logits tensor [batch_size, seq_len, vocab_size]
            masked_lm_labels (paddle.Tensor): Target labels tensor [batch_size, seq_len]
            loss_mask (Optional[paddle.Tensor]): Optional mask for valid tokens

        Returns:
            Same return format as forward()
        """
        if self.enable_parallel_cross_entropy:
            assert prediction_scores.shape[-1] != self.config.vocab_size, (
                f"enable_parallel_cross_entropy, the vocab_size should be splited:"
                f" {prediction_scores.shape[-1]}, {self.config.vocab_size}"
            )

        with paddle.amp.auto_cast(False):
            prediction_scores_dims = len(prediction_scores.shape)
            if prediction_scores_dims == 2 and prediction_scores.shape[
                0
            ] > self.config.get("loss_subbatch_seqlen", 32768):
                sb_loss_func = subbatch(
                    self.loss_impl,
                    [0, 1],
                    [0, 0],
                    self.config.get("loss_subbatch_seqlen", 32768),
                    0,
                )
                masked_lm_loss = sb_loss_func(prediction_scores, masked_lm_labels)
            elif prediction_scores_dims == 3 and prediction_scores.shape[
                1
            ] > self.config.get("loss_subbatch_seqlen", 32768):
                sb_loss_func = subbatch(
                    self.loss_impl,
                    [0, 1],
                    [1, 1],
                    self.config.get("loss_subbatch_seqlen", 32768),
                    1,
                )
                masked_lm_loss = sb_loss_func(prediction_scores, masked_lm_labels)
            else:
                masked_lm_loss = self.loss_impl(prediction_scores, masked_lm_labels)

            if loss_mask is None:
                loss_mask = masked_lm_labels != self.ignored_index

            lossmask = masked_lm_labels != self.ignored_index
            if (~lossmask).all():  # empty span
                logger.warning(
                    f"encounter empty span when calculate loss, ignored_index={self.ignored_index}"
                )
                loss = paddle.mean(masked_lm_loss) * 0.0
                loss_sum = masked_lm_loss.sum().detach()
            else:
                loss_mask = loss_mask.reshape([-1]).cast(paddle.float32)
                # 逐位对齐, 全精度聚合
                masked_lm_loss = paddle.sum(
                    masked_lm_loss.cast(paddle.float32).reshape([-1]) * loss_mask
                )
                loss = masked_lm_loss / loss_mask.sum()
                if self.token_balance_loss:
                    _loss = masked_lm_loss / self.config.token_balance_seqlen
                    loss = _loss - _loss.detach() + loss.detach()  # for 对线
                loss_sum = masked_lm_loss.sum().detach()
        if not self.return_tuple:  # only used in pp
            if self.training:
                return loss
            return loss_sum
        return loss, loss_sum


class Ernie4_5LMHead(nn.Layer):
    """Language model head for ERNIE with support for tensor parallelism."""

    def __init__(self, config):
        """Initialize the language model head.

        Args:
            config (PaddleOCRVLConfig): Model configuration containing:
                - vocab_size: Size of vocabulary
                - hidden_size: Dimension of hidden states
                - tensor_parallel_degree: Degree of tensor parallelism
                - tie_word_embeddings: Whether to tie input/output embeddings
                - weight_share_add_bias: Whether to add bias when weight sharing
                - use_bias: Whether to use bias term
                - use_recompute_loss_fn: Whether to defer logits computation to loss function
                - use_sparse_head_and_loss_fn: Whether to use sparse head computation
        """

        super(Ernie4_5LMHead, self).__init__()
        self.config = config
        if config.tensor_parallel_degree > 1:
            vocab_size = config.vocab_size // config.tensor_parallel_degree
        else:
            vocab_size = config.vocab_size

        self.weight = self.create_parameter(
            shape=(
                [vocab_size, config.hidden_size]
                if config.tie_word_embeddings
                else [config.hidden_size, vocab_size]
            ),
            dtype=paddle.get_default_dtype(),
        )
        logger.info(
            f"output-weight:{self.weight.shape} config.tie_word_embeddings={config.tie_word_embeddings}"
        )
        if config.weight_share_add_bias and config.use_bias:
            self.bias = self.create_parameter(
                shape=[vocab_size],
                dtype=paddle.get_default_dtype(),
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.constant.Constant(0.0)
                ),
            )
        else:
            self.bias = None

        # Must set distributed attr for Tensor Parallel !
        self.weight.is_distributed = (
            True if (vocab_size != config.vocab_size) else False
        )
        if config.weight_share_add_bias and config.use_bias:
            self.bias.is_distributed = (
                True if (vocab_size != config.vocab_size) else False
            )

        if self.weight.is_distributed:
            self.weight.split_axis = 1
        if (
            config.weight_share_add_bias
            and config.use_bias
            and self.bias.is_distributed
        ):
            self.bias.split_axis = 0

        if self.config.use_recompute_loss_fn:
            logger.info(
                "Using recompute_loss_fn, the calculation of logits will be moved into "
                "loss_fn for memory optimization"
            )

    def forward(self, hidden_states, tensor_parallel_output=None):
        """Project hidden states to vocabulary logits.

        Args:
            hidden_states (paddle.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]
            tensor_parallel_output (Optional[bool]): Whether to output parallel results. Defaults to None.

        Returns:
            Union[
                Tuple[paddle.Tensor, paddle.Tensor, Optional[paddle.Tensor]]:
                    # When use_recompute_loss_fn or use_sparse_head_and_loss_fn
                    - hidden_states: Original input
                    - weight: Projection weights
                    - bias: Optional bias term
                Tuple[paddle.Tensor, paddle.Tensor, Optional[paddle.Tensor], bool]:  # With tensor_parallel_output
                    Same as above plus tensor_parallel_output flag
                paddle.Tensor:  # Normal case
                    Logits tensor of shape [batch_size, seq_len, vocab_size]
            ]
        """
        #  will enter this branch when:
        # 1. use_recompute_loss_fn or use_sparse_head_and_loss_fn
        # 2. dpo training
        if self.config.use_recompute_loss_fn or self.config.use_sparse_head_and_loss_fn:
            return (
                hidden_states,
                self.weight,
                self.bias,
                self.config.tie_word_embeddings,
            )

        return calc_lm_head_logits(
            self.config,
            hidden_states,
            self.weight,
            self.bias,
            tensor_parallel_output,
            training=self.training,
        )


class Ernie4_5DecoderLayer(nn.Layer):
    """A single transformer decoder layer in ERNIE model.

    Contains self-attention and feed-forward components,
    support, residual connections, and layer normalization.
    """

    def __init__(self, config, layer_idx):
        """Initialize the decoder layer.

        Args:
            config (PaddleOCRVLConfig): Model configuration.
            layer_idx (int): Index of this layer in the transformer stack
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.config = config

        self.self_attn = Ernie4_5Attention(config, layer_idx)
        self.mlp = Ernie4_5MLP(config)

        Norm = RMSNorm if config.use_rmsnorm else LayerNorm

        self.input_layernorm = Norm(config)
        self.post_attention_layernorm = Norm(config)

        self.residual_add1 = FusedDropoutImpl(
            config.hidden_dropout_prob, mode="upscale_in_train"
        )
        self.residual_add2 = FusedDropoutImpl(
            config.hidden_dropout_prob, mode="upscale_in_train"
        )

        if config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.post_attention_layernorm.weight)
            if not hasattr(config, "disable_ffn_model_parallel"):
                mark_as_sequence_parallel_parameter(self.input_layernorm.weight)
                if config.use_bias:
                    mark_as_sequence_parallel_parameter(self.self_attn.o_proj.bias)
                    mark_as_sequence_parallel_parameter(self.mlp.down_proj.bias)

            if not config.use_rmsnorm and config.use_bias:
                mark_as_sequence_parallel_parameter(self.post_attention_layernorm.bias)
                mark_as_sequence_parallel_parameter(self.input_layernorm.bias)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_embeddings: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_start_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """Forward pass through the decoder layer.

        Args:
            hidden_states (paddle.Tensor): Input tensor [batch_size, seq_len, hidden_size]
            position_embeddings (paddle.Tensor): Position embeddings
            attention_mask (Optional[paddle.Tensor]): Attention mask tensor
            attn_mask_start_row_indices (Optional[paddle.Tensor]): Indices for variable length attention
            position_ids (Optional[paddle.Tensor]): Position indices for rotary embeddings
            output_attentions (Optional[bool]): Whether to return attention weights
            past_key_value (Optional[Tuple[paddle.Tensor]]): Cached key/value states
            use_cache (Optional[bool]): Whether to cache key/value states

        Returns:
            Union: Various output combinations depending on arguments:
                - Base case: Hidden states tensor
                - With attention: Tuple of (hidden_states, attention_weights)
                - With cache: Tuple of (hidden_states, cached_key_value)
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        has_gradient = not hidden_states.stop_gradient
        if (
            self.config.recompute
            and self.config.recompute_granularity == "full_attn"
            and has_gradient
        ):
            hidden_states, self_attn_weights, present_key_value = recompute(
                self.self_attn,
                hidden_states,
                position_embeddings,
                past_key_value,
                attention_mask,
                attn_mask_start_row_indices,
                position_ids,
                output_attentions,
                use_cache,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                attn_mask_start_row_indices=attn_mask_start_row_indices,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=use_cache,
                token_type_ids=token_type_ids,
            )

        with self.model_parallel_dropout():
            hidden_states = self.residual_add1(hidden_states, residual)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        with self.model_parallel_dropout():
            hidden_states = self.residual_add2(hidden_states, residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def model_parallel_dropout(self):
        """Get context manager for model-parallel dropout with proper seed control.

        Returns:
            Context manager for dropout operation
        """
        if (
            self.config.tensor_parallel_degree > 1
            and self.config.hidden_dropout_prob > 0.0
        ):
            current_seed = (
                "local_seed" if self.config.sequence_parallel else "global_seed"
            )
            return get_rng_state_tracker().rng_state(current_seed)
        return contextlib.nullcontext()


class Ernie4_5PretrainedModel(PretrainedModel):
    """Base class for ERNIE pretrained models."""

    config_class = PaddleOCRVLConfig
    base_model_prefix = "ernie"


class Ernie4_5Model(Ernie4_5PretrainedModel):
    """The core ERNIE transformer model"""

    def __init__(self, config: PaddleOCRVLConfig):
        """Initialize the ERNIE model architecture.

        Args:
            config (PaddleOCRVLConfig): Model configuration.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.config = config

        if config.tensor_parallel_degree > 1:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )

        self.layers = nn.LayerList(
            [Ernie4_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        Norm = RMSNorm if config.use_rmsnorm else LayerNorm
        self.norm = Norm(config)
        self.rotary_emb = KeyeRotaryEmbedding(config=config)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        """Get the input embedding layer.

        Returns:
            nn.Embedding: The embedding layer for input tokens
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """Set new input embeddings.

        Args:
            value (nn.Embedding): New embedding layer to use
        """
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        attn_mask_start_row_indices=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
    ):
        """Forward pass through the ERNIE model.

        Args:
            input_ids (Optional[paddle.Tensor]): Input token IDs
            position_ids (Optional[paddle.Tensor]): Position indices
            attention_mask (Optional[paddle.Tensor]): Attention mask
            attn_mask_start_row_indices (Optional[paddle.Tensor]): Variable length attention indices
            inputs_embeds (Optional[paddle.Tensor]): Precomputed embeddings
            use_cache (Optional[bool]): Whether to cache key/value states
            past_key_values (Optional[Tuple[Tuple[paddle.Tensor]]]): Cached key/value states
            output_attentions (Optional[bool]): Whether to output attention weights
            output_hidden_states (Optional[bool]): Whether to output all hidden states
            return_dict (Optional[bool]): Whether to return dict or tuple

        Returns:
            Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
                Various outputs depending on configuration, including:
                - last_hidden_state: Final layer hidden states
                - past_key_values: Cached key/value states if use_cache=True
                - hidden_states: All hidden states if output_hidden_states=True
                - attentions: Attention weights if output_attentions=True
        """
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if batch_size != 1:
            raise NotImplementedError

        layers = self.layers[: self.config.num_hidden_layers]

        if past_key_values is None:
            past_key_values = tuple([None] * len(layers))

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.astype(self.embed_tokens.weight.dtype)

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        hidden_states = inputs_embeds

        if position_ids is None or position_ids.dim() == 2:
            raise NotImplementedError
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            past_key_values,
            output_attentions,
        )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, (decoder_layer) in enumerate(layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )
            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute
                and self.config.recompute_granularity == "full"
                and has_gradient
            ):
                layer_outputs = recompute(
                    decoder_layer,
                    hidden_states,
                    position_embeddings,
                    causal_mask,
                    attn_mask_start_row_indices,
                    position_ids,
                    token_type_ids,
                    output_attentions,
                    past_key_value,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings,
                    causal_mask,
                    attn_mask_start_row_indices,
                    position_ids,
                    token_type_ids,
                    output_attentions,
                    past_key_value,
                    use_cache,
                )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
        )

    def _update_causal_mask(
        self,
        attention_mask: paddle.Tensor,
        input_tensor: paddle.Tensor,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]],
        output_attentions: bool = False,
    ):

        if self.config.use_flash_attention:
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = (
                    attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                )
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Keye. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = (
            past_key_values[0][0].shape[1]
            if past_key_values is not None and past_key_values[0] is not None
            else 0
        )

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, paddle.Tensor)
            else past_seen_tokens + sequence_length + 1
        )
        cache_position = paddle.arange(
            past_seen_tokens, past_seen_tokens + sequence_length
        )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length: int,
        target_length: int,
        dtype,
        cache_position,
        batch_size: int,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = paddle.finfo(dtype).min
            causal_mask = paddle.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype
            )
            diagonal_attend_mask = paddle.arange(
                target_length
            ) > cache_position.reshape((-1, 1))
            diagonal_attend_mask = diagonal_attend_mask.astype(causal_mask.dtype)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand((batch_size, 1, -1, -1))
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                    :, None, None, :
                ].astype(causal_mask.dtype)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
        return causal_mask
