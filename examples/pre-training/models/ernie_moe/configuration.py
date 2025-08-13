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

""" Ernie model configuration"""
import logging
import json
from typing import Union
import paddle.distributed.communication.group

from paddleformers.transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

__all__ = [
    "ERNIE_PRETRAINED_INIT_CONFIGURATION",
    "ErnieMoEConfig",
]

ERNIE_PRETRAINED_INIT_CONFIGURATION = {
    "ernie/tiny-random-ernie": {
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 2048,
        "model_type": "ernie",
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-06,
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "use_cache": False,
        "use_recompute": False,
        "use_flash_attn": True,
        "use_pure_fp16": False,
    },
}


class ErnieConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~ErnieModel`]. It is used to instantiate an Ernie
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Ernie-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Ernie model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~ErnieModel`] or [`~TFErnieModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        Example:
    ```python
    >>> from paddleformers.transformer import ErnieModel, ErnieConfig

    >>> # Initializing a Ernie ernie-7b style configuration
    >>> configuration = ErnieConfig()

    >>> # Initializing a model from the ernie-7b style configuration
    >>> model = ErnieModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ernie"
    attribute_map = {
        "n_positions": "max_position_embeddings",
        "n_embd": "hidden_size",
        "n_layer": "num_hidden_layers",
        "n_head": "num_attention_heads",
        "n_inner": "intermediate_size",
        "activation_function": "hidden_act",
    }
    pretrained_init_configuration = ERNIE_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=11008,
        max_position_embeddings=32768,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=None,
        initializer_range=0.02,  # no use
        rms_norm_eps=1e-6,
        use_cache=False,
        use_flash_attn=True,
        use_mem_eff_attn=False,
        use_flash_attn_with_mask=False,
        use_recompute=False,
        use_recompute_attn=False,
        recompute_use_reentrant=False,
        use_rmsnorm=True,
        z_loss_lambda=None,
        fuse_rms_norm=False,
        fuse_ln=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        fuse_attn_ffn=False,
        fuse_swiglu=False,
        use_bias=False,
        expert_mlp_use_bias=None,
        rope_reorder=True,
        rope_theta=10000,
        fuse_rope=False,
        use_fast_ln=False,
        weight_share_add_bias=True,
        fuse_linear=False,
        seqlen=False,
        ignored_index=-100,
        remove_tail_layer=False,
        use_recompute_lm_head=False,
        use_recompute_loss_fn=False,
        use_recompute_mtp=False,
        use_recompute_dnd=False,
        selective_no_recompute_num=0,
        use_mp_gathered_weight=False,
        refined_recompute=dict(),
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        compression_ratio: float = 1.0,
        quant_bits=-1,
        num_key_value_heads=None,
        submatrix_parallel=False,
        submatrix_parallel_low_memory=True,
        use_sparse_head_and_loss_fn=False,
        using_dynamic_sequence_length=False,
        micro_batch_size=-1,
        using_precision_check=False,
        use_qk_norm=False,
        use_tpsp_comm_overlap=False,
        offload_pp_data_chunk_size=0,
        use_fused_head_loss_fn=False,
        use_recompute_resampler=False,
        resampler_fuse_rms_norm=False,
        token_loss_equal_weight=False,
        token_balance_loss=False,
        token_balance_seqlen=False,
        use_fp8=False,
        fp8_configs=dict(),
        use_fp8_mlp=False,
        fp8_mem_configs=dict(),
        fp8_fused_ops_configs=dict(),
        drop_before_deepep=False,
        deepep_drop_padding=False,
        disable_pipeline_warmup=False,
        skip_align_position_id=False,
        rope_3d=False,
        freq_allocation=0,
        moe_layer_feed_fake_token=False,
        decoderlayer_act_offload_settings={"type": "", "value": ""},
        loss_subbatch_seqlen=32768,
        gate_force_zero_padding_grad=False,
        recompute_num_layers=None,
        use_combine_before_a2a=False,
        use_quant_before_a2a=False,
        rope_yarn_config={},
        **kwargs,
    ):
        if "tie_word_embeddings" not in kwargs:
            kwargs["tie_word_embeddings"] = False
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.use_recompute_attn = use_recompute_attn
        if use_recompute_attn:
            logger.warning("set `use_recompute_attn`=True, disabling `use_recompute`")
            use_recompute = False
        self.use_recompute = use_recompute
        self.recompute_num_layers = (
            recompute_num_layers
            if recompute_num_layers is not None
            else num_hidden_layers
        )
        self.use_flash_attn = use_flash_attn
        self.recompute_use_reentrant = recompute_use_reentrant
        self.use_mem_eff_attn = use_mem_eff_attn
        self.use_flash_attn_with_mask = use_flash_attn_with_mask
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.fuse_attn_ffn = fuse_attn_ffn
        self.fuse_swiglu = fuse_swiglu
        self.fuse_rms_norm = fuse_rms_norm
        self.fuse_ln = fuse_ln
        self.use_rmsnorm = use_rmsnorm
        self.z_loss_lambda = z_loss_lambda
        self.using_dynamic_sequence_length = using_dynamic_sequence_length
        if using_dynamic_sequence_length:
            assert (
                micro_batch_size > 0
            ), "micro_batch_size should be set when using_dynamic_sequence_length"
        self.micro_batch_size = micro_batch_size
        self.using_precision_check = using_precision_check
        self.use_qk_norm = use_qk_norm

        self.seqlen = seqlen
        self.use_bias = use_bias
        self.weight_share_add_bias = weight_share_add_bias
        self.rope_reorder = rope_reorder
        self.rope_yarn_config = rope_yarn_config
        self.rope_theta = rope_theta
        self.fuse_rope = fuse_rope
        self.use_fast_ln = use_fast_ln

        self.fuse_linear = fuse_linear
        self.ignored_index = ignored_index
        self.remove_tail_layer = remove_tail_layer
        self.use_recompute_lm_head = use_recompute_lm_head
        self.use_recompute_loss_fn = use_recompute_loss_fn
        self.use_recompute_mtp = use_recompute_mtp
        self.use_recompute_dnd = use_recompute_dnd

        self.use_mp_gathered_weight = use_mp_gathered_weight
        self.selective_no_recompute_num = selective_no_recompute_num  # only PP

        self.refined_recompute = refined_recompute
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.compression_ratio = compression_ratio
        self.skip_recompute_ops = dict()
        self.quant_bits = quant_bits
        self.num_key_value_heads = num_key_value_heads
        self.submatrix_parallel = submatrix_parallel
        self.submatrix_parallel_low_memory = submatrix_parallel_low_memory
        self.use_sparse_head_and_loss_fn = use_sparse_head_and_loss_fn
        self.use_tpsp_comm_overlap = use_tpsp_comm_overlap
        self.offload_pp_data_chunk_size = offload_pp_data_chunk_size
        self.use_fused_head_loss_fn = use_fused_head_loss_fn
        self.use_recompute_resampler = use_recompute_resampler
        self.resampler_fuse_rms_norm = resampler_fuse_rms_norm
        self.token_balance_loss = token_balance_loss
        self.token_balance_seqlen = token_balance_seqlen
        self.rope_3d = rope_3d
        self.freq_allocation = freq_allocation
        self.decoderlayer_act_offload_settings = decoderlayer_act_offload_settings
        self.loss_subbatch_seqlen = loss_subbatch_seqlen
        self.gate_force_zero_padding_grad = gate_force_zero_padding_grad

        # 默认的 fp8 设置
        default_fp8_configs = {
            "quant_scheme": "DelayedScaling",
            "recipe": {
                "format": "hybrid",
                "calibrating": True,
                "amax_history_len": 1024,
                "amax_compute_algo": "max",
                "fuse_wgrad_accumulation": False,
                "quant_weight_at_first_microbatch": False,
            },
            "layers": {
                "attn_fc1_linear": True,
                "attn_fc2_linear": True,
                "mlp_fc1_linear": True,
                "mlp_fc2_linear": True,
                "attn_tp_fc1_linear": True,
                "attn_tp_fc2_linear": True,
                "mlp_tp_fc1_linear": True,
                "mlp_tp_fc2_linear": True,
            },
            "smooth_swiglu": False,
        }

        def update_nested_dict(default_dict, update_dict):
            for key, value in update_dict.items():
                if (
                    isinstance(value, dict)
                    and key in default_dict
                    and isinstance(default_dict[key], dict)
                ):
                    update_nested_dict(default_dict[key], value)
                else:
                    default_dict[key] = value

        # 更新默认设置
        update_nested_dict(default_fp8_configs, fp8_configs)
        self.fp8_configs = default_fp8_configs
        self.use_fp8 = use_fp8
        self.expert_mlp_use_bias = expert_mlp_use_bias
        self.use_fp8_mlp = use_fp8_mlp
        default_fp8_mem_configs = {
            "shared_expert": False,
            "recompute_fwd_gate_up": False,
            "dequant_input": False,
        }
        update_nested_dict(default_fp8_mem_configs, fp8_mem_configs)
        self.fp8_mem_configs = default_fp8_mem_configs
        default_fp8_fused_ops_configs = {
            "stack_quant": False,
            "swiglu_probs_bwd": False,
            "split_group_gemm": True,
        }
        update_nested_dict(default_fp8_fused_ops_configs, fp8_fused_ops_configs)
        self.fp8_fused_ops_configs = default_fp8_fused_ops_configs
        self.drop_before_deepep = drop_before_deepep
        self.deepep_drop_padding = deepep_drop_padding
        self.disable_pipeline_warmup = disable_pipeline_warmup
        self.skip_align_position_id = skip_align_position_id
        self.moe_layer_feed_fake_token = moe_layer_feed_fake_token

        if self.sequence_parallel:
            assert (
                self.using_dynamic_sequence_length or self.seqlen
            ), "seqlen not provided in sequence-parallel when not using dygramic sequence length"

            assert (
                self.tensor_parallel_degree > 1
            ), f"senquence-parallel only works in mp, got mp={self.tensor_parallel_degree}"

        self.register_nonsaveable_keys("use_recompute")
        self.register_nonsaveable_keys("recompute_use_reentrant")
        self.register_nonsaveable_keys("refined_recompute")
        self.register_nonsaveable_keys("use_recompute_attn")
        self.register_nonsaveable_keys("use_recompute_lm_head")
        self.register_nonsaveable_keys("use_recompute_mtp")
        self.register_nonsaveable_keys("use_recompute_dnd")
        self.register_nonsaveable_keys("use_recompute_loss_fn")
        self.register_nonsaveable_keys("using_precision_check")
        self.register_nonsaveable_keys("skip_recompute_ops")

    def __setattr__(self, name: str, value):
        super().__setattr__(name, value)
        if getattr(self, "use_recompute", False):
            assert not getattr(
                self, "use_recompute_attn", False
            ), "cannot set `use_recompute_attn=True` when `use_recompute=True`"

    def register_nonsaveable_keys(self, keys):

        if hasattr(super(), "register_nonsaveable_keys"):
            return super().register_nonsaveable_keys(keys)
        elif hasattr(super(), "register_unsavable_keys"):
            return super().register_unsavable_keys(keys)
        else:
            raise AttributeError(
                "register_nonsaveable_keys not found in PretrainedConfig"
            )


class ErnieMoEConfig(ErnieConfig):
    r"""
    This is the configuration class to store the configuration of a [`~ErnieModel`]. It is used to instantiate an Ernie
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Ernie-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Ernie model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~ErnieModel`] or [`~TFErnieModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        Example:
    ```python
    >>> from paddleformers.transformer import ErnieModel, ErnieConfig

    >>> # Initializing a Ernie ernie-7b style configuration
    >>> configuration = ErnieConfig()

    >>> # Initializing a model from the ernie-7b style configuration
    >>> model = ErnieModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ernie"
    attribute_map = {
        "n_positions": "max_position_embeddings",
        "n_embd": "hidden_size",
        "n_layer": "num_hidden_layers",
        "n_head": "num_attention_heads",
        "n_inner": "intermediate_size",
        "activation_function": "hidden_act",
    }
    pretrained_init_configuration = ERNIE_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        moe_num_experts: Union[int, list] = 0,
        use_fake_gate=False,
        use_recompute_moe=False,
        moe_capacity=(),
        moe_layer_interval=2,
        moe_layer_start_index: Union[int, list] = 0,
        moe_layer_end_index: Union[int, list] = -1,
        moe_aux_loss_lambda=1e-2,
        moe_z_loss_lambda=1e-4,
        moe_orthogonal_loss_lambda=1e-2,
        moe_use_size_all2all=False,
        sinkhorn_2gate=True,
        sinkhorn_temp=3e-2,
        global_aux_loss=False,
        moe_dropout_prob=0.0,
        moe_group="world",
        moe_gate="top2",
        moe_num_attn_experts=False,
        moe_logging=False,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: Union[int, list] = 0,
        moe_num_shared_experts: int = 0,
        moe_num_dense_experts: int = 0,
        moe_dense_experts_token_type_id: int = 3,
        moe_multimodal_dispatch_use_allgather: str = "",
        moe_multimodal_paired_experts: bool = False,
        moe_reverse_token_drop: bool = False,
        moe_gate_act: str = "softmax",
        moe_norm_gate_logits=True,
        moe_use_hard_gate: bool = False,
        moe_use_bpr: bool = False,
        moe_fuse_experts: bool = False,
        moe_all_to_all_dropout: float = 0.0,
        moe_use_token_type_bias: bool = False,
        moe_k=2,
        moe_use_aux_free: bool = False,
        moe_group_experts: bool = False,
        moe_group_orthogonal_loss: bool = False,
        moe_with_send_router_loss: bool = True,
        enable_delay_scale_loss: bool = True,
        num_acc_steps: int = None,
        insert_empty_layer: list = None,
        pp_no_recompute_layer: list = None,
        multi_token_pred_depth: int = 0,
        multi_token_pred_lambda: float = 0.3,
        fuse_gate_detach_matmul: bool = False,
        enable_mtp_magic_send: bool = False,
        use_elastic_topk: bool = False,
        use_deepep: bool = False,
        use_elastic_expert_num: bool = False,
        elastic_min_expert_num: int = 0,
        all_expert_ratio: float = 1.0,
        use_elastic_topk_for_mbs: bool = False,
        elastic_min_topk: int = 1,
        elastic_max_topk: int = None,
        n_group: int = 0,
        topk_group: int = 0,
        scaling_factor: float = None,
        aux_loss_type: str = "",
        deepep_fine_grained: bool = False,
        deepep_use_fused: bool = False,
        deepep_tokens_per_subbatch: int = 0,
        use_linear_residual_norm_recompute: bool = False,
        use_rms_qkv_recompute: bool = False,
        build_skip_comm_buffer: bool = False,
        use_norm_gate_recompute: bool = False,
        moe_state_dict_use_global_expert_id: bool = False,
        enable_entropy_logging: bool = False,
        use_fp8_fuse_node: bool = False,
        use_combine_before_a2a: bool = False,
        use_fp8_dispatch_a2a: bool = False,
        use_ep_comm_overlap: bool = False,
        **kwargs,
    ):
        """
        config
        """
        if use_recompute_moe:
            logger.warning("set `use_recompute_moe`=True, disabling `use_recompute`")
            kwargs["use_recompute"] = False
        super().__init__(**kwargs)
        # moe
        self.use_fake_gate = use_fake_gate
        self.use_recompute_moe = use_recompute_moe
        self.moe_num_experts = moe_num_experts
        self.moe_capacity = moe_capacity
        self.moe_aux_loss_lambda = moe_aux_loss_lambda
        self.moe_z_loss_lambda = moe_z_loss_lambda
        self.moe_orthogonal_loss_lambda = moe_orthogonal_loss_lambda
        self.global_aux_loss = global_aux_loss
        self.sinkhorn_2gate = sinkhorn_2gate
        self.sinkhorn_temp = sinkhorn_temp
        self.moe_layer_interval = moe_layer_interval
        self.moe_dropout_prob = moe_dropout_prob
        self.moe_group = moe_group
        self.moe_gate = moe_gate
        self.moe_num_attn_experts = moe_num_attn_experts
        # implemtent size-all2all as https://arxiv.org/pdf/2303.06182.pdf
        self.moe_use_size_all2all = moe_use_size_all2all
        self.moe_logging = moe_logging
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_num_shared_experts = moe_num_shared_experts
        self.moe_num_dense_experts = moe_num_dense_experts
        self.moe_dense_experts_token_type_id = moe_dense_experts_token_type_id
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_reverse_token_drop = moe_reverse_token_drop
        self.moe_use_hard_gate = moe_use_hard_gate
        self.moe_fuse_experts = moe_fuse_experts
        self.moe_k = moe_k
        self.moe_all_to_all_dropout = moe_all_to_all_dropout
        self.moe_use_token_type_bias = moe_use_token_type_bias
        self.moe_use_bpr = moe_use_bpr
        self.moe_group_experts = moe_group_experts
        self.moe_group_orthogonal_loss = moe_group_orthogonal_loss
        # optimize send without router loss
        self.moe_with_send_router_loss = moe_with_send_router_loss
        self.enable_delay_scale_loss = enable_delay_scale_loss
        self.num_acc_steps = num_acc_steps
        self.moe_layer_start_index = moe_layer_start_index
        self.moe_layer_end_index = (
            self.num_hidden_layers - 1
            if moe_layer_end_index == -1
            else moe_layer_end_index
        )
        self.moe_multimodal_dispatch_use_allgather = (
            moe_multimodal_dispatch_use_allgather
        )
        self.moe_multimodal_paired_experts = moe_multimodal_paired_experts
        self.moe_gate_act = moe_gate_act
        self.moe_norm_gate_logits = moe_norm_gate_logits
        self.moe_use_aux_free = moe_use_aux_free
        self.fuse_gate_detach_matmul = fuse_gate_detach_matmul
        if insert_empty_layer is not None:
            assert isinstance(
                insert_empty_layer, list
            ), "insert_empty_layer should be a list"
        else:
            insert_empty_layer = []

        # Overlap A2A communication with shared expert and auxiliary loss.
        self.use_ep_comm_overlap = use_ep_comm_overlap
        # Move the combine operation before A2A communication.
        self.use_combine_before_a2a = use_combine_before_a2a
        # Use FP8 for dispatch communication.
        self.use_fp8_dispatch_a2a = use_fp8_dispatch_a2a

        # Multi-Token Prediction (MTP)
        self.multi_token_pred_depth = multi_token_pred_depth
        self.multi_token_pred_lambda = multi_token_pred_lambda
        self.enable_mtp_magic_send = enable_mtp_magic_send

        # The insert_empty_layer is a list of integer which will be used under pipeline parallel.
        # After each layer indicated in the insert_empty_layer, an empty layer will be inserted.
        # For example, a model with 4 layers, insert_empty_layer = [1, 3], the model actually passed to
        # pp is: transformer, transformer, EMPTY, transformer, transformer, EMPTY
        self.insert_empty_layer = insert_empty_layer

        # elastic
        self.use_elastic_topk = use_elastic_topk
        self.use_elastic_expert_num = use_elastic_expert_num
        self.elastic_min_expert_num = elastic_min_expert_num
        self.all_expert_ratio = all_expert_ratio
        self.use_elastic_topk_for_mbs = use_elastic_topk_for_mbs
        self.elastic_min_topk = elastic_min_topk
        if elastic_max_topk is None:
            self.elastic_max_topk = self.moe_k * 2 - 1

        # Using fusion expert node in moe layer.
        self.use_fp8_fuse_node = use_fp8_fuse_node

        # Perform MoE computation at expert granularity.
        self.deepep_fine_grained = deepep_fine_grained
        # Requires deepep_fine_grained to be enabled; further disperses token
        # granularity within experts to compute subbatches.
        self.deepep_tokens_per_subbatch = deepep_tokens_per_subbatch
        # Fuse combine and scatter operations when using BF16 for expert computation.
        self.deepep_use_fused = deepep_use_fused

        assert not (
            self.use_combine_before_a2a and self.use_deepep
        ), "combine_before_a2a is not supported for deepep now."

        assert not (
            self.use_fp8_dispatch_a2a and not self.use_fp8_fuse_node
        ), "fp8_dispatch_a2a must be used with use_fp8_fuse_node."

        assert not (
            self.use_fp8_dispatch_a2a and self.use_ep_comm_overlap
        ), "fp8_dispatch_a2a connot be used with use_ep_comm_overlap."

        if self.deepep_tokens_per_subbatch:
            assert (
                self.deepep_fine_grained
            ), "deepep_fine_grained must be enabled when deepep_tokens_per_subbatch is set."

        # node limit routing
        self.n_group = n_group
        self.topk_group = topk_group

        # router scaling_factor
        self.scaling_factor = scaling_factor

        self.build_skip_comm_buffer = build_skip_comm_buffer

        # router loss type
        assert aux_loss_type in ["", "default", "seq_aux_loss", "switch_aux_loss"]
        self.aux_loss_type = aux_loss_type

        self.use_deepep = use_deepep
        if self.moe_multimodal_paired_experts and isinstance(
            self.moe_num_experts, (tuple, list)
        ):
            logger.warning(
                "moe_num_experts must be one element when using paired experts"
            )
            self.moe_num_experts = self.moe_num_experts[0]

        if pp_no_recompute_layer is not None:
            assert isinstance(
                insert_empty_layer, list
            ), "pp_no_recompute_layer should be a list"

            # Indicating layers not do recompute under pipeline parallel.
            # Note that, when insert_empty_layer is not None, the pp_no_recompute_layer should be indicating
            # layers number in origin model structure, AKA model before insert empty layers.
        self.pp_no_recompute_layer = pp_no_recompute_layer
        self.register_nonsaveable_keys("moe_group")
        self.register_nonsaveable_keys("pp_no_recompute_layer")

        if (
            self.moe_group in ["dp", "data"]
            and self.moe_multimodal_dispatch_use_allgather
        ):
            assert (
                self.moe_num_shared_experts == 0
            ), "shared experts are not supported when using dp moe and moe_allgather_layer"
            assert (
                self.moe_num_dense_experts == 0
            ), "dense experts are not supported when using dp moe and moe_allgather_layer"

        self.use_linear_residual_norm_recompute = use_linear_residual_norm_recompute
        self.use_rms_qkv_recompute = use_rms_qkv_recompute
        self.use_norm_gate_recompute = use_norm_gate_recompute
        self.moe_state_dict_use_global_expert_id = moe_state_dict_use_global_expert_id
        self.enable_entropy_logging = enable_entropy_logging

    @property
    def multimodel_experts(self) -> bool:

        return (
            isinstance(self.moe_num_experts, (tuple, list))
            and len(self.moe_num_experts) > 1
        )

    @property
    def use_moe(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """
        return (
            sum(self.moe_num_experts) > 0
            if self.multimodel_experts
            else self.moe_num_experts > 0
        )

    def __setattr__(self, name: str, value):
        super().__setattr__(name, value)
        if getattr(self, "use_recompute", False):
            assert not getattr(
                self, "use_recompute_moe", False
            ), "cannot set `use_recompute_moe=True` when `use_recompute=True`"

    def to_json_string(self, use_diff: bool = True) -> str:

        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()

        def _serializer(obj):
            if isinstance(obj, paddle.distributed.communication.group.Group):
                return repr(obj)
            raise TypeError(f"Type {type(obj)} is not serializable")

        return (
            json.dumps(
                config_dict,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                default=_serializer,
            )
            + "\n"
        )
