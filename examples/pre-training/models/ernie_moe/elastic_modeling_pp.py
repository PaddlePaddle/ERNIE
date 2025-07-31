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
docstring
"""

import logging
import paddle
from models.sequence_parallel_utils import ScatterOp
from models.ernie_moe.modeling import (
    ErnieModel,
    ErnieDecoderLayer,
    RMSNorm,
    LayerNorm,
    ErnieMoELMHead,
)
from paddle.distributed.fleet.utils import recompute
from models.ernie.modeling import te_recompute

try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}

from models.ernie_moe.modeling_pp import (
    ErnieMoEForCausalLMPipe,
    ErnieEmbeddingPipe,
    ErnieDecoderLayerPipe,
    RMSNormPipe,
    LayerNormPipe,
    ErnieMoELMHeadPipe,
    MTPEmbeddingPipe,
    MTPLayer,
)

logger = logging.getLogger(__name__)

try:
    from models.ernie_moe.modeling_pp import input_ids_for_mtp
except ImportError:
    logger.error("input_ids_for_mtp not found. Update ErnieCore to use this opt")
    input_ids_for_mtp = None


class ErnieEmbeddingElasticPipe(ErnieEmbeddingPipe):
    """Extends ErnieEmbeddingPipe to forward attention_mask through the pipeline."""

    def forward(self, args):
        """
        elastic forward
        """
        if isinstance(args, tuple):
            if len(args) == 5:
                (
                    input_ids,
                    layer_elastic,
                    layer_use_info,
                    attention_mask,
                    position_ids,
                ) = args
                inbatch_pack_offset = None
            elif len(args) == 4:
                if self.use_mem_eff_attn:
                    input_ids, layer_elastic, layer_use_info, inbatch_pack_offset = args
                    position_ids, attention_mask = None, None
                    inbatch_pack_offset.stop_gradient = True
                else:
                    input_ids, layer_elastic, layer_use_info, attention_mask = args
                    position_ids = None
                    inbatch_pack_offset = None
            elif len(args) == 3:
                input_ids, layer_elastic, layer_use_info = args
                attention_mask, position_ids, inbatch_pack_offset = None, None, None
        else:
            input_ids, layer_elastic, layer_use_info = args
            attention_mask, position_ids, inbatch_pack_offset = None, None, None

        layer_elastic.stop_gradient = True
        layer_use_info.stop_gradient = True

        if position_ids is not None:
            position_ids.stop_gradient = True

        emb = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)

        if self.config.multi_token_pred_depth > 0:
            if self.config.enable_mtp_magic_send:
                emb = emb[:, : -self.config.multi_token_pred_depth, :]
                if self.sequence_parallel:
                    emb = emb.reshape([-1, emb.shape[-1]])
                    emb = ScatterOp.apply(emb)
            else:
                assert False, "elastic only support enable_mtp_magic_send now"
                # inputs_embeds_extra = emb[:, -self.config.multi_token_pred_depth :, :]  # [B, S, D]
                # inputs_embeds = emb[:, : -self.config.multi_token_pred_depth, :]
                # inputs_embeds_ori = inputs_embeds
                # batch_size, seq_length, _ = inputs_embeds.shape
                #
                # if self.sequence_parallel:
                #     inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
                #     inputs_embeds = ScatterOp.apply(inputs_embeds)
                # mtp_emb_res = [inputs_embeds]
                # for depth in range(self.config.multi_token_pred_depth):
                #     inputs_embeds_mtp = paddle.concat(
                #         [inputs_embeds_ori[:, (depth + 1) :, :], inputs_embeds_extra[:, : (depth + 1), :]], axis=1
                #     )
                #     if self.sequence_parallel:
                #         inputs_embeds_mtp = inputs_embeds_mtp.reshape([-1, inputs_embeds_mtp.shape[-1]])
                #         inputs_embeds_mtp = ScatterOp.apply(inputs_embeds_mtp)
                #     mtp_emb_res.append(inputs_embeds_mtp)
                # res = paddle.concat(mtp_emb_res)
                # return [res], layer_elastic.clone(), layer_use_info.clone()
        else:
            if self.sequence_parallel:
                emb = emb.reshape([-1, emb.shape[-1]])
                emb = ScatterOp.apply(emb)

        if attention_mask is not None:
            batch_size, seq_length = input_ids.shape
            attention_mask = ErnieModel._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), 0, emb.dtype
            )
            attention_mask.stop_gradient = True

        ret = (emb,)

        ret += (layer_elastic.clone(),)
        ret += (layer_use_info.clone(),)

        if attention_mask is not None:
            ret += (attention_mask.clone(),)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if inbatch_pack_offset is not None:
            ret += (inbatch_pack_offset.clone(),)
        if (
            self.config.multi_token_pred_depth > 0
            and not self.config.enable_mtp_magic_send
        ):
            ret += (input_ids,)
            assert len(ret) == 2, "mtp only support one input which is input_ids"
        if len(ret) == 1:
            ret = ret[0]
        return ret


class ErnieDecoderLayerElasticPipe(ErnieDecoderLayerPipe):
    """ErnieDecoderLayerElasticPipe"""

    def forward(self, args):
        """
        elastic forward
        """
        if (
            self.config.multi_token_pred_depth > 0
            and not self.config.enable_mtp_magic_send
        ):
            assert False, "elastic only support enable_mtp_magic_send now"
            # res = args[0]
            # tensor_list = paddle.split(res, self.config.multi_token_pred_depth + 1)
            # inputs_embeds = tensor_list[-self.config.multi_token_pred_depth :]
            # args[0] = tuple(tensor_list[: -self.config.multi_token_pred_depth])

        if isinstance(args, tuple):
            if len(args) == 5:  # layer_elastic bool layer_use_info [layer, data_num]
                (
                    hidden_states,
                    layer_elastic,
                    layer_use_info,
                    attention_mask,
                    position_ids,
                ) = args
            elif len(args) == 4:
                if self.use_mem_eff_attn:
                    (
                        hidden_states,
                        layer_elastic,
                        layer_use_info,
                        inbatch_pack_offset,
                    ) = args
                    position_ids, attention_mask = None, None
                    inbatch_pack_offset.stop_gradient = True
                else:
                    hidden_states, layer_elastic, layer_use_info, attention_mask = args
                    position_ids, inbatch_pack_offset = None, None
            elif len(args) == 3:
                hidden_states, layer_elastic, layer_use_info = args[:3]
                attention_mask, position_ids, inbatch_pack_offset = None, None, None

        else:
            hidden_states = args
            attention_mask, position_ids, inbatch_pack_offset = None, None, None

        layer_elastic.stop_gradient = True
        layer_use_info.stop_gradient = True

        if position_ids is not None:
            position_ids.stop_gradient = True

        if attention_mask is not None:
            attention_mask.stop_gradient = True

        # is_layer_elastic = layer_elastic.item()
        # if is_layer_elastic:
        #     print(f"elastic info get in elastic")
        out_hidden_states = hidden_states.clone()
        use_indices = paddle.nonzero(layer_use_info[self.layer_idx]).flatten()
        use_hidden_states = paddle.gather(hidden_states, use_indices, axis=0)

        if self.training and self.use_full_recompute:
            decoderlayer_act_offload_settings = self.config.get(
                "decoderlayer_act_offload_settings", {"type": "", "value": ""}
            )
            setting_type = decoderlayer_act_offload_settings["type"]
            offload_value = decoderlayer_act_offload_settings["value"]
            offload_kwargs = {}
            if "mod" == setting_type:
                assert isinstance(offload_value, (list, tuple))
                v1, v2 = offload_value
                offload_kwargs["offload_indices"] = (
                    [0] if self.layer_idx % v1 == v2 else []
                )
            elif "layer_idxs" == setting_type:
                offload_kwargs["offload_indices"] = (
                    [0] if self.layer_idx in offload_value else []
                )

            recompute_func = te_recompute if self.config.use_fp8 else recompute
            ret = recompute_func(
                ErnieDecoderLayer.forward,
                self,
                use_hidden_states,
                attention_mask,
                position_ids,
                None,  # token-type
                False,  # output-attention
                None,  # past-kv-cache
                False,  # use-cache
                inbatch_pack_offset,
                False,  # output_gate_logits
                **offload_kwargs,
            )
        else:
            ret = ErnieDecoderLayer.forward(
                self,
                use_hidden_states,
                attention_mask,
                position_ids,
                None,  # token-type
                False,  # output-attention
                None,  # past-kv-cache
                False,  # use-cache
                inbatch_pack_offset,
                False,  # output_gate_logits
            )
        if isinstance(ret, paddle.Tensor):
            ret = paddle.scatter(
                out_hidden_states, use_indices, ret, overwrite=True
            ).cast(ret.dtype)
            ret = (ret,)
            ret += (layer_elastic.clone(),)
            ret += (layer_use_info.clone(),)
        # print(f"elastic info ret is {ret}, layer_idx is {self.layer_idx}")

        if attention_mask is not None:
            ret += (attention_mask.clone(),)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if inbatch_pack_offset is not None:
            ret += (inbatch_pack_offset.clone(),)
        if self.config.multi_token_pred_depth > 0:
            # must tuple if `multi_token_pred_depth > 0`
            if not self.config.enable_mtp_magic_send:
                assert False, "elastic only support enable_mtp_magic_send now"
                # ret = (paddle.concat([ret[0], *inputs_embeds]), ret[1:])
        return ret


class RMSNormElasticPipe(RMSNormPipe):
    """RMSNormElasticPipe"""

    def forward(self, args):
        """
        elastic forward
        """
        if self.config.multi_token_pred_depth > 0:
            if self.config.enable_mtp_magic_send:
                assert (
                    len(args) == self.config.multi_token_pred_depth + 1
                ), "the length is not valid in mtp"
                mtp_outputs = []
                # 已在mtp layer做完elastic slice操作
                for hidden_states in args:
                    hidden_states = RMSNorm.forward(self, hidden_states)
                    mtp_outputs.append(hidden_states)
                return mtp_outputs
            else:
                assert False, "elastic only support enable_mtp_magic_send now"
                # tensor_list = paddle.split(args[0], self.config.multi_token_pred_depth + 1)
                # mtp_outputs = []
                # for hidden_states in tensor_list:
                #     mtp_outputs.append(RMSNorm.forward(self, hidden_states))
                # return mtp_outputs
        else:
            if self.use_moe:
                hidden_states, layer_elastic, layer_use_info = args[:3]
            if isinstance(args, tuple):
                if len(args) == 5:
                    (
                        hidden_states,
                        layer_elastic,
                        layer_use_info,
                        attention_mask,
                        position_ids,
                    ) = args
                elif len(args) == 4:
                    hidden_states, layer_elastic, layer_use_info, attention_mask = args
                    position_ids = None
            else:
                hidden_states, layer_elastic, layer_use_info = args
                attention_mask, position_ids = None, None

            layer_elastic.stop_gradient = True
            layer_use_info.stop_gradient = True

            is_layer_elastic = layer_elastic.item()
            out_hidden_states = hidden_states.clone()
            if not is_layer_elastic:
                use_indices = paddle.nonzero(layer_use_info[0]).flatten()
                hidden_states = paddle.gather(hidden_states, use_indices, axis=0)
            hidden_states = RMSNorm.forward(self, hidden_states)
            if not is_layer_elastic:
                out_hidden_states = paddle.scatter(
                    out_hidden_states, use_indices, hidden_states, overwrite=True
                ).cast(hidden_states.dtype)
                hidden_states = out_hidden_states
            return hidden_states, layer_elastic.clone(), layer_use_info.clone()


class LayerNormElasticPipe(LayerNormPipe):
    """LayerNormElasticPipe"""

    def forward(self, args):
        """
        elastic forward
        """
        if self.config.multi_token_pred_depth > 0:
            if self.config.enable_mtp_magic_send:
                assert (
                    len(args) == self.config.multi_token_pred_depth + 1
                ), "the length is not valid in mtp"
                mtp_outputs = []
                # 已在mtp layer做完elastic slice操作
                for hidden_states in args:
                    hidden_states = LayerNorm.forward(self, hidden_states)
                    mtp_outputs.append(hidden_states)
                return mtp_outputs
            else:
                assert False, "elastic only support enable_mtp_magic_send now"
                # tensor_list = paddle.split(args, self.config.multi_token_pred_depth + 1)
                # mtp_outputs = []
                # for hidden_states in tensor_list:
                #     mtp_outputs.append(LayerNorm.forward(self, hidden_states))
                # return mtp_outputs
        else:
            if self.use_moe:
                hidden_states, layer_elastic, layer_use_info = args[:3]

            if isinstance(args, tuple):
                if len(args) == 5:
                    (
                        hidden_states,
                        layer_elastic,
                        layer_use_info,
                        attention_mask,
                        position_ids,
                    ) = args
                elif len(args) == 4:
                    hidden_states, layer_elastic, layer_use_info, attention_mask = args
                    position_ids = None
                elif len(args) == 3:
                    hidden_states, layer_elastic, layer_use_info = args[:3]
                    attention_mask, position_ids = None, None
            else:
                hidden_states, layer_elastic, layer_use_info = args
                attention_mask, position_ids = None, None

            layer_elastic.stop_gradient = True
            layer_use_info.stop_gradient = True
            is_layer_elastic = layer_elastic.item()
            if not is_layer_elastic:
                out_hidden_states = hidden_states.clone()
                use_indices = paddle.nonzero(layer_use_info[0]).flatten()
                hidden_states = paddle.gather(hidden_states, use_indices, axis=0)
            hidden_states = LayerNorm.forward(self, hidden_states)
            if not is_layer_elastic:
                out_hidden_states = paddle.scatter(
                    out_hidden_states, use_indices, hidden_states, overwrite=True
                ).cast(hidden_states.dtype)
                hidden_states = out_hidden_states
        return hidden_states, layer_elastic.clone(), layer_use_info.clone()


class ErnieMoELMHeadElasticPipe(ErnieMoELMHeadPipe):
    """ErnieMoELMHeadElasticPipe"""

    def forward(self, args):
        """
        elastic forward
        """
        if self.config.multi_token_pred_depth > 0:
            logits = list()
            for _hidden_states in args:
                logits.append(ErnieMoELMHead.forward(self, _hidden_states))
            return logits
        else:
            hidden_states, layer_elastic, layer_use_info = args
            is_layer_elastic = layer_elastic.item()
            # print(f"elastic info is_layer_elastic {is_layer_elastic}, hidden_states is {hidden_states}")
            if not is_layer_elastic:
                use_indices = paddle.nonzero(layer_use_info[0]).flatten()
                hidden_states = paddle.gather(hidden_states, use_indices, axis=0)
            logits = ErnieMoELMHead.forward(self, hidden_states)

            return logits


class MTPEmbeddingElasticPipe(MTPEmbeddingPipe):
    """Extends ErnieEmbeddingPipe to forward attention_mask through the pipeline."""

    def forward(self, args):
        """MTPEmbeddingElasticPipe forward"""
        hidden_states, layer_elastic, layer_use_info = args
        layer_elastic.stop_gradient = True
        layer_use_info.stop_gradient = True
        is_layer_elastic = layer_elastic.item()
        if not is_layer_elastic:
            use_indices = paddle.nonzero(layer_use_info[0]).flatten()
            hidden_states = paddle.gather(hidden_states, use_indices, axis=0)
        assert (
            self.config.enable_mtp_magic_send
        ), "MTPEmbedding can only be added into model only support enable_mtp_magic_send=True"

        global input_ids_for_mtp
        assert len(input_ids_for_mtp) > 0, "input_ids for mtp is empty"
        input_ids = input_ids_for_mtp.popleft()
        if not is_layer_elastic:
            use_indices = paddle.nonzero(layer_use_info[0]).flatten()
            input_ids = paddle.gather(input_ids, use_indices, axis=0)

        input_embeds = self.embed_tokens(input_ids).astype(
            self.embed_tokens.weight.dtype
        )

        return (
            hidden_states,
            input_embeds,
            layer_elastic.clone(),
            layer_use_info.clone(),
        )


class MTPLayerElastic(MTPLayer):
    """_summary_

    Args:
        MTPLayer (_type_): _description_
    """

    def forward(self, args):
        """forward"""
        if self.config.enable_mtp_magic_send:
            assert isinstance(args, tuple), "Input for MTPLayer must be tuple"
            hidden_states, inputs_embeds, layer_elastic, layer_use_info = args
            layer_elastic.stop_gradient = True
            layer_use_info.stop_gradient = True
            # print(f"elastic info is_layer_elastic {is_layer_elastic}, hidden_states is {hidden_states}")
            inputs_embeds_extra = inputs_embeds[
                :, -self.config.multi_token_pred_depth :, :
            ]  # [B, S, D]
            inputs_embeds = inputs_embeds[:, : -self.config.multi_token_pred_depth, :]
            inputs_embeds_ori = inputs_embeds
        else:
            assert False, "elastic only support enable_mtp_magic_send now"
            # res, layer_elastic, layer_use_info = args
            # tensor_list = paddle.split(res, self.config.multi_token_pred_depth + 1)
            # hidden_states = tensor_list[0]
            # inputs_embeds_cur_depth_list = tensor_list[1:]

        has_gradient = not hidden_states.stop_gradient

        output_list = [hidden_states]
        for depth in range(self.config.multi_token_pred_depth):
            if self.config.enable_mtp_magic_send:
                # 构建输入向量
                inputs_embeds_cur_depth = paddle.concat(
                    [
                        inputs_embeds_ori[:, (depth + 1) :, :],
                        inputs_embeds_extra[:, : (depth + 1), :],
                    ],
                    axis=1,
                )

                if self.config.sequence_parallel or self.config.submatrix_parallel:
                    inputs_embeds_cur_depth = inputs_embeds_cur_depth.reshape(
                        [-1, inputs_embeds_cur_depth.shape[-1]]
                    )
                    inputs_embeds_cur_depth = ScatterOp.apply(inputs_embeds_cur_depth)
            else:
                assert False, "elastic only support enable_mtp_magic_send now"
                # inputs_embeds_cur_depth = inputs_embeds_cur_depth_list[depth]

            # Norm&Concat
            inputs_embeds_cur_depth_norm = self.mtp_emb_norm[depth](
                inputs_embeds_cur_depth
            )
            hidden_states_norm = self.mtp_hidden_norm[depth](hidden_states)

            inputs_embeds_cur_depth = self.mtp_linear_proj[depth](
                paddle.concat(
                    [inputs_embeds_cur_depth_norm, hidden_states_norm], axis=-1
                )
            )

            decoder_layer = self.mtp_block[depth]
            past_key_value = None
            layer_outputs = decoder_layer(
                inputs_embeds_cur_depth,
                None,  # attention_mask
                None,  # position_ids
                None,  # token-type
                False,  # output-attention
                None,  # past-kv-cache
                False,  # use-cache
                None,  # inbatch_pack_offset,
                False,  # output_gate_logits
            )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            output_list.append(hidden_states)

        if self.config.enable_mtp_magic_send:
            return tuple(output_list)
        else:
            res = paddle.concat(output_list)
            return (res,)


class ErnieElasticMoEForCausalLMPipe(ErnieMoEForCausalLMPipe):
    """支持 弹性层 Pipeline Parallel的ERNIE4 组网模型"""

    # for elastic
    ErnieEmbeddingPipeClass = ErnieEmbeddingElasticPipe
    ErnieDecoderLayerPipeClass = ErnieDecoderLayerElasticPipe
    MTPEmbeddingPipeClass = MTPEmbeddingElasticPipe
    MTPLayerClass = MTPLayerElastic
    RMSNormPipeClass = RMSNormElasticPipe
    LayerNormPipeClass = LayerNormElasticPipe
    ErnieMoELMHeadPipeClass = ErnieMoELMHeadElasticPipe

    @classmethod
    def _prepare_pipeline_inputs_func(cls, data):
        """
        从数据字典中提取输入和标签。

        Args:
            data (dict): 数据字典，包含以下键值对：
                - input_ids（paddle.Tensor）：输入的token id。
                - layer_elastic（paddle.Tensor）：是否是弹性层。
                - layer_use_info（paddle.Tensor）：各数据层使用信息。
                - attention_mask（paddle.Tensor）：输入的掩码。
                - position_ids（paddle.Tensor）：输入的位置id。
                - labels（paddle.Tensor）：预测的标签。
                - inbatch_pack_offset（paddle.Tensor）：在batch中的偏移量。

        Returns:
            Tuple[List[paddle.Tensor]]: 返回输入和标签列表。

        """
        inputs = tuple(
            [
                [d[k] for d in data]
                for k in [
                    "input_ids",
                    "layer_elastic",
                    "layer_use_info",
                    "attention_mask",
                    "position_ids",
                    "inbatch_pack_offset",
                ]
                if k in data[0]
            ]
        )
        if len(inputs) == 1:
            inputs = inputs[0]
        labels = [d["labels"] for d in data]
        return inputs, labels
