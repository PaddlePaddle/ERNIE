import argparse
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union
import humanfriendly
from matplotlib.collections import Collection
from matplotlib.pyplot import axis
import librosa
import soundfile as sf

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from typeguard import check_argument_types
import logging
import math
import yaml
from abc import ABC, abstractmethod
import warnings
from paddle.amp import auto_cast

import sys, os
pypath = '..'
for dir_name in os.listdir(pypath):
    dir_path = os.path.join(pypath, dir_name)
    if os.path.isdir(dir_path):
        sys.path.append(dir_path)

from paddlespeech.t2s.modules.nets_utils import initialize
from paddlespeech.t2s.modules.nets_utils import make_non_pad_mask
from paddlespeech.t2s.modules.nets_utils import make_pad_mask
from paddlespeech.t2s.modules.predictor.duration_predictor import DurationPredictor
from paddlespeech.t2s.modules.predictor.duration_predictor import DurationPredictorLoss
from paddlespeech.t2s.modules.predictor.length_regulator import LengthRegulator
from paddlespeech.t2s.modules.predictor.variance_predictor import VariancePredictor
from paddlespeech.t2s.modules.tacotron2.decoder import Postnet
from paddlespeech.t2s.modules.transformer.encoder import CNNDecoder
from paddlespeech.t2s.modules.transformer.encoder import CNNPostnet
from paddlespeech.t2s.modules.transformer.encoder import ConformerEncoder
from paddlespeech.t2s.modules.transformer.encoder import TransformerEncoder
from paddlespeech.t2s.modules.transformer.embedding import PositionalEncoding, ScaledPositionalEncoding, RelPositionalEncoding
from paddlespeech.t2s.modules.transformer.subsampling import Conv2dSubsampling
from paddlespeech.t2s.modules.masked_fill import masked_fill
from paddlespeech.t2s.modules.transformer.attention import MultiHeadedAttention, RelPositionMultiHeadedAttention
from paddlespeech.t2s.modules.transformer.positionwise_feed_forward import PositionwiseFeedForward
from paddlespeech.t2s.modules.transformer.multi_layer_conv import Conv1dLinear, MultiLayeredConv1d
from paddlespeech.t2s.modules.conformer.convolution import ConvolutionModule
from paddlespeech.t2s.modules.transformer.repeat import repeat
from paddlespeech.t2s.modules.conformer.encoder_layer import EncoderLayer
from paddlespeech.t2s.modules.layer_norm import LayerNorm
from paddlespeech.s2t.utils.error_rate import ErrorCalculator
from paddlespeech.t2s.datasets.get_feats import LogMelFBank

class Swish(nn.Layer):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swich activation function."""
        return x * F.sigmoid(x)


def get_activation(act):
    """Return activation function."""

    activation_funcs = {
        "hardtanh": nn.Hardtanh,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "swish": Swish,
    }

    return activation_funcs[act]()

class LegacyRelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """
    def __init__(self, d_model: int, dropout_rate: float, max_len: int=5000):
        """
        Args:
            d_model (int): Embedding dimension.
            dropout_rate (float): Dropout rate.
            max_len (int, optional): [Maximum input length.]. Defaults to 5000.
        """
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if paddle.shape(self.pe)[1] >= paddle.shape(x)[1]:
                # if self.pe.dtype != x.dtype or self.pe.device != x.device:
                #     self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = paddle.zeros((paddle.shape(x)[1], self.d_model))
        if self.reverse:
            position = paddle.arange(
                paddle.shape(x)[1] - 1, -1, -1.0, dtype=paddle.float32
            ).unsqueeze(1)
        else:
            position = paddle.arange(0, paddle.shape(x)[1], dtype=paddle.float32).unsqueeze(1)
        div_term = paddle.exp(
            paddle.arange(0, self.d_model, 2, dtype=paddle.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Compute positional encoding.
        Args:
            x (paddle.Tensor): Input tensor (batch, time, `*`).
        Returns:
            paddle.Tensor: Encoded tensor (batch, time, `*`).
            paddle.Tensor: Positional embedding tensor (1, time, `*`).
        """
        self.extend_pe(x)  
        x = x * self.xscale
        pos_emb = self.pe[:, :paddle.shape(x)[1]]
        return self.dropout(x), self.dropout(pos_emb)

def dump_tensor(var, do_trans = False):
    wf = open('/mnt/home/xiaoran/PaddleSpeech-develop/tmp_var.out', 'w')
    for num in var.shape:
        wf.write(str(num) + ' ')
    wf.write('\n')
    if do_trans:
        var = paddle.transpose(var, [1,0])
    if len(var.shape)==1:
        for _var in var:
            s = ("%.10f"%_var.item())
            wf.write(s+' ')
    elif len(var.shape)==2:
        for __var in var:
            for _var in __var:
                s = ("%.10f"%_var.item())
                wf.write(s+' ')
            wf.write('\n')
    elif len(var.shape)==3:
        for ___var in var:
            for __var in ___var:
                for _var in __var:
                    s = ("%.10f"%_var.item())
                    wf.write(s+' ')
                wf.write('\n')
            wf.write('\n')
    elif len(var.shape)==4:
        for ____var in var:
            for ___var in ____var:
                for __var in ___var:
                    for _var in __var:
                        s = ("%.10f"%_var.item())
                        wf.write(s+' ')
                    wf.write('\n')
                wf.write('\n')
            wf.write('\n')

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._sub_layers.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class NewMaskInputLayer(nn.Layer):
    __constants__ = ['out_features']
    out_features: int

    def __init__(self, out_features: int,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NewMaskInputLayer, self).__init__()
        self.mask_feature = paddle.create_parameter(
            shape=(1,1,out_features), 
            dtype=paddle.float32, 
            default_initializer=paddle.nn.initializer.Assign(paddle.normal(shape=(1,1,out_features))))

    def forward(self, input: paddle.Tensor, masked_position=None) -> paddle.Tensor:
        masked_position = paddle.expand_as(paddle.unsqueeze(masked_position, -1), input)
        masked_input = masked_fill(input, masked_position, 0) + masked_fill(paddle.expand_as(self.mask_feature, input), ~masked_position, 0)
        return masked_input

class LegacyRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (old version).
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias_attr=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3

        self.pos_bias_u = paddle.create_parameter(
            shape=(self.h, self.d_k),
            dtype='float32',
            default_initializer=paddle.nn.initializer.XavierUniform())
        self.pos_bias_v = paddle.create_parameter(
            shape=(self.h, self.d_k),
            dtype='float32',
            default_initializer=paddle.nn.initializer.XavierUniform())

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x(Tensor): Input tensor (batch, head, time1, time2).

        Returns:
            Tensor:Output tensor.
        """
        b, h, t1, t2 = paddle.shape(x)
        zero_pad = paddle.zeros((b, h, t1, 1))
        x_padded = paddle.concat([zero_pad, x], axis=-1)
        x_padded = paddle.reshape(x_padded, [b, h, t2 + 1, t1])
        # only keep the positions from 0 to time2
        x = paddle.reshape(x_padded[:, :, 1:], [b, h, t1, t2])

        if self.zero_triu:
            ones = paddle.ones((t1, t2))
            x = x * paddle.tril(ones, t2 - 1)[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query(Tensor): Query tensor (#batch, time1, size).
            key(Tensor): Key tensor (#batch, time2, size).
            value(Tensor): Value tensor (#batch, time2, size).
            pos_emb(Tensor): Positional embedding tensor (#batch, time1, size).
            mask(Tensor): Mask tensor (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        # (batch, time1, head, d_k)
        q = paddle.transpose(q, [0, 2, 1, 3])

        n_batch_pos = paddle.shape(pos_emb)[0]
        p = paddle.reshape(self.linear_pos(pos_emb), [n_batch_pos, -1, self.h, self.d_k])
        # (batch, head, time1, d_k)
        p = paddle.transpose(p, [0, 2, 1, 3])
        # (batch, head, time1, d_k)
        q_with_bias_u = paddle.transpose((q + self.pos_bias_u), [0, 2, 1, 3])
        # (batch, head, time1, d_k)
        q_with_bias_v = paddle.transpose((q + self.pos_bias_v), [0, 2, 1, 3])

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = paddle.matmul(q_with_bias_u, paddle.transpose(k, [0, 1, 3, 2]))

        # compute matrix b and matrix d
        # (batch, head, time1, time1)
        matrix_bd = paddle.matmul(q_with_bias_v, paddle.transpose(p, [0, 1, 3, 2]))
        matrix_bd = self.rel_shift(matrix_bd)
        # (batch, head, time1, time2)
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask)

class MLMEncoder(nn.Layer):
    """Conformer encoder module.

    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, paddle.nn.Layer]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        pos_enc_layer_type (str): Encoder positional encoding layer type.
        selfattention_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        use_cnn_module (bool): Whether to use convolution module.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.
        stochastic_depth_rate (float): Maximum probability to skip the encoder layer.
        intermediate_layers (Union[List[int], None]): indices of intermediate CTC layer.
            indices start from 1.
            if not None, intermediate outputs are returned (which changes return type
            signature.)

    """
    def __init__(
        self,
        idim,
        vocab_size=0,
        pre_speech_layer: int = 0,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        macaron_style=False,
        pos_enc_layer_type="abs_pos",
        pos_enc_class=None,
        selfattention_layer_type="selfattn",
        activation_type="swish",
        use_cnn_module=False,
        zero_triu=False,
        cnn_module_kernel=31,
        padding_idx=-1,
        stochastic_depth_rate=0.0,
        intermediate_layers=None,
        text_masking = False
    ):
        """Construct an Encoder object."""
        super(MLMEncoder, self).__init__()
        self._output_size = attention_dim
        self.text_masking=text_masking
        if self.text_masking:
            self.text_masking_layer = NewMaskInputLayer(attention_dim)
        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            pos_enc_class = LegacyRelPositionalEncoding
            assert selfattention_layer_type == "legacy_rel_selfattn"
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        self.conv_subsampling_factor = 1
        if input_layer == "linear":
            self.embed = nn.Sequential(
                nn.Linear(idim, attention_dim),
                nn.LayerNorm(attention_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
            self.conv_subsampling_factor = 4
        elif input_layer == "embed":
            self.embed = nn.Sequential(
                nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "mlm":
            self.segment_emb = None
            self.speech_embed = mySequential(
                NewMaskInputLayer(idim),
                nn.Linear(idim, attention_dim),
                nn.LayerNorm(attention_dim),
                nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
            self.text_embed = nn.Sequential(
                nn.Embedding(vocab_size, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer=="sega_mlm":
            self.segment_emb = nn.Embedding(500, attention_dim, padding_idx=padding_idx)
            self.speech_embed = mySequential(
                NewMaskInputLayer(idim),
                nn.Linear(idim, attention_dim),
                nn.LayerNorm(attention_dim),
                nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
            self.text_embed = nn.Sequential(
                nn.Embedding(vocab_size, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, nn.Layer):
            self.embed = nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before

        # self-attention module definition
        if selfattention_layer_type == "selfattn":
            logging.info("encoder self-attention layer type = self-attention")
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "rel_selfattn":
            logging.info("encoder self-attention layer type = relative self-attention")
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        # feed-forward module definition
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate * float(1 + lnum) / num_blocks,
            ),
        )
        self.pre_speech_layer = pre_speech_layer
        self.pre_speech_encoders = repeat(
            self.pre_speech_layer,
            lambda lnum: EncoderLayer(
                attention_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate * float(1 + lnum) / self.pre_speech_layer,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

        self.intermediate_layers = intermediate_layers


    def forward(self, speech_pad, text_pad, masked_position, speech_mask=None, text_mask=None,speech_segment_pos=None, text_segment_pos=None):
        """Encode input sequence.

        """
        if masked_position is not None:
            speech_pad = self.speech_embed(speech_pad, masked_position)
        else:
            speech_pad = self.speech_embed(speech_pad)
        # pure speech input
        if -2 in np.array(text_pad):
            text_pad = text_pad+3
            text_mask = paddle.unsqueeze(bool(text_pad), 1)
            text_segment_pos = paddle.zeros_like(text_pad)
            text_pad = self.text_embed(text_pad)
            text_pad = (text_pad[0] + self.segment_emb(text_segment_pos), text_pad[1])
            text_segment_pos=None
        elif text_pad is not None:
            text_pad = self.text_embed(text_pad)
        segment_emb = None
        if speech_segment_pos is not None and text_segment_pos is not None and self.segment_emb:
            speech_segment_emb = self.segment_emb(speech_segment_pos)
            text_segment_emb = self.segment_emb(text_segment_pos)
            text_pad = (text_pad[0] + text_segment_emb, text_pad[1])
            speech_pad = (speech_pad[0] + speech_segment_emb, speech_pad[1])
            segment_emb = paddle.concat([speech_segment_emb, text_segment_emb],axis=1)
        if self.pre_speech_encoders:
            speech_pad, _ = self.pre_speech_encoders(speech_pad, speech_mask)

        if text_pad is not None:
            xs = paddle.concat([speech_pad[0], text_pad[0]], axis=1)
            xs_pos_emb = paddle.concat([speech_pad[1], text_pad[1]], axis=1)
            masks = paddle.concat([speech_mask,text_mask],axis=-1)
        else:
            xs = speech_pad[0]
            xs_pos_emb = speech_pad[1]
            masks = speech_mask

        xs, masks = self.encoders((xs,xs_pos_emb), masks)

        if isinstance(xs, tuple):
            xs = xs[0]
        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks #, segment_emb


class MLMDecoder(MLMEncoder):

    def forward(self, xs, masks, masked_position=None,segment_emb=None):
        """Encode input sequence.

        Args:
            xs (paddle.Tensor): Input tensor (#batch, time, idim).
            masks (paddle.Tensor): Mask tensor (#batch, time).

        Returns:
            paddle.Tensor: Output tensor (#batch, time, attention_dim).
            paddle.Tensor: Mask tensor (#batch, time).

        """
        emb, mlm_position = None, None
        if not self.training:
            masked_position = None
        # if isinstance(self.embed, (Conv2dSubsampling, VGG2L)):
        #     xs, masks = self.embed(xs, masks)
        # else:
        xs = self.embed(xs)
        if segment_emb:
            xs = (xs[0] + segment_emb, xs[1])
        if self.intermediate_layers is None:
            xs, masks = self.encoders(xs, masks)
        else:
            intermediate_outputs = []
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs, masks = encoder_layer(xs, masks)

                if (
                    self.intermediate_layers is not None
                    and layer_idx + 1 in self.intermediate_layers
                ):
                    encoder_output = xs
                    # intermediate branches also require normalization.
                    if self.normalize_before:
                        encoder_output = self.after_norm(encoder_output)
                    intermediate_outputs.append(encoder_output)
        if isinstance(xs, tuple):
            xs = xs[0]
        if self.normalize_before:
            xs = self.after_norm(xs)

        if self.intermediate_layers is not None:
            return xs, masks, intermediate_outputs
        return xs, masks

class AbsESPnetModel(nn.Layer, ABC):
    """The common abstract class among each tasks

    "ESPnetModel" is referred to a class which inherits paddle.nn.Layer,
    and makes the dnn-models forward as its member field,
    a.k.a delegate pattern,
    and defines "loss", "stats", and "weight" for the task.

    If you intend to implement new task in ESPNet,
    the model must inherit this class.
    In other words, the "mediator" objects between
    our training system and the your task class are
    just only these three values, loss, stats, and weight.

    Example:
        >>> from espnet2.tasks.abs_task import AbsTask
        >>> class YourESPnetModel(AbsESPnetModel):
        ...     def forward(self, input, input_lengths):
        ...         ...
        ...         return loss, stats, weight
        >>> class YourTask(AbsTask):
        ...     @classmethod
        ...     def build_model(cls, args: argparse.Namespace) -> YourESPnetModel:
    """

    @abstractmethod
    def forward(
        self, **batch: paddle.Tensor
    ) -> Tuple[paddle.Tensor, Dict[str, paddle.Tensor], paddle.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def collect_feats(self, **batch: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        raise NotImplementedError

class AbsFeatsExtract(nn.Layer, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, input: paddle.Tensor, input_lengths: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        raise NotImplementedError

class AbsNormalize(nn.Layer, ABC):
    @abstractmethod
    def forward(
        self, input: paddle.Tensor, input_lengths: paddle.Tensor = None
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        # return output, output_lengths
        raise NotImplementedError



def pad_to_longformer_att_window(text, max_len, max_tlen,attention_window):
    round = max_len % attention_window
    if round != 0:
        max_tlen += (attention_window - round)
        n_batch = paddle.shape(text)[0]
        text_pad = paddle.zeros(shape = (n_batch, max_tlen, *paddle.shape(text[0])[1:]), dtype=text.dtype)
        for i in range(n_batch):
            text_pad[i, : paddle.shape(text[i])[0]] = text[i]
    else:
        text_pad = text[:, : max_tlen]
    return text_pad, max_tlen

class ESPnetMLMModel(AbsESPnetModel):
    def __init__(
        self,
        token_list: Union[Tuple[str, ...], List[str]],
        odim: int,
        feats_extract: Optional[AbsFeatsExtract],
        normalize: Optional[AbsNormalize],
        encoder: nn.Layer,
        decoder: Optional[nn.Layer],
        postnet_layers: int = 0,
        postnet_chans: int = 0,
        postnet_filts: int = 0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        masking_schema: str = "span",
        mean_phn_span: int = 3,
        mlm_prob: float = 0.25,
        dynamic_mlm_prob = False,
        decoder_seg_pos=False,
        text_masking=False
    ):

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.odim = odim
        self.ignore_id = ignore_id
        self.token_list = token_list.copy()

        self.normalize = normalize
        self.encoder = encoder

        self.decoder = decoder
        self.vocab_size = encoder.text_embed[0]._num_embeddings
        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None

        self.feats_extract = feats_extract
        self.mlm_weight = 1.0
        self.mlm_prob = mlm_prob
        self.mlm_layer = 12
        self.finetune_wo_mlm =True
        self.max_span = 50
        self.min_span = 4
        self.mean_phn_span = mean_phn_span
        self.masking_schema = masking_schema
        if self.decoder is None or not (hasattr(self.decoder, 'output_layer') and self.decoder.output_layer is not None):
            self.sfc = nn.Linear(self.encoder._output_size, odim)
        else:
            self.sfc=None
        if text_masking:
            self.text_sfc =  nn.Linear(self.encoder.text_embed[0]._embedding_dim, self.vocab_size, weight_attr = self.encoder.text_embed[0]._weight_attr)
            self.text_mlm_loss = nn.CrossEntropyLoss(ignore_index=ignore_id)
        else:
            self.text_sfc = None
            self.text_mlm_loss = None
        self.decoder_seg_pos = decoder_seg_pos
        if lsm_weight > 50:
            self.l1_loss_func = nn.MSELoss(reduce=False)
        else:
            self.l1_loss_func = nn.L1Loss(reduction='none')
        self.postnet = (
            None
            if postnet_layers == 0
            else Postnet(
                idim=self.encoder._output_size,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=True,
                dropout_rate=0.5,
            )
        )

    def collect_feats(self,
        speech, speech_lengths, text, text_lengths, masked_position, speech_mask, text_mask, speech_segment_pos, text_segment_pos, y_masks=None
    ) -> Dict[str, paddle.Tensor]:
        return {"feats": speech, "feats_lengths": speech_lengths}

    def _forward(self, batch, speech_segment_pos,y_masks=None):
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        speech_pad_placeholder = batch['speech_pad']
        if self.decoder is not None:
            ys_in = self._add_first_frame_and_remove_last_frame(batch['speech_pad'])
        encoder_out, h_masks = self.encoder(**batch)
        if self.decoder is not None:
            zs, _ = self.decoder(ys_in, y_masks, encoder_out, bool(h_masks), self.encoder.segment_emb(speech_segment_pos))
            speech_hidden_states = zs
        else:
            speech_hidden_states = encoder_out[:,:paddle.shape(batch['speech_pad'])[1], :]
        if self.sfc is not None:
            before_outs = paddle.reshape(self.sfc(speech_hidden_states), (paddle.shape(speech_hidden_states)[0], -1, self.odim))
        else:
            before_outs = speech_hidden_states
        if self.postnet is not None:
            after_outs = before_outs + paddle.transpose(self.postnet(
                paddle.transpose(before_outs, [0, 2, 1])
            ), (0, 2, 1))
        else:
            after_outs = None
        return before_outs, after_outs, speech_pad_placeholder, batch['masked_position']

    

    
    def inference(
        self,
        speech, text, masked_position, speech_mask, text_mask, speech_segment_pos, text_segment_pos,
        span_boundary,
        y_masks=None,
        speech_lengths=None, text_lengths=None,
        feats: Optional[paddle.Tensor] = None,
        spembs: Optional[paddle.Tensor] = None,
        sids: Optional[paddle.Tensor] = None,
        lids: Optional[paddle.Tensor] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_teacher_forcing: bool = False,
    ) -> Dict[str, paddle.Tensor]:
        
        
        batch = dict(
            speech_pad=speech,
            text_pad=text,
            masked_position=masked_position,
            speech_mask=speech_mask,
            text_mask=text_mask,
            speech_segment_pos=speech_segment_pos,
            text_segment_pos=text_segment_pos,
        )
        

        # # inference with teacher forcing
        # hs, h_masks = self.encoder(**batch)

        outs = [batch['speech_pad'][:,:span_boundary[0]]]
        z_cache = None
        if use_teacher_forcing:
            before,zs, _, _ = self._forward(
                batch, speech_segment_pos, y_masks=y_masks)
            if zs is None:
                zs = before
            outs+=[zs[0][span_boundary[0]:span_boundary[1]]]
            outs+=[batch['speech_pad'][:,span_boundary[1]:]]
            return dict(feat_gen=outs)
        
            # concatenate attention weights -> (#layers, #heads, T_feats, T_text)
        att_ws = paddle.stack(att_ws, axis=0)
        outs += [batch['speech_pad'][:,span_boundary[1]:]]
        return dict(feat_gen=outs, att_w=att_ws)


    def _add_first_frame_and_remove_last_frame(self, ys: paddle.Tensor) -> paddle.Tensor:
        ys_in = paddle.concat(
            [paddle.zeros(shape = (paddle.shape(ys)[0], 1, paddle.shape(ys)[2]), dtype = ys.dtype), ys[:, :-1]], axis=1
        )
        return ys_in


class ESPnetMLMEncAsDecoderModel(ESPnetMLMModel):

    def _forward(self, batch, speech_segment_pos, y_masks=None):
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        speech_pad_placeholder = batch['speech_pad']
        encoder_out, h_masks = self.encoder(**batch) # segment_emb
        if self.decoder is not None:
            zs, _ = self.decoder(encoder_out, h_masks)
        else:
            zs = encoder_out
        speech_hidden_states = zs[:,:paddle.shape(batch['speech_pad'])[1], :]
        if self.sfc is not None:
            before_outs = paddle.reshape(self.sfc(speech_hidden_states), (paddle.shape(speech_hidden_states)[0], -1, self.odim))
        else:
            before_outs = speech_hidden_states
        if self.postnet is not None:
            after_outs = before_outs + paddle.transpose(self.postnet(
                paddle.transpose(before_outs, [0, 2, 1])
            ), [0, 2, 1])
        else:
            after_outs = None
        return before_outs, after_outs, speech_pad_placeholder, batch['masked_position']

class ESPnetMLMDualMaksingModel(ESPnetMLMModel):

    def _calc_mlm_loss(
        self,
        before_outs: paddle.Tensor,
        after_outs: paddle.Tensor,
        text_outs: paddle.Tensor,
        batch
    ):
        xs_pad = batch['speech_pad']
        text_pad = batch['text_pad']
        masked_position = batch['masked_position']
        text_masked_position = batch['text_masked_position']
        mlm_loss_position = masked_position>0
        loss = paddle.sum(self.l1_loss_func(paddle.reshape(before_outs, (-1, self.odim)), 
                                            paddle.reshape(xs_pad, (-1, self.odim))), axis=-1)
        if after_outs is not None:
            loss += paddle.sum(self.l1_loss_func(paddle.reshape(after_outs, (-1, self.odim)), 
                                                paddle.reshape(xs_pad, (-1, self.odim))), axis=-1)
        loss_mlm = paddle.sum((loss * paddle.reshape(mlm_loss_position, axis=-1).float())) \
                                            / paddle.sum((mlm_loss_position.float()) + 1e-10)

        loss_text = paddle.sum((self.text_mlm_loss(paddle.reshape(text_outs, (-1,self.vocab_size)), paddle.reshape(text_pad, (-1))) * paddle.reshape(text_masked_position, (-1)).float())) \
            /  paddle.sum((text_masked_position.float()) + 1e-10)
        return loss_mlm, loss_text


    def _forward(self, batch, speech_segment_pos, y_masks=None):
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        speech_pad_placeholder = batch['speech_pad']
        encoder_out, h_masks = self.encoder(**batch) # segment_emb
        if self.decoder is not None:
            zs, _ = self.decoder(encoder_out, h_masks)
        else:
            zs = encoder_out
        speech_hidden_states = zs[:,:paddle.shape(batch['speech_pad'])[1], :]
        if self.text_sfc:
            text_hiddent_states = zs[:,paddle.shape(batch['speech_pad'])[1]:,:]
            text_outs = paddle.reshape(self.text_sfc(text_hiddent_states), (paddle.shape(text_hiddent_states)[0], -1, self.vocab_size))
        if self.sfc is not None:
            before_outs = paddle.reshape(self.sfc(speech_hidden_states),
            (paddle.shape(speech_hidden_states)[0], -1, self.odim))
        else:
            before_outs = speech_hidden_states
        if self.postnet is not None:
            after_outs = before_outs + paddle.transpose(self.postnet(
                paddle.transpose(before_outs, [0,2,1])
            ), [0, 2, 1])
        else:
            after_outs = None
        return before_outs, after_outs,text_outs, None #, speech_pad_placeholder, batch['masked_position'],batch['text_masked_position']

def build_model_from_file(config_file, model_file):
    
    state_dict = paddle.load(model_file)
    model_class = ESPnetMLMDualMaksingModel if 'conformer_combine_vctk_aishell3_dual_masking' in config_file \
        else ESPnetMLMEncAsDecoderModel

    # 构建模型
    args = yaml.safe_load(Path(config_file).open("r", encoding="utf-8"))
    args = argparse.Namespace(**args)

    model = build_model(args, model_class)

    model.set_state_dict(state_dict)
    return model, args


def build_model(args: argparse.Namespace, model_class = ESPnetMLMEncAsDecoderModel) -> ESPnetMLMModel:
    if isinstance(args.token_list, str):
        with open(args.token_list, encoding="utf-8") as f:
            token_list = [line.rstrip() for line in f]

        # Overwriting token_list to keep it as "portable".
        args.token_list = list(token_list)
    elif isinstance(args.token_list, (tuple, list)):
        token_list = list(args.token_list)
    else:
        raise RuntimeError("token_list must be str or list")
    vocab_size = len(token_list)
    logging.info(f"Vocabulary size: {vocab_size }")
    
    odim = 80


    # Normalization layer
    normalize = None

    pos_enc_class = ScaledPositionalEncoding if args.use_scaled_pos_enc else PositionalEncoding

    if "conformer" == args.encoder:
        conformer_self_attn_layer_type = args.encoder_conf['selfattention_layer_type']
        conformer_pos_enc_layer_type = args.encoder_conf['pos_enc_layer_type']
        conformer_rel_pos_type = "legacy"
        if conformer_rel_pos_type == "legacy":
            if conformer_pos_enc_layer_type == "rel_pos":
                conformer_pos_enc_layer_type = "legacy_rel_pos"
                logging.warning(
                    "Fallback to conformer_pos_enc_layer_type = 'legacy_rel_pos' "
                    "due to the compatibility. If you want to use the new one, "
                    "please use conformer_pos_enc_layer_type = 'latest'."
                )
            if conformer_self_attn_layer_type == "rel_selfattn":
                conformer_self_attn_layer_type = "legacy_rel_selfattn"
                logging.warning(
                    "Fallback to "
                    "conformer_self_attn_layer_type = 'legacy_rel_selfattn' "
                    "due to the compatibility. If you want to use the new one, "
                    "please use conformer_pos_enc_layer_type = 'latest'."
                )
        elif conformer_rel_pos_type == "latest":
            assert conformer_pos_enc_layer_type != "legacy_rel_pos"
            assert conformer_self_attn_layer_type != "legacy_rel_selfattn"
        else:
            raise ValueError(f"Unknown rel_pos_type: {conformer_rel_pos_type}")
        args.encoder_conf['selfattention_layer_type'] = conformer_self_attn_layer_type
        args.encoder_conf['pos_enc_layer_type'] = conformer_pos_enc_layer_type 
        if "conformer"==args.decoder:
            args.decoder_conf['selfattention_layer_type'] = conformer_self_attn_layer_type
            args.decoder_conf['pos_enc_layer_type'] = conformer_pos_enc_layer_type 


    # Encoder
    encoder_class = MLMEncoder

    if 'text_masking' in args.model_conf.keys() and args.model_conf['text_masking']:
        args.encoder_conf['text_masking'] = True
    else:
        args.encoder_conf['text_masking'] = False
    
    encoder = encoder_class(args.input_size,vocab_size=vocab_size, pos_enc_class=pos_enc_class,
    **args.encoder_conf)

    # Decoder
    if args.decoder != 'no_decoder':
        decoder_class = MLMDecoder
        decoder = decoder_class(
            idim=0,
            input_layer=None,
            **args.decoder_conf,
        )
    else:
        decoder = None

    # Build model
    model = model_class(
        feats_extract=None, # maybe should be LogMelFbank
        odim=odim,
        normalize=normalize,
        encoder=encoder,
        decoder=decoder,
        token_list=token_list,
        **args.model_conf,
    )


    # Initialize
    if args.init is not None:
        initialize(model, args.init)

    return model
