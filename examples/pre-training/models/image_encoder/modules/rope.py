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

""" rope """

import logging
from math import pi

import paddle


def broadcat(tensors, dim=-1):
    """_summary_

    Args:
        tensors (_type_): _description_
        dim (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = dim + shape_len if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(
        map(lambda t: t[0].expand(shape=t[1]), zip(tensors, expandable_shapes))
    )
    return paddle.concat(x=tensors, axis=dim)


def rotate_half(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    x = x.reshape(list(x.shape)[:-1] + [-1, 2])
    x1, x2 = x.unbind(axis=-1)
    x = paddle.stack(x=(-x2, x1), axis=-1)
    return x.reshape(list(x.shape)[:-2] + [-1])


class VisionRotaryEmbedding(paddle.nn.Layer):
    """_summary_

    Args:
        paddle (_type_): _description_
    """

    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / theta ** (
                paddle.arange(start=0, end=dim, step=2)[: dim // 2].astype(
                    dtype="float32"
                )
                / dim
            )
        elif freqs_for == "pixel":
            freqs = paddle.linspace(start=1.0, stop=max_freq / 2, num=dim // 2) * pi
        elif freqs_for == "constant":
            freqs = paddle.ones(shape=num_freqs).astype(dtype="float32")
        else:
            raise ValueError(f"unknown modality {freqs_for}")
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = paddle.arange(end=ft_seq_len) / ft_seq_len * pt_seq_len
        freqs_h = paddle.einsum("..., f -> ... f", t, freqs)
        # freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)
        freqs_h = freqs_h.repeat_interleave(2, axis=-1)
        freqs_w = paddle.einsum("..., f -> ... f", t, freqs)
        # freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)
        freqs_w = freqs_w.repeat_interleave(2, axis=-1)
        freqs = broadcat((freqs_h[:, (None), :], freqs_w[(None), :, :]), dim=-1)
        self.register_buffer("freqs_cos", freqs.cos(), persistable=False)
        self.register_buffer("freqs_sin", freqs.sin(), persistable=False)
        logging.info(f"Shape of rope freq: {self.freqs_cos.shape}")

    def forward(self, t, start_index=0):
        """_summary_

        Args:
            t (_type_): _description_
            start_index (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert (
            rot_dim <= t.shape[-1]
        ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        t_left, t, t_right = (
            t[(...), :start_index],
            t[(...), start_index:end_index],
            t[(...), end_index:],
        )
        t = t * self.freqs_cos + rotate_half(t) * self.freqs_sin
        return paddle.concat(x=(t_left, t, t_right), axis=-1)


class VisionRotaryEmbeddingFast(paddle.nn.Layer):
    """_summary_

    Args:
        paddle (_type_): _description_
    """

    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        patch_dropout=0.0,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / theta ** (
                paddle.arange(start=0, end=dim, step=2)[: dim // 2].astype(
                    dtype="float32"
                )
                / dim
            )
        elif freqs_for == "pixel":
            freqs = paddle.linspace(start=1.0, stop=max_freq / 2, num=dim // 2) * pi
        elif freqs_for == "constant":
            freqs = paddle.ones(shape=num_freqs).astype(dtype="float32")
        else:
            raise ValueError(f"unknown modality {freqs_for}")
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = paddle.arange(end=ft_seq_len) / ft_seq_len * pt_seq_len
        freqs = paddle.einsum("..., f -> ... f", t, freqs)
        # freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        freqs = freqs.repeat_interleave(2, axis=freqs.rank() - 1)
        freqs = broadcat((freqs[:, (None), :], freqs[(None), :, :]), dim=-1)
        freqs_cos = freqs.cos().reshape((-1, freqs.shape[-1]))
        freqs_sin = freqs.sin().reshape((-1, freqs.shape[-1]))
        self.patch_dropout = patch_dropout
        self.register_buffer("freqs_cos", freqs_cos, persistable=False)
        self.register_buffer("freqs_sin", freqs_sin, persistable=False)
        logging.info(f"Shape of rope freq: {self.freqs_cos.shape}")

    def forward(self, t, patch_indices_keep=None, position_ids=None):
        """_summary_

        Args:
            t (_type_): _description_
            patch_indices_keep (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if patch_indices_keep is not None:
            batch = t.shape[0]
            batch_indices = paddle.arange(end=batch)
            batch_indices = batch_indices[..., None]
            freqs_cos = self.freqs_cos.unsqueeze(0)
            freqs_cos = freqs_cos.unsqueeze(2)
            freqs_cos = freqs_cos.repeat_interleave(t.shape[0], axis=0)
            freqs_cos = freqs_cos.repeat_interleave(t.shape[1], axis=2)

            freqs_sin = self.freqs_sin.unsqueeze(0)
            freqs_sin = freqs_sin.unsqueeze(2)
            freqs_sin = freqs_sin.repeat_interleave(t.shape[0], axis=0)
            freqs_sin = freqs_sin.repeat_interleave(t.shape[1], axis=2)

            freqs_cos = freqs_cos[batch_indices, patch_indices_keep]
            freqs_cos = freqs_cos.transpose((0, 2, 1, 3))
            freqs_sin = freqs_sin[batch_indices, patch_indices_keep]
            freqs_sin = freqs_sin.transpose((0, 2, 1, 3))
            return t * freqs_cos + rotate_half(t) * freqs_sin
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class VisionRotaryEmbeddingME(paddle.nn.Layer):
    """
    2D Rope
    """

    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        """
        2D Rope
        """
        super().__init__()
        # note!: dim is half of the head dim coz it is 2D rope.
        self._cast_to_low_precision = False  # 兼容develop分支paddle
        self._cast_to_low_precison = False
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / theta ** (
                paddle.arange(start=0, end=dim, step=2)[: dim // 2].astype(
                    dtype="float32"
                )
                / dim
            )
        elif freqs_for == "pixel":
            freqs = paddle.linspace(start=1.0, stop=max_freq / 2, num=dim // 2) * pi
        elif freqs_for == "constant":
            freqs = paddle.ones(shape=num_freqs).astype(dtype="float32")
        else:
            raise ValueError(f"unknown modality {freqs_for}")
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = paddle.arange(end=ft_seq_len, dtype="float32") / ft_seq_len * pt_seq_len
        freqs = paddle.einsum("..., f -> ... f", t, freqs)

        # [t_shape, freqs_shape * 2], [i,:]是[i*theta0, i*theta0, i*theta1, i*theta1, ...]
        freqs_h = freqs.repeat_interleave(2, axis=freqs.rank() - 1)
        freqs_w = freqs.repeat_interleave(2, axis=freqs.rank() - 1)
        freqs_h = (
            freqs_h.unsqueeze(0).unsqueeze(0).transpose([0, 2, 1, 3])
        )  # [1, t_shape, 1, freqs_shape * 2]
        freqs_w = (
            freqs_w.unsqueeze(0).unsqueeze(0).transpose([0, 2, 1, 3])
        )  # [1, t_shape, 1, freqs_shape * 2]

        freqs_h_cos = freqs_h.cos()
        freqs_h_sin = freqs_h.sin()
        freqs_w_cos = freqs_w.cos()
        freqs_w_sin = freqs_w.sin()
        self.register_buffer("freqs_h_cos", freqs_h_cos, persistable=False)
        self.register_buffer("freqs_h_sin", freqs_h_sin, persistable=False)
        self.register_buffer("freqs_w_cos", freqs_w_cos, persistable=False)
        self.register_buffer("freqs_w_sin", freqs_w_sin, persistable=False)

    def forward(
        self, t, patch_indices_keep=None, position_ids=None, position_ids_2d=None
    ):
        """
        forward
        """
        ori_t_shape = t.shape
        if len(t.shape) == 3:
            adaptive = True
            B, n_head, head_dim = ori_t_shape
        else:
            adaptive = False
            B, n_head, N, head_dim = ori_t_shape

        if adaptive:
            out = paddle.zeros_like(t)
            cls, t = t[position_ids == 0], t[position_ids != 0]
            out[position_ids == 0] = cls

        first_half = t[
            ..., : head_dim // 2
        ]  # [B, n_head, N, head_dim//2] or [B, n_head, head_dim//2]
        second_half = t[..., head_dim // 2 :]
        first_half_interleaved = rotate_half(first_half)
        second_half_interleaved = rotate_half(second_half)

        if not adaptive:
            freqs_h_cos = self.freqs_h_cos.tile(repeat_times=[B, 1, n_head, 1])
            freqs_h_sin = self.freqs_h_sin.tile(repeat_times=[B, 1, n_head, 1])
            freqs_w_cos = self.freqs_w_cos.tile(repeat_times=[B, 1, n_head, 1])
            freqs_w_sin = self.freqs_w_sin.tile(repeat_times=[B, 1, n_head, 1])

            batch = t.shape[0]
            batch_indices = paddle.arange(end=batch)
            batch_indices = batch_indices[..., None]
            freqs_h_cos = freqs_h_cos[batch_indices, position_ids_2d[:, :, 0]]
            freqs_h_sin = freqs_h_sin[batch_indices, position_ids_2d[:, :, 0]]
            freqs_w_cos = freqs_w_cos[batch_indices, position_ids_2d[:, :, 1]]
            freqs_w_sin = freqs_w_sin[batch_indices, position_ids_2d[:, :, 1]]

            freqs_h_cos = freqs_h_cos.transpose([0, 2, 1, 3])
            freqs_h_sin = freqs_h_sin.transpose([0, 2, 1, 3])
            freqs_w_cos = freqs_w_cos.transpose([0, 2, 1, 3])
            freqs_w_sin = freqs_w_sin.transpose([0, 2, 1, 3])

        else:
            freqs_h_cos = self.freqs_h_cos.tile(repeat_times=[1, 1, n_head, 1]).squeeze(
                0
            )
            freqs_h_sin = self.freqs_h_sin.tile(repeat_times=[1, 1, n_head, 1]).squeeze(
                0
            )
            freqs_w_cos = self.freqs_w_cos.tile(repeat_times=[1, 1, n_head, 1]).squeeze(
                0
            )
            freqs_w_sin = self.freqs_w_sin.tile(repeat_times=[1, 1, n_head, 1]).squeeze(
                0
            )
            freqs_h_cos = freqs_h_cos[position_ids_2d[:, 0]]
            freqs_h_sin = freqs_h_sin[position_ids_2d[:, 0]]
            freqs_w_cos = freqs_w_cos[position_ids_2d[:, 1]]
            freqs_w_sin = freqs_w_sin[position_ids_2d[:, 1]]

        first_half_result = (
            first_half * freqs_h_cos + first_half_interleaved * freqs_h_sin
        )
        second_half_result = (
            second_half * freqs_w_cos + second_half_interleaved * freqs_w_sin
        )

        ret = paddle.concat([first_half_result, second_half_result], axis=-1)
        if adaptive:
            assert (
                out.shape[0] == position_ids.shape[0]
            ), f"out.shape[0]: {out.shape}, position_ids.shape[0]: {position_ids.shape}"
            out[position_ids != 0] = ret
            ret = out
        assert (
            ret.shape == ori_t_shape
        ), f"ret.shape: {ret.shape}, ori_t_shape: {ori_t_shape}"
        return ret
