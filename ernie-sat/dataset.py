


import paddle
import numpy as np
import math


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(paddle.shape(x)[0] for x in xs)
    pad = paddle.full((n_batch, max_len), pad_value, dtype = xs[0].dtype)

    for i in range(n_batch):
        pad[i, : paddle.shape(xs[i])[0]] = xs[i]
    
    return pad

def pad_to_longformer_att_window(text, max_len, max_tlen,attention_window):
    round = max_len % attention_window
    if round != 0:
        max_tlen += (attention_window - round)
        n_batch = paddle.shape(text)[0]
        text_pad = paddle.zeros((n_batch, max_tlen, *paddle.shape(text[0])[1:]), dtype=text.dtype)
        for i in range(n_batch):
            text_pad[i, : paddle.shape(text[i])[0]] = text[i]
    else:
        text_pad = text[:, : max_tlen]
    return text_pad, max_tlen


def make_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = list(lengths)
    # print('lengths', lengths)
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = paddle.shape(xs)[length_dim]

    seq_range = paddle.arange(0, maxlen, dtype=paddle.int64)
    seq_range_expand = paddle.expand(paddle.unsqueeze(seq_range, 0), (bs, maxlen))
    seq_length_expand = paddle.unsqueeze(paddle.to_tensor(lengths), -1)
    # print('seq_length_expand', paddle.shape(seq_length_expand))
    # print('seq_range_expand', paddle.shape(seq_range_expand))
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert paddle.shape(xs)[0] == bs, (paddle.shape(xs)[0], bs)

        if length_dim < 0:
            length_dim = len(paddle.shape(xs)) + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(len(paddle.shape(xs)))
        )
        # print('0:', paddle.shape(mask))
        # print('1:', paddle.shape(mask[ind]))
        # print('2:', paddle.shape(xs))
        mask = paddle.expand(mask[ind], paddle.shape(xs))
    return mask



def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 1, 0],
                 [1, 1, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_non_pad_mask(lengths, xs, 1)
        tensor([[[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)
        >>> make_non_pad_mask(lengths, xs, 2)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

    """
    return ~make_pad_mask(lengths, xs, length_dim)



def phones_masking(xs_pad, src_mask, align_start, align_end, align_start_lengths, mlm_prob, mean_phn_span, span_boundary=None):
    bz, sent_len, _ = paddle.shape(xs_pad)
    mask_num_lower = math.ceil(sent_len * mlm_prob)
    masked_position = np.zeros((bz, sent_len))
    y_masks = None
    # y_masks = torch.ones(bz,sent_len,sent_len,device=xs_pad.device,dtype=xs_pad.dtype)
    # tril_masks = torch.tril(y_masks)
    if mlm_prob == 1.0:
        masked_position += 1
        # y_masks = tril_masks
    elif mean_phn_span == 0:
        # only speech 
        length = sent_len
        mean_phn_span = min(length*mlm_prob//3, 50)
        masked_phn_indices = random_spans_noise_mask(length,mlm_prob, mean_phn_span).nonzero()
        masked_position[:,masked_phn_indices]=1
    else:
        for idx in range(bz):
            if span_boundary is not None:
                for s,e in zip(span_boundary[idx][::2], span_boundary[idx][1::2]):
                    masked_position[idx, s:e] = 1

                    # y_masks[idx, :, s:e] = tril_masks[idx, :, s:e]
                    # y_masks[idx, e:, s:e ] = 0
            else:
                length = align_start_lengths[idx].item()
                if length<2:
                    continue
                masked_phn_indices = random_spans_noise_mask(length,mlm_prob, mean_phn_span).nonzero()
                masked_start = align_start[idx][masked_phn_indices].tolist()
                masked_end = align_end[idx][masked_phn_indices].tolist()
                for s,e in zip(masked_start, masked_end):
                    masked_position[idx, s:e] = 1
                    # y_masks[idx, :, s:e] = tril_masks[idx, :, s:e]
                    # y_masks[idx, e:, s:e ] = 0
    non_eos_mask = np.array(paddle.reshape(src_mask, paddle.shape(xs_pad)[:2]).float().cpu())
    masked_position = masked_position * non_eos_mask
    # y_masks = src_mask & y_masks.bool()

    return paddle.cast(paddle.to_tensor(masked_position), paddle.bool), y_masks




def get_segment_pos(speech_pad, text_pad, align_start, align_end, align_start_lengths,sega_emb):
    bz, speech_len, _ = speech_pad.size()
    _, text_len = text_pad.size()

    # text_segment_pos = paddle.zeros_like(text_pad)
    # speech_segment_pos = paddle.zeros((bz, speech_len),dtype=text_pad.dtype)
    text_segment_pos = np.zeros((bz, text_len)).astype('int64')
    speech_segment_pos = np.zeros((bz, speech_len)).astype('int64')


    if not sega_emb:
        text_segment_pos = paddle.to_tensor(text_segment_pos)
        speech_segment_pos = paddle.to_tensor(speech_segment_pos)
        return speech_segment_pos, text_segment_pos
    for idx in range(bz):
        align_length = align_start_lengths[idx].item()
        for j in range(align_length):
            s,e = align_start[idx][j].item(), align_end[idx][j].item()
            speech_segment_pos[idx][s:e] = j+1
            text_segment_pos[idx][j] = j+1
    
    text_segment_pos = paddle.to_tensor(text_segment_pos)
    speech_segment_pos = paddle.to_tensor(speech_segment_pos)

    return speech_segment_pos, text_segment_pos