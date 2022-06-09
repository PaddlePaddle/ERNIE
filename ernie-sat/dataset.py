import math

import numpy as np
import paddle


def phones_text_masking(xs_pad: paddle.Tensor,
                        src_mask: paddle.Tensor,
                        text_pad: paddle.Tensor,
                        text_mask: paddle.Tensor,
                        align_start: paddle.Tensor,
                        align_end: paddle.Tensor,
                        align_start_lens: paddle.Tensor,
                        mlm_prob: float,
                        mean_phn_span: float,
                        span_bdy: paddle.Tensor=None):
    bz, sent_len, _ = paddle.shape(xs_pad)
    masked_pos = paddle.zeros((bz, sent_len))
    _, text_len = paddle.shape(text_pad)
    text_mask_num_lower = math.ceil(text_len * (1 - mlm_prob) * 0.5)
    text_masked_pos = paddle.zeros((bz, text_len))
    y_masks = None

    if mlm_prob == 1.0:
        masked_pos += 1
        # y_masks = tril_masks
    elif mean_phn_span == 0:
        # only speech 
        length = sent_len
        mean_phn_span = min(length * mlm_prob // 3, 50)
        masked_phn_idxs = random_spans_noise_mask(length, mlm_prob,
                                                  mean_phn_span).nonzero()
        masked_pos[:, masked_phn_idxs] = 1
    else:
        for idx in range(bz):
            if span_bdy is not None:
                for s, e in zip(span_bdy[idx][::2], span_bdy[idx][1::2]):
                    masked_pos[idx, s:e] = 1
            else:
                length = align_start_lens[idx]
                if length < 2:
                    continue
                masked_phn_idxs = random_spans_noise_mask(
                    length, mlm_prob, mean_phn_span).nonzero()
                unmasked_phn_idxs = list(
                    set(range(length)) - set(masked_phn_idxs[0].tolist()))
                np.random.shuffle(unmasked_phn_idxs)
                masked_text_idxs = unmasked_phn_idxs[:text_mask_num_lower]
                text_masked_pos[idx][masked_text_idxs] = 1
                masked_start = align_start[idx][masked_phn_idxs].tolist()
                masked_end = align_end[idx][masked_phn_idxs].tolist()
                for s, e in zip(masked_start, masked_end):
                    masked_pos[idx, s:e] = 1
    non_eos_mask = paddle.reshape(src_mask, paddle.shape(xs_pad)[:2])
    masked_pos = masked_pos * non_eos_mask
    non_eos_text_mask = paddle.reshape(text_mask, paddle.shape(xs_pad)[:2])
    text_masked_pos = text_masked_pos * non_eos_text_mask
    masked_pos = paddle.cast(masked_pos, 'bool')
    text_masked_pos = paddle.cast(text_masked_pos, 'bool')

    return masked_pos, text_masked_pos, y_masks


def get_seg_pos_reduce_duration(
        speech_pad: paddle.Tensor,
        text_pad: paddle.Tensor,
        align_start: paddle.Tensor,
        align_end: paddle.Tensor,
        align_start_lens: paddle.Tensor,
        sega_emb: bool,
        masked_pos: paddle.Tensor,
        feats_lens: paddle.Tensor, ):
    bz, speech_len, _ = paddle.shape(speech_pad)
    text_seg_pos = paddle.zeros(paddle.shape(text_pad))
    speech_seg_pos = paddle.zeros((bz, speech_len), dtype=text_pad.dtype)

    reordered_idx = paddle.zeros((bz, speech_len), dtype=align_start_lens.dtype)

    durations = paddle.ones((bz, speech_len), dtype=align_start_lens.dtype)
    max_reduced_length = 0
    if not sega_emb:
        return speech_pad, masked_pos, speech_seg_pos, text_seg_pos, durations
    for idx in range(bz):
        first_idx = []
        last_idx = []
        align_length = align_start_lens[idx]
        for j in range(align_length):
            s, e = align_start[idx][j], align_end[idx][j]
            if j == 0:
                if paddle.sum(masked_pos[idx][0:s]) == 0:
                    first_idx.extend(range(0, s))
                else:
                    first_idx.extend([0])
                    last_idx.extend(range(1, s))
            if paddle.sum(masked_pos[idx][s:e]) == 0:
                first_idx.extend(range(s, e))
            else:
                first_idx.extend([s])
                last_idx.extend(range(s + 1, e))
                durations[idx][s] = e - s
            speech_seg_pos[idx][s:e] = j + 1
            text_seg_pos[idx][j] = j + 1
        max_reduced_length = max(
            len(first_idx) + feats_lens[idx] - e, max_reduced_length)
        first_idx.extend(range(e, speech_len))
        reordered_idx[idx] = paddle.to_tensor(
            (first_idx + last_idx), dtype=align_start_lens.dtype)
        feats_lens[idx] = len(first_idx)
    reordered_idx = reordered_idx[:, :max_reduced_length]

    return reordered_idx, speech_seg_pos, text_seg_pos, durations, feats_lens


def random_spans_noise_mask(length: int, mlm_prob: float, mean_phn_span: float):
    """This function is copy of `random_spans_helper 
    <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
    Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
        length: an int32 scalar (length of the incoming token sequence)
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number
    Returns:
        a boolean tensor with shape [length]
    """

    orig_length = length

    num_noise_tokens = int(np.round(length * mlm_prob))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / mean_phn_span))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
    def _random_seg(num_items, num_segs):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
            num_items: an integer scalar > 0
            num_segs: an integer scalar in [1, num_items]
        Returns:
            a Tensor with shape [num_segs] containing positive integers that add
            up to num_items
        """
        mask_idxs = np.arange(num_items - 1) < (num_segs - 1)
        np.random.shuffle(mask_idxs)
        first_in_seg = np.pad(mask_idxs, [[1, 0]])
        segment_id = np.cumsum(first_in_seg)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lens = _random_seg(num_noise_tokens, num_noise_spans)
    nonnoise_span_lens = _random_seg(num_nonnoise_tokens, num_noise_spans)

    interleaved_span_lens = np.reshape(
        np.stack([nonnoise_span_lens, noise_span_lens], axis=1),
        [num_noise_spans * 2])
    span_starts = np.cumsum(interleaved_span_lens)[:-1]
    span_start_indicator = np.zeros((length, ), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return is_noise[:orig_length]


def pad_to_longformer_att_window(text: paddle.Tensor,
                                 max_len: int,
                                 max_tlen: int,
                                 attention_window: int=0):

    round = max_len % attention_window
    if round != 0:
        max_tlen += (attention_window - round)
        n_batch = paddle.shape(text)[0]
        text_pad = paddle.zeros(
            (n_batch, max_tlen, *paddle.shape(text[0])[1:]), dtype=text.dtype)
        for i in range(n_batch):
            text_pad[i, :paddle.shape(text[i])[0]] = text[i]
    else:
        text_pad = text[:, :max_tlen]
    return text_pad, max_tlen


def phones_masking(xs_pad: paddle.Tensor,
                   src_mask: paddle.Tensor,
                   align_start: paddle.Tensor,
                   align_end: paddle.Tensor,
                   align_start_lens: paddle.Tensor,
                   mlm_prob: float,
                   mean_phn_span: int,
                   span_bdy: paddle.Tensor=None):
    bz, sent_len, _ = paddle.shape(xs_pad)
    masked_pos = paddle.zeros((bz, sent_len))
    y_masks = None
    if mlm_prob == 1.0:
        masked_pos += 1
    elif mean_phn_span == 0:
        # only speech 
        length = sent_len
        mean_phn_span = min(length * mlm_prob // 3, 50)
        masked_phn_idxs = random_spans_noise_mask(length, mlm_prob,
                                                  mean_phn_span).nonzero()
        masked_pos[:, masked_phn_idxs] = 1
    else:
        for idx in range(bz):
            if span_bdy is not None:
                for s, e in zip(span_bdy[idx][::2], span_bdy[idx][1::2]):
                    masked_pos[idx, s:e] = 1
            else:
                length = align_start_lens[idx]
                if length < 2:
                    continue
                masked_phn_idxs = random_spans_noise_mask(
                    length, mlm_prob, mean_phn_span).nonzero()
                masked_start = align_start[idx][masked_phn_idxs].tolist()
                masked_end = align_end[idx][masked_phn_idxs].tolist()
                for s, e in zip(masked_start, masked_end):
                    masked_pos[idx, s:e] = 1
    non_eos_mask = paddle.reshape(src_mask, paddle.shape(xs_pad)[:2])
    masked_pos = masked_pos * non_eos_mask
    masked_pos = paddle.cast(masked_pos, 'bool')

    return masked_pos, y_masks


def get_seg_pos(speech_pad: paddle.Tensor,
                text_pad: paddle.Tensor,
                align_start: paddle.Tensor,
                align_end: paddle.Tensor,
                align_start_lens: paddle.Tensor,
                sega_emb: bool):
    bz, speech_len, _ = paddle.shape(speech_pad)
    _, text_len = paddle.shape(text_pad)

    text_seg_pos = paddle.zeros((bz, text_len), dtype='int64')
    speech_seg_pos = paddle.zeros((bz, speech_len), dtype='int64')

    if not sega_emb:
        return speech_seg_pos, text_seg_pos
    for idx in range(bz):
        align_length = align_start_lens[idx]
        for j in range(align_length):
            s, e = align_start[idx][j], align_end[idx][j]
            speech_seg_pos[idx, s:e] = j + 1
            text_seg_pos[idx, j] = j + 1

    return speech_seg_pos, text_seg_pos
