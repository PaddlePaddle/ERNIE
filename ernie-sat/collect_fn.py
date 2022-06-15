from typing import Collection
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import paddle

from dataset import get_seg_pos
from dataset import phones_masking
from dataset import phones_text_masking
from paddlespeech.t2s.datasets.get_feats import LogMelFBank
from paddlespeech.t2s.modules.nets_utils import make_non_pad_mask
from paddlespeech.t2s.modules.nets_utils import pad_list


class MLMCollateFn:
    """Functor class of common_collate_fn()"""

    def __init__(self,
                 feats_extract,
                 float_pad_value: Union[float, int]=0.0,
                 int_pad_value: int=-32768,
                 not_sequence: Collection[str]=(),
                 mlm_prob: float=0.8,
                 mean_phn_span: int=8,
                 attention_window: int=0,
                 pad_speech: bool=False,
                 seg_emb: bool=False,
                 text_masking: bool=False):
        self.mlm_prob = mlm_prob
        self.mean_phn_span = mean_phn_span
        self.feats_extract = feats_extract
        self.float_pad_value = float_pad_value
        self.int_pad_value = int_pad_value
        self.not_sequence = set(not_sequence)
        self.attention_window = attention_window
        self.pad_speech = pad_speech
        self.seg_emb = seg_emb
        self.text_masking = text_masking

    def __repr__(self):
        return (f"{self.__class__}(float_pad_value={self.float_pad_value}, "
                f"int_pad_value={self.float_pad_value})")

    def __call__(self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
                 ) -> Tuple[List[str], Dict[str, paddle.Tensor]]:
        return mlm_collate_fn(
            data,
            float_pad_value=self.float_pad_value,
            int_pad_value=self.int_pad_value,
            not_sequence=self.not_sequence,
            mlm_prob=self.mlm_prob,
            mean_phn_span=self.mean_phn_span,
            feats_extract=self.feats_extract,
            attention_window=self.attention_window,
            pad_speech=self.pad_speech,
            seg_emb=self.seg_emb,
            text_masking=self.text_masking)


def mlm_collate_fn(
        data: Collection[Tuple[str, Dict[str, np.ndarray]]],
        float_pad_value: Union[float, int]=0.0,
        int_pad_value: int=-32768,
        not_sequence: Collection[str]=(),
        mlm_prob: float=0.8,
        mean_phn_span: int=8,
        feats_extract=None,
        attention_window: int=0,
        pad_speech: bool=False,
        seg_emb: bool=False,
        text_masking: bool=False) -> Tuple[List[str], Dict[str, paddle.Tensor]]:
    uttids = [u for u, _ in data]
    data = [d for _, d in data]

    assert all(set(data[0]) == set(d) for d in data), "dict-keys mismatching"
    assert all(not k.endswith("_lens")
               for k in data[0]), f"*_lens is reserved: {list(data[0])}"

    output = {}
    for key in data[0]:
        # Each models, which accepts these values finally, are responsible
        # to repaint the pad_value to the desired value for each tasks.
        if data[0][key].dtype.kind == "i":
            pad_value = int_pad_value
        else:
            pad_value = float_pad_value

        array_list = [d[key] for d in data]

        # Assume the first axis is length:
        # tensor_list: Batch x (Length, ...)
        tensor_list = [paddle.to_tensor(a) for a in array_list]
        # tensor: (Batch, Length, ...)
        tensor = pad_list(tensor_list, pad_value)
        output[key] = tensor

        # lens: (Batch,)
        if key not in not_sequence:
            lens = paddle.to_tensor(
                [d[key].shape[0] for d in data], dtype=paddle.int64)
            output[key + "_lens"] = lens

    feats = feats_extract.get_log_mel_fbank(np.array(output["speech"][0]))
    feats = paddle.to_tensor(feats)
    feats_lens = paddle.shape(feats)[0]
    feats = paddle.unsqueeze(feats, 0)

    text = output["text"]
    text_lens = output["text_lens"]
    align_start = output["align_start"]
    align_start_lens = output["align_start_lens"]
    align_end = output["align_end"]

    max_tlen = max(text_lens)
    max_slen = max(feats_lens)

    speech_pad = feats[:, :max_slen]

    text_pad = text
    text_mask = make_non_pad_mask(
        text_lens, text_pad, length_dim=1).unsqueeze(-2)
    speech_mask = make_non_pad_mask(
        feats_lens, speech_pad[:, :, 0], length_dim=1).unsqueeze(-2)
    span_bdy = None
    if 'span_bdy' in output.keys():
        span_bdy = output['span_bdy']

    # dual_mask 的是混合中英时候同时 mask 语音和文本 
    # ernie sat 在实现跨语言的时候都 mask 了
    if text_masking:
        masked_pos, text_masked_pos = phones_text_masking(
            xs_pad=speech_pad,
            src_mask=speech_mask,
            text_pad=text_pad,
            text_mask=text_mask,
            align_start=align_start,
            align_end=align_end,
            align_start_lens=align_start_lens,
            mlm_prob=mlm_prob,
            mean_phn_span=mean_phn_span,
            span_bdy=span_bdy)
    # 训练纯中文和纯英文的 -> a3t 没有对 phoneme 做 mask, 只对语音 mask 了
    # a3t 和 ernie sat 的区别主要在于做 mask 的时候
    else:
        masked_pos = phones_masking(
            xs_pad=speech_pad,
            src_mask=speech_mask,
            align_start=align_start,
            align_end=align_end,
            align_start_lens=align_start_lens,
            mlm_prob=mlm_prob,
            mean_phn_span=mean_phn_span,
            span_bdy=span_bdy)
        text_masked_pos = paddle.zeros(paddle.shape(text_pad))

    output_dict = {}

    speech_seg_pos, text_seg_pos = get_seg_pos(
        speech_pad=speech_pad,
        text_pad=text_pad,
        align_start=align_start,
        align_end=align_end,
        align_start_lens=align_start_lens,
        seg_emb=seg_emb)
    output_dict['speech'] = speech_pad
    output_dict['text'] = text_pad
    output_dict['masked_pos'] = masked_pos
    output_dict['text_masked_pos'] = text_masked_pos
    output_dict['speech_mask'] = speech_mask
    output_dict['text_mask'] = text_mask
    output_dict['speech_seg_pos'] = speech_seg_pos
    output_dict['text_seg_pos'] = text_seg_pos
    output = (uttids, output_dict)
    return output


def build_collate_fn(
        sr: int=24000,
        n_fft: int=2048,
        hop_length: int=300,
        win_length: int=None,
        n_mels: int=80,
        fmin: int=80,
        fmax: int=7600,
        mlm_prob: float=0.8,
        mean_phn_span: int=8,
        train: bool=False,
        seg_emb: bool=False,
        epoch: int=-1, ):
    feats_extract_class = LogMelFBank

    feats_extract = feats_extract_class(
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax)

    pad_speech = False
    if epoch == -1:
        mlm_prob_factor = 1
    else:
        mlm_prob_factor = 0.8

    return MLMCollateFn(
        feats_extract=feats_extract,
        float_pad_value=0.0,
        int_pad_value=0,
        mlm_prob=mlm_prob * mlm_prob_factor,
        mean_phn_span=mean_phn_span,
        pad_speech=pad_speech,
        seg_emb=seg_emb)
