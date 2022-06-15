import os
from typing import List
from typing import Optional

import numpy as np
import paddle
import yaml
from sedit_arg_parser import parse_args
from yacs.config import CfgNode

from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.t2s.modules.normalizer import ZScore
from tools.torch_pwgan import TorchPWGAN

model_alias = {
    # acoustic model
    "speedyspeech":
    "paddlespeech.t2s.models.speedyspeech:SpeedySpeech",
    "speedyspeech_inference":
    "paddlespeech.t2s.models.speedyspeech:SpeedySpeechInference",
    "fastspeech2":
    "paddlespeech.t2s.models.fastspeech2:FastSpeech2",
    "fastspeech2_inference":
    "paddlespeech.t2s.models.fastspeech2:FastSpeech2Inference",
    "tacotron2":
    "paddlespeech.t2s.models.tacotron2:Tacotron2",
    "tacotron2_inference":
    "paddlespeech.t2s.models.tacotron2:Tacotron2Inference",
    "pwgan":
    "paddlespeech.t2s.models.parallel_wavegan:PWGGenerator",
    "pwgan_inference":
    "paddlespeech.t2s.models.parallel_wavegan:PWGInference",
}

def is_chinese(ch):
    if u'\u4e00' <= ch <= u'\u9fff':
        return True
    else:
        return False


def build_vocoder_from_file(
        vocoder_config_file=None,
        vocoder_file=None,
        model=None,
        device="cpu", ):
    # Build vocoder
    if str(vocoder_file).endswith(".pkl"):
        # If the extension is ".pkl", the model is trained with parallel_wavegan
        vocoder = TorchPWGAN(vocoder_file, vocoder_config_file)
        return vocoder.to(device)

    else:
        raise ValueError(f"{vocoder_file} is not supported format.")


def get_voc_out(mel):
    # vocoder
    args = parse_args()

    # print("current vocoder: ", args.voc)
    with open(args.voc_config) as f:
        voc_config = CfgNode(yaml.safe_load(f))
    voc_inference = voc_inference = get_voc_inference(
        voc=args.voc,
        voc_config=voc_config,
        voc_ckpt=args.voc_ckpt,
        voc_stat=args.voc_stat)

    with paddle.no_grad():
        wav = voc_inference(mel)
    return np.squeeze(wav)


# dygraph
def get_am_inference(am: str='fastspeech2_csmsc',
                     am_config: CfgNode=None,
                     am_ckpt: Optional[os.PathLike]=None,
                     am_stat: Optional[os.PathLike]=None,
                     phones_dict: Optional[os.PathLike]=None,
                     tones_dict: Optional[os.PathLike]=None,
                     speaker_dict: Optional[os.PathLike]=None,
                     return_am: bool=False):
    with open(phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)

    tone_size = None
    if tones_dict is not None:
        with open(tones_dict, "r") as f:
            tone_id = [line.strip().split() for line in f.readlines()]
        tone_size = len(tone_id)
        print("tone_size:", tone_size)

    spk_num = None
    if speaker_dict is not None:
        with open(speaker_dict, 'rt') as f:
            spk_id = [line.strip().split() for line in f.readlines()]
        spk_num = len(spk_id)
        print("spk_num:", spk_num)

    odim = am_config.n_mels
    # model: {model_name}_{dataset}
    am_name = am[:am.rindex('_')]
    am_dataset = am[am.rindex('_') + 1:]

    am_class = dynamic_import(am_name, model_alias)
    am_inference_class = dynamic_import(am_name + '_inference', model_alias)

    if am_name == 'fastspeech2':
        am = am_class(
            idim=vocab_size, odim=odim, spk_num=spk_num, **am_config["model"])
    elif am_name == 'speedyspeech':
        am = am_class(
            vocab_size=vocab_size,
            tone_size=tone_size,
            spk_num=spk_num,
            **am_config["model"])
    elif am_name == 'tacotron2':
        am = am_class(idim=vocab_size, odim=odim, **am_config["model"])

    am.set_state_dict(paddle.load(am_ckpt)["main_params"])
    am.eval()
    am_mu, am_std = np.load(am_stat)
    am_mu = paddle.to_tensor(am_mu)
    am_std = paddle.to_tensor(am_std)
    am_normalizer = ZScore(am_mu, am_std)
    am_inference = am_inference_class(am_normalizer, am)
    am_inference.eval()
    print("acoustic model done!")
    if return_am:
        return am_inference, am
    else:
        return am_inference


def get_voc_inference(
        voc: str='pwgan_csmsc',
        voc_config: Optional[os.PathLike]=None,
        voc_ckpt: Optional[os.PathLike]=None,
        voc_stat: Optional[os.PathLike]=None, ):
    # model: {model_name}_{dataset}
    voc_name = voc[:voc.rindex('_')]
    voc_class = dynamic_import(voc_name, model_alias)
    voc_inference_class = dynamic_import(voc_name + '_inference', model_alias)
    if voc_name != 'wavernn':
        voc = voc_class(**voc_config["generator_params"])
        voc.set_state_dict(paddle.load(voc_ckpt)["generator_params"])
        voc.remove_weight_norm()
        voc.eval()
    else:
        voc = voc_class(**voc_config["model"])
        voc.set_state_dict(paddle.load(voc_ckpt)["main_params"])
        voc.eval()

    voc_mu, voc_std = np.load(voc_stat)
    voc_mu = paddle.to_tensor(voc_mu)
    voc_std = paddle.to_tensor(voc_std)
    voc_normalizer = ZScore(voc_mu, voc_std)
    voc_inference = voc_inference_class(voc_normalizer, voc)
    voc_inference.eval()
    print("voc done!")
    return voc_inference


def evaluate_durations(phns, target_lang="chinese", fs=24000, hop_length=300):
    args = parse_args()

    if target_lang == 'english':
        args.am = "fastspeech2_ljspeech"
        args.am_config = "download/fastspeech2_nosil_ljspeech_ckpt_0.5/default.yaml"
        args.am_ckpt = "download/fastspeech2_nosil_ljspeech_ckpt_0.5/snapshot_iter_100000.pdz"
        args.am_stat = "download/fastspeech2_nosil_ljspeech_ckpt_0.5/speech_stats.npy"
        args.phones_dict = "download/fastspeech2_nosil_ljspeech_ckpt_0.5/phone_id_map.txt"

    elif target_lang == 'chinese':
        args.am = "fastspeech2_csmsc"
        args.am_config="download/fastspeech2_conformer_baker_ckpt_0.5/conformer.yaml"
        args.am_ckpt = "download/fastspeech2_conformer_baker_ckpt_0.5/snapshot_iter_76000.pdz"
        args.am_stat = "download/fastspeech2_conformer_baker_ckpt_0.5/speech_stats.npy"
        args.phones_dict ="download/fastspeech2_conformer_baker_ckpt_0.5/phone_id_map.txt"

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    # Init body.
    with open(args.am_config) as f:
        am_config = CfgNode(yaml.safe_load(f))

    am_inference, am = get_am_inference(
        am=args.am,
        am_config=am_config,
        am_ckpt=args.am_ckpt,
        am_stat=args.am_stat,
        phones_dict=args.phones_dict,
        tones_dict=args.tones_dict,
        speaker_dict=args.speaker_dict,
        return_am=True)

    vocab_phones = {}
    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    for tone, id in phn_id:
        vocab_phones[tone] = int(id)
    vocab_size = len(vocab_phones)
    phonemes = [phn if phn in vocab_phones else "sp" for phn in phns]

    phone_ids = [vocab_phones[item] for item in phonemes]
    phone_ids_new = phone_ids
    phone_ids_new.append(vocab_size - 1)
    phone_ids_new = paddle.to_tensor(np.array(phone_ids_new, np.int64))
    _, d_outs, _, _ = am.inference(phone_ids_new, spk_id=None, spk_emb=None)
    pre_d_outs = d_outs
    phoneme_durations_new = pre_d_outs * hop_length / fs
    phoneme_durations_new = phoneme_durations_new.tolist()[:-1]
    return phoneme_durations_new
