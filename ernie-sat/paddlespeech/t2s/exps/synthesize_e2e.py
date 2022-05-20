# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
from pathlib import Path

import paddle
import soundfile as sf
import yaml
from timer import timer
from yacs.config import CfgNode

from paddlespeech.t2s.exps.syn_utils import am_to_static
from paddlespeech.t2s.exps.syn_utils import get_am_inference
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_sentences
from paddlespeech.t2s.exps.syn_utils import get_voc_inference
from paddlespeech.t2s.exps.syn_utils import voc_to_static


def evaluate(args):

    # Init body.
    with open(args.am_config) as f:
        am_config = CfgNode(yaml.safe_load(f))
    with open(args.voc_config) as f:
        voc_config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(am_config)
    print(voc_config)

    sentences = get_sentences(args)

    # frontend
    frontend = get_frontend(args)

    # acoustic model
    am_inference, am_name, am_dataset = get_am_inference(args, am_config)

    # vocoder
    voc_inference = get_voc_inference(args, voc_config)

    # whether dygraph to static
    if args.inference_dir:
        # acoustic model
        am_inference = am_to_static(args, am_inference, am_name, am_dataset)

        # vocoder
        voc_inference = voc_to_static(args, voc_inference)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merge_sentences = False
    # Avoid not stopping at the end of a sub sentence when tacotron2_ljspeech dygraph to static graph
    # but still not stopping in the end (NOTE by yuantian01 Feb 9 2022)
    if am_name == 'tacotron2':
        merge_sentences = True
    N = 0
    T = 0
    for utt_id, sentence in sentences:
        with timer() as t:
            get_tone_ids = False
            if am_name == 'speedyspeech':
                get_tone_ids = True
            if args.lang == 'zh':
                input_ids = frontend.get_input_ids(
                    sentence,
                    merge_sentences=merge_sentences,
                    get_tone_ids=get_tone_ids)
                phone_ids = input_ids["phone_ids"]
                if get_tone_ids:
                    tone_ids = input_ids["tone_ids"]
            elif args.lang == 'en':
                input_ids = frontend.get_input_ids(
                    sentence, merge_sentences=merge_sentences)
                phone_ids = input_ids["phone_ids"]
            else:
                print("lang should in {'zh', 'en'}!")
            with paddle.no_grad():
                flags = 0
                for i in range(len(phone_ids)):
                    part_phone_ids = phone_ids[i]
                    # acoustic model
                    if am_name == 'fastspeech2':
                        # multi speaker
                        if am_dataset in {"aishell3", "vctk"}:
                            spk_id = paddle.to_tensor(args.spk_id)
                            mel = am_inference(part_phone_ids, spk_id)
                        else:
                            mel = am_inference(part_phone_ids)
                    elif am_name == 'speedyspeech':
                        part_tone_ids = tone_ids[i]
                        if am_dataset in {"aishell3", "vctk"}:
                            spk_id = paddle.to_tensor(args.spk_id)
                            mel = am_inference(part_phone_ids, part_tone_ids,
                                               spk_id)
                        else:
                            mel = am_inference(part_phone_ids, part_tone_ids)
                    elif am_name == 'tacotron2':
                        mel = am_inference(part_phone_ids)
                    # vocoder
                    wav = voc_inference(mel)
                    if flags == 0:
                        wav_all = wav
                        flags = 1
                    else:
                        wav_all = paddle.concat([wav_all, wav])
        wav = wav_all.numpy()
        N += wav.size
        T += t.elapse
        speed = wav.size / t.elapse
        rtf = am_config.fs / speed
        print(
            f"{utt_id}, mel: {mel.shape}, wave: {wav.shape}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )
        sf.write(
            str(output_dir / (utt_id + ".wav")), wav, samplerate=am_config.fs)
        print(f"{utt_id} done!")
    print(f"generation speed: {N / T}Hz, RTF: {am_config.fs / (N / T) }")


def parse_args():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(
        description="Synthesize with acoustic model & vocoder")
    # acoustic model
    parser.add_argument(
        '--am',
        type=str,
        default='fastspeech2_csmsc',
        choices=[
            'speedyspeech_csmsc', 'speedyspeech_aishell3', 'fastspeech2_csmsc',
            'fastspeech2_ljspeech', 'fastspeech2_aishell3', 'fastspeech2_vctk',
            'tacotron2_csmsc', 'tacotron2_ljspeech'
        ],
        help='Choose acoustic model type of tts task.')
    parser.add_argument(
        '--am_config',
        type=str,
        default=None,
        help='Config of acoustic model. Use deault config when it is None.')
    parser.add_argument(
        '--am_ckpt',
        type=str,
        default=None,
        help='Checkpoint file of acoustic model.')
    parser.add_argument(
        "--am_stat",
        type=str,
        default=None,
        help="mean and standard deviation used to normalize spectrogram when training acoustic model."
    )
    parser.add_argument(
        "--phones_dict", type=str, default=None, help="phone vocabulary file.")
    parser.add_argument(
        "--tones_dict", type=str, default=None, help="tone vocabulary file.")
    parser.add_argument(
        "--speaker_dict", type=str, default=None, help="speaker id map file.")
    parser.add_argument(
        '--spk_id',
        type=int,
        default=0,
        help='spk id for multi speaker acoustic model')
    # vocoder
    parser.add_argument(
        '--voc',
        type=str,
        default='pwgan_csmsc',
        choices=[
            'pwgan_csmsc',
            'pwgan_ljspeech',
            'pwgan_aishell3',
            'pwgan_vctk',
            'mb_melgan_csmsc',
            'style_melgan_csmsc',
            'hifigan_csmsc',
            'hifigan_ljspeech',
            'hifigan_aishell3',
            'hifigan_vctk',
            'wavernn_csmsc',
        ],
        help='Choose vocoder type of tts task.')
    parser.add_argument(
        '--voc_config',
        type=str,
        default=None,
        help='Config of voc. Use deault config when it is None.')
    parser.add_argument(
        '--voc_ckpt', type=str, default=None, help='Checkpoint file of voc.')
    parser.add_argument(
        "--voc_stat",
        type=str,
        default=None,
        help="mean and standard deviation used to normalize spectrogram when training voc."
    )
    # other
    parser.add_argument(
        '--lang',
        type=str,
        default='zh',
        help='Choose model language. zh or en')

    parser.add_argument(
        "--inference_dir",
        type=str,
        default=None,
        help="dir to save inference models")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")
    parser.add_argument(
        "--text",
        type=str,
        help="text to synthesize, a 'utt_id sentence' pair per line.")
    parser.add_argument("--output_dir", type=str, help="output dir.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    evaluate(args)


if __name__ == "__main__":
    main()
