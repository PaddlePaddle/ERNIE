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

""" process """
import copy
import io
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
from paddleformers.transformers.image_utils import ChannelDimension
from PIL import Image

from data_processor.image_preprocessor.image_preprocessor_adaptive import AdaptiveImageProcessor
from data_processor.process_video import (
    read_frames_decord,
    read_video_decord,
)
from data_processor.utils.io_utils import RAW_IMAGE_DIR, get_downloadable
from data_processor.utils.render_timestamp import render_frame_timestamp
from ernie.tokenizer_vl import Ernie4_5_VLTokenizer

IDS_TYPE_FLAG = {"text": 0, "image": 1, "video": 2}


def fancy_print(input_ids, tokenizer, image_patch_id=None):
    """
    input_ids: input_ids
    tokenizer: the tokenizer of models
    """
    i = 0
    res = ""
    text_ids = []
    real_image_token_len = 0
    while i < len(input_ids):
        if input_ids[i] == image_patch_id:
            if len(text_ids) > 0:
                res += tokenizer.decode(text_ids)
                text_ids = []

            real_image_token_len += 1
        else:
            if real_image_token_len != 0:
                res += f"<|IMAGE@{real_image_token_len}|>"
                real_image_token_len = 0

            text_ids.append(input_ids[i])

        i += 1
    if len(text_ids) > 0:

        res += tokenizer.decode(text_ids)
        text_ids = []
    return res


class DataProcessor:
    """
    Processes multimodal chat messages into model-ready inputs,
    handling text, images, and videos with 3D positional embeddings.
    """

    CLS_TOKEN = "<|begin_of_sentence|>"
    SEP_TOKEN = "<|end_of_sentence|>"
    IMG_START = "<|IMAGE_START|>"
    IMG_END = "<|IMAGE_END|>"
    VID_START = "<|VIDEO_START|>"
    VID_END = "<|VIDEO_END|>"

    def __init__(
        self,
        tokenizer_name: str,
        image_preprocessor_name: str,
        spatial_conv_size: int = 2,
        temporal_conv_size: int = 2,
        image_min_pixels: int = 4 * 28 * 28,
        image_max_pixels: int = 6177 * 28 * 28,
        video_min_pixels: int = 299 * 28 * 28,
        video_max_pixels: int = 1196 * 28 * 28,
        video_target_frames: int = -1,
        video_frames_sample: str = "leading",
        video_max_frames: int = 180,
        video_min_frames: int = 16,
        video_fps: int = 2,
    ) -> None:
        # Tokenizer and image preprocessor
        self.tokenizer = Ernie4_5_VLTokenizer.from_pretrained(tokenizer_name, verbose=False)
        self.tokenizer.ignored_index = -100
        self.image_preprocessor = AdaptiveImageProcessor.from_pretrained(image_preprocessor_name)

        # Convolution sizes for patch aggregation
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size

        # Pixel constraints
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels

        # Video sampling parameters
        self.target_frames = video_target_frames
        self.frames_sample = video_frames_sample
        self.max_frames = video_max_frames
        self.min_frames = video_min_frames
        self.fps = video_fps

        # Special tokens and IDs
        self.cls_token = self.CLS_TOKEN
        self.sep_token = self.SEP_TOKEN
        self.image_start = self.IMG_START
        self.image_end = self.IMG_END
        self.video_start = self.VID_START
        self.video_end = self.VID_END
        self.image_patch_id = self.tokenizer.convert_tokens_to_ids("<|IMAGE_PLACEHOLDER|>")

        self.token_type_mapping = self._build_token_type_mapping()
        self.is_training = True
        self.role_prefixes = {"system": "", "user": "User: ", "bot": "Assistant: "}

    def _build_token_type_mapping(self) -> Dict[Any, int]:
        mapping = defaultdict(lambda: IDS_TYPE_FLAG["text"])
        for token in (self.IMG_START, self.IMG_END, self.VID_START, self.VID_END):
            mapping[token] = IDS_TYPE_FLAG["image"]
        mapping[self.image_patch_id] = IDS_TYPE_FLAG["image"]
        return mapping

    def train(self) -> None:
        """Enable training mode (produces labels)."""
        self.is_training = True

    def eval(self) -> None:
        """Enable evaluation mode (doesn't produce labels)."""
        self.is_training = False

    def process(self, messages: List[Dict[str, Any]]) -> Dict[str, Union[np.ndarray, List[np.ndarray], None]]:
        """
        Convert chat messages into model inputs.
        Returns a dict with input_ids, token_type_ids, position_ids, images, grid_thw, image_type_ids, labels.
        """
        outputs = {
            "input_ids": [],
            "token_type_ids": [],
            "position_ids": [],
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "labels": [],
            "cur_position": 0,
            "pic_cnt": 0,
            "video_cnt": 0,
        }
        self._add_special_token(self.cls_token, outputs)

        for msg in messages:
            role = msg.get("role")
            assert role in self.role_prefixes, f"Unsupported role: {role}"
            prefix = self.role_prefixes[role]
            if prefix:
                self._add_text(prefix, outputs)

            content_items = msg.get("utterance")
            if not isinstance(content_items, list):
                content_items = [content_items]

            for item in content_items:
                if isinstance(item, str) or item.get("type") == "text":
                    text = item if isinstance(item, str) else item.get("text", "")
                    self._add_text(text, outputs)
                elif item.get("type") == "image_url":
                    self._add_image(item, outputs)
                elif item.get("type") == "video_url":
                    self._add_video(item, outputs)

            if role in ("user", "system"):
                self._add_text("\n", outputs)
            else:
                self._add_special_token(self.sep_token, outputs)

        if not self.is_training:
            # Append assistant prefix in eval
            self._add_text(self.role_prefixes["bot"], outputs)

        return self._pack_outputs(outputs)

    def _add_special_token(self, token: Union[str, int], outputs: Dict) -> None:
        token_id = token if isinstance(token, int) else self.tokenizer.convert_tokens_to_ids(token)
        outputs["input_ids"].append(token_id)
        outputs["token_type_ids"].append(self.token_type_mapping[token])
        pos = outputs["cur_position"]
        outputs["position_ids"].append([pos] * 3)
        outputs["cur_position"] += 1

    def _add_text(self, text: str, outputs: Dict) -> None:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)["input_ids"]
        outputs["input_ids"].extend(tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * len(tokens))

        start = outputs["cur_position"]
        for i in range(len(tokens)):
            outputs["position_ids"].append([start + i] * 3)
        outputs["cur_position"] += len(tokens)

    def _add_image(self, item: Dict, outputs: Dict) -> None:
        url_info = item.get("image_url", {})
        url = url_info.get("url")
        w = url_info.get("image_width", None)
        h = url_info.get("image_height", None)
        data = get_downloadable(url, download_dir=RAW_IMAGE_DIR, save_to_disk=False)

        img = Image.open(io.BytesIO(data) if isinstance(data, bytes) else data)
        if w and h:
            img = img.resize((w, h))

        outputs["pic_cnt"] += 1
        self._add_text(f"Picture {outputs['pic_cnt']}:", outputs)
        self._add_special_token(self.IMG_START, outputs)

        patches_h, patches_w = self.image_preprocessor.get_smarted_resize(
            img.height,
            img.width,
            min_pixels=self.image_min_pixels,
            max_pixels=self.image_max_pixels,
        )[1]
        num_tokens = (patches_h * patches_w) // (self.spatial_conv_size**2)

        outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)

        pos_ids = self._compute_3d_positions(1, patches_h, patches_w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

        # Preprocess pixels
        ret = self.image_preprocessor.preprocess(
            images=[img.convert("RGB")],
            do_normalize=False,
            do_rescale=False,
            predetermined_grid_thw=np.array([[patches_h, patches_w]]),
            do_convert_rgb=True,
            input_data_format=ChannelDimension.LAST,
        )
        outputs["images"].append(ret["pixel_values"])
        outputs["grid_thw"].append(ret["image_grid_thw"])
        outputs["image_type_ids"].append(0)

        self._add_special_token(self.IMG_END, outputs)

    def _add_video(self, item: Dict, outputs: Dict) -> None:
        url_info = item.get("video_url", {})
        url = url_info.get("url")
        outputs["video_cnt"] += 1
        self._add_text(f"Video {outputs['video_cnt']}:", outputs)
        self._add_special_token(self.VID_START, outputs)

        frames = self._load_and_process_video(url, item)
        patches_h, patches_w = self.image_preprocessor.get_smarted_resize(
            frames[0].height,
            frames[0].width,
            min_pixels=self.video_min_pixels,
            max_pixels=self.video_max_pixels,
        )[1]
        num_frames = len(frames)
        num_tokens = (num_frames * patches_h * patches_w) // (self.spatial_conv_size**2 * self.temporal_conv_size)

        pixel_stack = np.stack([np.array(f.convert("RGB")) for f in frames], axis=0)
        ret = self.image_preprocessor.preprocess(
            images=None,
            videos=pixel_stack,
            do_normalize=False,
            do_rescale=False,
            predetermined_grid_thw=np.array([[patches_h, patches_w]] * num_frames),
            do_convert_rgb=True,
            input_data_format=ChannelDimension.LAST,
        )
        outputs["images"].append(ret["pixel_values_videos"])
        outputs["grid_thw"].append(ret["video_grid_thw"])
        outputs["image_type_ids"].extend([1] * num_frames)

        outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)

        pos_ids = self._compute_3d_positions(num_frames, patches_h, patches_w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

        self._add_special_token(self.VID_END, outputs)

    def _load_and_process_video(self, url: str, item: Dict) -> List[Image.Image]:
        reader, meta, path = read_video_decord(url, save_to_disk=False)

        video_frame_args = dict()
        video_frame_args["fps"] = item.get("fps", self.fps)
        video_frame_args["min_frames"] = item.get("min_frames", self.min_frames)
        video_frame_args["max_frames"] = item.get("max_frames", self.max_frames)
        video_frame_args["target_frames"] = item.get("target_frames", self.target_frames)
        video_frame_args["frames_sample"] = item.get("frames_sample", self.frames_sample)

        video_frame_args = self._set_video_frame_args(video_frame_args, meta)

        frames_data, _, timestamps = read_frames_decord(
            path,
            reader,
            meta,
            target_frames=video_frame_args["target_frames"],
            target_fps=video_frame_args["fps"],
            frames_sample=video_frame_args["frames_sample"],
            save_to_disk=False,
        )

        frames: List[Image.Image] = []
        for img_array, ts in zip(frames_data, timestamps):
            frames.append(render_frame_timestamp(img_array, ts))
        # Ensure even number of frames for temporal conv
        if len(frames) % 2 != 0:
            frames.append(copy.deepcopy(frames[-1]))
        return frames

    def _set_video_frame_args(self, video_frame_args, video_meta):
        """
        Set the final frame extraction parameters based on known parameters and priorities
        """
        # Priority: video_target_frames > (video_min_frames, video_max_frames) > video_fps
        if video_frame_args["target_frames"] > 0:
            if video_frame_args["fps"] >= 0:
                raise ValueError("fps must be negative if target_frames is given")
            if (
                video_frame_args["min_frames"] > 0
                and video_frame_args["target_frames"] < video_frame_args["min_frames"]
            ):
                raise ValueError("target_frames must be larger than min_frames")
            if (
                video_frame_args["max_frames"] > 0
                and video_frame_args["target_frames"] > video_frame_args["max_frames"]
            ):
                raise ValueError("target_frames must be smaller than max_frames")
        else:
            if video_frame_args["fps"] < 0:
                raise ValueError("Must provide either positive target_fps or positive target_frames.")
            # First calculate the number of frames extracted under video_fps
            frames_to_extract = int(video_meta["duration"] * video_frame_args["fps"])
            # Determine whether it is within the target range. If not, take target_frames as the upper or lower bound
            if (
                video_frame_args["min_frames"] > 0
                and video_frame_args["max_frames"] > 0
                and video_frame_args["min_frames"] > video_frame_args["max_frames"]
            ):
                raise ValueError("min_frames must be smaller than max_frames")
            if video_frame_args["min_frames"] > 0 and frames_to_extract < video_frame_args["min_frames"]:
                video_frame_args["target_frames"] = video_frame_args["min_frames"]
                video_frame_args["fps"] = -1
            if video_frame_args["max_frames"] > 0 and frames_to_extract > video_frame_args["max_frames"]:
                video_frame_args["target_frames"] = video_frame_args["max_frames"]
                video_frame_args["fps"] = -1

        return video_frame_args

    def _compute_3d_positions(self, t: int, h: int, w: int, start_idx: int) -> List[List[int]]:
        # Downsample time if needed
        t_eff = t // self.temporal_conv_size if t != 1 else 1
        gh, gw = h // self.spatial_conv_size, w // self.spatial_conv_size
        time_idx = np.repeat(np.arange(t_eff), gh * gw)
        h_idx = np.tile(np.repeat(np.arange(gh), gw), t_eff)
        w_idx = np.tile(np.arange(gw), t_eff * gh)

        coords = list(zip(time_idx, h_idx, w_idx))
        return [[start_idx + ti, start_idx + hi, start_idx + wi] for ti, hi, wi in coords]

    def _pack_outputs(self, outs: Dict) -> Dict[str, Any]:
        # Stack or nullify image-related fields
        if not outs["images"]:
            outs["images"] = None
            outs["grid_thw"] = None
            outs["image_type_ids"] = None
        else:
            outs["images"] = np.vstack(outs["images"])
            outs["grid_thw"] = np.vstack(outs["grid_thw"])
            outs["image_type_ids"] = np.array(outs["image_type_ids"])

        # Convert lists to arrays
        outs["input_ids"] = np.array(outs["input_ids"], dtype=np.int64)
        outs["token_type_ids"] = np.array(outs["token_type_ids"], dtype=np.int64)
        outs["position_ids"] = np.array(outs["position_ids"], dtype=np.int64)
        return outs
