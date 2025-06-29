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
InputIdsMassageProcessor
"""
import json
from copy import deepcopy

from data_processor.steps.input_ids_massaging.chat_template_utils import (
    apply_deepseek_chat_training_template,
    apply_ernie_chat_training_template,
)
from data_processor.steps.input_ids_massaging.example_to_feature import ExampleToFeature
from data_processor.utils.format_utils import check_schema_format, get_format_type
from data_processor.utils.processor_base import ProcessorBase
from data_processor.utils.schema_clean_up import (
    clear_image_info,
    clear_text_info,
    clear_video_info,
    standardization,
)
from ernie.tokenizer_vl import coor_num, special_tokens_info


class InputIdsMassageProcessor(ProcessorBase):
    """
    processor for input_ids_massage
    """

    def __init__(self, args, tokenizer, image_preprocess):
        """
            Args:
            args (obj): Arguments.
            tokenizer (obj): Tokenizer.
            image_preprocess (obj): Image preprocess.

        Returns:
            None.

        Raises:
            None.
        """

        super().__init__(args)

        self.tokenizer = tokenizer
        image_processor = image_preprocess

        # reset max_pixels and min_pixels
        if args.variable_resolution:
            image_processor.set_pixels(
                min_pixels=args.min_pixels, max_pixels=args.max_pixels, msg="InputIdsMassageProcessor __init__()"
            )

        self.example_to_feature = ExampleToFeature(
            self.tokenizer,
            args.data_filelist,
            args.corpus_name,
            im_prefix_length=args.im_prefix_length,
            max_seq_length=args.max_seq_length,
            crop_width=args.crop_width,
            crop_height=args.crop_height,
            crop_tile_option=args.crop_tile_option,
            crop_tile_rate=args.crop_tile_rate,
            use_loc_specialtoken=args.use_loc_specialtoken,
            use_crop_specialtoken=args.use_crop_specialtoken,
            special_tokens_info=special_tokens_info,
            loc_coordinate_num=coor_num,
            augment_conf=args.augment_conf,
            prompt_dir=args.prompt_dir,
            min_crop_flag=args.min_crop_flag,
            one_sample_in_one_seq=args.one_sample_in_one_seq,
            video_max_tile=args.video_max_tile,
            use_multi_image_crop=args.use_multi_image_crop,
            video_fix_crop=args.video_fix_crop,
            min_tile=args.min_tile,
            use_smart_resize=args.use_smart_resize,
            variable_resolution=1,
            spatial_conv_size=args.spatial_conv_size,
            image_processor=image_processor,
            adaptive_max_imgtoken_option=args.adaptive_max_imgtoken_option,
            adaptive_max_imgtoken_rate=args.adaptive_max_imgtoken_rate,
            video_min_pixels=args.video_min_pixels,
            video_max_pixels=args.video_max_pixels,
            rope_3d=1,
            drop_untrainble_sample=args.drop_untrainble_sample,
            chat_template=args.chat_template,
        )
        self.image_start_token = self.example_to_feature.image_start_token
        self.image_end_token = self.example_to_feature.image_end_token
        self.video_start_token = self.example_to_feature.video_start_token
        self.video_end_token = self.example_to_feature.video_end_token
        self.video_temporal_conv_size = self.example_to_feature.video_temporal_conv_size

        self.cls_token = self.example_to_feature.cls_token
        self.sep_token = self.example_to_feature.sep_token
        self.bosys_token = self.example_to_feature.bosys_token
        self.eosys_token = self.example_to_feature.eosys_token
        self.use_pic_id = args.use_pic_id
        self.chat_template = args.chat_template

    def prepare_train_data(self, user_input):
        """prepare train data"""
        # input type is str
        if type(user_input) == str:
            user_input = json.loads(user_input)

        if "text_list" in user_input and "text_info" not in user_input:
            user_input["text_info"] = user_input["text_list"]
        image_info = user_input.get("image_info", [])
        text_info = user_input["text_info"]
        # get text
        image_index_to_text = {}
        for image in image_info:
            if "matched_text_index" not in image:
                image["matched_text_index"] = 0
            image_index_to_text.setdefault(image["matched_text_index"], []).append(image)

        # reformat to interleave format
        is_system = bool(user_input.get("is_system", 0))
        if len(text_info) == 0:
            text_info = [{"text": "", "tag": "mask"}]

        all_item_list = []
        if text_info[0]["tag"] != "mask":
            raise ValueError("first text must be mask")

        all_item_list = [[]]
        # resort text
        for idx, source in enumerate(text_info):
            if len(all_item_list[-1]) == 0 or all_item_list[-1][-1]["tag"] == source["tag"]:
                pass
            else:
                all_item_list.append([])
            if idx in image_index_to_text:
                if len(all_item_list) % 2 == 0:
                    if len(all_item_list[-1]) == 0:
                        all_item_list[-2].append(image_index_to_text[idx])
                    else:
                        raise Exception("Image should not in No-Mask Data.")
                else:
                    all_item_list[-1].append(image_index_to_text[idx])
            all_item_list[-1].append(source)

            if ["no_mask", "mask"][len(all_item_list) % 2] != all_item_list[-1][-1]["tag"]:
                raise Exception("ERROR Mask or No-Mask.")

        if not self.is_training and len(image_index_to_text) > 0:
            # image is in the last item-list
            if max(image_index_to_text.keys()) > idx:
                if text_info[-1]["tag"] == "mask":
                    all_item_list[-1].append(image_index_to_text[max(image_index_to_text.keys())])
                else:
                    all_item_list.append([image_index_to_text[max(image_index_to_text.keys())]])

        # support prefix in traning stage
        if user_input.get("prefix", "") and self.is_training:
            # step1
            # append wo/think tag in last item-list with no-mask
            all_item_list[-1].insert(0, {"text": user_input["prefix"], "tag": "mask"})
            # step2 append labels for multi-turn, we force only learn the last round.
            # the purpose is consistent for training and inference.
            for idx in range(len(all_item_list)):
                for inner_idx, item in enumerate(all_item_list[idx]):
                    if "text" in item:
                        if idx == len(all_item_list) - 1:
                            all_item_list[idx][inner_idx]["label"] = 1
                        else:
                            all_item_list[idx][inner_idx]["label"] = 0

        return {"all_item_list": all_item_list, "is_system": bool(user_input.get("is_system", 0))}

    def mapper(self, data, **kwargs):
        """mapper function"""
        # determined by data_type
        CLEAR_FN = {
            "image": [clear_text_info, clear_image_info],
            "video": [clear_text_info, clear_video_info],
        }

        if not isinstance(data, dict):
            line = json.loads(data)
        else:
            line = data

        sample = {}
        if self.is_training and self.is_pretraining:
            for clear_fn in CLEAR_FN[self.example_to_feature.data_type]:
                info_key, info_val = clear_fn(deepcopy(line), data_type=self.example_to_feature.data_type)
                if info_val is None:
                    return None
                else:
                    sample[info_key] = info_val
            if len(sample) != len(CLEAR_FN[self.example_to_feature.data_type]):
                return None
            # add id and relation
            sample["id"] = line.get("id", "") or line.get("data_id", "")
            sample["relation"] = line.get("relation", {})
            sample["order"] = line.get("order", {})
            sample["modality_interleave"] = line.get("modality_interleave", {})

            # check schema format
            check_schema_format(
                sample=sample,
                data_name=self.args.corpus_name,
                data_type=self.example_to_feature.data_type,
            )
            # standardization data
            sample = standardization(sample, self.example_to_feature.data_type)
        elif not self.is_pretraining:
            data = self.prepare_train_data(data)
            chat_template_func = None
            if self.chat_template == "ernie":
                chat_template_func = apply_ernie_chat_training_template
            elif self.chat_template == "deepseek":
                chat_template_func = apply_deepseek_chat_training_template
            else:
                raise NotImplementedError(f"{self.chat_template} is not supported now.")
            sample = chat_template_func(data, self.tokenizer, self.is_training, use_pic_id=self.use_pic_id, **kwargs)
        else:
            sample = line

        yield from self.example_to_feature.example_to_feature(sample, **kwargs)

    def process(self, data, **kwargs):
        """
        process function
        """
        assert get_format_type(data) == "schema", "input must be in schema format"
        items = []
        for feature in self.mapper(data, **kwargs):
            items.append(feature)
        if len(items) > 0:
            if self.args.serialize_output:
                return json.dumps(items, ensure_ascii=True, separators=(",", ":"))
            else:
                return items
        else:
            return None

    def process_generator(self, data, **kwargs):
        """
        process generator function
        """
        assert get_format_type(data) == "schema", f"input must be in schema format: {data}"
        if self.args.serialize_output:
            for feature in self.mapper(data, **kwargs):
                yield json.dumps(feature, ensure_ascii=True, separators=(",", ":"))
        else:
            return self.mapper(data, **kwargs)
        return self.mapper(data, **kwargs)
