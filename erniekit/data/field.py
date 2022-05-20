# -*- coding: utf-8 -*
"""
:py:class:`Reader` is an abstract class representing
"""
from ..common.rule import MaxTruncation, DataShape


class Field(object):
    """Filed"""
    def __init__(self):
        self.name = None
        self.data_type = DataShape.STRING
        self.reader_info = None  # 需要什么reader
        self.tokenizer_info = None
        self.need_convert = True  # 是否需要进行文本转id的操作，比如已经数值化了的文本就不需要再转了
        self.vocab_path = None
        self.max_seq_len = 512
        self.embedding_info = None
        self.truncation_type = MaxTruncation.KEEP_HEAD
        self.padding_id = 0
        self.field_reader = None
        self.label_start_id = 4
        self.label_end_id = 5
        self.join_calculation = True
        self.extra_params = {}
        self.is_prompt_tuning = False
        self.prompt_len = 5
        self.is_mask_res = False
        self.mask_res_len = 1
        self.use_label_map = False
        self.label_map_str = None
        self.num_labels = 2
        self.prompt = None
        self.text_tokenizer = None
        self.text_vocab_path = None
        self.label_map_path = None

    def build(self, params_dict):
        """
        :param params_dict:
        :return:
        """
        self.name = params_dict["name"]
        self.data_type = params_dict["data_type"]
        self.reader_info = params_dict["reader"]

        self.need_convert = params_dict.get("need_convert", True)
        self.vocab_path = params_dict.get("vocab_path", None)
        self.max_seq_len = params_dict.get("max_seq_len", 512)
        self.truncation_type = params_dict.get("truncation_type", MaxTruncation.KEEP_HEAD)
        self.padding_id = params_dict.get("padding_id", 0)
        self.join_calculation = params_dict.get("join_calculation", True)
        # for prompt tuning
        self.is_prompt_tuning = params_dict.get("is_prompt_tuning", False)
        self.prompt_len = params_dict.get("prompt_len", 5)
        self.is_mask_res = params_dict.get("is_mask_res", False)
        self.mask_res_len = params_dict.get("mask_res_len", 1)
        self.label_map_str = params_dict.get("label_map_str", None)
        self.num_labels = params_dict.get("num_labels", 2)
        self.use_label_map = params_dict.get("use_label_map", False)
        self.prompt = params_dict.get("prompt", None)
        # text tokenizer and vocab for label in prompt
        self.text_tokenizer = params_dict.get("text_tokenizer", None)
        self.text_vocab_path = params_dict.get("text_vocab_path", None)
        # for verbalizers for prompt
        self.label_map_path = params_dict.get("label_map_path", None)

        if "num_labels" in params_dict:
            self.num_labels = params_dict["num_labels"]

        # self.label_start_id = params_dict["label_start_id"]
        # self.label_end_id = params_dict["label_end_id"]

        if params_dict.__contains__("embedding"):
            self.embedding_info = params_dict["embedding"]
        if params_dict.__contains__("tokenizer"):
            self.tokenizer_info = params_dict["tokenizer"]
        self.extra_params.update(params_dict.get("extra_params", {}))






