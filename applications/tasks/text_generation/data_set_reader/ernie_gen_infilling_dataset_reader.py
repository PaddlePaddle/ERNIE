# -*- coding: utf-8 -*
"""
:py:class:`ErnieGenerationReader`
"""
import csv
import os
import sys
import traceback
import logging
from collections import namedtuple
import numpy as np
import time
import six
import paddle.fluid as fluid
from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
from erniekit.data.util_helper import pad_batch_data, convert_to_unicode

from erniekit.data.data_set_reader.base_dataset_reader import BaseDataSetReader
import paddle.distributed as dist
import random
from erniekit.data.data_set_reader.base_dataset_reader_ernie_gen import BaseDataSetReaderErnieGen
from paddle.fluid.core_avx import PaddleTensor, PaddleDType

@RegisterSet.data_set_reader.register
# class InfillingGenReader(BaseDataSetReader):
class InfillingGenReader(BaseDataSetReaderErnieGen):
    """ErnieGenerationReader:一个基础的data_set_reader，实现了文件读取，id序列化，token embedding化等基本操作
    """
    def __init__(self, name, fields, config):
        """__init__
        """
        BaseDataSetReaderErnieGen.__init__(self, name, fields, config)
        self.do_lower_case = self.config.extra_params.get("do_lower_case", True)
        self.vocab_path = self.config.extra_params.get("vocab_path")
        self.spm_model_path = self.config.extra_params.get("spm_model_path")
        self.tokenizer_name = self.config.extra_params.get("tokenizer", "FullTokenizer")
        self.tgt_type_id = self.config.extra_params.get("tgt_type_id", 1)
        self.max_seq_len = self.config.extra_params.get("max_seq_len", 512)
        self.max_src_len = self.config.extra_params.get("max_src_len", 320)
        self.max_tgt_len = self.config.extra_params.get("max_tgt_len", 64)
        self.max_dec_len = self.config.extra_params.get("max_dec_len", 32)
        # 输入是否分好词
        self.tokenized_input = self.config.extra_params.get("tokenized_input", False)
        self.in_tokens = self.config.extra_params.get("in_tokens", False)
        self.mask_prob = self.config.extra_params.get("mask_prob", 0.5)
        self.continuous_position = self.config.extra_params.get("continuous_position", True)
        self.two_stream =  self.config.extra_params.get("two_stream", True)
        self.random_noise = self.config.extra_params.get("random_noise", False)
        self.task_type = self.config.extra_params.get("task_type", 'normal')
        self.is_dialogue_task = (self.task_type == "dialog")
        self.is_trans_task = (self.task_type == "trans")
        # 对话任务参数
        self.turn_type_size = self.config.extra_params.get("turn_type_size", 16)
        # 解码，保存预测模型，预测/评估时候使用
        self.do_dec = self.config.extra_params.get("do_dec", True) and ("train" not in self.name)
        self.random_seed = self.config.extra_params.get("random_seed", 0)
        np.random.seed(self.random_seed)

        params = {}
        params["do_lower_case"] = self.do_lower_case
        params["spm_model_file"] = self.spm_model_path
        tokenizer_class = RegisterSet.tokenizer.__getitem__(self.tokenizer_name)
        self.tokenizer = tokenizer_class(self.vocab_path, params=params)
        if self.is_trans_task:
            src_params = {}
            src_params["do_lower_case"] = self.config.extra_params.get("src_do_lower_case", True)
            src_params["spm_model_file"] = self.config.extra_params.get("src_spm_model_path")
            src_tokenizer_class = RegisterSet.tokenizer.__getitem__(
                    self.config.extra_params.get("src_tokenizer"))
            self.src_tokenizer = tokenizer_class(self.config.extra_params.get("src_vocab_path"),
                    params=src_params)
            
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]

        if "train" in self.name:
            self.phase = InstanceName.TRAINING
        elif "dev" in self.name:
            self.phase = InstanceName.EVALUATE
        elif "test" in self.name:
            self.phase = InstanceName.TEST
        elif "predict" in self.name: 
            self.phase = "predict"

        # trainer_id 和 dev_count必须要设置，否则多卡的时候每张卡上的数据都是一样的
        self.dev_count = dist.get_world_size()
        self.trainer_id = dist.get_rank()
        self.trainer_nums = self.dev_count

        self.input_data_list = []

        self.place = fluid.CPUPlace()

        self.features = {}

    def create_reader(self):
        """ 静态图模式下用来初始化数据读取的op，调用op为paddle.static.data
        动态图模式下不需要调用
        """
        """create_reader"""

        if self.is_dialogue_task:
            names = ["src_ids", "role_ids", "turn_ids", "pos_ids"]
            ids_num = 4
        else:
            ids_num = 3
            names = ["src_ids", "sent_ids", "pos_ids"]

        input_shapes = [[-1, self.max_seq_len, 1]] * ids_num + [[-1, self.max_seq_len, self.max_seq_len]]
        query_input_shapes = [[-1, self.max_seq_len, 1]] * ids_num + [[-1, self.max_seq_len, self.max_seq_len * 2]]

        input_dtypes = ['int64'] * ids_num + ['float32']
        input_lod_levels = [0] * ids_num + [0]

        if self.do_dec:
            names += ["tgt_src_ids", "tgt_pos_ids", "init_scores", "parent_idx", "tgt_mask_ids", "data_ids"]
            shapes = input_shapes + [[-1, self.max_seq_len, 1], [-1, self.max_seq_len, 1],
                                     [-1, 1], [-1], [-1, 1, self.max_seq_len], [-1, 1]]
            dtypes = input_dtypes + ['int64', 'int64', 'float32', 'int32', 'float32', 'int64']
            lod_levels = input_lod_levels + [2, 2, 2, 0, 0, 0]
        else:
            names += ['tgt_label', 'tgt_pos']
            shapes = input_shapes + [[-1, 1], [-1, 1]]
            dtypes = input_dtypes + ['int64', 'int64']
            lod_levels = input_lod_levels + [0, 0]
            if self.two_stream:
                shapes += query_input_shapes
                dtypes += input_dtypes
                lod_levels += input_lod_levels

        for i in range(len(shapes)): 
            self.input_data_list.append(fluid.layers.data(name="placeholder_" + str(i), shape=shapes[i],
                                                          dtype=dtypes[i], lod_level=lod_levels[i]))


        self.paddle_data_loader = fluid.io.DataLoader.from_generator(feed_list=self.input_data_list,
                                                                     capacity=50, iterable=False)

        logging.debug("{0} create py_reader shape = {1}, types = {2}, \
                      level = {3}: ".format(self.name, shapes, dtypes, lod_levels))

    def get_num_examples(self):
        """get_num_examples"""
        examples = self.read_files(self.config.data_path)
        return len(examples)

    def run(self):
        """run
        """
        if self.paddle_data_loader:
            self.paddle_data_loader.set_batch_generator(self.data_generator())
            self.paddle_data_loader.start()
            logging.info("set data_generator and start.......")
        else:
            raise ValueError("paddle_data_loader is None")

    def instance_fields_dict(self):
       """instance_fields_dict
       """
       fields_instance = self.convert_fields_to_dict(self.input_data_list)
       return fields_instance

    def convert_fields_to_dict(self, field_list, need_emb=False, extra=None):
        """convert fileds to dict"""
        fields_instance = {}

        if self.is_dialogue_task: 
            input_keys = [InstanceName.SRC_IDS, InstanceName.ROLE_IDS,
                          InstanceName.TURN_IDS, InstanceName.POS_IDS]
        else:
            input_keys = [InstanceName.SRC_IDS, InstanceName.SENTENCE_IDS,
                          InstanceName.POS_IDS]
        input_keys += [InstanceName.MASK_IDS]
        input_num = len(input_keys)
        context = {}
        for index in range(input_num):
            context[input_keys[index]] = field_list[index]
        fields_instance["context"] = {InstanceName.RECORD_ID:context}

        if self.do_dec:
            decode_keys = [InstanceName.TGT_SRC_IDS, InstanceName.TGT_POS_IDS,
                           InstanceName.INIT_SCORES, InstanceName.PARENT_IDX,
                           InstanceName.TGT_MASK_IDS, InstanceName.DATA_IDS] 
            decode_inputs = {}
            #print(field_list)
            for index in range(len(decode_keys)):
                #print(len(field_list),input_num,decode_keys,index)
                #print(decode_keys)
                #print(field_list[input_num + index])
                decode_inputs[decode_keys[index]] = field_list[input_num + index]
                
            fields_instance["decode_inputs"] = {InstanceName.RECORD_ID: decode_inputs}
        else:
            mask_keys = [InstanceName.TGT_LABEL, InstanceName.TGT_POS]
            masks = {}
            for index in range(len(mask_keys)):
                masks[mask_keys[index]] = field_list[input_num + index]
            fields_instance["masks"] = {InstanceName.RECORD_ID: masks}
            
            if self.two_stream:
                query = {}
                query_field_list = field_list[-input_num:]
                for index in range(input_num):
                    query[input_keys[index]] = query_field_list[index]
                fields_instance["query"] = {InstanceName.RECORD_ID: query}

        #fields_instance = self.wrap_fields(fields_instance)


        return fields_instance

    def read_files(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        data_id = 0
        with open(input_file, "r", encoding='utf8') as f:
        #with open(input_file, "r") as f:
            try:
                headers = f.readline().strip().split("\t")
                src_indices = [
                    index for index, h in enumerate(headers) if h != "tgt" and h != "knowledge"
                ]
                assert len(src_indices) <= self.tgt_type_id, "len(src_indices) > self.tgt_type_id"
                assert len(src_indices) > 0, "len(src_indices) <= 0"

                Example = namedtuple('Example', ["src", "tgt", "knowledge", "data_id"])
                examples = []
                for line in f:
                    line = line.strip().split("\t")
                    src = []
                    tgt = None
                    knowledge = None
                    assert len(line) == len(headers), "len(line) != len(headers)"
                    for index, text in enumerate(line):
                        if index in src_indices:
                            src.append(text)
                        elif headers[index] == "tgt":
                            tgt = text
                        else:
                            knowledge = text
                    examples.append(Example(src=src, tgt=tgt, knowledge=knowledge, data_id=data_id))
                    data_id += 1
                return examples
            except Exception:
                logging.error("error in read tsv")
                logging.error("traceback.format_exc():\n%s" % traceback.format_exc())

    def _convert_dialogue_example_to_record(self, example):
        """convert_dialogue_example_to_record"""
        turn_split = " __eou__ "
        srcs = example.src[0].split(turn_split)
        if len(srcs) > self.turn_type_size - 1:
            srcs = srcs[len(srcs) - (self.turn_type_size - 1):]
        cur_role_type = len(srcs) % 2
        cur_turn_type = len(srcs)

        token_ids = [self.cls_id]
        role_type_ids = [cur_role_type]
        turn_type_ids = [cur_turn_type]
        position_ids = [0]

        if example.knowledge:
            text = example.knowledge
            if not self.tokenized_input:
                cur_tokens = self.tokenizer.tokenize(convert_to_unicode(text))
            else:
                cur_tokens = convert_to_unicode(text).split(" ")
            if len(cur_tokens) > self.max_src_len - 2:
                cur_tokens = cur_tokens[:self.max_src_len - 2]
            cur_ids = self.tokenizer.convert_tokens_to_ids(cur_tokens) + [self.sep_id]
            token_ids += cur_ids
            role_type_ids += [2] * len(cur_ids)
            turn_type_ids += [0] * len(cur_ids)
            position_ids += list(range(1, len(cur_ids) + 1))

        for text in srcs:
            if not self.tokenized_input:
                cur_tokens = self.tokenizer.tokenize(convert_to_unicode(text))
            else:
                cur_tokens = convert_to_unicode(text).split(" ")
            if len(cur_tokens) > self.max_src_len - 2:
                cur_tokens = cur_tokens[:self.max_src_len - 2]
            cur_ids = self.tokenizer.convert_tokens_to_ids(cur_tokens) + [self.sep_id]
            token_ids += cur_ids
            role_type_ids += [cur_role_type] * len(cur_ids)
            turn_type_ids += [cur_turn_type] * len(cur_ids)
            position_ids += list(range(1, len(cur_ids) + 1))
            cur_turn_type -= 1
            cur_role_type = (cur_role_type + 1) % 2
        if self.continuous_position and len(token_ids) > self.max_src_len:
            token_ids = token_ids[-self.max_src_len:]
            role_type_ids = role_type_ids[-self.max_src_len:]
            turn_type_ids = turn_type_ids[-self.max_src_len:]

        tgt_start_idx = len(token_ids)

        if not self.do_dec:
            assert example.tgt, "example.tgt is None"
            token_ids.append(self.cls_id)
            role_type_ids.append(0)
            turn_type_ids.append(0)
            position_ids.append(0)

            if not self.tokenized_input:
                tgt_tokens = self.tokenizer.tokenize(convert_to_unicode(example.tgt))
            else:
                tgt_tokens = convert_to_unicode(example.tgt).split(" ")
            tgt_ids = self.tokenizer.convert_tokens_to_ids(tgt_tokens) + [self.sep_id]
            if len(tgt_ids) > self.max_tgt_len - 1:
                tgt_ids = tgt_ids[:self.max_tgt_len - 1]
            token_ids += tgt_ids
            role_type_ids += [0] * len(tgt_ids)
            turn_type_ids += [0] * len(tgt_ids)
            position_ids += list(range(1, len(tgt_ids) + 1))

        if self.continuous_position:
            position_ids = list(range(len(token_ids)))

        assert len(token_ids) == len(position_ids) == len(role_type_ids) == len(turn_type_ids), \
            "not len(token_ids) == len(position_ids) == len(role_type_ids) == len(turn_type_ids)"

        Record = namedtuple(
            'Record',
            ['token_ids', 'position_ids', 'role_ids', 'turn_ids', 'tgt_start_idx', 'data_id'])
        record = Record(
            token_ids=token_ids,
            position_ids=position_ids,
            role_ids=role_type_ids,
            turn_ids=turn_type_ids,
            tgt_start_idx=tgt_start_idx,
            data_id=example.data_id)

        return record

    def convert_example_to_record(self, example, max_seq_length=512, tokenizer=None, is_zh=True):
        """convert_example_to_record"""
        token_ids = [self.cls_id]
        text_type_ids = [0]
        position_ids = [0]
        text_type = 0

        src_tokenizer = self.src_tokenizer if self.is_trans_task else self.tokenizer
        for text in example.src:
            if not self.tokenized_input:
                cur_tokens = src_tokenizer.tokenize(convert_to_unicode(text))
            else:
                cur_tokens = convert_to_unicode(text).split(" ")
            if len(cur_tokens) > self.max_src_len - 2:
                cur_tokens = cur_tokens[:self.max_src_len - 2]
            cur_ids = src_tokenizer.convert_tokens_to_ids(cur_tokens) + [self.sep_id]
            token_ids += cur_ids
            text_type_ids += [text_type] * len(cur_ids)
            position_ids += list(range(1, len(cur_ids) + 1))
            text_type += 1

        tgt_start_idx = len(token_ids) 
        
        if not self.do_dec:
            assert example.tgt, "example.tgt is None"
            token_ids.append(self.cls_id)
            text_type_ids.append(self.tgt_type_id)
            position_ids.append(0)

            if not self.tokenized_input:
                tgt_tokens = self.tokenizer.tokenize(convert_to_unicode(example.tgt))
            else:
                tgt_tokens = convert_to_unicode(example.tgt).split(" ")
            tgt_ids = self.tokenizer.convert_tokens_to_ids(tgt_tokens) + [self.sep_id]
            if len(tgt_ids) > self.max_tgt_len - 1:
                tgt_ids = tgt_ids[:self.max_tgt_len - 1]
            token_ids += tgt_ids
            text_type_ids += [self.tgt_type_id] * len(tgt_ids)
            position_ids += list(range(1, len(tgt_ids) + 1))

        if self.continuous_position:
            position_ids = list(range(len(token_ids)))

        assert len(token_ids) == len(position_ids) == len(text_type_ids), \
            "not len(token_ids) == len(position_ids) == len(text_type_ids)"

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'tgt_start_idx', 'data_id'])
        record = Record(
            token_ids=token_ids,
            text_type_ids=text_type_ids,
            position_ids=position_ids,
            tgt_start_idx=tgt_start_idx,
            data_id=example.data_id)

        return record        

    def prepare_batch_data(self, examples, batch_size):
        """generate batch records"""
        batch_records, max_len = [], 0
        for index, example in enumerate(examples):
            if "train" in self.name:
                self.current_example = index

            if self.is_dialogue_task:
                record = self._convert_dialogue_example_to_record(example)
            else:
                record = self.convert_example_to_record(example)

            max_len = max(max_len, len(record.token_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                #print(len(batch_records[0]))
                #print(len(self.pad_batch_records(batch_records)))
                #aaa
                yield self.pad_batch_records(batch_records)
                batch_records, max_len = [record], len(record.token_ids)

        if batch_records:
            yield self.pad_batch_records(batch_records)

    def _to_lodtensor(self, data, lod=None):
        data_tensor = fluid.LoDTensor()
        data_tensor.set(data, self.place)
        if lod is not None:
            data_tensor.set_lod(lod)
        return data_tensor

    def pad_batch_records(self, batch_records):
        """pad batch records"""

        def pad_batch_data_before(insts, pad_idx=0):
            """
            For generation task, 
            it's easier to extract the new generate token in batch mode 
            when paddings are appendded before real ids
            """
            max_len = max(len(inst) for inst in insts)
            inst_data = np.array([inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
            return inst_data.reshape([-1, max_len, 1])

        batch_token_ids = [record.token_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_tgt_start_idx = [record.tgt_start_idx for record in batch_records]
        input_mask = self.gen_unidirectional_mask(batch_token_ids, batch_tgt_start_idx)
        if self.is_dialogue_task:
            batch_role_ids = [record.role_ids for record in batch_records]
            batch_turn_ids = [record.turn_ids for record in batch_records]
            to_pad_list = [batch_token_ids, batch_role_ids, batch_turn_ids, batch_position_ids]
        else:
            batch_text_type_ids = [record.text_type_ids for record in batch_records]
            to_pad_list = [batch_token_ids, batch_text_type_ids, batch_position_ids]
        return_list = []
        for ids in to_pad_list:
            return_list.append(pad_batch_data_before(ids, pad_idx=self.pad_id))
        return_list.append(input_mask)

        max_len = return_list[0].shape[1]
        if self.do_dec:
            batch_data_ids = [record.data_id for record in batch_records]
            batch_data_ids = np.array(batch_data_ids).astype("int64").reshape([-1, 1])
            tgt_word = np.array([[self.cls_id]] * len(batch_token_ids), dtype="int64").reshape(-1, 1, 1)
            tgt_pos_id = np.array(batch_tgt_start_idx, dtype="int64").reshape(-1, 1, 1)

            lods = [list(range(tgt_word.shape[0] + 1))] * 2
            if self.phase == "predict":
                init_score = np.zeros_like(tgt_word, dtype="float32").reshape(-1, 1)
            else:    
                init_score = self._to_lodtensor(np.zeros_like(tgt_word, dtype="float32").reshape(-1, 1),
                        [list(range(tgt_word.shape[0] + 1))] * 2)
                tgt_word = self._to_lodtensor(tgt_word, [list(range(tgt_word.shape[0] + 1))] * 2)
                tgt_pos_id = self._to_lodtensor(tgt_pos_id, [list(range(tgt_pos_id.shape[0] + 1))] * 2)

            init_idx = np.array(list(range(len(batch_token_ids))), dtype="int32")
            tgt_src_attn_bias = np.tile(input_mask[:, ::max_len, :],
                    [1, 1, 1]).astype("float32")
            return_list += [tgt_word, tgt_pos_id, init_score, init_idx,
                    tgt_src_attn_bias, batch_data_ids]
            if self.phase == "predict":
                return_list += [lods]
        else:
            mask_id = self.vocab["[MASK]"]
            tgt_label = []
            tgt_pos = []

            def _gen_noise(tk):
                if self.two_stream:
                    if self.random_noise:
                        return random.randint(0, len(self.vocab)-1)
                    else:
                        return mask_id
                else: #UNILM Style
                    if random.random() < 0.8:
                        return mask_id
                    elif random.random() < 0.5:
                        return random.randint(0, len(self.vocab)-1)
                    else:
                        return tk

            for i in range(len(batch_token_ids)):
                if self.two_stream:
                    tgt_label.extend(batch_token_ids[i][idx] \
                        for idx in range(batch_tgt_start_idx[i] + 1, len(batch_token_ids[i])))
                    for idx in range(batch_tgt_start_idx[i] + 1, len(batch_token_ids[i])):
                        if random.random() < self.mask_prob:
                            batch_token_ids[i][idx] = _gen_noise(batch_token_ids[i][idx])
                else:
                    cur_pos = []
                    cur_label = []
                    for idx in range(batch_tgt_start_idx[i] + 1, len(batch_token_ids[i])):
                        if random.random() > self.mask_prob:
                            continue
                        cur_label.append(batch_token_ids[i][idx])
                        cur_pos.append(idx)
                        batch_token_ids[i][idx] = _gen_noise(batch_token_ids[i][idx])

                    tgt_pos.extend([idx + max_len * i for idx in cur_pos])
                    tgt_label.extend(cur_label)
            return_list[0] = pad_batch_data_before(batch_token_ids, pad_idx=self.pad_id)
            tgt_label = np.array(tgt_label).astype("int64").reshape([-1, 1])
            if self.two_stream:
                input_query_mask, query_token_ids, tgt_pos = self.gen_query_input(batch_token_ids,
                    max_len, batch_tgt_start_idx, mask_id)
                return_list += [tgt_label, tgt_pos, query_token_ids]
                for ids in to_pad_list[1:]:
                    return_list.append(pad_batch_data_before(
                        [ids[i][batch_tgt_start_idx[i]:] for i in range(len(ids))],
                        pad_idx=self.pad_id))
                return_list.append(input_query_mask)
            else:
                tgt_pos = np.array(tgt_pos).astype("int64").reshape([-1, 1])
                return_list += [tgt_label, tgt_pos]

        

        if self.config.need_generate_examples:
            #print(return_list)
            #aaa
            return return_list, batch_records
        else:
            return return_list

    def __iter__(self):
        """迭代器
        """

        epoch = self.config.epoch
        batch_size= self.config.batch_size
        shuffle = self.config.shuffle
        examples = self.read_files(self.config.data_path)
        if self.do_dec:
            features = {}
            for example in examples:
                features[example.data_id] = example
            self.features[self.phase] = features

        # trainer_id 和 dev_count必须要设置，否则多卡的时候每张卡上的数据都是一样的
        self.dev_count = dist.get_world_size()
        self.trainer_id = dist.get_rank()
        self.trainer_nums = self.dev_count

        all_dev_batches = []
        for epoch_index in range(epoch):
            if self.phase == InstanceName.TRAINING:
                self.current_example = 0
                self.current_epoch = epoch_index

            if shuffle:
                np.random.shuffle(examples)
            for batch_data in self.prepare_batch_data(examples, self.config.batch_size):
                if self.config.need_data_distribute:
                    if len(all_dev_batches) < self.dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == self.dev_count:
                        # trick: handle batch inconsistency caused by data sharding for each trainer
                        yield all_dev_batches[self.trainer_id]
                        all_dev_batches = []
                else:
                    yield batch_data

    def data_generator(self):
        """generate data"""

        epoch = self.config.epoch
        batch_size= self.config.batch_size
        shuffle = self.config.shuffle
        examples = self.read_files(self.config.data_path)
        if self.do_dec:
            features = {}
            for example in examples:
                features[example.data_id] = example
            self.features[self.phase] = features

        def wrapper():
            """wraper"""
            all_dev_batches = []
            trainer_id = self.trainer_id
            for epoch_index in range(epoch):
                if self.phase == InstanceName.TRAINING:
                    self.current_example = 0
                    self.current_epoch = epoch_index

                if shuffle:
                    np.random.shuffle(examples)
                for batch_data in self.prepare_batch_data(
                        examples, batch_size):
                    #print(batch_data)
                    #print(len(batch_data[0]))
                    
                    #aa
                    #aa
                    if len(all_dev_batches) < self.dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == self.dev_count:
                        if trainer_id < len(all_dev_batches):
                            #print(all_dev_batches)
                            #aaa
                            yield all_dev_batches[trainer_id]
                        all_dev_batches = []
                if self.phase != InstanceName.TRAINING:
                    if trainer_id < len(all_dev_batches):
                        yield all_dev_batches[trainer_id]

        return wrapper

    def api_generator(self, query_list):
        """api server"""

        if len(query_list) <= 0:
            raise ValueError("query can't be None")
        Example = namedtuple('Example', ["src", "tgt", "knowledge", "data_id"])
        examples = []
        for index, query in enumerate(query_list):
            examples.append(Example(src=[query['answer'], query['paragraph']], tgt=None, knowledge=None, data_id=index))

        for batch_data in self.prepare_batch_data(examples, self.config.batch_size):
            yield batch_data
    
    @staticmethod
    def gen_unidirectional_mask(insts, sent_b_starts=None):
        """
        generate input mask for seq2seq
        """
        max_len = max(len(inst) for inst in insts)
        input_mask_data = np.zeros((len(insts), max_len, max_len))
        for index, mask_data in enumerate(input_mask_data):
            start = sent_b_starts[index]
            end = len(insts[index])
            mask_data[:end, :start] = 1.0
            # Generate the lower triangular matrix using the slice of matrix
            b = np.tril(np.ones([end - start, end - start]), 0)
            mask_data[start:end, start:end] = b
        input_mask_data = np.array(input_mask_data, dtype='float32').reshape([-1, max_len, max_len])
        return input_mask_data

    @staticmethod
    def gen_query_input(token_ids, max_len, sent_b_starts, mask_id):
        """
        generate query input when using two-stream
        """
        bsz = len(sent_b_starts)
        dec_len = [len(token_ids[i]) - sent_b_starts[i] for i in range(bsz)]
        max_len_query = max(dec_len)
        mask_datas = np.zeros((bsz, max_len_query, max_len + max_len_query))
        mask_ids = np.ones((bsz, max_len_query, 1)) * mask_id
        tgt_pos = []
        for i in range(bsz):
            tgt_pos.extend(list(range(max_len_query * i + 1, max_len_query * i + dec_len[i])))
        for index, mask_data in enumerate(mask_datas):
            for i in range(dec_len[index]):
                mask_data[i, :sent_b_starts[index] + i] = 1.0
                mask_data[i, max_len + i] = 1.0

        return (mask_datas.astype('float32'),
            mask_ids.astype('int64'),
            np.array(tgt_pos).reshape([-1, 1]).astype('int64'))


