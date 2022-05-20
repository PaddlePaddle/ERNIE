# -*- coding: utf-8 -*
"""
IEReader
"""
import os
import re
import json
import logging
from collections import namedtuple
import numpy as np
import paddle
import paddle.fluid as fluid
from erniekit.common.rule import InstanceName
from erniekit.common.register import RegisterSet
from erniekit.data.data_set_reader.base_dataset_reader import BaseDataSetReader
from erniekit.data.util_helper import pad_batch_data

import paddle.distributed as dist

@RegisterSet.data_set_reader.register
class IEReader(BaseDataSetReader):
    """
    IEReader
    """
    def __init__(self, name, fields, config):
        """
           init params
        """
        BaseDataSetReader.__init__(self, name, fields, config)
        self.max_seq_len = config.extra_params.get("max_seq_len")
        self.do_lower_case = config.extra_params.get("do_lower_case")
        self.vocab_path = config.extra_params.get("vocab_path")
        self.num_labels = config.extra_params.get("num_labels")
        self.tokenizer = config.extra_params.get("tokenizer")

        self.need_generate_examples = config.extra_params.get("need_generate_examples", False)
        # self.need_data_distribute = config.get("need_data_distribute", False)



        tokenizer_class = RegisterSet.tokenizer.__getitem__(self.tokenizer)
        self.tokenizer = tokenizer_class(vocab_file=self.vocab_path, params={"do_lower_case": self.do_lower_case})

        self.vocab = self.tokenizer.vocabulary.vocab_dict
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.in_tokens = config.extra_params.get("in_tokens")

        if "train" in self.name:
            self.phase = InstanceName.TRAINING
        elif "dev" in self.name:
            self.phase = InstanceName.EVALUATE
        elif "test" in self.name or "predict" in self.name:
            self.phase = InstanceName.TEST
        else:
            logging.info(self.name)

        self.trainer_id = 0
        self.trainer_nums = 1
        if os.getenv("PADDLE_TRAINER_ID"):
            self.trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        if os.getenv("PADDLE_TRAINERS_NUM"):
            self.trainer_nums = int(os.getenv("PADDLE_TRAINERS_NUM"))

        if "train" in self.name or "predict" in self.name:
            self.dev_count = self.trainer_nums
        elif "dev" in self.name or "test" in self.name:
            self.dev_count = 1
            # if self.use_multi_gpu_test:
            #     self.dev_count = min(self.trainer_nums, 8)
        else:
            logging.info(self.name)

        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

        with open(config.extra_params.get("label_map_config")) as fp:
            self.label_map = json.load(fp)

    def read_json(self, input_file):
        """
        read json
        """
        examples = []
        with open(input_file, "r") as fp:
            for line in fp:
                examples.append(json.loads(line.strip()))
        return examples

    def _is_chinese_or_punct(self, char):
        """
        is chinese or punct
        """
        if re.match("[0-9a-zA-Z]", char):
            return False
        return True

    def convert_example_to_record(self, example, tokenizer, max_seq_len=512):
        """
        convert exammple to record
        """
        text = example["text"]
        if "spo_list" in example:
            spo_list = example["spo_list"]
        else:
            spo_list = None

        tokens = []
        buff = ""
        for char in text:
            if self._is_chinese_or_punct(char):
                if buff != "":
                    tokens.append(buff)
                    buff = ""
                tokens.append(char)
            else:
                buff += char
        if buff != "":
            tokens.append(buff)

        beg_ids = []
        end_ids = []
        subwords = []
        labels = []
        temp = ""
        
        for token in tokens:
            temp += token
            for i, subword in enumerate(tokenizer.tokenize(token)):
                beg_id = len(temp) - len(token)
                end_id = len(temp) - 1
                beg_ids.append(beg_id)
                end_ids.append(end_id)
                subwords.append(subword)
                label = [0] * self.num_labels
                if spo_list:
                    for spo in spo_list:
                        if beg_id == spo["subject"][0]:
                            if i == 0:
                                label[self.label_map["B-%s@S" % spo["predicate"]]] = 1
                            else:
                                label[1] = 1
                        elif spo["subject"][0] < beg_id and end_id < spo["subject"][1]:
                            label[1] = 1
                        if beg_id == spo["object"][0]:
                            if i == 0:
                                label[self.label_map["B-%s@O" % spo["predicate"]]] = 1
                            else:
                                label[1] = 1
                        elif spo["object"][0] < beg_id and end_id < spo["object"][1]:
                            label[1] = 1
                    if sum(label) == 0:
                        label[0] = 1
                labels.append(label)
                if len(subwords) >= max_seq_len - 2:
                    break
            else:
                continue
            break

        subwords = ["[CLS]"] + subwords + ["[SEP]"]
        special = [0] * self.num_labels
        special[0] = 1
        labels = [special] + labels + [special]
        beg_ids = [-1] + beg_ids + [-1]
        end_ids = [-1] + end_ids + [-1]

        src_ids = tokenizer.convert_tokens_to_ids(subwords)
        sent_ids = [0] * len(src_ids)
        pos_ids = list(range(len(src_ids)))
        task_ids = [0] * len(src_ids)
        
        Record = namedtuple("Record",
                            ["text", "src_ids", "sent_ids", "pos_ids", "task_ids", "beg_ids", "end_ids", "labels"])
        record = Record(
             text=text,
             src_ids=src_ids,
             sent_ids=sent_ids,
             pos_ids=pos_ids,
             beg_ids=beg_ids,
             end_ids=end_ids,
             task_ids=task_ids,
             labels=labels)
        return record

    def _get_rel_pos(self, batch_ids):
        """
        get rel pos
        """
        max_len = max([len(ids) for ids in batch_ids])
        rel_pos = np.reshape(np.tile(np.arange(max_len), [max_len]), [max_len, max_len])
        rel_pos = rel_pos - np.transpose(rel_pos)
        rel_pos = np.maximum(-4, np.minimum(4, rel_pos))
        rel_pos = rel_pos + 4
        return rel_pos.astype("int64").reshape([max_len, max_len, 1])

    def _get_deep_id(self, batch_ids):
        """
        get deep id
        """
        batch_size = len(batch_ids)
        max_len = max(len(ids) for ids in batch_ids)
        padded_deep_ids = np.zeros((batch_size, max_len))
        return padded_deep_ids.astype("int64").reshape([batch_size, max_len, 1])

    def _pad_batch_label(self, batch_labels):
        """
        pad batch label
        """
        special = [0] * self.num_labels
        special[0] = 1
        special = np.array(special)
        max_len = max(len(labels) for labels in batch_labels)
        padded_labels = []
        for labels in batch_labels:
            labels = np.concatenate((np.array(labels), np.tile(special, (max_len - len(labels), 1))))
            padded_labels.append(labels)
        padded_labels = np.stack(padded_labels).astype("float32")
        return padded_labels

    def pad_batch_records(self, batch_records):
        """
        pad batch records
        """
        batch_text = [record.text for record in batch_records]
        batch_src_ids = [record.src_ids for record in batch_records]
        batch_sent_ids = [record.sent_ids for record in batch_records]
        batch_pos_ids = [record.pos_ids for record in batch_records]
        batch_beg_ids = [record.beg_ids for record in batch_records]
        batch_end_ids = [record.end_ids for record in batch_records]
        batch_task_ids = [record.task_ids for record in batch_records]
        batch_labels = [record.labels for record in batch_records]

        padded_src_ids, input_mask, batch_seq_lens = pad_batch_data(batch_src_ids, 
            pad_idx=self.pad_id, 
            return_input_mask=True, 
            return_seq_lens=True)
        padded_sent_ids = pad_batch_data(batch_sent_ids, pad_idx=self.pad_id)
        padded_pos_ids = pad_batch_data(batch_pos_ids, pad_idx=self.pad_id)
        padded_beg_ids = pad_batch_data(batch_beg_ids, pad_idx=self.pad_id)
        padded_end_ids = pad_batch_data(batch_end_ids, pad_idx=self.pad_id)
        padded_rel_pos_ids = self._get_rel_pos(batch_src_ids)
        padded_deep_ids = self._get_deep_id(batch_src_ids)
        padded_task_ids = pad_batch_data(batch_task_ids, pad_idx=self.pad_id)
        padded_labels = self._pad_batch_label(batch_labels)

        return_list = [padded_src_ids,
                       padded_sent_ids, 
                       padded_pos_ids,
                       padded_rel_pos_ids, 
                       padded_deep_ids,
                       padded_task_ids,
                       input_mask,
                       padded_beg_ids, 
                       padded_end_ids,
                       batch_seq_lens.flatten(),
                       padded_labels]
        if self.need_generate_examples:
            return return_list, batch_text
        return return_list

    def prepare_batch_data(self, examples, batch_size):
        """
        prepare batch data
        """
        batch_records, max_len = [], 0
        for index, example in enumerate(examples):
            if self.phase == InstanceName.TRAINING:
                self.current_example += 1
            
            record = self.convert_example_to_record(example, self.tokenizer, self.max_seq_len)
            
            max_len = max(max_len, len(record.src_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                yield self.pad_batch_records(batch_records)
                batch_records, max_len = [record], len(record.src_ids)

        if batch_records:
            yield self.pad_batch_records(batch_records)

    def __iter__(self):
        """迭代器
        """
        assert os.path.isdir(self.config.data_path), "%s must be a directory that stores data files" \
                                                     % self.config.data_path
        data_files = os.listdir(self.config.data_path)

        assert len(data_files) > 0, "%s is an empty directory" % self.config.data_path

        # trainer_id 和 dev_count必须要设置，否则多卡的时候每张卡上的数据都是一样的
        self.dev_count = dist.get_world_size()
        self.trainer_id = dist.get_rank()
        self.trainer_nums = self.dev_count

        all_dev_batches = []
        for epoch_index in range(self.config.epoch):
            self.current_example = 0
            self.current_epoch = epoch_index

            for input_file in data_files:
                examples = self.read_json(os.path.join(self.config.data_path, input_file))
                if self.config.shuffle:
                    np.random.shuffle(examples)

                for batch_data in self.prepare_batch_data(examples, self.config.batch_size):
                    if self.config.need_data_distribute:
                        if len(all_dev_batches) < self.dev_count:
                            all_dev_batches.append(batch_data)
                        if len(all_dev_batches) == self.dev_count:
                            yield all_dev_batches[self.trainer_id]
                            all_dev_batches = []
                    else:
                        yield batch_data

    def create_reader(self):
        """
        create reader
        """
        shapes = [[-1, -1],    
                  [-1, -1],
                  [-1, -1],
                  [-1, -1, 1],
                  [-1, -1, 1],
                  [-1, -1],
                  [-1, -1],
                  [-1, -1],
                  [-1, -1],
                  [-1],
                  [-1, -1, self.num_labels]]
        dtypes = ["int64", "int64", "int64", "int64", "int64", "int64", "float32", "int64", "int64", "int64", "float32"]
        lod_levels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        feed_names = ['src_id', 'sent_id', 'pos_id', 'rel_pos_ids', 'deep_ids', 
        'task_ids', 'mask_ids', 'beg_ids', 'end_ids', 'seq_lens', 'label']

        for i in range(len(shapes)):
            self.input_data_list.append(paddle.static.data(name=feed_names[i], shape=shapes[i],
                                                   dtype=dtypes[i], lod_level=lod_levels[i]))    

    def convert_fields_to_dict(self, field_list, need_emb=False):
        """
        convert fields to dict
        """
        fields_instance = {}
        #将原本field需要的域中的structure_fields_dict、filed.name、get_field_length()等统一起来
        record_id_dict_text_a = {
             InstanceName.SRC_IDS: field_list[0],
             InstanceName.SENTENCE_IDS: field_list[1],
             InstanceName.POS_IDS: field_list[2],
             InstanceName.REL_POS_IDS: field_list[3],
             InstanceName.DEEP_IDS: field_list[4],
             InstanceName.TASK_IDS: field_list[5],
             InstanceName.MASK_IDS: field_list[6],
             InstanceName.BEG_IDS: field_list[7],
             InstanceName.END_IDS: field_list[8],
             InstanceName.SEQ_LENS: field_list[9]
        }
        record_dict_text_a = {
            InstanceName.RECORD_ID: record_id_dict_text_a,
            InstanceName.RECORD_EMB: None
        }
        fields_instance["text_a"] = record_dict_text_a

        record_id_dict_label = {
             InstanceName.SRC_IDS: field_list[10]
        }
        record_dict_label = {
            InstanceName.RECORD_ID: record_id_dict_label,
            InstanceName.RECORD_EMB: None
        }
        fields_instance["label"] = record_dict_label

        return fields_instance

    def instance_fields_dict(self):
        """
        instance fields dict
        """
        fields_instance = self.convert_fields_to_dict(self.input_data_list)
        return fields_instance

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def get_num_examples(self):
        """
        get num examples
        """
        data_path = self.config.data_path
        if os.path.isdir(data_path):
            data_files = [os.path.join(data_path, fn) for fn in os.listdir(data_path)]
            assert len(data_files) > 0, "%s is an empty directory" % data_path
        elif os.path.isfile(data_path):
            data_files = [data_path]
        else:
            raise ValueError("%s must be a directory that stores data files or path to a file" % data_path)
        self.num_examples = 0
        for input_file in data_files:
            current_examples = self.read_json(input_file)
            self.num_examples = len(current_examples)
        return self.num_examples

    def api_generator(self, query):
        """python api server
        :param query: list
        :return
        """
        if len(query) <= 0:
            raise ValueError("query can't be None")
        # for batch_data in self.prepare_batch_data(query, 1):
        #     yield batch_data
        batch_records = []
        record = self.convert_example_to_record(query, self.tokenizer, self.max_seq_len)
        batch_records.append(record)
        ids, samples = self.pad_batch_records(batch_records)
        return ids, samples