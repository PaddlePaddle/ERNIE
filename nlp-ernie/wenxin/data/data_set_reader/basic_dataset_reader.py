# -*- coding: utf-8 -*
"""
:py:class:`BasicDataSetReader`
"""
import csv
import os
import sys
import traceback
import logging
from collections import namedtuple
import numpy as np
import six
from ...common.register import RegisterSet
from ...common.rule import InstanceName
from .base_dataset_reader import BaseDataSetReader
import paddle.distributed as dist


@RegisterSet.data_set_reader.register
class BasicDataSetReader(BaseDataSetReader):
    """BasicDataSetReader:一个基础的data_set_reader，实现了文件读取，id序列化，token embedding化等基本操作
    """

    def __init__(self, name, fields, config):
        """__init__
        """
        BaseDataSetReader.__init__(self, name, fields, config)

    def create_reader(self):
        """ 静态图模式下用来初始化数据读取的op，调用op为paddle.static.data
        动态图模式下不需要调用
        """
        if not self.fields:
            raise ValueError("fields can't be None")

        for item in self.fields:
            if not item.field_reader:
                raise ValueError("{0}'s field_reader is None".format(item.name))
            if item.join_calculation:
                self.input_data_list.extend(item.field_reader.init_reader(dataset_type=InstanceName.TYPE_DATA_LOADER))

    def instance_fields_dict(self):
        """将输入进来的tensor数组经过转换，得到fields_id, 视情况构造embedding，然后结构化成dict类型返回给组网部分
        :return: 实例化的dict，保存了各个field的id和embedding(可以没有，是情况而定), 给trainer用
        """
        fields_instance = self.convert_fields_to_dict(self.input_data_list)
        return fields_instance

    def convert_fields_to_dict(self, field_list, need_emb=True):
        """实例化fields_dict，保存了各个field的id和embedding(可以没有，是情况而定),
        当need_emb=False的时候，可以直接给predictor调用
        :param field_list:
        :param need_emb:
        :return: dict
        """
        start_index = 0
        fields_instance = {}
        for index, filed in enumerate(self.fields):
            if not filed.join_calculation:
                continue
            item_dict = filed.field_reader.structure_fields_dict(field_list, start_index, need_emb=need_emb)
            fields_instance[filed.name] = item_dict
            start_index += filed.field_reader.get_field_length()

        return fields_instance

    # def convert_input_list_to_dict(self, input_list):
    #     """将dataloader读取的样本数据，由list类型转换成dict类型，静态图模式的execute.run调用
    #     """
    #     assert len(self.input_data_list) == len(input_list), "len of input_data_list must equal " \
    #                                                          "input_list in DataSet.convert_input_list_to_dict"
    #
    #     feed_dict = collections.OrderedDict()
    #     for index, data in enumerate(self.input_data_list):
    #         feed_dict[data.name] = input_list[index]
    #
    #     return feed_dict

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
                examples = self.read_files(os.path.join(self.config.data_path, input_file))
                if self.config.shuffle:
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

    def read_files(self, file_path, quotechar=None):
        """读取明文文件
        :param file_path
        :return: 以namedtuple数组形式输出明文样本对应的实例
        """
        line_index = 0
        with open(file_path, "r") as f:
            try:
                examples = []
                reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
                len_fields = len(self.fields)
                field_names = []

                for filed in self.fields:
                    field_names.append(filed.name)

                self.Example = namedtuple('Example', field_names)
                for linenum, line in enumerate(reader):
                    line_index = linenum + 1
                    if len(line) == len(field_names):
                        example = self.Example(*line)
                        examples.append(example)
                    else:
                        logging.warn('fileds in file %s of line %s not match: got %d, expect %d' \
                                     % (file_path, line_index, len(line), len_fields))
                return examples

            except Exception:
                logging.error("error in read tsv, maybe occur in linenum %s " % line_index)
                logging.error("traceback.format_exc():\n%s" % traceback.format_exc())

    def prepare_batch_data(self, examples, batch_size):
        """将明文样本按照data_loader需要的格式序列化成一个个batch输出
        :param examples:
        :param batch_size:
        :return:
        """
        batch_records = []
        for index, example in enumerate(examples):
            self.current_example += 1
            if len(batch_records) < batch_size:
                batch_records.append(example)
            else:
                yield self.pad_batch_records(batch_records)
                batch_records = [example]

        if batch_records:
            yield self.pad_batch_records(batch_records)

    def pad_batch_records(self, batch_records):
        """
        :param batch_records:
        :return:
        """
        return_list = []
        example = batch_records[0]
        linenums = []
        for index, key in enumerate(example._fields):
            text_batch = []
            for record in batch_records:
                text_batch.append(record[index])
            if key == 'linenum':
                linenums = text_batch
            try:
                if self.fields[index].join_calculation:
                    id_list = self.fields[index].field_reader.convert_texts_to_ids(text_batch)
                    return_list.extend(id_list)
            except Exception:
                lines = ''
                for linenum, text in zip(linenums, text_batch):
                    lines += 'linenum %s text: %s \n' % (linenum, text)
                logging.error("error occur! msg: %s, batch data: \n%s " % (traceback.format_exc(), lines))
                six.reraise(*sys.exc_info())

        if self.config.need_generate_examples:
            return return_list, batch_records
        else:
            return return_list

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def get_num_examples(self):
        """get number of example"""
        data_files = os.listdir(self.config.data_path)
        assert len(data_files) > 0, "%s is an empty directory" % self.config.data_path
        sum_examples = 0
        for input_file in data_files:
            examples = self.read_files(os.path.join(self.config.data_path, input_file))
            sum_examples += len(examples)

        self.num_examples = sum_examples

        return self.num_examples

    def api_generator(self, query):
        """python api server
        :param query: list
        :return
        """
        if len(query) <= 0:
            raise ValueError("query can't be None")

        field_names = []
        for filed in self.fields:
            field_names.append(filed.name)

        Example = namedtuple('Example', field_names)
        example = Example(*query)
        ids, samples = self.pad_batch_records([example])
        return ids, samples
