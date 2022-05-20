# -*- coding: utf-8 -*
"""
:py:class:`ErnieClassificationDataSetReader`
"""
import logging
import sys
import traceback
import six
from erniekit.common.register import RegisterSet
from erniekit.data.data_set_reader.basic_dataset_reader import BasicDataSetReader


@RegisterSet.data_set_reader.register
class ErnieClassificationDataSetReader(BasicDataSetReader):
    """ErnieClassificationDataSetReader:一个基础的data_set_reader，实现了文件读取，id序列化，token embedding化等基本操作
    """

    def pad_batch_records(self, batch_records):
        """
        :param batch_records:
        :return:
        """
        return_list = []
        example = batch_records[0]
        linenums = []
        text_combine_batch = []

        for index, key in enumerate(example._fields):
            text_batch = []
            for record in batch_records:
                text_batch.append(record[index])
            if "text_" in key:
                text_combine_batch.append(text_batch)

        for index, key in enumerate(example._fields):
            text_batch = []
            for record in batch_records:
                text_batch.append(record[index])
            if key == 'linenum':
                linenums = text_batch
            try:
                if not self.fields[index].join_calculation:
                    continue
                if "text_" in key:
                    id_list = self.fields[index].field_reader.convert_texts_to_ids(text_combine_batch)
                else:
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

