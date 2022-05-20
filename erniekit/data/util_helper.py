# -*- coding: utf-8 -*
"""
:py:`util_helper`
"""
import random
import numpy as np
import collections
import io
import json
import os
from .field import Field
from ..utils.util_helper import truncation_words
from ..utils.util_helper import convert_to_unicode

def read_doieat_file(file_path, data_filter=None, extra_fields=None):
    """
    read data file of doie annotation tool
    """
    fields = ["fn", "ln", "title", "content", "annotations"]
    if extra_fields:
        extra_fields = [x for x in extra_fields if x and x not in fields]
    else:
        extra_fields = []
    example_cls = collections.namedtuple('DoieatExample', fields + extra_fields)
    is_fileobj = isinstance(file_path, io.IOBase)
    if not is_fileobj:
        fn = os.path.basename(file_path)
    else:
        fn = file_path.name
    with is_fileobj and file_path or open(file_path, "r") as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                example_args = {
                    'fn': fn,
                    'content': d.get('content', ''),
                    'title': d.get('title'),
                    'ln':  d.get('ln', -1),
                    'annotations': d.get('annotations', {})
                }
            except:
                continue
            for field in extra_fields:
                example_args[field] = d.get(field)
            example = example_cls(**example_args)
            if data_filter and data_filter(example):
                continue
            yield example

def read_tsv_file(file_path, headers=None, data_filter=None):
    """
    read tsv data file
    """
    is_fileobj = isinstance(file_path, io.IOBase)
    if not is_fileobj:
        fn = os.path.basename(file_path)
    else:
        fn = file_path.name
    with is_fileobj and file_path or open(file_path, "r") as f:
        if not headers:
            headers = ['fn', 'ln'] + next(f).rstrip().split("\t")
        else:
            headers = ['fn', 'ln'] + headers
        example_cls = collections.namedtuple('TsvExample', headers)
        for ln, line in enumerate(f):
            line = convert_to_unicode(line.rstrip())
            items = line.rstrip().split("\t")
            example = example_cls(fn, ln + 1, *items)
            if data_filter and data_filter(example):
                continue
            yield example

def read_json_file(file_path):
    """
    read json data file, one json string per line
    """
    is_fileobj = isinstance(file_path, io.IOBase)
    if not is_fileobj:
        fn = os.path.basename(file_path)
    else:
        fn = file_path.name
    with is_fileobj and file_path or open(file_path, "r") as f:
        for ln, line in enumerate(f):
            line = line.strip()
            try:
                example = json.loads(line)
            except:
                continue
            example["fn"] = fn
            example["ln"] = ln
            yield example

def convert_text_to_id(text, field_config):
    """将一个明文样本转换成id
    :param text: 明文文本
    :param field_config : Field类型
    :return:
    """
    if not text:
        raise ValueError("text input is None")
    if not isinstance(field_config, Field):
        raise TypeError("field_config input is must be Field class")

    if field_config.need_convert:
        tokenizer = field_config.tokenizer
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
    else:
        ids = text.split(" ")

    # 加上截断策略
    if len(ids) > field_config.max_seq_len:
        ids = truncation_words(ids, field_config.max_seq_len, field_config.truncation_type)

    return ids


def padding_batch_data(insts,
                   pad_idx=0,
                   return_seq_lens=False,
                   paddle_version_code=1.6):
    """
    :param insts:
    :param pad_idx:
    :param return_seq_lens:
    :param paddle_version_code:
    :return:
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, max_len, 1])]

    if return_seq_lens:
        seq_lens_type = [-1]
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape(seq_lens_type)]

    return return_list if len(return_list) > 1 else return_list[0]


def mask_batch_data(insts, return_seq_lens=False, paddle_version_code=1.6):
    """
    :param insts:
    :param return_seq_lens:
    :param paddle_version_code:
    :return:
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)

    input_mask_data = np.array([[1] * len(inst) + [0] *
                                (max_len - len(inst)) for inst in insts])
    input_mask_data = np.expand_dims(input_mask_data, axis=-1)
    return_list += [input_mask_data.astype("float32")]

    if return_seq_lens:
        seq_lens_type = [-1]
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape(seq_lens_type)]

    return return_list if len(return_list) > 1 else return_list[0]


def pad_batch_data(insts,
                   insts_data_type="int64",
                   pad_idx=0,
                   final_cls=False,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False,
                   force_max_len=None,
                   force_3d=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    if force_max_len is not None:
        max_len = force_max_len
    else:
        max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    end_shape = [-1, max_len]
    # src-id、pos-id是否需要3维，默认为false，ernie-doc为true
    if force_3d:
        end_shape = [-1, max_len, 1]
    # id
    if final_cls:
        inst_data = np.array(
            [inst[:-1] + list([pad_idx] * (max_len - len(inst))) + [inst[-1]] for inst in insts])
    else:
        inst_data = np.array(
            [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype(insts_data_type).reshape(end_shape)]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape(end_shape)]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] *
                                    (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32").reshape(end_shape)]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens_type = [-1]
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape(seq_lens_type)]

    return return_list if len(return_list) > 1 else return_list[0]


def generate_pad_batch_data(insts,
                   insts_data_type="int64",
                   pad_idx=0,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False,
                   paddle_version_code=1.6):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    # id
    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype(insts_data_type).reshape([-1, max_len])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] *
                                    (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens_type = [-1]
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape(seq_lens_type)]

    return return_list if len(return_list) > 1 else return_list[0]


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


def gen_query_input2(token_ids, max_len, sent_b_starts, mask_id):
    """
    generate query input when using two-stream
    """
    bsz = len(sent_b_starts)
    dec_len = [len(token_ids[i]) - sent_b_starts[i] for i in range(bsz)]
    max_len_query = max(dec_len)
    mask_datas = np.zeros((bsz, max_len_query, max_len + max_len_query))
    mask_ids = np.ones((bsz, max_len_query, 1)) * mask_id
    mask_phonetic_a_ids = np.ones((bsz, max_len_query, 1)) * 0
    mask_phonetic_b_ids = np.ones((bsz, max_len_query, 1)) * 0
    mask_glyph_a_ids = np.ones((bsz, max_len_query, 1)) * 0
    mask_glyph_b_ids = np.ones((bsz, max_len_query, 1)) * 0
    mask_glyph_c_ids = np.ones((bsz, max_len_query, 1)) * 0
    mask_glyph_d_ids = np.ones((bsz, max_len_query, 1)) * 0
    tgt_pos = []
    for i in range(bsz):
        tgt_pos.extend(list(range(max_len_query * i + 1, max_len_query * i + dec_len[i])))
    for index, mask_data in enumerate(mask_datas):
        for i in range(dec_len[index]):
            mask_data[i, :sent_b_starts[index] + i] = 1.0
            mask_data[i, max_len + i] = 1.0
    return  {"mask_input": mask_datas.astype('float32'),
             "src_ids": mask_ids.astype('int64'),
             "phonetic_a_ids": mask_phonetic_a_ids.astype("int64"),
             "phonetic_b_ids": mask_phonetic_b_ids.astype("int64"),
             "glyph_a_ids": mask_glyph_a_ids.astype("int64"),
             "glyph_b_ids": mask_glyph_b_ids.astype("int64"),
             "glyph_c_ids": mask_glyph_c_ids.astype("int64"),
             "glyph_d_ids": mask_glyph_d_ids.astype("int64"),
             "tgt_pos": np.array(tgt_pos).reshape([-1, 1]).astype('int64')}

def get_hierar_relations_array(model_params):
    """
    generate hierarchical relations array when using hierarchical label classification
    """
    label_map, hierar_relations = {}, {}
    hierar_label_path=model_params.get("hierar_label_path", "")
    hierar_relations_path= model_params.get("hierar_relations_path", "")
    assert hierar_label_path != '', "'hierar_label_path' cannot be empty"
    assert hierar_relations_path != '', "'hierar_relations_path' cannot be empty" 
    
    file_hierar_label = open(hierar_label_path, "r")
    for line in file_hierar_label:
        label_name, label_id = convert_to_unicode(line.strip()).split("\t")
        label_map[label_name] = label_id
    
    file_hierar_relation = open(hierar_relations_path, "r")
    for line in file_hierar_relation:
        line_split = convert_to_unicode(line.strip("\n")).split("\t")
        parent_label, children_label = line_split[0], line_split[1:]
        if parent_label not in label_map:
            continue
        parent_label_id = label_map[parent_label]
        children_label_ids = [label_map[child_label] \
            for child_label in children_label if child_label in label_map]
        hierar_relations[parent_label_id] = children_label_ids

    num_label = len(label_map)
    hierar_array = np.zeros((num_label, num_label), dtype='float32')

    for parent_label, child_labels in hierar_relations.items():
        for child_label in child_labels:
            hierar_array[int(parent_label), int(child_label)] = 1
    return hierar_array

def get_random_pos_id(batch_pos_ids, max_pos_len=2048):
    """get random position ids"""
    batch_size = len(batch_pos_ids)
    random_batch_pos_ids = []
    for pos_ids in batch_pos_ids:
        len_pos = len(pos_ids)
        random_pos_start = random.randint(1, max_pos_len - 1)
        random_pos_ids = [0, random_pos_start]
        last_pos_id = random_pos_start
        for _ in range(len_pos - 2):
            random_gap = random.sample(range(1, 4), 1)[0]
            pos_id = (last_pos_id + random_gap) % max_pos_len
            if pos_id == 0:
                pos_id += 1
            random_pos_ids.append(pos_id)
            last_pos_id = pos_id
        assert len(random_pos_ids) == len_pos
        random_batch_pos_ids.append(random_pos_ids)
    return random_batch_pos_ids
