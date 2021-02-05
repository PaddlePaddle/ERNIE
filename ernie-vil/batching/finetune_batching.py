#    Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");                                                      
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
        
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" prepare data format for finetuning tasks """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from six.moves import xrange


def prepare_batch_data(batch_records, num_choice, pad_id, task_index, task_num):
    """
    prepare batch data for finetuning tasks
    """
    batch_input_ids = []
    batch_input_pos = []
    batch_seg_ids = []
    batch_input_masks = []
    num_sample = len(batch_records)
    batch_lens = [record["input_lens"] for record in batch_records]
    batch_labels = [record["target"] for record in batch_records]
    binary_labels = np.zeros([num_choice * num_sample, 1], dtype='float32')
    for i, l in enumerate(batch_labels):
        binary_labels[i * num_choice + l] = 1.0
    labels = np.array(batch_labels).astype("int64").reshape([-1, 1])
    image_features = [record["features"] for record in batch_records]
    image_boxes = [record["boxes"] for record in batch_records]
    batch_anno_ids = np.array([record["anno_id"] for record in batch_records]).astype("int64").reshape([-1, 1])
    max_len = max([max(lens) for lens in batch_lens])
    for i in range(len(batch_records)):
        batch_input_ids.append([inst + list([pad_id] * (max_len - len(inst))) \
            for inst in batch_records[i]["input_ids"]])
        batch_input_pos.append([inst + list([pad_id] * (max_len - len(inst))) \
            for inst in batch_records[i]["input_pos"]])
        batch_seg_ids.append([inst + list([pad_id] * (max_len - len(inst)))   \
            for inst in batch_records[i]["segment_ids"]])
        batch_input_masks.append([[1] * len(inst) + [0] * (max_len - len(inst))    \
            for inst in batch_records[i]["input_ids"]])

    image_embedding, image_mask = pad_feature_data(image_features, return_mask=True)
    image_loc = pad_feature_data(image_boxes)
    src_ids = np.array(batch_input_ids).astype("int64").reshape([num_choice * num_sample, max_len, 1])
    src_pos = np.array(batch_input_pos).astype("int64").reshape([num_choice * num_sample, max_len, 1])
    src_seg = np.array(batch_seg_ids).astype("int64").reshape([num_choice * num_sample, max_len, 1])
    src_masks = np.array(batch_input_masks).astype("float32").reshape([num_choice * num_sample, max_len, 1])
    batch, seq_len, fea_len = image_embedding.shape
    image_embedding = np.tile(np.expand_dims(image_embedding, axis=1),    \
        (1, num_choice, 1, 1)).reshape([num_choice * batch, seq_len, fea_len])
    image_mask = np.tile(np.expand_dims(image_mask, axis=1),        \
        (1, num_choice, 1, 1)).reshape([num_choice * batch, seq_len, 1])
    image_loc = np.tile(np.expand_dims(image_loc, axis=1),     \
        (1, num_choice, 1, 1)).reshape([num_choice * batch, seq_len, 5])
    return_list = [src_ids, src_pos, src_seg, src_masks, \
        image_embedding, image_loc, image_mask, labels, batch_anno_ids]
    return_list.append(np.array([task_index]).astype('int64'))
    return_list.append(binary_labels)
    for i in xrange(task_num):
        if i == task_index:
            return_list.append(np.array([1.0]).astype("float32"))
        else:
            return_list.append(np.array([0.0]).astype("float32"))
    return return_list


def prepare_vqa_batch_data(insts,
                       total_token_num,
                       task_index,
                       task_num,
                       voc_size=0,
                       pad_id=None,
                       cls_id=None,
                       sep_id=None,
                       mask_id=None,
                       return_input_mask=True,
                       return_max_len=True,
                       return_num_token=False):
    """
    prepare batch data for vqa tasks
    """
    batch_src_ids = [inst["token_ids"] for inst in insts]
    batch_sent_ids = [inst["sent_ids"] for inst in insts]
    batch_pos_ids = [inst["pos_ids"] for inst in insts]
    batch_image_embedding = [inst["image_embeddings"] for inst in insts]
    batch_image_loc = [inst["image_loc"] for inst in insts]
    batch_weight_label = [inst["weight_labels"] for inst in insts]
    q_ids = np.array([inst["question_id"] for inst in insts])

    #pad and trans to numpy array
    src_id, self_input_mask, seq_lens = pad_batch_data(
        batch_src_ids, pad_idx=pad_id, return_input_mask=True, return_seq_lens = True)
    pos_id = pad_batch_data(batch_pos_ids, pad_idx=pad_id)
    sent_id = pad_batch_data(batch_sent_ids, pad_idx=pad_id)
    weight_labels = np.array(batch_weight_label).astype("float32")
    #image_embedding_ori = copy.deepcopy(batch_image_embedding)
    image_embedding, image_mask = pad_feature_data(batch_image_embedding, return_mask = True)
    #image_embedding_ori = pad_feature_data(image_embedding_ori)
    image_loc = pad_feature_data(batch_image_loc)

    return_list = [
        src_id, pos_id, sent_id, self_input_mask,  \
            image_embedding, image_loc, image_mask, weight_labels, q_ids
    ]
    return return_list


def prepare_flickr_data(insts,
                       total_token_num,
                       task_index,
                       task_num,
                       voc_size=0,
                       pad_id=None,
                       cls_id=None,
                       sep_id=None,
                       mask_id=None,
                       outs=4,
                       return_input_mask=True,
                       return_max_len=True,
                       return_num_token=False):
    """
    prepare flickr data for finetuning tasks
    """
    if outs > 1:
        batch_src_ids = [inst["token_ids"][out] for inst in insts for out in range(outs)]
        batch_sent_ids = [inst["sent_ids"][out] for inst in insts for out in range(outs)]
        batch_pos_ids = [inst["pos_ids"][out] for inst in insts for out in range(outs)]
        batch_image_embedding = [inst["image_embeddings"][out] for inst in insts for out in range(outs)]
        batch_image_loc = [inst["image_loc"][out] for inst in insts for out in range(outs)]
    else:
        batch_src_ids = [inst["token_ids"] for inst in insts]
        batch_sent_ids = [inst["sent_ids"] for inst in insts]
        batch_pos_ids = [inst["pos_ids"] for inst in insts]
        batch_image_embedding = [inst["image_embeddings"] for inst in insts ]
        batch_image_loc = [inst["image_loc"] for inst in insts ]
    batch_ids = [inst["ids"] for inst in insts for out in range(outs)]
    batch_size = int(len(batch_src_ids) / outs)
    label = np.array([[0] for i in range(batch_size)], dtype = "int64")

    src_id, self_input_mask, seq_lens = pad_batch_data(
        batch_src_ids, pad_idx=pad_id, return_input_mask=True, return_seq_lens = True)
    pos_id = pad_batch_data(batch_pos_ids, pad_idx=pad_id)
    sent_id = pad_batch_data(batch_sent_ids, pad_idx=pad_id)
    image_embeddings, image_mask = pad_feature_data(batch_image_embedding, return_mask = True)
    image_loc = pad_feature_data(batch_image_loc)
    ids = np.array(batch_ids, dtype = "int64")

    return_list = [
        src_id, pos_id, sent_id, self_input_mask, image_embeddings, image_loc, image_mask, label, ids]
    
    return return_list


def prepare_refcoco_plus_batch_data(insts,
                       total_token_num,
                       task_index,
                       task_num,
                       voc_size=0,
                       pad_id=None,
                       return_input_mask=True,
                       return_max_len=True,
                       return_num_token=False):
    """
    prepare batch data for refcoco_plus tasks
    """
    batch_src_ids = [inst["token_ids"] for inst in insts]
    batch_sent_ids = [inst["sent_ids"] for inst in insts]
    batch_pos_ids = [inst["pos_ids"] for inst in insts]
    batch_image_embedding = [inst["image_embeddings"] for inst in insts]
    batch_image_loc = [inst["image_loc"] for inst in insts]
    batch_image_label = [inst["label"] for inst in insts]
    add_items = np.array([inst["add_item"] for inst in insts], dtype="float32")

    src_id, self_input_mask, seq_lens = pad_batch_data(
        batch_src_ids, pad_idx=pad_id, return_input_mask=True, return_seq_lens = True)
    pos_id = pad_batch_data(batch_pos_ids, pad_idx=pad_id)
    sent_id = pad_batch_data(batch_sent_ids, pad_idx=pad_id)
    image_embedding, image_mask = pad_feature_data(batch_image_embedding, return_mask = True)
    image_loc = pad_feature_data(batch_image_loc)
    image_label = pad_feature_data(batch_image_label)

    return_list = [
        src_id, pos_id, sent_id, self_input_mask, seq_lens,  \
            image_embedding, image_loc, image_mask, image_label, add_items
    ]
    return return_list


def pad_batch_data(insts,
                   pad_idx=0,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, max_len, 1])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] * (max_len - len(inst)) for inst in insts])
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
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape([-1, 1])]

    return return_list if len(return_list) > 1 else return_list[0]


def pad_feature_data(data, pad_value=0.0, dtype="float32", return_mask=False):
    """
    pad visual features with given pad value
    """
    max_length=max([len(item) for item in data])
    data_width = len(data[0][0])
    out_data = np.ones((len(data), max_length, data_width), dtype=dtype) * pad_value
    out_mask = np.zeros((len(data), max_length, 1), dtype=dtype)
    for i in range(len(data)):
        out_data[i, 0: len(data[i]), :] = data[i]
        if return_mask:
            out_mask[i, 0:len(data[i]):] = 1.0
    if return_mask:
        return out_data, out_mask
    else:
        return out_data

if __name__ == "__main__":
    pass
