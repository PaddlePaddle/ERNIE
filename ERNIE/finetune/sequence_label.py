#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import multiprocessing

import paddle
import paddle.fluid as fluid

from six.moves import xrange

from model.ernie import ErnieModel


def create_model(args, pyreader_name, ernie_config, is_prediction=False):
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, 1]],
        dtypes=['int64', 'int64', 'int64', 'float32', 'int64', 'int64'],
        lod_levels=[0, 0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)

    (src_ids, sent_ids, pos_ids, input_mask, labels,
     seq_lens) = fluid.layers.read_file(pyreader)

    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        config=ernie_config,
        use_fp16=args.use_fp16)

    enc_out = ernie.get_sequence_output()
    logits = fluid.layers.fc(
        input=enc_out,
        size=args.num_labels,
        num_flatten_dims=2,
        param_attr=fluid.ParamAttr(
            name="cls_seq_label_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_seq_label_out_b",
            initializer=fluid.initializer.Constant(0.)))

    ret_labels = fluid.layers.reshape(x=labels, shape=[-1, 1])
    ret_infers = fluid.layers.reshape(
        x=fluid.layers.argmax(
            logits, axis=2), shape=[-1, 1])

    labels = fluid.layers.flatten(labels, axis=2)
    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=fluid.layers.flatten(
            logits, axis=2),
        label=labels,
        return_softmax=True)
    loss = fluid.layers.mean(x=ce_loss)

    if args.use_fp16 and args.loss_scaling > 1.0:
        loss *= args.loss_scaling

    graph_vars = {
        "loss": loss,
        "probs": probs,
        "labels": ret_labels,
        "infers": ret_infers,
        "seq_lens": seq_lens
    }

    for k, v in graph_vars.items():
        v.persistable = True

    return pyreader, graph_vars


def chunk_eval(np_labels, np_infers, np_lens, tag_num, dev_count=1):
    def extract_bio_chunk(seq):
        chunks = []
        cur_chunk = None
        null_index = tag_num - 1
        for index in xrange(len(seq)):
            tag = seq[index]
            tag_type = tag // 2
            tag_pos = tag % 2

            if tag == null_index:
                if cur_chunk is not None:
                    chunks.append(cur_chunk)
                    cur_chunk = None
                continue

            if tag_pos == 0:
                if cur_chunk is not None:
                    chunks.append(cur_chunk)
                    cur_chunk = {}
                cur_chunk = {"st": index, "en": index + 1, "type": tag_type}

            else:
                if cur_chunk is None:
                    cur_chunk = {"st": index, "en": index + 1, "type": tag_type}
                    continue

                if cur_chunk["type"] == tag_type:
                    cur_chunk["en"] = index + 1
                else:
                    chunks.append(cur_chunk)
                    cur_chunk = {"st": index, "en": index + 1, "type": tag_type}

        if cur_chunk is not None:
            chunks.append(cur_chunk)
        return chunks

    null_index = tag_num - 1
    num_label = 0
    num_infer = 0
    num_correct = 0
    labels = np_labels.reshape([-1]).astype(np.int32).tolist()
    infers = np_infers.reshape([-1]).astype(np.int32).tolist()
    all_lens = np_lens.reshape([dev_count, -1]).astype(np.int32).tolist()

    base_index = 0
    for dev_index in xrange(dev_count):
        lens = all_lens[dev_index]
        max_len = 0
        for l in lens:
            max_len = max(max_len, l)

        for i in xrange(len(lens)):
            seq_st = base_index + i * max_len + 1
            seq_en = seq_st + (lens[i] - 2)
            infer_chunks = extract_bio_chunk(infers[seq_st:seq_en])
            label_chunks = extract_bio_chunk(labels[seq_st:seq_en])
            num_infer += len(infer_chunks)
            num_label += len(label_chunks)

            infer_index = 0
            label_index = 0
            while label_index < len(label_chunks) \
                   and infer_index < len(infer_chunks):
                if infer_chunks[infer_index]["st"] \
                    < label_chunks[label_index]["st"]:
                    infer_index += 1
                elif infer_chunks[infer_index]["st"] \
                    > label_chunks[label_index]["st"]:
                    label_index += 1
                else:
                    if infer_chunks[infer_index]["en"] \
                        == label_chunks[label_index]["en"] \
                        and infer_chunks[infer_index]["type"] \
                        == label_chunks[label_index]["type"]:
                        num_correct += 1

                    infer_index += 1
                    label_index += 1

        base_index += max_len * len(lens)

    return num_label, num_infer, num_correct


def calculate_f1(num_label, num_infer, num_correct):
    if num_infer == 0:
        precision = 0.0
    else:
        precision = num_correct * 1.0 / num_infer

    if num_label == 0:
        recall = 0.0
    else:
        recall = num_correct * 1.0 / num_label

    if num_correct == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate(exe,
             program,
             pyreader,
             graph_vars,
             tag_num,
             eval_phase,
             dev_count=1):
    fetch_list = [
        graph_vars["labels"].name, graph_vars["infers"].name,
        graph_vars["seq_lens"].name
    ]

    if eval_phase == "train":
        fetch_list.append(graph_vars["loss"].name)
        if "learning_rate" in graph_vars:
            fetch_list.append(graph_vars["learning_rate"].name)
        outputs = exe.run(fetch_list=fetch_list)
        np_labels, np_infers, np_lens, np_loss = outputs[:4]
        num_label, num_infer, num_correct = chunk_eval(
            np_labels, np_infers, np_lens, tag_num, dev_count)
        precision, recall, f1 = calculate_f1(num_label, num_infer, num_correct)
        outputs = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "loss": np.mean(np_loss)
        }
        if "learning_rate" in graph_vars:
            outputs["lr"] = float(outputs[4][0])
        return outputs

    else:
        total_label, total_infer, total_correct = 0.0, 0.0, 0.0
        time_begin = time.time()
        pyreader.start()
        while True:
            try:
                np_labels, np_infers, np_lens = exe.run(program=program,
                                                        fetch_list=fetch_list)
                label_num, infer_num, correct_num = chunk_eval(
                    np_labels, np_infers, np_lens, tag_num, dev_count)
                total_infer += infer_num
                total_label += label_num
                total_correct += correct_num

            except fluid.core.EOFException:
                pyreader.reset()
                break

        precision, recall, f1 = calculate_f1(total_label, total_infer,
                                             total_correct)
        time_end = time.time()

        print(
            "[%s evaluation] f1: %f, precision: %f, recall: %f, elapsed time: %f s"
            % (eval_phase, f1, precision, recall, time_end - time_begin))
