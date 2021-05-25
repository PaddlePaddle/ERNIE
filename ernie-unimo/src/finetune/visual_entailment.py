#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Model for visual_entailment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import time
import numpy as np

import paddle.fluid as fluid
from model.unimo_finetune import UNIMOModel
from eval import glue_eval
from collections import OrderedDict
from utils.utils import print_eval_log


def kl_divergence_with_logits(q_logits, p_logits):
    """
    symmetric KL-divergence (See SMART, Sec 3.1)
    q_logits: logits
    p_logits: delta_logits
    """
    q = fluid.layers.softmax(input=q_logits)
    p = fluid.layers.softmax(input=p_logits)
    kl_qp = fluid.layers.reduce_sum(q * (fluid.layers.log(q) - fluid.layers.log(p)), -1)
    kl_pq = fluid.layers.reduce_sum(p * (fluid.layers.log(p) - fluid.layers.log(q)), -1)
    vat_loss = fluid.layers.mean(x=kl_qp+kl_pq)
    return vat_loss


def create_model(args, config, pyreader_name="train_reader", is_train=True):
    """create_model"""
    shapes = [[-1, args.max_seq_len, 1],  # src_ids
              [-1, args.max_seq_len, 1],  # pos_ids
              [-1, args.max_seq_len, 1],  # sent_ids
              [-1, args.max_img_len + args.max_seq_len, args.max_img_len + args.max_seq_len],  # input_mask
              [-1, args.max_img_len, 1],  # v_mask
              [-1, args.max_seq_len, 1],  # t_mask
              [-1, args.max_img_len, config["image_embedding_size"]],  # image_embedding
              [-1, args.max_img_len, 5],  # image_loc
              [-1, 1]  # labels
              ]

    dtypes = ['int64', 'int64', 'int64', 'float32', 'float32', 'float32', 'float32','float32', 'int64']
    lod_levels = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    pyreader = fluid.layers.py_reader(
        capacity=70,
        shapes=shapes,
        dtypes=dtypes,
        lod_levels=lod_levels,
        name=pyreader_name,
        use_double_buffer=True)

    (src_ids, pos_ids, sent_ids, input_mask, v_mask, t_mask, image_embedding, image_loc, labels) \
        = fluid.layers.read_file(pyreader)

    emb_ids = {"word_embedding": src_ids, "sent_embedding": sent_ids, "pos_embedding": pos_ids}
    image_input = {"image_embedding": image_embedding, "loc_embedding": image_loc}
    
    adv_step, adv_lr, norm_type, adv_max_norm, adv_init_mag = \
            args.adv_step, args.adv_lr, args.norm_type, args.adv_max_norm, args.adv_init_mag
    assert adv_step > 0 and adv_init_mag > 0

    def get_loss_and_logits(text_feats, image_feats):
        feats = text_feats + image_feats
        cls_params_name = ["cls_out_w_0", "cls_out_b_0"]
        feats = fluid.layers.fc(
            input=feats,
            size=2048,
            param_attr=fluid.ParamAttr(
                name=cls_params_name[0],
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name=cls_params_name[1], initializer=fluid.initializer.Constant(0.)))
        feats = fluid.layers.dropout(
            x=feats,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")
        cls_params_name = ["cls_out_w_1", "cls_out_b_1"]
        logits = fluid.layers.fc(
            input=feats,
            size=args.num_labels,
            param_attr=fluid.ParamAttr(
                name=cls_params_name[0],
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name=cls_params_name[1], initializer=fluid.initializer.Constant(0.)))
        ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
            logits=logits, label=labels, return_softmax=True)
        loss = fluid.layers.mean(x=ce_loss) / adv_step
        return loss, logits, probs

    def init_delta(input, mask, shape, name='text'):
        real_seq_len = fluid.layers.shape(input)[1]

        fake = fluid.layers.data(name=name+"_fake", shape=shape, dtype='float32')
        mask_slice = fluid.layers.slice(mask, axes=[1], starts=[0], ends=fluid.layers.shape(mask)[1])
        length = fluid.layers.reduce_sum(mask_slice, dim=1, keep_dim=True) * shape[-1]
        
        # l2 norm
        delta = fluid.layers.uniform_random_batch_size_like(mask, shape=fake.shape, min=-1.0, max=1.0)
        delta = fluid.layers.slice(delta, axes=[1], starts=[0], ends=real_seq_len)
        delta = delta * mask_slice
        mag = adv_init_mag / fluid.layers.sqrt(length)
        delta = delta * mag
        return delta
    
    if is_train:
        text_emb_shape = [-1, args.max_seq_len, config['hidden_size']]
        text_delta = init_delta(src_ids, t_mask, text_emb_shape, name='text')
        image_emb_shape = [-1, args.max_img_len, config['image_embedding_size']]
        image_delta = init_delta(image_embedding, v_mask, image_emb_shape, name='img')
    else:
        text_delta, image_delta = None, None

    def pgd_with_l2(loss, delta):
        # grad
        delta_grad = fluid.backward.gradients(loss, delta)[0]

        # l2 norm
        delta_norm = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.pow(fluid.layers.reshape(delta_grad, \
                [fluid.layers.shape(delta_grad)[0], -1]), factor=2), dim=1, keep_dim=True))
        delta_norm = fluid.layers.clamp(delta_norm, min=float(1e-8))

        # pgd
        delta = delta + adv_lr * delta_grad / delta_norm

        # projection
        if adv_max_norm > 0:
            exceed_mask = (delta_norm > adv_max_norm).astype('float32')
            reweights = (adv_max_norm / delta_norm) * exceed_mask + (1 - exceed_mask)
            delta = delta * reweights
        delta_grad.stop_gradient=True
        return delta
    
    loss = None
    for iter in range(adv_step):
        vl_pure = UNIMOModel(
            emb_ids=emb_ids,
            input_mask=input_mask,
            config=config,
            image_input=image_input,
            weight_sharing=args.weight_sharing
        )
        vl_text = UNIMOModel(
            text_adv_delta=text_delta,
            emb_ids=emb_ids,
            input_mask=input_mask,
            config=config,
            image_input=image_input,
            weight_sharing=args.weight_sharing
        )
        vl_image = UNIMOModel(
            image_adv_delta=image_delta,
            emb_ids=emb_ids,
            input_mask=input_mask,
            config=config,
            image_input=image_input,
            weight_sharing=args.weight_sharing
        )

        h_pure_text, h_pure_image = vl_pure.get_pooled_output()
        h_text_text, h_text_image = vl_text.get_pooled_output()
        h_image_text, h_image_image = vl_image.get_pooled_output()

        loss_pure, logit_pure, probs_pure = get_loss_and_logits(h_pure_text, h_pure_image)
        loss_text, logit_text, probs_text = get_loss_and_logits(h_text_text, h_text_image)
        loss_image, logit_image, probs_image = get_loss_and_logits(h_image_text, h_image_image)
        
        if is_train:
            text_delta = pgd_with_l2(loss_text, text_delta)
            image_delta = pgd_with_l2(loss_image, image_delta)

        kl_adv_text_loss = kl_divergence_with_logits(logit_pure, logit_text)
        kl_adv_image_loss = kl_divergence_with_logits(logit_pure, logit_image)
        cur_loss = loss_pure + loss_text + loss_image + kl_adv_text_loss + kl_adv_image_loss
        loss = cur_loss if loss is None else loss + cur_loss

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs_pure, label=labels, total=num_seqs)

    graph_vars = {
        "loss": loss,
        "probs": probs_pure,
        "accuracy": accuracy,
        "labels": labels,
        "num_seqs": num_seqs
    }

    for k, v in graph_vars.items():
        v.persistable = False

    return pyreader, graph_vars


def evaluate(args, exe, test_pyreader, graph_vars, eval_phase, dev_count=1, gpu_id=0):
    """evaluate"""
    all_mat = []

    test_pyreader.start()
    time_begin = time.time()
    fetch_list = [graph_vars["probs"].name, graph_vars["labels"].name]

    while True:
        try:
            np_probs, np_labels = exe.run(fetch_list=fetch_list)
            np_preds = np.argmax(np_probs, axis=1).reshape((-1, 1))
            np_labels = np_labels.reshape((-1, 1))
            mat = np.concatenate([np_preds, np_labels], axis=1)
            all_mat.extend(mat.tolist())
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    all_mat = np.array(all_mat)
    time_end = time.time()

    save_file = "%s/%s.trainers_%d.part_%d.npy" % (args.eval_dir, eval_phase, dev_count, gpu_id)
    np.save(save_file, all_mat)

    tmp_file = "%s/%s.trainers_%d.part_%d.finish" % (args.eval_dir, eval_phase, dev_count, gpu_id)
    tmp_writer = open(tmp_file, "w")
    tmp_writer.close()

    if gpu_id == 0:
        while True:
            ret = os.popen('find %s -maxdepth 1 -name "%s.trainers_%d.part_*.finish"' %
                           (args.eval_dir, eval_phase, dev_count)).readlines()
            if len(ret) != dev_count:
                time.sleep(1)
                continue
            else:
                break

        all_mats = []
        save_files = glob.glob("%s/%s.trainers_%d.part_*.npy" % (args.eval_dir, eval_phase, dev_count))
        for cur_save_file in save_files:
            mat = np.load(cur_save_file).tolist()
            all_mats.extend(mat)
        all_mats = np.array(all_mats)

        cur_time = str(int(time.time()))
        os.system("mkdir %s/%s" % (args.eval_dir, cur_time))
        os.system("mv %s/%s.trainers_%d.* %s/%s" % (args.eval_dir, eval_phase, dev_count, args.eval_dir, cur_time))

        ret = OrderedDict()
        ret['phase'] = eval_phase
        ret['loss'] = -1
        ret['data_num'] = all_mats.shape[0]
        ret['used_time'] = round(time_end - time_begin, 4)

        metrics = OrderedDict()
        metrics["simple_accuracy"] = glue_eval.simple_accuracy

        if args.eval_mertrics in metrics:
            ret_metric = metrics[args.eval_mertrics](all_mats[:, 0], all_mats[:, 1])
            ret.update(ret_metric)
            print_eval_log(ret)
        else:
            raise ValueError('unsupported metric {}'.format(args.eval_mertrics))

        return ret
    else:
        return None

