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
"""seq2seq generation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import time
import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from model.unimo_finetune import UNIMOModel
from eval.gen_eval import GenerationEval
from finetune.trigram_blocking import TrigramBlocking
import codecs


class Seq2Seq(object):
    """finetuning for seq2seq generation"""

    def __init__(self, args, gene_config, tokenizer):
        self.gene_config = gene_config
        self.weight_sharing = args.weight_sharing
        self.task_type = args.task_type
        self.max_seq_len = args.max_seq_len
        self.label_smooth = args.label_smooth
        self.tgt_type_id = args.tgt_type_id
        self.continuous_position = args.continuous_position
        self.tokenizer = tokenizer
        self.vocab_size = gene_config["vocab_size"]

        self._emb_dtype = "float32"

        # for beam_search decoding
        self.do_decode = args.do_decode
        self.length_penalty = args.length_penalty
        self.max_out_len = args.max_out_len
        self.min_out_len = args.min_out_len
        self.block_trigram = args.block_trigram
        self.beam_size = args.beam_size

        self.bos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.mask_token_id
        self.evaluator = GenerationEval(args)
        if self.task_type == "dialog":
            self.emb_keys = ["word_embedding", "role_embedding", "turn_embedding", "pos_embedding"]
        else:
            self.emb_keys = ["word_embedding", "sent_embedding", "pos_embedding"]

    def cal_logit(self, enc_out, tgt_pos):
        """calculate logit"""
        enc_out = fluid.layers.reshape(x=enc_out,
                                       shape=[-1, self.gene_config["hidden_size"]])
        if tgt_pos:
            tgt_pos = fluid.layers.cast(x=tgt_pos, dtype='int32')
            tgt_feat = fluid.layers.gather(input=enc_out, index=tgt_pos)
        else:
            tgt_feat = enc_out

        tgt_trans_feat = fluid.layers.fc(
            input=tgt_feat,
            size=self.gene_config["emb_size"] or self.gene_config["hidden_size"],
            act=self.gene_config["hidden_act"],
            param_attr=fluid.ParamAttr(
                name="mask_lm_trans_fc.w_0",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="mask_lm_trans_fc.b_0",
                initializer=fluid.initializer.Constant(0.)))

        tgt_trans_feat = fluid.layers.layer_norm(
            tgt_trans_feat,
            begin_norm_axis=len(tgt_trans_feat.shape) - 1,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_layer_norm_scale',
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                name='mask_lm_trans_layer_norm_bias',
                initializer=fluid.initializer.Constant(1.)))

        seq2seq_out_bias_attr = fluid.ParamAttr(
            name="mask_lm_out_fc.b_0",
            initializer=fluid.initializer.Constant(value=0.0))

        if self.weight_sharing:
            fc_out = fluid.layers.matmul(
                x=tgt_trans_feat,
                y=fluid.default_main_program().global_block().var(
                    "word_embedding"),
                transpose_y=True)
            fc_out += fluid.layers.create_parameter(
                shape=[self.gene_config['vocab_size']],
                dtype="float32",
                attr=seq2seq_out_bias_attr,
                is_bias=True)
        else:
            out_size = self.gene_config["tgt_vocab_size"] or self.gene_config['vocab_size']
            fc_out = fluid.layers.fc(input=tgt_trans_feat,
                                     size=out_size,
                                     param_attr=fluid.ParamAttr(
                                         name="mask_lm_out_fc.w_0",
                                         initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
                                     bias_attr=seq2seq_out_bias_attr)

        return fc_out

    def to_tensor(self, shapes, dtypes, lod_levels):
        """convert to tensor"""
        return [fluid.layers.data(name="placeholder_" + str(i), shape=shapes[i], dtype=dtypes[i],
                                  lod_level=lod_levels[i]) for i in range(len(shapes))]

    def create_model(self, decoding=False):
        """create model for training"""
        if decoding:
            return self.fast_decode()

        if self.task_type == "dialog":
            emb_num = 4
        else:
            emb_num = 3
        input_shapes = [[-1, self.max_seq_len, 1]] * emb_num + \
                       [[-1, self.max_seq_len, self.max_seq_len]]
        input_dtypes = ['int64'] * emb_num + ['float32']
        input_lod_levels = [0] * emb_num + [0]
        shapes = input_shapes + [[-1, 1], [-1, 1]]
        dtypes = input_dtypes + ['int64', 'int64']
        lod_levels = input_lod_levels + [0, 0]

        inputs = self.to_tensor(shapes, dtypes, lod_levels)
        pyreader = fluid.io.DataLoader.from_generator(feed_list=inputs, capacity=70, iterable=False)

        emb_ids = {}
        for key, value in zip(self.emb_keys, inputs[:emb_num]):
            emb_ids[key] = value  # for embeddings

        # src_ids, sent_ids, pos_ids = inputs[:emb_num]
        input_mask = inputs[emb_num]
        tgt_labels, tgt_pos = inputs[-2:]

        unimo = UNIMOModel(
            emb_ids=emb_ids,
            input_mask=input_mask,
            config=self.gene_config,
            task_type=self.task_type)

        enc_out = unimo.get_sequence_output()
        fc_out = self.cal_logit(enc_out, tgt_pos)

        if self.label_smooth:
            out_size = self.gene_config['vocab_size']
            labels = fluid.layers.label_smooth(
                label=fluid.layers.one_hot(
                    input=tgt_labels, depth=out_size),
                epsilon=self.label_smooth)

            ce_loss = layers.softmax_with_cross_entropy(
                logits=fc_out, label=labels, soft_label=True)
        else:
            ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
                logits=fc_out, label=tgt_labels, return_softmax=True)

        loss = fluid.layers.mean(x=ce_loss)
        graph_vars = {"loss": loss}
        for k, v in graph_vars.items():
            v.persistable = True

        return pyreader, graph_vars

    def fast_decode(self):
        """create model for inference"""
        if self.task_type == "dialog":
            emb_num = 4
        else:
            emb_num = 3
        input_shapes = [[-1, self.max_seq_len, 1]] * emb_num + \
                       [[-1, self.max_seq_len, self.max_seq_len]]
        input_dtypes = ['int64'] * emb_num + ['float32']
        input_lod_levels = [0] * emb_num + [0]

        shapes = input_shapes + [[-1, 1, 1], [-1, 1, 1],
                                 [-1, 1], [-1], [-1, 1, self.max_seq_len], [-1, 1]]
        dtypes = input_dtypes + ['int64', 'int64', 'float32', 'int32', 'float32', 'int64']
        lod_levels = input_lod_levels + [2, 2, 2, 0, 0, 0]

        inputs = self.to_tensor(shapes, dtypes, lod_levels)
        pyreader = fluid.io.DataLoader.from_generator(feed_list=inputs, capacity=70, iterable=False)
        emb_ids = {}
        for key, value in zip(self.emb_keys, inputs[:emb_num]):
            emb_ids[key] = value

        input_mask = inputs[emb_num]
        tgt_ids, tgt_pos, init_scores, parent_idx, tgt_input_mask, data_ids = inputs[-6:]

        unimo = UNIMOModel(
            emb_ids=emb_ids,
            input_mask=input_mask,
            config=self.gene_config,
            task_type=self.task_type,
            decoding=True,
            gather_idx=parent_idx)

        max_len = layers.fill_constant(
            shape=[1], dtype=tgt_ids.dtype, value=self.max_out_len, force_cpu=True)
        min_len = layers.fill_constant(
            shape=[1], dtype=tgt_ids.dtype, value=self.min_out_len, force_cpu=True)
        neg_inf = layers.fill_constant(
            shape=[1], dtype='float32', value=-1e18)
        step_idx = layers.fill_constant(
            shape=[1], dtype=tgt_ids.dtype, value=0, force_cpu=True)
        step_next_idx = layers.fill_constant(
            shape=[1], dtype=tgt_ids.dtype, value=1, force_cpu=True)
        cond = layers.less_than(x=step_idx, y=max_len)
        while_op = layers.While(cond)

        ids = layers.array_write(layers.reshape(tgt_ids, (-1, 1)), step_idx)
        pos_biases = layers.array_write(tgt_pos, step_idx)
        scores = layers.array_write(init_scores, step_idx)
        tgt_masks = layers.array_write(tgt_input_mask, step_idx)

        trigram_blocking = TrigramBlocking(tgt_ids, self.tokenizer, beam_size=self.beam_size)

        with while_op.block():
            pre_ids = layers.array_read(array=ids, i=step_idx)
            pre_ids = layers.reshape(pre_ids, (-1, 1, 1), inplace=True)
            pre_scores = layers.array_read(array=scores, i=step_idx)
            pos_bias = layers.array_read(array=pos_biases, i=step_idx)
            pos_bias = layers.gather(input=pos_bias, index=parent_idx)

            def gen_batch_like(value, dtype="int64", shape=[-1, 1, 1], is_scalar=True):
                """generate batch"""
                if is_scalar:
                    return layers.fill_constant_batch_size_like(
                        input=parent_idx, value=value, shape=shape, dtype=dtype)
                else:
                    return layers.elementwise_mul(
                        x=layers.fill_constant_batch_size_like(
                            input=parent_idx, value=1, shape=shape, dtype=dtype),
                        y=value, axis=0)

            tmp_mask = layers.array_read(tgt_masks, i=step_idx)
            tmp_mask = layers.gather(input=tmp_mask, index=parent_idx)
            append_1_mask = gen_batch_like(1.0, dtype=tmp_mask.dtype)
            pre_mask = layers.concat([tmp_mask, append_1_mask], axis=2)

            pre_pos = gen_batch_like(step_idx, is_scalar=False)
            pre_pos = pre_pos + pos_bias  ####################### pos start from 2

            pre_sent = gen_batch_like(self.tgt_type_id, dtype=pre_ids.dtype)

            dec_emb_ids = {"word_embedding": pre_ids, "pos_embedding": pre_pos}
            if self.task_type == "dialog":
                role_ids = gen_batch_like(0)
                turn_ids = gen_batch_like(0)
                dec_emb_ids["role_embedding"] = role_ids
                dec_emb_ids["turn_embedding"] = turn_ids
            else:
                dec_emb_ids["sent_embedding"] = pre_sent

            dec_out = unimo.encode(emb_ids=dec_emb_ids,
                                   input_mask=pre_mask,
                                   gather_idx=parent_idx)
            fc_out = self.cal_logit(dec_out, None)

            # prevent generating end token if length less than min_out_len
            eos_index = layers.fill_constant(shape=[layers.shape(fc_out)[0]],
                                             dtype='int64',
                                             value=self.eos_id)
            eos_index = fluid.one_hot(eos_index, depth=self.vocab_size)
            less_cond = layers.cast(layers.less_than(x=step_idx, y=min_len), dtype='float32')
            less_val = layers.elementwise_mul(less_cond, neg_inf)
            eos_val = layers.elementwise_mul(eos_index, less_val, axis=0)
            revised_logits = layers.elementwise_add(fc_out, eos_val, axis=0)

            # topK reduction across beams, also contain special handle of
            # end beams and end sentences(batch reduction)
            topk_scores, topk_indices = layers.topk(
                input=layers.softmax(revised_logits), k=self.beam_size)

            # Roll-Back previous-scores for length-penalty
            # previous-scores has been length-penaltied, before this timestep length-penalty, need roll-back
            # because of doing this, we need store the length-penaltied score in `scores`
            # while calculating use the un-penaltied score
            # -> safe for step_idx == 0 (initialization state), because previous-score == 0
            pre_timestep_length_penalty = fluid.layers.pow(
                ((5.0 + fluid.layers.cast(step_idx, pre_scores.dtype)) / 6.0), self.length_penalty)
            pre_scores_wo_len_penalty = fluid.layers.elementwise_mul(pre_scores, pre_timestep_length_penalty)

            # calc trigram-blocking delta scores for current alive sequence
            if self.block_trigram:
                trigram_blocking.update_seq(pre_ids, parent_idx)
                trigram_blocking.expand_cand_seq(topk_indices)
                fluid.layers.py_func(func=trigram_blocking.blocking_forward,
                                     x=[trigram_blocking.cand_seq,
                                        trigram_blocking.id2is_full_token],
                                     out=trigram_blocking.delta_score_out,
                                     backward_func=None)
                pre_scores_wo_len_penalty = fluid.layers.elementwise_add(x=trigram_blocking.delta_score_out,
                                                                         y=pre_scores_wo_len_penalty,
                                                                         axis=0)
            # => [N, topk]
            accu_scores = layers.elementwise_add(
                x=layers.log(topk_scores), y=pre_scores_wo_len_penalty, axis=0)

            cur_timestep_length_penalty = layers.pow(((5.0 + layers.cast(step_next_idx, accu_scores.dtype)) / 6.0),
                                                     self.length_penalty)
            curr_scores = layers.elementwise_div(accu_scores, cur_timestep_length_penalty)

            # beam_search op uses lod to differentiate branches.
            curr_scores = layers.lod_reset(curr_scores, pre_ids)
            topk_indices = layers.lod_reset(topk_indices, pre_ids)
            selected_ids, selected_scores, gather_idx = layers.beam_search(
                pre_ids=pre_ids,
                pre_scores=pre_scores,
                ids=topk_indices,
                scores=curr_scores,
                beam_size=self.beam_size,
                end_id=self.eos_id,
                return_parent_idx=True)

            layers.increment(x=step_idx, value=1.0, in_place=True)
            layers.increment(x=step_next_idx, value=1.0, in_place=True)
            # cell states(caches) have been updated in wrap_decoder,
            # only need to update beam search states here.
            layers.array_write(selected_ids, i=step_idx, array=ids)
            layers.array_write(selected_scores, i=step_idx, array=scores)
            layers.array_write(pre_mask, i=step_idx, array=tgt_masks)
            layers.array_write(pos_bias, i=step_idx, array=pos_biases)
            layers.assign(gather_idx, parent_idx)

            length_cond = layers.less_than(x=step_idx, y=max_len)
            finish_cond = layers.logical_not(layers.is_empty(x=selected_ids))
            layers.logical_and(x=length_cond, y=finish_cond, out=cond)

        finished_ids, finished_scores = layers.beam_search_decode(
            ids, scores, beam_size=self.beam_size, end_id=self.eos_id)

        graph_vars = {
            "finished_ids": finished_ids,
            "finished_scores": finished_scores,
            "data_ids": data_ids
        }

        for k, v in graph_vars.items():
            v.persistable = True

        return pyreader, graph_vars

    def post_process_seq(self, seq):
        """
        Post-process the beam-search decoded sequence. Truncate from the first
        <eos> and remove the <bos> and <eos> tokens currently.
        """
        eos_pos = len(seq)
        for i, idx in enumerate(seq):
            if idx == self.eos_id:
                eos_pos = i
                break
        seq = seq[1:eos_pos]
        return seq

    def remove_special_tokens(self, seq, special_tokens):
        """Remove special tokens from output sequence"""
        seq = [idx for idx in seq if idx not in special_tokens]
        return seq

    def evaluate(self, resource, eval_phase, graph_vars, features=None,
                 output_path=None, dev_count=1, gpu_id=0):
        """evaluate model"""
        exe, program, pyreader = resource["exe"], resource["program"], resource["pyreader"]

        if eval_phase == "train":
            fetch_list = [graph_vars["loss"].name]
            if "learning_rate" in graph_vars:
                fetch_list.append(graph_vars["learning_rate"].name)
            outputs = exe.run(fetch_list=fetch_list)
            np_loss = outputs[0]
            ret = {"loss": np.mean(np_loss), "ppl": np.exp(np.mean(np_loss))}
            if "learning_rate" in graph_vars:
                ret["learning_rate"] = float(outputs[1][0])
            return ret

        if self.do_decode:
            return_numpy = False
            outfile = output_path + "/" + eval_phase
            outfile_part = outfile + ".part" + str(gpu_id)
            # writer = open(outfile_part, "w", encoding='utf-8')
            writer = codecs.open(outfile_part, 'w', encoding='utf-8')
            fetch_keys = ["finished_ids", "finished_scores", "data_ids"]
            special_tokens = [self.tokenizer.cls_token_id,
                              self.tokenizer.mask_token_id,
                              self.tokenizer.pad_token_id,
                              self.tokenizer.unk_token_id]
        else:
            steps = 0
            cost = 0.0
            return_numpy = True
            fetch_keys = ["loss"]

        fetch_list = [graph_vars[key].name for key in fetch_keys]

        time_begin = time.time()
        pyreader.start()
        while True:
            try:
                outputs = exe.run(program=program,
                                  fetch_list=fetch_list,
                                  return_numpy=return_numpy)
                if not self.do_decode:
                    np_loss = outputs[0]
                    cost += np.mean(np_loss)
                    steps += 1
                else:
                    seq_ids, seq_scores, data_ids = outputs
                    seq_ids_list, seq_scores_list = [seq_ids], [seq_scores] \
                        if isinstance(seq_ids, paddle.fluid.core.LoDTensor) else (seq_ids, seq_scores)

                    data_ids = np.array(data_ids).reshape(-1).tolist()
                    data_idx = 0

                    for seq_ids, seq_scores in zip(seq_ids_list, seq_scores_list):
                        # How to parse the results:
                        #   Suppose the lod of seq_ids is:
                        #     [[0, 3, 6], [0, 12, 24, 40, 54, 67, 82]]
                        #   then from lod[0]:
                        #     there are 2 source sentences, beam width is 3.
                        #   from lod[1]:
                        #     the first source sentence has 3 hyps; the lengths are 12, 12, 16
                        #     the second source sentence has 3 hyps; the lengths are 14, 13, 15
                        # hyps = [[] for i in range(len(seq_ids.lod()[0]) - 1)]
                        # scores = [[] for i in range(len(seq_scores.lod()[0]) - 1)]
                        for i in range(len(seq_ids.lod()[0]) - 1):  # for each source sentence
                            start = seq_ids.lod()[0][i]
                            end = seq_ids.lod()[0][i + 1]
                            max_cand = None
                            for j in range(end - start):  # for each candidate
                                sub_start = seq_ids.lod()[1][start + j]
                                sub_end = seq_ids.lod()[1][start + j + 1]
                                token_ids = [int(idx) for idx in self.post_process_seq(
                                    np.array(seq_ids)[sub_start:sub_end])]
                                # print(len(token_ids))

                                hyp_ids = self.remove_special_tokens(token_ids, special_tokens)
                                hyp_tokens = self.tokenizer.convert_ids_to_tokens(hyp_ids)
                                hyp_str = self.tokenizer.gptbpe_tokenizer.decode(hyp_tokens)
                                hyp_str = re.sub('\\s+', ' ', hyp_str)
                                # print(hyp_str)

                                score = np.array(seq_scores)[sub_end - 1]
                                if (not max_cand) or score > max_cand[1]:
                                    max_cand = (hyp_str, score)

                            data_id = data_ids[data_idx]
                            data_idx += 1
                            pred = max_cand[0]
                            writer.write("%d\t%s\n" % (data_id, pred))

            except fluid.core.EOFException:
                pyreader.reset()
                break

        time_end = time.time()
        if not self.do_decode:
            eval_result = "loss: %f, ppl: %f" % (cost / steps, np.exp(cost / steps))
            print("[%s evaluation] %s, elapsed time: %f s"
                  % (eval_phase, eval_result, time_end - time_begin))
        else:
            writer.close()
            # tmp_writer = open("%s/%s_dec_finish.%d" % (output_path, eval_phase, gpu_id), "w")
            tmp_writer = codecs.open("%s/%s_dec_finish.%d" % (output_path, eval_phase, gpu_id),
                                     'w', encoding='utf-8')
            tmp_writer.close()
            if gpu_id != 0:
                return

            while True:
                ret = os.popen('find %s -maxdepth 1 -name "%s_dec_finish.*"' %
                               (output_path, eval_phase)).readlines()
                if len(ret) != dev_count:
                    time.sleep(1)
                    continue
                else:
                    break

            os.system("sort -t '\t' -k 1 -n %s.part* | awk -F '\t' '{print $2}' > %s" % (outfile, outfile))
            os.system("rm %s.part*" % outfile)
            os.system("rm %s/%s_dec_finish.*" % (output_path, eval_phase))

            eval_result = self.evaluator.eval(outfile,
                                              phase=eval_phase.split("_")[0], features=features)
            print("[%s evaluation] %s, elapsed time: %f s"
                  % (eval_phase, eval_result, time_end - time_begin))
