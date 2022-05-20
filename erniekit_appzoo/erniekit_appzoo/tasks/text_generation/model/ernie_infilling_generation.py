# -*- coding: utf-8 -*
"""
ErnieClassificationPersonal
"""
import logging
import os
from collections import OrderedDict
import math
import paddle
import numpy as np
from paddle import fluid
from paddle.fluid import layers
import re
from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
from erniekit.model.model import BaseModel
from erniekit.modules.ernie_config import ErnieConfig
from erniekit.modules.ernie_gen import ErnieGenModel
from erniekit.utils.multi_process_eval import MultiProcessEval, MultiNodeWriter
from erniekit.utils.util_helper import printable_text
from erniekit.metrics.gen_eval import GenerationEval
from erniekit.modules.ernie_lr import LinearWarmupDecay


@RegisterSet.models.register
class ErnieInfillingGeneration(BaseModel):
    """ErnieInfillingGeneration"""

    def __init__(self, reader, model_params):
        BaseModel.__init__(self, model_params)
        logging.info("ErnieInfillingGeneration init....")
        self.reader = reader
        # args = other_params["args"]
        self.two_stream = self.reader.two_stream
        # self.use_fp16 = args.use_fp16
        self.task_type = self.reader.task_type
        self.label_smooth = self.model_params.get('label_smooth', 0.0)
        # self.loss_scaling = args.init_loss_scaling
        self.max_dec_len = self.reader.max_dec_len
        self.continuous_position = self.reader.continuous_position
        self.tgt_type_id = self.reader.tgt_type_id
        self.beam_size = self.model_params.get('beam_size', 1)
        self.weight_sharing = self.model_params.get('weight_sharing', True)
        self.do_dec = self.reader.do_dec
        self.output_path = './output/'
        self.length_penalty = self.model_params.get('weight_sharing', 0)
        self.use_multi_node_test = False

        self.tokenizer = self.reader.tokenizer
        self.merge_subword = self.tokenizer.merge_subword
        self.vocab = self.tokenizer.vocab
        self.inv_vocab = self.tokenizer.inv_vocab
        self.mask_idx = self.vocab["[MASK]"]
        self.eos_idx = self.vocab["[SEP]"]

        self.gpu_id = self.reader.trainer_id
        self.dev_count = self.reader.dev_count
        self.ernie_config = ErnieConfig(self.model_params.get("embedding").get("config_path"))
        if self.task_type == "dialog":
            self.ernie_config["turn_type_size"] = self.model_params.get('turn_type_size', 2)
            self.ernie_config["role_type_size"] = self.model_params.get('role_type_size', 16)

    def structure(self):
        """网络结构组织
        :return:
        """
        pass

    def _cal_logit(self, enc_out, tgt_pos=None):
        enc_out = fluid.layers.reshape(x=enc_out,
                                       shape=[-1, self.ernie_config["hidden_size"]])
        if tgt_pos:
            tgt_pos = fluid.layers.cast(x=tgt_pos, dtype='int32')
            tgt_feat = fluid.layers.gather(input=enc_out, index=tgt_pos)
        else:
            tgt_feat = enc_out
        tgt_trans_feat = fluid.layers.fc(
            input=tgt_feat,
            size=self.ernie_config["emb_size"] or self.ernie_config["hidden_size"],
            act=self.ernie_config["hidden_act"],
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
                y=fluid.default_main_program().global_block().var("word_embedding"),
                transpose_y=True)
            fc_out += fluid.layers.create_parameter(
                shape=[self.ernie_config['vocab_size']],
                dtype="float32",
                attr=seq2seq_out_bias_attr,
                is_bias=True)
        else:
            out_size = self.ernie_config["tgt_vocab_size"] or self.ernie_config['vocab_size']
            fc_out = fluid.layers.fc(input=tgt_trans_feat,
                                     size=out_size,
                                     param_attr=fluid.ParamAttr(
                                         name="mask_lm_out_fc.w_0",
                                         initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
                                     bias_attr=seq2seq_out_bias_attr)

        return fc_out

    def forward(self, reader, fields_dict, phase):
        """
        create forwrad net
        """
        fields_dict = self.fields_process(fields_dict, phase)
        self.do_dec = reader.do_dec
        if self.do_dec and phase != InstanceName.TRAINING:
            return self.fast_decode(fields_dict, phase)
        masks = fields_dict["masks"]
        tgt_labels, tgt_pos = masks[InstanceName.TGT_LABEL], masks[InstanceName.TGT_POS]
        # get output embedding of model
        emb_dict = self.make_embedding(fields_dict, phase)
        enc_out = emb_dict["enc_out"]
        fc_out = self._cal_logit(enc_out, tgt_pos)
        if phase == InstanceName.TRAINING and self.label_smooth:
            out_size = self.ernie_config["tgt_vocab_size"] or self.ernie_config['vocab_size']
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
        forward_return_dict = {
            InstanceName.LOSS: loss,
            'lr': "learning_rate_0"
        }
        return forward_return_dict

    def fast_decode(self, fields_dict, phase):
        """decode with cache"""
        emb_params = self.model_params.get("embedding")
        use_fp16 = emb_params.get("use_fp16")
        ernie_config = self.ernie_config

        context = fields_dict["context"]
        src_ids, pos_ids = context[InstanceName.SRC_IDS], context[InstanceName.POS_IDS]
        if len(context) == 4:
            sent_ids = context[InstanceName.SENTENCE_IDS]
            role_ids = None
            turn_ids = None
        else:
            sent_ids = None
            role_ids = context[InstanceName.ROLE_IDS]
            turn_ids = context[InstanceName.TURN_IDS]
        input_mask = context[InstanceName.MASK_IDS]
        decode_inputs = fields_dict["decode_inputs"]
        tgt_ids, tgt_pos, init_scores, parent_idx, tgt_input_mask, data_ids = \
            decode_inputs[InstanceName.TGT_SRC_IDS], decode_inputs[InstanceName.TGT_POS_IDS], \
            decode_inputs[InstanceName.INIT_SCORES], decode_inputs[InstanceName.PARENT_IDX], \
            decode_inputs[InstanceName.TGT_MASK_IDS], decode_inputs[InstanceName.DATA_IDS]


        
 
        ernie = ErnieGenModel(
            src_ids=src_ids,
            position_ids=pos_ids,
            sentence_ids=sent_ids,
            role_ids=role_ids,
            turn_ids=turn_ids,
            input_mask=input_mask,
            config=ernie_config,
            use_fp16=use_fp16,
            task_type=self.task_type,
            decoding=True,
            gather_idx=parent_idx,
            )

        max_len = layers.fill_constant(shape=[1], dtype=tgt_ids.dtype,
                                       value=self.max_dec_len, force_cpu=True)
        step_idx = layers.fill_constant(
            shape=[1], dtype=tgt_ids.dtype, value=0, force_cpu=True)
        pos_idx = layers.fill_constant(
            shape=[1], dtype=tgt_ids.dtype, value=1, force_cpu=True)
        length_cond = layers.less_than(x=step_idx, y=max_len)
        finish_cond = layers.fill_constant(shape=[1], dtype="bool", value=1)
        mask_id = layers.fill_constant(
            shape=[1], dtype=tgt_ids.dtype, value=self.mask_idx, force_cpu=True)
        ids = layers.array_write(
            layers.reshape(tgt_ids, (-1, 1)), step_idx)
        pos_biases = layers.array_write(layers.reshape(tgt_pos, (-1, 1)), step_idx)
        scores = layers.array_write(init_scores, step_idx)
        tgt_masks = layers.array_write(tgt_input_mask, step_idx)

        def cond(length_cond, finish_cond):
            """
            x循环条件
            :param length_cond: 长度条件
            :param finish_cond: 结束条件
            :return: 循环条件
            """
            return layers.logical_and(x=length_cond, y=finish_cond)

        def body(length_cond, finish_cond):
            """
            训练执行结构体
            :param length_cond: 长度条件
            :param finish_cond: 结束条件
            :return: 循环条件
            """
            pre_ids = layers.array_read(array=ids, i=step_idx)
            pre_ids = layers.reshape(pre_ids, (-1, 1, 1), inplace=True)
            pre_scores = layers.array_read(array=scores, i=step_idx)
            pos_bias = layers.array_read(array=pos_biases, i=step_idx)
            pos_bias = layers.gather(input=pos_bias, index=parent_idx)

            tmp_mask = layers.array_read(tgt_masks, i=step_idx)
            append_1_mask = layers.fill_constant_batch_size_like(
                input=tmp_mask, value=1.0, shape=[-1, 1, 1], dtype=tmp_mask.dtype)
            append_0_mask = layers.fill_constant_batch_size_like(
                input=tmp_mask, value=0.0, shape=[-1, 1, 1], dtype=tmp_mask.dtype)
            tmp_mask = layers.concat([tmp_mask, append_1_mask], axis=2)
            tmp_tgt_input_mask = layers.concat([tmp_mask, append_0_mask], axis=2)
            tmp_cur_input_mask = layers.concat([tmp_mask, append_1_mask], axis=2)
            tmp_mask = layers.gather(input=tmp_mask, index=parent_idx)
            pre_mask = layers.gather(input=tmp_tgt_input_mask, index=parent_idx)
            cur_mask = layers.gather(input=tmp_cur_input_mask, index=parent_idx)

            cur_ids = layers.elementwise_mul(
                x=layers.fill_constant_batch_size_like(
                    input=pre_mask, value=1, shape=[-1, 1, 1], dtype=pre_ids.dtype),
                y=mask_id, axis=0)

            pre_pos = layers.elementwise_mul(
                x=layers.fill_constant_batch_size_like(
                    input=pre_mask, value=1, shape=[-1, 1, 1], dtype=pre_ids.dtype),
                y=step_idx, axis=0)
            cur_pos = layers.elementwise_mul(
                x=layers.fill_constant_batch_size_like(
                    input=pre_mask, value=1, shape=[-1, 1, 1], dtype=pre_ids.dtype),
                y=pos_idx, axis=0)
            if self.continuous_position:
                unsqueeze_pos_bias = layers.unsqueeze(pos_bias, [1])
                pre_pos = pre_pos + unsqueeze_pos_bias
                cur_pos = cur_pos + unsqueeze_pos_bias

            type_ids = layers.fill_constant_batch_size_like(
                input=pre_mask, value=self.tgt_type_id, shape=[-1, 1, 1], dtype=pre_ids.dtype)
            role_type_ids = layers.fill_constant_batch_size_like(
                input=pre_mask, value=0, shape=[-1, 1, 1], dtype=pre_ids.dtype)
            turn_type_ids = layers.fill_constant_batch_size_like(
                input=pre_mask, value=0, shape=[-1, 1, 1], dtype=pre_ids.dtype)

            dec_ids = layers.concat([pre_ids, cur_ids], axis=1)
            dec_pos = layers.concat([pre_pos, cur_pos], axis=1)
            dec_type = layers.concat([type_ids, type_ids], axis=1)
            dec_role = layers.concat([role_type_ids, role_type_ids], axis=1)
            dec_turn = layers.concat([turn_type_ids, turn_type_ids], axis=1)
            dec_mask = layers.concat([pre_mask, cur_mask], axis=1)
            dec_out = ernie.encode(dec_ids, dec_pos, dec_type, dec_mask,
                                   parent_idx, remove_mask=True, role_ids=dec_role, turn_ids=dec_turn)
            fc_out = self._cal_logit(dec_out[:, 1:, :])
            topk_scores, topk_indices = layers.topk(
                input=layers.softmax(fc_out), k=self.beam_size)
            accu_scores = layers.elementwise_add(
                x=layers.log(topk_scores), y=pre_scores, axis=0)
            topk_indices = layers.lod_reset(topk_indices, pre_ids)
            accu_scores = layers.lod_reset(accu_scores, pre_ids)
            selected_ids, selected_scores, gather_idx = layers.beam_search(
                pre_ids=pre_ids,
                pre_scores=pre_scores,
                ids=topk_indices,
                scores=accu_scores,
                beam_size=self.beam_size,
                end_id=self.eos_idx,
                return_parent_idx=True)

            layers.increment(x=step_idx, value=1.0, in_place=True)
            layers.increment(x=pos_idx, value=1.0, in_place=True)
            layers.array_write(selected_ids, i=step_idx, array=ids)
            layers.array_write(selected_scores, i=step_idx, array=scores)
            layers.array_write(tmp_mask, i=step_idx, array=tgt_masks)
            layers.array_write(pos_bias, i=step_idx, array=pos_biases)
            layers.assign(gather_idx, parent_idx)

            length_cond = layers.less_than(x=step_idx, y=max_len)
            finish_cond = layers.logical_not(layers.is_empty(x=selected_ids))
            return [length_cond, finish_cond]

        _, _ = layers.while_loop(cond, body, [length_cond, finish_cond])

        finished_ids, finished_scores = layers.beam_search_decode(
            ids, scores, beam_size=self.beam_size, end_id=self.eos_idx)

        if phase == InstanceName.SAVE_INFERENCE:
            data_ids = layers.reshape(x=data_ids, shape=[-1])
            target_feed_list = list(context.values()) + list(decode_inputs.values())
            target_feed_name_list = [value.name for value in list(context.values()) + list(decode_inputs.values())]
            target_predict_list = [finished_ids, finished_scores, data_ids]
            forward_return_dict = {
                InstanceName.TARGET_FEED: target_feed_list,
                InstanceName.TARGET_FEED_NAMES: target_feed_name_list,
                InstanceName.TARGET_PREDICTS: target_predict_list
            }
            return forward_return_dict

        forward_return_dict = {
            "finished_ids": finished_ids,
            "finished_scores": finished_scores,
            "data_ids": data_ids
        }

        return forward_return_dict

    def fields_process(self, fields_dict, phase):
        """fields process"""
        ret = {}
        for key in fields_dict.keys():
            ret[key] = fields_dict[key][InstanceName.RECORD_ID]
        return ret

    def make_embedding(self, fields, phase):
        """make embedding"""
        emb_params = self.model_params.get("embedding")
        use_fp16 = emb_params.get("use_fp16")
        ernie_config = self.ernie_config

        context = fields["context"]
        if self.two_stream:
            query = fields["query"]
            src_ids = [context[InstanceName.SRC_IDS], query[InstanceName.SRC_IDS]]
            pos_ids = [context[InstanceName.POS_IDS], query[InstanceName.POS_IDS]]
            if len(context) == 4:
                sent_ids = [context[InstanceName.SENTENCE_IDS], query[InstanceName.SENTENCE_IDS]]
                role_ids = [None, None]
                turn_ids = [None, None]
            else:
                sent_ids = [None, None]
                role_ids = [context[InstanceName.ROLE_IDS], query[InstanceName.ROLE_IDS]]
                turn_ids = [context[InstanceName.TURN_IDS], query[InstanceName.TURN_IDS]]
            input_mask = [context[InstanceName.MASK_IDS], query[InstanceName.MASK_IDS]]
        else:
            src_ids = context[InstanceName.SRC_IDS]
            pos_ids = context[InstanceName.POS_IDS]
            if len(context) == 4:
                sent_ids = context[InstanceName.SENTENCE_IDS]
                role_ids = None
                turn_ids = None
            else:
                sent_ids = None
                role_ids = context[InstanceName.ROLE_IDS]
                turn_ids = context[InstanceName.TURN_IDS]
            input_mask = context[InstanceName.MASK_IDS]

        
        
        ernie = ErnieGenModel(
            src_ids=src_ids,
            position_ids=pos_ids,
            sentence_ids=sent_ids,
            role_ids=role_ids,
            turn_ids=turn_ids,
            input_mask=input_mask,
            config=ernie_config,
            use_fp16=use_fp16,
            task_type=self.task_type,
            two_stream=self.two_stream
            )

        enc_out = ernie.get_sequence_output()
        embedding_dict = {"enc_out": enc_out}
        return embedding_dict

    def _post_process_seq(self, seq):
        """
        Post-process the beam-search decoded sequence. Truncate from the first
        <eos> and remove the <bos> and <eos> tokens currently.
        """
        eos_pos = len(seq)
        for i, idx in enumerate(seq):
            if idx == self.eos_idx:
                eos_pos = i
                break
        seq = seq[1:eos_pos]
        return seq

    def init_metrics(self, meta_info, phase):
        """init output file for metric"""
        self.writer = MultiNodeWriter(self.output_path, self.dev_count, self.gpu_id)
        self.eval = GenerationEval(self.tokenizer)
        if self.do_dec and phase != InstanceName.TRAINING:
            self.writer.init_writer(phase + "_" + str(meta_info.get("stage", "")))

    def add_metric_result(self, forward_output_dict, phase):
        """write batch result"""
        if self.do_dec and phase != InstanceName.TRAINING:
            pred_list = []
            seq_ids = forward_output_dict["finished_ids"]
            seq_scores = forward_output_dict["finished_scores"]
            data_ids = forward_output_dict["data_ids"]
            seq_ids_list, seq_scores_list = [seq_ids], [seq_scores] if isinstance(
                seq_ids, fluid.core.LoDTensor) else (seq_ids, seq_scores)
            data_ids = np.array(data_ids).reshape(-1).tolist()
            data_idx = 0
            for seq_ids, seq_scores in zip(seq_ids_list, seq_scores_list):
                for i in range(len(seq_ids.lod()[0]) - 1):
                    start = seq_ids.lod()[0][i]
                    end = seq_ids.lod()[0][i + 1]
                    max_cand = None
                    for j in range(end - start):  # for each candidate
                        sub_start = seq_ids.lod()[1][start + j]
                        sub_end = seq_ids.lod()[1][start + j + 1]
                        tokens = [self.inv_vocab.get(str(idx), "[UNK]")
                                  for idx in self._post_process_seq(
                                np.array(seq_ids)[sub_start:sub_end])
                                  ]
                        score = np.array(seq_scores)[sub_end - 1]
                        if self.length_penalty > 0:
                            score = score / math.pow((5 + len(tokens)) / 6.0, self.length_penalty)
                        if (not max_cand) or score > max_cand[1]:
                            max_cand = (tokens, score)

                    data_id = data_ids[data_idx]
                    data_idx += 1
                    pred = self.merge_subword(max_cand[0])
                    pred_list.append((data_id, " ".join(pred)))

            self.writer.write_result_list(pred_list)

    def parse_predict_result(self, predict_result, sample_list, params_dict):
        """parse_predict_result
        """
        pred_list = []
        seq_ids, seq_scores, data_ids = predict_result
        batch_seq_ids = seq_ids.tolist()
        def post_process(seq):
            """ post_process """
            all_tokens = []
            prev = 1
            eos_pos = len(seq)
            for i, idx in enumerate(seq):
                if idx == self.eos_idx:
                    eos_pos = i
                    tmp = seq[prev:eos_pos]
                    all_tokens.append([self.inv_vocab.get(str(idx), "[UNK]") 
                            for idx in tmp])
                    prev = i + 2
            return all_tokens

        data_ids = data_ids.reshape(-1).tolist()
        data_idx = 0

        tokens = post_process(batch_seq_ids)
        #  取beam_size的第一个
        for i in range(0, len(tokens), self.beam_size):
            pred_list.append(tokens[i])
        return pred_list

    def get_metrics(self, forward_output_dict, meta_info, phase, reader):
        """get metrics"""

        if phase == InstanceName.TRAINING:
            # add normal log
            loss = np.mean(forward_output_dict[InstanceName.LOSS])
            ppl = np.exp(loss)
            log_info = "step: %d, loss: %.6f, ppl: %.3f, speed: %.3f steps/s"
            values = (meta_info[InstanceName.STEP], loss, ppl, meta_info["speed"])
            logging.info(log_info % values)

        elif phase == InstanceName.EVALUATE or phase == InstanceName.TEST or phase == "predict":
            if self.do_dec:
                if self.use_multi_node_test:
                    self.writer.finalize_writer(remove_sort_key=False)
                    logging.warning("Eval is not supported when use multi node test")
                    if self.gpu_id == 0:
                        eval_res = "Decoding is done "
                else:
                    outfile = self.writer.finalize_writer()
                    if self.gpu_id == 0:
                        eval_res = self.eval.eval(outfile, phase, reader.features[phase])
            else:
                loss_list = []
                loss = forward_output_dict[InstanceName.LOSS]
                for l in loss:
                    loss_list.append(np.mean(l))
                mean_loss = np.mean(np.array(loss_list))
                eval_res = "loss: %.3f, ppl: %.3f" % (mean_loss, np.exp(mean_loss))

            if self.gpu_id == 0:
                log_info = "[%s_%s evaluation] %s, elapsed time: %f s"
                values = (meta_info.get("stage", "infer"), phase, eval_res, meta_info[InstanceName.TIME_COST])
                logging.info(log_info % values)

        return None

    def set_optimizer(self):
        """
        :return: optimizer
        """
        # 学习率和权重的衰减设置在optimizer中，loss的缩放设置在amp中（各个trainer中进行设置）。
        # TODO:需要考虑学习率衰减、权重衰减设置、 loss的缩放设置
        opt_param = self.model_params.get('optimization', None)
        self.lr = opt_param.get("learning_rate", 2e-5)
        weight_decay = opt_param.get("weight_decay", 0.01)
        self.use_default_decay = opt_param.get("use_default_decay", False)
        self.use_lr_decay = opt_param.get("use_lr_decay", False)
        epsilon = opt_param.get("epsilon", 1e-6)

        g_clip = paddle.nn.ClipGradByGlobalNorm(1.0)

        param_name_to_exclue_from_weight_decay = re.compile(r'.*layer_norm_scale|.*layer_norm_bias|.*b_0')

        if self.use_lr_decay:
            max_train_steps = opt_param.get("max_train_steps", 0)
            warmup_steps = opt_param.get("warmup_steps", 0)
            self.lr_scheduler = LinearWarmupDecay(base_lr=self.lr, end_lr=0.0, warmup_steps=warmup_steps,
                                                  decay_steps=max_train_steps, num_train_steps=max_train_steps)

            self.optimizer = paddle.optimizer.AdamW(learning_rate=self.lr_scheduler,
                                                    weight_decay=weight_decay,
                                                    apply_decay_param_fun=lambda
                                                        n: not param_name_to_exclue_from_weight_decay.match(n),
                                                    epsilon=epsilon,
                                                    grad_clip=g_clip)
        else:
            self.optimizer = paddle.optimizer.AdamW(self.lr,
                                                    parameters=self.parameters(),
                                                    weight_decay=weight_decay,
                                                    apply_decay_param_fun=lambda
                                                        n: not param_name_to_exclue_from_weight_decay.match(n),
                                                    epsilon=epsilon,
                                                    grad_clip=g_clip)
        return self.optimizer
