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
"""Model for classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import os
import math
import json
import collections
import six
import subprocess

from scipy.stats import pearsonr, spearmanr
from six.moves import xrange
import paddle.fluid as fluid

from model.ernie import ErnieModel
import reader.tokenization as tokenization

if six.PY2:
    import commands as subprocess

def create_model(args, pyreader_name, ernie_config, is_training):
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, 1], [-1, 1], [-1, 1], [-1, args.max_seq_len, args.max_seq_len, 1]],
        dtypes=[
            'int64', 'int64', 'int64', 'int64', 'float32', 'int64', 'int64', 'int64', 'int64'],
        lod_levels=[0, 0, 0, 0, 0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)
    (src_ids, sent_ids, pos_ids, task_ids, input_mask, start_positions,
     end_positions, unique_id, rel_pos_scaler) = fluid.layers.read_file(pyreader)
    checkpoints = []

    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=[pos_ids, rel_pos_scaler],
        sentence_ids=sent_ids,
        task_ids=task_ids,
        input_mask=input_mask,
        config=ernie_config,
        use_fp16=args.use_fp16,
        has_sent_emb=True)
    checkpoints.extend(ernie.get_checkpoints())

    enc_out = ernie.get_sequence_output()
    enc_out = fluid.layers.dropout(
        x=enc_out,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")

    logits = fluid.layers.fc(
        input=enc_out,
        size=2,
        num_flatten_dims=2,
        param_attr=fluid.ParamAttr(
            name="cls_mrc_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_mrc_out_b", initializer=fluid.initializer.Constant(0.)))
    print(enc_out.shape, logits.shape)

    logits = fluid.layers.transpose(x=logits, perm=[2, 0, 1])
    start_logits, end_logits = fluid.layers.unstack(x=logits, axis=0)
    print(logits.shape, start_logits.shape, end_logits.shape)
    #input_mask = fluid.layers.flatten(input_mask, axis=1) 
    #mask_bias = (input_mask - 1.0) * 1e7
    #start_logits += mask_bias
    #end_logits += mask_bias

    batch_ones = fluid.layers.fill_constant_batch_size_like(
        input=start_logits, dtype='int64', shape=[1], value=1)
    print(batch_ones.shape)
    num_seqs = fluid.layers.reduce_sum(input=batch_ones)


    def compute_loss(logits, positions):
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=logits, label=positions)
        loss = fluid.layers.mean(x=loss)
        return loss

    start_loss = compute_loss(start_logits, start_positions)
    end_loss = compute_loss(end_logits, end_positions)
    loss = (start_loss + end_loss) / 2.0
    if args.use_fp16 and args.loss_scaling > 1.0:
        loss *= args.loss_scaling

    graph_vars = {
        "loss": loss,
        "num_seqs": num_seqs,
        "unique_id": unique_id,
        "start_logits": start_logits,
        "end_logits": end_logits,
        "checkpoints": checkpoints
    }

    for k, v in graph_vars.items():
        if k != "checkpoints":
            v.persistable = True

    return pyreader, graph_vars


def write_result(output_path, eval_phase, gpu_id, all_results):
    outfile = output_path + "/" + eval_phase
    outfile_part = outfile + ".part" + str(gpu_id)
    writer = open(outfile_part, "w")
    save_dict = json.dumps(all_results)
    writer.write(save_dict)
    writer.close()
    tmp_writer = open(output_path + "/" + eval_phase + "_dec_finish." + str(gpu_id), "w")
    tmp_writer.close()


def concat_result(output_path, eval_phase, dev_count, RawResult):
    outfile = output_path + "/" + eval_phase
    all_results_read = []
    while True:
        _, ret = subprocess.getstatusoutput('find ' + output_path + \
            ' -maxdepth 1 -name ' + eval_phase + '"_dec_finish.*"')
        ret = ret.split("\n")
        if len(ret) != dev_count:
            time.sleep(1)
            continue

        for dev_cnt in range(dev_count):
            fin_read = open(outfile + ".part" + str(dev_cnt), "rb")
            cur_rawresult = json.loads(fin_read.read())
            for tp in cur_rawresult:
                assert len(tp) == 3
                all_results_read.append(
                    RawResult(
                        unique_id=tp[0],
                        start_logits=tp[1],
                        end_logits=tp[2]))

        subprocess.getstatusoutput("rm " + outfile + ".*part*")
        subprocess.getstatusoutput("rm " + output_path + "/" + eval_phase + "_dec_finish.*")
        break

    return all_results_read


def evaluate(exe, 
             test_program,
             test_pyreader,
             graph_vars,
             eval_phase,
             tag_num=None,
             examples=None,
             features=None,
             args=None,
             use_multi_gpu_test=False,
             gpu_id=0, 
             dev_count=1,
             output_path="./tmpout",
             tokenizer=None,
             version_2_with_negative=False):
    if eval_phase == "train":
        train_fetch_list = [graph_vars["loss"].name]
        if "learning_rate" in graph_vars:
            train_fetch_list.append(graph_vars["learning_rate"].name)
        outputs = exe.run(fetch_list=train_fetch_list)
        ret = {"loss": np.mean(outputs[0])}
        if "learning_rate" in graph_vars:
            ret["learning_rate"] = float(outputs[1][0])
        return ret

    output_dir = output_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_prediction_file = os.path.join(output_dir, eval_phase + "_predictions.json")
    output_nbest_file = os.path.join(output_dir, eval_phase + "_nbest_predictions.json")
    if version_2_with_negative:
        output_null_log_odds_file = os.path.join(output_dir, eval_phase + "_null_odds.json")
    else:
        output_null_log_odds_file = None

    RawResult = collections.namedtuple("RawResult",
            ["unique_id", "start_logits", "end_logits"])

    test_pyreader.start()
    all_results = []
    time_begin = time.time()

    fetch_list = [
        graph_vars["unique_id"].name, graph_vars["start_logits"].name,
        graph_vars["end_logits"].name, graph_vars["num_seqs"].name
    ]
    while True:
        try:
            np_unique_ids, np_start_logits, np_end_logits, np_num_seqs = exe.run(
                program=test_program, fetch_list=fetch_list)
            for idx in range(np_unique_ids.shape[0]):
                if len(all_results) % 1000 == 0:
                    print("Processing example: %d" % len(all_results))
                unique_id = int(np_unique_ids[idx])
                start_logits = [float(x) for x in np_start_logits[idx].flat]
                end_logits = [float(x) for x in np_end_logits[idx].flat]
                all_results.append(
                    RawResult(
                        unique_id=unique_id,
                        start_logits=start_logits,
                        end_logits=end_logits))

        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    
    is_print = True
    if dev_count > 1:
        is_print = False
        write_result(output_dir, eval_phase, gpu_id, all_results)
        if gpu_id == 0:
            is_print = True
            all_results = concat_result(output_dir, eval_phase, dev_count, RawResult)

    if is_print:
        write_predictions(examples, features, all_results,
                      args.n_best_size, args.max_answer_length,
                      args.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, tokenizer, version_2_with_negative)

        if eval_phase.find("dev") != -1:
            data_file = args.dev_set
        elif eval_phase.find("test") != -1:
            data_file = args.test_set
            
        if version_2_with_negative:
            from utils.evaluate_v2 import eval_file
            eval_out = eval_file(data_file, output_prediction_file, output_null_log_odds_file) 
            print(eval_out)
            
            em, f1 = eval_out["exact"], eval_out["f1"]
            print("em: %f, f1: %f, best f1: %f"
                    % (em, f1, eval_out["best_f1"]))

            write_predictions(examples, features, all_results,
                          args.n_best_size, args.max_answer_length,
                          args.do_lower_case, output_prediction_file+"_1",
                          output_nbest_file+"_1", output_null_log_odds_file+"_1", tokenizer, version_2_with_negative, null_score_diff_threshold=eval_out['best_f1_thresh'])
            eval_out = eval_file(data_file, output_prediction_file+"_1", output_null_log_odds_file+"_1")
            print(eval_out)
            em, f1 = eval_out["exact"], eval_out["f1"]
            subprocess.getstatusoutput("rm " + output_dir + "/*")
        else:
            from utils.evaluate_v1 import eval_file
            em, f1 = eval_file(data_file, output_prediction_file) 

        time_end = time.time()
        elapsed_time = time_end - time_begin

        print("[%s evaluation] em: %f, f1: %f, elapsed time: %f"
                % (eval_phase, em, f1,  elapsed_time))


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, tokenizer, version_2_with_negative=True,
                      null_score_diff_threshold=0.0):

    """Write final predictions to the json file and log-odds of null if needed."""
    print("Writing predictions to: %s" % (output_prediction_file))
    print("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", [
            "feature_index", "start_index", "end_index", "start_logit",
            "end_logit"
        ])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        if version_2_with_negative:
            score_null = 1000000  # large and positive
            min_null_feature_index = 0  # the paragraph slice with min mull score
            null_start_logit = 0  # the start logit at the slice with min null score
            null_end_logit = 0  # the end logit at the slice with min null score

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[
                    0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]


            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))


        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1
                                                              )]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end +
                                                                 1)]
                #tok_text = tokenizer.encoder.decode(map(int, tok_tokens))
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                #tok_text = post_process(tok_text, tok_tokens, tokenizer)
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't inlude the empty option in the n-best, inlcude it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))


        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(
                    text="empty", start_logit=0.0, end_logit=0.0))
        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry and version_2_with_negative:
                if entry.text:
                    best_non_null_entry = entry
        # debug
        if best_non_null_entry is None and version_2_with_negative:
            print("Emmm..., sth wrong")


        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            try:
                # predict "" iff the null score - the score of best non-null > threshold
                score_diff = score_null - best_non_null_entry.start_logit - (
                    best_non_null_entry.end_logit)
                scores_diff_json[example.qas_id] = score_diff
                if score_diff > null_score_diff_threshold:
                    all_predictions[example.qas_id] = ""
                else:
                    all_predictions[example.qas_id] = best_non_null_entry.text
            except:
                scores_diff_json[example.qas_id] = 0
                all_predictions[example.qas_id] = ""



        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

def post_process(text, tok_tokens, tokenizer):
    prunc_pair = [[u"[", u"]"], [u"(", u")"], [u"{", u"}"]]
    prunc_pair_dic = {u"[":u"]", u"(":u")", u"{":u"}", u'"':'"', u"'":"'"}
    prunc_pair_flat = sum(prunc_pair, [])

    prunc = [u".", u",", u"%", u"-", u"!", u"?", u"~", u":", u";",u'"', u"'", u"#", u"$", 
                    u"&", u"*", u"/", u"<", u">", u"=", u"\\", u"+", u"_", u"^", u"|"] + prunc_pair_flat
    last_text = tokenizer.encoder.decode(map(int, [tok_tokens[-1]]))
    _last_text = tokenizer.encoder.decode(map(int, tok_tokens[:-1]))
    final_text = []
    start = -1
    for i,c in enumerate(last_text):
        if c in prunc and start == -1:
            start = i
        else:
            final_text.append(c)
            
        if c in prunc_pair_dic.keys() and prunc_pair_dic[c] in _last_text:
            final_text.append(c)
        elif c in prunc and i==start:
            final_text.append(c)
    return _last_text + "".join(final_text) 


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))
    #tok_text = orig_text

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
