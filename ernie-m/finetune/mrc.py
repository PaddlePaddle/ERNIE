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
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import six
import time
import math
import json
import logging
import subprocess
import collections 

from io import open

import numpy as np
import paddle.fluid as fluid

import reader.tokenization
from model.ernie import ErnieModel
from utils.eval_mlqa import mlqa_eval

log = logging.getLogger(__name__)

def create_model(args, ernie_config):
    src_ids = fluid.layers.data(name="src_ids", shape=[-1, args.max_seq_len, 1], dtype="int64")
    pos_ids = fluid.layers.data(name="pos_ids", shape=[-1, args.max_seq_len, 1], dtype="int64")
    input_mask = fluid.layers.data(name="input_mask", shape=[-1, args.max_seq_len, 1], dtype="float32")
    start_positions = fluid.layers.data(name="start_positions", shape=[-1, 1], dtype="int64")
    end_positions = fluid.layers.data(name="end_positions", shape=[-1, 1], dtype="int64")
    unique_id = fluid.layers.data(name="unique_id", shape=[-1, 1], dtype="int64")
    labels = fluid.layers.data(name="labels", shape=[-1, 1], dtype="int64")

    pyreader = fluid.io.DataLoader.from_generator(
         feed_list=[src_ids, pos_ids, input_mask, 
             start_positions, end_positions, unique_id, labels],
         capacity=70,
         iterable=False)

    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        input_mask=input_mask,
        config=ernie_config,
        use_fp16=args.use_fp16)

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

    logits = fluid.layers.transpose(x=logits, perm=[2, 0, 1])
    start_logits, end_logits = fluid.layers.unstack(x=logits, axis=0)

    batch_ones = fluid.layers.fill_constant_batch_size_like(
        input=start_logits, dtype='int64', shape=[1], value=1)
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
        "end_logits": end_logits
    }

    for k, v in graph_vars.items():
        v.persistable = True

    return pyreader, graph_vars


def write_result(output_path, eval_phase, trainer_id, results):
    output_file = os.path.join(output_path, 
                      "%s.output.%d" % (eval_phase, trainer_id))
    with open(output_file, "w") as fp:
         json.dump(results, fp)
    
    finish_file = os.path.join(output_path, 
                      "%s.finish.%d" % (eval_phase, trainer_id))
    with open(finish_file, "w") as fp:
        pass


def concat_result(output_path, eval_phase, dev_count, RawResult):
    all_results = []
    while True:
        ret = subprocess.check_output('find %s -maxdepth 3 -name "%s.finish.*"' 
                  %(output_path, eval_phase), shell=True)
        ret = bytes.decode(ret).strip("\n").split("\n")
        if len(ret) != dev_count:
            time.sleep(1)
            continue

        try:
            for trainer_id in range(dev_count):
                output_file = os.path.join(output_path,
                         "%s.output.%d" % (eval_phase, trainer_id))
                with open(output_file, "r") as fp:
                     results = json.load(fp)
                     for result in results:
                         assert len(result) == 3
                         all_results.append(
                             RawResult(
                                 unique_id=result[0],
                                 start_logits=result[1],
                                 end_logits=result[2]))
            break 
        except Exception as e:
            log.info('Error!!!!!!!!!!!!!!!!!!!!!!!')

    return all_results


def evaluate(exe, 
             test_program,
             test_pyreader,
             graph_vars,
             eval_phase,
             examples=None,
             features=None,
             args=None,
             trainer_id=0,
             dev_count=1,
             input_file=None,
             output_path=None,
             tokenizer=None,
             version_2_with_negative=False):
    if eval_phase == "train":
        train_fetch_list = [
            graph_vars["loss"].name
        ]
        if "learning_rate" in graph_vars:
            train_fetch_list.append(
                graph_vars["learning_rate"].name
            )
        outputs = exe.run(
            fetch_list=train_fetch_list
        )
        ret = {"loss": np.mean(outputs[0])}
        if "learning_rate" in graph_vars:
            ret["learning_rate"] = float(outputs[1][0])
        return ret

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_prediction_file = os.path.join(output_path, eval_phase + "_predictions.json")
    output_nbest_file = os.path.join(output_path, eval_phase + "_nbest_predictions.json")
    output_null_odds_file = os.path.join(output_path, eval_phase + "_null_odds.json")

    RawResult = collections.namedtuple("RawResult",
            ["unique_id", "start_logits", "end_logits"])

    all_results = [] 
    test_pyreader.start()
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
                # if len(all_results) % 500 == 0:
                #     log.info("Processing example: %d" % len(all_results))
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
    
    if dev_count > 1:
        write_result(output_path, 
                eval_phase, 
                trainer_id, 
                all_results)
        if trainer_id == 0:
            all_results = concat_result(output_path, 
                  eval_phase, 
                  dev_count, 
                  RawResult)

    if trainer_id == 0:
        write_predictions(examples, 
                   features,
                   all_results,
                   args.n_best_size,
                   args.max_answer_length,
                   args.do_lower_case,
                   output_prediction_file,
                   output_nbest_file, 
                   output_null_odds_file,
                   tokenizer, 
                   version_2_with_negative)

        with open(input_file) as fp:
            dataset_json = json.load(fp)

        with open(output_prediction_file) as fp:
            predictions = json.load(fp)

        dataset = dataset_json['data']
        lang = dataset_json['lang']

        eval_out = mlqa_eval(dataset, predictions, lang)

        time_end = time.time()
        elapsed_time = time_end - time_begin

        log.info("[%s evaluation] lang %s, em: %f, f1: %f, elapsed time: %f, eval file: %s"
                % (eval_phase, lang, eval_out["exact_match"], 
                   eval_out["f1"], elapsed_time, input_file))

def write_predictions(all_examples, 
                      all_features, 
                      all_results, 
                      n_best_size,
                      max_answer_length, 
                      do_lower_case, 
                      output_prediction_file,
                      output_nbest_file, 
                      output_null_odds_file, 
                      tokenizer, 
                      version_2_with_negative=True,
                      null_score_diff_threshold=0.0):

    """Write final predictions to the json file and log-odds of null if needed."""
    log.info("Writing predictions to: %s" % (output_prediction_file))
    log.info("Writing nbest to: %s" % (output_nbest_file))

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
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
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

                deal_tok_tokens = []
                for tok_token in tok_tokens:
                    tok_token = str(tok_token)
                    tok_token = tok_token.replace("▁", " ", 1)
                    deal_tok_tokens.append(tok_token)

                tok_text = "".join(deal_tok_tokens)
                tok_text = tok_text.strip()

                final_text = ""
                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=tok_text,
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
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry
        # debug
        if best_non_null_entry is None:
            log.info("Emmm..., sth wrong")


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
                all_predictions[example.qas_id] = ""
                scores_diff_json[example.qas_id] = 0

        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as fp:
        json.dump(all_predictions, fp, indent=4)

    with open(output_nbest_file, "w") as fp:
        json.dump(all_nbest_json, fp, indent=4)
    
    if version_2_with_negative:
        with open(output_null_odds_file, "w") as fp:
            json.dump(scores_diff_json, fp, indent=4)


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
