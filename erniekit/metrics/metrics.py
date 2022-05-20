# -*- coding: utf-8 -*
"""
文心中常用的指标与评估方式
"""
import sys
import re
import nltk
import six
import math
import logging

import numpy as np
import logging

from sklearn import metrics
from paddle import fluid
from six.moves import xrange
import json
import collections
from erniekit.metrics.tokenization import BasicTokenizer

log = logging.getLogger(__name__)

__all__ = [
    'Metrics', 'F1', 'Recall', 'Precision', 'Acc', "NerChunk", "MRC"
]


class Metrics(object):
    """Metrics"""
    def eval(self, run_value):
        """need overwrite， run_value是动态fetch回来的值，按需要进行计算和打印"""
        raise NotImplementedError


class Chunk(Metrics):
    """Chunk"""
    def eval(self, run_value):
        chunk_metrics = fluid.metrics.ChunkEvaluator()
        num_infer_chunks, num_label_chunks, num_correct_chunks = run_value
        if isinstance(num_infer_chunks[0], np.ndarray):
            for i in range(len(num_infer_chunks)):
                chunk_metrics.update(np.array(num_infer_chunks[i][0]), np.array(num_label_chunks[i][0]),
                                     np.array(num_correct_chunks[i][0]))
        else:
            for i in range(len(num_infer_chunks)):
                chunk_metrics.update(np.array(num_infer_chunks[i]), np.array(num_label_chunks[i]),
                                     np.array(num_correct_chunks[i]))
        precision, recall, f1_score = chunk_metrics.eval()
        result = {"precision": float("%.4f" % precision), "recall": float("%.4f" % recall),
                  "f1_score": float("%.4f" % f1_score)}
        return result


class Acc(Metrics):
    """Acc"""
    def eval(self, run_value):
        predict, label = run_value
        if isinstance(predict, list):
            tmp_arr = []
            for one_batch in predict:
                for pre in one_batch:
                    tmp_arr.append(np.argmax(pre))
        else:
            tmp_arr = []
            for pre in predict:
                tmp_arr.append(np.argmax(pre))

        predict_arr = np.array(tmp_arr)

        if isinstance(label, list):
            tmp_arr = []
            for one_batch in label:
                batch_arr = [one_label for one_label in one_batch]
                tmp_arr.extend(batch_arr)
            label_arr = np.array(tmp_arr)
        else:
            label_arr = np.array(label.flatten())

        score = metrics.accuracy_score(label_arr, predict_arr)

        return float("%.4f" % score)


class Precision(Metrics):
    """Precision"""
    def eval(self, run_value):
        predict, label = run_value
        if isinstance(predict, list):
            tmp_arr = []
            for one_batch in predict:
                for pre in one_batch:
                    tmp_arr.append(np.argmax(pre))
        else:
            tmp_arr = []
            for pre in predict:
                tmp_arr.append(np.argmax(pre))

        predict_arr = np.array(tmp_arr)

        if isinstance(label, list):
            tmp_arr = []
            for one_batch in label:
                batch_arr = [one_label for one_label in one_batch]
                tmp_arr.extend(batch_arr)
            label_arr = np.array(tmp_arr)
        else:
            label_arr = np.array(label.flatten())

        score = metrics.precision_score(label_arr, predict_arr, average="macro")
        # logging.info("sklearn precision macro score = ", score)
        return float("%.4f" % score)


class Recall(Metrics):
    """Recall"""
    def eval(self, run_value):
        predict, label = run_value
        predict_arr = None
        label_arr = None
        if isinstance(predict, list):
            tmp_arr = []
            for one_batch in predict:
                for pre in one_batch:
                    tmp_arr.append(np.argmax(pre))
        else:
            tmp_arr = []
            for pre in predict:
                tmp_arr.append(np.argmax(pre))

        predict_arr = np.array(tmp_arr)

        if isinstance(label, list):
            tmp_arr = []
            for one_batch in label:
                batch_arr = [one_label for one_label in one_batch]
                tmp_arr.extend(batch_arr)
            label_arr = np.array(tmp_arr)
        else:
            label_arr = np.array(label.flatten())

        score = metrics.recall_score(label_arr, predict_arr, average="macro")
        # logging.info("sklearn recall macro score = ", score)
        return float("%.4f" % score)


class F1(Metrics):
    """F1"""

    def eval(self, run_value):
        predict, label = run_value
        predict_arr = None
        label_arr = None
        if isinstance(predict, list):
            tmp_arr = []
            for one_batch in predict:
                for pre in one_batch:
                    tmp_arr.append(np.argmax(pre))
        else:
            tmp_arr = []
            for pre in predict:
                tmp_arr.append(np.argmax(pre))

        predict_arr = np.array(tmp_arr)

        if isinstance(label, list):
            tmp_arr = []
            for one_batch in label:
                batch_arr = [one_label for one_label in one_batch]
                tmp_arr.extend(batch_arr)
            label_arr = np.array(tmp_arr)
        else:
            label_arr = np.array(label.flatten())

        score = metrics.f1_score(label_arr, predict_arr, average="macro")
        # logging.info("sklearn f1 macro score = ", score)
        return float("%.4f" % score)


class Auc(Metrics):
    "Auc"
    def eval(self, run_value):
        predict, label = run_value
        predict_arr = None
        label_arr = None

        if isinstance(predict, list):
            tmp_arr = []
            for one_batch in predict:
                for pre in one_batch:
                    assert len(pre) == 2, "auc metrics only support binary classification, \
                                                          and the positive label must be 1, negative label must be 0"
                    tmp_arr.append(pre[1])
        else:
            tmp_arr = []
            for pre in predict:
                assert len(pre) == 2, "auc metrics only support binary classification, " \
                                      "and the positive label must be 1, negative label must be 0"
                tmp_arr.append(pre[1])

        predict_arr = np.array(tmp_arr)

        if isinstance(label, list):
            tmp_arr = []
            for one_batch in label:
                batch_arr = [one_label for one_label in one_batch]
                tmp_arr.extend(batch_arr)
            label_arr = np.array(tmp_arr)
        else:
            label_arr = np.array(label.flatten())

        fpr, tpr, thresholds = metrics.roc_curve(label_arr, predict_arr)
        score = metrics.auc(fpr, tpr)
        return float("%.4f" % score)


class Pn(Metrics):
    """Pn"""
    def eval(self, run_value):
        pos_score, neg_score = run_value
        wrong_cnt = np.sum(pos_score <= neg_score)
        right_cnt = np.sum(pos_score > neg_score)
        if wrong_cnt == 0:
            pn = float("inf")
        else:
            pn = float(right_cnt) / wrong_cnt

        return pn


class PairWiseAcc(Metrics):
    """PairWiseAcc"""
    def eval(self, run_value):
        pos_score, neg_score = run_value
        wrong_cnt = 0
        right_cnt = 0
        if isinstance(pos_score, list):
            for index in range(len(pos_score)):
                wrong_cnt += np.sum(pos_score[index] <= neg_score[index])
                right_cnt += np.sum(pos_score[index] > neg_score[index])
        else:
            wrong_cnt = np.sum(pos_score <= neg_score)
            right_cnt = np.sum(pos_score > neg_score)

        pn = float(right_cnt) / (wrong_cnt + right_cnt) * 1.0
        return float("%.4f" % pn)


class Ppl(Metrics):
    """ppl"""

    def eval(self, run_value):
        label_len, loss = run_value
        cost_train = np.mean(loss)

        if isinstance(label_len, list):
            tmp_arr = []
            for one_batch in label_len:
                batch_arr = [one_len for one_len in one_batch]
                tmp_arr.extend(batch_arr)
            label_len = np.array(tmp_arr)

        word_count = np.sum(label_len)
        total_loss = cost_train * label_len.shape[0]
        try_loss = total_loss / word_count
        ppl = np.exp(total_loss / word_count)

        p_inf = float("inf")
        if ppl < p_inf:
            ppl = int(ppl)

        result = {"ave_loss": float("%.4f" % try_loss), "ppl": ppl}
        return result


class MRC(Metrics):
    """MRC"""

    def write_predictions(self, all_examples, all_features, all_results, n_best_size,
                          max_answer_length, do_lower_case, output_prediction_file,
                          output_nbest_file):

        """Write final predictions to the json file and log-odds of null if needed."""
        logging.info("Writing predictions to: %s" % (output_prediction_file))
        logging.info("Writing nbest to: %s" % (output_nbest_file))

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

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                start_indexes = self._get_best_indexes(result.start_logits, n_best_size)
                end_indexes = self._get_best_indexes(result.end_logits, n_best_size)

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
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = "".join(orig_tokens)

                    final_text = self.get_final_text(tok_text, orig_text, do_lower_case)
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

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NbestPrediction(
                        text="empty", start_logit=0.0, end_logit=0.0))

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)

            probs = self._compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            all_predictions[example.qas_id] = nbest_json[0]["text"]
            all_nbest_json[example.qas_id] = nbest_json

        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")

        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n")

    def write_predictions_with_negative(self, all_examples, all_features, all_results, n_best_size,
                                        max_answer_length, do_lower_case, output_prediction_file,
                                        output_nbest_file, threshold=0.5):

        """Write final predictions to the json file and log-odds of null if needed."""
        logging.info("Writing predictions to: %s" % (output_prediction_file))
        logging.info("Writing nbest to: %s" % (output_nbest_file))

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

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]

            min_null_prediction = None
            prelim_predictions = []
            score_answerable = -1
            # keep track of the minimum score of null start+end of position 0
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]

                exp_answerable_scores = np.exp(result.cls_logits - np.max(result.cls_logits))
                feature_answerable_score = exp_answerable_scores / np.sum(exp_answerable_scores)

                if feature_answerable_score[-1] > score_answerable:
                    score_answerable = feature_answerable_score[-1]
                    answerable_probs = feature_answerable_score

                feature_null_score = result.start_logits[0] + result.end_logits[0]

                if min_null_prediction is None or \
                    min_null_prediction.start_logit + min_null_prediction.end_logit > feature_null_score:
                    min_null_prediction = _PrelimPrediction(
                        feature_index=feature_index,
                        start_index=0,
                        end_index=0,
                        start_logit=result.start_logits[0],
                        end_logit=result.end_logits[0]
                    )

                start_indexes = self._get_best_indexes(result.start_logits, n_best_size)
                end_indexes = self._get_best_indexes(result.end_logits, n_best_size)

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

            prelim_predictions.append(min_null_prediction)
            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: (x.start_logit + x.end_logit),
                reverse=True)[:n_best_size]

            if not any(p.start_index == 0 and p.end_index == 0 for p in prelim_predictions):
                prelim_predictions.append(min_null_prediction)

            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction", ["text", "start_logit", "end_logit"])

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                feature = features[pred.feature_index]
                if pred.start_index > 0:  # this is a non-null prediction
                    tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1
                                                                  )]
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end +
                                                                     1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = "".join(orig_tokens)

                    final_text = self.get_final_text(tok_text, orig_text, do_lower_case)
                    if final_text == "":
                        final_text = "no answer"
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = "no answer"
                    seen_predictions[final_text] = True

                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit))

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NbestPrediction(
                        text="no answer", start_logit=0.0, end_logit=0.0))

            total_scores = []
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)

            probs = self._compute_softmax(total_scores)
            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            i = 0
            while nbest_json[i]['text'] == "no answer" and i < len(nbest_json) - 1:
                i += 1
            best_non_null_entry = nbest_json[i]

            if answerable_probs[1] < threshold:
                all_predictions[example.qas_id] = "no answer"
            else:
                all_predictions[example.qas_id] = best_non_null_entry["text"]

            all_nbest_json[example.qas_id] = nbest_json

        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")

        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n")

    def write_cloze_predictions(self, all_examples, all_features, all_results, n_best_size,
                          max_answer_length, do_lower_case, output_prediction_file,
                          output_nbest_file):
        """Write final predictions to the json file and log-odds of null if needed."""

        logging.info("Writing predictions to: %s" % (output_prediction_file))
        logging.info("Writing nbest to: %s" % (output_nbest_file))

        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction", [
                "feature_index", "index", "logit"])

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                indexes = self._get_best_indexes(result.logits, n_best_size)

                for index in indexes:
                    if index >= len(feature.tokens):
                        continue
                    if index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(index, False):
                        continue
                    length = 1

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            index=index,
                            logit=result.logits[index]))

            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: (x.logit),
                reverse=True)

            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction", ["text", "logit"])

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break
                feature = features[pred.feature_index]
                if pred.index > 0:  # this is a non-null prediction
                    tok_tokens = feature.tokens[pred.index:(pred.index + 1)]
                    orig_doc = feature.token_to_orig_map[pred.index]
                    orig_tokens = example.doc_tokens[orig_doc:(orig_doc + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = "".join(orig_tokens)

                    final_text = self.get_final_text(tok_text, orig_text, do_lower_case)
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                    nbest.append(
                        _NbestPrediction(
                            text=final_text,
                            logit=pred.logit))

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NbestPrediction(
                        text="empty", logit=0.0))

            total_scores = []
            for entry in nbest:
                total_scores.append(entry.logit)

            probs = self._compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["logit"] = entry.logit
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            # all_predictions[example.qas_id] = nbest_json[0]["text"]
            all_predictions[example.qas_id] = \
                [[tmp["text"] for tmp in nbest_json], [tmp['probability'] for tmp in nbest_json]]
            all_nbest_json[example.qas_id] = nbest_json

        all_examples_dict = dict()
        for example in all_examples:
            all_examples_dict[example.qas_id]=example

        all_context_dict = dict()
        for key, value in all_predictions.items():
            context_id, answer_id = key.split("###")
            example = all_examples_dict[key]
            if context_id in all_context_dict:
                all_context_dict[context_id].append([example, answer_id, value])
            else:
                all_context_dict[context_id]=[[example, answer_id, value]]

        out_put_predict = dict()
        for key, value in all_context_dict.items():
            final_id = key
            answer_position_num = 0
            answer_position_to_id = None
            answer_id_to_predict_val = dict()

            for example, answer_id, val in value:
                if answer_position_num == 0:
                    answer_position_num = len(example.orig_answer_text)
                elif answer_position_num != len(example.orig_answer_text):
                    logging.error("error")
                answer_position_to_id = example.orig_answer_text
                answer_id_to_predict_val[answer_id] = val

            answer_text_dict = dict()
            while True:# search best answer for every choices
                is_update = False
                for key, value in answer_id_to_predict_val.items():
                    if len(value[0]) == 0:
                        continue
                    best_answer_text=value[0][0]
                    best_answer_score=value[1][0]
                    if best_answer_text in answer_text_dict:
                        answer_text_score_id = answer_text_dict[best_answer_text]
                        if best_answer_score > answer_text_score_id[0]:
                            tmp_value = answer_id_to_predict_val[answer_text_score_id[1]]
                            tmp_value[0] = tmp_value[0][1:]
                            tmp_value[1] = tmp_value[1][1:]
                            answer_id_to_predict_val[answer_text_score_id[1]] = tmp_value
                            answer_text_dict[best_answer_text] = [best_answer_score, key]
                            is_update = True
                        elif answer_text_score_id[1] != key:
                            tmp_value=answer_id_to_predict_val[key]
                            tmp_value[0] = tmp_value[0][1:]
                            tmp_value[1] = tmp_value[1][1:]
                            answer_id_to_predict_val[key] = tmp_value
                            is_update = True
                    else:
                        answer_text_dict[best_answer_text] = [best_answer_score, key]
                        is_update = True
                if is_update == False:
                    break

            out_put_predict[final_id] = [-1] * len(answer_position_to_id)
            for key, value_1 in answer_id_to_predict_val.items(): #get final predict file
                if len(value_1) > 0 and len(value_1[0]) > 0:
                    answer_pos=value_1[0][0]
                    answer_pos=answer_pos.replace("[unused", "")
                    answer_pos=answer_pos.replace("]", "")
                    try:
                        answer_pos = int(answer_pos)
                        out_put_predict[final_id][answer_pos - 5] = int(key)
                    except Exception as e:
                        continue

        json.dump(out_put_predict, open(output_prediction_file, mode="w"), indent=4)

    def get_final_text(self, pred_text, orig_text, do_lower_case):
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
        tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        tok_text = " ".join(tokenizer.tokenize(orig_text))

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


    def _get_best_indexes(self, logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(
            enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes


    def _compute_softmax(self, scores):
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

    def _mixed_segmentation(self, in_str, rm_punc=False):
        """mixed_segmentation"""
        if six.PY2:
            in_str = str(in_str).decode('utf-8')
        in_str = in_str.lower().strip()
        segs_out = []
        temp_str = ""
        sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
                   '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
                   '「', '」', '（', '）', '－', '～', '『', '』']
        for char in in_str:
            if rm_punc and char in sp_char:
                continue
            pattern = six.u(r'[\u4e00-\u9fa5]')
            if re.search(pattern, char) or char in sp_char:
                if temp_str != "":
                    ss = nltk.word_tokenize(temp_str)
                    segs_out.extend(ss)
                    temp_str = ""
                segs_out.append(char)
            else:
                temp_str += char

        #handling last part
        if temp_str != "":
            ss = nltk.word_tokenize(temp_str)
            segs_out.extend(ss)

        return segs_out

    # remove punctuation
    def _remove_punctuation(self, in_str):
        """remove_punctuation"""
        if six.PY2:
            in_str = str(in_str).decode('utf-8')
        in_str = in_str.lower().strip()
        sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
                   '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
                   '「', '」', '（', '）', '－', '～', '『', '』']
        out_segs = []
        for char in in_str:
            if char in sp_char:
                continue
            else:
                out_segs.append(char)
        return ''.join(out_segs)

    # find longest common string
    def _find_lcs(self, s1, s2):
        """find lcs"""
        m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
        mmax = 0
        p = 0
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    m[i + 1][j + 1] = m[i][j] + 1
                    if m[i + 1][j + 1] > mmax:
                        mmax = m[i + 1][j + 1]
                        p = i + 1
        return s1[p - mmax:p], mmax

    def _evaluate(self, ground_truth_file, prediction_file):
        """evaluate form files"""
        f1 = 0
        em = 0
        total_count = 0
        skip_count = 0
        for instances in ground_truth_file["data"]:
            for instance in instances["paragraphs"]:
                context_text = instance['context'].strip()
                for qas in instance['qas']:
                    total_count += 1
                    query_id = qas['id'].strip()
                    query_text = qas['question'].strip()
                    is_impossible = None
                    if "is_impossible" in qas.keys():
                        is_impossible = qas["is_impossible"]
                        if is_impossible:
                            assert len(qas["answers"]) == 1
                            answers = ["no answer"]
                        else:
                            answers = [ans["text"] for ans in qas["answers"]]
                    else:
                        answers = [ans["text"] for ans in qas["answers"]]

                    if query_id not in prediction_file:
                        sys.stderr.write('Unanswered question: {}\n'.format(query_id))
                        skip_count += 1
                        continue

                    prediction = str(prediction_file[query_id])
                    if is_impossible:
                        if prediction.lower() == "no answer":
                            _f1 = 1.0
                            _em = 1.0
                        else:
                            _f1 = 0.0
                            _em = 0.0
                    else:
                        _f1 = self._calc_f1_score(answers, prediction)
                        _em = self._calc_em_score(answers, prediction)

                    f1 += _f1
                    em += _em

        f1_score = 100.0 * f1 / total_count
        em_score = 100.0 * em / total_count
        return f1_score, em_score, (total_count, skip_count)


    def _calc_f1_score(self, answers, prediction):
        """calculate f1 score"""
        f1_scores = []
        for ans in answers:
            ans_segs = self._mixed_segmentation(ans, rm_punc=True)
            prediction_segs = self._mixed_segmentation(prediction, rm_punc=True)
            lcs, lcs_len = self._find_lcs(ans_segs, prediction_segs)
            if lcs_len == 0:
                f1_scores.append(0)
                continue
            precision = 1.0 * lcs_len / len(prediction_segs)
            recall = 1.0 * lcs_len / len(ans_segs)
            f1 = (2 * precision * recall) / (precision + recall)
            f1_scores.append(f1)
        return max(f1_scores)


    def _calc_em_score(self, answers, prediction):
        """calculate em score"""
        em = 0
        for ans in answers:
            ans_ = self._remove_punctuation(ans)
            prediction_ = self._remove_punctuation(prediction)
            if ans_ == prediction_:
                em = 1
                break
        return em


    def eval(self, run_value):
        """evaluation"""
        dataset_file, prediction_file = run_value
        ground_truth_file = json.load(open(dataset_file, 'rb'))
        prediction_file = json.load(open(prediction_file, 'rb'))
        F1, EM, (TOTAL, SKIP) = self._evaluate(ground_truth_file, prediction_file)
        AVG = (EM + F1) * 0.5
        return EM, F1, (AVG, TOTAL)


class NerChunk(Metrics):
    """Ner Chunck"""

    def eval(self, run_value):
        """evaluation"""
        np_labels, np_infers, np_lens, tag_num, max_len, label_map, is_glyce, is_cluener = run_value
        if is_glyce:
            pred_label, gold_label, pred_mask = [], [], []
            for i in xrange(len(np_lens)):
                seg_st = i * max_len + 1
                seg_en = seg_st + (np_lens[i] - 2)
                pred_label.append(np_infers[seg_st:seg_en])
                gold_label.append(np_labels[seg_st:seg_en])
                pred_mask.append([1]*len(np_infers[seg_st:seg_en]))
            num_label, num_infer, num_correct = self.chunk_eval_glyce(gold_label, pred_label, pred_mask, tag_num)

        elif is_cluener:
            num_label, num_infer, num_correct = \
                    self.chunk_eval_cluener(np_labels, np_infers, np_lens, tag_num, max_len, label_map)
        else:
            num_label, num_infer, num_correct = \
                    self.chunk_eval_std(np_labels, np_infers, np_lens, tag_num, max_len)

        precision, recall, f1 = calculate_f1(num_label, num_infer, num_correct)
        return precision, recall, f1

    def chunk_eval_std(self, np_labels, np_infers, np_lens, tag_num, max_len):
        """ chunk_eval_std """

        def extract_bio_chunk(seq):
            """ extract_bio_chunk """
            mod = 2
            chunks = []
            cur_chunk = None
            null_index = tag_num - 1
            for index in xrange(len(seq)):
                tag = seq[index]
                tag_type = tag // mod
                tag_pos = tag % mod

                if tag == null_index:
                    if cur_chunk is not None:
                        chunks.append(cur_chunk)
                        cur_chunk = None
                    continue

                cmd = (tag_pos == 0)
                if cmd:
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

        labels, infers, lens = np_labels, np_infers, np_lens

        base_index = 0

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

    def extract_entities(self, labels_lst, start_label="1_4"):
        """extract_entities"""

        def gen_entities(label_lst, start_label=1, dims=1):
            """gen_entities"""
            # rules -> if end_mark > start_label
            entities = dict()

            if "_" in start_label:
                start_label = start_label.split("_")
                start_label = [int(tmp) for tmp in start_label]
                ind_func = lambda x: (bool(label in start_label) for label in x)
                indicator = sum([int(tmp) for tmp in ind_func(label_lst)])
            else:
                start_label = int(start_label)
                indicator = 1 if start_label in labels_lst else 0

            if indicator > 0:
                if isinstance(start_label, list):
                    ixs, _ = zip(*filter(lambda x: x[1] in start_label, enumerate(label_lst)))
                elif isinstance(start_label, int):
                    ixs, _ = zip(*filter(lambda x: x[1] == start_label, enumerate(label_lst)))
                else:
                    raise ValueError("You Should Notice that The FORMAT of your INPUT")

                ixs = list(ixs)
                ixs.append(len(label_lst))
                for i in range(len(ixs) - 1):
                    sub_label = label_lst[ixs[i]: ixs[i + 1]]
                    end_mark = max(sub_label)
                    end_ix = ixs[i] + sub_label.index(end_mark) + 1
                    entities["{}_{}".format(ixs[i], end_ix)] = label_lst[ixs[i]: end_ix]
            return entities

        if start_label == "1":
            entities = gen_entities(labels_lst, start_label=int(start_label))
        elif start_label == "4":
            entities = gen_entities(labels_lst, start_label=int(start_label))
        elif "_" in start_label:
            entities = gen_entities(labels_lst, start_label=start_label)
        else:
            raise ValueError("You Should Check The FOMAT Of your SPLIT NUMBER !!!!!")

        return entities

    def chunk_eval_glyce(self, gold_label, pred_label, pred_mask, tag_num, dims=2):
        """chunk_eval_glyce"""

        start_label_1 = list(range(0, tag_num - 1, 4))
        start_label_2 = list(range(3, tag_num - 1, 4))
        assert len(start_label_1) == len(start_label_2)
        start_label = []
        for i in range(len(start_label_1)):
            start_label += [str(start_label_1[i])] + [str(start_label_2[i])]
        start_label = "_".join(start_label)

        #print(pred_label, gold_label, pred_mask)
        if dims == 1:
            mask_index = [tmp_idx for tmp_idx, tmp in enumerate(pred_mask) if tmp != 0]
            pred_label = [tmp for tmp_idx, tmp in enumerate(pred_label) if tmp_idx in mask_index]
            gold_label = [tmp for tmp_idx, tmp in enumerate(gold_label) if tmp_idx in mask_index]

            pred_entities = self.extract_entities(pred_label, start_label=start_label)
            truth_entities = self.extract_entities(gold_label, start_label=start_label)

            num_true = len(truth_entities)
            num_extraction = len(pred_entities)

            num_true_positive = 0
            for entity_idx in pred_entities.keys():
                try:
                    if truth_entities[entity_idx] == pred_entities[entity_idx]:
                        num_true_positive += 1
                except:
                    pass

            dict_match = list(filter(lambda x: x[0] == x[1], zip(pred_label, gold_label)))
            return num_true_positive, float(num_extraction), float(num_true)

        elif dims == 2:
            acc, posit, extra, true = 0, 0, 0, 0
            for pred_item, truth_item, mask_item in zip(pred_label, gold_label, pred_mask):
                if not pred_item or not truth_item or not mask_item:
                    pass
                else:
                    tmp_posit, tmp_extra, tmp_true = \
                        self.chunk_eval_glyce(truth_item, pred_item, mask_item, tag_num, dims=1)
                posit += tmp_posit
                extra += tmp_extra
                true += tmp_true

            return true, extra, posit

    def chunk_eval_cluener(self, np_labels, np_infers, np_lens, tag_num, max_len, label_map_config):
        """chunk_eval_cluener"""

        def get_entity_bios(seq, id2label):
            """Gets entities from sequence.
            note: BIOS
            Args:
                seq (list): sequence of labels.
            Returns:
                list: list of (chunk_type, chunk_start, chunk_end).
            Example:
                # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
                # >>> get_entity_bios(seq)
                [['PER', 0,1], ['LOC', 3, 3]]
            """
            chunks = []
            chunk = [-1, -1, -1]
            for indx, tag in enumerate(seq):
                if not isinstance(tag, str):
                    tag = id2label[tag]
                if tag.startswith("S-"):
                    if chunk[2] != -1:
                        chunks.append(chunk)
                    chunk = [-1, -1, -1]
                    chunk[1] = indx
                    chunk[2] = indx
                    chunk[0] = tag.split('-')[1]
                    chunks.append(chunk)
                    chunk = (-1, -1, -1)
                if tag.startswith("B-"):
                    if chunk[2] != -1:
                        chunks.append(chunk)
                    chunk = [-1, -1, -1]
                    chunk[1] = indx
                    chunk[0] = tag.split('-')[1]
                elif tag.startswith('I-') and chunk[1] != -1:
                    _type = tag.split('-')[1]
                    if _type == chunk[0]:
                        chunk[2] = indx
                    if indx == len(seq) - 1:
                        chunks.append(chunk)
                else:
                    if chunk[2] != -1:
                        chunks.append(chunk)
                    chunk = [-1, -1, -1]
            return chunks

        pred_label, gold_label = [], []
        label2id = json.load(open(label_map_config))
        id2label = dict(zip(label2id.values(), label2id.keys()))

        for i in xrange(len(np_lens)):
            seg_st = i * max_len + 1
            seg_en = seg_st + (np_lens[i] - 2)
            pred_label.append(np_infers[seg_st:seg_en])
            gold_label.append(np_labels[seg_st:seg_en])

        origins = []
        founds = []
        rights = []

        for gold, pred in zip(gold_label, pred_label):
            label_entities = get_entity_bios(gold, id2label)
            pre_entities = get_entity_bios(pred, id2label)
            right_entities = [pre_entity for pre_entity in pre_entities if pre_entity in label_entities]

            origins.extend(label_entities)
            founds.extend(pre_entities)
            rights.extend(right_entities)

        num_label = len(origins)
        num_infer = len(founds)
        num_correct = len(rights)

        return num_label, num_infer, num_correct


def chunk_eval(np_labels, np_infers, np_lens, tag_num, dev_count=1):
    """ chunk_eval """
    def extract_bio_chunk(seq):
        """ extract_bio_chunk """
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
    """ calculate_f1 """
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


def DCG(label_list):
    """ DCG """
    import math
    dcgsum = 0
    for i in range(len(label_list)):
        dcg = (2 ** label_list[i] - 1) / math.log(i + 2, 2)
        dcgsum += dcg
    return dcgsum

def NDCG(label_list, topK):
    """ NDCG """
    dcg = DCG(label_list[0:topK])
    ideal_list = sorted(label_list, reverse=True)
    ideal_dcg = DCG(ideal_list[0:topK])
    if ideal_dcg == 0:
        return 0
    return dcg / ideal_dcg


class LmPpl(Metrics):
    """ppl for language model"""
    def eval(self, run_value):
        label_len, loss = run_value
        total_ppl = 0.0
        for seq_loss in loss:
            avg_ppl = np.exp(seq_loss)
            seq_ppl = np.mean(avg_ppl)
            total_ppl += seq_ppl
        ave_ppl = total_ppl / len(loss)
        return int(ave_ppl)

def compare_list(ground_truth, prediction):
    """compare list"""
    right_count = 0
    min_len = len(ground_truth) if len(ground_truth)<len(prediction) else len(prediction)
    gap_count = len(ground_truth) - min_len

    for k in range(min_len):
        if str(ground_truth[k]) == str(prediction[k]):
            right_count += 1

    final_right_count = right_count - gap_count
    if final_right_count < 0:
        final_right_count = 0
    return final_right_count

def cal_qac_and_pac(ground_truth_file, prediction_file):
    """calculate qac and pac for cmrc2019"""
    qac = 0
    pac = 0
    qac_score = 0
    pac_score = 0
    total_question_count = 0
    skip_question_count = 0
    total_passage_count = 0

    ground_truth_file = json.load(open(ground_truth_file, 'rb'))
    prediction_file = json.load(open(prediction_file, 'rb'))

    for instance in ground_truth_file["data"]:
        context = instance["context"]
        context_id = instance["context_id"]
        choices = instance["choices"]
        answers = instance["answers"]

        predictions = []
        if context_id not in prediction_file:
            logging.info("Not found context_id in prediction: {}\n".format(context_id))
            right_question_count = 0
        else:
            predictions	= prediction_file[context_id]
            right_question_count = compare_list(answers, predictions)

        qac += right_question_count
        pac += (right_question_count == len(answers))

        total_question_count += len(answers)
        skip_question_count += len(answers) - len(predictions)
        total_passage_count += 1

    qac_score = 100.0 * qac / total_question_count
    pac_score = 100.0 * pac / total_passage_count

    if skip_question_count:
        logging.info("***Number of predicted samples is not equal to ground truth!***")

    return qac_score, pac_score, (total_question_count, skip_question_count)


def logits_matrix_to_array(logits_matrix, index_2_uniqueid, index_2_label):
    """evaluation for chid"""
    logits_matrix = np.array(logits_matrix)
    logits_matrix = np.transpose(logits_matrix)
    tmp = []
    for i, row in enumerate(logits_matrix):
        for j, col in enumerate(row):
            tmp.append((i, j, col))
    else:
        choice = set(range(i + 1))
        blanks = set(range(j + 1))
    tmp = sorted(tmp, key=lambda x: x[2], reverse=True)
    results = []
    for i, j, v in tmp:
        if (j in blanks) and (i in choice):
            results.append((i, j))
            blanks.remove(j)
            choice.remove(i)
    results = sorted(results, key=lambda x: x[1], reverse=False)

    ret_results = [[index_2_uniqueid[j], i] for i, j in results]
    ret_labels = [[index_2_label[j], i] for i, j in results]
    return ret_results, ret_labels


def chid_eval(logits, labels, qids, example_ids):
    """evaluation for chid"""
    assert len(logits) == len(labels) == len(qids) == len(example_ids)
    raw_resluts = {}

    for i in range(len(logits)):
        e_id = example_ids[i]
        if e_id not in raw_resluts:
            raw_resluts[e_id] = [(logits[i], qids[i], labels[i])]
        else:
            raw_resluts[e_id].append((logits[i], qids[i], labels[i]))

    results = []
    rel_labels = []

    for e_id, elem in raw_resluts.items():
        index_2_uniqueid = {index: q_id for index, (_, q_id, _) in enumerate(elem)}
        index_2_label = {index: label for index, (_, _, label) in enumerate(elem)}

        all_logits = [logits for (logits, _, _) in elem]

        tmp_result, tmp_label = logits_matrix_to_array(all_logits, index_2_uniqueid, index_2_label)
        results.extend(tmp_result)
        rel_labels.extend(tmp_label)


    num_correct = sum([int(ele[0] == ele[1]) for ele in rel_labels])
    acc = float(num_correct) / float(len(rel_labels))

    return acc