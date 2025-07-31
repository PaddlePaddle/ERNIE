# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from collections import namedtuple, defaultdict
import numpy as np

MaskedSpanInstance = namedtuple(
    "MaskedSpanInstance", ["index", "begin_label", "end_label"]
)

question_id, s_id, pad_id = (
    29984,
    29980,
    0,
)  # tokenizer.convert_tokens_to_ids(["[QUESTION]", "[<S>]", "[PAD]"])

STOPWORDS = {}
with open(
    "ernie4/src/datasets/streaming_pretrain_reader/cn_stopwords_ids.txt", "r"
) as fin:
    for line in fin:
        STOPWORDS[line.strip()] = 1


def get_candidate_span_clusters(
    tokens, seg_labels, max_span_length, include_sub_clusters=False, validate=True
):
    token_to_indices = defaultdict(list)
    for i, token in enumerate(tokens):
        token_to_indices[token].append(i)

    recurring_spans = []
    for token, indices in token_to_indices.items():
        for i, idx1 in enumerate(indices):
            for j in range(i + 1, len(indices)):
                idx2 = indices[j]
                assert idx1 < idx2

                max_recurring_length = 1
                for length in range(1, max_span_length):
                    if include_sub_clusters:
                        recurring_spans.append((idx1, idx2, length))
                    if (
                        (idx2 + length) >= len(tokens)
                        or tokens[idx1 + length] != tokens[idx2 + length]
                        or tokens[idx1 + length] == s_id
                    ):
                        break
                    max_recurring_length += 1

                if max_recurring_length == max_span_length or not include_sub_clusters:
                    recurring_spans.append((idx1, idx2, max_recurring_length))

    spans_to_clusters = {}
    spans_to_representatives = {}
    for idx1, idx2, length in recurring_spans:
        first_span, second_span = (idx1, idx1 + length - 1), (idx2, idx2 + length - 1)
        if first_span in spans_to_representatives:
            if second_span not in spans_to_representatives:
                rep = spans_to_representatives[first_span]
                cluster = spans_to_clusters[rep]
                cluster.append(second_span)
                spans_to_representatives[second_span] = rep
        else:
            cluster = [first_span, second_span]
            spans_to_representatives[first_span] = first_span
            spans_to_representatives[second_span] = first_span
            spans_to_clusters[first_span] = cluster

    if validate:
        recurring_spans = [
            cluster
            for cluster in spans_to_clusters.values()
            if validate_ngram(
                tokens, seg_labels, cluster[0][0], cluster[0][1] - cluster[0][0] + 1
            )
        ]
    else:
        recurring_spans = spans_to_clusters.values()
    return recurring_spans


def validate_ngram(tokens, seg_labels, start_index, length):
    # If the token is s_id, we don't want to consider this span.
    if tokens[start_index] == s_id and length == 1:
        return False

    # If the vocab at the beginning of the span is a part-of-word (##), we don't want to consider this span.
    # if vocab_word_piece[token_ids[start_index]]:
    if seg_labels[start_index] != 0:
        return False

    # If the token *after* this considered span is a part-of-word (##), we don't want to consider this span.
    if (start_index + length) < len(tokens) and seg_labels[start_index + length] != 0:
        return False

    # We filter out n-grams that are all stopwords (e.g. "in the", "with my", ...)
    if " ".join(map(str, tokens[start_index : start_index + length])) in STOPWORDS:
        return False
    return True


def get_span_clusters_by_length(span_clusters, seq_length):
    already_taken = [False] * seq_length
    span_clusters = sorted(
        [(cluster, cluster[0][1] - cluster[0][0] + 1) for cluster in span_clusters],
        key=lambda x: x[1],
        reverse=True,
    )
    filtered_span_clusters = []
    for span_cluster, _ in span_clusters:
        unpruned_spans = []
        for span in span_cluster:
            if any((already_taken[i] for i in range(span[0], span[1] + 1))):
                continue
            unpruned_spans.append(span)

        # Validating that the cluster is indeed "recurring" after the pruning
        if len(unpruned_spans) >= 2:
            filtered_span_clusters.append(unpruned_spans)
            for span in unpruned_spans:
                for idx in _iterate_span_indices(span):
                    already_taken[idx] = True

    return filtered_span_clusters


def _iterate_span_indices(span):
    return range(span[0], span[1] + 1)


def create_recurring_span_selection_predictions(
    tokens,
    seg_labels,
    max_recurring_predictions=15,
    max_span_length=10,
    masked_lm_prob=0.15,
    ngrams=None,
):
    masked_spans = []
    num_predictions = 0
    input_mask = [1] * len(tokens)
    new_tokens = list(tokens)

    already_masked_tokens = [False] * len(new_tokens)
    span_label_tokens = [False] * len(new_tokens)

    num_to_predict = min(
        max_recurring_predictions, max(1, int(round(len(tokens) * masked_lm_prob)))
    )

    # start_time = time.time()
    span_clusters = get_candidate_span_clusters(
        tokens, seg_labels, max_span_length, include_sub_clusters=True
    )
    span_clusters = get_span_clusters_by_length(span_clusters, len(tokens))
    span_clusters = [
        (cluster, tuple(tokens[cluster[0][0] : cluster[0][1] + 1]))
        for cluster in span_clusters
    ]
    # end_time = time.time()
    # tf.logging.info(f"Finding recurrent ngrams took {end_time - start_time} seconds, {len(tokens)} tokens")

    span_cluster_indices = np.random.permutation(range(len(span_clusters)))
    span_counter = 0
    while span_counter < len(span_cluster_indices):
        span_idx = span_cluster_indices[span_counter]
        span_cluster = span_clusters[span_idx][0]
        # self._assert_and_return_identical(token_ids, identical_spans)
        num_occurrences = len(span_cluster)

        unmasked_span_idx = np.random.randint(num_occurrences)
        unmasked_span = span_cluster[unmasked_span_idx]
        span_counter += 1
        if any(
            [already_masked_tokens[i] for i in _iterate_span_indices(unmasked_span)]
        ):
            # The same token can't be both masked for one pair and unmasked for another pair
            continue

        unmasked_span_beginning, unmasked_span_ending = unmasked_span
        for i, span in enumerate(span_cluster):
            if num_predictions >= num_to_predict:
                # logger.warning(f"Already masked {self.max_predictions} spans.")
                break

            if any(
                [already_masked_tokens[j] for j in _iterate_span_indices(unmasked_span)]
            ):
                break

            if i != unmasked_span_idx:
                if any(
                    [
                        already_masked_tokens[j] or span_label_tokens[j]
                        for j in _iterate_span_indices(span)
                    ]
                ):
                    # The same token can't be both masked for one pair and unmasked for another pair,
                    # or alternatively masked twice
                    continue

                if any(
                    [
                        new_tokens[j] != new_tokens[k]
                        for j, k in zip(
                            _iterate_span_indices(span),
                            _iterate_span_indices(unmasked_span),
                        )
                    ]
                ):
                    print(
                        f"Two non-identical spans: unmasked {new_tokens[unmasked_span_beginning:unmasked_span_ending + 1]}, "
                        f"masked:{new_tokens[span[0]:span[1] + 1]}"
                    )
                    continue

                is_first_token = True
                for j in _iterate_span_indices(span):
                    if is_first_token:
                        new_tokens[j] = question_id
                        masked_spans.append(
                            MaskedSpanInstance(
                                index=j,
                                begin_label=unmasked_span_beginning,
                                end_label=unmasked_span_ending,
                            )
                        )
                        num_predictions += 1
                    else:
                        new_tokens[j] = pad_id
                        input_mask[j] = 0

                    is_first_token = False
                    already_masked_tokens[j] = True

                for j in _iterate_span_indices(unmasked_span):
                    span_label_tokens[j] = True

    assert len(masked_spans) <= num_to_predict
    masked_spans = sorted(masked_spans, key=lambda x: x.index)

    masked_span_positions = []
    span_label_beginnings = []
    span_label_endings = []
    for p in masked_spans:
        masked_span_positions.append(p.index)
        span_label_beginnings.append(p.begin_label)
        span_label_endings.append(p.end_label)

    # Delete [PAD] to make the task harder without leaking length information
    new_tokens_after_delete = []
    new_seg_labels_after_delete = []
    ori_token_idx_to_deleted_token_idx_dict = {}
    assert input_mask[0] != 0, "first token can not be pad"
    for mask_i, mask in enumerate(input_mask):
        if mask != 0:  # pad
            new_tokens_after_delete.append(new_tokens[mask_i])
            new_seg_labels_after_delete.append(seg_labels[mask_i])
        ori_token_idx_to_deleted_token_idx_dict[mask_i] = (
            len(new_tokens_after_delete) - 1
        )

    masked_span_positions = [
        ori_token_idx_to_deleted_token_idx_dict[pos] for pos in masked_span_positions
    ]
    span_label_beginnings = [
        ori_token_idx_to_deleted_token_idx_dict[pos] for pos in span_label_beginnings
    ]
    span_label_endings = [
        ori_token_idx_to_deleted_token_idx_dict[pos] for pos in span_label_endings
    ]
    new_tokens = new_tokens_after_delete
    new_seg_labels = new_seg_labels_after_delete
    assert len(new_tokens) == len(new_seg_labels), "{} != {}".format(
        len(new_tokens), len(new_seg_labels)
    )

    return (
        new_tokens,
        new_seg_labels,
        masked_span_positions,
        span_label_beginnings,
        span_label_endings,
    )
