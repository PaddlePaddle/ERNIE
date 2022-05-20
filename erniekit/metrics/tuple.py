#-*-coding:utf8-*-
"""
infomation extraction task
span extraction evaluate functions
"""
import json
import numpy as np

def get_f1(num_correct, num_infer, num_label):
    """
    get p r f1
    input: 10, 15, 20
    output: (0.6666666666666666, 0.5, 0.5714285714285715)
    """
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
    return (precision, recall, f1)


def get_bool_ids_greater_than(probs, limit, return_prob=False):
    """
    get idx of the last dim in prob arraies, which is greater than a limitation
    input: [[0.1, 0.1, 0.2, 0.5, 0.1, 0.3], [0.7, 0.6, 0.1, 0.1, 0.1, 0.1]]
        0.4
    output: [[3], [0, 1]]
    """
    probs = np.array(probs)
    dim_len = len(probs.shape)
    if dim_len > 1:
        result = []
        for p in probs:
            result.append(get_bool_ids_greater_than(p, limit, return_prob))
        return result
    else:
        result = []
        for i, p in enumerate(probs):
            if p > limit:
                if return_prob:
                    result.append((i, p))
                else:
                    result.append(i)
        return result


def get_span_v2(start_ids, end_ids):
    """
    start id can only be used once
    end id can use multi times
    get span set from position start and end list
    input: [1, 2, 10] [4, 12]
    output: set((2, 4), (10, 12))
    """
    result = []
    start_ids = sorted(start_ids)
    end_ids = sorted(end_ids)
    for s_id in start_ids:
        e_id_list = filter(lambda x: x >= s_id, end_ids)
        if len(e_id_list) > 0:
            e_id = e_id_list[0]
            result.append((s_id, e_id))
    result = set(result)
    return result


def get_span_v3(start_ids, end_ids):
    """
    every id can only be used once
    pick the highest score among points options
    get span set from position start and end list
    input: [(1, 0.51), (2, 0.51), (10, 0.51)] [(4, 0.51), (12, 0.51)]
    output: set(((2, 0.51), (4, 0.51)), ((10, 0.51), (12, 0.51)))
    """
    id_dict = {}
    for x in start_ids:
        id_dict.setdefault(x[0], [x[0], x[1], ""])
        id_dict[x[0]][2] += "s"
    for x in end_ids:
        id_dict.setdefault(x[0], [x[0], x[1], ""])
        id_dict[x[0]][2] += "e"
    id_list = [tuple(x) for x in id_dict.values()]
    id_list = sorted(id_list, key=lambda x: x[0])

    start_pointer = 0
    start_best = None
    end_best = None
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    meet = False
    prev_start_pointer = True
    prev_end_pointer = True

    for this_id in id_list:
        # print "This is", this_id
        # meet end
        if prev_end_pointer and "s" in this_id[-1]:
            # print "MEET", start_best, end_best
            if start_best and end_best and \
                    start_best[0] <= end_best[0]:
                couple_dict[end_best] = start_best
                start_best = None
                end_best = None
        # keep best start
        if "s" in this_id[-1]:
            if start_best is None or \
                    this_id[1] > start_best[1]:
                start_best = this_id
        # keep best end
        if "e" in this_id[-1]:
            if end_best is None or \
                    this_id[1] > end_best[1]:
                end_best = this_id
        # print "best", start_best, end_best
        # next pointer
        prev_start_pointer = "s" in this_id[-1]
        prev_end_pointer = "e" in this_id[-1]

    if start_best and end_best and \
            start_best[0] <= end_best[0]:
        couple_dict[end_best] = start_best
        start_best = None
        end_best = None
    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result


def get_span(start_ids, end_ids, with_prob=False):
    """
    every id can only be used once
    get span set from position start and end list
    input: [1, 2, 10] [4, 12]
    output: set((2, 4), (10, 12))
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        #start_ids = sorted(start_ids)
        #end_ids = sorted(end_ids)
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            if start_ids[start_pointer][0] == end_ids[end_pointer][0]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                end_pointer += 1
                continue
            if start_ids[start_pointer][0] < end_ids[end_pointer][0]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                continue
            if start_ids[start_pointer][0] > end_ids[end_pointer][0]:
                end_pointer += 1
                continue
        else:
            if start_ids[start_pointer][0] == end_ids[end_pointer][0]:
                couple_dict[end_ids[end_pointer][0]] = start_ids[start_pointer][0]
                start_pointer += 1
                end_pointer += 1
                continue
            if start_ids[start_pointer][0] < end_ids[end_pointer][0]:
                couple_dict[end_ids[end_pointer][0]] = start_ids[start_pointer][0]
                start_pointer += 1
                continue
            if start_ids[start_pointer][0] > end_ids[end_pointer][0]:
                end_pointer += 1
                continue
    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result


def eval_span(predict_start_ids, predict_end_ids, label_start_ids, label_end_ids):
    """
    evaluate position extraction (start, end)
    return num_correct, num_infer, num_label
    input: [1, 2, 10] [4, 12] [2, 10] [4, 11]
    output: (1, 2, 2)
    """
    pred_set = get_span(predict_start_ids, predict_end_ids)
    label_set = get_span(label_start_ids, label_end_ids)
    num_correct = len(pred_set & label_set)
    num_infer = len(pred_set)
    num_label = len(label_set)
    return (num_correct, num_infer, num_label)


def eval_multi_class_span(predict_start_ids_list, predict_end_ids_list, label_start_ids_list, label_end_ids_list):
    """
    evaluate position extraction (start, end), input is an ids list
    return num_correct, num_infer, num_label
    input: [[1, 2, 10]], [[4, 12]], [[2, 10]], [[4, 11]]
    output: (1, 2, 2)
    """
    num_correct = 0
    num_infer = 0
    num_label = 0
    for predict_start_ids, predict_end_ids,\
            label_start_ids, label_end_ids in \
            zip(predict_start_ids_list, predict_end_ids_list,\
            label_start_ids_list, label_end_ids_list):
        result = eval_span(predict_start_ids, predict_end_ids, label_start_ids, label_end_ids)
        num_correct += result[0]
        num_infer += result[1]
        num_label += result[2]
    return (num_correct, num_infer, num_label)


def sets_compare(sets_predict, sets_label):
    """
    sets_compare_label
    input: [data [tuple list [ tuple ...] ... ] ...]
    output: num_correct, num_pred, num_label
    NOTE: basic set should be tuple
    """
    num_correct = 0
    num_pred = 0
    num_label = 0
    for set_1, set_2 in zip(sets_predict, sets_label):
        set_1 = set(set_1)
        set_2 = set(set_2)
        num_pred += len(set_1)
        num_label += len(set_2)
        num_correct += len(set_1 & set_2)
    return (num_correct, num_pred, num_label)


if __name__ == "__main__":
    print(get_f1(10, 15, 20))
    print(get_bool_ids_greater_than(
        np.array([[0.1, 0.1, 0.2, 0.5, 0.1, 0.3], [0.7, 0.6, 0.1, 0.1, 0.1, 0.1]]),
        0.4))
    print(get_span([1, 2, 10, 11, 15], [4, 11, 12, 14, 20]))
    print(get_span_v2([1, 2, 10, 11, 15], [4, 11, 12, 14, 20]))
    print(get_span_v3(
        [(0, 0.51), (1, 0.51), (2, 0.52), (5, 0.55), (6, 0.52), (9, 0.52), (10, 0.51)], 
        [(0, 0.51), (4, 0.51), (7, 0.55), (8, 0.52), (12, 0.55), (13, 0.51)]))
    print(eval_span([1, 2, 10], [4, 12], [2, 10], [4, 11]))
    print(eval_multi_class_span([[1, 2, 10]], [[4, 12]], [[2, 10]], [[4, 11]]))
    print(sets_compare(
            [
                [(1, 2, 10)]
            ], 
            [
                [(4, 12, 1), (1, 2, 10), (1, 2, 2)]
            ]))

