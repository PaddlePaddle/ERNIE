""" Eval cross-modal retrieval """
import numpy as np
import sys
import logging
from tabulate import tabulate



def parse_vec(line, dilter="\t"):
    """
    parse_vec
    """
    vec = [float(x) for x in line.split(dilter)]
    return np.array(vec)


def matmul(a, b, na):
    """
    multi-thread matmul
    """
    assert a.shape[0] % na == 0
    k = a.shape[0] // na
    lst = [a[i * k: (i + 1) * k] for i in range(na)]
    rlst = []
    
    for i, x in enumerate(lst):

        rlst.append(np.matmul(x, b))
        logging.info(i)
    del lst
    c = np.concatenate(rlst, 0)
    del rlst
    return c


def single_eval(score, label, na):
    """
    score: n * m
    label: n * 5
    """
    assert score.shape[0] % na == 0
    n, m = score.shape
    k = n // na
    lst = []

    for i in range(na):
        x, y = score[i * k: (i + 1) * k], label[i * k: (i + 1) * k]
        y = y + np.expand_dims(np.arange(k), 1) * m
        z = np.take(x, y)
        c = np.expand_dims(x, 2) > np.expand_dims(z, 1)
        lst.append(c.sum(1).min(1))
        del x, y, z
    ans_idx = np.concatenate(lst, 0)
    n = float(n)
    r1 = (ans_idx < 1).sum() / n
    r5 = (ans_idx < 5).sum() / n
    r10 = (ans_idx < 10).sum() / n
    mrr = (1.0 / (1.0 + ans_idx)).sum() / n
    mr = (r1 + r5 + r10) / 3.0
    return [r1, r5, r10, mr, mrr]


def read_single(filename):
    """
    read_single
    """
    text_lst, image_lst, idx_lst= [], [], []
    with open(filename) as f:
        for i in f:
            text_emb, img_emb, idx=i.strip("\n").split("\t")
            text_lst.append(parse_vec(text_emb, dilter=" "))
            image_lst.append(parse_vec(img_emb, dilter=" "))
            idx_lst.append(int(idx))
    text = np.array(text_lst, dtype=np.float32)
    image = np.array(image_lst, dtype=np.float32)
    del text_lst, image_lst
    text_label = [[i] for i in range(len(text))]
    image_label = [[i] for i in range(len(image))]
    text_label, image_label = np.array(text_label), np.array(image_label)
    return [text, image, text_label, image_label]


def eval_scores(text, image, text_label=None, image_label=None, na=10):
    """
    eval_scores with text and image
    """
    if text_label == []:
        text_label = [[i] for i in range(len(text))]
    if image_label == []:
        image_label = [[i] for i in range(len(image))]
    
    text_label, image_label = np.array(text_label), np.array(image_label)
    
    
    image = image.transpose()
    score = matmul(text, image, na)
    del text, image
    return single_eval(score, text_label, 10), single_eval(score.transpose(), image_label, 10)


if __name__ == "__main__":
    text, image, text_label, image_label = read_single(sys.argv[1])
    image = image.transpose()
    num_text, num_image = len(text_label), len(image_label)
    logging.info("calculating score")
    score = matmul(text, image, 10)
    table=[]
    del text, image

    t2i=single_eval(score, text_label, 10)
    table.append(["Text2Image"]+[round(j * 100, 2) for j in t2i][:4])
    i2t=single_eval(score.transpose(), image_label, 10)

    table.append(["Image2Text"]+[round(j * 100, 2) for j in i2t][:4])


    table.append(["MeanRecall"]+[round((t2i[i]+i2t[i])/2 * 100, 2) for i in range(len(t2i))][:4])
    print(tabulate(table, headers = ["Name", "R@1", "R@5", "R@10", "meanRecall"], tablefmt="github", floatfmt=".2f"))

