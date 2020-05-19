import sys
import json
import random

pred_fin = open(sys.argv[1])
info_fin = open(sys.argv[2])

data = []

def clean_str(text):
    text = text.replace(" ' s ", " 's ")
    text = text.replace(" ' ll ", " 'll ")
    text = text.replace(" ' re ", " 're ")
    text = text.replace("n ' t ", " n't ")
    return text


for pred in pred_fin:
    answer = clean_str(pred.strip())
    id, turn_id = info_fin.readline().strip().split("\t")
    data.append({"id": id, "turn_id": int(turn_id), "answer": answer})


fo = open(sys.argv[3], "w")
json.dump(data, fo, indent=4)
