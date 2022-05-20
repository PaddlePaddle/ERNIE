#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A tool to convert DuIE2.0 dataset """
import json 
import os

def mkdir(path):
    """ mkdir """
    path = path.strip()
    path = path.rstrip("/")
    path = path.split("/")
    path = "/".join(path[:-1])
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
 

def convert_data(ip_fil, op_fil):
    """ convert train and dev datasets """
    mkdir(op_fil)
    op_f = open(op_fil, 'w')
    with open(ip_fil, 'r') as f: 
        for line in f:
            data = json.loads(line)
            text = data['text']
            spo_list = data['spo_list']
           

            op_data={
                "text": text,
                "spo_list": []
            }
            for pre in spo_list:
                predicate_name = pre['predicate']
                subject_name = pre['subject']
                object_name = pre['object']['@value']
                try:
                    subject_id = [text.index(subject_name), text.index(subject_name)+len(subject_name)]
                except ValueError:
                    print(data)
                    print(text, '*', subject_name)
                    continue
                object_id = [text.index(object_name), text.index(object_name)+len(object_name)]
                op_data["spo_list"].append(
                    {
                        "predicate": predicate_name,
                        "subject": subject_id,
                        "object": object_id
                    }
                )

            if len(op_data["spo_list"]) == 0:
                print(op_data)
                continue
            op_data = json.dumps(op_data, ensure_ascii=False)

            op_f.write(op_data + '\n')
    op_f.close()

    
def conver_label_map(ip_fil, op_fil): 
    """ convert label map """
    mkdir(op_fil)
    op_f = open(op_fil, 'w')
    data_lis=['O', 'I']
    with open(ip_fil, 'r') as f: 
        for i, line in enumerate(f): 
            data = json.loads(line)
            predicate = data['predicate']
            data_lis.append('B-' + predicate + '@S')
            data_lis.append('B-' + predicate + '@O')
    data_dict = {}
    for i, k in enumerate(data_lis):
        data_dict[k]=i
    json.dump(data_dict, op_f, ensure_ascii=False, indent=4)

    op_f.close()


def main():
    """ main function """
    fil = ["duie_dev.json/duie_dev.json", "duie_train.json/duie_train.json"]
    save = ["dev_data/dev.json", "train_data/train.json"]

    n = len(fil)

    schema = "duie_schema/duie_schema.json"
    label_map = "label_map/label_map.json"

    for i in range(n):
        convert_data(fil[i], save[i])
    conver_label_map(schema, label_map)

if __name__ == '__main__':

    main()