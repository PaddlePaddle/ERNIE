import logging
import os
import numpy as np
from collections import namedtuple
from tqdm import tqdm


def read_files(dir_path):
    """
    :param dir_path
    """
    examples = []
    Example = namedtuple('Example', ['qid', 'text_a', 'label', 'score'])

    def _read_files(dir_p, label):
        logging.info('loading data from %s' % dir_p)
        data_files = os.listdir(dir_p)
        desc = "loading " + dir_p
        for f_idx, data_file in tqdm(enumerate(data_files), desc=desc):
            file_path = os.path.join(dir_p, data_file)
            qid, score = data_file.split('_')
            score = score.split('.')[0]
            with open(file_path, 'r') as f:
                doc = []
                for line in f:
                    line = line.strip().replace('<br /><br />', ' ')
                    doc.append(line)
                doc_text = ' '.join(doc)
                example = Example(
                    qid=len(examples)+1,
                    text_a=doc_text,
                    label=label,
                    score=score
                )
                examples.append(example)
    
    neg_dir = os.path.join(dir_path, 'neg')
    pos_dir = os.path.join(dir_path, 'pos')
    _read_files(neg_dir, label=0) 
    _read_files(pos_dir, label=1)  
    logging.info('loading data finished')
    return examples

def write_to_one(dir, o_file_name):
    exampels = read_files(dir)
    logging.info('ex nums:%d' % (len(exampels)))
    with open(o_file_name, 'w') as fout:
        fout.write("qid\tlabel\tscore\ttext_a\n")
        for ex in exampels:
            try:
                fout.write("{}\t{}\t{}\t{}\n".format(ex.qid, ex.label, ex.score, ex.text_a.replace('\t', '')))
            except Exception as e:
                print(ex.qid, ex.text_a, ex.label, ex.score)
                raise e

if __name__ == "__main__":
    write_to_one("train", 'train.txt')
    write_to_one("test", "test.txt")




