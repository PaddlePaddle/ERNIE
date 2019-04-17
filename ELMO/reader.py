#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np

Py3 = sys.version_info[0] == 3


def listDir(rootDir):
    res = []
    for filename in os.listdir(rootDir):
        pathname = os.path.join(rootDir, filename)
        if (os.path.isfile(pathname)):
            res.append(pathname)
    return res


_unk = -1
_bos = -1
_eos = -1


def _read_words(filename):
    data = []
    with open(filename, "r") as f:
        return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))

    print("vocab word num", len(words))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _load_vocab(filename):
    with open(filename, "r") as f:
        words = f.read().decode("utf-8").replace("\n", " ").split()
        word_to_id = dict(zip(words, range(len(words))))
        _unk = word_to_id['<S>']
        _eos = word_to_id['</S>']
        _unk = word_to_id['<UNK>']
        return word_to_id


def _file_to_word_ids(filenames, word_to_id):
    for filename in filenames:
        data = _read_words(filename)
        for id in [word_to_id[word] for word in data if word in word_to_id]:
            yield id


def ptb_raw_data(data_path=None, vocab_path=None, args=None):
    """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """
    if vocab_path:
        word_to_id = _load_vocab(vocab_path)

    if not args.train_path:
        train_path = os.path.join(data_path, "train")
        train_data = _file_to_word_ids(listDir(train_path), word_to_id)
    else:
        train_path = args.train_path
        train_data = _file_to_word_ids([train_path], word_to_id)
    valid_path = os.path.join(data_path, "dev")
    test_path = os.path.join(data_path, "dev")
    valid_data = _file_to_word_ids(listDir(valid_path), word_to_id)
    test_data = _file_to_word_ids(listDir(test_path), word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


def get_data_iter(raw_data, batch_size, num_steps):
    def __impl__():
        buf = []
        while True:
            if len(buf) >= num_steps * batch_size + 1:
                x = np.asarray(
                    buf[:-1], dtype='int64').reshape((batch_size, num_steps))
                y = np.asarray(
                    buf[1:], dtype='int64').reshape((batch_size, num_steps))
                yield (x, y)
                buf = [buf[-1]]
            try:
                buf.append(raw_data.next())
            except StopIteration:
                break

    return __impl__
