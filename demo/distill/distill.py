#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os

import numpy as np
from sklearn.metrics import f1_score
import paddle as P
from paddle.nn import functional as F
import propeller.paddle as propeller

from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.modeling_ernie import ErnieModelForSequenceClassification
from demo.utils import create_if_not_exists, get_warmup_and_linear_decay

# 本例子采用chnsenticorp中文情感识别任务作为示范；并且事先通过数据增强扩充了蒸馏所需的无监督数据
#
# 下载数据；并存放在 ./chnsenticorp-data/
# 数据分为3列：原文；空格切词；情感标签
# 其中第一列为ERNIE的输入；第二列为BoW词袋模型的输入
# 事先统计好的BoW 词典在 ./chnsenticorp-data/vocab.bow.txt

# 定义finetune teacher模型所需要的超参数
DATA_DIR = './chnsenticorp-data/'
SEQLEN = 256
BATCH = 32
EPOCH = 10
LR = 5e-5

tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

student_vocab = {
    i.strip(): l
    for l, i in enumerate(
        open(
            os.path.join(DATA_DIR, 'vocab.bow.txt'), encoding='utf8')
        .readlines())
}


def space_tokenizer(i):
    return i.decode('utf8').split()


feature_column = propeller.data.FeatureColumns([
    propeller.data.TextColumn(
        'seg_a',
        unk_id=tokenizer.unk_id,
        vocab_dict=tokenizer.vocab,
        tokenizer=tokenizer.tokenize),
    propeller.data.TextColumn(
        'seg_a_student',
        unk_id=student_vocab['[UNK]'],
        vocab_dict=student_vocab,
        tokenizer=space_tokenizer),
    propeller.data.LabelColumn(
        'label', vocab_dict={
            b"0": 0,
            b"1": 1,
        }),
])


def map_fn(seg_a, seg_a_student, label):
    seg_a, _ = tokenizer.truncate(seg_a, [], seqlen=SEQLEN)
    sentence, segments = tokenizer.build_for_ernie(seg_a)
    return seg_a_student, sentence, segments, label


train_ds = feature_column.build_dataset('train', data_dir=os.path.join(DATA_DIR, 'train/'), shuffle=True, repeat=False, use_gz=False) \
    .map(map_fn) \
    .padded_batch(BATCH)

train_ds_unlabel = feature_column.build_dataset('train-da', data_dir=os.path.join(DATA_DIR, 'train-data-augmented/'), shuffle=True, repeat=False, use_gz=False) \
    .map(map_fn) \
    .padded_batch(BATCH)

dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(DATA_DIR, 'dev/'), shuffle=False, repeat=False, use_gz=False) \
    .map(map_fn) \
    .padded_batch(BATCH,)

shapes = ([-1, SEQLEN], [-1, SEQLEN], [-1, SEQLEN], [-1])
types = ('int64', 'int64', 'int64', 'int64')

train_ds.data_shapes = shapes
train_ds.data_types = types
train_ds_unlabel.data_shapes = shapes
train_ds_unlabel.data_types = types
dev_ds.data_shapes = shapes
dev_ds.data_types = types

place = P.CUDAPlace(0)


def evaluate_teacher(model, dataset):
    all_pred, all_label = [], []
    with P.no_grad():
        model.eval()
        for step, (ids_student, ids, _, labels) in enumerate(
                P.io.DataLoader(
                    dataset, places=place, batch_size=None)):
            _, logits = model(ids)
            pred = logits.argmax(-1)
            all_pred.extend(pred.numpy())
            all_label.extend(labels.numpy())
        f1 = f1_score(all_label, all_pred, average='macro')
        model.train()
        return f1


teacher_model = ErnieModelForSequenceClassification.from_pretrained(
    'ernie-1.0', num_labels=2)
teacher_model.train()
if not os.path.exists('./teacher_model.bin'):
    g_clip = P.nn.ClipGradByGlobalNorm(1.0)  #experimental
    lr_scheduler = P.optimizer.lr.LambdaDecay(
        LR,
        get_warmup_and_linear_decay(9600 * EPOCH / BATCH,
                                    9600 * EPOCH * 0.1 / BATCH))

    opt = P.optimizer.AdamW(
        lr_scheduler,
        parameters=teacher_model.parameters(),
        weight_decay=0.01,
        grad_clip=g_clip)
    for epoch in range(EPOCH):
        for step, (ids_student, ids, sids, labels) in enumerate(
                P.io.DataLoader(
                    train_ds, places=place, batch_size=None)):
            loss, logits = teacher_model(ids, labels=labels)
            loss.backward()
            opt.step()
            lr_scheduler.step()
            teacher_model.clear_gradients()

            if step % 10 == 0:
                _lr = lr_scheduler.get_lr()
                _l = loss.numpy()
                msg = '[step-%d] train loss %.5f lr %.3e' % (step, _l, _lr)
                print(msg)
            if step % 100 == 0:
                f1 = evaluate_teacher(teacher_model, dev_ds)
                print('teacher f1: %.5f' % f1)
    P.save(teacher_model.state_dict(), './teacher_model.bin')
else:
    state_dict = P.load('./teacher_model.bin')
    teacher_model.set_state_dict(state_dict)
    f1 = evaluate_teacher(teacher_model, dev_ds)
    print('teacher f1: %.5f' % f1)

# 定义finetune student 模型所需要的超参数
SEQLEN = 256
BATCH = 32
EPOCH = 10
LR = 1e-4


def evaluate_student(model, dataset):
    all_pred, all_label = [], []
    with P.no_grad():
        model.eval()
        for step, (ids_student, ids, _, labels) in enumerate(
                P.io.DataLoader(
                    dataset, places=place, batch_size=None)):
            _, logits = model(ids_student)
            pred = logits.argmax(-1)
            all_pred.extend(pred.numpy())
            all_label.extend(labels.numpy())
        f1 = f1_score(all_label, all_pred, average='macro')
        model.train()
        return f1


class BOW(P.nn.Layer):
    def __init__(self):
        super().__init__()
        self.emb = P.nn.Embedding(len(student_vocab), 128, padding_idx=0)
        self.fc = P.nn.Linear(128, 2)

    def forward(self, ids, labels=None):
        embbed = self.emb(ids)
        pad_mask = (ids != 0).cast('float32').unsqueeze(-1)

        embbed = (embbed * pad_mask).sum(1)
        embbed = F.softsign(embbed)
        logits = self.fc(embbed)
        if labels is not None:
            if len(labels.shape) == 1:
                labels = labels.reshape([-1, 1])
            loss = F.cross_entropy(logits, labels).mean()
        else:
            loss = None
        return loss, logits


class CNN(P.nn.Layer):
    def __init__(self):
        super().__init__()
        self.emb = P.nn.Embedding(30002, 128, padding_idx=0)
        self.cnn = P.nn.Conv2D(128, 128, (1, 3), padding=(0, 1), act='relu')
        self.pool = P.nn.Pool2D((1, 3), pool_padding=(0, 1))
        self.fc = P.nn.Linear(128, 2)

    def forward(self, ids, labels=None):
        embbed = self.emb(ids)
        #d_batch, d_seqlen = ids.shape
        hidden = embbed
        hidden = hidden.transpose([0, 2, 1]).unsqueeze(2)  #change to NCWH
        hidden = self.cnn(hidden)
        hidden = self.pool(hidden).squeeze(2).transpose([0, 2, 1])
        pad_mask = (ids != 0).cast('float32').unsqueeze(-1)
        hidden = P.nn.funcional.softsign(L(hidden * pad_mask).sum(1))
        logits = self.fc(hidden)
        if labels is not None:
            if len(labels.shape) == 1:
                labels = labels.reshape([-1, 1])
            loss = F.cross_entropy(logits, labels).mean()
        else:
            loss = None
        return loss, logits


def KL(pred, target):
    pred = F.log_softmax(pred)
    target = F.softmax(target)
    loss = F.kl_div(pred, target)
    return loss


teacher_model.eval()
model = BOW()
g_clip = P.nn.ClipGradByGlobalNorm(1.0)  #experimental

lr_scheduler = P.optimizer.lr.LambdaDecay(
    LR,
    get_warmup_and_linear_decay(9600 * EPOCH / BATCH,
                                9600 * EPOCH * 0.1 / BATCH))

opt = P.optimizer.AdamW(
    lr_scheduler,
    parameters=model.parameters(),
    weight_decay=0.01,
    grad_clip=g_clip)
model.train()

for epoch in range(EPOCH - 1):
    for step, (
            ids_student, ids, sids, label
    ) in enumerate(P.io.DataLoader(
            train_ds, places=place, batch_size=None)):
        with P.no_grad():
            _, logits_t = teacher_model(ids, sids)  # teacher 模型输出logits
        _, logits_s = model(ids_student)  # student 模型输出logits
        loss_ce, _ = model(ids_student, labels=label)
        loss_kd = KL(logits_s, logits_t.detach())  # 由KL divergence度量两个分布的距离
        loss = loss_ce + loss_kd
        loss.backward()
        opt.step()
        lr_scheduler.step()
        model.clear_gradients()
        if step % 10 == 0:
            _lr = lr_scheduler.get_lr()
            _l = loss.numpy()
            msg = '[step-%d] train loss %.5f lr %.3e' % (step, _l, _lr)
            print(msg)

    f1 = evaluate_student(model, dev_ds)
    print('student f1 %.5f' % f1)

# 最后再加一轮hard label训练巩固结果
for step, (
        ids_student, ids, sids, label
) in enumerate(P.io.DataLoader(
        train_ds, places=place, batch_size=None)):
    loss, _ = model(ids_student, labels=label)
    loss.backward()
    opt.step()
    model.clear_gradients()
    if step % 10 == 0:
        _lr = lr_scheduler.get_lr()
        _l = loss.numpy()
        msg = '[step-%d] train loss %.5f lr %.3e' % (step, _l, _lr)
        print(msg)

f1 = evaluate_student(model, dev_ds)
print('final f1 %.5f' % f1)
