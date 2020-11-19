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
import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D
import propeller.paddle as propeller

from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.modeling_ernie import ErnieModelForSequenceClassification
from ernie.optimization import AdamW, LinearDecay

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
        open(os.path.join(DATA_DIR, 'vocab.bow.txt')).readlines())
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

place = F.CUDAPlace(0)
D.guard(place).__enter__()


def evaluate_teacher(model, dataset):
    all_pred, all_label = [], []
    with D.base._switch_tracer_mode_guard_(is_train=False):
        model.eval()
        for step, (ids_student, ids, _, labels) in enumerate(dataset.start()):
            _, logits = model(ids)
            pred = L.argmax(logits, -1)
            all_pred.extend(pred.numpy())
            all_label.extend(labels.numpy())
        f1 = f1_score(all_label, all_pred, average='macro')
        model.train()
        return f1


teacher_model = ErnieModelForSequenceClassification.from_pretrained(
    'ernie-1.0', num_labels=2)
teacher_model.train()
if not os.path.exists('./teacher_model.pdparams'):
    g_clip = F.clip.GradientClipByGlobalNorm(1.0)
    opt = AdamW(
        learning_rate=LinearDecay(LR, 9600 * EPOCH * 0.1 / BATCH,
                                  9600 * EPOCH / BATCH),
        parameter_list=teacher_model.parameters(),
        weight_decay=0.01,
        grad_clip=g_clip)
    for epoch in range(EPOCH):
        for step, (ids_student, ids, sids,
                   labels) in enumerate(train_ds.start(place)):
            loss, logits = teacher_model(ids, labels=labels)
            loss.backward()
            if step % 10 == 0:
                print('[step %03d] teacher train loss %.5f lr %.3e' %
                      (step, loss.numpy(), opt.current_step_lr()))
            opt.minimize(loss)
            teacher_model.clear_gradients()
            if step % 100 == 0:
                f1 = evaluate_teacher(teacher_model, dev_ds)
                print('teacher f1: %.5f' % f1)
    D.save_dygraph(teacher_model.state_dict(), './teacher_model')
else:
    state_dict, _ = D.load_dygraph('./teacher_model')
    teacher_model.set_dict(state_dict)
    f1 = evaluate_teacher(teacher_model, dev_ds)
    print('teacher f1: %.5f' % f1)

# 定义finetune student 模型所需要的超参数
SEQLEN = 256
BATCH = 100
EPOCH = 10
LR = 1e-4


def evaluate_student(model, dataset):
    all_pred, all_label = [], []
    with D.base._switch_tracer_mode_guard_(is_train=False):
        model.eval()
        for step, (ids_student, ids, _, labels) in enumerate(dataset.start()):
            _, logits = model(ids_student)
            pred = L.argmax(logits, -1)
            all_pred.extend(pred.numpy())
            all_label.extend(labels.numpy())
        f1 = f1_score(all_label, all_pred, average='macro')
        model.train()
        return f1


class BOW(D.Layer):
    def __init__(self):
        super().__init__()
        self.emb = D.Embedding([len(student_vocab), 128], padding_idx=0)
        self.fc = D.Linear(128, 2)

    def forward(self, ids, labels=None):
        embbed = self.emb(ids)
        pad_mask = L.unsqueeze(L.cast(ids != 0, 'float32'), [-1])

        embbed = L.reduce_sum(embbed * pad_mask, 1)
        embbed = L.softsign(embbed)
        logits = self.fc(embbed)
        if labels is not None:
            if len(labels.shape) == 1:
                labels = L.reshape(labels, [-1, 1])
            loss = L.softmax_with_cross_entropy(logits, labels)
            loss = L.reduce_mean(loss)
        else:
            loss = None
        return loss, logits


class CNN(D.Layer):
    def __init__(self):
        super().__init__()
        self.emb = D.Embedding([30002, 128], padding_idx=0)
        self.cnn = D.Conv2D(128, 128, (1, 3), padding=(0, 1), act='relu')
        self.pool = D.Pool2D((1, 3), pool_padding=(0, 1))
        self.fc = D.Linear(128, 2)

    def forward(self, ids, labels=None):
        embbed = self.emb(ids)
        #d_batch, d_seqlen = ids.shape
        hidden = embbed
        hidden = L.transpose(hidden, [0, 2, 1])  #change to NCWH
        hidden = L.unsqueeze(hidden, [2])
        hidden = self.cnn(hidden)
        hidden = self.pool(hidden)
        hidden = L.squeeze(hidden, [2])
        hidden = L.transpose(hidden, [0, 2, 1])
        pad_mask = L.unsqueeze(L.cast(ids != 0, 'float32'), [-1])
        hidden = L.softsign(L.reduce_sum(hidden * pad_mask, 1))
        logits = self.fc(hidden)
        if labels is not None:
            if len(labels.shape) == 1:
                labels = L.reshape(labels, [-1, 1])
            loss = L.softmax_with_cross_entropy(logits, labels)
            loss = L.reduce_mean(loss)
        else:
            loss = None
        return loss, logits


def KL(pred, target):
    pred = L.log(L.softmax(pred))
    target = L.softmax(target)
    loss = L.kldiv_loss(pred, target)
    return loss


teacher_model.eval()
model = BOW()
g_clip = F.clip.GradientClipByGlobalNorm(1.0)  #experimental
opt = AdamW(
    learning_rate=LR,
    parameter_list=model.parameters(),
    weight_decay=0.01,
    grad_clip=g_clip)
model.train()
for epoch in range(EPOCH):
    for step, (ids_student, ids, sids,
               label) in enumerate(train_ds.start(place)):
        _, logits_t = teacher_model(ids, sids)  # teacher 模型输出logits
        logits_t.stop_gradient = True
        _, logits_s = model(ids_student)  # student 模型输出logits
        loss_ce, _ = model(ids_student, labels=label)
        loss_kd = KL(logits_s, logits_t)  # 由KL divergence度量两个分布的距离
        loss = loss_ce + loss_kd
        loss.backward()
        if step % 10 == 0:
            print('[step %03d] distill train loss %.5f lr %.3e' %
                  (step, loss.numpy(), opt.current_step_lr()))
        opt.minimize(loss)
        model.clear_gradients()
    f1 = evaluate_student(model, dev_ds)
    print('student f1 %.5f' % f1)

# 最后再加一轮hard label训练巩固结果
for step, (ids_student, ids, sids, label) in enumerate(train_ds.start(place)):
    loss, _ = model(ids_student, labels=label)
    loss.backward()
    if step % 10 == 0:
        print('[step %03d] train loss %.5f lr %.3e' %
              (step, loss.numpy(), opt.current_step_lr()))
    opt.minimize(loss)
    model.clear_gradients()

f1 = evaluate_student(model, dev_ds)
print('final f1 %.5f' % f1)
