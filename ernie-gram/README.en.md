English|[简体中文](./README.zh.md)

`Remind`: *ERNIE-Gram* model has been officially released in [here](#3-download-pretrained-models-optional). Our reproduction codes have been released to [repro branch](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-gram).


## _ERNIE-Gram_: Pre-Training with Explicitly N-Gram Masked Language Modeling for Natural Language Understanding

![ERNIE-Gram](.meta/ernie-gram.jpeg)

- [Framework](#ernie-gram-framework)
- [Quick Tour](#quick-tour)
- [Setup](#setup)
    * [Install PaddlePaddle](#1-install-paddlepaddle)
    * [Install ERNIE Kit](#2-install-ernie-kit)
    * [Download pre-trained models](#3-download-pretrained-models-optional)
    * [Download datasets](#4-download-datasets)
- [Fine-tuning](#fine-tuning)
- [Citation](#citation)

### ERNIE-Gram Framework

Since **ERNIE 1.0**, Baidu researchers have introduced **knowledge-enhanced representation learning** in pre-training to achieve better pre-training learning by masking consecutive words, phrases, named entities, and other semantic knowledge units. Furthermore, we propose **ERNIE-Gram**, an explicitly n-gram masking language model to enhance the integration of coarse-grained information for pre-training. In **ERNIE-Gram**, **n-grams** are masked and predicted directly using **explicit** n-gram identities rather than contiguous sequences of tokens.

In downstream tasks, **ERNIE-gram** uses a `bert-style` fine-tuning approach, thus maintaining the same parameter size and computational complexity.

We pre-train **ERNIE-Gram** on `English` and `Chinese` text corpora and fine-tune on `19` downstream tasks. Experimental results show that **ERNIE-Gram** outperforms previous pre-training models like *XLNet* and *RoBERTa* by a large margin, and achieves comparable results with state-of-the-art methods.

The **ERNIE-Gram** paper has been accepted for **NAACL-HLT 2021**, for more details please see in [here](https://arxiv.org/abs/2010.12148).

### Quick Tour

```shell
mkdir -p data
cd data
wget https://ernie-github.cdn.bcebos.com/data-xnli.tar.gz
tar xf data-xnli.tar.gz
cd ..
#demo for NLI task
sh run_cls.sh task_configs/xnli_conf
```

### Setup

##### 1. Install PaddlePaddle

This repo requires PaddlePaddle 2.0.0+, please see [here](https://www.paddlepaddle.org.cn/install/quick) for installaton instruction.

##### 2. Install ERNIE Kit

```shell
git clone https://github.com/PaddlePaddle/ERNIE.git --depth 1
cd ERNIE
pip install -r requirements.txt
pip install -e .
```

##### 3. Download pretrained models (optional)

| Model                                              | Description                                                  |abbreviation|
| :------------------------------------------------- | :----------------------------------------------------------- |:-----------|
| [ERNIE-Gram Base for Chinese](https://ernie-github.cdn.bcebos.com/model-ernie-gram-zh.1.tar.gz) | Layer:12, Hidden:768, Heads:12 | ernie-gram|
| [ERNIE-Gram Base for English](https://ernie-github.cdn.bcebos.com/model-ernie-gram-en.1.tar.gz) | Layer:12, Hidden:768, Heads:12 | ernie-gram-en |

##### 4. Download datasets

**English Datasets**

Download the [GLUE datasets](https://gluebenchmark.com/tasks) by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)

the `--data_dir` option in the following section assumes a directory tree like this:

```shell
data/xnli
├── dev
│   └── 1
├── test
│   └── 1
└── train
    └── 1
```

see [demo](https://ernie-github.cdn.bcebos.com/data-mnli-m.tar.gz) data for MNLI task.

### Fine-tuning

try eager execution with `dygraph model` :

  - [Natural Language Inference](./demo/finetune_classifier_distributed.py)
  - [Sentiment Analysis](./demo/finetune_sentiment_analysis.py)
  - [Semantic Similarity](./demo/finetune_classifier.py)
  - [Name Entity Recognition(NER)](./demo/finetune_ner.py)
  - [Machine Reading Comprehension](./demo/finetune_mrc.py)


**recomended hyper parameters:**

 - See **ERNIE-Gram** paper [Appendix B.1-4](https://arxiv.org/abs/2010.12148)

For full reproduction of paper results, please checkout to `repro` branch of this repo, the site is at [ernie-gram](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-gram).

# Citation

```
@article{xiao2020ernie,
  title={ERNIE-Gram: Pre-Training with Explicitly N-Gram Masked Language Modeling for Natural Language Understanding},
  author={Xiao, Dongling and Li, Yu-Kun and Zhang, Han and Sun, Yu and Tian, Hao and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:2010.12148},
  year={2020}
}
```

### Communication

- [ERNIE homepage](https://wenxin.baidu.com/)
- [Github Issues](https://github.com/PaddlePaddle/ERNIE/issues): bug reports, feature requests, install issues, usage issues, etc.
- QQ discussion group: 760439550 (ERNIE discussion group).
- QQ discussion group: 958422639 (ERNIE discussion group-v2).
- [Forums](http://ai.baidu.com/forum/topic/list/168?pageNo=1): discuss implementations, research, etc.
