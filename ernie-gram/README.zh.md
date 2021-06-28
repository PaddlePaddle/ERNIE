[English](./README.en.md)|简体中文

`提醒`: *ERNIE-Gram* 中/英文模型已经[正式开源](#3-下载预训练模型可选)，paper 复现代码开源至 [repro分支](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-gram)。现在您可以使用基于 Paddle 2.0 全新升级、基于动静结合的新版 ERNIE 套件体验 *ERNIE-Gram* 中/英文开源模型。


## _ERNIE-Gram_: Pre-Training with Explicitly N-Gram Masked Language Modeling for Natural Language Understanding

![ERNIE-Gram](.meta/ernie-gram.jpeg)

- [模型框架](#模型框架)
- [快速上手](#快速上手)
- [安装& 使用](#安装)
  * [安装 PaddlePaddle](#1-安装-paddlepaddle)
  * [安装 ERNIE 套件](#2-安装-ernie-套件)
  * [下载预训练模型（可选）](#3-下载预训练模型可选)
  * [下载任务数据集](#4-下载数据集)
- [支持的NLP任务](#支持的-nlp-任务)
- [文献引用](#文献引用)

### 模型框架

从 **ERNIE 1.0** 起，百度研究者们就在预训练中引入**知识增强**学习，通过掩码连续的词、phrase、named entity 等语义知识单元，实现更好的预训练学习。本次开源的通用语义理解模型 **ERNIE-Gram** 更进一步，提出的**显式**、**完备**的 n-gram 掩码语言模型，实现了显式的 n-gram 语义单元知识建模。

#### ERNIE 多粒度预训练语义理解技术
 作为自然语言处理的基本语义单元，更充分的语言粒度学习能帮助模型实现更强的语义理解能力：
 - **ERNIE-Gram** 提出显式完备的 **n-gram** 多粒度掩码语言模型，同步建模 n-gram **内部**和 n-gram **之间**的语义关系，实现同时学习**细粒度（fine-grained）**和**粗粒度（coarse-grained）**语义信息
 - **ERNIE-Gram** 采用双流结构，在预训练过程中实现了单一位置多语义粒度层次预测，进一步增强了语义知识学习

**ERNIE-Gram** 多粒度预训练语义理解技术，在**预训练 (pre-training)** 阶段实现了显式的多粒度语义信号学习，在**微调 (fine-tuning)** 阶段采用 bert-style 微调方式，在不增加参数和计算复杂度的前提下，取得 **10 项**英文权威任务的 **SOTA**。在中文任务上，**ERNIE-Gram** 在包括 NLI、阅读理解等需要丰富、多层次的语义理解任务上取得公开 **SOTA**。

**ERNIE-Gram** 工作已被 **NAACL-HLT 2021** 作为长文收录，更多细节见 [link](https://arxiv.org/abs/2010.12148)。

### 快速上手
```shell
mkdir -p data
cd data
wget https://ernie-github.cdn.bcebos.com/data-xnli.tar.gz
tar xf data-xnli.tar.gz
cd ..
#demo for NLI task
sh run_cls.sh task_configs/xnli_conf
```


### 安装

##### 1. 安装 PaddlePaddle

本项目依赖 PaddlePaddle 2.0.0+， 请参考[这里](https://www.paddlepaddle.org.cn/install/quick)安装 PaddlePaddle。

##### 2. 安装 ERNIE 套件

```shell
git clone https://github.com/PaddlePaddle/ERNIE.git --depth 1
cd ERNIE
pip install -r requirements.txt
pip install -e .
```
`propeller`是辅助模型训练的高级框架，包含NLP常用的前、后处理流程。你可以通过将本repo根目录放入`PYTHONPATH`的方式导入`propeller`:
```shell
export PYTHONPATH=$PWD:$PYTHONPATH
```

##### 3. 下载预训练模型（可选）


| Model                                              | 细节参数                                                                  |下载简写|
| :------------------------------------------------- |:------------------------------------------------------------------------- |:-------|
| [ERNIE-Gram 中文](https://ernie-github.cdn.bcebos.com/model-ernie-gram-zh.1.tar.gz)           | Layer:12, Hidden:768, Heads:12  |ernie-gram|
| [ERNIE-Gram 英文](https://ernie-github.cdn.bcebos.com/model-ernie-gram-en.1.tar.gz)                  | Layer:12, Hdden:768, Heads:12   |ernie-gram-en|

##### 4. 下载数据集

请将数据目录整理成以下格式，方便使用（通过`--data_dir`参数将数据路径传入训练脚本）；

```shell
data/xnli
├── dev
│   └── 1
├── test
│   └── 1
└── train
    └── 1
```

**中文数据**

| 数据集|描述|
|:--------|:----------|
| [XNLI](https://ernie-github.cdn.bcebos.com/data-xnli.tar.gz)                 |XNLI 是由 Facebook 和纽约大学的研究者联合构建的自然语言推断数据集，包括 15 种语言的数据。我们用其中的中文数据来评估模型的语言理解能力。[链接](https://github.com/facebookresearch/XNLI)|
| [ChnSentiCorp](https://ernie-github.cdn.bcebos.com/data-chnsenticorp.tar.gz) |ChnSentiCorp 是一个中文情感分析数据集，包含酒店、笔记本电脑和书籍的网购评论。|
| [MSRA-NER](https://ernie-github.cdn.bcebos.com/data-msra_ner.tar.gz)         |MSRA-NER (SIGHAN2006) 数据集由微软亚研院发布，其目标是识别文本中具有特定意义的实体，包括人名、地名、机构名。|
| [NLPCC2016-DBQA](https://ernie-github.cdn.bcebos.com/data-dbqa.tar.gz)       |NLPCC2016-DBQA 是由国际自然语言处理和中文计算会议 NLPCC 于 2016 年举办的评测任务，其目标是从候选中找到合适的文档作为问题的答案。[链接](http://tcci.ccf.org.cn/conference/2016/dldoc/evagline2.pdf)|
|[CMRC2018](https://ernie-github.cdn.bcebos.com/data-cmrc2018.tar.gz)|CMRC2018 是中文信息学会举办的评测，评测的任务是抽取类阅读理解。[链接](https://github.com/ymcui/cmrc2018)


### 支持的 NLP 任务

使用 `动态图` 模型进行finetune:
  
  - [句对分类](./demo/finetune_classifier_distributed.py)
  - [语义匹配](./demo/finetune_classifier_distributed.py)
  - [机器阅读理解](./demo/finetune_mrc.py)


**推荐超参数设置：**

|任务|batch size|learning rate|epoch|dropout rate|
|--|--|--|--|--|
| XNLI         | 256             | 1.5e-4 | 3 | 0.1 |
| LCQMC        | 32              | 4e-5   | 2 | 0.1 |
| DRCD         | 64              | 1e-4   | 3 | 0.2 |
| CMRC2018     | 64              | 1.5e-4 | 5 | 0.2 |
| DuReader     | 64              | 1.5e-4 | 5 | 0.1 |
| MSRA-NER(SIGHAN2006) | 16      | 5e-5   | 10| 0.1 |


若希望复现 paper 中的所有实验，请切换至本 repo 的 `repro` 分支, 地址为[ernie-gram](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-gram)。

### 文献引用

```
@article{xiao2020ernie,
  title={ERNIE-Gram: Pre-Training with Explicitly N-Gram Masked Language Modeling for Natural Language Understanding},
  author={Xiao, Dongling and Li, Yu-Kun and Zhang, Han and Sun, Yu and Tian, Hao and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:2010.12148},
  year={2020}
}
```

### 讨论组
- [ERNIE官方主页](https://wenxin.baidu.com/)
- [Github Issues](https://github.com/PaddlePaddle/ERNIE/issues): bug reports, feature requests, install issues, usage issues, etc.
- QQ 群: 760439550 (ERNIE discussion group).
- QQ 2群: 958422639 (ERNIE discussion group-v2).
- [Forums](http://ai.baidu.com/forum/topic/list/168?pageNo=1): discuss implementations, research, etc.
