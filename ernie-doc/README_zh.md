[English](./README.md) | 简体中文

## _ERNIE-Doc_: A Retrospective Long-Document Modeling Transformer

- [模型框架](#模型框架)
- [预训练模型](#预训练模型)
- [下游任务](#下游任务)
  * [语言建模](#语言建模)
  * [篇章级分类](#篇章级分类)
  * [阅读理解](#阅读理解)
  * [信息抽取](#信息抽取)
  * [语义匹配](#语义匹配)
- [使用说明](#使用说明)
  * [安装飞桨](#安装飞桨)
  * [运行微调](#运行微调)
- [引用](#引用)

关于算法的详细描述，请参见我们的论文：
>[_**ERNIE-Doc: A Retrospective Long-Document Modeling Transformer**_](https://arxiv.org/abs/2012.15688)
>
>Siyu Ding\*, Junyuan Shang\*, Shuohuan Wang, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang (\* : equal contribution)
>
>Preprint December 2020
>
>Accepted by **ACL-2021**

![ERNIE-Doc](https://img.shields.io/badge/预训练-长文本建模-green) ![paper](https://img.shields.io/badge/论文-ACL2021-yellow) 

---
**ERNIE-Doc 是面向篇章级长文本建模的预训练-微调框架**，ERNIE-Doc 受到人类先粗读后精读的阅读方式启发，提出了**回顾式建模机制**和**增强记忆机制**，突破了 Transformer 在文本长度上的建模瓶颈。ERNIE-Doc 在业界首次实现了全篇章级无限长文本的双向建模，在包括阅读理解、信息抽取、篇章分类、语言模型在内的13个权威中英文长文本语言理解任务上取得了SOTA效果。

## 模型框架

我们提出了三种方法解决长文本建模问题:

- **回顾式建模机制（Retrospective Feed Mechanism）**:  通过将文本片段重复输入两次，使得在回顾阶段的每一个文本片段可以双向建模并利用在粗读阶段获取到的全篇章语义信息。
- **增强记忆机制（Enhanced Recurrence Mechansim）**: 通过改进Recurrence Transformer模型（Transformer-XL等）为同层循环的方式，可建模无限长文本。
- **段落重排序目标（Segment-reordering Objective）**:  该预训练目标通过让模型学习篇章级本文段落间的关系，使得模型可以对篇章整体信息进行建模。

下图展示了ERNIE-Doc 与Recurrence Transformer在3层网络，4个片段输入情况下的建模方式与建模长度的对比。

![framework](.meta/framework.png)

## 预训练模型

我们发布了 **ERNIE-Doc _base_** 中英文模型和 **ERNIE-Doc _large_** 英文模型。 

- [**ERNIE-Doc _base_en_**](https://ernie-github.cdn.bcebos.com/model-ernie-doc-base-en.tar.gz) (_12-layer, 768-hidden, 12-heads_)
- [**ERNIE-Doc _base_zh_**](https://ernie-github.cdn.bcebos.com/model-ernie-doc-base-zh.tar.gz) (_12-layer, 768-hidden, 12-heads_)
- [**ERNIE-Doc _large_en_**](https://ernie-github.cdn.bcebos.com/model-ernie-doc-large-en.tar.gz) (_24-layer, 1024-hidden, 16-heads_)


## 下游任务

我们在语言建模、篇章级分类、阅读理解以及信息抽取等任务上选取了广泛使用的数据集进行模型效果验证，并且与当前效果最优的模型（[Longformer](https://arxiv.org/abs/2004.05150)、[BigBird](https://arxiv.org/abs/2007.14062)、[ETC](https://arxiv.org/abs/2004.08483)、[ERNIE2.0](https://arxiv.org/abs/1907.12412)等）进行对比。

### 语言建模

- [WikiText-103](https://arxiv.org/abs/1609.07843)

| 模型                   | Param. | PPL  |
|--------------------------|:--------:|:------:|
| _Results of base models_   |        |      |
| LSTM                     |    -   | 48.7 |
| LSTM+Neural cache        |    -   | 40.8 |
| GCNN-14                  |    -   | 37.2 |
| QRNN                     |  151M  | 33.0 |
| Transformer-XL Base      |  151M  | 24.0 |
| SegaTransformer-XL Base  |  151M  | 22.5 |
| **ERNIE-Doc** Base           |  151M  | **21.0** |
| _Results of large models_  |        |      |
| Adaptive Input           |  247M  | 18.7 |
| Transformer-XL Large     |  247M  | 18.3 |
| Compressive Transformer  |  247M  | 17.1 |
| SegaTransformer-XL Large |  247M  | 17.1 |
| **ERNIE-Doc** Large          |  247M  | **16.8** |

### 篇章级分类

- [IMDB reviews](http://ai.stanford.edu/~amaas/data/sentiment/index.html)

| 模型          | Acc. | F1 | 
|-----------------|:----:|:----:|
| RoBERTa         | 95.3 | 95.0 | 
| Longformer      | 95.7 |   -  | 
| BigBird         |   -  | 95.2 |
| **ERNIE-Doc** Base  | **96.1** | **96.1** |
| XLNet-Large     | 96.8 |   -  |   -  |
| **ERNIE-Doc** Large | **97.1** | **97.1** | 

- [Hyperpartisan News Dection](https://pan.webis.de/semeval19/semeval19-web/)

| 模型          | F1 |
|-----------------|:----:|
| RoBERTa         | 87.8 | 
| Longformer      | 94.8 |   
| BigBird         |  92.2  | 
| **ERNIE-Doc** Base  |  **96.3** | 
| **ERNIE-Doc** Large | **96.6** | 

- [THUCNews(THU)](http://thuctc.thunlp.org/)、[IFLYTEK(IFK)](https://arxiv.org/abs/2004.05986)

| 模型          |    THU   |    THU   |    IFK   |
|-----------------|:--------:|:--------:|:--------:|
|                 |   Acc.   |   Acc.   |   Acc.   |
|                 |    Dev   | Test     |    Dev   |
| BERT            |   97.7   |   97.3   |   60.3   |
| BERT-wwm-ext    |   97.6   |   97.6   |   59.4   |
| RoBERTa-wwm-ext |     -    |     -    |   60.3   |
| ERNIE 1.0       |   97.7   |   97.3   |   59.0   |
| ERNIE 2.0       |   98.0   |   97.5   |   61.7   |
| **ERNIE-Doc**       | **98.3** | **97.7** | **62.4** |

### 阅读理解

- [TriviaQA](http://nlp.cs.washington.edu/triviaqa/)验证集效果

| 模型          | F1 |
|-----------------|:----:|
| RoBERTa         | 74.3 | 
| Longformer      | 75.2 |   
| BigBird         |  79.5 | 
| **ERNIE-Doc** Base  |  **80.1** | 
| Longformer Large  |  77.8 | 
|   BigBird Large  |  - | 
| **ERNIE-Doc** Large | **82.5** | 

- [HotpotQA](https://hotpotqa.github.io/)验证集效果

| 模型          | Span-F1 | Supp.-F1 | Joint-F1 |
|-----------------|:----:|:----:|:----:|
| RoBERTa         | 73.5 | 83.4 | 63.5 | 
| Longformer      | 74.3 |  84.4 | 64.4 |  
| BigBird         |  75.5 | **87.1** | 67.8 | 
| **ERNIE-Doc** Base  |  **79.4** | 86.3 | **70.5** | 
| Longformer Large  |  81.0 | 85.8 | 71.4 | 
|   BigBird Large  |  81.3 | **89.4** | - | 
| **ERNIE-Doc** Large | **82.2** | 87.6 | **73.7** | 

- [DRCD](https://arxiv.org/abs/1806.00920)、[CMRC2018](https://arxiv.org/abs/1810.07366)、[DuReader](https://arxiv.org/abs/1711.05073)、[C3](https://arxiv.org/abs/1904.09679)

| 模型            | DRCD          | DRCD          | CMRC2018      | DuReader      | C3       | C3       |
|-----------------|---------------|---------------|---------------|---------------|----------|----------|
|                 | dev           | test          | dev           | dev           | dev      | test     |
|                 | EM/F1         | EM/F1         | EM/F1         | EM/F1         | Acc.     | Acc.     |
| BERT            | 85.7/91.6     | 84.9/90.9     | 66.3/85.9     | 59.5/73.1     |   65.7   |   64.5   |
| BERT-wwm-ext    | 85.0/91.2     | 83.6/90.4     | 67.1/85.7     | -/-           |   67.8   |   68.5   |
| RoBERTa-wwm-ext | 86.6/92.5     | 85.2/92.0     | 67.4/87.2     | -/-           |   67.1   |   66.5   |
| MacBERT         | 88.3/93.5     | 87.9/93.2     | 69.5/87.7     | -/-           |     -    |     -    |
| XLNet-zh        | 83.2/92.0     | 82.8/91.8     | 63.0/85.9     | -/-           |     -    |     -    |
| ERNIE 1.0       | 84.6/90.9     | 84.0/90.5     | 65.1/85.1     | 57.9/72/1     |   65.5   |   64.1   |
| ERNIE 2.0       | 88.5/93.8     | 88.0/93.4     | 69.1/88.6     | 61.3/74.9     |   72.3   |   73.2   |
| **ERNIE-Doc**   | **90.5/95.2** | **90.5/95.1** | **76.1/91.6** | **65.8/77.9** | **76.5** | **76.5** |

### 信息抽取

- [Open Domain Web Keyphrase Extraction (OpenKP)](https://www.aclweb.org/anthology/D19-1521/)

| 模型    | F1@1 | F1@3 | F1@5 |
|-----------|:----:|:----:|:----:|
| BLING-KPE | 26.7 | 29.2 | 20.9 |
| JointKPE  | 39.1 | 39.8 | 33.8 |
| ETC       |   -  | 40.2 |   -  |
| ERNIE-Doc | **40.2** | **40.5** | **34.4** |

### 语义匹配

- [CAIL2019-SCM](https://arxiv.org/abs/1911.08962)

| 模型    |      Dev (Acc.)     |     Test  (Acc.)       |
|-----------|:-------------:|:-------------:|
| BERT      |      61.9     |      67.3     |
| ERNIE 2.0 |      64.9     |      67.9     |
| ERNIE-Doc | **65.6** | **68.8** |


## 使用说明

### 安装飞桨

我们的代码基于 Paddle(version>=2.0)，推荐使用python3运行。 ERNIE-Doc 依赖的其他模块也列举在 `requirements.txt`，可以通过下面的指令安装:
```script
pip install -r requirements.txt
```

### 运行微调
我们开源了中英文分类任务以及中文阅读理解任务的微调代码，运行以下脚本即可进行实验
```shell
sh script/run_imdb.sh # 英文分类任务 
sh script/run_iflytek.sh # 中文分类任务
sh script/run_dureader.sh # 中文阅读理解任务
```
[imdb数据处理说明](./data/imdb/README.md)


具体微调参数均可在上述脚本中进行修改，训练和评估的日志在 `log/job.log.0`。 

**注意**: 训练时实际的 batch size 等于 `配置的 batch size * GPU 卡数`。


## 引用

可以按下面的格式引用我们的论文:

```
@article{ding2020ernie,
  title={ERNIE-DOC: The Retrospective Long-Document Modeling Transformer},
  author={Ding, Siyu and Shang, Junyuan and Wang, Shuohuan and Sun, Yu and Tian, Hao and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:2012.15688},
  year={2020}
}
```