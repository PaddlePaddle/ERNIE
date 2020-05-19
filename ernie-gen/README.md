English | [简体中文](./README.zh.md)

## _ERNIE-GEN_: An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation

- [Proposed Generation Framework](#proposed-generation-framework)
- [Pre-trained Models](#pre-trained-models)
- [Fine-tuning on Downstream Tasks](#fine-tuning-on-downstream-tasks)
  * [Abstractive Summarization](#abstractive-summarization)
  * [Question Generation](#question-generation)
  * [Generative Dialogue Response](#generative-dialogue-response)
  * [Generative Question Answering](#generative-question-answering)
- [Usage](#usage)
  * [Install PaddlePaddle](#install-paddlepaddle)
  * [Fine-tuning](#fine-tuning)
  * [Employ Dynamic Computation Graph](#employ-dynamic-computation-graph)
  * [The ERNIE 1.0 is avaliable](#the-ernie-10-is-avaliable-for-chinese-generation-tasks)
- [Citation](#citation)

For technical description of the algorithm, please see our paper:
>[_**ERNIE-GEN:An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation**_](https://arxiv.org/abs/2001.11314.pdf)
>
>Dongling Xiao\*, Han Zhang\*, Yukun Li, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang (\* : equal contribution)
>
>Preprint January 2020
>
>Accepted by **IJCAI-2020**

![ERNIE-GEN](https://img.shields.io/badge/Pretraining-Generation-green) ![Gigaword](https://img.shields.io/badge/Abstractive%20Summarization-Gigaword-yellow) ![Gigaword](https://img.shields.io/badge/Abstractive%20Summarization-CNN/Daily%20Mail-blue) ![SQuAD](https://img.shields.io/badge/Question%20Generation-SQuAD-green) ![Personal-Chat](https://img.shields.io/badge/Dialogue%20Response-Personal%20Chat-yellowgreen) ![CoQA](https://img.shields.io/badge/Generative%20Question%20Answering-CoQA-orange) 
---
**[ERNIE-GEN](https://arxiv.org/abs/2001.11314.pdf) is a multi-flow language generation framework for both pre-training and fine-tuning.** We propose a novel **span-by-span generation** pre-training task to enable the model to **generate a semantically-complete span** at each step rather than a word, in light of the fact that entities, phrases in human writing are organized in a coherent manner. An **infilling generation mechanism** and a **noise-aware generation method** are incorporated into both pre-training and fine-tuning to alleviate **the problem of exposure bias**. In the pre-training phase, ERNIE-GEN adopts a **multi-granularity target fragments sampling** strategy to force decoder to rely more on the encoder representations other than the previous generated words to enhancing the correlation between encoder and decoder.

## Proposed Generation Framework

We construct three novel methods to enhance the language generation ability:

- **Span-by-span Generation Pre-training Task**:  to enable model to generate a semantically-complete span at each step rather than a word.
- **Infilling Genration and Noise-aware Generation**:  to alleviate the problem of exposure bias.
- **Multi-Granularity Target Fragments**:  to enhance the correlation between encoder and decoder during pre-training.

Specifically, the span-by-span generation task and word-by-word generation task based on infilling generation mechanism are impemented by a carefully designed **Multi-Flow Attention** architecture as shown below.

![multi-flow-attention](.meta/multi-flow-attention.png)

## Pre-trained Models

We release the checkpoints for **ERNIE-GEN _base_** model and **ERNIE-GEN _large_** model which are both pre-trained on English Wikipedia and [BookCorpus](https://arxiv.org/abs/1506.06724) (totally 16GB). Besides, **ERNIE-GEN _large_** pre-trained on the 160GB corpus (used by [RoBERTa](https://arxiv.org/abs/1907.11692) and [BART](https://arxiv.org/abs/1910.13461)) is available as well.

- [**ERNIE-GEN _base_**](https://ernie.bj.bcebos.com/ernie_gen_base.tgz) (_lowercased | 12-layer, 768-hidden, 12-heads, 110M parameters_)
- [**ERNIE-GEN _large_**](https://ernie.bj.bcebos.com/ernie_gen_large.tgz) (_lowercased | 24-layer, 1024-hidden, 16-heads, 340M parameters_)
- [**ERNIE-GEN _large with 160G_**](https://ernie.bj.bcebos.com/ernie_gen_large_160g.tgz) (_lowercased | 24-layer, 1024-hidden, 16-heads, 340M parameters_)


## Fine-tuning on Downstream Tasks

We compare the performance of [ERNIE-GEN](https://arxiv.org/pdf/2001.11314.pdf) with the existing SOTA pre-training models for natural language generation ([UniLM](https://arxiv.org/abs/1905.03197), [MASS](https://arxiv.org/abs/1905.02450), [PEGASUS](https://arxiv.org/abs/1912.08777), [BART](https://arxiv.org/abs/1910.13461) and [T5](https://arxiv.org/abs/1910.10683)) on 5 genration tasks, including abstractive summarization (**_Gigaword_** and **_CNN/DailyMail_**), question generation (**_SQuAD_**), dialogue generation (**_Persona-Chat_**) and generative question answering (**_CoQA_**). 

### Abstractive Summarization 

- _**Gigaword**_

The results on Gigaword-10k (10K examples of Gigaword) are presented as follows:

| Model                                                     | <strong>Data / Params</strong> | <strong>Rouge-1</strong> | <strong>Rouge-2</strong> | <strong>Rouge-L</strong> |
| :-------------------------------------------------------- | :----------------------------: | :----------------------: | :----------------------: | :----------------------: |
| UniLM                 |           16G / 340M           |          34.21           |          15.28           |          31.54           |
| **ENRIE-GEN** _base_  |           16G / 110M           |          33.75           |          15.23           |          31.35           |
| **ERNIE-GEN** _large_ |           16G / 340M           |        35.05         |        16.10         |        32.50         |
| **ERNIE-GEN** _large_ (160G) |           160G / 340M           |        **35.51**         |        **16.79**         |        **33.23**         |

The results on Gigaword are presented as follows: 

| Model                                                     | <strong>Data / Params</strong> | <strong>Rouge-1</strong> | <strong>Rouge-2</strong> | <strong>Rouge-L</strong> |
| :-------------------------------------------------------- | :----------------------------: | :----------------------: | :----------------------: | :----------------------: |
| MASS                  |           18G / 160M           |          38.73           |          19.71           |          35.96           |
| BERTSHARE             |           16G / 110M           |          38.13           |          19.81           |          35.62           |
| UniLM                |           16G / 340M           |          38.45           |          19.45           |          35.75           |
| PEGASUS (_C4_)        |          750G / 568M           |          38.75           |          19.96           |          36.14           |
| PEGASUS (_HugeNews_)  |          3.8T / 568M           |          39.12           |          19.86           |          36.24           |
| **ENRIE-GEN** _base_  |           16G / 110M           |          38.83           |          20.04           |          36.20           |
| **ERNIE-GEN** _large_ |           16G / 340M           |        39.25         |        20.25         |        36.53         |
| **ERNIE-GEN** _large_ (160G) |           160G / 340M           |        **39.46**         |        **20.34**         |        **36.74**         |

We preprocess the raw Gigaword dataset following UniLM, the preprocessed data is avalilable at this [Gigaword](https://ernie.bj.bcebos.com/gigaword.tgz).

- _**CNN/Daily Mail**_

The results on CNN/Daily Mail are presented as follows: 

| <strong>Model</strong>                                    | Data / Params | <strong>Rouge-1</strong> | <strong>Rouge-2</strong> | <strong>Rouge-L</strong> |
| :-------------------------------------------------------- | :-----------: | :----------------------: | :----------------------: | :----------------------: |
| MASS                  |  18G / 160M   |          42.12           |          19.50           |          39.01           |
| UniLM                 |  16G / 340M   |          43.33           |          20.21           |          40.51           |
| T5 _large_            |  750G / 340M  |          42.50           |          20.68           |          39.75           |
| T5 _xlarge_           |  750G / 11B   |          43.52           |        **21.55**         |          40.69           |
| BART                  |  160G / 400M  |          44.16           |          21.28           |          40.90           |
| PEGASUS (_C4_)        |  750G / 568M  |          43.90           |          21.20           |          40.76           |
| PEGASUS (_HugeNews_)  |  3.8T / 568M  |          44.17           |          21.47           |          41.11           |
| **ENRIE-GEN** _base_  |  16G / 110M   |          42.30           |          19.92           |          39.68           |
| **ENRIE-GEN** _large_ |  16G / 340M   |          44.02           |          21.17           |          41.26           |
| **ENRIE-GEN** _large_ (160G) |  160G / 340M   |        **44.31**         |          21.35           |        **41.60**         |

We preprocess the raw CNN/Daily Mail dataset following UniLM, the preprocessed data is avalilable at this [CNN/Daily Mail](https://ernie.bj.bcebos.com/cnndm.tgz).

### Question Generation

- _**SQuAD**_

The results on the [SQuAD 1.1](https://arxiv.org/abs/1806.03822) dataset following the data split in [[Du et al., 2017]](https://arxiv.org/pdf/1705.00106.pdf) are presented as follows:

| Model                                                        | <strong>BLEU-4</strong> | <strong>METEOR</strong> | <strong>Rouge-L</strong> |
| :----------------------------------------------------------- | :----------------------: | :----------------------: | :----------------------: |
| [SemQG](https://arxiv.org/abs/1909.06356)                    |          18.37           |          22.65           |          46.68           |
| UniLM _large_ (beam size=1) |          22.12           |          25.06           |          51.07           |
| **ENRIE-GEN** _base_ (beam size=1) |          22.28           |          25.13           |          50.38           |
| **ERNIE-GEN** _large_ (beam size=1) |        24.03         |        26.31         |        52.36         |
| **ERNIE-GEN** _large_ (beam size=5) |        25.40         |        **26.92**         |        52.84         |
| **ERNIE-GEN** _large_ (beam size=5) + (160G) |        **25.41**         |        26.77         |        **52.91**         |

The results following the reversed dev-test data split in [[Zhao et al., 2018]](https://www.aclweb.org/anthology/D18-1424/) are presented as follows:

| Model                                                        | <strong>BLEU-4</strong> | <strong>METEOR</strong> | <strong>Rouge-L</strong> |
| :----------------------------------------------------------- | :----------------------: | :----------------------: | :----------------------: |
| SemQG                    |          20.76           |          24.20           |          48.91           |
| UniLM _large_ (beam size=1) |          23.75           |          25.61           |          52.04           |
| **ENRIE-GEN** _base_ (beam size=1) |          23.52           |          25.61           |          51.45           |
| **ERNIE-GEN** _large_ (beam size=1) |        25.57         |        26.89         |        53.31         |
| **ERNIE-GEN** _large_ (beam size=5) |        26.95         |        **27.57**         |        53.77         |
| **ERNIE-GEN** _large_ (beam size=5) + (160G) |        **27.05**         |        27.43         |        **53.83**         |

*_Note that we also report the results with higher beam size to 5._

The preprocessed data for question generation task can be downloaded from [SQuAD](https://ernie.bj.bcebos.com/squad_qg.tgz).

### Generative Dialogue Response

- _**Personal-Chat**_

 Comparison with current state-of-the-art results on the multi-turn conversations task ([Persona-Chat](https://arxiv.org/abs/1801.07243)) is presented as follows:

| Model                                                     | <strong>BLEU-1</strong> | <strong>BLEU-2</strong> | <strong>Distinct-1</strong> | <strong>Distinct-2</strong> |
| :-------------------------------------------------------- | :---------------------: | :---------------------: | :-------------------------: | :---------------------------: |
| [LIC](https://arxiv.org/abs/1910.07931)                   |          40.5           |          32.0           |            0.019            | 0.113                       |
| [PLATO](https://arxiv.org/abs/1910.07931)                 |          45.8           |          35.7           |            0.012            | 0.064                       |
| PLATO _w/o latent_    |          40.6           |          31.5          |            0.021            | 0.121                    |
| **ERNIE-GEN** _large_ |        **46.8**         |        **36.4**         |          **0.023**          | **0.168**                   |

The training data can be downloaded from [Personal-Chat](https://ernie.bj.bcebos.com/persona_chat.tgz).

### Generative Question Answering

- _**CoQA**_

Results of development set on CoQA task is presented as follows:

| Model                                                     | F1-score |
| :-------------------------------------------------------- | :------: |
| [Seq2Seq](https://arxiv.org/abs/1910.07931)               |   27.5   |
| [PGNet](https://arxiv.org/abs/1910.07931)                 |   45.4   |
| UniLM _large_         |   82.5   |
| **ERNIE-GEN** _large_ | **84.5** |

We preprocess the raw [CoQA](https://arxiv.org/abs/1808.07042) dataset, the preprocessed data is avalilable at this [CoQA-preprocessed](https://ernie.bj.bcebos.com/coqa.tgz).

Finally, we also compared with a concurrent work [ProphetNet](https://arxiv.org/abs/2001.04063), the fine-tuning results on Gigaword, CNN/Daily Mail and SQuAD are reported as follows:

- _**Abstractive Summarization**_

| Model / Task                                                     | <strong>Data / Params</strong> | <strong>Gigaword</strong> |<strong>CNN/Daily Mail</strong>|
| :-------------------------------------------------------- | :----------------------------: | :----------------------: | :----------------------: |
| Metric                                                     | - | <strong>Rouge-1 / Rouge-2 / Rouge-L</strong> |<strong>Rouge-1 / Rouge-2 / Rouge-L</strong>|
| **ProphetNet** _large_ (160G) |           160G / 340M           |     **39.51** / **20.42** / 36.69       |44.20 / 21.17 / 41.30|
| **ERNIE-GEN** _large_ (160G) |           160G / 340M           |        39.46 / 20.34 / **36.74**         |**44.31** / **21.35** / **41.60**|

- _**Question Generation**_

| Model                                                     | <strong>Data / Params</strong> | <strong>BLEU-4 / METEOR / Rouge-L</strong> |<strong>BLEU-4 / METEOR / Rouge-L</strong>|
| :-------------------------------------------------------- | :----------------------------: | :----------------------: |:----------------------: |
| Data split                                                     | - | <strong>Original</strong> |<strong>Reversed dev-test</strong>|
| **ProphetNet** _large_ (16G) |           16G / 340M           |     25.01 / 26.83 / 52.57       |26.72 / **27.64** / **53.79** |
| **ERNIE-GEN** _large_ (16G) |           16G / 340M           |        **25.40** / **26.92** / **52.84**       |**26.95** / 27.57 / **53.77**|

## Usage

### Install PaddlePaddle

This code base has been tested with Paddle Fluid 1.7 with Python 2.7. Other dependency of ERNIE-GEN is listed in `requirements.txt`, you can install it by
```script
pip install -r requirements.txt
```

### Fine-tuning
Please update LD_LIBRARY_PATH about CUDA, cuDNN, NCCL2 before running ERNIE-GEN. We have put the parameter configurations of the above downstream tasks in `config/`. You can easily run finetuning through these configuration files. For example, you can finetune ERNIE-GEN base model on Gigaword by
```script
MODEL="base"      # base or large or large_160g
TASK="gigaword"   # cnndm, coqa, gigaword, squad_qg or persona-chat
sh run_seq2seq.sh ./configs/${MODEL}/${TASK}_conf
```
The log of training and the evaluation results are in `log/job.log.0`. To finetune on your own task data, you can refer to the data format we provide for processing your data.

Our fine-tuning experiments are carried on 8 NVIDIA V100 (32GB) GPUs. If your GPU memory is not enough, you can reduce the batch size in the corresponding configuration file.

**NOTICE: ** The actual total batch size is equal to `configured batch size * number of used gpus`.

### Employ Dynamic Computation Graph

The ERNIE-GEN code using dynamic graph is more concise and flexible, please refer to  [ERNIE-GEN Dygraph](https://github.com/PaddlePaddle/ERNIE/tree/develop/experimental/seq2seq) for specific use.

### The ERNIE 1.0 is avaliable for Chinese Generation Tasks

The ERNIE-GEN code is compatible with [ERNIE 1.0](https://ernie.bj.bcebos.com/ERNIE_1.0_max-len-512.tar.gz) model. After specifying the parameters related to the model and data in the configuration file, you can use ERNIE 1.0 to fine-tune chinese generation tasks.

## Citation

You can cite the paper as below:

```
@article{xiao2020ernie-gen,
  title={ERNIE-GEN: An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation},
  author={Xiao, Dongling and Zhang, Han and Li, Yukun and Sun, Yu and Tian, Hao and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:2001.11314},
  year={2020}
}
```




