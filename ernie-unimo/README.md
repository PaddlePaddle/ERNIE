# <p align=center>`UNIMO`</p>

Code for the main conference of ACL 2021 long paper [UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning](https://arxiv.org/pdf/2012.15409.pdf)


## Abstract

Existed pre-training methods either focus on single-modal tasks or multi-modal tasks, and cannot effectively adapt to each other.
They can only utilize single-modal data (i.e., text or image) or limited multi-modal data (i.e., image-text pairs).
In this work, we propose a UNIfied-MOdal pre-training architecture, namely `UNIMO`, which can effectively adapt to both single-modal and multi-modal understanding and generation tasks.
Large scale of free text corpus and image collections are utilized to improve the capability of visual and textual understanding, and cross-modal contrastive learning (CMCL) is leveraged to align the textual and visual information into a unified semantic space over a corpus of image-text pairs augmented with related images and texts.
With the help of rich non-paired single-modal data, our model is able to learn more generalizable representations, by allowing textual knowledge and visual knowledge to enhance each other in the unified semantic space.
The experimental results show that `UNIMO` greatly improves the performance of several single-modal and multi-modal downstream tasks.

![UNIMO](images/framework.png#pic_center)


## Performance

Results on multi-modal understanding and generation tasks:

![UNIMO](images/multiple.png#pic_center)

Results on single-modal understanding and generation tasks:

![UNIMO](images/single.png#pic_center)

---

## TODOs
- [] Add all downstream tasks
- [] Add unimo large model

## Dependencies
python 3.7.4\
paddlepaddle-gpu==1.8.4.post107\
pyrouge==0.1.3

## Pre-trained Models
`UNIMO` adopts large-scale text corpus, image collections and image-text aligned datasets as the pre-training data. 
We provide `UNIMO` models of 1 scale settings which are pretrained:

[UNIMO base](https://unimo.bj.bcebos.com/model/unimo_base_en.tar.gz) (lowercased | 12 layers)

```
MODEL_SIZE=base
cd /path/to/model_files
wget --no-check-certificate -q https://unimo.bj.bcebos.com/model/unimo_${MODEL_SIZE}_en.tar.gz
tar -zxf unimo_${MODEL_SIZE}_en.tar.gz
```

## Experiments
Our fine-tuning experiments are carried on V100 GPU. Here are the results from the `UNIMO` model:

<table>
    <tr>
        <td><strong><center>Task Type</strong></td>
        <td><strong><center>Datatset</strong></td>
        <td><strong><center>Pre-trained Models</strong></td>
        <td><strong><center>Start Command</strong></td>
        <td><strong><center>V100 GPU Cards</strong></td>
        <td><strong><center>Running Time</strong></td>
    </tr>
    <tr>
        <td rowspan="1"><center>Text Understanding<center></td>
        <td rowspan="1"><center>SST-2<center></td>
        <td><center>UNIMO base</td>
        <td><center>sh ./script/classification/SST-2/run.sh</td>
        <td><center>8</td>
        <td><center>9h</td>
    </tr>
    <tr>
        <td rowspan="1"><center>Text Generation<center></td>
        <td rowspan="1"><center>CoQA<center></td>
        <td><center>UNIMO base</td>
        <td><center>sh ./script/seq2seq/coqa/run.sh</td>
        <td><center>4</td>
        <td><center>7h</td>
    </tr>
    <tr>
        <td rowspan="1"><center>Multi-Modal Understanding<center></td>
        <td rowspan="1"><center>Flickr30k<center></td>
        <td><center>UNIMO base</td>
        <td><center>sh ./script/retrieval/Flickr30k/run.sh</td>
        <td><center>16</td>
        <td><center>3d</td>
    </tr>
<table>

---
## Text Understanding Tasks

### (1) Sentiment Classification

#### Download SST-2 dataset:
```
cd /path/to/data
wget --no-check-certificate -q https://unimo.bj.bcebos.com/data/SST-2.tar.gz
tar -zxf SST.tar.gz
```

#### Run the following common to train and evaluate on the SST-2 dataset:

For base model:
```
bash ./script/classification/SST-2/run.sh
```

#### Evaluation Results:

<table>
    <tr>
        <td><strong><center>Model</strong></td>
        <td><strong><center>Acc</strong></td>
    </tr>
    <tr>
        <td><center>UNIMO-base</td>
        <td><center>95.1</td>
    </tr>
<table>

##  Text Generation Tasks

### (1) Conversation Question Answering

#### Download CoQA dataset:
```
cd /path/to/data
wget --no-check-certificate -q https://unimo.bj.bcebos.com/data/coqa.tar.gz
tar -zxf coqa.tar.gz
```

#### Download evaluation script:
```
cd src/eval/tasks
wget --no-check-certificate -q https://unimo.bj.bcebos.com/eval_script/coqa.tar.gz
tar -zxf coqa.tar.gz
```


#### Run the following common to train and evaluate on the CoQA dataset:

For base model:
```
bash ./script/seq2seq/coqa/run.sh
```

#### Evaluation Results:

<table>
    <tr>
        <td><strong><center>Model</strong></td>
        <td><strong><center>Acc</strong></td>
    </tr>
    <tr>
        <td><center>UNIMO-base</td>
        <td><center>80.2</td>
    </tr>
<table>


##  Multi-Modal Understanding Tasks

### (1) Image-Text Retrieval

#### Download Flickr30k dataset:

##### Note: Visual features are extracted by [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)

```
cd /path/to/data
wget --no-check-certificate -q https://unimo.bj.bcebos.com/data/Flickr30k.tar.gz # occupies about 37G disk space
tar -zxf Flickr30k.tar.gz
```

#### Run the following common to train and evaluate on the Flickr30k dataset:

For base model:
```
bash ./script/retrieval/Flickr30k/run.sh
```

#### Evaluation Results:

Results of Image Retrieval task on Flickr30k dataset

<table>
    <tr>
        <td><strong><center>Model</strong></td>
        <td><strong><center>R@1</strong></td>
        <td><strong><center>R@5</strong></td>
        <td><strong><center>R@10</strong></td>
    </tr>
    <tr>
        <td><center>UNIMO-base</td>
        <td><center>74.66</td>
        <td><center>93.40</td>
        <td><center>96.08</td>
    </tr>
<table>

Results of Text Retrieval task on Flickr30k dataset

<table>
    <tr>
        <td><strong><center>Model</strong></td>
        <td><strong><center>R@1</strong></td>
        <td><strong><center>R@5</strong></td>
        <td><strong><center>R@10</strong></td>
    </tr>
    <tr>
        <td><center>UNIMO-base</td>
        <td><center>89.70</td>
        <td><center>98.40</td>
        <td><center>99.10</td>
    </tr>
<table>

---

Citation
---
If you find our paper and code useful, please cite the following paper:
```
@article{li2020unimo,
  title={UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning},
  author={Li, Wei and Gao, Can and Niu, Guocheng and Xiao, Xinyan and Liu, Hao and Liu, Jiachen and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:2012.15409},
  year={2020}
}
```

Contact information
---

For help or issues using `UNIMO`, please submit a GitHub issue.

For personal communication related to `UNIMO`, please contact Wei Li (liwei85@baidu.com), Guocheng Niu (niuguocheng@baidu.com) , Can Gao (gaocan01@baidu.com).