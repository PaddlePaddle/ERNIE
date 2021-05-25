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
- [] Add VQA tasks

## Dependencies
python 3.7.4\
paddlepaddle-gpu==1.8.4.post107\
pyrouge==0.1.3
regex==2020.7.14

## Pre-trained Models
`UNIMO` adopts large-scale text corpus, image collections and image-text aligned datasets as the pre-training data. 
We provide `UNIMO` pre-trained models below:

[UNIMO base](https://unimo.bj.bcebos.com/model/unimo_base_en.tar.gz) (lowercased | 12 layers)

[UNIMO-mnli base](https://unimo.bj.bcebos.com/model/unimo_mnli_base_en.tar.gz) (lowercased | 12 layers)

[UNIMO large](https://unimo.bj.bcebos.com/model/unimo_large_en.tar.gz) (lowercased | 24 layers)

[UNIMO-mnli large](https://unimo.bj.bcebos.com/model/unimo_mnli_large_en.tar.gz) (lowercased | 24 layers)

```
MODEL_SIZE=base # base | mnli_base | large | mnli_large
cd /path/to/model_files
wget --no-check-certificate -q https://unimo.bj.bcebos.com/model/unimo_${MODEL_SIZE}_en.tar.gz
tar -zxf unimo_${MODEL_SIZE}_en.tar.gz
```

## Experiments
Our fine-tuning experiments are carried on V100 GPU. The following are the startup methods and basic settings of all downstream tasks:

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
        <td rowspan="8"><center>Text Understanding<center></td>
        <td rowspan="2"><center>SST-2<center></td>
        <td><center>UNIMO base</td>
        <td><center>sh ./script/classification/SST-2/run.sh</td>
        <td><center>8</td>
        <td><center>9h</td>
    </tr>
    <tr>
        <td><center>UNIMO large</td>
        <td><center>sh ./script/classification/SST-2_large/run.sh</td>
        <td><center>8</td>
        <td><center>14h</td>
    </tr>
    <tr>
        <td rowspan="2"><center>CoLA<center></td>
        <td><center>UNIMO base</td>
        <td><center>sh ./script/classification/CoLA/run.sh</td>
        <td><center>4</td>
        <td><center>2h</td>
    </tr>
    <tr>
        <td><center>UNIMO large</td>
        <td><center>sh ./script/classification/CoLA_large/run.sh</td>
        <td><center>4</td>
        <td><center>4h</td>
    </tr>
    <tr>
        <td rowspan="2"><center>MNLI-AX<center></td>
        <td><center>UNIMO base</td>
        <td><center>sh ./script/classification/MNLI-AX/run.sh</td>
        <td><center>8</td>
        <td><center>1d20h</td>
    </tr>
    <tr>
        <td><center>UNIMO large</td>
        <td><center>sh ./script/classification/MNLI-AX_large/run.sh</td>
        <td><center>8</td>
        <td><center>2d13h</td>
    </tr>
    <tr>
        <td rowspan="2"><center>STS-B<center></td>
        <td><center>UNIMO-mnli base</td>
        <td><center>sh ./script/regression/STS-B/run.sh</td>
        <td><center>8</td>
        <td><center>2h</td>
    </tr>
    <tr>
        <td><center>UNIMO-mnli large</td>
        <td><center>sh ./script/regression/STS-B_large/run.sh</td>
        <td><center>8</td>
        <td><center>4h</td>
    </tr>
    <tr>
        <td rowspan="8"><center>Text Generation<center></td>
        <td rowspan="2"><center>CNN/DailyMail<center></td>
        <td><center>UNIMO base</td>
        <td><center>sh ./script/seq2seq/cnndm/run.sh</td>
        <td><center>4</td>
        <td><center>1d8h</td>
    </tr>
    <tr>
        <td><center>UNIMO large</td>
        <td><center>sh ./script/seq2seq/cnndm_large/run.sh</td>
        <td><center>4</td>
        <td><center>3d18h</td>
    </tr>
    <tr>
        <td rowspan="2"><center>Gigaword<center></td>
        <td><center>UNIMO base</td>
        <td><center>sh ./script/seq2seq/gigaword/run.sh</td>
        <td><center>4</td>
        <td><center>1d3h</td>
    </tr>
    <tr>
        <td><center>UNIMO large</td>
        <td><center>sh ./script/seq2seq/gigaword_large/run.sh</td>
        <td><center>4</td>
        <td><center>2d3h</td>
    </tr>
    <tr>
        <td rowspan="2"><center>CoQA<center></td>
        <td><center>UNIMO base</td>
        <td><center>sh ./script/seq2seq/coqa/run.sh</td>
        <td><center>4</td>
        <td><center>7h</td>
    </tr>
    <tr>
        <td><center>UNIMO large</td>
        <td><center>sh ./script/seq2seq/coqa_large/run.sh</td>
        <td><center>4</td>
        <td><center>22h</td>
    </tr>
    <tr>
        <td rowspan="2"><center>Squad_QG<center></td>
        <td><center>UNIMO base</td>
        <td><center>sh ./script/seq2seq/squad_qg/run.sh</td>
        <td><center>4</td>
        <td><center>4h</td>
    </tr>
    <tr>
        <td><center>UNIMO large</td>
        <td><center>sh ./script/seq2seq/squad_qg_large/run.sh</td>
        <td><center>4</td>
        <td><center>8h</td>
    </tr>
    <tr>
        <td rowspan="6"><center>Multi-Modal Understanding<center></td>
        <td rowspan="2"><center>Flickr30k<center></td>
        <td><center>UNIMO base</td>
        <td><center>sh ./script/retrieval/Flickr30k/run.sh</td>
        <td><center>16</td>
        <td><center>3d</td>
    </tr>
    <tr>
        <td><center>UNIMO large</td>
        <td><center>sh ./script/retrieval/Flickr30k_large/run.sh</td>
        <td><center>16</td>
        <td><center>3d</td>
    </tr>
    <tr>
        <td rowspan="2"><center>SNLI-VE<center></td>
        <td><center>UNIMO base</td>
        <td><center>sh ./script/visual_entailment/SNLI-VE/run.sh</td>
        <td><center>16</td>
        <td><center>16h</td>
    </tr>
    <tr>
        <td><center>UNIMO large</td>
        <td><center>sh ./script/visual_entailment/SNLI-VE_large/run.sh</td>
        <td><center>16</td>
        <td><center>2d</td>
    </tr>
    <tr>
        <td rowspan="2"><center>VQA<center></td>
        <td><center>UNIMO base</td>
        <td><center>-</td>
        <td><center>-</td>
        <td><center>-</td>
    </tr>
    <tr>
        <td><center>UNIMO large</td>
        <td><center>-</td>
        <td><center>-</td>
        <td><center>-</td>
    </tr>
    <tr>
        <td rowspan="6"><center>Multi-Modal Generation<center></td>
        <td rowspan="2"><center>COCO Caption<center></td>
        <td><center>UNIMO base</td>
        <td><center>sh ./script/img2txt/coco/run.sh</td>
        <td><center>16</td>
        <td><center>3d</td>
    </tr>
    <tr>
        <td><center>UNIMO large</td>
        <td><center>sh ./script/img2txt/coco_large/run.sh</td>
        <td><center>16</td>
        <td><center>4d</td>
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
For large model:
```
bash ./script/classification/SST-2_large/run.sh
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
    <tr>
        <td><center>UNIMO-large</td>
        <td><center>96.8</td>
    </tr>
<table>

### (2) Natural Language Inference

#### Download MNLI-AX dataset:
```
cd /path/to/data
wget --no-check-certificate -q https://unimo.bj.bcebos.com/data/MNLI-AX.tar.gz
tar -zxf MNLI-AX.tar.gz
```

#### Run the following common to train and evaluate on the MNLI-AX dataset:

For base model:
```
bash ./script/classification/MNLI-AX/run.sh
```
For large model:
```
bash ./script/classification/MNLI-AX_large/run.sh
```

#### Evaluation Results:

<table>
    <tr>
        <td><strong><center>Model</strong></td>
        <td><strong><center>Acc-(m/mm)</strong></td>
    </tr>
    <tr>
        <td><center>UNIMO-base</td>
        <td><center>86.8/86.7</td>
    </tr>
    <tr>
        <td><center>UNIMO-large</td>
        <td><center>89.8/89.5</td>
    </tr>
<table>


### (3) Similarity Tasks

#### Download STS-B dataset:
```
cd /path/to/data
wget --no-check-certificate -q https://unimo.bj.bcebos.com/data/STS-B.tar.gz
tar -zxf STS-B.tar.gz
```

#### Run the following common to train and evaluate on the STS-B dataset:

For base model:
```
bash ./script/regression/STS-B/run.sh
```
For large model:
```
bash ./script/regression/STS-B_large/run.sh
```

#### Evaluation Results:

<table>
    <tr>
        <td><strong><center>Model</strong></td>
        <td><strong><center>Pearson correlation</strong></td>
    </tr>
    <tr>
        <td><center>UNIMO-base</td>
        <td><center>91.0</td>
    </tr>
    <tr>
        <td><center>UNIMO-large</td>
        <td><center>92.6</td>
    </tr>
<table>


### (4) Linguistic Acceptability Judgments

#### Download CoLA dataset:
```
cd /path/to/data
wget --no-check-certificate -q https://unimo.bj.bcebos.com/data/CoLA.tar.gz
tar -zxf CoLA.tar.gz
```

#### Run the following common to train and evaluate on the CoLA dataset:

For base model:
```
bash ./script/classification/CoLA/run.sh
```
For large model:
```
bash ./script/classification/CoLA_large/run.sh
```

#### Evaluation Results:

<table>
    <tr>
        <td><strong><center>Model</strong></td>
        <td><strong><center>Matthews correlation</strong></td>
    </tr>
    <tr>
        <td><center>UNIMO-base</td>
        <td><center>65.4</td>
    </tr>
    <tr>
        <td><center>UNIMO-large</td>
        <td><center>68.5</td>
    </tr>
<table>


##  Text Generation Tasks

### (1) Document Summarization

#### Download CNN/DailyMail dataset:
```
cd /path/to/data
wget --no-check-certificate -q https://unimo.bj.bcebos.com/data/cnndm.tar.gz
tar -zxf cnndm.tar.gz
```

#### Download evaluation script:
```
cd src/eval/tasks
wget --no-check-certificate -q https://unimo.bj.bcebos.com/eval_script/cnndm.tar.gz
tar -zxf cnndm.tar.gz
```

#### Run the following common to train and evaluate on the CNN/DailyMail dataset:

For base model:
```
bash ./script/seq2seq/cnndm/run.sh
```
For large model:
```
bash ./script/seq2seq/cnndm_large/run.sh
```

#### Evaluation Results:


<table>
    <tr>
        <td><strong><center>Model</strong></td>
        <td><strong><center>ROUGE-1</strong></td>
        <td><strong><center>ROUGE-2</strong></td>
        <td><strong><center>ROUGE-L</strong></td>
    </tr>
    <tr>
        <td><center>UNIMO-base</td>
        <td><center>42.42</td>
        <td><center>20.12</td>
        <td><center>39.61</td>
    </tr>
    <tr>
        <td><center>UNIMO-large</td>
        <td><center>43.51</td>
        <td><center>20.65</td>
        <td><center>40.63</td>
    </tr>
<table>


### (2) Sentence Compression

#### Download Gigaword dataset:
```
cd /path/to/data
wget --no-check-certificate -q https://unimo.bj.bcebos.com/data/gigaword.tar.gz
tar -zxf gigaword.tar.gz
```

#### Download evaluation script:
```
cd src/eval/tasks
wget --no-check-certificate -q https://unimo.bj.bcebos.com/eval_script/gigaword.tar.gz
tar -zxf gigaword.tar.gz
```

#### Run the following common to train and evaluate on the Gigaword dataset:

For base model:
```
bash ./script/seq2seq/gigaword/run.sh
```
For large model:
```
bash ./script/seq2seq/gigaword_large/run.sh
```

#### Evaluation Results:

<table>
    <tr>
        <td><strong><center>Model</strong></td>
        <td><strong><center>ROUGE-1</strong></td>
        <td><strong><center>ROUGE-2</strong></td>
        <td><strong><center>ROUGE-L</strong></td>
    </tr>
    <tr>
        <td><center>UNIMO-base</td>
        <td><center>38.80</td>
        <td><center>19.99</td>
        <td><center>36.27</td>
    </tr>
    <tr>
        <td><center>UNIMO-large</td>
        <td><center>39.71</td>
        <td><center>20.37</td>
        <td><center>36.88</td>
    </tr>
<table>


### (3) Question Generation

#### Download Squad dataset:
```
cd /path/to/data
wget --no-check-certificate -q https://unimo.bj.bcebos.com/data/squad_qg.tar.gz
tar -zxf squad_qg.tar.gz
```

#### Download evaluation script:
```
cd src/eval/tasks
wget --no-check-certificate -q https://unimo.bj.bcebos.com/eval_script/squad_qg.tar.gz
tar -zxf squad_qg.tar.gz
```

#### Run the following common to train and evaluate on the Squad dataset:

For base model:
```
bash ./script/seq2seq/squad_qg/run.sh
```
For large model:
```
bash ./script/seq2seq/squad_qg_large/run.sh
```

#### Evaluation Results:

<table>
    <tr>
        <td><strong><center>Model</strong></td>
        <td><strong><center>BLUE4</strong></td>
        <td><strong><center>METEOR</strong></td>
        <td><strong><center>ROUGE-L</strong></td>
    </tr>
    <tr>
        <td><center>UNIMO-base</td>
        <td><center>22.78</td>
        <td><center>25.24</td>
        <td><center>51.34</td>
    </tr>
    <tr>
        <td><center>UNIMO-large</td>
        <td><center>24.59</td>
        <td><center>26.39</td>
        <td><center>52.47</td>
    </tr>
<table>

### (4) Conversation Question Answering

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
For large model:
```
bash ./script/seq2seq/coqa_large/run.sh
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
    <tr>
        <td><center>UNIMO-large</td>
        <td><center>84.9</td>
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
For large model:
```
bash ./script/retrieval/Flickr30k_large/run.sh
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
    <tr>
        <td><center>UNIMO-large</td>
        <td><center>78.04</td>
        <td><center>94.24</td>
        <td><center>97.12</td>
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
    <tr>
        <td><center>UNIMO-large</td>
        <td><center>89.40</td>
        <td><center>98.90</td>
        <td><center>99.80</td>
    </tr>
<table>

### (2) Visual Entailment

#### Download SNLI-VE dataset:

##### Note: Visual features are extracted by [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)

```
cd /path/to/data
wget --no-check-certificate -q https://unimo.bj.bcebos.com/data/SNLI-VE.tar.gz
tar -zxf SNLI-VE.tar.gz
```

#### Run the following common to train and evaluate on the SNLI-VE dataset:

For base model:
```
bash ./script/visual_entailment/SNLI-VE/run.sh
```
For large model:
```
bash ./script/visual_entailment/SNLI-VE_large/run.sh
```

#### Evaluation Results:

Results of Visual Entailment task on SNLI-VE dataset

<table>
    <tr>
        <td><strong><center>Model</strong></td>
        <td><strong><center>dev</strong></td>
        <td><strong><center>test</strong></td>
    </tr>
    <tr>
        <td><center>UNIMO-base</td>
        <td><center>80.00</td>
        <td><center>79.10</td>
    </tr>
    <tr>
        <td><center>UNIMO-large</td>
        <td><center>81.11</td>
        <td><center>80.63</td>
    </tr>
<table>


##  Multi-Modal Generation Tasks

### (1) Image Caption Generation

#### Download COCO Caption dataset:

##### Note: Visual features are extracted by [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)

```
cd /path/to/data
wget --no-check-certificate -q https://unimo.bj.bcebos.com/data/coco.tar.gz
tar -zxf coco.tar.gz
```

#### Download evaluation script:
```
cd src/eval/tasks
wget --no-check-certificate -q https://unimo.bj.bcebos.com/eval_script/coco.tar.gz
tar -zxf coco.tar.gz
```

#### Run the following common to train and evaluate on the COCO Caption dataset:

For base model:
```
bash ./script/img2txt/coco/run.sh
```
For large model:
```
bash ./script/img2txt/coco_large/run.sh
```

#### Evaluation Results:

<table>
    <tr>
        <td><strong><center>Model</strong></td>
        <td><strong><center>BLUE4</strong></td>
        <td><strong><center>CIDEr</strong></td>
    </tr>
    <tr>
        <td><center>UNIMO-base</td>
        <td><center>38.8</td>
        <td><center>124.4</td>
    </tr>
    <tr>
        <td><center>UNIMO-large</td>
        <td><center>39.6</td>
        <td><center>127.7</td>
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
