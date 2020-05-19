ERNIE 2.0 is a continual pre-training framework for language understanding in which pre-training tasks can be incrementally built and learned through multi-task learning.
ERNIE 2.0 builds a strong basic for nearly every NLP tasks: Text Classification, Ranking, NER, Reading Comprehension, Genration and so on.

# News

- May.20.2020: Try ERNIE in "`dygraph`", with:

	- Pretrain and finetune ERNIE with [PaddlePaddle v1.7](https://github.com/PaddlePaddle/Paddle/tree/release/1.7).
	- Eager execution with `paddle.fluid.dygraph`.
	- Distributed training.
	- Easy deployment.
	- Learn NLP in Aistudio tutorials.
	- Backward compatibility for old-styled checkpoint
- 2020/4/30: Release [ERNIESage](https://github.com/PaddlePaddle/PGL/tree/master/examples/erniesage), a novel Graph Neural Network Model using ERNIE as its aggregator. It is implemented through [PGL](https://github.com/PaddlePaddle/PGL)
- 2020/3/27: [Champion on 5 SemEval2020 sub tasks](https://www.jiqizhixin.com/articles/2020-03-27-8)
- 2019/12/26: [1st place on GLUE leaderboard](https://www.technologyreview.com/2019/12/26/131372/ai-baidu-ernie-google-bert-natural-language-glue/)
- 2019/11/6: [Introducing ERNIE-tiny](https://www.jiqizhixin.com/articles/2019-11-06-9)
- 2019/7/31: [Introducing ERNIE2.0](https://www.jiqizhixin.com/articles/2019-07-31-10)
- 2019/3/16: [Introducing ERNIE1.0](https://www.jiqizhixin.com/articles/2019-03-16-3)

	
# Table of contents
* [Tutorials](#tutorials)
* [Setup](#setup)
* [Finetune](#finetune)
* [Distributed pretrain](#distributed-pretrain)
* [Online inference](#online-inference)
* [Distillation](#distillation)
* [Citation](#citation)
* [Contact us](#contact-us)

# Quick Tour

```python
import numpy as np
import paddle.fluid.dygraph as D
from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.modeling_ernie import ErnieModel

D.guard().__enter__() # activate paddle `dygrpah` mode

model = ErnieModel.from_pretrained('ernie-1.0')    # Try to get pretrained model from server, make sure you have network connection
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

ids, _ = tokenizer.encode('hello world')
ids = D.to_variable(np.expand_dims(ids, 0))  # insert extra `batch` dimension
pooled, encoded = model(ids)                 # eager execution
print(pooled.numpy())                        # convert  results to numpy

```

# Tutorials

Don't have GPU? try ERNIE in [AIStudio](https://aistudio.baidu.com/aistudio/index)!
(please choose the latest version and apply for a GPU environment)

1. [Text classification walkthrough](https://aistudio.baidu.com/aistudio/projectdetail/266623)
1. [Sementic analysis](https://aistudio.baidu.com/aistudio/projectdetail/427482)
2. [Cloze test](https://aistudio.baidu.com/aistudio/projectdetail/433491)
3. [Knowledge Distillation](https://aistudio.baidu.com/aistudio/projectdetail/439460)
4. [Ask Ernie](https://aistudio.baidu.com/aistudio/projectdetail/456443)
5. ...

# Setup

##### 1. install ernie

```script
pip install -i https://test.pypi.org/simple/ paddle-ernie==0.0.1
```

or 

```shell
git clone -b dygraph https://github.com/PaddlePaddle/ERNIE.git --single-branch
cd ERNIE
pip install -r requirement.txt
pip setup.py -e .

```

##### 3. download pretrained models

| Model                                              | Description                                                  |
| :------------------------------------------------- | :----------------------------------------------------------- |
| [ERNIE 1.0 Base for Chinese](https://ernie-github.cdn.bcebos.com/model-ernie1.0.1.tar.gz)           | ernie 1.0 base: L12H768A12|
| [ERNIE tiny](https://ernie-github.cdn.bcebos.com/model-ernie_tiny.1.tar.gz)                         | erine tiny: L3H1024A16|
| [ERNIE 2.0 Base for English](https://ernie-github.cdn.bcebos.com/model-ernie2.0-en.1.tar.gz)        | ernie 2.0 base: L12H768A12  |
| [ERNIE 2.0 Large for English](https://ernie-github.cdn.bcebos.com/model-ernie2.0-large-en.1.tar.gz) | ernie 2.0 large: L24H1024A16 |

##### 4. download datasets
 
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

**Chinese Datasets**

| Datasets|Description|
|:--------|:----------|
| [XNLI](https://ernie-github.cdn.bcebos.com/data-xnli.tar.gz)                 |XNLI is a natural language inference dataset in 15 languages. It was jointly built by Facebook and New York University. We use Chinese data of XNLI to evaluate language understanding ability of our model. [url: https://github.com/facebookresearch/XNLI]|
| [ChnSentiCorp](https://ernie-github.cdn.bcebos.com/data-chnsenticorp.tar.gz) |ChnSentiCorp is a sentiment analysis dataset consisting of reviews on online shopping of hotels, notebooks and books.|
| [MSRA-NER](https://ernie-github.cdn.bcebos.com/data-msra_ner.tar.gz)         |MSRA-NER (SIGHAN2006) dataset is released by MSRA for recognizing the names of people, locations and organizations in text.|
| [NLPCC2016-DBQA](https://ernie-github.cdn.bcebos.com/data-dbqa.tar.gz)       |NLPCC2016-DBQA is a sub-task of NLPCC-ICCPOL 2016 Shared Task which is hosted by NLPCC(Natural Language Processing and Chinese Computing), this task targets on selecting documents from the candidates to answer the questions. [url: http://tcci.ccf.org.cn/conference/2016/dldoc/evagline2.pdf]|


# Finetune 

- try eager execution with `dygraph model` :

```script
python3 ./demo/finetune_classifier_dygraph.py \
    --from_pretrained ernie_1.0 \
    --data_dir ./data/xnli 
```

- Distributed finetune

`paddle.distributed.launch` is a process manager, we use it to launch python processes on each avalible GPU devices:

when in distributed training, `max_steps` is used as stopping criteria rather than `epoch` to prevent dead block.
also notice than we shard the train data according to device id to prevent over fitting.

demo: 

```script
python3 -m paddle.distributed.launch \
./demo/finetune_classifier_dygraph_distributed.py \
    --data_dir data/mnli \
    --max_steps 10000 \
    --from_pretrained ernie2.0-en
```


many other demo python scripts:

1. [Sementic Analysis](./demo/finetune_sementic_analysis_dygraph.py)
1. [Pointwise Ranking](./demo/finetune_classifier_dygraph.py)
1. [Name Entity Recognition(NER)](./demo/finetune_ner_dygraph.py)
1. [Machine Reading Comprehension](./demo/finetune_mrc_dygraph.py)
1. Text generation...




**recomended hyper parameters:**

|tasks|batch size|learning rate|
|--|--|--|
| CoLA         | 32 / 64 (base)  | 3e-5                     |
| SST-2        | 64 / 256 (base) | 2e-5                     |
| STS-B        | 128             | 5e-5                     |
| QQP          | 256             | 3e-5(base)/5e-5(large)   |
| MNLI         | 256 / 512 (base)| 3e-5                     |
| QNLI         | 256             | 2e-5                     |
| RTE          | 16 / 4 (base)   | 2e-5(base)/3e-5(large)   |
| MRPC         | 16 / 32 (base)  | 3e-5                     |
| WNLI         | 8               | 2e-5                     |
| XNLI         | 512             | 1e-4(base)/4e-5(large)   |
| CMRC2018     | 64              | 3e-5                     |
| DRCD         | 64              | 5e-5(base)/3e-5(large)   |
| MSRA-NER(SIGHAN2006)  | 16     | 5e-5(base)/1e-5(large)   |
| ChnSentiCorp | 24              | 5e-5(base)/1e-5(large)   |
| LCQMC        | 32              | 2e-5(base)/5e-6(large)   |
| NLPCC2016-DBQA| 64             | 2e-5(base)/1e-5(large)   |

# Distributed pretrain

see [here](./demo/pretrain/README.md)


# Online inference

If `--inference_model_dir` is passed to `finetune_classifier_dygraph.py`, 
a deployable model will be generated at the end of finetuning and your model is ready to serve.

For details about online inferece, see [C++ inference API](./inference/README.md),
or you can start a multi-gpu inference server with a few lines of codes:

```shell
python -m propeller.tools.start_server -m /path/to/saved/inference_model  -p 8881
```

and call the server just like calling local function (python3 only):

```python
from propeller.service.client import InferenceClient
from ernie.tokenizing_ernie import ErnieTokenizer

client = InferenceClient('tcp://localhost:8881')
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
ids, sids = tokenizer.encode('hello world')
ids = np.expand_dims(ids, 0)
sids = np.expand_dims(sids, 0)
result = client(ids, sids)
```

A pre-made `inference model` for ernie-1.0 can be downloaded at [here](https://ernie.bj.bcebos.com/ernie1.0_zh_inference_model.tar.gz). 
It can be used for feature-based finetuning or feature extraction.

# Distillation

Knowledge distillation is good way to compress and accelerate ERNIE. 

For details about distillation, see [here](./distill/README.md)

# Citation

please cite [ERNIE 2.0](https://arxiv.org/abs/1907.12412):

```
@article{SunERNIE,
  title={ERNIE 2.0: A Continual Pre-training Framework for Language Understanding},
  author={Sun, Yu and Wang, Shuohuan and Li, Yukun and Feng, Shikun and Tian, Hao and Wu, Hua and Wang, Haifeng},
}
```

and [ERNIE-gen](https://arxiv.org/abs/2001.11314)

```
@article{Xiao2020ERNIE,
  title={ERNIE-GEN: An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation},
  author={Xiao, Dongling and Zhang, Han and Li, Yukun and Sun, Yu and Tian, Hao and Wu, Hua and Wang, Haifeng},
  year={2020},
}
```
# Contact us

- [Github Issues](https://github.com/PaddlePaddle/ERNIE/issues): bug reports, feature requests, install issues, usage issues, etc.
- QQ discussion group: 760439550 (ERNIE discussion group).
- [Forums](http://ai.baidu.com/forum/topic/list/168?pageNo=1): discuss implementations, research, etc.

