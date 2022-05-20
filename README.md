

# ![ERNIE_milestone_20210519_zh](./ERNIE.png)

文心大模型ERNIE是百度发布的产业级知识增强大模型，涵盖了NLP大模型和跨模态大模型。知识增强是文心大模型的核心特色，其能够从大规模知识和海量无结构数据中融合学习，学习效率更高、效果更好，具有良好的可解释性。 百度是国内首个在预训练大模型上做出突破性工作的公司，早在2019年3月就发布了国内首个开源预训练模型文心ERNIE 1.0，此后在语言与跨模态的理解和生成等领域取得一系列技术突破，并在 GLUE、SuperGLUE、VCR、XTREME、DocVQA、SROIE、SemEval 等国际权威评测上斩获数十项冠军。文心ERNIE大模型核心技术荣获2021年度国家技术发明二等奖、2020年世界人工智能大会最高荣誉SAIL奖、中国人工智能学会优秀科技成果奖、相关创新成果也被国际顶级学术会议AAAI、ACL、EMNLP、IJCAI、NAACL收录。 文心大模型ERNIE已应用于百度搜索、信息流、智能音箱等互联网产品，显著提升了亿万用户的体验。 文心大模型ERNIE通过百度飞桨平台对外开源与开放，并通过百度智能云赋能工业、能源、金融、通信、媒体、教育等不同行业，促进各行各业的产业智能化升级。
提醒: ERNIE老版本代码已经迁移至repro分支，欢迎使用我们全新升级的基于动静结合的新版ERNIE套件进行开发。另外，也欢迎上[EasyDL](https://ai.baidu.com/easydl/pro)、[BML](https://ai.baidu.com/bml/app/overview)体验更丰富的功能。
[【了解更多】](https://wenxin.baidu.com/)

# 开源Roadmap

- 2022.5.20:
  - 最新开源ERNIE 3.0系列预训练模型:
    - 110M参数通用模型ERNIE 3.0 Base
    - 280M参数重量级通用模型ERNIE 3.0 XBase
    - 74M轻量级通用模型ERNIE 3.0 Medium
  - 新增语音-语言跨模态模型ERNIE-SAT（链接待补充）
  - 新增ERNIE-Gen（中文）预训练模型，支持多类主流生成任务：主要包括摘要、问题生成、对话、问答
  - 动静结合的文心ERNIE开发套件：基于飞桨动态图功能，支持文心ERNIE模型动态图训练。您仅需要在模型训练开启前，修改一个参数配置，即可实现模型训练的动静切换。
  - 将文本预处理、预训练模型、网络搭建、模型评估、上线部署等NLP开发流程规范封装。
  - 支持NLP常用任务：文本分类、文本匹配、序列标注、信息抽取、文本生成、数据蒸馏等。
  - 提供数据清洗、数据增强、分词、格式转换、大小写转换等数据预处理工具。
- 2021.12.3:
  - 多语言预训练模型`ERNIE-M` [正式开源](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-m)
- 2021.5.20:
  - ERNIE 最新开源四大预训练模型:
    - 多粒度语言知识模型`ERNIE-Gram` [正式开源](https://github.com/PaddlePaddle/ERNIE/blob/develop/ernie-gram)
    - 超长文本双向建模预训练模型`ERNIE-Doc` [正式开源](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-doc)
    - 融合场景图知识的跨模态预训练模型教程`ERNIE-ViL` [正式开源](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-vil)
    - 语言与视觉一体的预训练模型`ERNIE-UNIMO` [正式开源](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-unimo)
- 2020.9.24:
  - `ERNIE-ViL` 技术发布! ([点击进入](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-vil))
    - 面向视觉-语言知识增强的预训练框架，首次在视觉-语言预训练引入结构化的知识。
      - 利用场景图中的知识，构建了物体、属性和关系预测任务，精细刻画模态间细粒度语义对齐。
    - 五项视觉-语言下游任务取得最好效果，[视觉常识推理榜单](https://visualcommonsense.com/)取得第一。
- 2020.5.20:
  - `ERNIE-GEN` 模型正式开源! ([点击进入](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-gen))
    - 最强文本生成预训练模型正式开源，相关工作已被 `IJCAI-2020` 收录。
      - 首次把 ERNIE 预训练技术能力扩展至文本生成领域，在多个典型任务上取得最佳。
      - 您现在即可下载论文报告的所有模型（包含 [base/large/large-430G](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-gen/README.zh.md#预训练模型)）。
    - 首次在预训练阶段加入span-by-span 生成任务，让模型每次能够生成一个语义完整的片段。
    - 提出填充式生成机制和噪声感知机制来缓解曝光偏差问题。
    - 精巧的 Mulit-Flow Attention 实现框架。
- 2020.4.30 发布[ERNIESage](https://github.com/PaddlePaddle/PGL/tree/master/examples/erniesage)， 一种新型图神经网络模型，采用ERNIE做为aggreagtor. 由[PGL](https://github.com/PaddlePaddle/PGL)实现。
- 2020.3.27 [在SemEval2020五项子任务上夺冠](https://www.jiqizhixin.com/articles/2020-03-27-8)。
- 2019.12.26 [GLUE榜第一名](https://www.technologyreview.com/2019/12/26/131372/ai-baidu-ernie-google-bert-natural-language-glue/)。
- 2019.11.6 发布[ERNIE Tiny](https://www.jiqizhixin.com/articles/2019-11-06-9)。
- 2019.7.7 发布[ERNIE 2.0](https://www.jiqizhixin.com/articles/2019-07-31-10)。
- 2019.3.16 发布[ERNIE 1.0](https://www.jiqizhixin.com/articles/2019-03-16-3)。

# 环境安装

1. 安装环境依赖：[环境安装](./README_ENV.md)
2. 安装Ernie套件

```plain
git clone https://github.com/PaddlePaddle/ERNIE.git
```

# 快速上手：使用文心ERNIE大模型进行训练

- 使用ERNIE3.0作为预训练模型，准备工作包括：
  - 下载模型
  - 准备数据
  - 配置训练json文件
  - 启动训练模型
  - 配置预测json文件
  - 启动预测
- 我们以文本分类任务为例，来快速上手ERNIE大模型的使用

## 下载模型

- 使用ERNIE3.0预训练模型进行文本分类任务
- ERNNIE3.0预训练模型的下载与配置

```plain
# ernie_3.0 模型下载
# 进入models_hub目录
cd ./nlp-ernie/wenxin_appzoo/wenxin_appzoo/models_hub
# 运行下载脚本
sh download_ernie_3.0_base_ch.sh
```

## 准备数据

- 文心各个任务的data目录下自带一些示例数据，能够实现直接使用，方便快速熟悉文心的使用。
- 文本分类任务的数据

```shell
#进入文本分类任务文件夹
cd ./nlp-ernie/wenxin_appzoo/wenxin_appzoo/tasks/text_classification/
#查看文本分类任务自带数据集
ls ./data
```

- 注：示例数据仅作为格式演示使用，在真正训练模型时请替换为真实数据。

## 配置训练json文件

- 其预置json文件在./examples/目录下，使用ERNIE3.0预训练模型进行训练的配置文件为的./examples/cls_ernie_fc_ch.json，在该json文件中对数据、模型、训练方式等逻辑进行了配置。

```shell
#查看 ERNIE3.0预训练模型 训练文本分类任务的配置文件
cat ./examples/cls_ernie_fc_ch.json
```

## 启动训练

- 将数据集存放妥当，并配置好cls_ernie_fc_ch.json，我们就可以运行模型训练的命令。
- 其中，单卡指令为`python run_trainer.py`，如下所示，使用基于ernie的中文文本分类模型在训练集上进行本地模型训练。

```shell
# ernie 中文文本分类模型
# 基于json实现预置网络训练。其调用了配置文件./examples/cls_ernie_fc_ch.json
python run_trainer.py --param_path ./examples/cls_ernie_fc_ch.json
```

- 多卡指令为:

```plain
fleetrun --gpus=x,y run_trainer.py./examples/cls_ernie_fc_ch.json
```

- 训练运行的日志会自动保存在**./log/test.log**文件中。
- 训练中以及结束后产生的模型文件会默认保存在./output/**目录下，其中**save_inference_model/文件夹会保存用于预测的模型文件，**save_checkpoint/** 文件夹会保存用于热启动的模型文件。

## 配置预测json文件

- 其预置json文件在./examples/目录下，使用ERNIE2.0预训练模型训练的模型进行预测的配置文件为的./examples/cls_ernie_fc_ch_infer.json
- 主要修改./examples/cls_ernie_fc_ch_infer.json文件的预测模型的输入路径、预测文件的输入路径、预测结果的输出路径，对应修改配置如下：

```
{
"dataset_reader":{"train_reader":{"config":{"data_path":"./data/predict_data"}}},
"inference":{"inference_model_path":"./output/cls_ernie_fc_ch/save_inference_model/inference_step_251",
                        "output_path": "./output/predict_result.txt"}
}
```

## 启动预测

- 运行run_infer.py ，选择对应的参数配置文件即可。如下所示：

```plain
python run_infer.py --param_path ./examples/cls_enrie_fc_ch_infer.json
```

- 预测过程中的日志自动保存在./output/predict_result.txt文件中。

# 预训练模型介绍

- 参考预训练模型原理介绍:[模型介绍](./nlp-ernie/wenxin_appzoo/wenxin_appzoo/models_hub/README.md)
- 预训练模型下载：进入./wenxin_appzoo/models_hub目录下,下载示例：

```plain
#进入预训练模型下载目录
cd ./wenxin_appzoo/models_hub
#下载ERNIE3.0 base模型
sh downlaod_ernie3.0_base_ch.sh
```

- 更多开源模型，见[Research](./Research/README.md)

# 模型效果评估

[模型效果评估](./README_SOCRE.md)

# 数据集下载

[CLUE数据集](https://www.cluebenchmarks.com/)

[DuIE2.0数据集](https://www.luge.ai/#/luge/dataDetail?id=5)

[MSRA_NER数据集](https://ernie-github.cdn.bcebos.com/data-msra_ner.tar.gz)

# 应用场景

文本分类（[文本分类](./nlp-ernie/wenxin_appzoo/wenxin_appzoo/tasks/text_classification/README.md)）

文本匹配（[文本匹配](./nlp-ernie/wenxin_appzoo/wenxin_appzoo/tasks/text_matching/README.md)）

系列标注（[序列标注](./nlp-ernie/wenxin_appzoo/wenxin_appzoo/tasks/sequence_labeling/README.md)）

信息抽取（[信息抽取](./nlp-ernie/wenxin_appzoo/wenxin_appzoo/tasks/information_extraction_many_to_many/README.md)）

文本生成（[文本生成](./nlp-ernie/wenxin_appzoo/wenxin_appzoo/tasks/text_generation/README.md)）

数据蒸馏（[数据蒸馏](./nlp-ernie/wenxin_appzoo/wenxin_appzoo/tasks/data_distillation/README.md)）

工具使用（[工具使用](./nlp-ernie/wenxin_appzoo/wenxin_appzoo/tools/README.md)）

# 文献引用

### ERNIE 1.0

```
@article{sun2019ernie,
  title={Ernie: Enhanced representation through knowledge integration},
  author={Sun, Yu and Wang, Shuohuan and Li, Yukun and Feng, Shikun and Chen, Xuyi and Zhang, Han and Tian, Xin and Zhu, Danxiang and Tian, Hao and Wu, Hua},
  journal={arXiv preprint arXiv:1904.09223},
  year={2019}
}
```

### ERNIE 2.0

```
@inproceedings{sun2020ernie,
  title={Ernie 2.0: A continual pre-training framework for language understanding},
  author={Sun, Yu and Wang, Shuohuan and Li, Yukun and Feng, Shikun and Tian, Hao and Wu, Hua and Wang, Haifeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={05},
  pages={8968--8975},
  year={2020}
}
```

### ERNIE-GEN

```
@article{xiao2020ernie,
  title={Ernie-gen: An enhanced multi-flow pre-training and fine-tuning framework for natural language generation},
  author={Xiao, Dongling and Zhang, Han and Li, Yukun and Sun, Yu and Tian, Hao and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:2001.11314},
  year={2020}
}
```

### ERNIE-ViL

```
@article{yu2020ernie,
  title={Ernie-vil: Knowledge enhanced vision-language representations through scene graph},
  author={Yu, Fei and Tang, Jiji and Yin, Weichong and Sun, Yu and Tian, Hao and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:2006.16934},
  year={2020}
}
```

### ERNIE-Gram

```
@article{xiao2020ernie,
  title={ERNIE-Gram: Pre-Training with Explicitly N-Gram Masked Language Modeling for Natural Language Understanding},
  author={Xiao, Dongling and Li, Yu-Kun and Zhang, Han and Sun, Yu and Tian, Hao and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:2010.12148},
  year={2020}
}
```

### ERNIE-Doc

```
@article{ding2020ernie,
  title={ERNIE-Doc: A retrospective long-document modeling transformer},
  author={Ding, Siyu and Shang, Junyuan and Wang, Shuohuan and Sun, Yu and Tian, Hao and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:2012.15688},
  year={2020}
}
```

### ERNIE-UNIMO

```
@article{li2020unimo,
  title={Unimo: Towards unified-modal understanding and generation via cross-modal contrastive learning},
  author={Li, Wei and Gao, Can and Niu, Guocheng and Xiao, Xinyan and Liu, Hao and Liu, Jiachen and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:2012.15409},
  year={2020}
}
```

### ERNIE-M

```
@article{ouyang2020ernie,
  title={Ernie-m: Enhanced multilingual representation by aligning cross-lingual semantics with monolingual corpora},
  author={Ouyang, Xuan and Wang, Shuohuan and Pang, Chao and Sun, Yu and Tian, Hao and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:2012.15674},
  year={2020}
}
```
