# 文本生成

- 文本生成是自然语言处理中一个重要的研究领域，具有广阔的应用前景。简单来说，模型接受源文本（source text）输入，完成目标文本（target text）输出。经典地应用场景如对话生成、翻译、摘要、纠错、改写、Text2SQL、生成式问答、问题生成等。
- 目前文本生成任务只支持单机单卡。

## 代码结构

- 文本生成任务位于 ./applications/tasks/text_generation

```shell
text_generation/
├── data                                        #数据集存放文件夹
│   └── ernie_gen_dureader
│       ├── dev.tsv
│       ├── test.tsv
│       └── train.tsv
├── data_set_reader
│   └── ernie_gen_infilling_dataset_reader.py
├── examples
│   ├── cls_ernie_gen_infilling_ch_infer.json
│   └── cls_ernie_gen_infilling_ch.json
├── inference
│   ├── custom_inference.py
│   └── __init__.py
├── model
│   ├── ernie_infilling_generation.py
│   └── __init__.py
├── run_infer.py
├── run_trainer_ernie_gen.py
```

## 准备工作

### 数据准备

- 文心预置为**问题生成任务**数据集，该数据集来自**Dureader**，DuReader 是百度在自然语言处理国际顶会 ACL 2018 发布的机器阅读理解数据集，所有的问题、原文都来源于百度搜索引擎数据和百度知道问答社区，答案是由人工整理的。实验是在 DuReader 的单文档、抽取类的子集上进行的，训练集包含15763个文档和问题，验证集包含1628个文档和问题。[链接: https://arxiv.org/pdf/1711.05073.pdf]
- 问题生成任务的训练集、验证集分别存放在./data/ernie_gen_dureader目录下的train.tsv, dev.tsv文件中，训练集、验证集的数据格式相同，数据分为3列，列与列之间用\t进行分割。第一行为列名，从第二行开始为数据样例。数据示例如下：

| answer  | paragraph                                                    | tgt                  |
| ------- | ------------------------------------------------------------ | -------------------- |
| XPE     | 爬行垫根据中间材料的不同可以分为:XPE爬行垫、EPE爬行垫、EVA爬行垫、PVC爬行垫；其中XPE爬行垫、EPE爬行垫都属于PE材料加保鲜膜复合而成，都是无异味的环保材料，但是XPE爬行垫是品质较好的爬行垫，韩国进口爬行垫都是这种爬行垫，而EPE爬行垫是国内厂家为了减低成本，使用EPE(珍珠棉)作为原料生产的一款爬行垫，该材料弹性差，易碎，开孔发泡防水性弱。EVA爬行垫、PVC爬行垫是用EVA或PVC作为原材料与保鲜膜复合的而成的爬行垫，或者把图案转印在原材料上，这两款爬行垫通常有异味，如果是图案转印的爬 | 爬行垫什么材质的好   |
| 360影视 | 在360影视里看试试看吧沈珍珠已经决心入宫，她向安庆绪辞行，安庆绪冲动地要带着她逃走，被安禄山带人拦住。安庆绪想告诉珍珠安禄山其实是在利用她，但遭到安禄山的警告，他担心珍珠会因此遭遇不测，只好三缄其口。结果，他被安禄山命人拉入密室暴打，珍珠则独自启程去了长安。 | 大唐荣耀哪里能看全集 |

- 问题生成任务可以通过已知的文章和用户回答，生成相关的问题来增加搜索引擎的泛化能力
- 在该数据集中分为两类文本：源文本、目标文本，问题生成任务就是根据源文本来生成目标文本：
  - 源文本：
    - answert：答案
    - paragraph：文章
  - 目标文本：
    - tgt：问题

### 网络（模型）选择

- 文心预置的可用于生成任务的模型源文件在./applications/tasks/text_generation/model/目录下

| 网络名称（py文件的类型）                                | 简介                                                         | 支持类型             | 备注       |
| ------------------------------------------------------- | ------------------------------------------------------------ | -------------------- | ---------- |
| ErnieInfillingGeneration(ernie_infilling_generation.py) | ERNIE-GEN是针对通用生成任务的预训练模型，在4类生成任务的5个英文公开数据集上超过微软 MASS 和 UNILM、Facebook BART、谷歌 T5 等参数规模更大、预训练数据更多的竞品模型取得SOTA效果，在中文任务上较通用ERNIE模型提升显著 | 通用生成Finetune任务 | 序列到序列 |

### ERNIE预训练模型下载

- 文心提供的[ERNIE预训练模型](../../models_hub)的参数文件和配置文件在 applications/applications/models_hub目录下，使用对应的sh脚本，即可拉取对应的模型、字典、必要环境等文件。

| 模型名称  | 下载脚本                         | 备注                                                     |
| --------- | -------------------------------- | -------------------------------------------------------- |
| ERNIE-Gen | sh download_ernie_gen_base_ch.sh | 下载并解压后，参数、字典和配置存放于ernie_gen_ch_dir目录 |

### 模型评估指标选择

- 采用序列生成常用指标：BLEU-4与ROUGE-L 。

## 开始训练

- 进入指定任务的目录：./applications/tasks/text_generation

```
cd ./applications/tasks/text_generation
```

### 训练的配置文件

- 配置文件：./examples/cls_ernie_gen_infilling_ch.json

```json
{
  "dataset_reader": {
    "train_reader": {
      "name": "train_reader",
      "type": "InfillingGenReader",
      "fields": [],
      "config": {
        "data_path": "./data/ernie_gen_dureader/train.tsv",
        "shuffle": true,
        "batch_size": 8,
        "epoch": 20,
        "sampling_rate": 1.0,
        "need_data_distribute": true,
        "extra_params":{
          "vocab_path":"../../models_hub/ernie_gen_ch_dir/vocab_ernie_gen_ch.txt",
          "max_seq_len":512,
          "do_lower_case":true,
          "in_tokens":false,
          "tokenizer": "FullTokenizer",
          "tgt_type_id": 7,
          "max_src_len": 320,
          "max_tgt_len": 64,
          "max_dec_len": 32
        }
      }
    },
    "dev_reader": {
      "name": "dev_reader",
      "type": "InfillingGenReader",
      "fields": [],
      "config": {
        "data_path": "./data/ernie_gen_dureader/dev.tsv",
        "shuffle": false,
        "batch_size": 16,
        "epoch": 1,
        "sampling_rate": 1.0,
        "need_data_distribute": true,
        "extra_params":{
          "vocab_path":"../../models_hub/ernie_gen_ch_dir/vocab_ernie_gen_ch.txt",
          "max_seq_len":512,
          "do_lower_case":true,
          "in_tokens":false,
          "tokenizer": "FullTokenizer",
          "tgt_type_id": 7,
          "max_src_len": 320,
          "max_tgt_len": 64,
          "max_dec_len": 32
        }
      }
    }
  },
  "model": {
    "type": "ErnieInfillingGeneration",
    "is_dygraph": 0,
    "embedding": {
      "emb_dim": 768,
      "use_amp": false,
      "config_path": "../../models_hub/ernie_gen_ch_dir/ernie_gen_ch_config.json"
    },
    "optimization":{
      "learning_rate": 5e-5,
      "use_lr_decay": true,
      "use_default_decay": false,
      "lr_scheduler": "linear_warmup_decay",
      "epsilon": 1e-6,
      "warmup_steps": 0,
      "warmup_proportion": 0.1,
      "weight_decay": 0.01,
      "use_dynamic_loss_scaling": false,
      "init_loss_scaling": 128,
      "incr_every_n_steps": 100,
      "decr_every_n_nan_or_inf": 2,
      "incr_ratio": 2.0,
      "decr_ratio": 0.8
    },
    "label_smooth": 0.1,
    "beam_size": 4,
    "weight_sharing": true,
    "length_penalty": 1.0
  },
  "trainer": {
    "type" : "CustomGenerationTrainer",
    "PADDLE_PLACE_TYPE": "gpu",
    "ramdom_seed": 1,
    "save_inference_model": true,
    "train_log_step": 10,
    "is_do_train": 1,
    "is_eval_dev": 1,
    "is_eval_test": 0,
    "eval_step": 10000,
    "save_model_step": 100000000,
    "load_parameters": "",
    "load_checkpoint": "",
    "pre_train_model": [
      {
        "name": "ernie_gen_ch",
        "params_path": "../../models_hub/ernie_gen_ch_dir/params_dec"
      }
    ],
    "output_path": "./output/ernie_gen_ch"
  }
}
```

### 训练ERNIE-Gen

```shell
python run_trainer_ernie_gen.py --param_path ./examples/cls_ernie_gen_infilling_ch.json
```

- 训练模型保存于./output/ernie_gen_ch文件夹下（可在配置文件./examples/cls_ernie_gen_infilling_ch.json中修改输出路径），其中save_inference_model/文件夹会保存用于预测的模型文件，save_checkpoint/文件夹会保存用于热启动的模型文件
- 训练模型的日志文件保存于./log文件夹下

## 开始预测

### 预测的配置文件

- 配置文件：./examples/cls_ernie_gen_infilling_ch_infer.json
- 在配置文件./examples/cls_ernie_gen_infilling_ch_infer.json中需要更改 inference.inference_model_path 为上面训练过程中所保存的**预测模型的路径**

```
{
  "dataset_reader": {
    "predict_reader": {
      "name": "predict_reader",
      "type": "InfillingGenReader",
      "fields": [],
      "config": {
        "data_path": "./data/ernie_gen_dureader/test.tsv",
        "shuffle": false,
        "batch_size": 8,
        "epoch": 1,
        "sampling_rate": 1.0,
        "need_data_distribute": false,
        "need_generate_examples": true,
        "extra_params":{
          "vocab_path":"../../models_hub/ernie_gen_ch_dir/vocab_ernie_gen_ch.txt",
          "max_seq_len":512,
          "do_lower_case":true,
          "in_tokens":false,
          "tokenizer": "FullTokenizer",
          "tgt_type_id": 7,
          "max_src_len": 320,
          "max_tgt_len": 64,
          "max_dec_len": 32
        }
      }
    }
  },
  "model": {
    "type": "ErnieInfillingGeneration",
    "is_dygraph": 0,
    "embedding": {
      "emb_dim": 768,
      "use_amp": false,
      "config_path": "../../models_hub/ernie_gen_ch_dir/ernie_gen_ch_config.json"
    },
    "optimization":{
      "learning_rate": 5e-5,
      "use_lr_decay": true,
      "use_default_decay": false,
      "lr_scheduler": "linear_warmup_decay",
      "epsilon": 1e-6,
      "warmup_steps": 0,
      "warmup_proportion": 0.1,
      "weight_decay": 0.01,
      "use_dynamic_loss_scaling": false,
      "init_loss_scaling": 128,
      "incr_every_n_steps": 100,
      "decr_every_n_nan_or_inf": 2,
      "incr_ratio": 2.0,
      "decr_ratio": 0.8
    },
    "label_smooth": 0.1,
    "beam_size": 4,
    "weight_sharing": true,
    "length_penalty": 1.0
  },
  "inference": {
    "type": "CustomInference",
    "output_path": "./output/predict_result.txt",  #预测输出文件路径
    "PADDLE_PLACE_TYPE": "gpu",
    "thread_num": 1,
    "inference_model_path": "output/ernie_gen_ch/save_inference_model/inference_step_39421/", ## 配置对应预测模型的路径
    "extra_param": {
      "meta":{
        "job_type": "text_generation"
      }

    }
  }
}
```

### ERNIE-Gen预测

```shell
python run_infer.py --param_path ./examples/cls_ernie_gen_infilling_ch_infer.json
```

- 预测结果保存于./output/predict_result.txt文件中（可在./examples/cls_ernie_gen_infilling_ch_infer.json中修改输出路径）
