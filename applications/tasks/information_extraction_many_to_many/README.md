# 多对多信息抽取

- 实体关系，实体属性抽取是信息抽取的关键任务；实体关系抽取是指从一段文本中抽取关系三元组，实体属性抽取是指从一段文本中抽取属性三元组；信息抽取一般分以下几种情况一对一，一对多，多对一，多对多的情况：
- 一对一：“张三男汉族硕士学历”含有一对一的属性三元组（张三，民族，汉族）。
- 一对多：“华扬联众数字技术股份有限公司于2017年8月2日在上海证券交易所上市”，含有一对多的属性三元组（华扬联众数字技术股份有限公司，上市时间，2017年8月2日）和（华扬联众数字技术股份有限公司，上市地点，上海证券交易所上市）
- 多对一：“上海森焱软件有限公司和上海欧提软件有限公司的注册资本均为100万人民币”，含有多对一的属性三元组（上海森焱软件有限公司，注册资本，100万人民币）和（上海欧提软件有限公司，注册资本，100万人民币）
- 多对多：“大华种业稻麦种子加工36.29万吨、销售37.5万吨；苏垦米业大米加工22.02万吨、销售24.86万吨”，含有多对多的属性三元组（大华种业，稻麦种子产量，36.29万吨）和（苏垦米业，大米加工产量，22.02万吨）

## 代码结构

- 多对多信息抽取任务位于 ./applications/tasks/information_extraction_many_to_many

```
information_extraction_many_to_many/
├── data
│   ├── DuIE2.0
│   │   └── convert_data.py
│   ├── entity_attribute_data
│   │   ├── dev_data
│   │   │   └── dev.json
│   │   ├── predict_data
│   │   │   └── predict.json
│   │   ├── test_data
│   │   │   └── test.json
│   │   └── train_data
│   │       └── train.json
│   └── entity_relation_data
│       ├── dev_data
│       │   └── dev.json
│       ├── predict_data
│       │   └── predict.json
│       ├── test_data
│       │   └── test.json
│       └── train_data
│           └── train.json
├── data_set_reader
│   └── ie_data_set_reader.py
├── dict
│   ├── entity_attribute_label_map.json
│   └── entity_relation_label_map.json
├── examples
│   ├── many_to_many_ie_attribute_ernie_fc_ch_infer.json
│   ├── many_to_many_ie_attribute_ernie_fc_ch.json
│   ├── many_to_many_ie_relation_ernie_fc_ch_infer.json
│   └── many_to_many_ie_relation_ernie_fc_ch.json
├── inference
│   ├── custom_inference.py
│   └── __init__.py
├── model
│   ├── ernie_fc_ie_many_to_many.py
│   └── __init__.py
├── run_infer.py
├── run_trainer.py
└── trainer
    ├── custom_dynamic_trainer.py
    ├── custom_trainer.py
    └── __init__.py
```

## 准备工作：数据准备、网络（模型）选择、ERNIE预训练模型选择

#### 数据准备

- 这里我们提供三份已标注的数据集：属性抽取数据集（demo示例数据集）、关系抽取数据集（demo示例数据集）、[DuIE2.0](https://www.luge.ai/#/luge/dataDetail?id=5)（全量数据集）。
- 属性抽取训练集、测试集、验证集和预测集分别存放在./data/entity_attribute_data目录下的train_data、test_data、dev_data和predict_data文件夹下，对应的示例标签词表存放在./dict目录下。
- 关系抽取训练集、测试集、验证集和预测集分别存放在./data/entity_relation_data目录下的train_data、test_data、dev_data和predict_data文件夹下，对应的示例标签词表存放在./dict目录下。
- DuIE2.0需要去官网下载，下载链接：https://www.luge.ai/#/luge/dataDetail?id=5
- 注：数据集（包含词表）均为utf-8格式。

##### Demo示例数据集（属性抽取数据集、关系抽取数据集）

- demo示例数据集中属性抽取数据集与关系抽取数据集的结构一样，他们都只包含少量数据集，可用于快速开始模型的训练与预测。

###### 训练集/测试集/

- 训练集、测试集的数据格式相同，每个样例分为两个部分文本和对应标签

```
{"text": "倪金德，1916年生，奉天省营口（今辽宁省营口市）人", "spo_list": [{"predicate": "出生日期", "subject": [0, 3], "object": [4, 9]}, {"predicate": "出生地", "subject": [0, 3], "object": [11, 16]}]}
{"text": "基本介绍克里斯蒂娜·塞寇丽（Christina Sicoli）身高163cm，在加拿大安大略出生和长大，毕业于伦道夫学院", "spo_list": [{"predicate": "毕业院校", "subject": [4, 13], "object": [55, 60]}]}
```

###### 预测集

- 预测集只有一个key（"text"）：

```
{"text": "倪金德，1916年生，奉天省营口（今辽宁省营口市）人"}
{"text": "基本介绍克里斯蒂娜·塞寇丽（Christina Sicoli）身高163cm，在加拿大安大略出生和长大，毕业于伦道夫学院"}
```

###### 标签词表

- 标签列表是一个json字符串，key是标签值，value是标签对应id，示例词表采用BIO标注，B表示关系，分为主体（S）与客体（O），如下所示：

```
{
     "O": 0,
     "I": 1,
     "B-毕业院校@S": 2,
     "B-毕业院校@O": 3,
     "B-出生地@S": 4,
     "B-出生地@O": 5,
     "B-祖籍@S": 6,
     "B-祖籍@O": 7,
     "B-国籍@S": 8,
     "B-国籍@O": 9,
     "B-出生日期@S": 10,
     "B-出生日期@O": 11
}
```

- 注意：O, I对应的ID必须是0， 1，B-XXX@O对应的id需要必须为B-XXX@S对应的id+1（B-XXX@s须为偶数,B-XXX@O须为奇数）

##### DuIE2.0数据集

- DuIE2.0是业界规模最大的中文关系抽取数据集，其schema在传统简单关系类型基础上添加了多元复杂关系类型，此外其构建语料来自百度百科、百度信息流及百度贴吧文本，全面覆盖书面化表达及口语化表达语料，能充分考察真实业务场景下的关系抽取能力。下载链接：https://www.luge.ai/#/luge/dataDetail?id=5
- DuIE2.0数据集的格式与本框架所需要的文本输入格式不一致，需要进行转化成**demo示例数据集**的格式才能使用，具体转化步骤如下：
  - 从链接 https://www.luge.ai/#/luge/dataDetail?id=5 下载数据集到 ./data/DuIE2.0 文件夹中，并解压
  - 进入./data/DuIE2.0目录

```
cd ./data
mkdir DuIE2.0
cd DuIE2.0
#下载DuIE2.0数据并解压，下载地址https://www.luge.ai/#/luge/dataDetail?id=5 
```

- - 运行./data/DuIE2.0/convert_data.py 脚本

```
python convert_data.py
```

### 网络（模型）选择

- 文心预置的可用于生成任务的模型源文件在applications/applications/tasks/text_generation/model/目录下

| 网络名称（py文件的类型）               | 简介                                                         | 支持类型                 | 备注     |
| -------------------------------------- | ------------------------------------------------------------ | ------------------------ | -------- |
| ErnieFcIe(ernie_fc_ie_many_to_many.py) | ErnieFcIe多对多信息抽取任务模型源文件，可加载ERNIE2.0-Base、ERNIE2.0-large、ERNIE3.0-Base、ERNIE3.0-x-Base、ERNIE3.0-Medium | 通用信息抽取Finetune任务 | 信息抽取 |

### ERNIE预训练模型下载

- 文心提供的[ERNIE预训练模型](../../models_hub)的参数文件和配置文件在 applications/applications/models_hub目录下，使用对应的sh脚本，即可拉取对应的模型、字典、必要环境等文件。

| 模型名称        | 下载脚本                           | 备注                                       |
| --------------- | ---------------------------------- | ------------------------------------------ |
| ERNIE2.0-Base   | sh download_ernie_2.0_base_ch.sh   | 下载并解压后得到对应模型的参数、字典和配置 |
| ERNIE2.0-large  | sh download_ernie_2.0_large_ch.sh  |                                            |
| ERNIE3.0-Base   | sh download_ernie_3.0_base_ch.sh   |                                            |
| ERNIE3.0-x-Base | sh download_ernie_3.0_x_base_ch.sh |                                            |
| ERNIE3.0-Medium | sh download_ernie_3.0_medium.sh    |                                            |

### 模型评估指标选择

- 采用评估指标：precison、recall、f1

## 开始训练

- 以属性抽取数据集的训练为例，进入指定任务目录./applications/tasks/information_extraction_many_to_many

```
cd ./applications/tasks/information_extraction_many_to_many
```

#### 训练的配置文件

- 配置文件：./examples/many_to_many_ie_attribute_ernie_fc_ch.json

```
{
  "dataset_reader": {
    "train_reader": {
      "name": "train_reader",
      "type": "IEReader",
      "fields": [],
      "config": {
        "data_path": "./data/entity_attribute_data/train_data/",
        "shuffle": false,
        "batch_size": 2,
        "epoch": 5,
        "sampling_rate": 1.0,
        "need_data_distribute": true,
        "need_generate_examples": false,
        "extra_params": {
          "vocab_path": "../../models_hub/ernie_3.0_base_ch_dir/vocab.txt",        #选择对应预训练模型的词典路径，在models_hub路径下
          "label_map_config": "./dict/entity_attribute_label_map.json",
          "num_labels": 12,
          "max_seq_len": 512,
          "do_lower_case":true,
          "in_tokens":false,
          "tokenizer": "FullTokenizer"
        }
      }
    },
    "test_reader": {
      "name": "test_reader",
      "type": "IEReader",
      "fields": [],
      "config": {
        "data_path": "./data/entity_attribute_data/test_data/",
        "shuffle": false,
        "batch_size": 2,
        "epoch": 1,
        "sampling_rate": 1.0,
        "need_data_distribute": false,
        "need_generate_examples": false,
        "extra_params": {
          "vocab_path": "../../models_hub/ernie_3.0_base_ch_dir/vocab.txt",  #选择对应预训练模型的词典路径，在models_hub路径下
          "label_map_config": "./dict/entity_attribute_label_map.json",
          "num_labels": 12,
          "max_seq_len": 512,
          "do_lower_case":true,
          "in_tokens":false,
          "tokenizer": "FullTokenizer"
        }
      }
    }
  },
  "model": {
    "type": "ErnieFcIe",
    "is_dygraph":1,
    "num_labels":12,
    "optimization": {
      "learning_rate": 5e-05,
      "use_lr_decay": true,
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
    "embedding": {
      "config_path": "../../models_hub/ernie_3.0_base_ch_dir/ernie_config.json"  #选择对应预训练模型的配置文件路径，在models_hub路径下
    }
  },
  "trainer": {
    "type": "CustomDynamicTrainer",
    "PADDLE_PLACE_TYPE": "gpu",
    "PADDLE_IS_FLEET": 0,
    "train_log_step": 10,
    "use_amp": true,
    "is_eval_dev": 0,
    "is_eval_test": 1,
    "eval_step": 50,
    "save_model_step": 100,
    "load_parameters": "",
    "load_checkpoint": "",
    "pre_train_model": [
      {
        "name": "ernie_3.0_base_ch",
        "params_path": "../../models_hub/ernie_3.0_base_ch_dir/params"   #选择对应预训练模型的参数路径，在models_hub路径下
      }
    ],
    "output_path": "./output/ie_attribute_ernie_3.0_base_fc_ch",    #输出路径
    "extra_param": {
      "meta":{
        "job_type": "entity_attribute_extraction"
      }
    }
  }
}
```

#### 训练ERNIE模型

- 基于示例的数据集，可以运行以下命令在训练集（train.txt）上进行模型训练，并在测试集（test.txt）上进行验证；

```
python run_trainer.py --param_path ./examples/many_to_many_ie_attribute_ernie_fc_ch.json
```

- 训练模型保存于./output/ie_attribute_ernie_3.0_base_fc_ch文件夹下（可在配置文件./examples/many_to_many_ie_attribute_ernie_fc_ch.json中修改输出路径），其中save_inference_model/文件夹会保存用于预测的模型文件，save_checkpoint/文件夹会保存用于热启动的模型文件
- 训练模型的日志文件保存于./log文件夹下

## 开始预测

- 以属性抽取数据集的预测为例

#### 预测的配置文件

- 配置文件：./examples/many_to_many_ie_attribute_ernie_fc_ch_infer.json
- 在配置文件./examples/many_to_many_ie_attribute_ernie_fc_ch_infer.json中需要更改 inference.inference_model_path 为上面训练过程中所保存的**预测模型的路径**

```
{
  "dataset_reader": {
      "predict_reader": {
          "name": "predict_reader",
          "type": "IEReader",
          "fields": [],
          "config": {
              "data_path": "./data/entity_attribute_data/predict_data/",
              "shuffle": false,
              "batch_size": 2,
              "epoch": 1,
              "sampling_rate": 1.0,
              "extra_params": {
                  "vocab_path": "../../models_hub/ernie_3.0_base_ch_dir/vocab.txt",
                  "label_map_config": "./dict/entity_attribute_label_map.json",
                  "num_labels": 12,
                  "max_seq_len": 512,
                  "do_lower_case":true,
                  "in_tokens":false,
                  "tokenizer": "FullTokenizer",
                  "need_data_distribute": false,
                  "need_generate_examples": true
              }
          }
      }
  },

  "inference": {
      "output_path": "./output/predict_result.txt",               #输出文件路径
      "label_map_config": "./dict/entity_attribute_label_map.json",
      "PADDLE_PLACE_TYPE": "gpu",
      "inference_model_path": "./output/ie_attribute_ernie_3.0_base_fc_ch/save_inference_model/inference_step_1000",   #加载推理模型
      "extra_param": {
          "meta": {
              "job_type": "information_extraction"
          }
      }
  }
}
```

#### 预测ERNIE模型

```
python run_infer.py --param_path ./examples/many_to_many_ie_relation_ernie_fc_ch_infer.json
```

- 预测结果保存于./output/predict_result.txt文件中（可在./examples/many_to_many_ie_attribute_ernie_fc_ch_infer.json中修改输出路径）
