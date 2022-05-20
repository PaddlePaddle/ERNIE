# 序列标注

- 序列标注是经典的NLP问题之一，其本质就是对于一个一维线性输入序列中的每个元素打上标签集合y中的某个标签。 所以，其本质上是对线性序列中每个元素根据上下文内容进行分类的问题。一般情况下，对于NLP任务来说，线性序列就是输入的文本，往往可以把一个汉字看做线性序列的一个元素，而不同任务其标签集合代表的含义可能不太相同，但是相同的问题都是：如何根据汉字的上下文给汉字打上一个合适的标签。

## 代码结构

- 序列标注任务位于 ./applications/tasks/sequence_labeling

```
.
├── data                                                                                    ### 示例数据文件夹，包括各任务所需的训练集、测试集、验证集和预测集
│   ├── dev_data
│   │   └── dev.txt
│   ├── predict_data
│   │   └── infer.txt
│   ├── test_data
│   │   └── test.txt
│   └── train_data
│       └── train.txt
│   ├── download_data.sh
├── examples                                                                            ### 各典型网络的json配置文件，infer后缀的为对应的预测配置文件
│   ├── seqlab_ernie_fc_ch.json
│   
│   ├── seqlab_ernie_fc_ch_infer.json
├── inference                                                                            ### 模型预测代码
│   └── custom_inference.py                                                ### 序列标注任务通用的模型预测代码
├── model                                                                                    ### 序列标注任务相关的网络文件
│   ├── ernie_fc_sequence_label.py
├── run_infer.py                                                                    ### 依靠json进行模型预测的入口脚本
├── run_trainer.py                                                                ### 依靠json进行模型训练的入口脚本
└── trainer                                                                                ### 模型训练和评估代码
    ├── custom_dynamic_trainer.py                                    ### 动态库模式下的模型训练评估代码
    └── custom_trainer.py                                                    ### 静态图模式下的模型训练评估代码
└── dict                                                                                ### 模型训练和评估代码
    ├── vocab_label_map.txt                                    ### 序列标注的标签配置
    └── vocab.txt                                                    ### 词典
```

## 准备工作

### 数据准备

- 在文心中，基于ERNIE的模型都不需要用户自己分词和生成词表文件，非ERNIE的模型需要用户自己提前切好词，词之间以空格分隔，并生成词表文件。切词和词表生成可以使用「[分词工具与词表生成工具](../../tools/data/wordseg)」进行处理。
- 文心中的所有数据集、包含词表文件、label_map文件等都必须为为utf-8格式，如果你的数据是其他格式，请使用「[编码识别及转换工具](../../tools/data/data_cleaning)」进行格式转换。
- 文心中的训练集、测试集、验证集、预测集和词表分别存放在./applications/tasks/sequence_labeling/data目录下的train_data、test_data、dev_data、predict_data、dict文件夹下。

#### 训练集/测试集/验证集文件格式

- 在序列标注任务中，训练集、测试集和验证集的数据格式相同，数据分为两列，列与列之间用\t进行分隔。第一列为文本，第二列为标签，如下所示

```plain
海 钓 比 赛 地 点 在 厦 门 与 金 门 之 间 的 海 域 。	O O O O O O O B-LOC I-LOC O B-LOC I-LOC O O O O O O
周 恩 来 总 理 说 ， 那 就 送 一 株 万 古 常 青 的 友 谊 红 杉 吧 ！	B-PER I-PER I-PER O O O O O O O O O O O O O O O O O O O O
沙 特 队 教 练 佩 雷 拉 ： 两 支 队 都 想 胜 ， 因 此 都 作 出 了 最 大 的 努 力 。	B-ORG I-ORG I-ORG O O B-PER I-PER I-PER O O O O O O O O O O O O O O O O O O O O
```

#### 词表

- 词表：./applications/tasks/sequence_labeling/data/dict/vocab.txt 
- 词表文件：词表分为两列，第一列为词，第二列为id（从0开始），列与列之间用**\t**进行分隔，一般是vocab.txt文件。文心的词表中，[PAD]、[CLS]、[SEP]、[MASK]、[UNK]这5个词是必须要有的，若用户自备词表，需保证这5个词是存在的。

```plain
[PAD]	0
[CLS]	1
[SEP]	2
[MASK]	3
，	4
的	5
、	6
一	7
人	8
有	9
是	10
```

#### 标签词表

- 标签词表：./applications/tasks/sequence_labeling/data/dict/vocab_label_map.txt
- 标签词表文件：标签词表分为两列，第一列为词，第二列为id（从0开始），列与列之间用**\t**进行分隔，一般是vocab_label_map.txt文件。标签顺序需要满足对应的标签体系，比如下面是BIO的标签体系，同一类别的标签B要排在I前面，O排在整个标签词表的最后。

```plain
B-PER   0
 I-PER  1
 B-ORG  2
 I-ORG  3
 B-LOC  4
 I-LOC  5
 O  6
```

### 网络（模型）选择

- 文心预置的可用于序列标注的模型在applications/tasks/sequence_labeling/model目录下，目前支持模型的特点如下所示：

| 网络名称（py文件的类名）                    | 简介                                                         | 支持类型 | 支持预训练模型                                               | 备注 |
| ------------------------------------------- | ------------------------------------------------------------ | -------- | ------------------------------------------------------------ | ---- |
| ErnieFcSeqLabel(ernie_fc_sequence_label.py) | 基于ERNIE和FC的序列标注模型，本质是对每个token做分类，分类的类别总数是标签的数目 | 序列标注 | ERNIE2.0-Base、ERNIE2.0-large、ERNIE3.0-Base、ERNIE3.0-x-Base、ERNIE3.0-Medium |      |

### ERNIE预训练模型下载

- 文心提供的[ERNIE预训练模型](../../models_hub)的参数文件和配置文件在applications/models_hub目录下，由对应的download_xx.sh文件是下载得到，包括模型的参数文件、配置文件以及词表等。

| 模型名称        | 下载脚本                           | 备注                                       |
| --------------- | ---------------------------------- | ------------------------------------------ |
| ERNIE2.0-Base   | sh download_ernie_2.0_base_ch.sh   | 下载并解压后得到对应模型的参数、字典和配置 |
| ERNIE2.0-large  | sh download_ernie_2.0_large_ch.sh  |                                            |
| ERNIE3.0-Base   | sh download_ernie_3.0_base_ch.sh   |                                            |
| ERNIE3.0-x-Base | sh download_ernie_3.0_x_base_ch.sh |                                            |
| ERNIE3.0-Medium | sh download_ernie_3.0_medium.sh    |                                            |

### 模型评估指标选择

- 序列标注任务常用的指标有：Acc（准确率）、Precision（精确率）、Recall（召回率）、F1，这几个指标与常见的分类、匹配等任务的计算有所区别，[详情见常用指标解析](https://ai.baidu.com/ai-doc/ERNIE-Ultimate/nkmlroqy2)中的Chunk指标计算。

### 运行环境选择

- 非ERNIE网络，优先考虑CPU机器
- ERNIE网络优先考虑GPU机器，显存大小最好在10G以上。

## 开始训练

- 进入制定任务目录 ./applications/tasks/sequence_labeling

```
cd ./applications/tasks/sequence_labeling
```

### 训练的配置文件

- 配置文件：./examples/seqlab_ernie_fc_ch.json

```
{
  "dataset_reader": {                                  
    "train_reader": {                                   ### 训练、验证、测试各自基于不同的数据集，数据格式也可能不一样，可以在json中配置不同的reader，此处为训练集的reader。
      "name": "train_reader",
      "type": "BasicDataSetReader",                     ### 采用BasicDataSetReader，其封装了常见的读取tsv文件、组batch等操作。
      "fields": [                                       ### 域（field）是文心的高阶封装，对于同一个样本存在不同域的时候，不同域有单独的数据类型（文本、数值、整型、浮点型）、单独的词表(vocabulary)等，可以根据不同域进行语义表示，如文本转id等操作，field_reader是实现这些操作的类。
        {
          "name": "text_a",                             ### 序列标注的文本特征域，命名为"text_a"。
          "data_type": "string",                        ### data_type定义域的数据类型，文本域的类型为string，整型数值为int，浮点型数值为float。
          "reader": {"type":"CustomTextFieldReader"},   ### 采用针对文本域的通用reader "CustomTextFieldReader"。数值数组类型域为"ScalarArrayFieldReader"，数值标量类型域为"ScalarFieldReader"。
          "tokenizer":{
              "type":"CustomTokenizer",                 ### 指定该文本域的tokenizer为CustomTokenizer。
              "split_char":" ",                         ### 通过空格区分不同的token。
              "unk_token":"[UNK]",                      ### unk标记为"[UNK]"。
              "params":null
            },
          "need_convert": true,                         ### "need_convert"为true说明数据格式是明文字符串，需要通过词表转换为id。
          "vocab_path": "../../models_hub/ernie_3.0_base_ch_dir/vocab.txt",             ### 指定该文本域的词表。
          "max_seq_len": 512,                           ### 设定每个域的最大长度。
          "truncation_type": 0,                         ### 选择截断策略，0为从头开始到最大长度截断，1为从头开始到max_len-1的位置截断，末尾补上最后一个id（词或字），2为保留头和尾两个位置，然后按从头开始到最大长度方式截断。
          "padding_id": 0                               ### 设定padding时对应的id值。
        },                                              ### 如果每一个样本有多个特征域（文本类型、数值类型均可），可以仿照前面对每个域进行设置，依次增加每个域的配置即可。此时样本的域之间是以\t分隔的。
        {
          "name": "label",                              ### 标签也是一个单独的域，命名为"label"。如果多个不同任务体系的标签存在于多个域中，则可实现最基本的多任务学习。
          "data_type": "string",                        ### 序列标注任务中，标签是文本类型。
          "reader":{"type":"CustomTextFieldReader"},
          "tokenizer":{
              "type":"CustomTokenizer",
              "split_char":" ",
              "unk_token":"O",
              "params":null
          },
          "need_convert": true,
          "vocab_path": "./dict/vocab_label_map.txt",   ### 配置标签的标注方式
          "max_seq_len": 512,
          "truncation_type": 0,
          "padding_id": 0
        }
      ],
      "config": {
        "data_path": "./data/train_data/",              ### 训练数据train_reader的数据路径，写到文件夹目录。
        "shuffle": false,
        "batch_size": 8,
        "epoch": 10,
        "sampling_rate": 1.0
        "need_data_distribute": true,        ### 表示数据读取过程中是否需要按卡数进行分发，true表示每张卡在同一个step中读取到的数据是不一样的，false表示每张卡在同一个step中读取到的数据是一样的，训练集默认为true，测试集、验证集、预测集都是false。
        "need_generate_examples": false        ### 表示在数据读取过程中除了id化好的tensor数据外，是否需要返回原始明文样本。
      }
    },
    ……
  },
  "model": {
  "type": "ErnieFcSeqLabel",        ### 使用的模型网络类。
    "is_dygraph": 0,
    "optimization": {                                            ### 优化器设置，建议使用文心ERNIE推荐的默认设置。
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
    "embedding": {                                                ### ERNIE中的embedding参数设置，必填参数。
      "config_path": "../../models_hub/ernie_3.0_base_ch_dir/ernie_config.json"        ### 当前ERNIE模型的配置文件，请填入所采用的ERNIE预训练模型对应的模型配置文件。
    }
},
"trainer": {
  "type": "CustomTrainer",        ### 表示使用的trainer对应的类名，注意要区分静态图（CustomTrainer）和动态图(CustomDynamicTrainer)。
  "PADDLE_PLACE_TYPE": "gpu",                ### 表示运行时的设备类别，取值为cpu和gpu。
  "PADDLE_IS_FLEET": 0,                            ### 表示是否使用fleetrun模式进行训练，gpu多卡情况下必须设置为1，并使用fleetrun命令进行训练。
  "train_log_step": 10,                            ### 训练时打印训练日志的间隔步数。
  "use_amp": true,                                    ### 是否开启混合精度模式的训练。
  "is_eval_dev": 0,                                    ### 是否在训练的时候评估验证集，1为需评估，此时必须配置dev_reader。
  "is_eval_test": 1,                                ### 是否在训练的时候评估测试集，1为需评估，此时必须配置test_reader。
  "eval_step": 100,                                    ### 进行测试集或验证集评估的间隔步数。
  "save_model_step": 200,                        ### 保存模型时的间隔步数，建议设置为eval_step的整数倍。
  "load_parameters": "",                        ### 加载已训练好的模型的op参数值，不会加载训练步数、学习率等训练参数，可用于加载预训练模型。如需使用填写具体文件夹路径即可。
  "load_checkpoint": "",                        ### 加载已训练好的模型的所有参数，包括学习率等，可用于热启动。如需使用填写具体文件夹路径即可。
  "pre_train_model": [                            ### 加载预训练模型，ERNIE任务的必填参数，非ERNIE任务置为[]即可。
    {
      "name": "ernie_3.0_base_ch_dir",    ### 预训练模型的名称。
      "params_path": "../../models_hub/ernie_3.0_base_ch_dir/params"        ### 预训练模型的参数目录。
    }
  ],
  "output_path": "./output/seqlab_ernie_3.0_base_fc_ch",                ### 保存模型的输出路径，若为空则默认。为"./output"
  "extra_param": {"meta":{"job_type": "sequence_labeling"}                                    ### 额外的参数信息。
}
}
```

### 训练ERNIE模型

```
python run_trainer.py --param_path ./examples/seqlab_ernie_fc_ch.json
```

- 训练模型保存于./output/seqlab_ernie_3.0_base_fc_ch文件夹下（可在配置文件./examples/seqlab_ernie_fc_ch.json中修改输出路径），其中save_inference_model/文件夹会保存用于预测的模型文件，save_checkpoint/文件夹会保存用于热启动的模型文件
- 训练模型的日志文件保存于./log文件夹下

## 开始预测

### 预测的配置文件

- 配置文件：./examples/seqlab_ernie_fc_ch_infer.json
- 在配置文件./examples/seqlab_ernie_fc_ch_infer.json中需要更改 inference.inference_model_path 为上面训练过程中所保存的**预测模型的路径**

```shell
{
  "dataset_reader": {
    "predict_reader": {
      "name": "predict_reader",
      "type": "BasicDataSetReader",
      "fields": [
        {
          "name": "text_a",
          "data_type": "string",
          "reader": {
            "type": "ErnieTextFieldReader"
          },
          "tokenizer": {
            "type": "FullTokenizer",
            "split_char": " ",
            "unk_token": "[UNK]",
            "params": null
          },
          "need_convert": true,
          "vocab_path": "../../models_hub/ernie_3.0_base_ch_dir/vocab.txt",
          "max_seq_len": 512,
          "truncation_type": 0,
          "padding_id": 0,
          "embedding": null
        }
      ],
      "config": {
        "data_path": "./data/predict_data",
        "shuffle": false,
        "batch_size": 16,
        "epoch": 1,
        "sampling_rate": 1.0,
        "need_data_distribute": false,
        "need_generate_examples": true
      }
    }
  },
  "inference": {
    "PADDLE_PLACE_TYPE": "gpu",
    "PADDLE_IS_LOCAL": 1,
    "is_ernie": true,
    "inference_model_path": "./output/seqlab_ernie_3.0_base_fc_ch/save_inference_model/inference_step_601",  #加载推理模型的路径
    "extra_param": {
      "meta":{
        "job_type": "sequence_labeling"
      }

    }
  }
}
```

### 预测ERNIE模型

```shell
python run_infer.py --param_path ./examples/seqlab_ernie_fc_ch_infer.json
```

- 预测结果保存于./output/predict_result.txt文件中（可在./examples/seqlab_ernie_fc_ch_infer.json中修改输出路径）
