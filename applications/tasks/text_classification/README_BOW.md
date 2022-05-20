# Bow(BowClassification)模型训练、预测

## 开始训练

- 进入分类任务的目录./erniekit_appzoo/tasks/text_classification

```
cd ./erniekit_appzoo/tasks/text_classification
```

### 训练的配置文件

- 配置文件 ./examples/cls_bow_ch.json
- 文心中的各种参数都是在json文件中进行配置的，您可以通过修改所加载的json文件来进行参数的自定义配置。json配置文件主要分为三个部分：
  - dataset_reader
  - model
  - trainer

```
{                                        
  "dataset_reader": {                ## 用于配置模型训练或者预测时的数据相关配置，训练任务的dataset_reader中必须有train_reader、test_reader、dev_reader，预测推理任务的dataset_reader仅需要predict_reader。
    "train_reader": {                ## 训练、验证、测试各自基于不同的数据集，数据格式也可能不一样，可以在json中配置不同的reader，此处为训练集的reader。
      "name": "train_reader",
      "type": "BasicDataSetReader",    ## 采用BasicDataSetReader，其封装了常见的读取tsv、txt文件、组batch等操作。
      "fields": [                ## 域（field）是文心的高阶封装，对于同一个样本存在不同域的时候，不同域有单独的数据类型（文本、数值、整型、浮点型）、单独的词表(vocabulary)等，可以根据不同域进行语义表示，如文本转id等操作，field_reader是实现这些操作的类。
        {
          "name": "text_a",    ## 文本分类只有一个文本特征域，命名为"text_a"。
          "data_type": "string",    ## data_type定义域的数据类型，文本域的类型为string，整型数值为int，浮点型数值为float。
          "reader": {
            "type": "CustomTextFieldReader"    ## 采用针对文本域的通用reader "CustomTextFieldReader"。数值数组类型域为"ScalarArrayFieldReader"，数值标量类型域为"ScalarFieldReader"，这里的取值是对应FieldReader的类名
          },
          "tokenizer": {
            "type": "CustomTokenizer",        ## 指定该文本域的tokenizer为CustomTokenizer，type的取值是对应Tokenizer的类名
            "split_char": " ",            ## 非Ernie任务需要自己切词，切词之后的明文使用的分隔符在这里设置，默认是通过空格区分不同的token。
            "unk_token": "[UNK]",        ## unk标记为"[UNK]"， 即词表之外的token所对应的默认id，unk必须是词表文件中存在的token。
            "params": null                ## 如果需要一些额外的参数传入tokenizer的时候可以使用该字段
          },
          "need_convert": true,            ## "need_convert"为true说明数据格式是明文字符串，需要通过词表转换为id。
          "vocab_path": "./data/dict/vocab.txt",    ## 指定该文本域的词表，"need_convert"为true时一定要设置
          "max_seq_len": 512,            ## 设定当前域转为id之后的最大长度
          "truncation_type": 0,        ## 选择文本超长截断的策略，0为从头开始到最大长度截断，1为从头开始到max_len-1的位置截断，末尾补上最后一个id（词或字），2为保留头和尾两个位置，然后按从头开始到最大长度方式截断。
          "padding_id": 0                ## 设定padding时对应的id值，文心内部会按batch中的最长文本大小对整个batch中的数据进行padding补齐。
        },                            ## 如果每一个样本有多个特征域（文本类型、数值类型均可），可以仿照前面对每个域进行设置，依次增加每个域的配置即可。此时样本的域之间是以\t分隔的。
        {
          "name": "label",            ## 标签也是一个单独的域，在当前例子中命名为"label"。如果多个不同任务体系的标签存在于多个域中，则可实现最基本的多任务学习。
          "data_type": "int",            ## 标签是整型数值。
          "reader": {
            "type": "ScalarFieldReader"    ## 整型数值域的reader为"ScalarFieldReader"
          },
          "tokenizer": null,            ## 如果你的label是明文文本，且需要分词的话，这里就需要配置对应的tokenizer，规则如上方文本域的tokenizer配置
          "need_convert": false,        ## "need_convert"为true说明数据格式是明文字符串，需要通过词表转换为id。
          "vocab_path": "",            ## "need_convert"为true的时候需要填词表路径 。
          "max_seq_len": 1,            ## 设定每个域的最大长度，当前例子中的label域是一个int数值，所以最大长度是1。
          "truncation_type": 0,        ## 超过max_seq_len长度之后的截断策略，同上。
          "padding_id": 0,            ## 设定padding时对应的id值。
          "embedding": null             ## 历史遗留参数，设置为null即可。
        }
      ],
      "config": {
        "data_path": "./data/train_data",    ## 训练数据train_reader的数据路径，只写到文件夹目录。
        "shuffle": false,                    ## 数据在读取过程中是否需要打乱顺序。
        "batch_size": 128,                    ## 超参数之一，表示每个step训练多少个样本。
        "epoch": 5,                            ## 超参数之一，表示这个数据集中的数据会被重复训练多少轮。
        "sampling_rate": 1.0,                ## 数据集的采样率，文心预留参数，暂时不起作用，后续版本会升级。
        "need_data_distribute": true,        ## 表示数据读取过程中是否需要按卡数进行分发，true表示每张卡在同一个step中读取到的数据是不一样的，false表示每张卡在同一个step中读取到的数据是一样的，训练集默认为true，测试集、验证集、预测集都是false。
        "need_generate_examples": false,    ## 表示在数据读取过程中除了id化好的tensor数据外，是否需要返回原始明文样本，测试集默认取值为true，训练集、验证集为false
        "key_tag": false                    ## 需全部置为false
      }
    },
    "test_reader": {        ## 若要评估测试集，需配置test_reader，其配置方式与train_reader类似， 需要注意的是shuffle参数要设置为false，epoch参数必须是1。
      "name": "test_reader",
      "type": "BasicDataSetReader",
      "fields": [
        {
          "name": "text_a",
          "data_type": "string",
          "reader": {
            "type": "CustomTextFieldReader"
          },
          "tokenizer": {
            "type": "CustomTokenizer",
            "split_char": " ",
            "unk_token": "[UNK]",
            "params": null
          },
          "need_convert": true,
          "vocab_path": "./data/dict/vocab.txt",
          "max_seq_len": 512,
          "truncation_type": 0,
          "padding_id": 0,
          "embedding": null
        },
        {
          "name": "label",
          "data_type": "int",
          "reader": {
            "type": "ScalarFieldReader"
          },
          "tokenizer": null,
          "need_convert": false,
          "vocab_path": "",
          "max_seq_len": 1,
          "truncation_type": 0,
          "padding_id": 0,
          "embedding": null
        }
      ],
      "config": {
        "data_path": "./data/test_data",
        "shuffle": false,
        "batch_size": 128,
        "epoch": 1,
        "sampling_rate": 1.0,
        "need_data_distribute": false,
        "need_generate_examples": false,
        "key_tag": false
      }
    },
    "dev_reader": {        ## 若要评估验证集，需配置dev_reader，其配置方式与test_reader类似，需要注意的是shuffle参数要设置为false，epoch参数必须是1。
      "name": "dev_reader",
      "type": "BasicDataSetReader",
      "fields": [
        {
          "name": "text_a",
          "data_type": "string",
          "reader": {
            "type": "CustomTextFieldReader"
          },
          "tokenizer": {
            "type": "CustomTokenizer",
            "split_char": " ",
            "unk_token": "[UNK]",
            "params": null
          },
          "need_convert": true,
          "vocab_path": "./data/dict/vocab.txt",
          "max_seq_len": 512,
          "truncation_type": 0,
          "padding_id": 0,
          "embedding": null
        },
        {
          "name": "label",
          "data_type": "int",
          "reader": {
            "type": "ScalarFieldReader"
          },
          "tokenizer": null,
          "need_convert": false,
          "vocab_path": "",
          "max_seq_len": 1,
          "truncation_type": 0,
          "padding_id": 0,
          "embedding": null
        }
      ],
      "config": {
        "data_path": "./data/dev_data",
        "shuffle": false,
        "batch_size": 128,
        "epoch": 1,
        "sampling_rate": 1.0,
        "need_data_distribute": false,
        "need_generate_examples": false,
        "key_tag": false
      }
    }
  },
  "model": {
    "type": "BowClassification",    ## 文心采用模型(models)的方式定义神经网络的基本操作，本例采用预置的模型BowClassification实现文本分类，具体网络可参考models目录。
    "is_dygraph": 1,                ## 区分动态图模型和静态图模型，1表示动态图，0表示静态图
    "optimization": {
      "learning_rate": 2e-05        ## 预置模型的优化器所需的参数配置，如学习率等。
    },
    "vocab_size": 33261,            ## 该模型（model）使用的词表大小，必填参数。
    "num_labels": 2                    ## 该分类模型的类别数目是多少，必填参数，不填则默认是二分类
  },    
  "trainer": {
    "type": "CustomDynamicTrainer",     ## 上面设置为0表示静态图时，此处改为CustomTrainer
    "PADDLE_PLACE_TYPE": "cpu",        ## 表示运行时的设备类别，取值为cpu和gpu。 
    "PADDLE_IS_FLEET": 0,                ## 表示是否使用fleetrun模式进行训练，gpu多卡情况下必须设置为1，并使用fleetrun命令进行训练。
    "train_log_step": 20,                ## 训练时打印训练日志的间隔步数，
    "is_eval_dev": 0,                    ## 是否在训练的时候评估开发集，如果取值为1，则一定需要配置dev_reader及其数据路径
    "is_eval_test": 1,                    ## 是否在训练的时候评估测试集，如果取值为1，则一定需要配置test_reader及其数据路径
    "eval_step": 100,                    ## 进行测试集或验证集评估的间隔步数
    "save_model_step": 10000,            ## 保存模型时的间隔步数，建议设置为eval_step的整数倍
    "load_parameters": "",                ## 加载包含各op参数值的训练好的模型，用于热启动。此处填写checkpoint路径。不填则表示不使用热启动
    "load_checkpoint": "",                ## 加载包含学习率等所有参数的训练模型，用于热启动。此处填写checkpoint路径。不填则表示不使用热启动
    "pre_train_model": [],                ## 加载预训练模型，ERNIE任务的必填参数，非ERNIE任务将当前参数置为[]即可。
    "output_path": "./output/cls_bow_ch",    ## 保存模型的输出路径，如置空或者不配置则默认输出路径为"./output"
    "extra_param": {                     ## 除核心必要信息之外，需要额外标明的参数信息，比如一些meta信息可以作为日志统计的关键字
      "meta":{
        "job_type": "text_classification"
      }

    }
  }
}
```

### 训练BOW模型

```
python run_trainer.py --param_path ./examples/cls_bow_ch.json
```

- 训练模型保存于./output/cls_bow_ch文件夹下（可在配置文件./examples/cls_bow_ch.json中修改输出路径），其中save_inference_model/文件夹会保存用于预测的模型文件，save_checkpoint/文件夹会保存用于热启动的模型文件
- 训练模型的日志文件保存于./log文件夹下

## 开始预测

### 预测的配置文件

- 配置文件 ./examples/cls_bow_ch_infer.json 
- 在配置文件./examples/cls_bow_ch_infer.json中需要更改 inference.inference_model_path 为上面训练过程中所保存的**预测模型的路径**

```
{
  "dataset_reader": {
    "predict_reader": {        ## 如果是预测推理，则必须配置predict_reader，其配置方式与train_reader、test_reader类似，需要注意的是predict_reader不需要label域，shuffle参数必须是false，epoch参数必须是1。
      "name": "predict_reader",    
      "type": "BasicDataSetReader",
      "fields": [
        {
          "name": "text_a",
          "data_type": "string",
          "reader": {
            "type": "CustomTextFieldReader"
          },
          "tokenizer": {
            "type": "CustomTokenizer",
            "split_char": " ",
            "unk_token": "[UNK]",
            "params": null
          },
          "need_convert": true,
          "vocab_path": "./data/dict/vocab.txt",
          "max_seq_len": 512,
          "truncation_type": 0,
          "padding_id": 0,
          "embedding": null
        }
      ],
      "config": {
        "data_path": "./data/predict_data",
        "shuffle": false,        ## 注意！这里的参数必须关掉，打乱顺序输出之后不方便比对数据看结果
        "batch_size": 8,
        "epoch": 1,                ## 注意！这里的epoch要设置为1，重复多次预测没意义。
        "sampling_rate": 1.0,
        "need_data_distribute": false,
        "need_generate_examples": true,
        "key_tag": false
      }
    }
  },

  "inference": {    ## 用于配置模型预测推理的启动器，包括待预测模型路径、结果输出等参数。
    "type": "CustomInference",
    "output_path": "./output/predict_result.txt", ## 预测结果的输出路径，如果不填则默认输出路径为"./output/predict_result.txt"
    "PADDLE_PLACE_TYPE": "cpu",
    "num_labels": 2,        ## 必填参数，表示分类模型的类别数目是多少，预测结果解析时会用到
    "thread_num": 2,
    "inference_model_path": "./output/cls_bow_ch/save_inference_model_0/inference_step_381/", ## 待预测模型的路径
    "extra_param": { ## 同trainer，除核心必要信息之外，需要额外标明的参数信息，比如一些meta信息可以作为日志统计的关键字。 
      "meta":{
        "job_type": "text_classification"
      }

    }
  }
}
```

### Bow模型预测

```
python run_infer.py --param_path ./examples/cls_bow_ch_infer.json
```

- 预测结果保存于./output/predict_result.txt文件中（可在./examples/cls_bow_ch_infer.json中修改输出路径）
