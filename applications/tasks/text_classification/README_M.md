# ERNIE-M(ErnieClassification)模型训练、预测

## 开始训练

- 进入分类任务的目录./applications/tasks/text_classification

```
cd ./applications/tasks/text_classification
```

### 预训练模型准备

- 模型均存放于./applications/models_hub文件夹下，进入该文件夹下载对应ERNIE-M模型

```
cd ../../models_hub
sh download_ernie_m_1.0_base.sh
cd ../tasks/text_classification
```

### 训练的配置文件

- 配置文件：./examples/cls_ernie_m_1.0_base_one_sent.json

```
{
  "dataset_reader": {
    "train_reader": {
      "name": "train_reader",
      "type": "BasicDataSetReader",
      "fields": [
        {
          "name": "text_a",
          "data_type": "string",
          "reader": {"type":"ErnieTextFieldReaderForMultilingual"},
          "tokenizer":{
              "type":"FullTokenizerErnieM",
              "split_char":" ",
              "unk_token":"[UNK]",
              "params":{
                "do_lower_case": "false",
                "spm_model_file": "./data/dict/sentencepiece.bpe.model"
              }
            },
          "need_convert": true,
          "vocab_path": "../../models_hub/ernie_m_1.0_base_dir/erniem.vocab.txt",
          "max_seq_len": 256,
          "truncation_type": 0,
          "padding_id": 1,
          "embedding": {
            "type":"ErnieTokenEmbedding",
            "use_reader_emb":false,
            "emb_dim":768,
            "config_path":"../../models_hub/ernie_m_1.0_base_dir/ernie_m_1.0_base_config.json"
          }
        },
        {
          "name": "label",
          "data_type": "int",
          "reader": {"type":"ScalarFieldReader"},
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
        "data_path": "./data/train_data_one_sent_multilingual",
        "shuffle": true,
        "batch_size": 32,
        "epoch": 3,
        "sampling_rate": 1.0,
        "key_tag": false,
        "need_data_distribute": true,
        "need_generate_examples": false
      }
    },
    "test_reader": {
      "name": "test_reader",
      "type": "BasicDataSetReader",
      "fields": [
        {
          "name": "text_a",
          "data_type": "string",
          "reader": {"type":"ErnieTextFieldReaderForMultilingual"},
          "tokenizer":{
              "type":"FullTokenizerErnieM",
              "split_char":" ",
              "unk_token":"[UNK]",
              "params":{
                "do_lower_case": false,
                "spm_model_file": "./data/dict/sentencepiece.bpe.model"
              }
            },
          "need_convert": true,
          "vocab_path": "../../models_hub/ernie_m_1.0_base_dir/erniem.vocab.txt",
          "max_seq_len": 512,
          "truncation_type": 0,
          "padding_id": 0,
          "embedding": {
            "type":"ErnieTokenEmbedding",
            "use_reader_emb":false,
            "emb_dim":768,
            "config_path":"../../models_hub/ernie_m_1.0_base_dir/ernie_m_1.0_base_config.json",
            "other":""
          }
        },
        {
          "name": "label",
          "data_type": "int",
          "reader": {"type":"ScalarFieldReader"},
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
        "data_path": "./data/test_data_one_sent_multilingual",
        "shuffle": false,
        "batch_size": 8,
        "key_tag": false,
        "epoch": 1,
        "sampling_rate": 1.0,
        "need_data_distribute": false,
        "need_generate_examples": false
      }
    },
    "dev_reader": {
      "name": "dev_reader",
      "type": "BasicDataSetReader",
      "fields": [
        {
          "name": "text_a",
          "data_type": "string",
          "reader": {"type":"ErnieTextFieldReaderForMultilingual"},
          "tokenizer":{
              "type":"FullTokenizerErnieM",
              "split_char":" ",
              "unk_token":"[UNK]",
              "params":{
                "do_lower_case":false,
                "spm_model_file": "./data/dict/sentencepiece.bpe.model"
              }
            },
          "need_convert": true,
          "vocab_path": "../../models_hub/ernie_m_1.0_base_dir/erniem.vocab.txt",
          "max_seq_len": 512,
          "truncation_type": 0,
          "padding_id": 0,
          "embedding": {
            "type":"ErnieTokenEmbedding",
            "use_reader_emb":false,
            "emb_dim":768,
            "config_path":"../../models_hub/ernie_m_1.0_base_dir/ernie_m_1.0_base_config.json",
            "other":""
          }
        },
        {
          "name": "label",
          "data_type": "int",
          "reader": {"type":"ScalarFieldReader"},
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
        "data_path": "./data/dev_data_one_sent_multilingual",
        "shuffle": false,
        "key_tag": false,
        "batch_size": 8,
        "epoch": 1,
        "sampling_rate": 1.0,
        "need_data_distribute": false,
        "need_generate_examples": false
      }
    }
  },
  "model": {
    "type": "ErnieClassification",
    "is_dygraph": 1,
    "embedding": {
      "type":"ErnieTokenEmbedding",
      "emb_dim":768,
      "config_path":"../../models_hub/ernie_m_1.0_base_dir/ernie_m_1.0_base_config.json",
      "other":""
    },
    "optimization":{
      "learning_rate": 2e-5,
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
    "num_labels": 2
  },
  "trainer": {
    "type": "CustomDynamicTrainer",
    "PADDLE_PLACE_TYPE": "gpu",
    "PADDLE_IS_FLEET": 1,
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
        "name": "ernie_m_1.0_base",
        "params_path": "../../models_hub/ernie_m_1.0_base_dir/params"
      }
    ],
    "output_path": "./output/cls_ernie_m_1.0_base_one_sent",
    "extra_param": {
      "meta":{
        "job_type": "text_classification"
      }

    }
  }
}
```

### 训练ERNIE-M模型

```
python run_trainer.py --param_path ./examples/cls_ernie_m_1.0_base_one_sent.json
```

- 训练模型保存于./output/cls_ernie_m_1.0_base_one_sent文件夹下（可在配置文件./examples/cls_ernie_m_1.0_base_one_sent.json中修改输出路径），其中save_inference_model/文件夹会保存用于预测的模型文件，save_checkpoint/文件夹会保存用于热启动的模型文件
- 训练模型的日志文件保存于./log文件夹下

## 开始预测

### 预测的配置文件

- 配置文件 ./examples/cls_ernie_m_1.0_base_one_sent_infer.json
- 在配置文件./examples/cls_ernie_m_1.0_base_one_sent_infer.json中需要更改 inference.inference_model_path 为上面训练过程中所保存的**预测模型的路径**

```
{
  "dataset_reader": {
    "predict_reader": {
     "name": "predict_reader",
     "type": "BasicDataSetReader",
     "fields" : [
       {
          "name": "text_a",
          "data_type": "string",
          "reader": {"type":"ErnieTextFieldReaderForMultilingual"},
          "tokenizer":{
              "type":"FullTokenizerErnieM",
              "split_char":" ",
              "unk_token":"[UNK]",
              "params":{
                "do_lower_case": "false",
                "spm_model_file": "./data/dict/sentencepiece.bpe.model"
              }
            },
          "need_convert": true,
          "vocab_path": "../../models_hub/ernie_m_1.0_base_dir/erniem.vocab.txt",
          "max_seq_len": 512,
          "truncation_type": 0,
          "padding_id": 1,
          "embedding": {
            "type":"ErnieTokenEmbedding",
            "use_reader_emb":false,
            "emb_dim":768,
            "config_path":"../../models_hub/ernie_m_1.0_base_dir/ernie_m_1.0_base_config.json"
          }
       }
      ],
      "config": {
        "data_path": "./data/predict_data_one_sent_multilingual",
        "shuffle": false,
        "key_tag": false,
        "batch_size": 8,
        "epoch": 1,
        "sampling_rate": 1.0,
        "need_data_distribute": false,
        "need_generate_examples": true
      }
    }
  },

  "inference": {
    "output_path": "./output/predict_result.txt",
    "inference_model_path": "./output/cls_ernie_m_1.0_base_one_sent/save_inference_model/inference_step_100",
    "PADDLE_PLACE_TYPE": "gpu",
    "num_labels": 2,
    "thread_num": 2,
    "extra_param": {
      "meta":{
        "job_type": "text_classification"
      }

    }
  }
}
```

### ERNIE-M模型预测

```
python ./run_infer.py  --param_path ./examples/cls_ernie_m_1.0_base_one_sent_infer.json
```

- 预测结果保存于./output/predict_result.txt文件中（可在./examples/cls_ernie_m_1.0_base_one_sent_infer.json中修改输出路径）。
