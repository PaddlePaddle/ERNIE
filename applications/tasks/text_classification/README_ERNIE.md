# ERNIE(ErnieClassification)模型训练、预测

# 开始训练

- 进入分类任务的目录./applications/tasks/text_classification

```
cd ./applications/tasks/text_classification
```

## 预训练模型准备

- 模型均存放于./applications/models_hub文件夹下，进入该文件夹下载对应ERNIE模型

```
cd ../../models_hub
sh ./download_ernie_xxx.sh
cd ../tasks/text_classification
```

## 训练的配置文件

- 配置文件：./examples/cls_ernie_fc_ch.json

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
          "reader": {
            "type": "ErnieTextFieldReader"
          },
          "tokenizer": {
            "type": "FullTokenizer",
            "split_char": " ",
            "unk_token": "[UNK]"
          },
          "need_convert": true,
          "vocab_path": "../../models_hub/ernie_3.0_base_ch_dir/vocab.txt",
          "max_seq_len": 512,
          "truncation_type": 0,
          "padding_id": 0
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
        "data_path": "./data/train_data",
        "shuffle": false,
        "batch_size": 8,
        "epoch": 5,
        "sampling_rate": 1.0,
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
          "reader": {
            "type": "ErnieTextFieldReader"
          },
          "tokenizer": {
            "type": "FullTokenizer",
            "split_char": " ",
            "unk_token": "[UNK]"
          },
          "need_convert": true,
          "vocab_path": "../../models_hub/ernie_3.0_base_ch_dir/vocab.txt",
          "max_seq_len": 512,
          "truncation_type": 0,
          "padding_id": 0
        },
        {
          "name": "label",
          "data_type": "int",
          "need_convert": false,
          "reader": {
            "type": "ScalarFieldReader"
          },
          "tokenizer": null,
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
    "optimization": {            ## 优化器设置，文心ERNIE推荐的默认设置。
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
      "config_path": "../../models_hub/ernie_3.0_base_ch_dir/ernie_config.json"
    },
    "num_labels": 2
  },
  "trainer": {
    "type": "CustomDynamicTrainer",
    "PADDLE_PLACE_TYPE": "gpu",
    "PADDLE_IS_FLEET": 1,
    "train_log_step": 10,
    "use_amp": true,         # 是否开启混合精度训练，默认开启。
    "is_eval_dev": 0,
    "is_eval_test": 1,
    "eval_step": 100,
    "save_model_step": 200,
    "load_parameters": "",    ## 加载包含各op参数值的训练好的模型，用于热启动。此处填写checkpoint路径。不填则表示不使用热启动。
    "load_checkpoint": "",     ## 加载包含学习率等所有参数的训练模型，用于热启动。此处填写checkpoint路径。不填则表示不使用热启动。
    "pre_train_model": [
      {
        "name": "ernie_3.0_base_ch",
        "params_path": "../../models_hub/ernie_3.0_base_ch_dir/params"
      }
    ],
    "output_path": "./output/cls_ernie_3.0_base_fc_ch_dy",
    "extra_param": {
      "meta":{
        "job_type": "text_classification"
      }

    }
  }
}
```

## 训练ERNIE模型

```
python run_trainer.py --param_path ./examples/cls_ernie_fc_ch.json
```

- 训练模型保存于./output/cls_ernie_3.0_base_fc_ch_dy文件夹下（可在配置文件./examples/cls_ernie_fc_ch.json中修改输出路径），其中save_inference_model/文件夹会保存用于预测的模型文件，save_checkpoint/文件夹会保存用于热启动的模型文件
- 训练模型的日志文件保存于./log文件夹下

# 开始预测

## 预测的配置文件

- 配置文件 ./examples/cls_ernie_fc_ch_infer.json
- 在配置文件./examples/cls_ernie_fc_ch_infer.json中需要更改 inference.inference_model_path 为上面训练过程中所保存的**预测模型的路径**

```
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
        "batch_size": 8,
        "epoch": 1,
        "sampling_rate": 1.0,
        "need_data_distribute": false,
        "need_generate_examples": true
      }
    }
  },

  "inference": {
    "type": "CustomInference",
    "output_path": "./output/predict_result.txt",
    "PADDLE_PLACE_TYPE": "cpu",
    "num_labels": 2,
    "thread_num": 2,
    "inference_model_path": "./output/cls_ernie_3.0_base_fc_ch_dy/save_inference_model/inference_step_126/",
    "extra_param": {
      "meta":{
        "job_type": "text_classification"
      }

    }
  }
}
```

## ERNIE模型预测

```
python ./run_infer.py  --param_path ./examples/cls_ernie_fc_ch_infer.json
```

- 预测结果保存于./output/predict_result.txt文件中（可在./examples/cls_ernie_fc_ch_infer.json中修改输出路径）。
