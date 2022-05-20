# 数据蒸馏

- 在ERNIE强大的语义理解能力背后，是需要同样强大的算力才能支撑起如此大规模模型的训练和预测。很多工业应用场景对性能要求较高，若不能有效压缩则无法实际应用。

- 因此，我们基于数据蒸馏技术构建了数据蒸馏系统。其原理是通过数据作为桥梁，将ERNIE模型的知识迁移至小模型，以达到损失很小的效果却能达到上千倍的预测速度提升的效果。

## 代码结构

- 数据蒸馏任务位于 applications/tasks/data_distillation

```plain
data_distillation/
├── data
│   ├── dev_data
│   ├── dict
│   ├── download_data.sh
│   ├── predict_data
│   ├── test_data
│   └── train_data
├── distill
│   └── chnsenticorp
│       ├── student
│       └── teacher
├── examples
│   ├── cls_bow_ch.json
│   ├── cls_cnn_ch.json
│   ├── cls_ernie_fc_ch_infer.json
│   └── cls_ernie_fc_ch.json
├── inference
│   ├── custom_inference.py
│   ├── __init__.py

├── model
│   ├── base_cls.py
│   ├── bow_classification.py
│   ├── cnn_classification.py
│   ├── ernie_classification.py
│   ├── __init__.py

├── run_distill.sh
├── run_infer.py
├── run_trainer.py
└── trainer
    ├── custom_dynamic_trainer.py
    ├── __init__.py
```

## 数据准备

- 目前采用三种数据增强策略策略，对于不用的任务可以特定的比例混合。三种[数据增强](../../tools/data/data_aug)策略包括：
  - 添加噪声：对原始样本中的词，以一定的概率（如0.1）替换为”UNK”标签
  - 同词性词替换：对原始样本中的所有词，以一定的概率（如0.1）替换为本数据集中随机一个同词性的词
  - N-sampling：从原始样本中，随机选取位置截取长度为m的片段作为新的样本，其中片段的长度m为0到原始样本长度之间的随机值
- 数据增强策略可参考https://arxiv.org/pdf/1903.12136.pdf ,我们已**准备好了**采用上述3种增强策略制作的**ChnSentiCorp增强数据**。

## 开始离线蒸馏（demo演示）

- 使用预置的ERNIE 3.0 base模型

```plain
cd applications/models_hub
bash download_ernie_3.0_base_ch.sh
```

- 下载预置的原始数据以及增强数据。

```plain
cd applications/tasks/data_distillation/distill
bash download_data.sh
```

- 运行以下命令，开始数据蒸馏

```plain
cd applications/tasks/data_distillation 
bash run_distill.sh
```

## 蒸馏过程说明（run_distill.sh脚本运行过程说明）

- 事先构造好的增强数据放在`./distill/chnsenticorp/student/unsup_train_aug`文件夹下
- **run_distill.sh**脚本涉及教师和学生模型的json文件，其中：
  - `./examples/cls_ernie_fc_ch.json`负责教师模型的finetune；
  - `./examples/cls_ernie_fc_ch_infer.json` 负责教师模型的预测；
  - `./examples/cls_cnn_ch.json`，负责学生模型的训练。
- **run_distill.sh**脚本会进行下面三步操作：
  - 在任务数据上Fine-tune：
    - 使用 python run_trainer.py --param_path ./examples/cls_ernie_fc_ch.json 训练teacher模型
  - 加载Fine-tune好的模型对增强数据进行打分：
    - 通过 python run_infer.py --param_path ./examples/cls_ernie_fc_ch_infer.json 脚本使用teacher模型对无监督数据进行进行预测，并将打分输出到标准输出。最终的标注结果放在 `./distill/chnsenticorp/student/train/part.1`文件中。标注结果包含两列, 第一列为明文，第二列为标注label。用这种方式对数据增强后的无监督训练预料进行标注。
  - 使用Student模型进行训练，脚本采用hard-label蒸馏：
    - 使用 python run_trainer.py --param_path ./examples/cls_cnn_ch.json 训练student模型，其训练数据放在 `distill/chnsenticorp/student/train/` 中，part.0 为原监督数据，part.1 为 ERNIE 标注数据。

- **注**：
  - 如果用户已经拥有了无监督数据，则可以将无监督数据放入 `distill/chnsenticorp/student/unsup_train_aug` 即可。
  - 需注意的是：因学生模型训练数据分别为原监督数据part.0和ERNIE标注数据part.1，通常情况下，两份数据文件大小不均衡。

## 效果验证

我们将实际应用场景分类为两种：用户提供“无标注数据”和用户未提供“无标注数据”（通过数据增强生成数据）。

### Case#1 用户提供“无标注数据”

| 模型               | 评论低质识别【分类｜ACC】 | 中文情感【分类｜ACC】 | 问题识别【分类｜ACC】 | 搜索问答匹配【匹配｜正逆序】 |
| ------------------ | ------------------------- | --------------------- | --------------------- | ---------------------------- |
| ERNIE-Finetune     | 90.6%                     | 96.2%                 | 97.5%                 | 4.25                         |
| 非ERNIE基线（BOW） | 80.8%                     | 94.7%                 | 93.0%                 | 1.83                         |
| + 数据蒸馏         | 87.2%                     | 95.8%                 | 96.3%                 | 3.30                         |

### Case#2 用户未提供“无标注数据”（通过数据增强生成数据）

| 模型                | ChnSentiCorp |
| ------------------- | ------------ |
| ERNIE-Finetune      | 95.4%        |
| 非ERNIE基线(BOW)    | 90.1%        |
| + 数据蒸馏          | 91.4%        |
| 非ERNIE基线（CNN）  | 91.6%        |
| + 数据蒸馏          | 92.4%        |
| 非ERNIE基线（LSTM） | 91.2%        |
| + 数据蒸馏          | 93.9%        |
