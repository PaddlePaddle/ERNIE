# 代码结构、准备工作

## 代码结构

- 分类任务位于applications/tasks/text_classification

```plain
.
├── data        # 训练数据
│   ├── dev_data
│   │   └── dev_1.txt
│   ├── dev_data_one_sent_multilingual
│   │   └── multi_dev.tsv
│   ├── dict
│   │   ├── sentencepiece.bpe.model
│   │   └── vocab.txt
│   ├── download_data.sh
│   ├── multi_label_data
│   │   ├── dev_data
│   │   │   └── dev.txt
│   │   ├── test_data
│   │   │   └── test.txt
│   │   └── train_data
│   │       └── train.txt
│   ├── predict_data
│   │   └── infer.txt
│   ├── predict_data_one_sent_multilingual
│   │   └── multi_predict.tsv
│   ├── test_data
│   │   └── test.txt
│   ├── test_data_one_sent_multilingual
│   │   └── multi_test.tsv
│   ├── train_data
│   │   └── train.txt
│   └── train_data_one_sent_multilingual
│       └── multi_train.tsv
├── data_set_reader    # 数据加载相关类
│   ├── dataset_reader_for_ernie_doc_classification.py
│   └── __init__.py
├── examples
│   ├── cls_bow_ch_infer.json
│   ├── cls_bow_ch.json
│   ├── cls_ernie_doc_1.0_one_sent_ch_infer_dy.json
│   ├── cls_ernie_doc_1.0_one_sent_ch.json
│   ├── cls_ernie_fc_ch_infer.json
│   ├── cls_ernie_fc_ch.json
│   ├── cls_ernie_fc_ch_with_data_aug.json
│   ├── cls_ernie_m_1.0_base_one_sent_infer.json
│   ├── cls_ernie_m_1.0_base_one_sent.json
│   ├── cls_ernie_multi_label_ch_infer.json
│   └── cls_ernie_multi_label_ch.json
├── inference       # 推理相关类
│   ├── custom_inference.py
│   ├── ernie_doc_inference_dy.py
│   └── __init__.py
├── __init__.py
├── model
│   ├── base_cls.py
│   ├── bow_classification.py
│   ├── ernie_classification.py
│   ├── ernie_doc_classification.py
│   ├── __init__.py
│   └── multi_label_classification.py
├── reader
│   ├── categorical_field_reader.py
│   └── multi_label_field_reader.py
├── run_infer_dy.py   #ernie_doc 动态图预测的py文件
├── run_infer.py      # 除enrie_doc外的预测文件入口文件，只依靠json进行模型训练的入口脚本
├── run_trainer.py.   # 训练文件入口文件，只依靠json进行模型训练的入口脚本
├── run_with_data_aug.sh
└── trainer
    ├── custom_dynamic_ernie_doc_trainer.py
    ├── custom_dynamic_trainer.py
    ├── custom_trainer.py
    └── __init__.py
```

## 数据准备

- 在文心中，基于ERNIE的模型都不需要用户自己分词和生成词表文件，非ERNIE的模型需要用户自己提前切好词，词之间以空格分隔，并生成词表文件。切词和词表生成可以使用「[分词工具与词表生成工具](../../tools/data/wordseg/README.md)」进行处理。
- 文心中的所有数据集、包含词表文件、label_map文件等都必须为为utf-8格式，如果你的数据是其他格式，请使用「[编码识别及转换工具](../../tools/data/data_cleaning/README.md)」进行格式转换。
- 文心中的训练集、测试集、验证集、预测集和词表分别存放在./applications/tasks/text_classification/data目录下的train_data、test_data、dev_data、predict_data、dict文件夹下。
- 在分类任务中，训练集、测试集和验证集的数据格式相同，数据分为两列，列与列之间用**\t**进行分隔。第一列为文本，第二列为标签。以下为示例：

### 单标签分类

- 非ERNIE训练集数据示例：数据分为两列，列与列之间用**\t**进行分隔。第一列为文本，第二列为标签。

```js
房间 太 小 。 其他 的 都 一般 。 。 。 。 。 。 。 。 。         0
LED屏 就是 爽 ， 基本 硬件 配置 都 很 均衡 ， 镜面 考 漆 不错 ， 小黑 ， 我喜欢 。         1
差 得 要命 , 很大 股霉味 , 勉强 住 了 一晚 , 第二天 大早 赶紧 溜。         0
```

- 非ERNIE预测数据集示例：仅一列为文本，不需要标签列

```js
USB接口 只有 2个 ， 太 少 了 点 ， 不能 接 太多 外 接 设备 ！ 表面 容易 留下 污垢 ！ 
平时 只 用来 工作 ， 上 上网 ， 挺不错 的 ， 没有 冗余 的 功能 ， 样子 也 比较 正式 ！ 还 可以 吧 ， 价格 实惠   宾馆 反馈   2008年4月17日   ：   谢谢 ！ 欢迎 再次 入住 其士 大酒店 。
```

- 非ERNIE模型的词表文件示例：词表分为两列，第一列为词，第二列为id（从0开始），列与列之间用**\t**进行分隔。文心的词表中，[PAD]、[CLS]、[SEP]、[MASK]、[UNK]这5个词是必须要有的，若用户自备词表，需保证这5个词是存在的。部分词表示例如下所示：

```js
[PAD]	0 
[CLS]	1 
[SEP]	2 
[MASK]	3 
[UNK]	4 
郑重	5 
天空	6 
工地	7 
神圣	8
```

- ERNIE数据集与非ERNIE数据集格式一致，不同之处在于不用分词，如下所示：

```js
选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。酒店装修一般，但还算整洁。         1 
15.4寸笔记本的键盘确实爽，基本跟台式机差不多了，蛮喜欢数字小键盘，输数字特方便，样子也很美观，做工也相当不错         1 
房间太小。其他的都一般。。。。。。。。。         0
```

- ERNIE词表文件格式与非ERNIE的格式一致，由文心提供。

```js
[PAD]   0 
[CLS]  1 
[SEP]  2 
[MASK] 3 
，  4 
的  5 
、  6 
一  7 
人  8 
有  9
```

### 多标签分类

- ERNIE训练集数据示例：数据分为两列，列与列之间用**\t**进行分隔。第一列为文本，第二列为标签，当该样本拥有多个标签时，标签之间使用空格进行分隔。比如你的标签有100种，某个样例的标签是第50个标签和第87个标签，其标签列就是“49 86”。像这个情况就是标签的第49和86维是正例1，其他维是负例0。非ERNIE的训练集数据与ERNIE一致，区别就是非ERNIE的文本需要切词，这里不再赘述。

```js
互联网创业就如选秀 需求与服务就是价值	0 1 
郭德纲式生存：时代的变与未变	2 
快讯！腾讯市值突破10000亿港元	3
```

- ERNIE预测数据集示例：仅一列为文本，不需要标签列，非ERNIE的训练集数据与ERNIE一致，区别就是非ERNIE的文本需要切词，这里不再赘述。

```js
互联网创业就如选秀 需求与服务就是价值 
郭德纲式生存：时代的变与未变 
快讯！腾讯市值突破10000亿港元
```

- 词表的格式与单标签分类一致，不再赘述

## 网络（模型）选择

文心预置的可用于文本分类的模型源文件在./applications/tasks/text_classification/model目录下，各个模型的特点如下所示（后面章节会依次展示使用方法）：

| 网络名称（py文件的类名）                                | 简介                                                         | 支持类型   | 支持预训练模型                                               | 备注 |
| ------------------------------------------------------- | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ | ---- |
| BowClassification(bow_classification.py)                | 词袋模型，不考虑语法和语序，用一组无序单词来表达一段文本。   | 单标签分类 | 无                                                           |      |
| ErnieClassification(ernie_classification.py)            | 基于ERNIE预训练模型的最简单的分类模型，在ERNIE的embedding输出层之后直接添加FC（全链接层）降维到标签数量的纬度，loss使用交叉熵。网络结构简单，效果好。 | 单标签分类 | ERNIE2.0-Base、ERNIE2.0-large、ERNIE3.0-Base、ERNIE3.0-x-Base、ERNIE3.0-Medium、ERNIE-M |      |
| MultiLabelClassification(multi_label_classification.py) | 处理多标签分类任务的网络结构，在ErnieClassification的基础上，loss由二分类交叉熵更换为 sigmoid cross entropy | 多标签分类 | ERNIE2.0-Base、ERNIE2.0-large、ERNIE3.0-Base、ERNIE3.0-x-Base、ERNIE3.0-Medium、ERNIE-M |      |
| ErnieDocClassification(ernie_doc_classification.py)     | 长文本分类任务的网络                                         | 单标签分类 | ERNIE-Doc                                                    |      |

## ERNIE预训练模型下载

文心提供的ERNIE预训练模型的参数文件和配置文件在./applications/tasks/models_hub/目录下，由对应的download_xx.sh文件是下载得到。ERNIE部分模型介绍，请详见文档「[ERNIE模型介绍](../../models_hub)」

## 模型评估指标选择

分类任务常用的指标有：Acc（准确率）、Precision（精确率）、Recall（召回率）、F1。
