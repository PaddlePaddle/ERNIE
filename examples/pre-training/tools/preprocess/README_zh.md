[English](README.md) | 简体中文

# 预训练数据转换工具

本工具用于将文本格式的数据集转换为预训练使用的`bin/idx`格式数据集

## 下载预训练权重

下载已发布的预训练权重，请参考[Introduction to ERNIE 4.5](/README.md)。

注意：每个模型的Tokenizer是不同的，因此必须下载你所训练的模型才能生成训练数据。

## 准备文本数据

本工具的输入为`jsonl`格式，每一行是一份预训练文本，记录在`"text"`字段中。例如：

```json
{"text": "动静统一自动并行只需在单卡基础上进行少量的张量切分标记，飞桨能自动寻找最高效的分布式并行策略..."}
{"text": "同一套框架支持训练和推理，实现训练、推理代码复用和无缝衔接，为大模型的全流程提供了..."}
...
```

## 生成预训练数据

```bash
python -u create_pretraining_data.py \
    --model_name "/path/to/your/ERNIE-4.5-21B-A3B-Base-Paddle" \
    --data_format "JSON" \
    --input_path "/path/to/your/text/dataset.jsonl" \
    --append_eos \
    --output_prefix "./pretrain_data"  \
    --workers 1 \
    --log_interval 10000 \
    --data_impl "mmap"
```

输出将保存为`./pretrain_data.bin`和`./pretrain_data.idx`。
