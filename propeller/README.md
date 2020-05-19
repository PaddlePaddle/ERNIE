简体中文|[English](./README.en.md)
# Introducing paddle-propeller
本文档介绍propeller，一种可极大地简化机器学习编程的高阶 Paddle API。propeller 会封装下列操作：
-   训练
-   评估
-   预测
-   导出以供使用（上线）  
  
Propeller 具有下列优势：

-   您可以在本地主机上或分布式多服务器环境中运行基于 Propeller 的模型，而无需更改模型。此外，您可以在 CPU、GPU上运行基于 Propeller 的模型，而无需重新编码模型。
-   Propeller 简化了在模型开发者之间共享实现的过程。
-   只需关注模型实现以及数据输入，而无需关注其他辅助代码（保存、热启动、打log等）
-   Propeller 会为您构建Program以及PyReader。
-   Propeller 提供安全的分布式训练循环，可以控制如何以及何时：
    -   构建Program
    -   初始化变量
    -   处理异常
    -   创建检查点文件并从故障中恢复
    -   保存可视化的摘要结果

## Getting Started|快速开始
```python

    #定义训练模型
    class BowModel(propeller.Model):
        def __init__(self, config, mode):
            self.embedding = Embedding(config['emb_size'], config['vocab_size'])
            self.fc1 = FC(config['hidden_size'])
            self.fc2 = FC(config['hidden_size']

        def forward(self, features):
            q, t = features 
            q_emb = softsign(self.embedding(q))
            t_emb = softsign(self.embedding(t))
            q_emb = self.fc1(q_emb)
            t_emb = self.fc2(t_emn)
            prediction = dot(q_emb,  emb)
            return prediction

        def loss(self, predictions, label):
            return sigmoid_cross_entropy_with_logits(predictions, label)

        def backward(self, loss):
            opt = AdamOptimizer(1.e-3)
            opt.mimize(loss)

        def metrics(self, predictions, label):
            auc = atarshi.metrics.Auc(predictions, label)
            return {'auc': auc}

    # 超参可以来自于文件/ 环境变量/ 命令行
    run_config = propeller.parse_runconfig(args)
    hparams = propeller.parse_hparam(args)
    
    # 定义数据： 
    # `FeatureColumns` 用于管理训练、预测文件. 会自动进行二进制化.
    feature_column = propeller.data.FeatureColumns(columns=[
            propeller.data.TextColumn('query', vocab='./vocab'),
            propeller.data.TextColumn('title', vocab='./vocab'),
            propeller.data.LabelColumn('label'),
        ])
    train_ds = feature_column.build_dataset(data_dir='./data',  shuffle=True, repeat=True)
    eval_ds = feature_column.build_dataset(data_dir='./data', shuffle=False, repeat=False)

    # 开始训练！
    propeller.train_and_eval(BowModel, hparams, run_config, train_ds, eval_ds)
```
详细详细请见example/toy/

## 主要构件
1. train_and_eval

    会根据用户提供的`propeller.Model`类，实例化两种模式下的训练模型： 1. TRAIN模式 2. EVAL模式。
    然后开始训练，同时执行评估（Evaluation）

2. FeatureColumns
    
    用`FeatureColumns`来管理训练数据. 根据自定义`Column`来适配多种ML任务（NLP/CV...).
    `FeatureColumns`会自动对提供的训练数据进行批量预处理(tokenization, 查词表, etc.)并二进制化，并且生成训练用的dataset

3. Dataset

    `FeatureColumns`生成`Dataset`，或者您可以调用`propeller.Dataset.from_generator_func`来构造自己的`Dataset`，配合shuffle/ interleave/ padded_batch/ repeat 等方法满足定制化需求.

4. Summary
    对训练过程中的某些参数进行log追踪，只需要：
```python
            propeller.summary.histogram('loss', tensor) 

```


## Contributing|贡献

1. 本项目处于初期阶段，欢迎贡献！
2. functional programing is welcomed


## TODO

1. dataset output_types/ output_shapes 自动推断
2. 自动超参数搜索
3. propeller server
4. ...
