[简体中文](./README.md)|English
# Introducing Propeller
This doc introduct Propeller, a high level paddle API for general ML, Propeller encapsulate the following actions:：
-  training
-  evaluation
-  prediction
-  export serving
  
Propeller provide the following benefits:

-   You can run Propeller-based models on a local host or on a distributed multi-server environment without changing your model. Furthermore, you can run Propeller-based models on CPUs, GPUs without recoding your model.
-   Propeller simplify sharing implementations between model developers.
-   Propeller do many things for you (logging, hot-start...)
-   Propeller buids Program and PyReader or you.
-   Propeller provide a safe distributed training loop that controls how and when to:
    -   build the Program
    -   initialize variables
    -   create checkpoint files and recover from failures
    -   save visualizable results

## install

```script
cd propeller && pip install .
```

## Getting Started
```python

    #Define model
    class BowModel(propeller.Model):
        def __init__(self, config, mode):
            self.embedding = Embedding(config['emb_size'], config['vocab_size'])
            self.fc1 = FC(config['hidden_size'])
            self.fc2 = FC(config['hidden_size'])

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

    # hyper param comes from files/command line prompt/env vir
    run_config = propeller.parse_runconfig(args)
    hparams = propeller.parse_hparam(args)
    
    # Define data
    # `FeatureColumns` helps you to organize training/evluation files.
    feature_column = propeller.data.FeatureColumns(columns=[
            propeller.data.TextColumn('query', vocab='./vocab'),
            propeller.data.TextColumn('title', vocab='./vocab'),
            propeller.data.LabelColumn('label'),
        ])
    train_ds = feature_column.build_dataset(data_dir='./data',  shuffle=True, repeat=True)
    eval_ds = feature_column.build_dataset(data_dir='./data', shuffle=False, repeat=False)

    # Start training!
    propeller.train_and_eval(BowModel, hparams, run_config, train_ds, eval_ds)
```
More detail see example/toy/

## Main Feature
1. train_and_eval

    according to user-specified `propeller.Model`class，initialize training model in the following 2 modes: 1. TRAIN mode 2. EVAL mode and
    perform train_and_eval

2. FeatureColumns
    
    `FeatureColumns`is used to ogranize train data. With custmizable `Column` property, it can adaps to many ML tasks（NLP/CV...).
    `FeatureColumns` also do the preprocessing for you (tokenization, vocab lookup, serialization, batcing etc.)


3. Dataset

    `FeatureColumns` generats `Dataset`，or you can call `propeller.Dataset.from_generator_func` to build your own `Dataset`.

4. Summary
    To trace tensor histogram in training, simply：
    ```python
        propeller.summary.histogram('loss', tensor) 
    ```


## Contributing

1. This project is in alpha stage, any contribution is welcomed. Fill free to create a PR.
