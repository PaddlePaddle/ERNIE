<h1 align="center">ELMO</h1>

## 介绍

ELMO(Embeddings from Language Models)是一种新型深度语境化词表征，可对词进行复杂特征(如句法和语义)和词在语言语境中的变化进行建模(即对多义词进行建模)。ELMO作为词向量，解决了两个重要问题：（1）词使用的复杂特性，如句法和语法。（2）如何在具体的语境下使用词，比如多义词的问题。

ELMO在大语料上以language model为训练目标，训练出bidirectional LSTM模型，利用LSTM产生词语的表征, 对下游NLP任务(如问答、分类、命名实体识别等）进行微调。

此版本发布要点：
1. 发布预训练模型完整代码。
2. 支持多卡训练，训练速度比主流实现快约1倍。
3. 发布[ELMO中文预训练模型](https://dureader.gz.bcebos.com/elmo/baike_elmo_checkpoint.tar.gz),
训练约38G中文百科数据。
4. 发布基于ELMO微调步骤和[LAC微调示例代码](finetune)，验证在中文词法分析任务LAC上f1值提升了1.1%。


## 基本配置及第三方安装包

Python==2.7

PaddlePaddle lastest版本

numpy ==1.15.1

six==1.11.0

glob


## 预训练模型

1. 把文档文件切分成句子，并基于词表（参考[`data/vocabulary_min5k.txt`](data/vocabulary_min5k.txt)）对句子进行切词。把文件切分成训练集trainset和测试集testset。训练数据参考[`data/train`](data/train)，测试数据参考[`data/dev`](data/dev)，
训练集和测试集比例推荐为5：1。

```
本 书 介绍 了 中国 经济 发展 的 内外 平衡 问题 、 亚洲 金融 危机 十 周年 回顾 与 反思 、 实践 中 的 城乡 统筹 发展 、 未来 十 年 中国 需要 研究 的 重大 课题 、 科学 发展 与 新型 工业 化 等 方面 。
```
```
吴 敬 琏 曾经 提出 中国 股市 “ 赌场 论 ” ， 主张 维护 市场 规则 ， 保护 草根 阶层 生计 ， 被 誉 为 “ 中国 经济 学界 良心 ” ， 是 媒体 和 公众 眼中 的 学术 明星 
```

2. 训练模型

```shell
sh run.sh
```

3. 把checkpoint结果写入文件中。


## 单机多卡训练

模型支持单机多卡训练，需要在[`run.sh`](run.sh)里export CUDA_VISIBLE_DEVICES设置指定卡,如下所示：
```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## 如何利用ELMO做微调

   在深度学习训练中，例如图像识别训练，每次从零开始训练都要消耗大量的时间和资源。而且当数据集比较少时，模型也难以拟合的情况。基于这种情况下，就出现了迁移学习，通过使用已经训练好的模型来初始化即将训练的网络，可以加快模型的收敛速度，而且还能提高模型的准确率。这个用于初始化训练网络的模型是使用大型数据集训练得到的一个模型，而且模型已经完全收敛。最好训练的模型和预训练的模型是同一个网络，这样可以最大限度地初始化全部层。
   
   利用ELMO做微调，与Bert方式不同，ELMO微调是把ELMO部分作为已预训练好的词向量，接入到NLP下游任务中。
   
   在原论文中推荐的使用方式是，NLP下游任务输入的embedding层与ELMO的输出向量直接做concat。其中，ELMO部分是直接加载预训练出来的模型参数（PaddlePaddle中通过fluid.io.load_vars接口来加载参数），模型参数输入到NLP下游任务是fix的（在PaddlePaddle中通过stop_gradient = True来实现）。
   
   ELMO微调部分可参考[LAC微调示例代码](finetune)，百度词法分析工具[LAC官方发布代码地址](https://github.com/baidu/lac/tree/a4eb73b2fb64d8aab8499a1184edf4fc386f8268)。

ELMO微调任务的要点如下：

1)下载预训练模型的参数文件。

2)加载elmo网络定义部分bilm.py。

3)在网络启动时加载预训练模型。

4)基于elmo字典对输入做切词并转化为id。

5)elmo词向量与网络embedding层做concat。

具体步骤如下：
1. 下载ELMO Paddle官方发布预训练模型文件，预训练模型文件训练约38G中文百科数据。

[ELMO中文预训练模型](https://dureader.gz.bcebos.com/elmo/baike_elmo_checkpoint.tar.gz)

2. 在网络初始化启动中加载ELMO Checkpoint文件。加载参数接口（fluid.io.load_vars）,可加在网络参数（exe.run(fluid.default_startup_program())）初始化之后。

```shell
# 定义一个使用CPU的执行器
place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

```

```shell
src_pretrain_model_path = '490001' #490001为ELMO checkpoint文件
def if_exist(var):
    path = os.path.join(src_pretrain_model_path, var.name)
    exist = os.path.exists(path)
    if exist:
        print('Load model: %s' % path)
    return exist

fluid.io.load_vars(executor=exe, dirname=src_pretrain_model_path, predicate=if_exist, main_program=main_program) 
```

3. 在下游NLP任务代码中加入[`bilm.py`](bilm.py) 文件，[`bilm.py`](finetune/bilm.py) 是ELMO网络定义部分。

4. 基于elmo词表（参考[`data/vocabulary_min5k.txt`](data/vocabulary_min5k.txt) ）对输入的句子或段落进行切词，并把切词的词转化为id,放入feed_dict中。

5. 在NLP下游任务代码，网络定义中embedding部分加入ELMO网络的定义

```shell
#引入 bilm.py embedding部分和encoder部分
from bilm import elmo_encoder
from bilm import emb

#word为输入elmo部分切词后的字典
elmo_embedding = emb(word)
elmo_enc= elmo_encoder(elmo_embedding)

#与NLP任务中生成词向量word_embedding做连接操作
word_embedding=layers.concat(input=[elmo_enc, word_embedding], axis=1)

```

## 参考论文
[Deep contextualized word representations](https://arxiv.org/abs/1802.05365)


## Contributors
本项目由百度深度学习技术平台部PaddlePaddle团队和百度自然语言处理部合作完成。欢迎贡献代码和反馈问题。
