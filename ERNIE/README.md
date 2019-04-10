## ERNIE: **E**nhanced **R**epresentation through k**N**owledge **I**nt**E**gration

**** **2019-04-10 更新**: update ERNIE_stable-1.0.1.tar.gz, 将模型参数、配置 ernie_config.json、vocab.txt 打包发布 ****

**** **2019-03-18 更新**: update ERNIE_stable.tgz ****

**ERNIE** 通过建模海量数据中的词、实体及实体关系，学习真实世界的语义知识。相较于 **BERT** 学习原始语言信号，**ERNIE** 直接对先验语义知识单元进行建模，增强了模型语义表示能力。

这里我们举个例子：

```Learnt by BERT ：哈 [mask] 滨是 [mask] 龙江的省会，[mask] 际冰 [mask] 文化名城。```

```Learnt by ERNIE：[mask] [mask] [mask] 是黑龙江的省会，国际 [mask] [mask] 文化名城。```

在 **BERT** 模型中，我们通过『哈』与『滨』的局部共现，即可判断出『尔』字，模型没有学习与『哈尔滨』相关的任何知识。而 **ERNIE** 通过学习词与实体的表达，使模型能够建模出『哈尔滨』与『黑龙江』的关系，学到『哈尔滨』是 『黑龙江』的省会以及『哈尔滨』是个冰雪城市。

训练数据方面，除百科类、资讯类中文语料外，**ERNIE** 还引入了论坛对话类数据，利用 **DLM**（Dialogue Language Model）建模 Query-Response 对话结构，将对话 Pair 对作为输入，引入 Dialogue Embedding 标识对话的角色，利用 Dialogue Response Loss 学习对话的隐式关系，进一步提升模型的语义表示能力。

我们在自然语言推断，语义相似度，命名实体识别，情感分析，问答匹配 5 个公开的中文数据集合上进行了效果验证，**ERNIE** 模型相较 **BERT** 取得了更好的效果。

<table>
  <tbody>
    <tr>
      <th><strong>数据集</strong>
        <br></th>
      <th colspan="2"><strong>XNLI</strong></th>
      <th colspan="2"><strong>LCQMC</strong></th>
      <th colspan="2"><strong>MSRA-NER(SIGHAN 2006)</strong></th>
      <th colspan="2"><strong>ChnSentiCorp</strong></th>
      <th colspan="4"><strong>nlpcc-dbqa</strong></th></tr>
    <tr>
      <td rowspan="2">
        <p>
          <strong>评估</strong></p>
        <p>
          <strong>指标</strong>
          <br></p>
      </td>
      <td colspan="2">
        <strong>acc</strong>
        <br></td>
      <td colspan="2">
        <strong>acc</strong>
        <br></td>
      <td colspan="2">
        <strong>f1-score</strong>
        <br></td>
      <td colspan="2">
        <strong>acc</strong>
        <strong></strong>
        <br></td>
      <td colspan="2">
        <strong>mrr</strong>
        <br></td>
      <td colspan="2">
        <strong>f1-score</strong>
        <br></td>
    </tr>
    <tr>
      <th colspan="1" width="">
        <strong>dev</strong>
        <br></th>
      <td colspan="1" width="">
        <strong>test</strong>
        <br></td>
      <td colspan="1" width="">
        <strong>dev</strong>
        <br></td>
      <td colspan="1" width="">
        <strong>test</strong>
        <br></td>
      <td colspan="1" width="">
        <strong>dev</strong>
        <br></td>
      <td colspan="1" width="">
        <strong>test</strong>
        <br></td>
      <td colspan="1" width="">
        <strong>dev</strong>
        <br></td>
      <td colspan="1" width="">
        <strong>test</strong>
        <br></td>
      <td colspan="1" width="">
        <strong>dev</strong>
        <br></td>
      <td colspan="1" width="">
        <strong>test</strong>
        <br></td>
      <td colspan="1" width="">
        <strong>dev</strong>
        <br></td>
      <td colspan="1" width="">
        <strong>test</strong>
        <br></td>
    </tr>
    <tr>
      <td>
        <strong>BERT
          <br></strong></td>
      <td>78.1</td>
      <td>77.2</td>
      <td>88.8</td>
      <td>87.0</td>
      <td>94.0
        <br></td>
      <td>
        <span>92.6</span></td>
      <td>94.6</td>
      <td>94.3</td>
      <td colspan="1">94.7</td>
      <td colspan="1">94.6</td>
      <td colspan="1">80.7</td>
      <td colspan="1">80.8</td></tr>
    <tr>
      <td>
        <strong>ERNIE
          <br></strong></td>
      <td>79.9 <span>(<strong>+1.8</strong>)</span></td>
      <td>78.4 <span>(<strong>+1.2</strong>)</span></td>
      <td>89.7 <span>(<strong>+0.9</strong>)</span></td>
      <td>87.4 <span>(<strong>+0.4</strong>)</span></td>
      <td>95.0 <span>(<strong>+1.0</strong>)</span></td>
      <td>93.8 <span>(<strong>+1.2</strong>)</span></td>
      <td>95.2 <span>(<strong>+0.6</strong>)</span></td>
      <td>95.4 <span>(<strong>+1.1</strong>)</span></td>
      <td colspan="1">95.0 <span>(<strong>+0.3</strong>)</span></td>
      <td colspan="1">95.1 <span>(<strong>+0.5</strong>)</span></td>
      <td colspan="1">82.3 <span>(<strong>+1.6</strong>)</span></td>
      <td colspan="1">82.7 <span>(<strong>+1.9</strong>)</span></td></tr>
  </tbody>
</table>

 - **自然语言推断任务** XNLI

```text
XNLI 由 Facebook 和纽约大学的研究者联合构建，旨在评测模型多语言的句子理解能力。目标是判断两个句子的关系（矛盾、中立、蕴含）。[链接: https://github.com/facebookresearch/XNLI]
```

 - **语义相似度** LCQMC

```text
LCQMC 是哈尔滨工业大学在自然语言处理国际顶会 COLING2018 构建的问答匹配数据集，其目标是判断两个问题的语义是否相同。[链接: http://aclweb.org/anthology/C18-1166]
```

 - **命名实体识别任务** MSRA-NER(SIGHAN 2006)

```text
MSRA-NER(SIGHAN 2006) 数据集由微软亚研院发布，其目标是命名实体识别，是指识别文本中具有特定意义的实体，主要包括人名、地名、机构名等。
```

 - **情感分析任务** ChnSentiCorp

```text
ChnSentiCorp 是中文情感分析数据集，其目标是判断一段话的情感态度。
```

 - **检索式问答任务** nlpcc-dbqa

 ```text
nlpcc-dbqa是由国际自然语言处理和中文计算会议NLPCC于2016年举办的评测任务，其目标是选择能够回答问题的答案。[链接: http://tcci.ccf.org.cn/conference/2016/dldoc/evagline2.pdf]
```

### 模型&数据

1) 预训练模型下载

| Model | Description |
| :------| :------ |
| [模型](https://ernie.bj.bcebos.com/ERNIE_stable.tgz) | 包含预训练模型参数 |
| [模型(含配置文件及词典)](https://baidu-nlp.bj.bcebos.com/ERNIE_stable-1.0.1.tar.gz)) | 包含预训练模型参数、词典 vocab.txt、模型配置 ernie_config.json|

2) [任务数据下载](https://ernie.bj.bcebos.com/task_data.tgz)

### 安装
本项目依赖于 Paddle Fluid 1.3.1，请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装。

**Note**: 预训练任务和finetune任务测试机器为P40, 显存22G；如果显存低于22G, 某些任务可能会因显存不足报错；

### 预训练

#### 数据预处理
基于百科类、资讯类、论坛对话类数据构造具有上下文关系的句子对数据，利用百度内部词法分析工具对句对数据进行字、词、实体等不同粒度的切分，然后基于 [`tokenization.py`](tokenization.py) 中的 CharTokenizer 对切分后的数据进行 token 化处理，得到明文的 token 序列及切分边界，然后将明文数据根据词典 [`config/vocab.txt`](config/vocab.txt) 映射为 id 数据，在训练过程中，根据切分边界对连续的 token 进行随机 mask 操作；

我们给出了 id 化后的部分训练数据：[`data/demo_train_set.gz`](./data/demo_train_set.gz`)、和测试数据：[`data/demo_valid_set.gz`](./data/demo_valid_set.gz)，每行数据为1个训练样本，示例如下:

```
1 1048 492 1333 1361 1051 326 2508 5 1803 1827 98 164 133 2777 2696 983 121 4 19 9 634 551 844 85 14 2476 1895 33 13 983 121 23 7 1093 24 46 660 12043 2 1263 6 328 33 121 126 398 276 315 5 63 44 35 25 12043 2;0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55;-1 0 0 0 0 1 0 1 0 0 1 0 0 1 0 1 0 0 0 0 0 0 1 0 1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 -1 0 0 0 1 0 0 1 0 1 0 0 1 0 1 0 -1;0
```

每个样本由5个 '`;`' 分隔的字段组成，数据格式: `token_ids; sentence_type_ids; position_ids; seg_labels; next_sentence_label`；其中 `seg_labels` 表示分词边界信息: 0表示词首、1表示非词首、-1为占位符, 其对应的词为 `CLS` 或者 `SEP`；

#### 开始训练
预训练任务的启动脚本是 [`script/pretrain.sh`](./script/pretrain.sh)，
在开始预训练之前需要把 CUDA、cuDNN、NCCL2 等动态库路径加入到环境变量 LD_LIBRARY_PATH 之中；然后执行 `bash script/pretrain.sh` 就可以基于demo数据和默认参数配置开始预训练；

预训练任务进行的过程中会输出当前学习率、训练数据所经过的轮数、当前迭代的总步数、训练误差、训练速度等信息，根据 --validation_steps ${N} 的配置，每间隔 N 步输出模型在验证集的各种指标:
```
current learning_rate:0.000001
epoch: 1, progress: 1/1, step: 30, loss: 10.540648, ppl: 19106.925781, next_sent_acc: 0.625000, speed: 0.849662 steps/s, file: ./data/demo_train_set.gz, mask_type: mask_word
feed_queue size 70
current learning_rate:0.000001
epoch: 1, progress: 1/1, step: 40, loss: 10.529287, ppl: 18056.654297, next_sent_acc: 0.531250, speed: 0.849549 steps/s, file: ./data/demo_train_set.gz, mask_type: mask_word
feed_queue size 70
current learning_rate:0.000001
epoch: 1, progress: 1/1, step: 50, loss: 10.360563, ppl: 16398.287109, next_sent_acc: 0.625000, speed: 0.843776 steps/s, file: ./data/demo_train_set.gz, mask_type: mask_word
```
如果用自定义的真实数据进行训练，请参照[`script/pretrain.sh`](./script/pretrain.sh)脚本对参数做相应修改。

### Fine-tuning 任务

在完成 ERNIE 模型的预训练后，即可利用预训练参数在特定的 NLP 任务上做 Fine-tuning。以下基于 ERNIE 的预训练模型，示例如何进行分类任务和序列标注任务的 Fine-tuning，如果要运行这些任务，请通过 [模型&数据](#模型-数据) 一节提供的链接预先下载好对应的预训练模型。

将下载的模型解压到 `${MODEL_PATH}` 路径下，`${MODEL_PATH}` 路径下包含模型参数目录 `params`;

将下载的任务数据解压到 `${TASK_DATA_PATH}` 路径下，`${TASK_DATA_PATH}` 路径包含 `LCQMC`、`XNLI`、`MSRA-NER`、`ChnSentCorp`、 `nlpcc-dbqa` 5个任务的训练数据和测试数据；

#### 单句和句对分类任务

1) **单句分类任务**:

 以 `ChnSentiCorp` 情感分类数据集作为单句分类任务示例，数据格式为包含2个字段的tsv文件，2个字段分别为: `text_a  label`, 示例数据如下:
 ```
label  text_a
0   当当网名不符实，订货多日不见送货，询问客服只会推托，只会要求用户再下订单。如此服务留不住顾客的。去别的网站买书服务更好。
0   XP的驱动不好找！我的17号提的货，现在就降价了100元，而且还送杀毒软件！
1   <荐书> 推荐所有喜欢<红楼>的红迷们一定要收藏这本书,要知道当年我听说这本书的时候花很长时间去图书馆找和借都没能如愿,所以这次一看到当当有,马上买了,红迷们也要记得备货哦!
 ```

执行 `bash script/run_ChnSentiCorp.sh` 即可开始finetune，执行结束后会输出如下所示的在验证集和测试集上的测试结果:

```
[dev evaluation] ave loss: 0.189373, ave acc: 0.954167, data_num: 1200, elapsed time: 14.984404 s
[test evaluation] ave loss: 0.189387, ave acc: 0.950000, data_num: 1200, elapsed time: 14.737691 s
```

2) **句对分类任务**：

以 `LCQMC` 语义相似度任务作为句对分类任务示例，数据格式为包含3个字段的tsv文件，3个字段分别为: `text_a    text_b   label`，示例数据如下:
```
text_a  text_b  label
开初婚未育证明怎么弄？  初婚未育情况证明怎么开？    1
谁知道她是网络美女吗？  爱情这杯酒谁喝都会醉是什么歌    0
这腰带是什么牌子    护腰带什么牌子好    0
```
执行 `bash script/run_lcqmc.sh` 即可开始finetune，执行结束后会输出如下所示的在验证集和测试集上的测试结果:

```
[dev evaluation] ave loss: 0.290925, ave acc: 0.900704, data_num: 8802, elapsed time: 32.240948 s
[test evaluation] ave loss: 0.345714, ave acc: 0.878080, data_num: 12500, elapsed time: 39.738015 s
```

#### 序列标注任务

1) **实体识别**

 以 `MSRA-NER(SIGHAN 2006)` 作为示例，数据格式为包含2个字段的tsv文件，2个字段分别为: `text_a  label`, 示例数据如下:
 ```
 label  text_a
 在 这 里 恕 弟 不 恭 之 罪 ， 敢 在 尊 前 一 诤 ： 前 人 论 书 ， 每 曰 “ 字 字 有 来 历 ， 笔 笔 有 出 处 ” ， 细 读 公 字 ， 何 尝 跳 出 前 人 藩 篱 ， 自 隶 变 而 后 ， 直 至 明 季 ， 兄 有 何 新 出 ？    O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O
相 比 之 下 ， 青 岛 海 牛 队 和 广 州 松 日 队 的 雨 中 之 战 虽 然 也 是 0 ∶ 0 ， 但 乏 善 可 陈 。   O O O O O B-ORG I-ORG I-ORG I-ORG I-ORG O B-ORG I-ORG I-ORG I-ORG I-ORG O O O O O O O O O O O O O O O O O O O
理 由 多 多 ， 最 无 奈 的 却 是 ： 5 月 恰 逢 双 重 考 试 ， 她 攻 读 的 博 士 学 位 论 文 要 通 考 ； 她 任 教 的 两 所 学 校 ， 也 要 在 这 段 时 日 大 考 。    O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O
 ```

执行 `bash script/run_msra_ner.sh` 即可开始 finetune，执行结束后会输出如下所示的在验证集和测试集上的测试结果:

```
[dev evaluation] f1: 0.951949, precision: 0.944636, recall: 0.959376, elapsed time: 19.156693 s
[test evaluation] f1: 0.937390, precision: 0.925988, recall: 0.949077, elapsed time: 36.565929 s
```

### FAQ

#### 如何获取输入句子经过 ERNIE 编码后的 Embedding 表示?

可以通过 ernie_encoder.py 抽取出输入句子的 Embedding 表示和句子中每个 token 的 Embedding 表示，数据格式和 [Fine-tuning 任务](#Fine-tuning-任务) 一节中介绍的各种类型 Fine-tuning 任务的训练数据格式一致；以获取 LCQM dev 数据集中的句子 Embedding 和 token embedding 为例，示例脚本如下:

```
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=7

python -u ernir_encoder.py \
                   --use_cuda true \
                   --batch_size 32 \
                   --output_dir "./test" \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --data_set ${TASK_DATA_PATH}/lcqmc/dev.tsv \
                   --vocab_path config/vocab.txt \
                   --max_seq_len 128 \
                   --ernie_config_path config/ernie_config.json
```

上述脚本运行结束后，会在当前路径的 test 目录下分别生成 `cls_emb.npy` 文件存储句子 embeddings 和 `top_layer_emb.npy` 文件存储 token embeddings; 实际使用时，参照示例脚本修改数据路径、embeddings 文件存储路径等配置即可运行；

#### 如何获取输入句子中每个 token 经过 ERNIE 编码后的 Embedding 表示？

[解决方案同上](#如何获取输入句子经过-ERNIE-编码后的-Embedding-表示?)
