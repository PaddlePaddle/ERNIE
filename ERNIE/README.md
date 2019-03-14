## Ernie: **E**nhanced **R**epresentation from k**N**owledge **I**nt**E**gration

*Ernie* 通过建模海量数据中的词、实体及实体关系，学习真实世界的语义知识。相较于 *Bert* 学习局部语言共现的语义表示，*Ernie* 直接对语义知识进行建模，增强了模型语义表示能力。

这里我们举个例子：

```Learnt by Bert ：哈 [mask] 滨是 [mask] 龙江的省会，[mask] 际冰 [mask] 文化名城。```

```Learnt by Ernie：[mask] [mask] [mask] 是黑龙江的省会，国际 [mask] [mask] 文化名城。```

在 *Bert* 模型中，我们通过『哈』与『滨』的局部共现，即可判断出『尔』字，模型没有学习与『哈尔滨』相关的任何知识。而 *Ernie* 通过学习词与实体的表达，使模型能够建模出『哈尔滨』与『黑龙江』的关系，学到『哈尔滨』是 『黑龙江』的省会以及『哈尔滨』是个冰雪城市。

训练数据方面，百科类、资讯类中文语料外，*Ernie* 还引入了论坛对话类数据，利用 **DLM**（Dialogue Language Model）建模 Query-Response 对话结构，将对话 Pair 对作为输入，引入 Dialogue Embedding 标识对话的角色，利用 Dialogue Response Loss 学习对话的隐式关系，进一步提升模型的语义表示能力。

我们在自然语言推断，语义相似度，命名实体识别，情感分析，问答匹配 5 个公开的中文数据集合上进行了效果验证，*Ernie* 模型相较 *Bert* 取得了更好的效果。

<table>
  <tbody>
    <tr>
      <th><strong>数据集</strong>
        <br></th>
      <th colspan="2"><strong>XNLI</strong></th>
      <th colspan="2"><strong>LCQMC</strong></th>
      <th colspan="2"><strong>MSRA-NER</strong></th>
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
        <strong>Bert
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
        <strong>Ernie
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
LCQMC 是哈尔滨工业大学在自然语言处理国际顶会 COLING2018 构建的问答匹配数据集其目，标是判断两个问题的语义是否相同。[链接: http://aclweb.org/anthology/C18-1166]
```

 - **命名实体识别任务** MSRA-NER

```text
MSRA-NER 数据集由微软亚研院发布，其目标是命名实体识别，是指识别文本中具有特定意义的实体，主要包括人名、地名、机构名等。[链接: http://sighan.cs.uchicago.edu/bakeoff2005/]
```

 - **情感分析任务** ChnSentiCorp

```text
ChnSentiCorp 是中文情感分析数据集，其目标是判断一段话的情感态度。
```

 - **检索式问答任务** nlpcc-dbqa 

 ```text
nlpcc-dbqa是由国际自然语言处理和中文计算会议NLPCC于2016年举办的评测任务，其目标是选择能够回答问题的答案。[链接: http://tcci.ccf.org.cn/conference/2016/dldoc/evagline2.pdf]
```
