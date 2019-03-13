## Ernie: **E**nhanced **R**epresentation from k**N**owledge **I**nt**E**gration

*Ernie* 通过建模海量数据中的词、实体及实体关系，学习真实世界的语义知识。相较于 *Bert* 学习局部语言共现的语义表示，*Ernie* 直接对语义知识进行建模，增强了模型语义表示能力。

这里我们举个例子：

```Learnt by Bert ：哈 [mask] 滨是 [mask] 龙江的省会，[mask] 际冰 [mask] 文化名城。```

```Learnt by Ernie：[mask] [mask] [mask] 是黑龙江的省会，国际 [mask] [mask] 文化名城。```

在 *Bert* 模型中，我们通过『哈』与『滨』的局部共现，即可判断出『尔』字，模型没有学习与『哈尔滨』相关的任何知识。而 *Ernie* 通过学习词与实体的表达，使模型能够建模出『哈尔滨』与『黑龙江』的关系，学到『哈尔滨』是 『黑龙江』的省会以及『哈尔滨』是个冰雪城市。

此外， *Ernie* 引入了百科、新闻、论坛回帖等多源中文语料进行训练。

我们在多个公开的中文数据集合上进行了效果验证，*Ernie* 模型相较 *Bert*， 取得了更好的效果。

<table style="margin-left: 30.0px;">
  <tbody style="margin-left: 30.0px;">
    <tr style="margin-left: 30.0px;">
      <th class="confluenceTh"><strong>数据集</strong>
        <br></th>
      <th style="text-align: center;margin-left: 30.0px;" colspan="2"><strong>xnli</strong></th>
      <th style="text-align: center;margin-left: 30.0px;" colspan="2"><strong>lcqmc</strong></th>
      <th style="text-align: center;margin-left: 30.0px;" colspan="2"><strong>msra ner</strong></th>
      <th style="text-align: center;margin-left: 30.0px;" colspan="2"><strong>chnsenticorp</strong></th>
      <th style="text-align: center;margin-left: 30.0px;" colspan="4"><strong>nlpcc-dbqa</strong></th></tr>
    <tr style="margin-left: 30.0px;">
      <td  rowspan="2">
        <p>
          <strong>评估</strong></p>
        <p>
          <strong>指标</strong>
          <br></p>
      </td>
      <td  style="margin-left: 30px; text-align: center;" colspan="2">
        <strong>acc</strong>
        <br></td>
      <td  style="margin-left: 30px; text-align: center;" colspan="2">
        <strong>acc</strong>
        <br></td>
      <td  style="margin-left: 30px; text-align: center;" colspan="2">
        <strong>f1-score</strong>
        <br></td>
      <td  style="margin-left: 30px; text-align: center;" colspan="2">
        <strong>acc</strong>
        <strong></strong>
        <br></td>
      <td  style="margin-left: 30px; text-align: center;" colspan="2">
        <strong>mrr</strong>
        <br></td>
      <td  style="margin-left: 30px; text-align: center;" colspan="2">
        <strong>f1-score</strong>
        <br></td>
    </tr>
    <tr style="margin-left: 30.0px;">
      <td colspan="1"  style="text-align: center;" width="">
        <strong>dev</strong>
        <br></td>
      <td colspan="1"  style="text-align: center;" width="">
        <strong>test</strong>
        <br></td>
      <td colspan="1"  style="text-align: center;" width="">
        <strong>dev</strong>
        <br></td>
      <td colspan="1"  style="text-align: center;" width="">
        <strong>test</strong>
        <br></td>
      <td colspan="1"  style="text-align: center;" width="">
        <strong>dev</strong>
        <br></td>
      <td colspan="1"  style="text-align: center;" width="">
        <strong>test</strong>
        <br></td>
      <td colspan="1"  style="text-align: center;" width="">
        <strong>dev</strong>
        <br></td>
      <td colspan="1"  style="text-align: center;" width="">
        <strong>test</strong>
        <br></td>
      <td colspan="1"  style="text-align: center;" width="">
        <strong>dev</strong>
        <br></td>
      <td colspan="1"  style="text-align: center;" width="">
        <strong>test</strong>
        <br></td>
      <td colspan="1"  style="text-align: center;" width="">
        <strong>dev</strong>
        <br></td>
      <td colspan="1"  style="text-align: center;" width="">
        <strong>test</strong>
        <br></td>
    </tr>
    <tr style="margin-left: 30.0px;">
      <td  style="margin-left: 30.0px;">
        <strong>Bert
          <br></strong></td>
      <td style="margin-left: 30px; text-align: center;">78.1</td>
      <td style="margin-left: 30px; text-align: center;">77.2</td>
      <td style="margin-left: 30px; text-align: center;">88.8</td>
      <td style="margin-left: 30px; text-align: center;">87.0</td>
      <td style="margin-left: 30px; text-align: center;">94.0
        <br></td>
      <td style="margin-left: 30px; text-align: center;">
        <span>92.6</span></td>
      <td style="margin-left: 30px; text-align: center;">94.6</td>
      <td style="margin-left: 30px; text-align: center;">94.3</td>
      <td style="margin-left: 30px; text-align: center;" colspan="1">94.7</td>
      <td style="margin-left: 30px; text-align: center;" colspan="1">94.6</td>
      <td style="margin-left: 30px; text-align: center;" colspan="1">80.7</td>
      <td style="margin-left: 30px; text-align: center;" colspan="1">80.8</td></tr>
    <tr style="margin-left: 30.0px;">
      <td style="margin-left: 30.0px;">
        <strong>Ernie
          <br></strong></td>
      <td style="margin-left: 30px; text-align: center;">79.9 <span style="color: red;">(<strong>+1.8</strong>)</span></td>
      <td style="margin-left: 30px; text-align: center;">78.4 <span style="color: red;">(<strong>+1.2</strong>)</span></td>
      <td style="margin-left: 30px; text-align: center;">89.7 <span style="color: red;">(<strong>+0.9</strong>)</span></td>
      <td style="margin-left: 30px; text-align: center;">87.4 <span style="color: red;">(<strong>+0.4</strong>)</span></td>
      <td style="margin-left: 30px; text-align: center;">95.0 <span style="color: red;">(<strong>+1.0</strong>)</span></td>
      <td style="margin-left: 30px; text-align: center;">93.8 <span style="color: red;">(<strong>+1.2</strong>)</span></td>
      <td style="margin-left: 30px; text-align: center;">95.2 <span style="color: red;">(<strong>+0.6</strong>)</span></td>
      <td style="margin-left: 30px; text-align: center;">95.4 <span style="color: red;">(<strong>+1.1</strong>)</span></td>
      <td style="margin-left: 30px; text-align: center;" colspan="1">95.0 <span style="color: red;">(<strong>+0.3</strong>)</span></td>
      <td style="margin-left: 30px; text-align: center;" colspan="1">95.1 <span style="color: red;">(<strong>+0.5</strong>)</span></td>
      <td style="margin-left: 30px; text-align: center;" colspan="1">82.3 <span style="color: red;">(<strong>+1.6</strong>)</span></td>
      <td style="margin-left: 30px; text-align: center;" colspan="1">82.7 <span style="color: red;">(<strong>+1.9</strong>)</span></td></tr>
  </tbody>
</table>

#### 数据集介绍

 - **自然语言推断任务** XNLI
XNLI 由 Facebook 和纽约大学的研究者联合构建，旨在评测模型多语言的句子理解能力。目标是判断两个句子的关系（矛盾、中立、蕴含）。[链接](https://github.com/facebookresearch/XNLI)

 - **语义匹配任务** LCQMC
LCQMC 是哈尔滨工业大学在自然语言处理国际顶会 COLING2018 构建的问答匹配数据集其目，标是判断两个问题的语义是否相同。[链接](http://aclweb.org/anthology/C18-1166)

 - **命名实体识别任务** MSRA-NER
MSRA-NER 数据集由微软亚研院发布，其目标是命名实体识别，是指识别文本中具有特定意义的实体，主要包括人名、地名、机构名等。[链接](http://sighan.cs.uchicago.edu/bakeoff2005/)

 - **情感分析任务** ChnSentiCorp
ChnSentiCorp 是中文情感分析数据集，其目标是判断一段话的情感态度。

 - **检索式问答任务** nlpcc-dbqa 
nlpcc-dbqa是由国际自然语言处理和中文计算会议NLPCC于2016年举办的评测任务，其目标是选择能够回答问题的答案。[链接](http://tcci.ccf.org.cn/conference/2016/dldoc/evagline2.pdf)
