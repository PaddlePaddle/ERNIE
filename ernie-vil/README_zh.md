
[English](./README.md) | 简体中文

## _ERNIE-ViL_: Knowledge Enhanced Vision-Language Representations Through Scene Graph
- [模型框架](#模型框架)
- [预训练模型](#预训练模型)
- [下游任务](#下游任务)
  * [视觉常识推理](#视觉常识推理)
  * [视觉问答](#视觉问答)
   * [跨模态检索](#跨模态检索)
  * [引用表达式理解](#引用表达式理解)
- [使用说明](#使用说明)
  * [安装飞桨](#安装飞桨)
  * [运行微调](#运行微调)
  * [预测](#预测)
- [引用](#引用)

关于算法的详细描述，请参见我们的论文

>[_**ERNIE-ViL:Knowledge Enhanced Vision-Language Representations Through Scene Graph**_](https://arxiv.org/abs/2006.16934)
>
>Fei Yu\*, Jiji Tang\*, Weichong Yin, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang (\* : equal contribution)
>
>Preprint June 2020
>
![ERNIE-ViL](https://img.shields.io/badge/预训练-视觉语言联合表示-green)![VQA](https://img.shields.io/badge/视觉问答-VQA-yellow) ![VCR](https://img.shields.io/badge/视觉常识推理-VCR-blue) ![RefCOCO](https://img.shields.io/badge/引用表达式理解-RefCOCO+-green) ![IRTR](https://img.shields.io/badge/跨模态检索-IR&TR-yellowgreen) 


---
**ERNIE-ViL
是面向视觉-语言任务的知识增强预训练框架**，首次在视觉-语言预训练中引入了结构化的知识。ERNIE-ViL利用场景图中的结构化知识，构建了**物体预测，属性预测，关系预测**三种预训练任务，精细地刻画了视觉-语言模态之间细粒度语义的对齐，从而获得了更好的视觉-语言联合表示。

## 模型框架

基于文本中解析出的场景图，ERNIE-ViL提出了三个多模态场景图预测任务：
- **物体预测**：随机选取图中的一部分物体，然后对其在句子中对应的词进行掩码和预测；
- **属性预测**：对于场景图中的属性-物体组合，随机选取一部分词对其中属性词进行掩码和预测；
- **关系预测**：对于场景图中的物体-关系-物体三元组，对其中的关系词进行掩码和预测。

![ernie_vil_struct](.meta/ernie_vil_struct.png)

ERNIE-ViL 场景图预训练任务结构

## 预训练模型


ERNIE-ViL使用大规模图文对齐数据作为预训练数据，基于[**Conceptual
Captions**](https://www.aclweb.org/anthology/P18-1238.pdf)和[**SBU
Captions**](http://papers.nips.cc/paper/4470-im2text-describing-images-using-1-million-captio)两个out-of-domain数据集，训练两种参数规模模型如下：

- [**ERNIE-ViL _base_**](https://ernie-github.cdn.bcebos.com/model-ernie-vil-base-en.1.tar.gz) (_lowercased | 12-text-stream-layer, 6-visual-stream-layer_)
- [**ERNIE-ViL _large_**](https://ernie-github.cdn.bcebos.com/model-ernie-vil-large-en.1.tar.gz) (_lowercased | 24-text-stream-layer, 6-visual-stream-layer_)

基于两个out-of-domian数据集([**Conceptual
Captions**](https://www.aclweb.org/anthology/P18-1238.pdf)，[**SBU
Captions**](http://papers.nips.cc/paper/4470-im2text-describing-images-using-1-million-captio))和两个in-domain数据集([**MS-COCO**](https://arxiv.org/abs/1405.0312)，[**Visual-Genome**](https://arxiv.org/abs/1602.07332))训练了large参数规模的模型：

- [**ERNIE-ViL-Out&in-domain _large_**](https://ernie-github.cdn.bcebos.com/model-ernie-vil-all-domain-large-en.1.tar.gz) (_lowercased | 24-text-stream-layer, 6-visual-stream-layer_)

## 下游任务

ERNIE-ViL在五个视觉语言下游任务进行了实验，包括[**视觉常识推理**](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zellers_From_Recognition_to_Cognition_Visual_Commonsense_Reasoning_CVPR_2019_paper.pdf)，
[**视觉问答**](https://openaccess.thecvf.com/content_iccv_2015/papers/Antol_VQA_Visual_Question_ICCV_2015_paper.pdf)，
[**跨模态图片检索**](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00166)，
[**跨模态文本检索**](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00166)，
[**引用表达式理解**](https://www.aclweb.org/anthology/D14-1086.pdf)，与主流模型的效果对比可以参考开源论文。



### **视觉常识推理**
   * 数据集合
      * 训练、验证和测试集合相关数据可以由[**视觉常识推理官网**](http://visualcommonsense.com/download/)获取；
      * 视觉端特征的组织方式借鉴[**ViLBERT**](https://github.com/jiasenlu/vilbert_beta), 因此项目直接使用**ViLBERT**中的数据，数据[下载地址](https://github.com/jiasenlu/vilbert_beta/tree/master/data);
      * 将所有获取的文件放在 data/vcr 目录下；
      
   * 任务预训练： 基于ERNIE-ViL的out-of-domain模型，在视觉推理任务中进行了任务预训练，预训练获得模型如下
      * [**ERNIE-ViL-VCR-task-pretrain _base_**](https://ernie-github.cdn.bcebos.com/model-ernie-vil-base-VCR-task-pre-en.1.tar.gz)
      * [**ERNIE-ViL-VCR-task-pretrain _large_**](https://ernie-github.cdn.bcebos.com/model-ernie-vil-large-VCR-task-pre-en.1.tar.gz)
      
   * 效果: ERNIE-ViL在视觉常识推理任务上的效果对比如下：

      | 模型                                |      <strong>Q->A</strong>    |      <strong>QA->R</strong>    |     <strong>Q->AR</strong>       |
      | :---------------------------------- | :---------------------------: | :----------------------------: | :---------------------------:    |
      | ERNIE-ViL (task-pretrain) _base_    |           76.37(77.0)         |            79.65(80.3)         |           61.24(62.1)            |
      | ERNIE-ViL (task-pretrain) _large_   |  <strong>78.52(79.2)</strong> |  <strong>83.37(83.5)</strong>  |  <strong/>65.81(66.3) </strong>  |

      _注：括号外表示验证集效果，括号内表示测试集效果，测试集效果提交到[VCR榜单](https://visualcommonsense.com/leaderboard/)获得。_

### **视觉问答**
   * 数据集合
       * 原始图片、问题和答案可以由[**视觉问答官网**](https://visualqa.org/)获取。
       * 视觉端特征使用[**bottom-up attention**](https://github.com/jiasenlu/bottom-up-attention)中的工具提取，提取的box动态值为100-100。
       * 训练 & 测试数据按照如下方式组织:
           ```script
           question_id, question, answer_label, answer_score, image_w, image_h, number_box, image_loc, image_embeddings
           ```
           _多个答案的label和score用 ‘|’ 分隔，和image相关的项均可以从bottom up attention的工具提取。_
           
   * 效果：ERNIE-ViL的三种预训练模型在**视觉问答**任务下的效果如下表
   
      | 模型                               |      <strong>test-dev</strong>    |      <strong>test-std</strong>    |
      | :-------------------------------- | :-------------------------------: | :------------------------------:  | 
      | ERNIE-ViL _base_                  |           73.18                   |              73.36                |         
      | ERNIE-ViL _large_                 |           73.78                   |              73.96                |
      | ERNIE-ViL-Out&in-domain _large_   |           74.95                   |              75.10                |    
      
      
### **跨模态检索**
   * 数据集合
       * 原始图片和文本描述相关的数据，可以从[**这里**](https://www.kaggle.com/hsankesara/flickr-image-dataset)获取。
       * 视觉端特征使用[**bottom-up attention**](https://github.com/jiasenlu/bottom-up-attention)提取，提取的box动态值为0-36。
       * 文本相关的数据可以参见data/flickr给出的示例 flickr.dev.data，图片端特征组织方式为
           ```script
           image_w, image_h, number_box, image_loc, image_embeddings
           ```
           
   * 效果
       * ERNIE-ViL的三种预训练模型在**跨模态图片检索（Flickr30k 数据集）**上的效果如下表
          | 模型                               |    <strong>R@1</strong>  |    <strong>R@5</strong>   |   <strong>R@10</strong>   |
          | :-------------------------------- | :---------------------:  | :----------------------:  | :----------------------:  | 
          | ERNIE-ViL _base_                  |           74.44          |          92.72            |           95.94           |        
          | ERNIE-ViL _large_                 |           75.10          |          93.42            |           96.26           |
          | ERNIE-ViL-Out&in-domain _large_   |           76.66          |          94.16            |           96.76           |
          
       * ERNIE-ViL的三种预训练模型在**跨模态文本检索（Flickr30k 数据集）**任务上的效果如下表
          | 模型                               |    <strong>R@1</strong>  |    <strong>R@5</strong>   |   <strong>R@10</strong>   |
          | :-------------------------------- | :---------------------:  | :----------------------:  | :----------------------:  | 
          | ERNIE-ViL _base_                  |           86.70          |          97.80            |           99.00           |        
          | ERNIE-ViL _large_                 |           88.70          |          97.30            |           99.10           |
          | ERNIE-ViL-Out&in-domain _large_   |           89.20          |          98.50            |           99.20           |
         
### **引用表达式理解**
   * 数据集合
       * 视觉端特征参考了[MAttNet](https://github.com/lichengunc/MAttNet)的提取方式。
       * 单条训练 & 验证 数据的组织方式为
           ```script
           expressions, image_w, image_h, number_box, number_boxes_gt, image_loc, image_embeddings, box_label, label
           ```
  * 效果
      * ERNIE-ViL的三种预训练模型在**引用表达式理解**任务上的效果如下表：
     
          | 模型                               |   <strong>val</strong>  |    <strong>testA</strong>   |   <strong>testB</strong>   |
          | :-------------------------------- | :---------------------:  | :----------------------:  | :----------------------:  | 
          | ERNIE-ViL _base_                  |           74.02          |          80.33            |           64.74           |        
          | ERNIE-ViL _large_                 |           74.24          |          80.97            |           64.70           |
          | ERNIE-ViL-Out&in-domain _large_   |           75.89          |          82.39            |           66.91           |
   
      
  
    

## 使用说明

### 安装飞桨

ERNIE-ViL代码基于Paddle Fluid 1.8 和 Python 2.7， 依赖的其他模块也列举在 requirements.txt，可以通过下面的指令安装: 
 ```script
      pip install -r requirements.txt
  ```
### 运行微调
在运行 ERNIE-ViL 微调前，需要将 CUDA 、cuDNN 、NCCL2 的动态库路径添加到 LD_LIBRARY_PATH 。 我们把下游任务的参数配置文件放到了 conf/ ，可以简单地通过配置文件运行。 例如，您可以通过下面的指令在各个下游任务上进行微调：

```script
    sh run_finetuning.sh $task_name(vqa/flickr/refcoco_plus/vcr) conf/${task_name}/model_conf_${task_name} $vocab_file $ernie_vil_config $pretrain_models_params
```

前面提供的模型链接中包含了所有需要的文件, 包含词表文件，配置文件和预训练参数。微调相关的模型配置和参数配置可以通过conf/ 目录下的文件找到，这里对论文最优结果（large模型）的一些关键参数进行汇总：

|  Tasks   | Batch Size | Learning Rate | # of Epochs |  GPUs    | Layer Decay rate | Hidden dropout |
|   -----  | ----------:| -------------:| -----------:| --------:| ----------------:| --------------:| 
|  VCR     |   16(x4)   |    1e-4       |      6      |  4x V100 |        0.9       |       0.1      |
|  VQA 2.0 |   64(x4)   |    1e-4       |     15      |  4x V100 |        0.9       |       0.1      |
| RefCOCO+ |   64(x2)   |    1e-4       |     30      |  2x V100 |        0.9       |       0.2      |
| Flickr   |   8(x8)    |    2e-5       |     40      |  8x V100 |        0.0       |       0.1      | 


所有的下游任务的微调实验是在 32 GB 的英伟达V100 GPU上运行，如果您的GPU显存不够，可以考虑更多张卡运行或者减小配置中的batch_size。


### 预测
基于已经训练的模型，您可以通过下面的命令测试下游任务的效果（相关的配置文件可以从之前下载的包获得）

#### VCR
      
 
  ```script
     Task Q->A: sh run_inference.sh vcr qa $split(val/test) conf/vcr/model_conf_vcr $vocab_file $ernie_vil_config $model_params $res_file
  ```
 
  ```script
     Task Q->AR: sh run_inference.sh vcr qar $split(val/test) conf/vcr/model_conf_vcr $vocab_file $ernie_vil_config $model_params $res_file
  ```
  
  
  _VCR的测试可以在一张32GB的英伟达V100 GPU上运行，测试的结果包含Q->A 任务、QA->R任务和Q->AR任务，其中Q->AR任务由前两个任务结果合并所得._

#### VQA
 
  ```script
       sh run_inference.sh vqa eval $split(val/test_dev/test_std) conf/vqa/model_conf_vqa $vocab_file $ernie_vil_config $model_params $res_file
  ```
   注:_VQA的测试样本没有label信息，需要将结果文件提交到[**VQA网站**](https://visualqa.org/)查看结果。_
   
#### RefCOCO+

  ```script
       sh run_inference.sh refcoco_plus eval $split(val/test_A/test_B) conf/refcoco_plus/model_conf_refcoco_plus $vocab_file $ernie_vil_config $model_params $res_file
  ```
  
#### Flickr
   
  ```script
       sh run_inference.sh flickr eval $split(dev/test) conf/flickr/model_conf_flickr $vocab_file $ernie_vil_config $model_params $res_file
  ```
  注：_Flickr的结果是一个预测结果文件，可以参考 tools/get_recall.py 统计一下最终结果。_
  
## 引用

可以按下面的格式引用我们的论文:

```
@article{yu2020ernie,
  title={ERNIE-ViL: Knowledge Enhanced Vision-Language Representations Through Scene Graph},
  author={Yu, Fei and Tang, Jiji and Yin, Weichong and Sun, Yu and Tian, Hao and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:2006.16934},
  year={2020}
}

```

