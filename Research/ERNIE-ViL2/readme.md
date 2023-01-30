简体中文|[English](./readme_en.md) 
# ERNIE-ViL 2.0 基于多视角对比学习的跨模态预训练模型
更多技术细节请参考 我们的论文：
>[_**ERNIE-ViL 2.0: Multi-view Contrastive Learning for Image-Text Pre-training**_](https://arxiv.org/pdf/2209.15270.pdf)
>
>Bin Shan, Weichong Yin, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang
>
>
近年来，基于大规模数据预训练的跨模态模型取得了令人瞩目的成绩。基于**对比学习**的双塔预训练框架能够利用大规模的噪声图文数据，在跨模态检索等任务上展现出较大的效果提升，同时具备计算效率高等优势，受到了广泛的关注（如[CLIP](https://arxiv.org/pdf/2103.00020.pdf)，[ALIGN](https://arxiv.org/pdf/2102.05918.pdf)等）。然而，已有的视觉-语言预训练技术基于单视角的对比学习，无法同时学习多种模态间和模态内的关联性。  
**ERNIE-ViL 2.0**提出了一种*基于多视角对比学习*的预训练框架，通过构建丰富的视觉/文本视角，能够同时学习模态间和模态内的多种关联性，从而学习到更鲁棒的跨模态对齐，在跨模态检索等任务上取得了业界领先水平。

## 方法
ERNIE-ViL 2.0 的多视角对比学习包括：
- 跨模态对比学习：图-文（描述），图-文（object tags序列）
- 模态内对比学习：图-图，文-文

![ERNIE-ViL2.0](./packages/src/framework.png)
## 跨模态检索效果
以下为以中、英文模型在Flickr30K、COCO-CN的zero-shot结果，其他详见论文。
* **ERNIE-ViL 2.0 英文 on Flickr30k**:

| Name       |   R@1 |   R@5 |   R@10 |  
|------------|-------|-------|--------|
| Text2Image | 85.0 | 97.0 |  98.3 |      
| Image2Text | 96.1 | 99.9 |  100.0 |  

* **ERNIE-ViL 2.0 中文 COCO-CN**:

| Name       |   R@1 |   R@5 |   R@10 |   
|------------|-------|-------|--------|  
| Text2Image | 69.6 | 91.2 |  96.9 |    
| Image2Text | 69.1 | 92.9 |  97.1 |


* 这里结果均为论文最好结果

## 例子
这里以ERNIE-ViL 2.0 Base（ViT）（开源）,在COCO-CN上进行ZERO-SHOT的图文检索任务为例子：
* 模型下载:
[ERNIE-ViL 2.0 Base（ViT)](http://bj.bcebos.com/wenxin-models/OPEN_ERNIE_VIL2_BASE_ViT.pdparams)
* 数据准备：下载[COCO-CN的测试集](https://github.com/li-xirong/coco-cn),然后在配置文件设置输入路径,处理数据格式(默认为UTF-8编码), 为三列，由\t分开，第一列是文本，第二列是coco中的图像ID, 第三列是由base64编码的图片。
* 首先安装环境, 安装 [paddle>=2.1.3](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html), 以及[requirements.txt](requirements.txt) 要求的包，  
* 然后，对 ./packages/configs/ernie_vil_base.yaml 进行各项配置，具体请参考配置中的各项注释(包括输入/输出路径位置和模型参数路径)。
* 最后，运行以下命令，得到跨模态的图文embeddings
```bash
# Usage: bash $0 gpu-card-index config-path
$ bash run_infer.sh 2 ./packages/configs/ernie_vil_base.yaml 
```

通过在./packages/configs/ernie_vil_base.yaml定义的输出结果的位置，使用下面脚本进行评测：


```bash
# Usage: python $0 output-embedding-path
$ python eval_retri.py test_out/cross_modal_embeddings.out
```
以下是ERNIE-ViL 2.0 Base模型在COCO-CN的结果,详细结果见论文  
| Name       |   R@1 |   R@5 |   R@10 |   meanRecall |
|------------|-------|-------|--------|--------------|
| Text2Image | 65.9 | 90.1 |  96.1 |        84.0 |
| Image2Text | 66.5 | 91.6 |  96.2 |        84.8 | 
| MeanRecall | 66.2 | 90.9 |  96.2 |        84.4 |  

## 备注
- ERNIE-ViL 2.0 base模型已经开源，Large模型和最好效果请移步[文心官网](https://wenxin.baidu.com/)。
