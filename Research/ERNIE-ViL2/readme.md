简体中文|[English](./readme_en.md) 
# ERNIE-ViL 2.0 跨模态理解大模型 
近年来，基于大规模数据预训练的跨模态模型取得了令人瞩目的成绩。基于**对比学习**的双塔预训练框架能够利用大规模的噪声图文数据，在跨模态检索等任务上展现出较大的效果提升，同时具备计算效率高等优势，受到了广泛的关注（如[CLIP](https://arxiv.org/pdf/2103.00020.pdf)，[ALIGN](https://arxiv.org/pdf/2102.05918.pdf)等）。然而，已有的视觉-语言预训练技术基于单视角的对比学习，无法同时学习多种模态间和模态内的关联性。  
**ERNIE-ViL 2.0**提出了一种*基于多视角对比学习*的预训练框架，通过构建丰富的视觉/文本视角，能够同时学习模态间和模态内的多种关联性，从而学习到更鲁棒的跨模态对齐，在跨模态检索等任务上取得了业界领先水平。

## 方法
ERNIE-ViL 2.0 的多视角对比学习包括：
- 跨模态对比学习：图-文（描述），图-文（object tags序列）
- 模态内对比学习：图-图，文-文

![ERNIE-ViL2.0](./packages/src/framework.png)
## 跨模态检索效果 （Zero-shot）
* **ERNIE-ViL 2.0 BASE（ViT）**：ViT-B-16（视觉backbone）+ERNIE 3.0 Base （文本backbone）

| 数据集     | T2I R@1 |   I2T R@1 |   meanRecall  |  
|------------|-------|--------|----|
| [COCO-CN](https://arxiv.org/pdf/1805.08661.pdf) | 66.00 | 65.90 |  84.28 |    
| [AIC-ICC](https://arxiv.org/pdf/1711.06475.pdf) | 17.93 | 30.41 |  38.57 |
* **ERNIE-ViL 2.0 Large（ViT）**：ViT-L-14（视觉backbone）+ERNIE 3.0 Large （文本backbone）

| 数据集     | T2I R@1 |   I2T R@1 |   meanRecall  |  
|------------|-------|--------|----|
| [COCO-CN](https://arxiv.org/pdf/1805.08661.pdf) | 70.30 | 68.80| 86.32 |    
| [AIC-ICC](https://arxiv.org/pdf/1711.06475.pdf) | 20.17 | 32.29 | 41.08 |

* 这里AIC-ICC 为validation 集合的前10,000 行效果 

## 例子
这里以ERNIE-ViL 2.0 Base（ViT）（开源）,在COCO-CN上进行ZERO-SHOT的图文检索任务为例子：
* 模型下载:
[ERNIE-ViL 2.0 Base（ViT)](http://bj.bcebos.com/wenxin-models/ERNIE_VIL2_BASE_ViT.pdparams)
* 数据准备：下载[COCO-CN的测试集](http://bj.bcebos.com/wenxin-models/test.coco_cn.data),然后在配置文件设置输入路径,数据格式(默认为UTF-8编码), 为三列，由\t分开，第一列是文本，第二列是coco中的图像ID, 第三列是由base64编码的图片。
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
以下是ERNIE-ViL 2.0 Base模型在COCO-CN的结果  
| Name       |   R@1 |   R@5 |   R@10 |   meanRecall |
|------------|-------|-------|--------|--------------|
| Text2Image | 66.00 | 90.00 |  96.10 |        84.03 |
| Image2Text | 65.90 | 91.40 |  96.30 |        84.53 | 
| MeanRecall | 65.95 | 90.70 |  96.20 |        84.28 |  

## 备注
- ERNIE-ViL 2.0 base模型已经开源，Large模型请移步[文心官网](https://wenxin.baidu.com/)。
