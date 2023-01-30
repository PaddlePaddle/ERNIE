Simplified [Chinese](./readme.md) |English

# ERNIE-ViL 2.0: Multi-View Contrastive Learning for Image-Text Pre-training
>[_**ERNIE-ViL 2.0: Multi-view Contrastive Learning for Image-Text Pre-training**_](https://arxiv.org/pdf/2209.15270.pdf)
>
>Bin Shan, Weichong Yin, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang
>
>

## Methods
ERNIE-ViL 2.0's multi-view  contrastive learning includes:
- Cross modal contrastive learning: image-caption, image-objects
- Modal contrast learning: iamge-image, text-text

![ERNIE-ViL2.0](./packages/src/framework.png)
## Cross modal retrieval effect
The following is the zero shot results of Chinese and English models in Flickr30K and COCO-CN. See the paper for other details.

* **ERNIE-ViL 2.0 （English）on Flickr30k**:

| Name       |   R@1 |   R@5 |   R@10 |  
|------------|-------|-------|--------|
| Text2Image | 85.0 | 97.0 |  98.3 |      
| Image2Text | 96.1 | 99.9 |  100.0 |  
* **ERNIE-ViL 2.0 （Chinese） on COCO-CN**:

| Name       |   R@1 |   R@5 |   R@10 |   
|------------|-------|-------|--------|  
| Text2Image | 69.6 | 91.2 |  96.9 |    
| Image2Text | 69.1 | 92.9 |  97.1 |
 

## Examples
Here, ERNIE-ViL 2.0 base (ViT) (open source)（chinese model） is used as an example to perform the text retrieval task of zero-shot on COCO-CN:  

* Model Download:
[ERNIE-ViL 2.0 Base（ViT)](http://bj.bcebos.com/wenxin-models/OPEN_ERNIE_VIL2_BASE_ViT.pdparams)
* Data preparation: we have built in a [COCO-CN](https://github.com/li-xirong/coco-cn) test set. The data format (UTF-8 encoding by default) is three columns separated by \t. The first column is text, the second column is the image ID in coco, and the third column is the image encoded by Base64.
* First, install the environment and install [paddle>=2.1.3](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.HTML) and [requirements.txt](requirements.txt),
* Then, for ./packages/configs/ernie_vil_base.yaml performs various configurations. For details, please refer to the notes in the configuration (including input/output path location and model parameter path).
* Finally, run the following command to get cross modal graphic embeddings

```bash
# Usage: bash $0 gpu-card-index config-path
$ bash run_infer.sh 2 ./packages/configs/ernie_vil_base.yaml 
```
By define in /packages/configs/ernie_vil_base.yaml The location of the output result defined by base.yaml is evaluated using the following script:

```bash
# Usage: python $0 output-embedding-path
$ python eval_retri.py test_out/cross_modal_embeddings.out
```
The following is the results of ERNIE-ViL 2.0 Base model in COCO-CN. See the paper for detailed results

| Name       |   R@1 |   R@5 |   R@10 |   meanRecall |
|------------|-------|-------|--------|--------------|
| Text2Image | 65.9 | 90.1 |  96.1 |        84.0 |
| Image2Text | 66.5 | 91.6 |  96.2 |        84.8 | 
| MeanRecall | 66.2 | 90.9 |  96.2 |        84.4 | 
