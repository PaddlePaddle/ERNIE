# 数据增强

## 策略简介

- 支持环境： py3.7+PaddlePaddle2.0
- 目前文心2.1.0数据增强工具提供4种增强策略：遮盖、删除、同词性词替换、词向量近义词替换
- 可通过入口参数配置各增强策略的概率比例，在数据的一次增强中使用多种增强策略
- 依赖说明：

| 模块     | 依赖              | 原理简介                                    |
| -------- | ----------------- | ------------------------------------------- |
| unk      | no                | 随机mask                                    |
| truncate | no                | 随机删除                                    |
| pos      | lac               | 同词性词替换，依赖LAC算子进行分词和词性标注 |
| w2v      | lac, gensim, tqdm | 词向量近义词替换                            |

- - gensim安装最新版本：pip install gensim
  - tqdm安装：pip install tqdm
  - [LAC](https://github.com/baidu/lac)安装最新版本：pip install lac

## 脚本运行（说明）

开发套件用于数据增强的python脚本位于目录./wenxin_appzoo/tools/data_aug/data_aug.py
- 运行data_aug.py脚本

```shell
python data_aug.py "输入文件夹的目录" "输出文件夹的目录"
```

- data_aug.py脚本传参说明

```shell
shell输入：
    python data_aug.py -h

shell输出：
    usage: data_aug.py [-h] [-n AUG_TIMES] [-c COLUMN_NUMBER] [-u UNK]
                       [-t TRUNCATE] [-r POS_REPLACE] [-w W2V_REPLACE]
                       [-e ERNIE_REPLACE] [--unk_token UNK_TOKEN]
                       input output
    
    main
    
    positional arguments:
      input                                                #原始待增强数据文件所在文件夹，带label的，一个或多个文本列
      output                                               #输出文件路径
    
    optional arguments:
      -h, --help            show this help message and exit
      -n AUG_TIMES, --aug_times AUG_TIMES                  #数据集数目放大n倍，output行数为input的n+1倍      
      -c COLUMN_NUMBER, --column_number COLUMN_NUMBER      #明文文件中所要增强列的列序号，多列用逗号分割，如：1,2
      -u UNK, --unk UNK                                    #unk 增强策略的概率
      -t TRUNCATE, --truncate TRUNCATE                     #truncate 增强策略的概率
      -r POS_REPLACE, --pos_replace POS_REPLACE            #pos_replace 增强策略的概率
      -w W2V_REPLACE, --w2v_replace W2V_REPLACE            #w2v_replace 增强策略的概率
      --unk_token UNK_TOKEN                    
```

## 下游任务使用demo

### 分类任务

- 使用文心框架中的增强工具在下游任务中进行数据增强，然后再训练
- 进去下游任务（分类任务）

```java
cd ./wenxin_appzoo/tasks/text_classification/
```

- 一键启动数据增强和训练脚本

```shell
sh run_with_data_aug.sh
```

### 效果评测

| **Model** | **训练集样本数目：100条****测试集样本数目：50条**            | **训练集样本数目：200条****测试集样本数目：500条**           | **训练集样本数目：500条****测试集样本数目：500条**           | **训练集样本数目：1000条****测试集样本数目：500条**          |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Base+CNN  | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=7ed6755c53ad49a3a3bbc59a17c444cb)acc/pre: 0.8/0.7971 | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=762eb023d60e46759b3fff3a2619db56)acc/pre: 0.87,0.8725 | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=6aea774fa56c47ef8f4a6fab40dc8c21)acc/pre: 0.906,0.9063 | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=6cf359fda4084454a6b1af52a25d02a4)acc/pre: 0.912,0.9126 |
| +unk      | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=60ea9366705d4503ad61ec6563b8d2d0)acc/pre: 0.84/0.8507 ↑提升4-5个百分点 | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=b3790de6ebff46ec9d00c672d395ad22)acc/pre: 0.906,0.9059 ↑提升3个百分点 | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=35c31f72054d4326a09892c802904864)acc/pre: 0.922,0.9225 ↑提升1.6个百分点 | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=76f2938f494c4aaca0186fd1c71257c7)acc/pre: 0.928,0.9297 ↑提升1.6个百分点 |
| +truncate | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=f4a1a72e38574a49afc8e6ba1b4e18a8)acc/pre: 0.86/0.8667 ↑提升6-7个百分点 | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=dab738beddb849c88fea4d9c03595aa4)acc/pre: 0.908,0.9082 ↑提升3个百分点 | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=69f8ec5f250d40ab99fe53b5d7b1dbbd)acc/pre: 0.916,0.9164 ↑提升1个百分点 | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=25172c7c60a24bb5bf33c36197dce9a0)acc/pre: 0.928,0.9289 ↑提升1.6个百分点 |
| +pos      | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=223bf8d17bb1415698907385bee5d757)acc/pre: 0.82/0.8243 ↑提升2-3个百分点 | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=070d7fb412674901b24d353a60f968ab)acc/pre: 0.906,0.9061 ↑提升3个百分点 | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=5c0939c8568744d397b8d8e7d9fd7db1)acc/pre: 0.926,0.9279 ↑提升4个百分点 | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=e0af74790bb64b91845039faecd455d3)acc/pre: 0.924,0.9261 ↑提升1.2个百分点 |
| +w2v      | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=7f7d269f5241483dbad575cbf454155e)acc/pre: 0.84/0.8381 ↑提升4个百分点 | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=985f71c6720241c590981e49f981c1e3)acc/pre: 0.904,0.904 ↑提升3个百分点 | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=6c0461e6645c463fb7c4e20f2c6628b1)acc/pre: 0.916,0.9164 ↑提升1个百分点 | ![img](http://rte.weiyun.baidu.com/api/imageDownloadAddress?attachId=72f1057a03be40078381a0756118d86e)acc/pre: 0.926,0.926 ↑提升1.4个百分点 |