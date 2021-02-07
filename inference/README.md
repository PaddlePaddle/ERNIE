# ERNIE fast inference (C++)

ERNIE C++ fast inference API提供了一种更为高效的在线预测方案，可以直接联编译至生产环境以获取更好的性能。
其实现基于[fluid inference](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/inference_deployment/inference/native_infer.html).
**请确保您的 fluid inference 版本高于 1.7 以获得正确的预测结果。**
本页面提供了一个ERNIE C++ fast inference 的 demo benchmark.

## 准备工作

demo 数据取自XNLI数据集test集合，位于./data 中。采用明文id格式，一行代表一个 batch, 包含四个字段：
```text
src_ids, pos_ids, sent_ids, self_attn_mask
```
字段之间按照分号(;)分隔；各字段内部包含 `shape` 和 `data` 两部分，按照冒号(:)分隔； `shape` 和 `data` 内部按空格分隔；`self_attn_mask` 为 FLOAT32 类型，其余字段为 INT64 类型。

ERNIE fast inference 需要输入 inference\_model 格式的模型，可以参考[这里](../README.zh.md#生成inference_model)生成 inference\_model .

**使用propeller产出的 inference\_model 只需要`src_ids`，`sent_ids` 两个字段，因此需要适当修改数据文件**


## 编译和运行

为了编译本 demo，c++ 编译器需要支持 C++11 标准。

下载对应的 [fluid_inference库](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html) , 根据使用的 paddle 的版本和配置状况 (是否使用 avx, mkl, 以及 cuda, cudnn 版本) 选择下载对应的版本并解压，会得到 `fluid_inference` 文件夹，将其放在与`inference.cc`同一级目录。

用以下命令编译：
``` bash
cd ./gpu # cd ./cpu
mkdir build
cd build
cmake ..
make
```

用以下命令运行：
```
./run.sh ../data/sample /path/to/inference_mode_dir
```

## 性能测试

测试样本：XNLI test集合，输入BatchSize=1, SequenceLength=128.
重复5遍取平均值。

| 测试环境 | 延迟(ms) |
| ----- | -----    |
| CPU（Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz (20 线程)） | 29.8818|
| GPU （P4）  | 8.5 |
