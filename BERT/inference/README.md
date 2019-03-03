# BERT模型inference demo

## 数据预处理
实际应用场景中，模型部署之后用户还需要编写对应的程序对输入进行处理，然后把得到的数据传给模型进行预测。这里为了演示的需要，用 `gen_demo_data.py` 来进行数据处理，包括 tokenization，batching，numericalization，并且把处理后的数据输出为文本文件。使用方法如下：

``` bash
TASK_NAME="xnli"
DATA_PATH=/path/to/xnli/data/
BERT_BASE_PATH=/path/to/bert/pretrained/model/
python gen_demo_data.py \
    --task_name ${TASK_NAME} \
    --data_path ${DATA_PATH} \
    --vocab_path "${BERT_BASE_PATH}/vocab.txt" \
    --batch_size 4096 \
    --in_tokens \
    > data.txt
```

**生成的数据格式**

生成的数据一行代表一个 `batch`, 包含五个字段

```text
src_id, pos_id, segment_id, self_attention_bias, next_segment_index
```

字段之间按照分号(;)分隔，其中各字段内部 `shape` 和 `data` 按照冒号(:)分隔，`shape` 和 `data` 内部按空格分隔，`self_attention_bias` 为 FLOAT32 类型，其余字段为 INT64 类型。

## 编译和运行

为了编译 inference demo，`c++` 编译器需要支持 `C++11` 标准。

首先下载对应的 [Fluid_inference库](http://paddlepaddle.org/documentation/docs/zh/1.2/advanced_usage/deploy/inference/build_and_install_lib_cn.html) , 根据使用的 paddle 的版本和配置状况 (是否使用 avx, mkl, 以及 cuda, cudnn 版本) 选择下载对应的版本，并解压至 `inference` 目录，会得到 `fluid_inference` 文件夹。

设置运行相关的环境变量(以 `cpu_avx_mkl` 版本为例)

``` bash
LD_LIBRARY_PATH=fluid_inference/paddle/lib/:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=fluid_inference/third_party/install/mklml/lib:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=fluid_inference/third_party/install/mkldnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
```

编译 demo

``` bash
mkdir build
cd build
cmake
make
```

这会在 `build` 目录下生成运行 `inference` 可执行文件。

运行 demo

```bash
./inference --logtostderr \
    --model_dir $MODEL_PATH \
    --data $DATA_PATH \
    --repeat $REPEAT_NUM
```

## 性能测试

E5-2680 CPU，单线程下196样本性能如下

| repeat | 单样本耗时(ms) |
| -----  | -----          |
| 10     | 264.696        |
| 50     | 264.84         |
