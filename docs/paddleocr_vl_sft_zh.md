[English](./paddleocr_vl_sft.md) | 简体中文

# PaddleOCR-VL-0.9B SFT

## 1. 引言
PaddleOCR-VL 是一款为文档解析任务量身打造的、性能顶尖 (SOTA) 且轻量高效的模型。它的核心是 PaddleOCR-VL-0.9B——一个紧凑而强大的视觉语言模型 (VLM)。该模型创新地集成了 NaViT 风格的动态分辨率视觉编码器与 ERNIE-4.5-0.3B 语言模型，从而能够精准地识别各类文档元素。

这款模型不仅能高效支持 109 种语言，还擅长识别文本、表格、公式、图表等复杂元素，并始终保持极低的资源占用。在多个权威的公开及内部基准测试中，PaddleOCR-VL 的页面级文档解析与元素级识别性能均达到了业界顶尖水平。其性能远超现有方案，面对顶级视觉语言模型也极具竞争力，且推理速度飞快。这些杰出特性使其成为在真实场景中落地部署的理想选择。

虽然 PaddleOCR-VL-0.9B 在常见场景下表现出色，但在许多特定或复杂的业务场景中，其性能会遇到瓶颈。例如：
- 特定行业与专业领域
    - 金融与财会领域：识别发票、收据、银行对账单、财务报表等
    - 医疗领域：识别病历、化验单、医生手写处方、药品说明书等
    - 法律领域：识别合同、法律文书、法庭文件、证书等

- 非标准化的文本与字体
    - 手写体识别：识别手写的表单、笔记、信件、问卷调查等
    - 艺术字体与设计字体：识别海报、广告牌、产品包装、菜单上的艺术字体等
    - 古籍与历史文献：识别古代手稿、旧报纸、历史档案等

- 特定任务与输出格式
    - 表格识别与结构化输出：将图像中的表格转换为 Excel、CSV 或 JSON 格式
    - 数学公式识别：识别教科书、论文中的数学公式，并输出为 LaTeX 等格式


这时，就需要通过 SFT (Supervised Fine-Tuning) 来提升模型的准确性和鲁棒性。


## 2. 环境配置

请参考 [ERNIEKit-安装文档](./erniekit.md#2-installation) 中的安装步骤对训练环境进行配置。

## 3. 模型和数据集准备

### 3.1 模型准备

在 [huggingface](https://huggingface.co/PaddlePaddle/PaddleOCR-VL/tree/main/PaddleOCR-VL-0.9B) 或者 [modelscope](https://modelscope.cn/models/PaddlePaddle/PaddleOCR-VL/files) 可以下载 PaddleOCR-VL-0.9B 模型。

```
huggingface-cli download PaddlePaddle/PaddleOCR-VL --local-dir PaddlePaddle/PaddleOCR-VL
```

### 3.2 数据集准备

请参考 [SFT VL 数据集格式](./datasets.md#sft-vl-dataset) 来构建微调数据集，为了方便起见，我们也提供了一个快速上手的[孟加拉语训练数据集](https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-train_Bengali.jsonl)，可用于微调 PaddleOCR-VL-0.9B 对孟加拉语进行识别，使用以下命令下载：

```
wget https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-train_Bengali.jsonl
```

## 4. 训练配置

我们针对孟加拉语示例数据集提供了[配置文件](../examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml)，其中的关键训练超参数如下：

- `max_steps=926`：训练总步数, 约等于 `(D × E) / (G × B × A)`。
    - `D=29605`：数据集中训练样本数目。
    - `E=2`：训练轮次数目。
    - `G=1`：用于训练的 GPU 数目。
    - `B=8`：单卡的训练 Batch Size。
    - `A=8`：梯度累积步数。
- `warmup_steps=10`：线性预热步数, 建议设置成最大步数的 1% `0.01 × max_steps`。
- `packing_size=8`：序列中打包的样本数目，作用等同于 `batch_size`。
- `max_seq_len=16384`：最大序列长度，建议设置成训练过程中显存允许的最大值。
- `gradient_accumulation_steps=8`：梯度累积步数。
    - 每达到该步数整数倍更新一次模型参数。
    - 当显存不足时，可以减小 `packing_size` 并增大 `gradient_accumulation_steps`。
    - 用时间换空间策略，可以减少显存占用，但会延长训练时间。
- `learning_rate=5e-6`：学习率，即每次参数更新的幅度。

## 5. SFT 训练

使用以下命令行即可启动训练：
```
erniekit train examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml \
        model_name_or_path=PaddlePaddle/PaddleOCR-VL \
        train_dataset_path=./ocr_vl_sft-train_Bengali.jsonl \
```
在 1*A800-80 G 上训练时长约为 2 小时。

ERNIEKit 默认使用机器上的全部 GPU，可以通过环境变量 `CUDA_VISIBLE_DEVICES` 设置 ERNIEKit 能够使用的 GPU。

GPU 的数目 `GPU_num` 会影响训练超参数 `learning_rate & packing_size & gradient_accumulation_steps` 配置。理论上，每个更新步使用的样本数目 `sample_num = G*B*A`，近似与学习率 `learning_rate` 成正线形关系，因此，当 GPU 数目增加 `N` 倍变为 `N*GPU` 时，有两种调整方式：
1. 保持 `sample_num` 不变
    - 将 `packing_size` 减少 `x` 倍，变成 `packing_size/x`
    - 将 `gradient_accumulation_steps` 减少 `y` 倍，变成 `gradient_accumulation_steps/y` 
    - 满足 `x*y = N` 即可
2. 将 `learning_rate` 增加 `N` 倍，变成 `N*learning_rate`

可以通过 `tensorboard` 对训练过程可视化，使用以下命令行即可启动（下方命令将端口 port 设置为 `8084`，需要根据实际情况设置可用端口）：

```
pip install tensorboard
tensorboard --logdir ./PaddleOCR-VL-SFT-Bengali/tensorboard_logs/ --port 8084
```

成功启动后该服务后，在浏览器输入 `ip:port` ，则可以看到训练日志（通过 `hostname -i` 命令可以查看机器的 ip 地址）

## 6. 模型结构说明

训练结束后，模型会保存在 `output_dir=./PaddleOCR-VL-SFT-Bengali` 指定路径下，其中包含：

- preprocessor_config.json：图像预处理配置文件
- config.json：模型配置文件
- model-00001-of-00001.safetensors：模型权重文件
    - 保存的模型格式可以通过 `save_to_hf` 控制，默认是 huggingface safetensors 格式
- model.safetensors.index.json & static_name_to_dyg_name.json：模型权重索引文件等，用于辅助模型在多 GPU 上切分与加载
- tokenizer.model & tokenizer_config.json & special_tokens_map.json & added_tokens.json：分词器文件
- train_args.bin：训练参数文件，记录训练使用的参数等
- train_state.json：训练状态文件，记录训练步数和最优指标等
- train_results.json & all_results.json：训练结果文件，记录训练进度&用时&每步耗时&每样本耗时等
- generation.json：生成配置文件
- checkpoint-[save_steps*n]：检查点文件夹，在 `save_steps` 整数倍保存训练状态，除以上文件外，还会保存 master-weight & optimizer-state & scheduler-state 等，可用于训练中断后恢复训练

## 7. 推理

### 7.1 推理环境配置

安装 PaddleX 用于推理

```
python -m pip install paddlex
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
```

### 7.2 推理模型准备

从 PaddleOCR-VL 中拷贝必要的推理配置文件到 SFT 训练完成后保存的模型目录中

```
cp PaddlePaddle/PaddleOCR-VL/chat-template.jinja PaddleOCR-VL-SFT-Bengali
cp PaddlePaddle/PaddleOCR-VL/inference.yaml PaddleOCR-VL-SFT-Bengali
```

### 7.3 推理数据集准备

我们提供了[孟加拉语测试数据集](https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-test_Bengali.jsonl)，可用于推理来观察微调效果，使用以下命令下载：

```
wget https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-test_Bengali.jsonl
```

### 7.4 单样本推理

执行以下 python 代码即可加载模型并对单样本推理：

```python
from paddlex import create_model

model = create_model("PaddleOCR-VL-0.9B", model_dir="PaddleOCR-VL-SFT-Bengali")

# one sample
sample= {"image": "https://paddle-model-ecology.bj.bcebos.com/PPOCRVL/dataset/bengali_sft/5b/7a/5b7a5c1c-207a-4924-b5f3-82890dc7b94a.png", "query": "OCR:"}
# GT： নট চলল রফযনর পঠ সওযর\nহয গলয গলয ভব এখন দটত, মঝ মঝ খবর নয যদও লগ যয\nঝগড\nদরগর কছ চল এল

res = next(model.predict(sample, max_new_tokens=2048, use_cache=True))
res.print()

# Excepted Answer = নট চলল রফযনর পঠ সওযর\nহয গলয গলয ভব এখন দটত, মঝ মঝ খবর নয যদও লগ যয\nঝগড\nদরগর কছ চল এল

```

### 7.5 数据集推理

执行以下 python 代码即可加载模型并对测试数据集推理：

```python
import json
import jsonlines
from paddlex import create_model

model = create_model("PaddleOCR-VL-0.9B", model_dir="PaddleOCR-VL-SFT-Bengali")

with open("./ocr_vl_sft-test_Bengali.jsonl", 'r') as f:
    sample_list = [json.loads(line) for line in f]

for sample in sample_list:
    sample['image'] = sample['image_info'][0]['image_url']
    sample['query'] = "OCR:"
    res = next(model.predict(sample, max_new_tokens=2048, use_cache=True))
    sample['response'] = res['result']

with jsonlines.open("ocr_vl_sft-test_Bengali_response.jsonl", mode='w') as writer:
    writer.write_all(sample_list)
```
