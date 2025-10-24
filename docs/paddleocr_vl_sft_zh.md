[English](./paddleocr_vl_sft.md) | 简体中文

# PaddleOCR-VL-0.9B SFT

## 引言
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


## SFT 训练

- 在 [huggingface](https://huggingface.co/PaddlePaddle/PaddleOCR-VL/tree/main/PaddleOCR-VL-0.9B) 或者 [modelscope](https://modelscope.cn/models/PaddlePaddle/PaddleOCR-VL/files) 可以下载 PaddleOCR-VL-0.9B 模型。

```
huggingface-cli download PaddlePaddle/PaddleOCR-VL --local-dir PaddlePaddle/PaddleOCR-VL
```

- 我们提供了[孟加拉语训练数据集](https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-train_Bengali.jsonl)可用于微调 PaddleOCR-VL-0.9B 对孟加拉语进行识别，或者你可以按照 [SFT VL 数据集格式](./datasets.md#sft-vl-dataset) 来构建自己的微调数据集。
- 使用命令行和 YAML 配置文件来启动训练：
```
erniekit train examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml \
        model_name_or_path=PaddlePaddle/PaddleOCR-VL \
        train_dataset_path=/PAHT/TO/DATASET \
```
- 我们提供了[孟加拉语测试数据集](https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-test_Bengali.jsonl)，可用于推理来观察微调效果。

### 超参数说明

- `max_steps`：训练总步数, 约等于 `(D × E) / (G × B × A)`。
    - `D`：数据集中训练样本数目。
    - `E`：训练轮次数目。
    - `G`：数据并行的 GPU 数目。
    - `B`：每 GPU 每步的批大小 (打包大小)。
    - `A`：梯度累积的步数。
- `warmup_steps`：线性预热步数, 建议设置成最大步数的 1% `0.01 × max_steps`。
- `packing_size`：单个序列中打包的样本数目。
- `padding`：设置为 `False` 以避免将序列填充到最大序列长度。
- `max_seq_len`：确保该值大于输入样本的最大序列长度。