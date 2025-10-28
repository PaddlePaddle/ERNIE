English | [简体中文](./paddleocr_vl_sft_zh.md)

# PaddleOCR-VL-0.9B SFT

## 1. Introduction

PaddleOCR-VL, a SOTA and resource-efficient model tailored for document parsing. Its core component is PaddleOCR-VL-0.9B, a compact yet powerful vision-language model (VLM) that integrates a NaViT-style dynamic resolution visual encoder with the ERNIE-4.5-0.3B language model to enable accurate element recognition. This innovative model efficiently supports 109 languages and excels in recognizing complex elements (e.g., text, tables, formulas, and charts), while maintaining minimal resource consumption. Through comprehensive evaluations on widely used public benchmarks and in-house benchmarks, PaddleOCR-VL achieves SOTA performance in both page-level document parsing and element-level recognition. It significantly outperforms existing solutions, exhibits strong competitiveness against top-tier VLMs, and delivers fast inference speeds. These strengths make it highly suitable for practical deployment in real-world scenarios.

While PaddleOCR-VL-0.9B excels in common scenarios, its performance often faces limitations in many specific or complex business applications. For instance:

- Domain-Specific Applications
    - Finance & Accounting: Recognizing documents such as invoices, receipts, bank statements, and financial reports
    - Healthcare: Processing medical records, lab reports, handwritten prescriptions, and pharmaceutical instructions
    - Legal Sector: Identifying text in contracts, legal instruments, court filings, and certificates.
- Non-Standard Text and Typography
    - Handwriting Recognition: Deciphering handwritten forms, notes, letters, and questionnaires.
    - Stylized & Artistic Fonts: Recognizing text on posters, billboards, product packaging, and menus.
    - Historical & Archival Documents: Processing ancient manuscripts, old newspapers, and historical archives.
- Task-Specific Structured Output
    - Table Recognition & Structuring: Converting tables within images into structured formats like Excel, CSV, or JSON.
    - Mathematical Formula Recognition: Identifying mathematical equations in textbooks or research papers and exporting them into formats like LaTeX.

This is where SFT (Supervised Fine-Tuning) becomes necessary to enhance the model’s accuracy and robustness for these specialized tasks.


## 2. Environment Setup

Please ensure that you install ERNIE and its related dependencies in an environment with CUDA 12 or a later version. To avoid potential environment issues, we recommend building a container based on the official PaddlePaddle image.

### 2.1. Build the Container

The image already includes the PaddlePaddle framework, so no additional installation is required.

```bash
docker run --gpus all --name erniekit-ft-paddleocr-vl -v $PWD:/paddle --shm-size=128g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.2.0-gpu-cuda12.6-cudnn9.5 /bin/bash
```

### 2.2. Install ERNIEKit

Clone ERNIEKit and install dependencies:

```bash
git clone https://github.com/PaddlePaddle/ERNIE
cd ERNIE
python -m pip install -r requirements/gpu/requirements.txt
python -m pip install -e .
python -m pip install tensorboard
python -m pip install opencv-python-headless
python -m pip install numpy==1.26.4
```

For more installation methods, please refer to the [ERNIEKit Installation Guide]((./erniekit.md#2-installation)).

## 3. Model and Dataset Preparation

### 3.1. Model Preparation
The PaddleOCR-VL-0.9B model can be downloaded from [huggingface](https://huggingface.co/PaddlePaddle/PaddleOCR-VL/tree/main/PaddleOCR-VL-0.9B) or [modelscope](https://modelscope.cn/models/PaddlePaddle/PaddleOCR-VL/files).

```bash
huggingface-cli download PaddlePaddle/PaddleOCR-VL --local-dir PaddlePaddle/PaddleOCR-VL
```

### 3.2. Dataset Preparation

For the training dataset format, please refer to [SFT VL Dataset Format]((./datasets.md#sft-vl-dataset)). Required fields are as follows:
* `text_info`: The list of text data, each element contains a `text` and a `tag`
  * `text`: The text content from User question or System response
  * `tag`: The mask tag (`no_mask`=include in training, `mask`=exclude)
* `image_info`: The list of image data, each element contains a `image_url` and a `matched_text_index`
  * `image_url`: The url to download image online or the path to access image locally
  * `matched_text_index`: The index of matched text in `text_info`
    * Default: `matched_text_index=0` means the image is matched with the first text, and will be palced before the first text

Notes:
* Each training sample is in JSON format, with multiple samples separated by newlines
* Please ensure that `mask` items and `no_mask` items alternate in the `text_info`

For your convenience, we also provide a quick-start [Bengali training dataset]((https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-train_Bengali.jsonl)) for fine-tuning PaddleOCR-VL-0.9B on Bengali recognition. Download it using the following command:

```bash
wget https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-train_Bengali.jsonl
```

Bengali training example:

<p align="center">
  <img src="./assets/bengali_train_example.png" width="400px"></a>
</p>


```json
{
    "image_info": [
        {"matched_text_index": 0, "image_url": "./assets/table_example.jps"},
    ],
    "text_info": [
        {"text": "OCR:", "tag": "mask"},
        {"text": "দডর মথ বধ বকসট একনজর দখই চনত পরল তর অনমন\nঠক পনতই লকয রখছ\nর নচ থকই চচয বলল কশর, “এইই; পযছ! পযছ!'\nওপর", "tag": "no_mask"},
    ]
}
```

Tables, formulas, and charts use a special data format. For details, please refer to [8.1. Table/Formula/Chart Data Format](#81-tableformulachart-data-format)

## 4. Training Configuration

We provide a [configuration](../examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml) file for the Bengali sample dataset. The key training hyperparameters are as follows:

- `max_steps=926`: Total number of training steps, approximately `(D × E) / (G × B × A)`.
    - `D`: Number of training samples in the dataset.
    - `E`: Number of training epochs.
    - `G`: Number of GPUs used for training.
    - `B`: Batch size per GPU per step.
    - `A`: Number of gradient accumulation steps.
- `warmup_steps=10`: Number of linear warmup steps. It is recommended to set this to 1% of max_steps (0.01 × max_steps).
- `packing_size=8`: Number of samples packed into a sequence. Its effect is functionally equivalent to batch_size.
- `max_seq_len=16384`: The maximum sequence length. It’s recommended to set this to the largest value that your GPU memory can accommodate during training.
- `gradient_accumulation_steps=8`: Number of gradient accumulation steps.
    - Model parameters are updated once every `gradient_accumulation_steps`.
    - When GPU memory is insufficient, you can decrease `packing_size` and increase `gradient_accumulation_steps`.
    - This is a time-for-space tradeoff: it reduces GPU memory usage but extends training time.
- `learning_rate=5e-6`: Learning rate, which determines the magnitude of each parameter update.

## 5. SFT Training

Start the training using the following command:

```bash
CUDA_VISIBLE_DEVICES=0 \
erniekit train examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml \
        model_name_or_path=PaddlePaddle/PaddleOCR-VL \
        train_dataset_path=./ocr_vl_sft-train_Bengali.jsonl \
```

The training takes approximately 2 hours on a single A800-80G GPU.

By default, ERNIEKit uses all available GPUs on the machine. You can specify which GPUs ERNIEKit can use with the `CUDA_VISIBLE_DEVICES` environment variable.

The number of GPUs `GPU_num` affects the configuration of training hyperparameters like `learning_rate`, `packing_size`, and `gradient_accumulation_steps`. Theoretically, the number of samples used per update step, `sample_num = G*B*A`, has an approximately linear relationship with the `learning_rate`. Therefore, when the number of GPUs increases by a factor of `N` (to `N*GPU`), there are two adjustment methods:

- Keep sample_num constant:
    - Decrease `packing_size` by a factor of `x` to `packing_size/x`.
    - Decrease `gradient_accumulation_steps` by a factor of `y` to `gradient_accumulation_steps/y`.
    - Where `x * y = N`.
- Increase `learning_rate` by a factor of `N` to `N*learning_rate`.

You can visualize the training process using `tensorboard`. Launch it with the following command (the command below sets the `port` to 8084; please adjust it to an available port as needed):

```bash
tensorboard --logdir /PaddleOCR-VL-SFT-Bengali/tensorboard_logs/ --port 8084
```

After the service starts successfully, you can view the training logs by entering `ip:port` in your browser (You can find the machine’s IP address using the `hostname -i` command).

Loss curve as follows:

![SFT-loss](./assets/PaddleOCR-Bengali-SFT-Loss.png)

## 6. Output Directory Structure
After training, the model will be saved in the path specified by `output_dir=./PaddleOCR-VL-SFT-Bengali`. The directory contains:

- preprocessor_config.json: Image preprocessing configuration file.
- config.json: Model configuration file.
- model-00001-of-00001.safetensors: Model weights file.
    - The format of the saved model can be controlled by `save_to_hf`, defaulting to the Hugging Face safetensors format.
- model.safetensors.index.json & static_name_to_dyg_name.json: Model weight index files, etc., used to assist in sharding and loading the model across multiple GPUs.
- tokenizer.model & tokenizer_config.json & special_tokens_map.json & added_tokens.json: Tokenizer files.
- train_args.bin: Training arguments file, which records the parameters used for training.
- train_state.json: Training state file, which records the training step and best metrics.
- train_results.json & all_results.json: Training results files, which record training progress, duration, time per step, time per sample, etc.
- generation.json: Generation configuration file.
- checkpoint-[save_steps\*n]: Checkpoint folders. Saves the training state at multiples of `save_steps`. In addition to the files above, it also saves master-weight, optimizer-state, scheduler-state, etc., which can be used to resume training after an interruption.


## 7. Inference

### 7.1. Inference Environment Setup

Install PaddleOCR for inference:

```bash
python -m pip install -U "paddleocr[doc-parser]"
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
python -m pip install --force-reinstall opencv-python-headless
python -m pip install numpy==1.26.4
```

### 7.2. Inference Model Preparation
Copy the necessary inference configuration files from the original PaddleOCR-VL model to the directory where the SFT-trained model is saved:

```
cp PaddlePaddle/PaddleOCR-VL/chat_template.jinja PaddleOCR-VL-SFT-Bengali
cp PaddlePaddle/PaddleOCR-VL/inference.yml PaddleOCR-VL-SFT-Bengali
```

### 7.3. Inference Dataset Preparation
We provide a [Bengali test dataset]((https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-test_Bengali.jsonl)) that can be used for inference to observe the fine-tuning results. Download it using the following command:

```bash
wget https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-test_Bengali.jsonl
```

### 7.4. Single-Sample Inference

Bengali test image：
<p align="center">
  <img src="./assets/bengali_test_example.png" width="400px"></a>
</p>

Use the following command for single-sample inference:

```bash
paddleocr doc_parser -i https://paddle-model-ecology.bj.bcebos.com/PPOCRVL/dataset/bengali_sft/5b/7a/5b7a5c1c-207a-4924-b5f3-82890dc7b94a.png \
    --vl_rec_model_name "PaddleOCR-VL-0.9B" \
    --vl_rec_model_dir "./PaddleOCR-VL-SFT-Bengali" \
    --save_path="./PaddleOCR-VL-SFT-Bengali_response"

# GT = নট চলল রফযনর পঠ সওযর\nহয গলয গলয ভব এখন দটত, মঝ মঝ খবর নয যদও লগ যয\nঝগড\nদরগর কছ চল এল
# Excepted Answer = নট চলল রফযনর পঠ সওযর\nহয গলয গলয ভব এখন দটত, মঝ মঝ খবর নয যদও লগ যয\nঝগড\nদরগর কছ চল এল
```

The above command will save the results and visualization images in the PaddleOCR-VL-SFT-Bengali_response directory, where the prediction results are stored in files with the `.md` extension. For more information on the inference capabilities of the paddleocr tool, please refer to: https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html.

## 8. Notes

### 8.1. Table/Formula/Chart Data Format

In particular, the following formats are used for specific data types:

Table Data: OTSL format

<p align="center">
  <img src="./assets/table_example.png" width="400px"></a>
</p>

```json
{
    "image_info": [
        {"matched_text_index": 0, "image_url": "./assets/table_example.jps"},
    ],
    "text_info": [
        {"text": "Table Recognition:", "tag": "mask"},
        {"text": "<fcel>分组<fcel>频数<fcel>频率<nl><fcel>[41,51)<fcel>2<fcel>\\( \\frac{2}{30} \\)<nl><fcel>[51,61)<fcel>1<fcel>\\( \\frac{1}{30} \\)<nl><fcel>[61,71)<fcel>4<fcel>\\( \\frac{4}{30} \\)<nl><fcel>[71,81)<fcel>6<fcel>\\( \\frac{6}{30} \\)<nl><fcel>[81,91)<fcel>10<fcel>\\( \\frac{10}{30} \\)<nl><fcel>[91,101)<fcel>5<fcel>\\( \\frac{5}{30} \\)<nl><fcel>[101,111)<fcel>2<fcel>\\( \\frac{2}{30} \\)<nl>", "tag": "no_mask"},
    ]
}
```

Formula Data: LaTeX format

<p align="center">
  <img src="./assets/formula_example.jpg" width="200px"></a>
</p>

```json
{
    "image_info": [
        {"matched_text_index": 0, "image_url": "./assets/formula_example.jps"},
    ],
    "text_info": [
        {"text": "Formula Recognition:", "tag": "mask"},
        {"text": "\\[t_{n}\\in[0,\\infty]\\]", "tag": "no_mask"},
    ]
}
```

Chart Data: Markdown format

<p align="center">
  <img src="./assets/chart_example.png" width="400px"></a>
</p>

```json
{
    "image_info": [
        {"matched_text_index": 0, "image_url": "./assets/chart_example.png"},
    ],
    "text_info": [
        {"text": "Chart Recognition:", "tag": "mask"},
        {"text": "  | 22Q3 | 22Q3yoy\n电商 | 85 | 100%\n川渝 | 140 | 8%\n云贵陕 | 95 | 12%\n外围地区 | 45 | 20%", "tag": "no_mask"},
    ]
}
```

### Common Issues

If you encounter the following problem while using the above command, it is generally due to a conflict between cv2 and the environment. This can be resolved by installing `opencv-python-headless`.

**Error message**

```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python3.10/dist-packages/cv2/__init__.py", line 181, in <module>
    bootstrap()
  File "/usr/local/lib/python3.10/dist-packages/cv2/__init__.py", line 153, in bootstrap
    native_module = importlib.import_module("cv2")
  File "/usr/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

**Solution**

```shell
python -m pip install --force-reinstall opencv-python-headless
python -m pip install numpy==1.26.4
```
