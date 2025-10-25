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

Please refer to the installation steps in the [ERNIEKit Installation Guide]((./erniekit.md#2-installation)) to set up the training environment.

## 3. Model and Dataset Preparation

### 3.1 Model Preparation
The PaddleOCR-VL-0.9B model can be downloaded from [huggingface](https://huggingface.co/PaddlePaddle/PaddleOCR-VL/tree/main/PaddleOCR-VL-0.9B) or [modelscope](https://modelscope.cn/models/PaddlePaddle/PaddleOCR-VL/files).

```
huggingface-cli download PaddlePaddle/PaddleOCR-VL --local-dir PaddlePaddle/PaddleOCR-VL
```

### 3.2 Dataset Preparation

You can build your fine-tuning dataset according to the [SFT VL Dataset Format]((./datasets.md#sft-vl-dataset)). For your convenience, we also provide a quick-start [Bengali training dataset]((https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-train_Bengali.jsonl)) for fine-tuning PaddleOCR-VL-0.9B on Bengali recognition. Download it using the following command:

```
wget https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-train_Bengali.jsonl
```

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

```
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

```
pip install tensorboard
tensorboard --logdir /PaddleOCR-VL-SFT-Bengali/tensorboard_logs/ --port 8084
```

After the service starts successfully, you can view the training logs by entering `ip:port` in your browser (You can find the machine’s IP address using the `hostname -i` command).

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

### 7.1 Inference Environment Setup

Install PaddleX for inference:

```
python -m pip install paddlex
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
```

### 7.2 Inference Model Preparation
Copy the necessary inference configuration files from the original PaddleOCR-VL model to the directory where the SFT-trained model is saved:

```
cp PaddlePaddle/PaddleOCR-VL/chat-template.jinja PaddleOCR-VL-SFT-Bengali
cp PaddlePaddle/PaddleOCR-VL/inference.yaml PaddleOCR-VL-SFT-Bengali
```

### 7.3 Inference Dataset Preparation
We provide a [Bengali test dataset]((https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-test_Bengali.jsonl)) that can be used for inference to observe the fine-tuning results. Download it using the following command:

```
wget https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-test_Bengali.jsonl
```

### 7.4 Single-Sample Inference

Execute the following Python code to load the model and perform inference on a single sample:

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

### 7.5 Dataset Inference

Execute the following Python code to load the model and perform inference on the test dataset:

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