English | [简体中文](./paddleocr_vl_sft_zh.md)

# PaddleOCR-VL-0.9B SFT

## Introduction

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


## SFT Training

- Download the PaddleOCR-VL-0.9B model from [huggingface](https://huggingface.co/PaddlePaddle/PaddleOCR-VL/tree/main/PaddleOCR-VL-0.9B) or [modelscope](https://modelscope.cn/models/PaddlePaddle/PaddleOCR-VL/files). 
- Download the [Bengali language train dataset](https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-train_Bengali.jsonl), or construct your own dataset following the [SFT VL Dataset Foramt](./datasets.md#sft-vl-dataset).
- Training with CLI and YAML configuration:
```
erniekit train examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml \
        model_name_or_path=/PATH/TO/MODEL \
        train_dataset_path=/PAHT/TO/DATASET \
```
- Download the [Bengali language test dataset](https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-test_Bengali.jsonl) and inference with [PaddleX](https://github.com/PaddlePaddle/PaddleX).

### Hyper Parameters

- `max_steps`: the number of steps to train, approximately obtained by `(D × E) / (G × B × A)` .
    - `D`: the number of training samples in dataset.
    - `E`: the number of training epochs.
    - `G`: the number of GPUs for data parallel.
    - `B`: the batch size (packing size) per step per GPU.
    - `A`: the number of graident accumulation steps.
- `warmup_steps`: the number of steps for linear warmup, recommended to be `0.01 × max_steps`.
- `packing_size`: the number of samples packed into one sequence.
- `padding`: set `False` to avoid padding the sequence to the max sequence length.
- `max_seq_len`: make sure it is greater than the max sequence length of the input sample.

