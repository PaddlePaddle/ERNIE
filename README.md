<!-- filepath: [README.md](http://_vscodecontentref_/0) -->


<p align="center">
  <img src="https://github.com/user-attachments/assets/9ad1ffce-2310-4f80-a3cd-7a117bfb4f17" width="300px"></a>
</p>

<div align="center">

[ERNIE Bot](https://ernie.baidu.com/) | [AI Studio](https://aistudio.baidu.com/modelsoverview) | [Hugging Face](https://huggingface.co/baidu)

📑 [Blog](https://yiyan.baidu.com/blog/posts/ernie4.5) | 📚 [Cookbook](./cookbook/) | 📑 [Paper](https://yiyan.baidu.com/blog/publication/)

</div>

## Introduction to ERNIE 4.5

We introduce ERNIE 4.5, a new family of large-scale multimodal models comprising 10 distinct variants. The model family consist of Mixture-of-Experts (MoE) models with 47B and 3B active parameters, with the largest model having 424B total parameters, as well as a 0.3B dense model. For the MoE architecture, we propose a novel heterogeneous modality structure, which supports parameter sharing across modalities while also allowing dedicated parameters for each individual modality.  This MoE architecture has the advantage to enhance multimodal understanding without compromising, and even improving, performance on text-related tasks. All of our models are trained with optimal efficiency using the [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) deep learning framework, which also enables high-performance inference and streamlined deployment for them. We achieve 47% Model FLOPs Utilization (MFU) in our largest ERNIE 4.5 language model pre-training. Experimental results show that our models achieve state-of-the-art performance across multiple text and multimodal benchmarks, especially in instruction following, world knowledge memorization, visual understanding and multimodal reasoning. All models are publicly accessible under Apache 2.0 to support future research and development in the field. Additionally, we open source the development toolkits for ERNIE 4.5, featuring industrial-grade capabilities, resource-efficient training and inference workflows, and multi-hardware compatibility.

</br>

<div align="center">

 **ERNIE 4.5**
<table style="table-layout: auto; border-collapse: collapse; border: 1px solid #ddd; text-align: center;">
  <thead class="ant-table-thead">
    <tr>
      <th colspan="2" style="border: 1px solid #ddd;text-align: center;background: lightgray;vertical-align: middle;color:black" >ERNIE 4.5 Models </th>
      <th colspan="3" style="border: 1px solid #ddd;text-align: center;background: lightgray;vertical-align: middle;color:black">Model Information</th>
    </tr>
    <tr>
      <th style="border: 1px solid #ddd;width: 100px;text-align: center;background: lightgray;vertical-align: middle;color:black">Model Category</th>
      <th style="border: 1px solid #ddd;width: 250px;text-align: center;background: lightgray;vertical-align: middle;color:black">Model</th>
      <th style="border: 1px solid #ddd; width: 100px;text-align: center;background: lightgray;vertical-align: middle;color:black">Input Modality</th>
      <th style="border: 1px solid #ddd; width: 100px;text-align: center;background: lightgray;vertical-align: middle;color:black">Output Modality</th>
      <th style="border: 1px solid #ddd; width: 100px;text-align: center;background: lightgray;vertical-align: middle;color:black">Context Window

</th>
    </tr>
  </thead>
  <tbody class="ant-table-tbody">
    <tr>
      <td rowspan="4" style="border: 1px solid #ddd;vertical-align: middle;">Large Language Model (LLMs)</td>
      <td style="border: 1px solid #ddd;">ERNIE-4.5-300B-A47B-Base</td>
      <td rowspan="4"style="border: 1px solid #ddd;">Text</td>
      <td rowspan="4"style="border: 1px solid #ddd;">Text</td>
      <td rowspan="10" style="border: 1px solid #ddd;">128K</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd;">ERNIE-4.5-300B-A47B</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd;">ERNIE-4.5-21B-A3B-Base</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd;">ERNIE-4.5-21B-A3B</td>
    </tr>
    <tr>
      <td rowspan="4" style="border: 1px solid #ddd;vertical-align: middle;"> Vision-Language Models (VLMs)</td>
      <td style="border: 1px solid #ddd;">ERNIE-4.5-VL-424B-A47B-Base</td>
      <td rowspan="4"style="border: 1px solid #ddd;">Text/Image/Video</td>
      <td rowspan="4"style="border: 1px solid #ddd;">Text</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd;">ERNIE-4.5-VL-424B-A47B</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd;">ERNIE-4.5-VL-28B-A3B-Base</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd;">ERNIE-4.5-VL-28B-A3B</td>
    </tr>
    <tr>
      <td rowspan="2" style="border: 1px solid #ddd;vertical-align: middle;">Dense Models</td>
      <td style="border: 1px solid #ddd;">ERNIE-4.5-0.3B-Base</td>
      <td rowspan="2"style="border: 1px solid #ddd;">Text</td>
      <td rowspan="2"style="border: 1px solid #ddd;">Text</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd;">ERNIE-4.5-0.3B</td>
    </tr>
  </tbody>
</table>
</div>

_Note:All models (including pre-trained weights and inference code) have been released on [Hugging Face](https://huggingface.co/baidu), and [AI Studio](https://aistudio.baidu.com/index). Check our [blog](https://yiyan.baidu.com/blog/posts/ernie4.5) for more details.

</br>

## Highlights

Our model family is characterized by three key innovations:

1. **Multimodal Heterogeneous MoE Pre-Training:** Our models are jointly trained on both textual and visual modalities to better capture the nuances of multimodal information and improve performance on tasks involving text understanding and generation, image understanding, and cross-modal reasoning. To achieve this without one modality hindering the learning of another, we designed a *heterogeneous MoE structure*, incorporated *modality-isolated routing*, and employed *router orthogonal loss* and *multimodal token-balanced loss*. These architectural choices ensure that both modalities are effectively represented, allowing for mutual reinforcement during training.

2. **Scaling-Efficient Infrastructure:** We propose a novel heterogeneous hybrid parallelism and hierarchical load balancing strategy for efficient training of ERNIE 4.5 models. By using intra-node expert parallelism, memory-efficient pipeline scheduling, FP8 mixed-precision training and finegrained recomputation methods, we achieve remarkable pre-training throughput. For inference, we propose *multi-expert parallel collaboration* method and *convolutional code quantization* algorithm to achieve 4-bit/2-bit lossless quantization. Furthermore, we introduce PD disaggregation with dynamic role switching for effective resource utilization to enhance inference performance for ERNIE 4.5 MoE models. Built on [PaddlePaddle](https://github.com/PaddlePaddle/Paddle), ERNIE 4.5 delivers high-performance inference across a wide range of hardware platforms.

3. **Modality-Specific Post-Training:** To meet the diverse requirements of real-world applications, we fine-tuned variants of the pre-trained model for specific modalities. Our LLMs are optimized for general-purpose language understanding and generation. The VLMs focuses on visuallanguage understanding and supports both thinking and non-thinking modes. Each model employed a combination of *Supervised Fine-tuning (SFT)*, *Direct Preference Optimization (DPO)* or a modified reinforcement learning method named *Unified Preference Optimization (UPO)* for post-training.

All released models (including pretrained weights and inference code) are now fully open source. For relationships between different model architectures, see the diagram below. Additional technical details are available in the technical report.

<div align="center">
<img src="https://github.com/user-attachments/assets/44060b7a-6dbe-442b-80e8-b399cab1ce84" width="1080" height="636" >
</div>

</br>

## Performance and Benchmark Results

ERNIE-4.5-300B-A47B-Base surpasses DeepSeek-V3-671B-A37B-Base on 22 out of 28 benchmarks, demonstrating leading performance across all major capability categories. This underscores the substantial improvements in generalization, reasoning, and knowledge-intensive tasks brought about by scaling up the ERNIE-4.5-Base model relative to other state-of-the-art large models. With a total parameter size of 21B (approximately 70% that of Qwen3-30B), ERNIE-4.5-21B-A3B-Base outperforms Qwen3-30B-A3B-Base on several math and reasoning benchmarks, including BBH and CMATH. ERNIE-4.5-21B-A3B-Base remains highly competitive given its significantly smaller model size, demonstrating notable parameter efficiency and favorable performance trade-offs.

ERNIE-4.5-300B-A47B, the post trained model, demonstrates significant strengths in instruction following and knowledge tasks, as evidenced by the state-of-the-art scores on benchmarks such as IFEval, Multi-IF, SimpleQA, and ChineseSimpleQA. The lightweight model ERNIE-4.5-21B-A3B achieves competitive performance compared to Qwen3-30B-A3B, despite having approximately 30% fewer total parameters.

In the non-thinking mode, ERNIE-4.5-VL exhibits outstanding proficiency in visual perception, document and chart understanding, and visual knowledge, performing strongly across a range of established benchmarks. Under the thinking mode, ERNIE-4.5-VL not only demonstrates enhanced reasoning abilities compared to the non-thinking mode, but also retains the strong perception capabilities of the latter. ERNIE-4.5-VL-424B-A47B delivers consistently strong results across the various multimodal evaluation benchmarks. Its thinking mode offers a distinct advantage on challenging benchmarks such as MathVista, MMMU, and VisualPuzzle, while maintaining competitive performance on perception-focused datasets like CV-Bench and RealWorldQA. The lightweight vision-language model ERNIE-4.5-28B-A3B achieves competitive or even superior performance compared to Qwen2.5-VL-7B and Qwen2.5-VL-32B across most benchmarks, despite using significantly fewer activation parameters. Notably, our lightweight model also supports both thinking and non-thinking modes, offering functionalities consistent with ERNIE-4.5-VL-424B-A47B.

### Performace of ERNIE-4.5 pre-trained models

<div align="center">
<img src="https://github.com/user-attachments/assets/a31a0101-4d47-4e76-97f6-332a0cff7098" style="max-width: 80%; height: auto;">
</div>

### Performance of post-trained model ERNIE-4.5-300B-A47B

<div align="center">
<img src="https://github.com/user-attachments/assets/2b5657ef-b61f-44c2-a945-106cf919d5bf" style="max-width: 80%; height: auto;">
</div>

### Performance of post-trained model ERNIE-4.5-21B-A3B

<div align="center">
<img src="https://github.com/user-attachments/assets/a89d9414-16ea-4afa-8888-bbc1ca51f509" style="max-width: 80%; height: auto;">
</div>

### Performance of post-trained multimodal models in thinking mode

<div align="center">
<img src="https://github.com/user-attachments/assets/d91f898d-3b5c-4628-ad3a-c1d231ae415a" style="max-width: 80%; height: auto;">
</div>

### Performance of post-trained multimodal models in non-thinking mode

<div align="center">
<img src="https://github.com/user-attachments/assets/4874712f-47b0-413a-b51c-f9536ca03d51" style="max-width: 80%; height: auto;">
</div>

</br>

## Model Development

ERNIE 4.5 models are trained and deployed for inference using the [PaddlePaddle]((https://github.com/PaddlePaddle/Paddle)) framework. The full workflow of training, compression, and inference for ERNIE 4.5 is supported through the [ERNIEKit](./docs/erniekit.md) and [FastDeploy](https://github.com/PaddlePaddle/FastDeploy) toolkit. The table below details the feature matrix of the ERNIE 4.5 model family for training and inference.
<div align="center">

| Model             |  Training      |  Inference              |
| ------------------------------ | ------------------------- | -------------------------------- |
| ERNIE-4.5-300B-A47B-Base       | SFT/SFT-LoRA/DPO/DPO-LoRA | BF16 / W4A16C16 / W8A16C16 / FP8 |
| ERNIE-4.5-300B-A47B            | SFT/SFT-LoRA/DPO/DPO-LoRA/QAT | BF16 / W4A16C16 / W8A16C16 / W4A8C8 / FP8  / 2Bits |
| ERNIE-4.5-21B-A3B-Base         | SFT/SFT-LoRA/DPO/DPO-LoRA | BF16 / W4A16C16 / W8A16C16 / FP8 |
| ERNIE-4.5-21B-A3B              | SFT/SFT-LoRA/DPO/DPO-LoRA | BF16 / W4A16C16 / W8A16C16 / FP8 |
| ERNIE-4.5-VL-424B-A47B-Base    | Coming Soon               | BF16 / W4A16C16 / W8A16C16 / FP8 |
| ERNIE-4.5-VL-424B-A47B         | Coming Soon               | BF16 / W4A16C16 / W8A16C16 / FP8 |
| ERNIE-4.5-VL-28B-A3B-Base      | Coming Soon               | BF16 / W4A16C16 / W8A16C16 / FP8 |
| ERNIE-4.5-VL-28B-A3B           | Coming Soon               | BF16 / W4A16C16 / W8A16C16 / FP8 |
| ERNIE-4.5-0.3B-Base            | SFT/SFT-LoRA/DPO/DPO-LoRA | BF16 / W8A16C16 / FP8            |
| ERNIE-4.5-0.3B                 | SFT/SFT-LoRA/DPO/DPO-LoRA | BF16 / W8A16C16 / FP8            |

</div>

_Note: For different ERNIE 4.5 model, we provide diverse quantization schemes using the notation WxAxCx, where: W indicates weight precision, A indicates activation precision, C indicates KV Cache precision, x represents numerical precision._


### ERNIEKit: ERNIE Development Toolkit Based on PaddlePaddle

**ERNIEKit** is an industrial-grade training and compression development toolkit for ERNIE models based on PaddlePaddle, offering full-cycle development support for the ERNIE 4.5 model family. Key capabilities include:
* High-performance pre-training implementation
* Full-parameter supervised fine-tuning (SFT)
* Direct Preference Optimization (DPO)
* Parameter-efficient fine-tuning and alignment (SFT-LoRA/DPO-LoRA)
* Quantization-Aware Training (QAT)
* Post-Training Quantization (PTQ) [WIP]

Minimum hardware requirements for training each model are documented [here](./docs/erniekit.md).


#### Quick Start

When you install ERNIEKit successfully, you can start training ERNIE 4.5 models with the following command:

```bash
# download model from huggingface
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# 8K Sequence Length, SFT
erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_sft_8k.yaml
```

For detailed guides on installation, CLI usage, WebUI, multi-node training, and advanced features, please refer to [ERNIEKit Training Document](./docs/erniekit.md).

**ERNIEKit WebUI demo:**

https://github.com/user-attachments/assets/6d44cb92-0826-42df-aa80-7656445e0f73

### FastDeploy：High-performance Inference and Deployment Toolkit for LLMs and VLMs Based on PaddlePaddle

**FastDeploy** is an inference and deployment toolkit for large language models and visual language models, developed based on PaddlePaddle. It delivers production-ready, easy-to-use multi-hardware deployment solutions with multi-level load-balanced PD disaggregation, comprehensive quantization format support, OpenAI API server and vLLM compatible etc.

For installation please refer to [FastDeploy](https://github.com/PaddlePaddle/FastDeploy).

#### Offline Inference

```python
from fastdeploy import LLM, SamplingParams

prompt = "Write me a poem about large language model."
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="baidu/ERNIE-4.5-0.3B-Paddle", max_model_len=32768)

outputs = llm.generate(prompt, sampling_params)
```

#### Online Serving

```bash
python -m fastdeploy.entrypoints.openai.api_server \
    --model "baidu/ERNIE-4.5-0.3B-Paddle" \
    --max-model-len 32768 \
    --port 9904
```

For more inference and deployment guides, please refer to [FastDeploy](https://github.com/PaddlePaddle/FastDeploy).

</br>

## Cookbooks

Discover best-practice guides showcasing ERNIE’s capabilities across multiple domains:

<div align="center">

| Cookbook | Description | Gradio Demo |
| --- | --- | --- |
| [Conversation](/cookbook/notebook/conversation_demo_en.ipynb) | Building conversational applications.  | [conversation_demo.py](/cookbook/conversation_demo.py) |
| [Simple ERNIE Bot](/cookbook/notebook/simple_ernie_bot_demo_en.ipynb) | Creating a lightweight web-based ERNIE Bot.   |[simple_ernie_bot_demo.py](/cookbook/simple_ernie_bot_demo.py) |
| [Web-Search-Enhanced Conversation](/cookbook/notebook/web_search_demo_en.ipynb) | Building conversational apps with integrated web search. | [web_search_demo.py](/cookbook/web_search_demo.py) |
| [Knowledge Retrieval-based Q&A](/cookbook/notebook/knowledge_retrieval_demo_en.ipynb) | Building intelligent Q&A systems with private knowledge bases. | [knowledge_retrieval_demo.py](/cookbook/knowledge_retrieval_demo.py) |
| [Advanced Search](/cookbook/notebook/advanced_search_demo_en.ipynb)    | Building article-generation applications using deep information extraction. | [advanced_search_demo.py](/cookbook/advanced_search_demo.py) |
| [SFT tutorial](/cookbook/notebook/sft_tutorial_en.ipynb) | Optimizing task performance through supervised fine-tuning with ERNIEKit. | - |
| [DPO tutorial](/cookbook/notebook/dpo_tutorial_en.ipynb) | Aligning models with human preferences using ERNIEKit. | - |
| [Text Recognition](/cookbook/notebook/text_recognition_tutorial_en.ipynb) | A Comprehensive Guide to Developing Text Recognition for Non-Chinese and Non-English Languages Using ERNIE and PaddleOCR. | - |
| [Document Translation](/cookbook/notebook/document_translation_tutorial_en.ipynb)          |  Document Translation Practice Based on ERNIE and PaddleOCR. | - |
| [Key Information Extraction](/cookbook/notebook/key_information_extraction_tutorial_en.ipynb) |  Key Information Extraction in Contract Scenarios Based on ERNIE and PaddleOCR. | - |

</div>

</br>

## License

The ERNIE 4.5 models are provided under the Apache License 2.0. This license permits commercial use, subject to its terms and conditions.
</br>

## Citation

If you find ERNIE 4.5 useful or wish to use it in your projects, please kindly cite our technical report:

```bibtex
@misc{ernie2025technicalreport,
      title={ERNIE 4.5 Technical Report},
      author={Baidu ERNIE Team},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={}
}
```
