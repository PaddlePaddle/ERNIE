# ERNIEKit Data Format Specification

ERNIEKit currently supports reading local datasets and downloading specified Hugging Face datasets in two formats: erniekit and alpaca.

## Local Datasets

- **CLI**: Modify the following fields in the YAML config file:
  - Set `train_dataset_path`/`eval_dataset_path` to the absolute or relative path of your local dataset file
  - Set `train_dataset_type`/`eval_dataset_type` to the dataset format (erniekit/alpaca)
  - Set `train_dataset_prob`/`eval_dataset_prob` for multi-source dataset mixing probabilities
```yaml
# single-source
train_dataset_type: "erniekit"
train_dataset_path: "./examples/data/sft-train.jsonl"
train_dataset_prob: "1.0"

# multi-source
train_dataset_type: "erniekit,erniekit"
train_dataset_path: "./examples/data/sft-train1.jsonl,./examples/data/sft-train2.jsonl"
train_dataset_prob: "0.8,0.2"
```

- **WebUI**:
  - Under `Set Custom Dataset`, input the local file path in `Dataset Path`
  - Select the corresponding format (erniekit/alpaca) in `Optional Data Type`

## Hugging Face Datasets

- **CLI**: Modify the following fields in the YAML config file:
  - Set `train_dataset_path`/`eval_dataset_path` to the Hugging Face repo ID
  - Set `train_dataset_type`/`eval_dataset_type` to alpaca
  - Set `train_dataset_prob`/`eval_dataset_prob` for multi-source dataset mixing probabilities
```yaml
# single-source
train_dataset_type: "alpaca"
train_dataset_path: "BelleGroup/train_2M_CN"
train_dataset_prob: "1.0"

# multi-source
train_dataset_type: "alpaca,alpaca"
train_dataset_path: "llamafactory/alpaca_gpt4_zh,BelleGroup/train_2M_CN"
train_dataset_prob: "0.8,0.2"
```
- **WebUI**:
  - Under `Set Built-in Dataset`, select the dataset name in `Dataset Selection`
  - The system will automatically configure the path and type, then download and read from Hugging Face

Supported Hugging Face datasets are defined in `ernie.dataset.hf.data_info.json`:

### Supported Hugging Face Datasets
| Dataset Name | Format | File | File Format |
|--------------|--------|------|-------------|
| [llamafactory/alpaca_en](https://huggingface.co/datasets/llamafactory/alpaca_en) | alpaca | alpaca_data_en_52k.json | json |
| [llamafactory/alpaca_zh](https://huggingface.co/datasets/llamafactory/alpaca_zh) | alpaca | alpaca_data_zh_51k.json | json |
| [llamafactory/alpaca_gpt4_en](https://huggingface.co/datasets/llamafactory/alpaca_gpt4_en) | alpaca | alpaca_gpt4_data_en.json | json |
| [llamafactory/alpaca_gpt4_zh](https://huggingface.co/datasets/llamafactory/alpaca_gpt4_zh) | alpaca | alpaca_gpt4_data_zh.json | json |
| [BelleGroup/train_2M_CN](https://huggingface.co/datasets/BelleGroup/train_2M_CN) | alpaca | train_2M_CN.json | jsonl |
| [BelleGroup/train_1M_CN](https://huggingface.co/datasets/BelleGroup/train_1M_CN) | alpaca | Belle_open_source_1M.json | jsonl |
| [BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) | alpaca | Belle_open_source_0.5M.json | jsonl |
| [BelleGroup/generated_chat_0.4M](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M) | alpaca | generated_chat_0.4M.json | jsonl |
| [BelleGroup/school_math_0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M) | alpaca | school_math_0.25M.json | jsonl |
| [sahil2801/CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) | alpaca | code_alpaca_20k.json | json |
| [TIGER-Lab/MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) | alpaca | MathInstruct.json | json |
| [YeungNLP/firefly-train-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) | alpaca | firefly-train-1.1M.jsonl | jsonl |
| [suolyer/webqa](https://huggingface.co/datasets/suolyer/webqa) | alpaca | train.json | jsonl |
| [zxbsmk/webnovel_cn](https://huggingface.co/datasets/zxbsmk/webnovel_cn) | alpaca | novel_cn_token512_50k.json | json |
| [AstraMindAI/SFT-Nectar](https://huggingface.co/datasets/AstraMindAI/SFT-Nectar) | alpaca | sft_data_structured.json | json |
| [hfl/stem_zh_instruction](https://huggingface.co/datasets/hfl/stem_zh_instruction) | alpaca | bio_50282.json | jsonl |
| [llamafactory/OpenO1-SFT](https://huggingface.co/datasets/llamafactory/OpenO1-SFT) | alpaca | OpenO1-SFT-Pro.jsonl | jsonl |
| [Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT) | alpaca | distill_r1_110k_sft.jsonl | jsonl |
| [mayflowergmbh/oasst_de](https://huggingface.co/datasets/mayflowergmbh/oasst_de) | alpaca | oasst_de.json | json |
| [mayflowergmbh/dolly-15k_de](https://huggingface.co/datasets/mayflowergmbh/dolly-15k_de) | alpaca | dolly_de.json | json |
| [mayflowergmbh/alpaca-gpt4_de](https://huggingface.co/datasets/mayflowergmbh/alpaca-gpt4_de) | alpaca | alpaca_gpt4_data_de.json | json |
| [mayflowergmbh/openschnabeltier_de](https://huggingface.co/datasets/mayflowergmbh/openschnabeltier_de) | alpaca | openschnabeltier.json | json |
| [mayflowergmbh/evol-instruct_de](https://huggingface.co/datasets/mayflowergmbh/evol-instruct_de) | alpaca | evol_instruct_de.json | json |
| [mayflowergmbh/dolphin_de](https://huggingface.co/datasets/mayflowergmbh/dolphin_de) | alpaca | dolphin.json | json |
| [mayflowergmbh/booksum_de](https://huggingface.co/datasets/mayflowergmbh/booksum_de) | alpaca | booksum.json | json |
| [mayflowergmbh/airoboros-3.0_de](https://huggingface.co/datasets/mayflowergmbh/airoboros-3.0_de) | alpaca | airoboros_3.json | json |
| [mayflowergmbh/ultra-chat_de](https://huggingface.co/datasets/mayflowergmbh/ultra-chat_de) | alpaca | ultra_chat_german.json | json |
| [pleisto/wikipedia-cn-20230720-filtered](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered) | alpaca | wikipedia-cn-20230720-filtered.json | json |

## erniekit Data Format

### SFT Dataset

We provide [demo](../examples/data/) data for quick testing. You can either use these samples or train with your own data.

Required fields for SFT:

* `system(optional)`: System configuration
* `src`: User conversation content
* `tgt`: System response content
* `label(optional)`: Training flag (1=include in training, 0=exclude)

Notes:
* `src` and `tgt` are List objects supporting multi-turn conversations
* Each training sample is in JSON format, with multiple samples separated by newlines
```json
{
    "system": "你是一个生活小助理",
    "src": [
        "我们如何在日常生活中减少用水？",
        "还有别的建议吗？"
    ],
    "tgt": [
        "1. 使用节水装置，如节水淋浴喷头和水龙头。",
        "2. 使用水箱或水桶收集家庭废水，例如洗碗和洗浴。 \n3. 在社区中提高节水意识。 \n4. 检查水管和灌溉系统的漏水情况，并及时修复它们。 \n5. 洗澡时间缩短，使用低流量淋浴头节约用水。 \n6. 收集雨水，用于园艺或其他非饮用目的。 \n7. 刷牙或擦手时关掉水龙头。 \n8. 减少浇水草坪的时间。 \n9. 尽可能多地重复使用灰水（来自洗衣机、浴室水槽和淋浴的水）。 \n10. 只购买能源效率高的洗碗机和洗衣机。"
    ],
    "label": [0, 1],
}
```

### DPO Dataset

We provide [demo](../examples/data/) data for quick testing. You can either use these samples or train with your own data.

Required fields for DPO:

* `system(optional)`: System configuration
* `src`: User conversation content (first item=question1, second=question2, etc.)
* `tgt`: System response content (one fewer item than src)
* `response`: Contains chosen/rejected responses (must contain odd number of strings)
* `sort`: Differentiates chosen/rejected (lower value=rejected, higher=chosen)
* Each training sample is in JSON format, with multiple samples separated by newlines
```json
{
    "system": "你是一个生活小助理",
    "src": [
        "你好。",
        "哪一个富含蛋白质，床还是墙？"
    ],
    "tgt": ["你好呀，我是你的生活小助理。"],
    "response": [
        [
            "床和墙都不是蛋白质的来源，因为它们都是无生命的物体。蛋白质通常存在于肉类、奶制品、豆类和坚果等食物中。"
        ],
        [
            "对不起，我无法回答那个问题。请提供更具体的信息，让我知道你需要什么帮助。"
        ]
    ],
    "sort": [
        1,
        0
    ]
}
```

## alpaca Format

### SFT Dataset

Supports json and jsonl file formats:

* **json**: Each line contains one JSON object:
```json
{"instruction":"instructionA", "input":"inputA", "output":"outputA"}
{"instruction":"instructionB", "input":"inputB", "output":"outputB"}
{"instruction":"instructionC", "input":"inputC", "output":"outputC"}
```

* **jsonl**: All data in a single JSON array:
```json
[
    {"instruction":"instructionA", "input":"inputA", "output":"outputA"},
    {"instruction":"instructionB", "input":"inputB", "output":"outputB"},
    {"instruction":"instructionC", "input":"inputC", "output":"outputC"}
]
```

**Field Mapping Between alpaca and erniekit**

| alpaca | erniekit | Mapping |
|--------|----------|---------|
| instruction <br> input | src | src[-1] = instruction + input |
| output | tgt | tgt[-1] = output |
| history | src <br> tgt | history = zip(src[:-1], tgt[:-1]) |
| system | system | system=system |

### DPO Dataset

(Coming soon)
