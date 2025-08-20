# ERNIEKit Data Packing Strategy

`Packing` is a technique used to optimize batch processing by combining multiple short input sequences into a single longer sequence before feeding them into the LLM. This reduces padding overhead and improves hardware utilization (e.g., GPU/TPU efficiency).

`The greedy intokens strategy` is a token-level optimization that prioritizes filling the available token budget (e.g., max sequence length) in a greedy manner during batch processing. It ensures that the model generates as many tokens as possible within the constraints, minimizing wasted capacity.

| packing      | greedy_intokens | Packing Strategy |
|--------------|-----------------|------------------|
| false | any   | No packing  |
| true  | false | packing is enabled without greedy intokens strategy |
| true  | true  | greedy intokens packing is enabled |

# ERNIEKit Data Sampling Strategy

Currently, four data sampling strategies are supported: `random`, `concat`, `interleave_under`, `interleave_over`

| Data Sampling Strategy | Applicable Scenarios	| Limitations | Description |
|------------------|-----------------|------------------|------------------|
| `random`           | The dataset is extremely large and strict data proportioning is required | max_steps > 0 | In `random` mode, based on the input dataset probs, a fixed-size sample pool of `num_samples_each_epoch` is constructed, and the data loader randomly acquires data from this sample pool. |
| `concat`           | Need to train all data in the datasets | None | In `concat` mode, the input dataset probs are not used. Instead, multiple datasets are directly concatenated. The size of the dataset is equal to the total size of the input multi-source datasets. When max_steps = -1, setting `num_train_epochs` allows for a complete traversal of the input datasets for `num_train_epochs` rounds. |
| `interleave_under` | When small datasets are important but have limited samples | None | The `interleave` strategy involves cross-concatenating multiple datasets according to data proportioning. `interleave_under` indicates undersampling, meaning that sampling stops as soon as one of the datasets is exhausted. |
| `interleave_over`  | When small datasets are important but have limited samples | None | The `interleave` strategy involves cross-concatenating multiple datasets according to data proportioning. `interleave_over` indicates oversampling, meaning that sampling stops only after all datasets have been exhausted. |

- Note: `num_samples_each_epoch` only works in `random` data sampling strategy.

# ERNIEKit Data Format Specification

ERNIEKit currently supports reading local datasets and downloading specified Hugging Face datasets in two formats: erniekit and alpaca.

## Local Datasets

- **CLI**: Modify the following fields in the YAML config file:
  - Set `train_dataset_path` / `eval_dataset_path` to the absolute or relative path of your local dataset file
  - Set `train_dataset_type` / `eval_dataset_type` to the dataset format (erniekit/alpaca)
  - Set `train_dataset_prob` / `eval_dataset_prob` for multi-source dataset mixing probabilities
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
  - Set `train_dataset_path` / `eval_dataset_path` to the Hugging Face repo ID
  - Set `train_dataset_type` / `eval_dataset_type` to alpaca
  - Set `train_dataset_prob` / `eval_dataset_prob` for multi-source dataset mixing probabilities
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
| Dataset Name | Type |Format | File | File Format |
|--------------|------|-------|------|-------------|
| [llamafactory/alpaca_en](https://huggingface.co/datasets/llamafactory/alpaca_en) | sft | alpaca | alpaca_data_en_52k.json | json |
| [llamafactory/alpaca_zh](https://huggingface.co/datasets/llamafactory/alpaca_zh) | sft | alpaca | alpaca_data_zh_51k.json | json |
| [llamafactory/alpaca_gpt4_en](https://huggingface.co/datasets/llamafactory/alpaca_gpt4_en) | sft | alpaca | alpaca_gpt4_data_en.json | json |
| [llamafactory/alpaca_gpt4_zh](https://huggingface.co/datasets/llamafactory/alpaca_gpt4_zh) | sft | alpaca | alpaca_gpt4_data_zh.json | json |
| [BelleGroup/train_2M_CN](https://huggingface.co/datasets/BelleGroup/train_2M_CN) | sft | alpaca | train_2M_CN.json | jsonl |
| [BelleGroup/train_1M_CN](https://huggingface.co/datasets/BelleGroup/train_1M_CN) | sft | alpaca | Belle_open_source_1M.json | jsonl |
| [BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) | sft | alpaca | Belle_open_source_0.5M.json | jsonl |
| [BelleGroup/generated_chat_0.4M](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M) | sft | alpaca | generated_chat_0.4M.json | jsonl |
| [BelleGroup/school_math_0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M) | sft | alpaca | school_math_0.25M.json | jsonl |
| [sahil2801/CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) | sft | alpaca | code_alpaca_20k.json | json |
| [TIGER-Lab/MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) | sft | alpaca | MathInstruct.json | json |
| [YeungNLP/firefly-train-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) | sft | alpaca | firefly-train-1.1M.jsonl | jsonl |
| [suolyer/webqa](https://huggingface.co/datasets/suolyer/webqa) | sft | alpaca | train.json | jsonl |
| [zxbsmk/webnovel_cn](https://huggingface.co/datasets/zxbsmk/webnovel_cn) | sft | alpaca | novel_cn_token512_50k.json | json |
| [AstraMindAI/SFT-Nectar](https://huggingface.co/datasets/AstraMindAI/SFT-Nectar) | sft | alpaca | sft_data_structured.json | json |
| [hfl/stem_zh_instruction](https://huggingface.co/datasets/hfl/stem_zh_instruction) | sft | alpaca | bio_50282.json | jsonl |
| [llamafactory/OpenO1-SFT](https://huggingface.co/datasets/llamafactory/OpenO1-SFT) | sft | alpaca | OpenO1-SFT-Pro.jsonl | jsonl |
| [Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT) | sft | alpaca | distill_r1_110k_sft.jsonl | jsonl |
| [mayflowergmbh/oasst_de](https://huggingface.co/datasets/mayflowergmbh/oasst_de) | sft | alpaca | oasst_de.json | json |
| [mayflowergmbh/dolly-15k_de](https://huggingface.co/datasets/mayflowergmbh/dolly-15k_de) | sft | alpaca | dolly_de.json | json |
| [mayflowergmbh/alpaca-gpt4_de](https://huggingface.co/datasets/mayflowergmbh/alpaca-gpt4_de) | sft | alpaca | alpaca_gpt4_data_de.json | json |
| [mayflowergmbh/openschnabeltier_de](https://huggingface.co/datasets/mayflowergmbh/openschnabeltier_de) | sft | alpaca | openschnabeltier.json | json |
| [mayflowergmbh/evol-instruct_de](https://huggingface.co/datasets/mayflowergmbh/evol-instruct_de) | sft | alpaca | evol_instruct_de.json | json |
| [mayflowergmbh/dolphin_de](https://huggingface.co/datasets/mayflowergmbh/dolphin_de) | sft | alpaca | dolphin.json | json |
| [mayflowergmbh/booksum_de](https://huggingface.co/datasets/mayflowergmbh/booksum_de) | sft | alpaca | booksum.json | json |
| [mayflowergmbh/airoboros-3.0_de](https://huggingface.co/datasets/mayflowergmbh/airoboros-3.0_de) | sft | alpaca | airoboros_3.json | json |
| [mayflowergmbh/ultra-chat_de](https://huggingface.co/datasets/mayflowergmbh/ultra-chat_de) | sft | alpaca | ultra_chat_german.json | json |
| [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs) | dpo | alpaca | orca_rlhf.jsonl | jsonl |

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

### SFT VL Dataset

We provide demo data for quick training, please download the [image](https://paddleformers.bj.bcebos.com/datasets/DoclingMatix.tar.gz) or [video](
https://paddleformers.bj.bcebos.com/datasets/NExTVideo.tar.gz) data according to your needs and unzip it to the [demo](../examples/data/)  data directory. You can either use these samples or train with your own data.

Required fields for SFT VL:


* `text_info`: The list of text data, each element contains a `text` and a `tag`
  * `text`: The text content from User question or System response
  * `tag`: The mask tag (`no_mask`=include in training, `mask`=exclude)
* `image_info`: The list of image data, each element contains a `image_url` and a `matched_text_index`
  * `image_url`: The url to download image online or the path to access image locally
  * `matched_text_index`: The index of matched text in `text_info`
    * Default: `matched_text_index=0` means the image is matched with the first text, and will be palced before the first text
* `is_system(optional)`: The system flag (1=system configuration, 0=no system configuration)
  * system configuration = `text_info[0]` if `is_system=1`

Notes:
* Each training sample is in JSON format, with multiple samples separated by newlines
* Video data is supported by replacing the `image_info` with `video_info`
  * the `image_url` can be a video url or video path
* Please ensure that `mask` items and `no_mask` items alternate in the `text_info`

Here is a multi-image example of SFT VL dataset:
```json
{
    "image_info": [
        {"matched_text_index": 0, "image_url": "./DoclingMatix/218/0.png"},
        {"matched_text_index": 0, "image_url": "./DoclingMatix/218/1.png"}
    ],
    "text_info": [
        {"text": "What is the purpose of the resolution discussed in the text?", "tag": "mask"},
        {"text": "The purpose of the resolution is to approve the redevelopment contract of the Philadelphia Redevelopment Authority for the redevelopment and urban renewal of a portion of the Haddington Urban Renewal Area, Unit Nos. 2 and 3, and to authorize the Redevelopment Authority to execute the redevelopment contract with Danielle M. Carson-Varns.", "tag": "no_mask"},
        {"text": "Who introduced Resolution No. 160204 to the City Council?", "tag": "mask"},
        {"text": "Councilmember Blackwell introduced Resolution No. 160204 to the City Council.", "tag": "no_mask"},
        ...
    ]
}
```

Here is a video example of SFT VL dataset:
```json
{
    "video_info": [
        {"matched_text_index": 0, "image_url": "./NExTVideo/1027/4789497818.mp4"}
    ],
    "text_info": [
        {"text": "how does the man sit on the grass?\nA. kneel\nB. one leg in the air\nC. sitting on bicycle seat\nD. legs spread out\nE. squatting down\n Answer with the option's letter from the given choices directly.", "tag": "mask"},
        {"text": "D", "tag": "no_mask"}
    ]
}
```

Here is a system configuration example of SFT VL dataset:
```json
{
    "is_system": 1,
    "text_info": [
        {"text": "Your role as ...", "tag": "mask"},
        {"text": "好的", "tag": "no_mask"},
        {"text": "What is written...", "tag": "mask"},
        {"text": "<think>So I've got...", "tag": "no_mask"},
        ...
    ]
    "image_info": [...]
}
```


## alpaca Format

### SFT Dataset

**Required fields for SFT**

* `instruction`: A clear task directive (e.g., "Translate the following Chinese text to English").
* `input`: Task-specific input content (may be empty for tasks like "Write a poem").
* `output`: The expected model response.

**Supports json and jsonl file formats**

* **json**: Each line contains one JSON object:
```json
[
    {"instruction":"instructionA", "input":"inputA", "output":"outputA"},
    {"instruction":"instructionB", "input":"inputB", "output":"outputB"},
    {"instruction":"instructionC", "input":"inputC", "output":"outputC"}
]
```

* **jsonl**: All data in a single JSON array:
```json
{"instruction":"instructionA", "input":"inputA", "output":"outputA"}
{"instruction":"instructionB", "input":"inputB", "output":"outputB"}
{"instruction":"instructionC", "input":"inputC", "output":"outputC"}
```

**Field Mapping Between alpaca and erniekit**

| alpaca | erniekit | Mapping |
|--------|----------|---------|
| instruction <br> input | src | src[-1] = instruction + input |
| output | tgt | tgt[-1] = output |
| history | src <br> tgt | history = zip(src[:-1], tgt[:-1]) |
| system | system | system=system |

### DPO Dataset

**Required fields for DPO**

* `system(optional)`: System configuration
* `question`: User question.
* `chosen`: The higher-quality output selected by human annotators.
* `rejected`: The lower-quality output for the same question.

**Supports json and jsonl file formats**

* **json**: Each line contains one JSON object:
```json
[
    {"system": "你是一个AI小助理", "question": "哪一个富含蛋白质，床还是墙？", "chosen": "床和墙都不是蛋白质的来源，因为它们都是无生命的物体。蛋白质通常存在于肉类、奶制品、豆类和坚果等食物中。", "rejected": "对不起，我无法回答那个问题。请提供更具体的信息，让我知道你需要什么帮助。"},
    {"system": "你是一个AI小助理", "question": "哪一个富含蛋白质，床还是墙？", "chosen": "床和墙都不是蛋白质的来源，因为它们都是无生命的物体。蛋白质通常存在于肉类、奶制品、豆类和坚果等食物中。", "rejected": "对不起，我无法回答那个问题。请提供更具体的信息，让我知道你需要什么帮助。"},
    {"system": "你是一个AI小助理", "question": "哪一个富含蛋白质，床还是墙？", "chosen": "床和墙都不是蛋白质的来源，因为它们都是无生命的物体。蛋白质通常存在于肉类、奶制品、豆类和坚果等食物中。", "rejected": "对不起，我无法回答那个问题。请提供更具体的信息，让我知道你需要什么帮助。"}
]
```

* **jsonl**: All data in a single JSON array:
```json
{"system": "你是一个AI小助理", "question": "哪一个富含蛋白质，床还是墙？", "chosen": "床和墙都不是蛋白质的来源，因为它们都是无生命的物体。蛋白质通常存在于肉类、奶制品、豆类和坚果等食物中。", "rejected": "对不起，我无法回答那个问题。请提供更具体的信息，让我知道你需要什么帮助。"}
{"system": "你是一个AI小助理", "question": "哪一个富含蛋白质，床还是墙？", "chosen": "床和墙都不是蛋白质的来源，因为它们都是无生命的物体。蛋白质通常存在于肉类、奶制品、豆类和坚果等食物中。", "rejected": "对不起，我无法回答那个问题。请提供更具体的信息，让我知道你需要什么帮助。"}
{"system": "你是一个AI小助理", "question": "哪一个富含蛋白质，床还是墙？", "chosen": "床和墙都不是蛋白质的来源，因为它们都是无生命的物体。蛋白质通常存在于肉类、奶制品、豆类和坚果等食物中。", "rejected": "对不起，我无法回答那个问题。请提供更具体的信息，让我知道你需要什么帮助。"}
```
