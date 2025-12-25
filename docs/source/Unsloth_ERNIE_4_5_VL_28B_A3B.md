# Unsloth

## **Unsloth Fine-tuning Guide for ERNIE_4_5_VL_28B_A3B**
> **Reference:** This tutorial is adapted and extended based on the official Unsloth Colab notebook for ERNIE_4_5_VL_28B_A3B fine-tuning and inference.  
> You can run this notebook on Google Colab (paid) or for free using AMD resources:
> - Google Colab: [https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/ERNIE_4_5_VL_28B_A3B_PT_Vision.ipynb](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/ERNIE_4_5_VL_28B_A3B_PT_Vision.ipynb)  
> - Free on AMD: [https://oneclickamd.ai/github/unslothai/notebooks/blob/main/nb/ERNIE_4_5_VL_28B_A3B_PT_Vision.ipynb](https://oneclickamd.ai/github/unslothai/notebooks/blob/main/nb/ERNIE_4_5_VL_28B_A3B_PT_Vision.ipynb)  
> More notebooks are available by replacing the path after `nb/` with the corresponding notebook name in the GitHub repository.


This tutorial provides a comprehensive, end-to-end guide on how to fine-tune the **ERNIE_4_5_VL_28B_A3B** vision-language model using Unsloth, covering the complete workflow from environment setup and dataset construction to LoRA training, inference, and model export.

By the end of this tutorial, you will understand how to perform:

- Data Preparation
- Model Training
- Inference
- Model Saving & Deployment

GitHub Repository: [Unsloth](https://github.com/unslothai/unsloth)

## **Fine-tuning ERNIE_4_5_VL_28B_A3B with Unsloth**

### **Installation**
**Local Installation (Linux Recommended)**
```
pip install unsloth
```
For full installation instructions, refer to the official documentation:[installation instructions (English)](https://docs.unsloth.ai/get-started/installing-+-updating)

### **Model Loading & LoRA Configuration**
**Loading ERNIE-4.5-VL Model**
```
from unsloth import FastVisionModel # 对应 LLM 使用 FastLanguageModel
import torch
from transformers import AutoModelForCausalLM ,AutoProcessor

model_path = "unsloth/ERNIE-4.5-VL-28B-A3B-PT"
model, tokenizer = FastVisionModel.from_pretrained(
    model_path,
    auto_model=AutoModelForCausalLM,
    load_in_4bit = False, # 该模型不支持 4bit
    trust_remote_code = True,
    unsloth_force_compile = True,
    use_gradient_checkpointing = False,
    attn_implementation="eager"
)
```
**Load Processor & Register Image Preprocessing**
```
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
processor.eval()
model.add_image_preprocess(processor)
```
**Configure LoRA Adapter (PEFT)**
> Train only about 1% of the parameters, significantly reducing memory usage while maintaining model expressiveness, suitable for training 28B-level VL models in a single-card environment.

**[New Features]**

You can choose to:
- Fine-tune vision only
- Fine-tune language only
- Fine-tune both
- Selectively fine-tune Attention and/or MLP layers

```
model = FastVisionModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "fc1", "fc2",
    ]
)
```
### **Data Preparation**

**Unified Vision Conversation Format**

```
[
  {
    "role": "user",
    "content": [
      {"type": "text", "text": Q},
      {"type": "image", "image": image}
    ]
  },
  {
    "role": "assistant",
    "content": [
      {"type": "text", "text": A}
    ]
  }
]
```

**We use a subset of a handwritten math formula dataset, aiming to convert images into structured LaTeX expressions.**

Subset: ```unsloth/LaTeX_OCR```

Full Dataset: ```linxy/LaTeX_OCR```

```
from datasets import load_dataset
dataset = load_dataset("unsloth/LaTeX_OCR", split="train")
```

**The dataset must be converted into a multi-turn dialogue list, with each entry clearly distinguishing between text and image.**

```
instruction = "为这张图片写出对应的 LaTeX 表达式。"

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["text"]} ],
            "reasoning_content": "\n" # 如果保持为 "\n"，则会训练模型输出空的思考过程        },
    ]
    return { "messages" : conversation }

converted_dataset = [convert_to_conversation(sample) for sample in dataset]
```

**Inference Check**

```
FastVisionModel.for_inference(model) # Enable for inference!

image = dataset[2]["image"]

instruction = "Write the LaTeX representation for this image."
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"}, # Placeholder required for the template
            {"type": "text", "text": instruction}
        ]
    }
]
text_prompt = processor.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
inputs = processor(
    text=[text_prompt],
    images=[image],
    videos=[],
    padding=True,
    return_tensors="pt",
)

# Move inputs to GPU
device = next(model.parameters()).device
inputs = inputs.to(device)

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens=128,
                   use_cache=False, temperature=1.5, min_p=0.1)
```
### **Training Components (Collator & Trainer)**
**Using Custom ErnieVisionDataCollator and Custom ErnieSFTTrainer**
> ERNIE-4.5-VL uses 3D position_ids and image patch tokens, so the default collator of the standard SFTTrainer cannot be reused.

```
# @title Setup Collator & Trainer

from trl import SFTTrainer, SFTConfig
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class ErnieVisionDataCollator:
    processor: Any
    tokenizer: Any
    ignore_index: int = -100
    max_seq_length: int = 2048
    train_on_responses_only: bool = False

    _img_patch_id: int = field(init=False, default=-1)

    def __post_init__(self):
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        patch_token = "<|IMAGE_PLACEHOLDER|>"
        converted_id = self.tokenizer.convert_tokens_to_ids(patch_token)
        self._img_patch_id = converted_id if converted_id is not None else -1

    def _extract_visuals(self, msgs: List[Dict]) -> tuple:
        image_inputs, video_inputs = [], []
        needs_extraction = False

        for msg in msgs:
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if "image" in part:
                            image_inputs.append(part["image"])
                        elif part.get("type") in ["image_url", "video_url"]:
                            needs_extraction = True

        if needs_extraction and not image_inputs:
            try:
                return self.processor.process_vision_info(msgs)
            except Exception:
                return [], []

        return image_inputs, video_inputs

    def _mask_prompt(self, msgs: List[Dict], image_inputs: List, labels: torch.Tensor, full_input_ids: torch.Tensor) -> torch.Tensor:
        last_asst_idx = -1
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i]['role'] == 'assistant':
                last_asst_idx = i
                break

        if last_asst_idx == -1:
            return labels

        prompt_msgs = msgs[:last_asst_idx]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        prompt_inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            return_tensors="pt"
        )

        prompt_ids = prompt_inputs['input_ids'][0]

        len_full = full_input_ids.size(0)
        len_prompt = prompt_ids.size(0)
        limit = min(len_full, len_prompt)

        matches = (full_input_ids[:limit] == prompt_ids[:limit])

        mismatches = (~matches).nonzero(as_tuple=False)

        if len(mismatches) > 0:
            mask_len = mismatches[0].item()
        else:
            mask_len = limit

        labels[:mask_len] = self.ignore_index

        return labels

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {k: [] for k in ["input_ids", "labels", "token_type_ids", "position_ids", "images", "grid_thw", "image_type_ids"]}

        for example in features:
            msgs = example.get("messages", example.get("conversations", []))
            image_inputs, video_inputs = self._extract_visuals(msgs)

            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt"
            )

            input_ids = inputs['input_ids'][0]
            tt = inputs['token_type_ids'][0]
            pos = inputs['position_ids'][0]

            if input_ids[-1] != self.tokenizer.eos_token_id:
                input_ids = torch.cat([input_ids, torch.tensor([self.tokenizer.eos_token_id])])
                tt = torch.cat([tt, torch.tensor([0], dtype=tt.dtype)])
                pos = torch.cat([pos, (pos[-1] + 1).unsqueeze(0)])

            labels = input_ids.clone()


            if self._img_patch_id != -1:
                labels[labels == self._img_patch_id] = self.ignore_index

            if self.train_on_responses_only:
                labels = self._mask_prompt(msgs, image_inputs, labels, input_ids)

            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)
            batch["token_type_ids"].append(torch.cat([tt, torch.tensor([0])]))
            batch["position_ids"].append(pos)

            if inputs.get('images') is not None: batch["images"].append(inputs['images'])
            if inputs.get('grid_thw') is not None: batch["grid_thw"].append(inputs['grid_thw'])
            if inputs.get('image_type_ids') is not None: batch["image_type_ids"].append(inputs['image_type_ids'])

        padded_input = torch.nn.utils.rnn.pad_sequence(batch["input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_label = torch.nn.utils.rnn.pad_sequence(batch["labels"], batch_first=True, padding_value=self.ignore_index)
        padded_tt = torch.nn.utils.rnn.pad_sequence(batch["token_type_ids"], batch_first=True, padding_value=0)

        max_len = padded_input.shape[1]
        padded_pos = torch.zeros((len(batch["position_ids"]), max_len, 3), dtype=torch.long)
        for i, p in enumerate(batch["position_ids"]):
            l = min(p.shape[0], max_len)
            padded_pos[i, :l, :] = p[:l]

        if padded_input.shape[1] > self.max_seq_length:
            padded_input = padded_input[:, :self.max_seq_length]
            padded_label = padded_label[:, :self.max_seq_length]
            padded_pos = padded_pos[:, :self.max_seq_length, :]
            padded_tt = padded_tt[:, :self.max_seq_length + 1]

        final_batch = {
            "input_ids": padded_input,
            "labels": padded_label,
            "attention_mask": padded_input.ne(self.tokenizer.pad_token_id).long(),
            "token_type_ids": padded_tt,
            "position_ids": padded_pos,
        }

        if batch["images"]: final_batch["images"] = torch.cat(batch["images"], dim=0)
        if batch["grid_thw"]: final_batch["grid_thw"] = torch.cat(batch["grid_thw"], dim=0)
        if batch["image_type_ids"]: final_batch["image_type_ids"] = torch.cat(batch["image_type_ids"], dim=0)

        return final_batch

class ErnieSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)

        logits = outputs.logits
        labels = inputs.get("labels")

        loss = None
        if labels is not None:

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            loss = loss_fct(shift_logits, shift_labels)

            if hasattr(outputs, "router_loss") and outputs.router_loss is not None:
                aux_loss = outputs.router_loss.to(loss.device)
                loss += aux_loss

        if return_outputs:
            return (loss, outputs)
        return loss
```
### **Model Training (Train)**
**For quick demonstration, we only train for 30 steps.**
For formal training, set num_train_epochs=1 and turn off max_steps.
```
from trl import  SFTConfig

FastVisionModel.for_training(model) # Enable for training!

custom_collator = ErnieVisionDataCollator(
    processor=processor,
    tokenizer=tokenizer,
    max_seq_length=2048,
    train_on_responses_only = True,
)

trainer = ErnieSFTTrainer(
    model = model,
    tokenizer = processor.tokenizer,
    data_collator = custom_collator,
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 2,
        warmup_steps = 5,
        max_steps = 30,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        gradient_checkpointing = False,
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
    ),
)
trainer_stats = trainer.train()
```
### **Inference**
**We use:**
```
temperature = 1.5
min_p = 0.1
```
> This combination suppresses low-probability noise while preserving creativity — ideal for structured formula output.
👉 For detailed reasoning, see this tweet: [https://x.com/menhguin/status/1826132708508213629](https://x.com/menhguin/status/1826132708508213629)

```
FastVisionModel.for_inference(model) # Enable for inference!

image = dataset[2]["image"]

instruction = "Write the LaTeX representation for this image."
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"}, # Placeholder required for the template
            {"type": "text", "text": instruction}
        ]
    }
]
text_prompt = processor.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
inputs = processor(
    text=[text_prompt],
    images=[image],
    videos=[],
    padding=True,
    return_tensors="pt",
)

# Move inputs to GPU
device = next(model.parameters()).device
inputs = inputs.to(device)

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens=128,
                   use_cache=False, temperature=1.5, min_p=0.1)
```
### **Saving & Loading **
**Save LoRA adapters**
```
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
```
**Load for inference**
```
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "lora_model",
    load_in_4bit = False,
)
FastVisionModel.for_inference(model)
```
**Merge & export (float16, for vLLM)**
```
model.save_pretrained_merged("finetune", tokenizer)
```
### Conclusion

You now have a complete, production-grade pipeline for fine-tuning ERNIE-4.5-VL-28B using Unsloth, covering dataset design, training, inference, and deployment. 

This workflow enables efficient multimodal training on limited hardware while preserving the model’s full reasoning capabilities.
