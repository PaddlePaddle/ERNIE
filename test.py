import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM

model_path = '/root/paddlejob/workspace/env/output/lrl/ERNIE/eb45lite_think_feed_vid_im_v2_midtrain_1021_step70_1r_torch'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    # torch_dtype=torch.bfloat16,
    dtype=torch.bfloat16,
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
processor.eval()
model.add_image_preprocess(processor)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's the whether in Beijing."},
            {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"}},
        ]
    },
]

tools = [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Determine weather in my location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": [
                "c",
                "f"
              ]
            }
          },
          "additionalProperties": False,
          "required": [
            "location",
            "unit"
          ]
        },
        "strict": True
      }
    }]

text = processor.tokenizer.apply_chat_template(
    messages,
    tools=tools,
    tokenize=False,
    add_generation_prompt=True, 
    chat_template_kwargs={"options": {"thinking_mode": "true", "tool_choice": {"mode": "required"}}},
)
print("text: ", text)
image_inputs, video_inputs = processor.process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

device = next(model.parameters()).device
inputs = inputs.to(device)

import time
start_time = time.time()
generated_ids = model.generate(
    inputs=inputs['input_ids'].to(device),
    **inputs,
    max_new_tokens=8192,
    use_cache=False
    )
end_time = time.time()
output_text = processor.decode(generated_ids[0][len(inputs['input_ids']):])
print("output_text: ", output_text[len(inputs['input_ids']):])
print("耗时: ", end_time - start_time)