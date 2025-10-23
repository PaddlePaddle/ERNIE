import json

f = open('/root/paddlejob/workspace/env/output/lrl/ERNIE/examples/data/sft-train.jsonl', 'rb')
output_file = "sft-pretrain.jsonl"
oral_data = []
for line in f:
    oral_data.append(json.loads(line))

with open(output_file, "w", encoding="utf-8") as f:
    for item in oral_data:
        new_item = dict()
        new_item["tgt"] = item["tgt"]
        f.write(json.dumps(new_item, ensure_ascii=False) + '\n')

print('over')

