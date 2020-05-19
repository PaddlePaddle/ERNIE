import sys
import json

data = {}
for line in sys.stdin:
    key, value = line.strip().split(": ")
    data[key] = float(value)

print(json.dumps(data))
