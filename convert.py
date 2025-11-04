# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sys

from typing import List


def thinking_merge(reasoning_content, content):
    assert isinstance(content, List)
    if len(content) > 0:
        assert len(content) == 1
        return "<think>" + reasoning_content + "</think>" + content[0]["text"]
    else:
        return "<think>" + reasoning_content + "</think>"


def convert_jsonl(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:

        for line in infile:
            try:
                data = json.loads(line.strip())
                messages = data["prompt"]
                video_info = []
                text_info = []
                for i, message in enumerate(messages):
                    # user / tool
                    if "role" in message and (
                        message["role"] == "user" or message["role"] == "tool"
                    ):
                        if isinstance(message["content"], List):
                            # 有时候只有图片
                            have_text = False
                            for item in message["content"]:
                                if item["type"] == "image_url":
                                    video_info.append(
                                        {
                                            "matched_text_index": i,
                                            "image_url": item["image_url"]["url"],
                                        }
                                    )
                                elif item["type"] == "text":
                                    have_text = True
                                    text_info.append(
                                        {
                                            "text": item["text"],
                                            "tag": "mask",
                                            "tool_response": (
                                                True
                                                if message["role"] == "tool"
                                                else False
                                            ),
                                        }
                                    )
                            if not have_text:
                                text_info.append(
                                    {
                                        "text": "",
                                        "tag": "mask",
                                        "tool_response": (
                                            True if message["role"] == "tool" else False
                                        ),
                                    }
                                )
                    # assistant
                    if "role" in message and message["role"] == "assistant":
                        text_info.append(
                            {
                                "text": thinking_merge(
                                    message["reasoning_content"], message["content"]
                                ),
                                "tool_calls": message["tool_calls"],
                                "tag": "mask",
                            }
                        )

                # tgt
                candidate = data["candidates"]
                if "role" in candidate and candidate["role"] == "assistant":
                    text_info.append(
                        {
                            "text": thinking_merge(
                                candidate["reasoning_content"], candidate["content"]
                            ),
                            "tool_calls": candidate["tool_calls"],
                            "tag": "no_mask",
                        }
                    )

                # 写入新的JSONL格式
                new_data = {
                    "video_info": video_info,
                    "text_info": text_info,
                    "tools": data["tools"],
                }
                outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")

            except json.JSONDecodeError:
                print(
                    f"Warning: Skipping invalid JSON line: {line.strip()}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input.jsonl output.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_jsonl(input_file, output_file)
