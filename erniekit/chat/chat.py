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

# The file has been adapted from hiyouga LLaMA-Factory project
# Copyright (c) 2025 LLaMA-Factory
# Licensed under the Apache License - https://github.com/hiyouga/LLaMA-Factory/blob/main/LICENSE

from typing import Any, Optional

import openai

from ..hparams import get_server_args, read_args


def run_chat(args: Optional[dict[str, Any]] = None) -> None:
    """
    Launch a conversation in the command line.
    Note: Service-oriented deployment needs to be carried out using 'erniekit server' first.
    """
    args = read_args(args)
    model_args, generating_args, finetuning_args, server_args = get_server_args(args)

    messages = []
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    ip = "0.0.0.0"
    service_http_port = str(server_args.port)
    client = openai.Client(base_url=f"http://{ip}:{service_http_port}/v1", api_key="EMPTY_API_KEY")

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            messages = []
            print("History has been removed.")
            continue

        messages.append({"role": "user", "content": query})
        print("Assistant: ", end="", flush=True)

        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=generating_args.temperature,
            top_p=generating_args.top_p,
            max_tokens=generating_args.max_new_tokens,
            frequency_penalty=generating_args.frequency_penalty,
            presence_penalty=generating_args.presence_penalty,
            stream=generating_args.stream,
            stream_options=generating_args.stream_options,
        )

        assistant_response = ""
        if generating_args.stream:
            for chunk in response:
                if chunk.choices[0].delta is not None:
                    print(chunk.choices[0].delta.content, end='')
                    assistant_response += chunk.choices[0].delta.content
        else:
            print(response.choices[0].message.content)
            assistant_response += response.choices[0].message.content
        print()
        messages.append({"role": "assistant", "content": assistant_response})
