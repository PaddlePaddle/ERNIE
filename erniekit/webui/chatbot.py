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

"""
Chatbot, send, terminate, and return the model's dialogue.
"""


import asyncio

import gradio as gr
import openai

from erniekit.webui.common import config


class ChatBotGenerator:
    """

    Chatbot Generator, supporting multimodal response and thinking process generation

    """

    def __init__(self):
        self.default_ip = "0.0.0.0"
        self.openai_client = None
        self.stop_generation = False  # 添加停止标志

    def stop(self):
        """
        Set a stop flag to interrupt the generation process

        Args:
            self (object): Instance of the class
        """

        self.stop_generation = True

    def reset(self):
        """
        Reset the stop flag

        Args:
            self (object): Instance of the class
        """

        self.stop_generation = False

    def create_openai_client(self, port):
        """
        Create an OpenAI client connection

        Args:
            self (object): Instance of the class
            port (int): Port number for the connection
        """

        base_url = f"http://{self.default_ip}:{port}/v1"
        return openai.Client(base_url=base_url, api_key="EMPTY_API_KEY")

    async def build_message_history(self, message, history, role_setting, system_prompt):
        """
        Build message history and uniformly process different history record formats

        Args:
            self (object): Instance of the class
            message (str): Current message content
            history (list): Conversation history
            role_setting (dict): Role configuration information
            system_prompt (str): System prompt message
        """

        messages = []

        if role_setting or system_prompt:
            system_content = ""
            if role_setting:
                system_content += f"你现在扮演: {role_setting}"
            if system_prompt:
                system_content += system_prompt
            if system_content:
                messages.append({"role": "system", "content": system_content})

        if history:
            for entry in history:
                if isinstance(entry, dict) and "role" in entry:
                    role = entry["role"]
                    content = entry.get("content", "")
                    if role == "user":
                        messages.append({"role": "user", "content": content})
                    elif role == "assistant":
                        messages.append({"role": "assistant", "content": content})
                elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                    user_msg, bot_msg = entry
                    messages.append({"role": "user", "content": user_msg})
                    if bot_msg:
                        messages.append({"role": "assistant", "content": bot_msg})
                else:
                    print(f"Warning: Unresolvable historical record format: {entry}")

        messages.append({"role": "user", "content": message})
        return messages

    async def mm_response(
        self,
        message,
        history,
        role_setting,
        system_prompt,
        max_length,
        top_p,
        temperature,
        port=8188,
    ):
        """
        Generate multimodal response and uniformly process history record format

        Args:
            self (object): Instance of the class
            message (str): User message
            history (list): Conversation history
            role_setting (dict): Role settings
            system_prompt (str): System instruction prompt
            max_length (int): Maximum response length
            top_p (float): Nucleus sampling probability
            temperature (float): Sampling temperature
            port (int, optional): Service port (default: 8188)
        """

        if not message:
            yield [], gr.update(value="")
            return

        self.reset()

        try:
            client = self.create_openai_client(port)
            messages = await self.build_message_history(
                message,
                history,
                role_setting,
                system_prompt,
            )

            response = client.chat.completions.create(
                model="default",
                messages=messages,  # 确保使用完整的消息历史
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_length,
                stream=True,
            )

            new_history = list(history) if history else []

            user_message = {"role": "user", "content": message}
            new_history.append(user_message)

            assistant_response = {"role": "assistant", "content": ""}
            new_history.append(assistant_response)

            for chunk in response:
                if self.stop_generation:
                    break

                if chunk.choices[0].delta and chunk.choices[0].delta.content:
                    assistant_response["content"] += chunk.choices[0].delta.content
                    yield new_history, gr.update(value="")
                    await asyncio.sleep(0.01)

            yield new_history, gr.update(value="")

        except Exception as e:
            print(f"mm_response error: {e}")
            error_msg = {"role": "assistant", "content": f"API call failed：{e!s}"}
            yield [{"role": "user", "content": message}, error_msg], gr.update(value="")
        finally:
            self.reset()

    async def thought_response(
        self,
        message,
        history,
        role_setting=None,
        system_prompt=None,
        max_length=1000,
        top_p=0.8,
        temperature=0.7,
        port=8188,
    ):
        """
        Generate response with thought process and unify history record format

        Args:
            self (object): Instance of the class
            message (str): User message
            history (list): Conversation history
            role_setting (dict, optional): Role settings (default: None)
            system_prompt (str, optional): System instruction prompt (default: None)
            max_length (int, optional): Maximum response length (default: 1000)
            top_p (float, optional): Nucleus sampling probability (default: 0.8)
            temperature (float, optional): Sampling temperature (default: 0.7)
            port (int, optional): Service port (default: 8188)
        """

        if not message:
            yield [], gr.update(value="")
            return

        self.reset()

        try:
            client = self.create_openai_client(port)

            messages = await self.build_message_history(message, history, role_setting, system_prompt)

            response = client.chat.completions.create(
                model="default",
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_length,
                stream=True,
            )

            new_history = list(history) if history else []

            user_message = {"role": "user", "content": message}
            new_history.append(user_message)

            assistant_response = {"role": "assistant", "content": ""}
            new_history.append(assistant_response)

            current_thought = ""
            current_response = ""

            for chunk in response:
                if self.stop_generation:
                    break

                if chunk.choices[0].delta:
                    thought_part = getattr(chunk.choices[0].delta, "reasoning_content", "")
                    answer_part = getattr(chunk.choices[0].delta, "content", "")

                    current_thought += thought_part
                    current_response += answer_part

                    formatted_response = (
                        f"<details open><summary>思考过程</summary>\n"
                        f"<div class='thought-container' style='font-size: 13px;opacity: 0.85;"
                        f"padding-left:20px;border-left:3px solid #ddd;"
                        f"margin-bottom: 1em;'>{current_thought}</div>\n"
                        f"</details>\n"
                        f"<div class='response-container' style='line-height: 1.5;'>{current_response}</div>"
                    )

                    assistant_response["content"] = formatted_response
                    yield new_history, gr.update(value="")
                    await asyncio.sleep(0.01)

            yield new_history, gr.update(value="")

        except Exception as e:
            print(f"thought_response error: {e}")
            error_msg = {"role": "assistant", "content": f"The thinking process generation failed：{e!s}"}
            yield [{"role": "user", "content": message}, error_msg], gr.update(value="")
        finally:
            self.reset()

    def check_thought_model(self, model_name):
        """
        Validate if the specified model supports thought process generation

        Args:
            self (object): Instance of the class
            model_name (str): Name of the model to check
        """

        return config.is_thought_model(model_name)


chatbot = ChatBotGenerator()
