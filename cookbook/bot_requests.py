# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BotClient class for interacting with bot models."""

import argparse
import json
import logging
import traceback

import jieba
import requests
from openai import OpenAI


class BotClient:
    """Client for interacting with various AI models."""

    def __init__(self, args: argparse.Namespace):
        """
        Initializes the BotClient instance by configuring essential parameters from command line arguments
        including retry limits, character constraints, model endpoints and API credentials while setting up
        default values for missing arguments to ensure robust operation.

        Args:
            args (argparse.Namespace): Command line arguments containing configuration parameters.
                                      Uses getattr() to safely retrieve values with fallback defaults.
        """
        self.logger = logging.getLogger(__name__)

        self.max_retry_num = getattr(args, "max_retry_num", 3)
        self.max_char = getattr(args, "max_char", 8000)

        self.model_map = getattr(args, "model_map", {})
        self.api_key = getattr(args, "api_key", "bce-v3/xxx")

        self.embedding_service_url = getattr(
            args, "embedding_service_url", "embedding_service_url"
        )
        self.embedding_model = getattr(args, "embedding_model", "embedding_model")

        self.web_search_service_url = getattr(
            args, "web_search_service_url", "web_search_service_url"
        )
        self.max_search_results_num = getattr(args, "max_search_results_num", 15)

        self.qianfan_api_key = getattr(args, "qianfan_api_key", "bce-v3/xxx")

    def call_back(self, host_url: str, req_data: dict) -> dict:
        """
        Executes an HTTP request to the specified endpoint using the OpenAI client, handles the response
        conversion to a compatible dictionary format, and manages any exceptions that may occur during
        the request process while logging errors appropriately.

        Args:
            host_url (str): The URL to send the request to.
            req_data (dict): The data to send in the request body.

        Returns:
            dict: Parsed JSON response from the server. Returns empty dict
                if request fails or response is invalid.
        """
        try:
            client = OpenAI(base_url=host_url, api_key=self.api_key)
            response = client.chat.completions.create(**req_data)

            # Convert OpenAI response to compatible format
            return response.model_dump()

        except Exception as e:
            self.logger.error(f"Stream request failed: {e}")
            raise

    def call_back_stream(self, host_url: str, req_data: dict) -> dict:
        """
        Makes a streaming HTTP request to the specified host URL using the OpenAI client and yields response chunks
        in real-time while handling any exceptions that may occur during the streaming process.

        Args:
            host_url (str): The URL to send the request to.
            req_data (dict): The data to send in the request body.

        Returns:
            generator: Generator that yields parsed JSON responses from the server.
        """
        try:
            client = OpenAI(base_url=host_url, api_key=self.api_key)
            response = client.chat.completions.create(
                **req_data,
                stream=True,
            )
            for chunk in response:
                if not chunk.choices:
                    continue

                # Convert OpenAI response to compatible format
                yield chunk.model_dump()

        except Exception as e:
            self.logger.error(f"Stream request failed: {e}")
            raise

    def process(
        self,
        model_name: str,
        req_data: dict,
        max_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.7,
    ) -> dict:
        """
        Handles chat completion requests by mapping the model name to its endpoint, preparing request parameters
        including token limits and sampling settings, truncating messages to fit character limits, making API calls
        with built-in retry mechanism, and logging the full request/response cycle for debugging purposes.

        Args:
            model_name (str): Name of the model, used to look up the model URL from model_map.
            req_data (dict): Dictionary containing request data, including information to be processed.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature to control the diversity of generated text.
            top_p (float): Cumulative probability threshold to control the diversity of generated text.

        Returns:
            dict: Dictionary containing the model's processing results.
        """
        model_url = self.model_map[model_name]

        req_data["model"] = model_name
        req_data["max_tokens"] = max_tokens
        req_data["temperature"] = temperature
        req_data["top_p"] = top_p
        req_data["messages"] = self.truncate_messages(req_data["messages"])
        for _ in range(self.max_retry_num):
            try:
                self.logger.info(f"[MODEL] {model_url}")
                self.logger.info("[req_data]====>")
                self.logger.info(json.dumps(req_data, ensure_ascii=False))
                res = self.call_back(model_url, req_data)
                self.logger.info("model response")
                self.logger.info(res)
                self.logger.info("-" * 30)
            except Exception as e:
                self.logger.info(e)
                self.logger.info(traceback.format_exc())
                res = {}
            if len(res) != 0 and "error" not in res:
                break

        return res

    def process_stream(
        self,
        model_name: str,
        req_data: dict,
        max_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.7,
    ) -> dict:
        """
        Processes streaming requests by mapping the model name to its endpoint, configuring request parameters,
        implementing a retry mechanism with logging, and streaming back response chunks in real-time while
        handling any errors that may occur during the streaming session.

        Args:
            model_name (str): Name of the model, used to look up the model URL from model_map.
            req_data (dict): Dictionary containing request data, including information to be processed.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature to control the diversity of generated text.
            top_p (float): Cumulative probability threshold to control the diversity of generated text.

        Yields:
            dict: Dictionary containing the model's processing results.
        """
        model_url = self.model_map[model_name]
        req_data["model"] = model_name
        req_data["max_tokens"] = max_tokens
        req_data["temperature"] = temperature
        req_data["top_p"] = top_p
        req_data["messages"] = self.truncate_messages(req_data["messages"])

        last_error = None
        for _ in range(self.max_retry_num):
            try:
                self.logger.info(f"[MODEL] {model_url}")
                self.logger.info("[req_data]====>")
                self.logger.info(json.dumps(req_data, ensure_ascii=False))

                yield from self.call_back_stream(model_url, req_data)
                return

            except Exception as e:
                last_error = e
                self.logger.error(
                    f"Stream request failed (attempt {_ + 1}/{self.max_retry_num}): {e}"
                )

        self.logger.error("All retry attempts failed for stream request")
        yield {"error": str(last_error)}

    def cut_chinese_english(self, text: str) -> list:
        """
        Segments mixed Chinese and English text into individual components using Jieba for Chinese words
        while preserving English words as whole units, with special handling for Unicode character ranges
        to distinguish between the two languages.

        Args:
            text (str): Input string to be segmented.

        Returns:
            list: A list of segments, where each segment is either a letter or a word.
        """
        words = jieba.lcut(text)
        en_ch_words = []

        for word in words:
            if word.isalpha() and not any(
                "\u4e00" <= char <= "\u9fff" for char in word
            ):
                en_ch_words.append(word)
            else:
                en_ch_words.extend(list(word))
        return en_ch_words

    def truncate_messages(self, messages: list[dict]) -> list:
        """
        Truncates conversation messages to fit within the maximum character limit (self.max_char)
        by intelligently removing content while preserving message structure. The truncation follows
        a prioritized order: historical messages first, then system message, and finally the last message.

        Args:
            messages (list[dict]): List of messages to be truncated.

        Returns:
            list[dict]: Modified list of messages after truncation.
        """
        if not messages:
            return messages

        processed = []
        total_units = 0

        for msg in messages:
            # Handle two different content formats
            if isinstance(msg["content"], str):
                text_content = msg["content"]
            elif isinstance(msg["content"], list):
                text_content = msg["content"][1]["text"]
            else:
                text_content = ""

            # Calculate unit count after tokenization
            units = self.cut_chinese_english(text_content)
            unit_count = len(units)

            processed.append(
                {
                    "role": msg["role"],
                    "original_content": msg["content"],  # Preserve original content
                    "text_content": text_content,  # Extracted plain text
                    "units": units,
                    "unit_count": unit_count,
                }
            )
            total_units += unit_count

        if total_units <= self.max_char:
            return messages

        # Number of units to remove
        to_remove = total_units - self.max_char

        # 1. Truncate historical messages
        for i in range(len(processed) - 1, 1):
            if to_remove <= 0:
                break

            # current = processed[i]
            if processed[i]["unit_count"] <= to_remove:
                processed[i]["text_content"] = ""
                to_remove -= processed[i]["unit_count"]
                if isinstance(processed[i]["original_content"], str):
                    processed[i]["original_content"] = ""
                elif isinstance(processed[i]["original_content"], list):
                    processed[i]["original_content"][1]["text"] = ""
            else:
                kept_units = processed[i]["units"][:-to_remove]
                new_text = "".join(kept_units)
                processed[i]["text_content"] = new_text
                if isinstance(processed[i]["original_content"], str):
                    processed[i]["original_content"] = new_text
                elif isinstance(processed[i]["original_content"], list):
                    processed[i]["original_content"][1]["text"] = new_text
                to_remove = 0

        # 2. Truncate system message
        if to_remove > 0:
            system_msg = processed[0]
            if system_msg["unit_count"] <= to_remove:
                processed[0]["text_content"] = ""
                to_remove -= system_msg["unit_count"]
                if isinstance(processed[0]["original_content"], str):
                    processed[0]["original_content"] = ""
                elif isinstance(processed[0]["original_content"], list):
                    processed[0]["original_content"][1]["text"] = ""
            else:
                kept_units = system_msg["units"][:-to_remove]
                new_text = "".join(kept_units)
                processed[0]["text_content"] = new_text
                if isinstance(processed[0]["original_content"], str):
                    processed[0]["original_content"] = new_text
                elif isinstance(processed[0]["original_content"], list):
                    processed[0]["original_content"][1]["text"] = new_text
                to_remove = 0

        # 3. Truncate last message
        if to_remove > 0 and len(processed) > 1:
            last_msg = processed[-1]
            if last_msg["unit_count"] > to_remove:
                kept_units = last_msg["units"][:-to_remove]
                new_text = "".join(kept_units)
                last_msg["text_content"] = new_text
                if isinstance(last_msg["original_content"], str):
                    last_msg["original_content"] = new_text
                elif isinstance(last_msg["original_content"], list):
                    last_msg["original_content"][1]["text"] = new_text
            else:
                last_msg["text_content"] = ""
                if isinstance(last_msg["original_content"], str):
                    last_msg["original_content"] = ""
                elif isinstance(last_msg["original_content"], list):
                    last_msg["original_content"][1]["text"] = ""

        result = []
        for msg in processed:
            if msg["text_content"]:
                result.append({"role": msg["role"], "content": msg["original_content"]})

        return result

    def embed_fn(self, text: str) -> list:
        """
        Generate an embedding for the given text using the QianFan API.

        Args:
            text (str): The input text to be embedded.

        Returns:
            list: A list of floats representing the embedding.
        """
        client = OpenAI(
            base_url=self.embedding_service_url, api_key=self.qianfan_api_key
        )
        response = client.embeddings.create(input=[text], model=self.embedding_model)
        return response.data[0].embedding

    def get_web_search_res(self, query_list: list) -> list:
        """
        Send a request to the AI Search service using the provided API key and service URL.

        Args:
            query_list (list): List of queries to send to the AI Search service.

        Returns:
            list: List of responses from the AI Search service.
        """
        headers = {
            "Authorization": "Bearer " + self.qianfan_api_key,
            "Content-Type": "application/json",
        }

        results = []
        top_k = self.max_search_results_num // len(query_list)
        for query in query_list:
            payload = {
                "messages": [{"role": "user", "content": query}],
                "resource_type_filter": [{"type": "web", "top_k": top_k}],
            }
            response = requests.post(
                self.web_search_service_url, headers=headers, json=payload
            )

            if response.status_code == 200:
                response = response.json()
                self.logger.info(response)
                results.append(response["references"])
            else:
                self.logger.info(f"请求失败，状态码: {response.status_code}")
                self.logger.info(response.text)
        return results
