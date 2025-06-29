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

"""This script provides a simple web interface that allows users to interact with."""

import argparse
import asyncio
import base64
import json
import logging
import os
import textwrap
from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
from functools import partial

import gradio as gr
import pdfplumber
from bot_requests import BotClient
from crawl_utils import CrawlUtils
from docx import Document

os.environ["NO_PROXY"] = "localhost,127.0.0.1"  # Disable proxy

logging.root.setLevel(logging.INFO)

IMAGE_FILE_TYPE = [".png", ".jpeg", ".jpg"]
TEXT_FILE_TYPE = [".pdf", ".txt", ".md", ".docx"]

SEARCH_INFO_PROMPT = textwrap.dedent(
    """\
    ## 当前时间
    {date}

    ## 对话
    {context}
    问题：{query}

    根据当前时间和对话完成以下任务：
    1. 查询判断：是否需要借助搜索引擎查询外部知识回答用户当前问题。
    2. 问题改写：改写用户当前问题，使其更适合在搜索引擎查询到相关知识。
    注意：只在**确有必要**的情况下改写，输出不超过 5 个改写结果，不要为了凑满数量而输出冗余问题。

    ## 输出如下格式的内容（只输出 JSON ，不要给出多余内容）：
    ```json
    {{
        "is_search":true/false,
        "query_list":["改写问题1"，"改写问题2"...]
    }}```
    """
)
ANSWER_PROMPT = textwrap.dedent(
    """\
    下面你会收到多段参考资料和一个问题。你的任务是阅读参考资料，并根据参考资料中的信息回答对话中的问题。
    以下是当前时间和参考资料：
    ---------
    ## 当前时间
    {date}

    ## 参考资料
    {reference}

    请严格遵守以下规则：
    1. 回答必须结合问题需求和当前时间，对参考资料的可用性进行判断，避免在回答中使用错误或过时的信息。
    2. 当参考资料中的信息无法准确地回答问题时，你需要在回答中提供获取相应信息的建议，或承认无法提供相应信息。
    3. 你需要优先根据百度高权威信息、百科、官网、权威机构、专业网站等高权威性来源的信息来回答问题，
       但务必不要用“（来源：xx）”这类格式给出来源，
       不要暴露来源网站中的“_百度高权威信息”，
       也不要出现'根据参考资料'，'根据当前时间'等表述。
    4. 更多地使用参考文章中的相关数字、案例、法律条文、公式等信息，让你的答案更专业。
    5. 只要使用了参考资料中的任何内容，必须在句末或段末加上资料编号，如 "[1]" 或 "[2][4]"。不要遗漏编号，也不要随意编造编号。编号必须来源于参考资料中已有的标注。
    ---------
    下面请结合以上信息，回答问题，补全对话:
    ## 对话
    {context}
    问题：{query}

    直接输出回复内容即可。
    """
)


def get_args() -> argparse.Namespace:
    """
    Parse and return command line arguments for the ERNIE chatbot demo.
    Configures server settings, model endpoints, and operational parameters.

    Returns:
        argparse.Namespace: Parsed command line arguments containing all the above settings.
    """
    parser = ArgumentParser(description="ERNIE models web chat demo.")

    parser.add_argument("--server-port", type=int, default=8666, help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Demo server name.")
    parser.add_argument("--max_char", type=int, default=20000, help="Maximum character limit for messages.")
    parser.add_argument("--max_retry_num", type=int, default=3, help="Maximum retry number for request.")
    parser.add_argument(
        "--model_map",
        type=str,
        required=True,
        default='{"ERNIE-4.5-VL": "http://localhost:port/v1"}',
        help="""JSON string defining model name to endpoint mappings.
            Required Format:
            {"ERNIE-4.5-VL": "http://localhost:port/v1"}

            Note:
            - Endpoint must be valid HTTP URL
            - Specify ONE model endpoint in JSON format.
            - Prefix determines model capabilities:
            * ERNIE-4.5-VL: Multimodal models (image+text)
            """,
    )
    parser.add_argument(
        "--web_search_service_url",
        type=str,
        default="https://qianfan.baidubce.com/v2/ai_search/chat/completions",
        help="Web Search Service URL.",
    )
    parser.add_argument(
        "--qianfan_api_key", type=str, default="bce-v3/xxx", help="Web Search Service API key.", required=True
    )

    args = parser.parse_args()
    try:
        args.model_map = json.loads(args.model_map)

        # Validation: Check at least one model exists
        if len(args.model_map) < 1:
            raise ValueError("model_map must contain at least one model configuration")
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON format for --model-map") from e
    return args


class GradioEvents:
    """
    Handles Gradio UI events and manages chatbot interactions including conversation flow and file processing.

    Provides methods for maintaining chat history, extracting text from files, and generating image URLs.
    Supports both text and multimodal interactions with web search integration when needed.

    Manages chatbot state including conversation history, file attachments and UI updates.
    Includes utilities for reading various file formats and handling streaming AI responses.
    """

    @staticmethod
    def get_history_conversation(task_history: list, image_history: dict, file_history: dict) -> tuple:
        """
        Constructs complete conversation history from stored components including text messages,
        attached files and images. Processes each dialogue turn by combining the raw query/response
        pairs with any associated multimedia attachments. For multimodal models, image URLs are
        formatted with base64 encoding while text files have their content extracted inline.

        Args:
            task_history (list): List of tuples containing user queries and responses.
            image_history (dict): Dictionary mapping indices to lists of image urls.
            file_history (dict): Dictionary mapping indices to lists of file urls.

        Returns:
            tuple: Tuple containing two elements:
                - conversation (list): List of dictionaries representing the conversation history.
                - conversation_str (str): String representation of the conversation history.
        """
        conversation = []
        conversation_str = ""
        for idx, (query_h, response_h) in enumerate(task_history):
            conversation_str += f"user:\n{query_h}\nassistant:\n{response_h}\n"
            if idx in file_history:
                for file_url in file_history[idx]:
                    query_h += f"参考资料[{idx + 1}]:\n资料来源：用户上传\n{GradioEvents.get_file_text(file_url)}\n"
            if idx in image_history:
                content = []
                for image_url in image_history[idx]:
                    content.append({"type": "image_url", "image_url": {"url": GradioEvents.get_image_url(image_url)}})
                content.append({"type": "text", "text": query_h})
                conversation.append({"role": "user", "content": content})
            else:
                conversation.append({"role": "user", "content": query_h})
            conversation.append({"role": "assistant", "content": response_h})
        return conversation, conversation_str

    @staticmethod
    def get_search_query(conversation: list, model_name: str, bot_client: BotClient) -> list:
        """
        Processes conversation history to generate search queries by sending the conversation context
        to the model and parsing its JSON response. Handles model output validation and extracts
        structured search queries containing query lists. Raises Gradio errors for
        invalid JSON responses from the model.

        Args:
            conversation (list): List of dictionaries representing the conversation history.
            model_name (str): Name of the model being used.
            bot_client (BotClient): An instance of BotClient.

        Returns:
            list: List of strings representing the search query.
        """
        req_data = {"messages": conversation}
        try:
            response = bot_client.process(model_name, req_data)
            search_query = response["choices"][0]["message"]["content"]
            start = search_query.find("{")
            end = search_query.rfind("}") + 1
            if start >= 0 and end > start:
                search_query = search_query[start:end]
            search_query = json.loads(search_query)
            return search_query
        except json.JSONDecodeError:
            logging.error("error: model output is not valid JSON format ")
            return None

    @staticmethod
    def process_files(
        diologue_turn: int,
        files_url: list,
        file_history: dict,
        image_history: dict,
        bot_client: BotClient,
        max_file_char: int,
    ):
        """
        Processes file URLs and generates input content for the model.
        Handles both text and image files by:
        1. For text files (PDF, TXT, MD, DOCX): extracts content and adds to file history with reference numbering
        2. For image files (PNG, JPEG, JPG): generates base64 encoded URLs for model input
        Maintains character limits for text references and ensures no duplicate file processing.

        Args:
            diologue_turn (int): Index of the current dialogue turn.
            files_url (list): List of uploaded file urls.
            file_history (dict): Dictionary mapping indices to lists of file urls.
            image_history (dict): Dictionary mapping indices to lists of image urls.
            bot_client (BotClient): An instance of BotClient.
            max_file_char (int): Maximum number of characters allowed for references.

        Returns:
            tuple: A tuple containing three elements:
                - input_content (list): List of dictionaries representing the input content.
                - file_contents (str): String representation of the file contents.
                - ref_file_num (int): Number of reference files added.
        """
        input_content = []
        file_contents = ""
        ref_file_num = 0
        if not files_url:
            return input_content, file_contents, ref_file_num

        for file_url in files_url:
            extionsion = "." + file_url.split(".")[-1]
            if extionsion in TEXT_FILE_TYPE and (
                len(file_history) == 0 or file_url not in list(file_history.values())[-1]
            ):
                file_history[diologue_turn] = file_history.get(diologue_turn, []) + [file_url]
                file_name = file_url.split("/")[-1]
                file_contents_words = bot_client.cut_chinese_english(file_contents)

                if len(file_contents_words) < max_file_char - 20:
                    ref_file_num += 1
                    file_content = (
                        f"\n参考资料[{len(file_history[diologue_turn])}]:\n资料来源："
                        + f"用户上传\n{file_name}\n{GradioEvents.get_file_text(file_url)}\n"
                    )
                    file_content_words = bot_client.cut_chinese_english(file_content)
                    max_char = min(len(file_content_words), max_file_char - len(file_contents_words))
                    file_content_words = file_content_words[:max_char]
                    file_contents += "".join(file_content_words) + "\n"
            elif extionsion in IMAGE_FILE_TYPE and (
                len(image_history) == 0 or file_url not in list(image_history.values())[-1]
            ):
                image_history[diologue_turn] = image_history.get(diologue_turn, []) + [file_url]
                input_content.append({"type": "image_url", "image_url": {"url": GradioEvents.get_image_url(file_url)}})
        return input_content, file_contents, ref_file_num

    @staticmethod
    async def chat_stream(
        query: str,
        task_history: list,
        image_history: dict,
        file_history: dict,
        model_name: str,
        files_url: list,
        search_state: bool,
        bot_client: BotClient,
        max_ref_char: int = 18000,
    ) -> dict:
        """
        Handles streaming chat queries with text and multimodal inputs.
        Builds conversation history with attachments, checks if web search
        is needed, and streams responses.

        Args:
            query (str): User input query string.
            task_history (list): List of tuples containing user queries and responses.
            image_history (dict): Dictionary mapping indices to lists of image urls.
            file_history (dict): Dictionary mapping indices to lists of file urls.
            model_name (str): Name of the model being used.
            files_url (list): List of uploaded file urls.
            search_state (bool): Whether to perform a search.
            bot_client (BotClient): An instance of BotClient.
            max_ref_char (int): Maximum number of characters allowed for references.

        Returns:
            dict: Dictionary containing the following keys:
                - "type": The message type.
                - "content": The content of the message.
        """
        conversation, conversation_str = GradioEvents.get_history_conversation(
            task_history, image_history, file_history
        )

        # Step 1: Determine whether a search is needed and obtain the corresponding query list
        search_info_res = {}
        if search_state:
            search_info_message = SEARCH_INFO_PROMPT.format(
                date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), context=conversation_str, query=query
            )
            search_conversation = [{"role": "user", "content": search_info_message}]
            search_info_res = GradioEvents.get_search_query(search_conversation, model_name, bot_client)
            if search_info_res is None:
                search_info_res = {"is_search": True, "query_list": [query]}

        # Process files
        diologue_turn = len(task_history)
        if search_info_res.get("is_search", False) and search_info_res.get("query_list", []):
            max_file_char = max_ref_char // 2
        else:
            max_file_char = max_ref_char
        input_content, file_contents, ref_file_num = GradioEvents.process_files(
            diologue_turn, files_url, file_history, image_history, bot_client, max_file_char
        )

        # Step 2: If a search is needed, obtain the corresponding query results
        if search_info_res.get("is_search", False) and search_info_res.get("query_list", []):
            search_result = bot_client.get_web_search_res(search_info_res["query_list"])

            max_search_result_char = max_ref_char - len(bot_client.cut_chinese_english(file_contents))
            complete_search_result = await GradioEvents.get_complete_search_content(
                ref_file_num, search_result, bot_client, max_search_result_char
            )
            complete_ref = file_contents + "\n" + complete_search_result

            query = ANSWER_PROMPT.format(
                date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                reference=complete_ref,
                context=conversation_str,
                query=query,
            )
            yield {"type": "search_result", "content": complete_ref}
        else:
            query += "\n" + file_contents

        # Step 3: Answer the user's query
        if image_history.get(diologue_turn, []):
            input_content.append({"type": "text", "text": query})
            conversation.append({"role": "user", "content": input_content})
        else:
            conversation.append({"role": "user", "content": query})

        try:
            req_data = {"messages": conversation}
            for chunk in bot_client.process_stream(model_name, req_data):
                if "error" in chunk:
                    raise Exception(chunk["error"])

                message = chunk.get("choices", [{}])[0].get("delta", {})
                content = message.get("content", "")

                if content:
                    yield {"type": "answer", "content": content}

        except Exception as e:
            raise gr.Error("Exception: " + repr(e))

    @staticmethod
    async def predict(
        query: str,
        chatbot: list,
        task_history: list,
        image_history: dict,
        file_history: dict,
        model: str,
        file_url: list,
        search_state: bool,
        bot_client: BotClient,
    ) -> tuple:
        """
        Processes user queries and generates responses through streaming interaction.
        Handles both text and file inputs, manages conversation history updates,
        and optionally performs web searches when enabled. Yields intermediate
        answers as they become available.

        Args:
            query (str): User input query string.
            chatbot (list): List of dictionaries representing the chatbot history.
            task_history (list): List of tuples containing user queries and responses.
            image_history (dict): Dictionary mapping indices to lists of image urls.
            file_history (dict): Dictionary mapping indices to lists of file urls.
            model (str): Name of the model being used.
            file_url (list): List of uploaded file urls.
            search_state (bool): Whether to perform a search.
            bot_client (BotClient): An instance of BotClient.

        Returns:
            tuple: Tuple containing two elements:
                - chatbot (list): Updated chatbot history after adding the user's query.
                - search_result (str): Search result obtained from the AI search service.
        """

        logging.info(f"User: {query}")
        # First yield the chatbot with user message
        chatbot.append({"role": "user", "content": query})
        yield chatbot, None

        response = ""
        search_result = None
        async for new_text in GradioEvents.chat_stream(
            query, task_history, image_history, file_history, model, file_url, search_state, bot_client
        ):
            if not isinstance(new_text, dict):
                continue

            if new_text.get("type") == "search_result":
                search_result = new_text["content"]
                yield chatbot, search_result
                continue
            elif new_text.get("type") == "answer":
                response += new_text["content"]

            # Remove previous message if exists
            if chatbot[-1].get("role") == "assistant":
                chatbot.pop(-1)

            if response:
                chatbot.append({"role": "assistant", "content": response})
                yield chatbot, search_result
                await asyncio.sleep(0)  # Wait to refresh

        logging.info(f"History: {task_history}")
        task_history.append((query, response))
        logging.info(f"ERNIE models: {response}")

    @staticmethod
    async def regenerate(
        chatbot: list,
        task_history: list,
        image_history: dict,
        file_history: dict,
        model: str,
        file_url: list,
        search_state: bool,
        bot_client: BotClient,
    ) -> tuple:
        """
        Regenerates the chatbot's last response by reprocessing the previous user query with current context.
        Maintains conversation continuity by preserving history while removing the last interaction,
        then reinvokes the prediction pipeline with identical parameters to generate a fresh response.

        Args:
            chatbot (list): List of dictionaries representing the chatbot history.
            task_history (list): List of tuples containing user queries and responses.
            image_history (dict): Dictionary mapping indices to lists of image urls.
            file_history (dict): Dictionary mapping indices to lists of file urls.
            model (str): Name of the model being used.
            file_url (list): List of uploaded file urls.
            search_state (bool): Whether to perform a search.
            bot_client (Botclient): An instance of BotClient.

        Returns:
            tuple: Tuple containing two elements:
                - chatbot (list): Updated chatbot history after removing the last user query and response.
                - search_result (str): Search result obtained from the AI search service.
        """
        if not task_history:
            yield chatbot, None
            return
        # Pop the last user query and bot response from task_history
        item = task_history.pop(-1)
        dialogue_turn = len(task_history)
        if (dialogue_turn) in image_history:
            del image_history[dialogue_turn]
        if (dialogue_turn) in file_history:
            del file_history[dialogue_turn]
        while len(chatbot) != 0 and chatbot[-1].get("role") == "assistant":
            chatbot.pop(-1)
        chatbot.pop(-1)

        async for chunk, search_result in GradioEvents.predict(
            item[0], chatbot, task_history, image_history, file_history, model, file_url, search_state, bot_client
        ):
            yield chunk, search_result

    @staticmethod
    def reset_user_input() -> gr.update:
        """
        Reset user input box content.

        Returns:
            gr.update: Update object indicating that the value should be set to an empty string
        """
        return gr.update(value="")

    @staticmethod
    def reset_state() -> namedtuple:
        """
        Reset the state of the chatbot.

        Returns:
            namedtuple: A namedtuple containing the following fields:
                - chatbot (list): Empty list
                - task_history (list): Empty list
                - image_history (dict): Empty dictionary
                - file_history (dict): Empty dictionary
                - file_btn (gr.update): Value set to None
                - search_result (gr.update): Value set to None
        """
        GradioEvents.gc()

        reset_result = namedtuple(
            "reset_result", ["chatbot", "task_history", "image_history", "file_history", "file_btn", "search_result"]
        )
        return reset_result(
            [],  # clear chatbot
            [],  # clear task_history
            {},  # clear image_history
            {},  # clear file_history
            gr.update(value=None),  # clear file_btn
            gr.update(value=None),  # reset search_result
        )

    @staticmethod
    def gc():
        """Run garbage collection."""
        import gc

        gc.collect()

    @staticmethod
    def search_toggle_state(search_state: bool) -> bool:
        """
        Toggle search state between enabled and disabled.

        Args:
            search_state (bool): Current search state

        Returns:
            bool: New search result visible state
        """
        return gr.update(visible=search_state)

    @staticmethod
    def get_image_url(image_path: str) -> str:
        """
        Encode image file to Base64 format and generate data URL.
        Reads an image file from disk, encodes it as Base64, and formats it
        as a data URL that can be used directly in HTML or API requests.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The URL of the image file.
        """
        base64_image = ""
        extension = image_path.split(".")[-1]
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        url = f"data:image/{extension};base64,{base64_image}"
        return url

    @staticmethod
    def get_file_text(file_path: str) -> str:
        """
        Get the contents of a file as plain text.

        Args:
            file_path (str): The path to the file to read.

        Returns:
            str: The contents of the file as plain text.
        """
        if file_path is None:
            return ""
        if file_path.endswith(".pdf"):
            return GradioEvents.read_pdf(file_path)
        elif file_path.endswith(".docx"):
            return GradioEvents.read_docx(file_path)
        elif file_path.endswith(".txt") or file_path.endswith(".md"):
            return GradioEvents.read_txt_md(file_path)
        else:
            return ""

    @staticmethod
    def read_pdf(pdf_path: str) -> str:
        """
        Extracts text content from a PDF file using pdfplumber library. Processes each page sequentially
        and concatenates all extracted text. Handles potential extraction errors gracefully by returning
        an empty string and logging the error.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: Text extracted from the PDF file.
        """
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            logging.info(f"Error reading PDF file: {e}")
            return ""

    @staticmethod
    def read_docx(file_path: str) -> str:
        """
        Extracts text content from a DOCX file using python-docx library. Processes all paragraphs
        sequentially and joins them with newline characters. Handles potential file reading errors
        gracefully by returning an empty string and logging the error.

        Args:
            file_path (str): Path to the DOCX file.

        Returns:
            str: Text extracted from the DOCX file.
        """
        try:
            doc = Document(file_path)
            full_text = []
            for paragraph in doc.paragraphs:
                full_text.append(paragraph.text)
            return "\n".join(full_text)
        except Exception as e:
            logging.info(f"Error reading DOCX file: {e}")
            return ""

    @staticmethod
    def read_txt_md(file_path: str) -> str:
        """
        Read a TXT or MD file and extract its text content.

        Args:
            file_path (str): Path to the TXT or MD file.

        Returns:
            str: Text extracted from the TXT or MD file.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logging.info(f"Error reading TXT or MD file: {e}")
            return ""

    @staticmethod
    async def get_complete_search_content(
        ref_file_num: int, search_results: list, bot_client: BotClient, max_search_results_char
    ) -> str:
        """
        Combines and formats multiple search results into a single string.
        Processes each result, extracts URLs, crawls content, and enforces length limits.

        Args:
            ref_file_num (int): Reference file number
            search_results (list): List of search results
            bot_client (BotClient): Chatbot client instance
            max_search_results_char (int): Maximum character length of each search result

        Returns:
            str: Complete search content string
        """
        results = []
        crawl_utils = CrawlUtils()
        for search_res in search_results:
            for item in search_res:
                new_content = await crawl_utils.get_webpage_text(item["url"])
                if not new_content:
                    continue
                item_text = "Title: {title} \nURL: {url} \nContent:\n{content}\n".format(
                    title=item["title"], url=item["url"], content=new_content
                )

                # Truncate the search result to max_search_results_char characters
                search_res_words = bot_client.cut_chinese_english(item_text)
                res_words = bot_client.cut_chinese_english("".join(results))
                if len(search_res_words) + len(res_words) > max_search_results_char:
                    break

                results.append(
                    f"参考资料[{len(results) + 1 + ref_file_num}]:\n" + f"资料来源：素材检索\n{item_text}\n"
                )

        return "".join(results)


def launch_demo(args: argparse.Namespace, bot_client: BotClient):
    """
    Launch demo program
    Args:
        args (argparse.Namespace): argparse Namespace object containing parsed command line arguments
        bot_client (BotClient): Bot client instance
    """
    css = """
    .input-textbox textarea {
        height: 200px !important;
    }
    #file-upload {
        height: 247px !important;
        min-height: 247px !important;
        max-height: 247px !important;
    }
    /* Hide original Chinese text */
    #file-upload .wrap {
        font-size: 0 !important;
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    /* Insert English prompt text below the SVG icon */
    #file-upload .wrap::after {
        content: "Drag and drop files here or click to upload";
        font-size: 18px;
        color: #555;
        margin-top: 8px;
        white-space: nowrap;
    }
    """

    with gr.Blocks(css=css) as demo:
        logo_url = GradioEvents.get_image_url("assets/logo.png")
        gr.Markdown(
            f"""\
                <p align="center"><img src="{logo_url}" \
                style="height: 60px"/><p>"""
        )
        gr.Markdown(
            """\
<center><font size=3>This demo is based on ERNIE models. \
(本演示基于文心大模型实现。)</center>"""
        )

        chatbot = gr.Chatbot(label="ERNIE", elem_classes="control-height", type="messages")

        search_result = gr.Textbox(label="Search Result", lines=10, max_lines=10, visible=False)

        with gr.Row():
            search_check = gr.Checkbox(label="🌐 Search the web(联网搜索)")

        with gr.Row():
            query = gr.Textbox(label="Input", lines=1, scale=6, elem_classes="input-textbox")
            file_btn = gr.File(
                label="File upload (Accepted formats: PNG, JPEG, JPG, PDF, TXT, MD, DOC, DOCX)",
                scale=4,
                elem_id="file-upload",
                file_types=IMAGE_FILE_TYPE + TEXT_FILE_TYPE,
                file_count="multiple",
            )

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History(清除历史)")
            submit_btn = gr.Button("🚀 Submit(发送)")
            regen_btn = gr.Button("🤔️ Regenerate(重试)")

        task_history = gr.State([])
        image_history = gr.State({})
        file_history = gr.State({})
        model_name = gr.State(next(iter(args.model_map.keys())))

        search_check.change(fn=GradioEvents.search_toggle_state, inputs=search_check, outputs=search_result)

        predict_with_clients = partial(GradioEvents.predict, bot_client=bot_client)
        regenerate_with_clients = partial(GradioEvents.regenerate, bot_client=bot_client)
        query.submit(
            predict_with_clients,
            inputs=[query, chatbot, task_history, image_history, file_history, model_name, file_btn, search_check],
            outputs=[chatbot, search_result],
            show_progress=True,
        )
        query.submit(GradioEvents.reset_user_input, [], [query])
        submit_btn.click(
            predict_with_clients,
            inputs=[query, chatbot, task_history, image_history, file_history, model_name, file_btn, search_check],
            outputs=[chatbot, search_result],
            show_progress=True,
        )
        submit_btn.click(GradioEvents.reset_user_input, [], [query])
        empty_btn.click(
            GradioEvents.reset_state,
            outputs=[chatbot, task_history, image_history, file_history, file_btn, search_result],
            show_progress=True,
        )
        regen_btn.click(
            regenerate_with_clients,
            inputs=[chatbot, task_history, image_history, file_history, model_name, file_btn, search_check],
            outputs=[chatbot, search_result],
            show_progress=True,
        )

    demo.queue().launch(server_port=args.server_port, server_name=args.server_name)


def main():
    """Main function that runs when this script is executed."""
    args = get_args()
    bot_client = BotClient(args)
    launch_demo(args, bot_client)


if __name__ == "__main__":
    main()
