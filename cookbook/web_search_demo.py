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

"""
This script provides a Gradio interface for interacting with a web search-powered chatbot
with live web search functionality.
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import textwrap
from argparse import ArgumentParser
from datetime import datetime
from functools import partial

import gradio as gr
from bot_requests import BotClient
from crawl_utils import CrawlUtils

os.environ["NO_PROXY"] = "localhost,127.0.0.1"  # Disable proxy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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
    {search_result}

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
    Configures and parses command line arguments for the web chat demo application.
    Handles server settings, model endpoints, and operational parameters.

    Returns:
        args: Parsed command line arguments object.
    """
    parser = ArgumentParser(description="ERNIE models web chat demo.")

    parser.add_argument(
        "--server-port", type=int, default=8969, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="0.0.0.0", help="Demo server name."
    )
    parser.add_argument(
        "--max_char",
        type=int,
        default=20000,
        help="Maximum character limit for messages.",
    )
    parser.add_argument(
        "--max_retry_num", type=int, default=3, help="Maximum retry number for request."
    )
    parser.add_argument(
        "--model_map",
        required=True,
        type=str,
        default='{"ERNIE-4.5": "http://localhost:port/v1"}',
        help="""JSON string defining model name to endpoint mappings.
            Required Format:
            {"ERNIE-4.5": "http://localhost:port/v1"}

            Note:
            - Endpoint must be valid HTTP URL
            - Specify ONE model endpoint in JSON format.
            - Prefix determines model capabilities:
            * ERNIE-4.5: Text-only model
            """,
    )
    parser.add_argument(
        "--web_search_service_url",
        type=str,
        default="https://qianfan.baidubce.com/v2/ai_search/chat/completions",
        help="Web Search Service URL.",
    )
    parser.add_argument(
        "--qianfan_api_key",
        type=str,
        default="bce-v3/xxx",
        help="QianFan API Key.",
        required=True,
    )
    parser.add_argument(
        "--max_crawler_threads",
        type=int,
        default=10,
        help="The maximum number of concurrent crawler threads.",
    )
    parser.add_argument(
        "--concurrency_limit", type=int, default=10, help="Default concurrency limit."
    )
    parser.add_argument(
        "--max_queue_size", type=int, default=50, help="Maximum queue size for request."
    )

    args = parser.parse_args()
    try:
        args.model_map = json.loads(args.model_map)

        # Validation: Check at least one model exists
        if len(args.model_map) < 1:
            raise ValueError("model_map must contain at least one model configuration")
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON format for --model_map") from e
    return args


class GradioEvents:
    """
    Handles all Gradio UI events and interactions for the chatbot demo.
    Manages conversation flow, search functionality, and response generation.
    """

    @staticmethod
    def get_history_conversation(task_history: list) -> tuple:
        """
        Converts task history into conversation format for model processing.
        Transforms query-response pairs into structured message history and plain text.

        Args:
            task_history (list): List of tuples containing queries and responses.

        Returns:
            tuple: Tuple containing two elements:
                - conversation (list): List of dictionaries representing the conversation history.
                - conversation_str (str): String representation of the conversation history.
        """
        conversation = []
        conversation_str = ""
        for query_h, response_h in task_history:
            conversation.append({"role": "user", "content": query_h})
            conversation.append({"role": "assistant", "content": response_h})
            conversation_str += f"user:\n{query_h}\nassistant:\n{response_h}\n"
        return conversation, conversation_str

    @staticmethod
    def get_search_query(
        conversation: list, model_name: str, bot_client: BotClient
    ) -> dict:
        """
        Determines if a web search is needed by analyzing conversation context.
        Processes model response to extract structured search decision and queries.

        Args:
            conversation (list): List of dictionaries representing the conversation history.
            model_name (str): Name of the model being used.
            bot_client (BotClient): Instance of BotClient.

        Returns:
            dict: Dictionary containing the search query information.
        """
        req_data = {"messages": conversation}
        try:
            response = bot_client.process(model_name, req_data)
            search_info_res = response["choices"][0]["message"]["content"]
            start = search_info_res.find("{")
            end = search_info_res.rfind("}") + 1
            if start >= 0 and end > start:
                search_info_res = search_info_res[start:end]
            search_info_res = json.loads(search_info_res)
            if search_info_res.get("query_list", []):
                unique_list = list(set(search_info_res["query_list"]))
                search_info_res["query_list"] = unique_list
            return search_info_res
        except json.JSONDecodeError:
            logging.error("error: model output is not valid JSON format ")
            return None

    @staticmethod
    async def chat_stream(
        query: str,
        task_history: list,
        model_name: str,
        search_state: bool,
        max_crawler_threads: int,
        bot_client: BotClient,
    ) -> dict:
        """
        Orchestrates the chatbot conversation flow with optional web search integration.
        Handles three key steps: search determination, search execution, and response generation.

        Args:
            query (str): User's query string.
            task_history (list): Task history list.
            model_name (str): Model name.
            search_state (bool): Searching state.
            max_crawler_threads (int): Maximum number of concurrent crawler threads.
            bot_client (BotClient): Bot client instance.

        Yields:
            dict: A dictionary containing the event type and its corresponding content.
        """
        conversation, conversation_str = GradioEvents.get_history_conversation(
            task_history
        )

        # Step 1: Determine whether a search is needed and obtain the corresponding query list
        search_info_res = {}
        if search_state:
            search_info_message = SEARCH_INFO_PROMPT.format(
                date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                context=conversation_str,
                query=query,
            )
            search_conversation = [{"role": "user", "content": search_info_message}]
            search_info_res = GradioEvents.get_search_query(
                search_conversation, model_name, bot_client
            )
            if search_info_res is None:
                search_info_res = {"is_search": True, "query_list": [query]}

        # Step 2: If a search is needed, obtain the corresponding query results
        if search_info_res.get("is_search", False) and search_info_res.get(
            "query_list", []
        ):
            yield {"type": "search_result", "content": "🧐 努力搜索中... ✨"}
            search_result = bot_client.get_web_search_res(search_info_res["query_list"])

            complete_search_result = await GradioEvents.get_complete_search_content(
                search_result, max_crawler_threads, bot_client
            )

            if complete_search_result:
                query = ANSWER_PROMPT.format(
                    date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    search_result=complete_search_result,
                    context=conversation_str,
                    query=query,
                )
            yield {"type": "search_result", "content": complete_search_result}

        # Step 3: Answer the user's query
        content = []
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
        model: str,
        search_state: bool,
        max_crawler_threads: int,
        bot_client: BotClient,
    ) -> list:
        """
        Handles the complete chatbot interaction from user input to response.
        Manages message display, streaming responses, optional web search, and conversation history.
        Updates UI in real-time and stores final conversation state.

        Args:
            query (str): The content of the user's input query.
            chatbot (list): The chatbot's historical message list.
            task_history (list): The task history record list.
            model (Model): The model used to generate responses.
            search_state (bool): The searching state of the chatbot.
            max_crawler_threads (int): The maximum number of concurrent crawler threads.
            bot_client (object): The chatbot client object.

        Yields:
            list: The chatbot's response list.
        """
        logging.info(f"User: {query}")
        # First yield the chatbot with user message
        chatbot.append({"role": "user", "content": query})
        yield chatbot, "🛠️ 正在解析问题意图，判断是否需要搜索... 🔍"
        await asyncio.sleep(0.05)  # Wait to refresh

        content = ""
        search_result = None
        async for new_text in GradioEvents.chat_stream(
            query, task_history, model, search_state, max_crawler_threads, bot_client
        ):
            if not isinstance(new_text, dict):
                continue

            if new_text.get("type") == "search_result":
                search_result = new_text["content"]
                yield chatbot, search_result
                continue
            elif new_text.get("type") == "answer":
                content += new_text["content"]

            # Remove previous message if exists
            if chatbot[-1].get("role") == "assistant":
                chatbot.pop(-1)

            if content:
                chatbot.append({"role": "assistant", "content": content})
                yield chatbot, search_result
                await asyncio.sleep(0)  # Wait to refresh

        logging.info(f"History: {task_history}")
        task_history.append((query, content))
        logging.info(f"ERNIE models: {content}")

    @staticmethod
    async def regenerate(
        chatbot: list,
        task_history: list,
        model: str,
        search_state: bool,
        max_crawler_threads: int,
        bot_client: BotClient,
    ) -> tuple:
        """
        Regenerate the chatbot's response based on the latest user query.

        Args:
            chatbot (list): The chatbot's historical message list.
            task_history (list): The task history record list.
            model (Model): The model used to generate responses.
            search_state (bool): The searching state of the chatbot.
            max_crawler_threads (int): The maximum number of concurrent crawler threads.
            bot_client (object): The chatbot client object.

        Yields:
            list: The chatbot's response list.
        """
        if not task_history:
            yield chatbot, None
            return
        # Pop the last user query and bot response from task_history
        item = task_history.pop(-1)
        while len(chatbot) != 0 and chatbot[-1].get("role") == "assistant":
            chatbot.pop(-1)
        chatbot.pop(-1)

        async for chunk, search_result in GradioEvents.predict(
            item[0],
            chatbot,
            task_history,
            model,
            search_state,
            max_crawler_threads,
            bot_client,
        ):
            yield chunk, search_result

    @staticmethod
    def reset_user_input() -> dict:
        """
        Reset user input box content.

        Returns:
            dict: Dictionary containing updated input box value for Gradio's update method
        """
        return gr.update(value="")

    @staticmethod
    def reset_state() -> tuple:
        """
        Reset chat state and clear all history.

        Returns:
            tuple: Updated chatbot, task history, and search result
        """
        GradioEvents.gc()
        return [], [], ""

    @staticmethod
    def gc():
        """Run garbage collection to free up memory."""
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
            image_path (str): Path to the image file

        Returns:
            str: Image URL
        """
        base64_image = ""
        extension = image_path.split(".")[-1]
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        url = f"data:image/{extension};base64,{base64_image}"
        return url

    @staticmethod
    async def get_complete_search_content(
        search_results: list,
        max_crawler_threads: int,
        bot_client: BotClient,
        max_search_results_char: int = 18000,
    ) -> str:
        """
        Combines and formats multiple search results into a single string.
        Processes each result, extracts URLs, crawls content, and enforces length limits.

        Args:
            search_results (list): List of search results
            max_crawler_threads (int): Maximum number of concurrent crawler threads
            bot_client (BotClient): Chatbot client instance
            max_search_results_char (int): Maximum character length of each search result

        Returns:
            str: Complete search content string
        """
        results = []
        crawl_utils = CrawlUtils()

        items_to_crawl = []
        for search_res in search_results:
            for item in search_res:
                items_to_crawl.append(item)

        # Create a semaphore to limit concurrent crawls
        semaphore = asyncio.Semaphore(max_crawler_threads)

        async def crawl_with_semaphore(url):
            async with semaphore:
                return await crawl_utils.get_webpage_text(url)

        # Crawl all webpages with limited concurrency
        crawl_tasks = [crawl_with_semaphore(item["url"]) for item in items_to_crawl]
        crawled_contents = await asyncio.gather(*crawl_tasks, return_exceptions=True)

        # Process crawled contents
        for item, new_content in zip(items_to_crawl, crawled_contents):
            if not new_content or isinstance(new_content, Exception):
                continue

            item_text = "Title: {title} \nURL: {url} \nContent:\n{content}\n".format(
                title=item["title"], url=item["url"], content=new_content
            )

            # Truncate the search result to max_search_results_char characters
            search_res_words = bot_client.cut_chinese_english(item_text)
            res_words = bot_client.cut_chinese_english("".join(results))
            if len(res_words) >= max_search_results_char:
                break
            elif len(search_res_words) + len(res_words) > max_search_results_char:
                max_char = max_search_results_char - len(res_words)
                print(f"max_char: {max_char}\n")
                search_res_words = search_res_words[:max_char]
                item_text = "".join(search_res_words)

            results.append(f"\n参考资料[{len(results) + 1}]:\n{item_text}\n")

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
        gr.Markdown(
            """\
<center><font size=3>    <a href="https://ernie.baidu.com/">ERNIE Bot</a> | \
<a href="https://github.com/PaddlePaddle/ERNIE">GitHub</a> | \
<a href="https://huggingface.co/baidu">Hugging Face</a> | \
<a href="https://aistudio.baidu.com/modelsoverview">BAIDU AI Studio</a> | \
<a href="https://yiyan.baidu.com/blog/publication/">Technical Report</a></center>"""
        )

        chatbot = gr.Chatbot(
            label="ERNIE", elem_classes="control-height", type="messages"
        )

        search_result = gr.Textbox(
            label="Search Result", lines=10, max_lines=10, visible=True
        )

        search_check = gr.Checkbox(
            label="🌐 Search the web(联网搜索)", value=True, interactive=True
        )

        with gr.Row():
            query = gr.Textbox(
                label="Input", lines=1, scale=6, elem_classes="input-textbox"
            )

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History(清除历史)")
            submit_btn = gr.Button("🚀 Submit(发送)")
            regen_btn = gr.Button("🤔️ Regenerate(重试)")

        task_history = gr.State([])
        model_name = gr.State(next(iter(args.model_map.keys())))
        max_crawler_threads = gr.State(args.max_crawler_threads)

        search_check.change(
            fn=GradioEvents.search_toggle_state,
            inputs=search_check,
            outputs=search_result,
        )

        predict_with_clients = partial(GradioEvents.predict, bot_client=bot_client)
        regenerate_with_clients = partial(
            GradioEvents.regenerate, bot_client=bot_client
        )
        query.submit(
            predict_with_clients,
            inputs=[
                query,
                chatbot,
                task_history,
                model_name,
                search_check,
                max_crawler_threads,
            ],
            outputs=[chatbot, search_result],
            show_progress=True,
        )
        query.submit(GradioEvents.reset_user_input, [], [query])
        submit_btn.click(
            predict_with_clients,
            inputs=[
                query,
                chatbot,
                task_history,
                model_name,
                search_check,
                max_crawler_threads,
            ],
            outputs=[chatbot, search_result],
            show_progress=True,
        )
        submit_btn.click(GradioEvents.reset_user_input, [], [query])
        empty_btn.click(
            GradioEvents.reset_state,
            outputs=[chatbot, task_history, search_result],
            show_progress=True,
        )
        regen_btn.click(
            regenerate_with_clients,
            inputs=[
                chatbot,
                task_history,
                model_name,
                search_check,
                max_crawler_threads,
            ],
            outputs=[chatbot, search_result],
            show_progress=True,
        )

    demo.queue(
        default_concurrency_limit=args.concurrency_limit, max_size=args.max_queue_size
    )
    demo.launch(server_port=args.server_port, server_name=args.server_name)


def main():
    """Main function that runs when this script is executed."""
    args = get_args()
    bot_client = BotClient(args)
    launch_demo(args, bot_client)


if __name__ == "__main__":
    main()
