"""
Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import base64
import json
import logging
import os
import textwrap
from argparse import ArgumentParser
from typing import List

import gradio as gr
from appbuilder.mcp_server.client import MCPClient
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
logging.root.setLevel(logging.INFO)

TASK_PLAN_PROMPT = textwrap.dedent(
    """\
    # 任务
    我是一个专业的研究员。我会根据给定的query需求，逐层深入分析，并以清晰易懂的语言组织成一份完整的研究方案规划，便于执行和实施。
    ## 深度研究参考思路
    1. 目标确认与基础信息收集
        （1）确定研究目的：收集相关领域的信息，明确研究背景、细化通过研究想要达成的具体成果
        （2）数据收集与整理：我会系统收集目标领域的历史数据和案例，将这些信息整理成标准化的内容，可选择用数据表格形态整理。关键是确保数据的完整性、准确性和时序性，为后续所有分析提供可靠的事实基础，数据收集必须覆盖足够的时间范围\
    ，包含所有相关的关键信息字段。
    2. 深度分析与信息深度挖掘
        （1）深度模式分析：基于收集到的数据，我会深入分析其中的新研究对象、关键模式、规律和趋势等。这包括频率统计、周期性变化、发展趋势等量化分析，目标是揭示隐藏在数据背后的内在逻辑和规律性特征。对于上一步中出现的新的重要概念或\
    实体，需对该类需要探究的内容进行二次信息搜集。分析结果尽可能用统计数据和可视化图表来呈现。
        （2）核心驱动因素提取：通过对模式的深度分析，我需要识别出真正影响结果的核心驱动因素。这些因素需要按照影响力大小进行排序，并评估各自的权重。重点是找到那些具有决定性作用的关键变量，而不是表面的相关性因素。
        （3）现实背景信息补强：针对已识别的核心驱动因素，我会收集当前相关的现实背景信息。这包括最新的政策变化、市场环境、技术发展、社会趋势等可能影响分析结果的现实因素。目标是将历史规律与当前实际情况相结合，确保分析的时效性和准\
    确性。
    3. 研究成果产出
        （1）综合推理与结论：最后将整合上述获取到的所有信息，运用严密的逻辑推理得出最终结论。这不仅包括具体的预测结果，还要包含完整的推理过程、逻辑链条、以及对结论可信度的评估。同时需要考虑可能存在的风险因素和不确定性。
        （2）产出符合需求的交付形态：根据输入query需求或场景通用的形态，组织最终研究成果或调研报告，需包含上述研究过程的各类细节信息，
    *以上思路步骤仅供参考，可根据实际需求或命题情况进行调整


    ## query示例及样例输出：
    **输入query1：**
    "请你深入研究基于事件的光流估计，梳理经典代表算法与最新的研究趋势，分析当前这个领域的瓶颈及可行的解决方案"

    **样例输出1：**
    1. 资料收集
        (1) 调研基于事件的视觉传感器（事件相机）的工作原理、特点及其与传统帧式相机的区别。
        (2) 梳理基于事件的光流估计领域的经典代表算法，包括但不限于基于事件匹配、基于事件累积和基于学习的方法。
    2. 深入研究并梳理趋势并识别挑战
        (1) 深入分析基于事件的光流估计的最新研究趋势，例如与深度学习的结合、多模态融合以及在特定应用场景（如高速运动、低光照）下的进展。
        (2) 识别并分析当前基于事件的光流估计领域面临的主要瓶颈和挑战，例如数据稀疏性、噪声敏感性、计算效率、泛化能力以及缺乏大规模标准数据集。
    3. 瓶颈和方案探索
        (1) 针对上述瓶颈和挑战，探讨可行的解决方案和未来研究方向，例如新型事件表示方法、更鲁棒的算法设计、高效的硬件加速以及跨领域知识迁移。
        (2) 查找并分析基于事件的光流估计在实际应用中的案例，例如机器人导航、自动驾驶或工业检测，以了解其应用前景和实际需求。
    4. 报告撰写与交付
        （1）将研究成果整理成深入的研究报告，包括研究过程、结论和建议

    **输入query2：**
    "请帮我梳理一下面向AIGC/扩散模型/大模型训练的数据集所有权保护方法，并分析当前的挑战，给出一些创新性的思路方案"
    **样例输出2：**
    1. 信息收集和整理
        (1) 梳理当前适用于人工智能训练数据所有权和知识产权保护的法律和技术框架。
        (2) 深入研究专门针对AIGC、扩散模型和大型模型训练数据所有权保护的现有方法，例如数据水印、溯源技术、加密技术和合同协议。
    2. 分析和方案制定
        (1) 分析在大型AI模型训练中实施和执行这些数据所有权保护方法所面临的挑战，包括技术限制、法律模糊性及实际操作困难。
        (2) 查找并评估与AI模型训练数据所有权保护相关的实际案例或争议，以了解其成功和失败的经验。
        (3) 探索新兴技术和研究方向，如区块链、先进密码学或新型数字版权管理，它们如何为AI数据所有权保护提供创新解决方案。
        (4) 思考在AI数据所有权领域可能出现的政策或监管创新，例如新的许可模式、集体权利管理或国际合作框架。
    3. 报告撰写
        (1) 综合上述研究，提出结合法律、技术和伦理考量的创新性概念框架或混合方案，以实现AIGC/大型模型训练数据的稳健所有权保护。

    ## 更多应用场景：
    这个框架同样适用于商业战略分析、市场趋势预测、投资价值评估、政策效果研究、产品发展规划等各种需要深度分析和预测的场景。无论是分析企业发展策略、预测行业发展趋势，还是评估投资机会，都会按照同样的五层逻辑进行系统化的深度研究。

    ## 要求
    我不会对输入query作出任何评价，也不会解释相关的动作，我只会生成方案规划，一定不要直接回复输入query。
    我输出的规划是按步骤顺序组织的整体思路，每个步骤的规划用一两句话或一两个列表元素表示具体步骤，每个步骤的标题要严格和输入query相关，不要简单重复上述参考思路的小标题。
    但对于一些简单的问题，不用输出详细的规划步骤（也不需要给出规划示例）；如果遇到一些困难的问题，我可能会更进一步地扩展深度研究的要求。

    ## 当前输入query：{query}
    """
)

SEARCH_INFO_PROMPT = textwrap.dedent(
    """\
    现在需要你完成以下 **[深度研究]任务**：
    **{query}**
    为了完成该 [深度研究] 任务，你需要通过**动态、多次地调用检索工具**，以获取足够且相关的 [参考信息](工具返回的搜索结果)。
    [参考信息] 必须满足以下两个要求：
    1. 全面覆盖任务问题中的各个细节需求；
    2. 对前序 [参考信息] 中涉及的延伸概念进行进一步的具体说明。

    在信息量不足或理解尚不充分的情况下，请持续调用工具补充信息，直到所收集的信息足以支撑对任务问题的完整理解和回答为止。

    你应当根据**历史检索结果**，判断下一步搜索的方向和重点，并决定是否需要继续调用工具。这一过程应具有**自适应性和递进性**。

    请严格参照以下任务规划指导，辅助你进行任务执行：
    ```
    {task_plan}
    ```
    历史检索结果：
    ```
    {search_results_string}
    ```

    请以JSON格式输出，格式要求如下：
    {{
        "reasoning": "决策理由",
        "need_more_tool_info": true/false
    }}
    注意：need_more_tool_info为true表示需要调用工具获取更多信息，若任务过于简单或需要向用户澄清获取信息时则为false。
    """
)

GEN_QUERY_PROMPT = textwrap.dedent(
    """\
    可用工具信息：
    {available_tools}

    历史检索结果：
    ```
    {search_results_string}
    ```
    用户问题：{query}

    请严格参照以下任务规划指导，执行下一步工具调用：
    ```
    {task_plan}
    ```

    请根据以上历史检索结果判断回复用户问题所缺少的信息内容，或根据任务规划指导判断下一步需要完成的检索任务，调用合适的工具，生成新的检索query来补充所缺少的信息或需要执行的其他检索任务.
    回复用户问题所需的完整参考信息需包含对用户问题中各细节需求及信息，以及参考信息中与需求相关的延伸概念的具体说明，同时需要完成任务规划中指导的各检索任务。
    生成检索query的要求如下：
    - 生成的检索query要与历史检索结果中的检索词不同，生成更深入的信息补充类query。
    - 单个检索的query需独立为一个简单直接的搜索主题，不能过长或过于复杂，若有多个需要补充的信息，则拆分成多个query，通过json列表形式分成多次工具调用。

    以下为生成本轮query的示例1：
    ```
    用户问题：帮我做竞品调研，深入了解当前Agent产品在旅行场景下如何识别、响应和解决用户的具体需求，包括需求识别的准确性、解决方案的有效性，以及在满足用户需求过程中面临的挑战。请结合当前市面上的产品和AI能力帮我输出详细的调研报告。
    上一轮检索query为："旅行场景 AI Agent 产品 竞品"
    上一轮得的检索结果包括以下内容："Manus"、"飞猪问一问"、"字节豆包AI"等Agent产品介绍
    本轮生成query则应根据上一轮检索结果进行下一步有关产品细节的深度挖掘，生成如下检索query：
    - Manus 需求应用案例
    - 飞猪问一问 需求应用案例
    - 字节豆包AI 需求应用案例
    ```

    以下为生成本轮query的示例2：
    ```
    用户问题：帮我做竞品调研，深入了解当前Agent产品在旅行场景下如何识别、响应和解决用户的具体需求，包括需求识别的准确性、解决方案的有效性，以及在满足用户需求过程中面临的挑战。请结合当前市面上的产品和AI能力帮我输出详细的调研报告。
    上一轮检索query为："Manus 需求方案 应用案例"、 "飞猪问一问 需求方案 应用案例"、"字节豆包AI 需求方案 应用案例"
    上一轮得的检索结果包括以下内容："Manus相关产品方案和案例"、"飞猪问一问应用案例"、"字节豆包AI产品方案"等产品案例内容
    本轮生成query则应根据上一轮检索结果判断，缺少有关产品满足用户需求中面临的挑战相关内容，因此进行下一步检索，生成如下检索query：
    - Manus 满足需求面临的挑战
    - 飞猪问一问 满足需求面临的挑战
    - 字节豆包AI 满足需求面临的挑战
    ```

    请根据用户问题和历史检索结果中的结果，参考任务规划指导进行工具调用，并参照上述示例生成恰当的参数，直接以JSON格式输出需要调用的工具名称（tool_name）和工具调用参数（tool_args）。
    """
)

FINAL_ANSWER_PROMPT = textwrap.dedent(
    """\
    参考信息：{reference_results_string}
    用户问题：{query}
    请参考以上信息回复用户问题，需遵循以下要求：
    1. 结合问题需求，对参考信息中的检索内容进行可用性判断，避免在回复中使用错误或不恰当的信息；
    2. 优先根据官网、百科、权威机构、专业网站等高权威性来源的信息来回答问题，更多地参考其中的相关数字和案例。
    3. 回复内容需覆盖用户需求的各个细节，尽可能的全面的解答问题，输出内容尽可能详细且结构化，按照用户需求的格式（如有）组织输出形式。
    """
)


def get_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        ArgumentParser: Parsed command line arguments object.
    """
    parser = ArgumentParser(description="Yiyan web chat demo.")

    parser.add_argument("--server-port", type=int, default=8733, help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Demo server name.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="ERNIE-4.5",
        help="Model name. Prefix determines model capabilities: ERNIE-4.5: Text-only model",
        required=True,
    )
    parser.add_argument(
        "--model_url",
        type=str,
        default="http://localhost:port/v1",
        help="Url of model endpoint. It must be valid HTTP URL.",
        required=True,
    )
    parser.add_argument("--max_chars", type=int, default=25000, help="Maximum character limit for messages.")
    return parser.parse_args()


def chat(
    query: str, task_history: list, model: str, max_tokens: int = 2048, temperature: float = 1.0, top_p: float = 0.7
) -> str:
    """Process chat request and generate response.

    Args:
        query: User input query.
        model: Model name to use.
        max_tokens: Maximum tokens for response generation.
        temperature: Controls randomness of generated responses.
        top_p: Controls diversity of generated responses.

    Returns:
        str: response content.
    """
    args = get_args()
    model_url = args.model_url
    model_name = args.model_name

    conversation = []
    for query_h, response_h in task_history:
        conversation.append({"role": "user", "content": query_h})
        conversation.append({"role": "assistant", "content": response_h})
    conversation.append({"role": "user", "content": query})
    try:
        client = OpenAI(base_url=model_url, api_key=os.getenv('api_key'))
        response_text = client.chat.completions.create(
            model=model_name, messages=conversation, temperature=temperature, max_tokens=max_tokens, top_p=top_p
        )
        response_text = response_text.model_dump()
        logging.info(f"Model Response text: {response_text}")
        content_text = response_text["choices"][0]["message"]["content"]
        return content_text
    except Exception as e:
        raise gr.Error("Exception: " + repr(e))


class ChatInterface:
    """Chat interface class for MCP server.

    This class handles the interaction with MCP server, including initialization,
    query processing, and cleanup.
    """

    def __init__(self):
        """Initialize ChatInterface instance."""
        self.service_url = f"http://appbuilder.baidu.com/v2/ai_search/mcp/sse?api_key=Bearer+{os.getenv('api_key')}"
        self.client = MCPClient()
        self.search_results = []
        self.reference_results = []
        self.conversation_history = []
        self.model_decisions = []
        self.init_mcp = False

    def get_image_url(self, image_path: str) -> str:
        """
        Converts an image file at the given path to a base64 encoded data URL
        that can be used directly in HTML or Gradio interfaces.
        Reads the image file, encodes it in base64 format, and constructs
        a data URL with the appropriate image MIME type.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Image URL.
        """
        base64_image = ""
        extension = image_path.split(".")[-1]
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        url = f"data:image/{extension};base64,{base64_image}"
        return url

    async def initialize(self) -> None:
        """Initialize MCP server connection.

        Raises:
            ConnectionError: If connection to MCP server fails.
        """
        if not self.init_mcp:
            await self.client.connect_to_server(service_url=self.service_url)
            logging.info("MCP client initialized.")
            logging.info(f"MCP tool list: {self.client.tools}")
            self.init_mcp = True

    async def cleanup(self) -> None:
        """Clean up MCP server resources."""
        await self.client.cleanup()

    async def process_query(
        self,
        query: str,
        history: List[List[str]],
        model: str,
        max_chars: int,
        max_iterations: int = 5,
    ) -> dict:
        """Process user query and generate response.

        Args:
            query: User input query.
            history: Conversation history.
            model: Model name to use.
            max_chars: Maximum character limit for messages.
            max_iterations: Maximum number of search iterations.

        Yields:
            dict: Updated conversation history.
        """

        self.search_results = []
        self.reference_results = []
        self.model_decisions = []

        current_iteration = 0

        # Step 1: Generate task plan
        task_plan = ""
        plan_prompt = TASK_PLAN_PROMPT.format(query="{query}")
        task_plan = chat(plan_prompt.format(query=query), history, model)

        yield {"type": "plan", "content": task_plan}

        while current_iteration < max_iterations:
            current_iteration += 1

            # Step 1: Check if more information is needed
            need_more_info = None
            search_results_string = "\n\n".join(self.search_results)
            content = chat(
                SEARCH_INFO_PROMPT.format(
                    query=query, task_plan=task_plan, search_results_string=search_results_string
                ),
                history,
                model,
            )

            try:
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    content = content[start:end]
                logging.info(f"Output json result: {content}")
                result = json.loads(content)
                self.model_decisions.append(
                    f"***第{len(self.model_decisions) + 1}次决策结果***：{json.dumps(result, ensure_ascii=False, indent=2)}"
                )
                yield {"type": "content", "content": "\n\n".join(self.model_decisions)}
            except json.JSONDecodeError:
                yield history + [[query, "错误：模型输出不是有效的JSON格式，请求失败，请重新请求"]]
                return

            if not result:
                return

            if not result.get("need_more_tool_info", False):
                break

            # Step 2: Generate search parameters
            available_tools = json.dumps(
                [
                    {"name": tool.name, "description": tool.description, "input_schema": tool.inputSchema}
                    for tool in self.client.tools
                ],
                ensure_ascii=False,
                indent=2,
            )

            content_text = chat(
                GEN_QUERY_PROMPT.format(
                    available_tools=available_tools,
                    query=query,
                    task_plan=task_plan,
                    search_results_string=search_results_string,
                ),
                history,
                model,
            )

            try:
                start = content_text.find("```json") + 7
                end = content_text.rfind("```")
                if start >= 0 and end > start:
                    content_text = content_text[start:end]
                logging.info(f"Output json result: {content_text}")
                content = json.loads(content_text)
                self.model_decisions.append(
                    f"***第{len(self.model_decisions) + 1}次决策结果***：{json.dumps(content, ensure_ascii=False, indent=2)}"
                )
                yield {"type": "content", "content": "\n\n".join(self.model_decisions)}
            except json.JSONDecodeError:
                yield history + [[query, "错误：模型输出不是有效的JSON格式，请求失败，请重新请求"]]
                return

            if not content:
                return

            # Step 3: Call MCP server tool
            if type(content) != list:
                content = [content]

            try:
                search_result = []
                reference_result = []
                for content_item in content:
                    tool_name = content_item.get("tool_name")
                    tool_args = content_item.get("tool_args")
                    if not tool_name or not tool_args:
                        yield history + [[query, "错误：工具参数不完整，请求失败，请重新请求"]]
                        return

                    if "model" in tool_args:
                        del tool_args["model"]
                    for key in list(tool_args.keys()):
                        if tool_args[key] == "" or tool_args[key] is None:
                            del tool_args[key]
                    tool_result = await self.client.call_tool(tool_name, tool_args)
                    logging.info(f"Tool returned result: {tool_result}")
                    if any(
                        text.type == "text" and "Error executing tool AIsearch" in text.text
                        for text in tool_result.content
                    ):
                        raise ValueError(tool_result.content[0].text)
                    tool_result = "\n\n".join(
                        [text.text[:1000] for text in tool_result.content if text.type == "text"]
                    )
                    logging.info(f"Merged tool result: {tool_result}")

                    search_result.append(f"检索词：{tool_args.get('query')}\n检索结果：\n{tool_result}\n")
                    self.reference_results.append(tool_result)
                search_result_string = "\n\n".join(search_result)
                self.search_results.append(
                    f"*********第{len(self.search_results) + 1}次检索结果*********：\n{search_result_string}\n"
                )
                yield {"type": "tool_result", "content": "\n\n".join(self.search_results)}
            except Exception as e:
                yield history + [[query, f"检索出错：{e!s}，请重新尝试"]]
                return

        # Step 4: Generate final response
        reference_results_string = "\n\n".join(self.reference_results)[:max_chars]
        final_response = chat(
            FINAL_ANSWER_PROMPT.format(reference_results_string=reference_results_string, query=query),
            history,
            model,
        )
        yield history + [[query, final_response]]


def launch_demo(args: argparse.Namespace) -> None:
    """
    Launch demo program

    Args:
        args (argparse.Namespace): argparse Namespace object containing parsed command line arguments
    """
    chat_interface = ChatInterface()

    with gr.Blocks() as demo:

        # State component to track MCP initialization
        mcp_state = gr.State(False)

        logo_url = chat_interface.get_image_url("assets/logo.png")
        gr.Markdown(
            f"""\
                <p align="center"><img src="{logo_url}" \
                style="height: 60px"/><p>"""
        )
        gr.Markdown(
            """\
<center><font size=3>Advanced Search Demo: This is a demonstration of advanced AI search capabilities by integrating\
 with AI Search MCP Server. </br>This page is based on ERNIE model, developed by Baidu. (本演示基于文心大模型实现。)</center>"""
        )

        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(height=600)
                with gr.Row():
                    msg = gr.Textbox(label="Input", placeholder="Please input your query...", scale=9)
                with gr.Row():
                    submit_btn = gr.Button("Submit", scale=1)
                    clear = gr.Button("Clear messages", scale=1)

        with gr.Row():
            plan_box = gr.Textbox(label="Task Plan", lines=5, interactive=False)
            content_box = gr.Textbox(label="Model Decision", lines=5, interactive=False)
            tool_result_box = gr.Textbox(label="Tool Result", lines=5, interactive=False)

        async def respond(
            message: str,
            chat_history: List[List[str]],
            content_box: str,
            tool_result_box: str,
            plan_box: str,
            mcp_initialized: bool,
        ) -> tuple:
            """Process user message and update chat history.

            Args:
                message: User input message.
                chat_history: Current chat history.
                content_box: Current content box value.
                tool_result_box: Current tool result box value.
                plan_box: Current plan box value.
                mcp_initialized: Whether MCP is initialized.

            Returns:
                tuple: Updated chat history and box values.
            """
            chat_history = chat_history or []
            plan_box = ""
            content_box = ""
            tool_result_box = ""

            # Initialize MCP if not already done
            if not mcp_initialized:
                await chat_interface.initialize()
                mcp_initialized = True

            async for response in chat_interface.process_query(
                message,
                chat_history,
                model=args.model_name,
                max_chars=args.max_chars,
            ):
                if isinstance(response, list):
                    chat_history = response
                elif isinstance(response, dict):
                    if response.get("type") == "content":
                        content_box = response.get("content", "")
                    elif response.get("type") == "tool_result":
                        tool_result_box = response.get("content", "")
                    elif response.get("type") == "plan":
                        plan_box = response.get("content", "")
                yield chat_history, content_box, tool_result_box, plan_box, mcp_initialized

        msg.submit(
            respond,
            [msg, chatbot, content_box, tool_result_box, plan_box, mcp_state],
            [chatbot, content_box, tool_result_box, plan_box, mcp_state],
            concurrency_limit=10,
            show_progress=True,
        )
        submit_btn.click(
            respond,
            [msg, chatbot, content_box, tool_result_box, plan_box, mcp_state],
            [chatbot, content_box, tool_result_box, plan_box, mcp_state],
            concurrency_limit=10,
            show_progress=True,
        )
        clear.click(
            lambda: ("", "", "", "", False),
            None,
            [chatbot, content_box, tool_result_box, plan_box, mcp_state],
            queue=False,
            concurrency_limit=10,
        )

    demo.launch(server_name=args.server_name, server_port=args.server_port, share=False)


if __name__ == "__main__":
    args = get_args()
    launch_demo(args)
